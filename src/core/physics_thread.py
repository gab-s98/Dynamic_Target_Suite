from src.utils.track_manager import TrackCollisionMesh, RoadSurface2D, extrude_verge_geometry
"""
Interactive 14-DOF Real-Time Driving Simulator with 3D Bird's-Eye View & Full Kinematics Telemetry
=================================================================================================
Features:
- Physical Caster & KPI Wheel Jacking (Standstill diagonal corner weight redistribution):
  * Caster angle (nu = 5.5 deg), KPI (sigma = 6.5 deg), Mechanical Trail & Scrub Radius.
  * Steering from standstill dynamically redistributes Fz_FL, Fz_FR, Fz_RL, Fz_RR cross-diagonally!
- Exact 3D & 2D Wheel & Chassis Visual Rendering:
  * Front wheels rotate visually in the exact direction of steering (Right -> Right, Left -> Left).
  * 3D Depth-sorted track rendering with grass green outfield and horizon sky gradient.
  * 2D Top-down track map with steered wheels and dynamic grip coloring.
- Complete 14-DOF Telemetry Suite & Crash-Proof Analysis:
  * Thread-safe snapshot buffer preventing concurrent mutation crashes.
  * Suspension wheel travel (z_fl..z_rr) [mm], dynamic camber & toe, roll centers (h_rc_f, h_rc_r)
  * 4-Wheel tire forces (Fz, Fy, Fx), slip angles (alpha_i), and slip ratios (kappa_i)
  * G-G diagram with velocity coloring and friction circle.
- Advanced Force Feedback & Haptic Trigger Feedback (Impulse Triggers & 4-Channel Haptics):
  * Left Trigger (LT): Brake lockup threshold & ABS pulsation haptics.
  * Right Trigger (RT): Traction loss, throttle wheelspin & longitudinal slip buzz.
  * Low-Frequency Heavy Rumble: Chassis inertia, oversteer snap, tail drift & kerb impacts.
  * High-Frequency Light Rumble: Front tire understeer scrub & road surface grip limit.
- Universal controller auto-detection (Windows Gaming Input + Native XInput + Pygame DirectInput + Keyboard).
- Configurable RK4 Physics loop (250 Hz default / 333 Hz / 500 Hz) with real-time performance budget monitor.
- Global text crash logger saving traceback to crash_log.txt and displaying user-friendly dialogs.
"""

import sys
import os
import time
import math
import ctypes
from ctypes import wintypes
import threading
import traceback
import numpy as np
import pandas as pd
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QSplitter, QMessageBox, QFileDialog,
    QSlider, QDoubleSpinBox, QSpinBox, QAbstractSpinBox, QLineEdit,
    QTextEdit, QPlainTextEdit, QTabWidget, QScrollArea, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QToolButton, QMenu, QFrame,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPolygonF,
    QPainterPath, QLinearGradient
)

try:
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import QSurfaceFormat
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True

    # Configure high-performance OpenGL surface format (zero VSync latency / tear-free double-buffering)
    _gl_fmt = QSurfaceFormat()
    _gl_fmt.setDepthBufferSize(24)
    _gl_fmt.setStencilBufferSize(8)
    _gl_fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(_gl_fmt)
except ImportError:
    HAS_OPENGL = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import (
    mf61_lateral_force, mf61_longitudinal_force,
    mf61_combined_lateral_force, mf61_combined_longitudinal_force,
    mf61_aligning_torque
)
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_direct_collocation import build_track_geometry
from src.core.road_profile import RoadProfile, RoadProfileConfig
from src.models.tire_degradation import TireState, TireDegradationModel, TireDegradationParams


# =============================================================================
# 0. GLOBAL EXCEPTION HOOK & CRASH LOGGER
# =============================================================================
def setup_global_exception_handler():
    """Installs a global exception hook to catch unhandled crashes, write to crash_log.txt, and show a Qt dialog."""
    def exception_hook(exc_type, exc_value, exc_tb):
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(f"\n[CRASH DETECTED]\n{err_msg}\n")
        try:
            with open("crash_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- CRASH LOG [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
                f.write(err_msg)
                f.write("\n" + "="*60 + "\n")
        except Exception:
            pass

        app = QApplication.instance()
        if app is not None:
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Errore Imprevisto (Crash Log)")
                msg_box.setText(f"<b>Si è verificato un errore imprevisto:</b><br><br><code>{exc_value}</code><br><br>I dettagli sono stati registrati in <code>crash_log.txt</code>.")
                msg_box.setDetailedText(err_msg)
                msg_box.exec_()
            except Exception:
                pass
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_hook

setup_global_exception_handler()


# =============================================================================
# 1. NATIVE WINDOWS XINPUT & CONTROLLER DEFINITIONS
# =============================================================================
from src.core.gamepad_manager import GamepadManager

class PhysicsThread14DOF(QThread):
    state_updated = pyqtSignal(dict)

    def __init__(self, vehicle_params=None, tir_file=None, target_hz=500.0):
        super().__init__()
        self.vp = vehicle_params if vehicle_params is not None else VehicleParams.nominal_fsae_params()
        
        self.tir_path = TireParams._resolve_path(tir_file)
        self.tp = TireParams(self.tir_path)
        self.model = Vehicle14DOF(self.vp, self.tp, self.tp)
        self.ode = self.model.get_fast_ode()

        self.target_hz = float(target_hz)  # Default 500 Hz (Precise unsprung mass integration)
        self.dt = 1.0 / self.target_hz
        self.target_fps = 60.0
        self.target_emit_dt = 0.015
        self.running = False
        self.gamepad = GamepadManager()
        sw_max = float(getattr(self.vp, 'steering_wheel_max_deg', 125.0))
        sr = float(getattr(self.vp, 'steering_ratio', 5.0))
        self.gamepad.max_steer_deg = sw_max / max(sr, 0.1)
        self.key_states = {}
        self.speed_sensitive_steer = False

        # Traction Control System (TCS) — Switchable (Default: ON)
        self.tcs_enabled = True
        self.tcs_target_slip = 0.095
        self.tcs_active = False
        self.tcs_cut_pct = 0.0
        
        self.actual_physics_hz = self.target_hz
        self.physics_step_ms = 0.25
        self.real_time_ratio = 1.00
        
        self.dist_total = 0.0
        self.pos_s = 0.0
        self._closest_track_idx = 0
        
        self._history_lock = threading.Lock()
        self.telemetry_history = []
        self.last_stint_telemetry = []
        self._tuning_lock = threading.Lock()
        self._tuning_dirty = False
        self.max_history = 50000
        self.track_data = build_track_geometry(track_length_m=850.0, ds=2.0, track_width_m=3.8)
        self.track_mesh = TrackCollisionMesh(self.track_data, road_surface_2d=getattr(self, 'road_surface_2d', None))

        # Thermal Degradation Integration Rate (can be 10, 25, 50, 100 Hz or 0=sync)
        self.thermal_hz = 50.0
        self._thermal_accumulator = 0.0
        self.tire_deg_model = TireDegradationModel(mode="off")
        self.tire_T_start = 25.0
        self.tire_states = [TireState() for _ in range(4)]

        # Wheel Spin Angle Accumulators [rad] for 3D rotation rendering: fl, fr, rl, rr
        self._wheel_angle = [0.0, 0.0, 0.0, 0.0]

        # Lap Invalidation: 4-wheel off-track detection
        self._lap_invalid = False           # True = current lap is invalid (no best-lap archiving)
        self._offtrack_4wheel_timer = 0.0   # Continuous time [s] with all 4 wheels off-track
        self._offtrack_4wheel_thresh = 0.15 # Invalidation threshold [s]

        # Powertrain RWD Dual-Motor Differentials & Active Torque Vectoring (TV)
        self._tv_integral = 0.0

        # Continuous Spatial Lap Trajectory & Ghost Car Interpolation
        self.current_lap_trajectory = []    # [(s, x, y, z, psi, t_lap), ...]
        self.best_lap_trajectory = []       # Archived personal best lap
        self._ghost_s = None
        self._ghost_x = None
        self._ghost_y = None
        self._ghost_z = None
        self._ghost_psi = None
        self._ghost_t = None
        self.live_delta_lap = 0.0           # Spatial delta [s]: t_current - t_best(s)
        self.ghost_car_state = None         # {'x', 'y', 'z', 'psi'}

        # 3D Road Profile (Macro Altimetry & Micro Roughness)
        self.road_config = RoadProfileConfig(enable_macro=False, enable_micro=False)
        self.road_profile = RoadProfile.flat(self.track_data.get('s', np.linspace(0, 850, len(self.track_data['x_center']))), config=self.road_config)
        self.road_profile.apply_to_track_data(self.track_data)
        
        self.road_surface_2d = None
        extrude_verge_geometry(self.track_data, w_verge=12.0, h_drop=0.4)

        # Zero-Allocation pre-allocated buffers for RK4 integration
        self._k1 = np.empty(28, dtype=np.float64)
        self._k2 = np.empty(28, dtype=np.float64)
        self._k3 = np.empty(28, dtype=np.float64)
        self._k4 = np.empty(28, dtype=np.float64)
        self._x_tmp = np.empty(28, dtype=np.float64)
        self._x_cand = np.empty(28, dtype=np.float64)

        self.reset_state()

    def set_vehicle_params(self, vp):
        """Allows dynamically updating the vehicle parameters and kinematic maps in real-time."""
        self.vp = vp
        sw_max = float(getattr(self.vp, 'steering_wheel_max_deg', 125.0))
        sr = float(getattr(self.vp, 'steering_ratio', 5.0))
        self.gamepad.max_steer_deg = sw_max / max(sr, 0.1)

        # Automatically update tire model if specified in vp
        tire_file = getattr(self.vp, 'tire_file', None)
        if tire_file:
            from src.models.params_tire import TireParams
            self.tp = TireParams(tire_file)

        self.model = Vehicle14DOF(self.vp, self.tp, self.tp)
        self.ode = self.model.get_fast_ode()
        if hasattr(self, 'tire_deg_model') and self.tire_deg_model is not None:
            self.tire_deg_model.update_from_vehicle_params(self.vp)
        self._params_dirty = True

    def set_tire_params(self, tp_or_path):
        """Updates tire property model with a new TireParams or .tir filepath."""
        from src.models.params_tire import TireParams
        if isinstance(tp_or_path, str):
            self.tp = TireParams(tp_or_path)
        elif isinstance(tp_or_path, TireParams):
            self.tp = tp_or_path
        self.model = Vehicle14DOF(self.vp, self.tp, self.tp)
        self.ode = self.model.get_fast_ode()

    def set_tcs_enabled(self, enabled: bool):
        """Enable or disable the closed-loop Traction Control System."""
        self.tcs_enabled = bool(enabled)
        if not self.tcs_enabled:
            self.tcs_active = False
            self.tcs_cut_pct = 0.0

    def toggle_tcs(self) -> bool:
        """Toggles the Traction Control System on/off and returns the new state."""
        self.set_tcs_enabled(not getattr(self, 'tcs_enabled', True))
        return self.tcs_enabled

    def get_last_stint_snapshot(self):
        """Thread-safe snapshot of the last completed stint telemetry before reset."""
        with self._history_lock:
            return list(self.last_stint_telemetry)

    def update_live_tuning(self, camber_f=None, camber_r=None, toe_f=None, toe_r=None,
                           preload_f=None, preload_r=None, aero_bal=None, Cz=None, Cd=None,
                           press_f=None, press_r=None, k_tire_z=None,
                           diff_mode=None, damper_mode=None,
                           lsc_f=None, hsc_f=None, lsr_f=None, hsr_f=None,
                           lsc_r=None, hsc_r=None, lsr_r=None, hsr_r=None,
                           k_phi_f=None, k_phi_r=None, lltd=None, arb_model=None,
                           rh_f=None, rh_r=None,
                           k_spring_f=None, k_spring_r=None, l0_f=None, l0_r=None):
        """Allows dynamically updating vehicle setup in real-time in a thread-safe manner."""
        with self._tuning_lock:
            if camber_f is not None:
                self.vp.camber_f_static = float(camber_f)
            if camber_r is not None:
                self.vp.camber_r_static = float(camber_r)
            if toe_f is not None:
                self.vp.toe_f_static = float(toe_f)
            if toe_r is not None:
                self.vp.toe_r_static = float(toe_r)
            if k_tire_z is not None:
                self.vp.tire_vertical_stiffness_12psi = float(k_tire_z)
            if k_spring_f is not None:
                self.vp.spring_rate_installed_f_Nmm = float(k_spring_f)
                mr_f = getattr(self.vp, 'motion_ratio_f', 0.8)
                self.vp.K_z_f = float(k_spring_f * 1000.0 * (mr_f ** 2))
                st_f = self.vp.calc_damper_strokes_from_preload('f')
                self.vp.bumpstop_gap_f_m = float(st_f['wheel_bumpstop_gap_mm'] / 1000.0)
            if k_spring_r is not None:
                self.vp.spring_rate_installed_r_Nmm = float(k_spring_r)
                mr_r = getattr(self.vp, 'motion_ratio_r', 0.8)
                self.vp.K_z_r = float(k_spring_r * 1000.0 * (mr_r ** 2))
                st_r = self.vp.calc_damper_strokes_from_preload('r')
                self.vp.bumpstop_gap_r_m = float(st_r['wheel_bumpstop_gap_mm'] / 1000.0)
            if l0_f is not None:
                self.vp.pushrod_offset_f_mm = float(l0_f)
            if l0_r is not None:
                self.vp.pushrod_offset_r_mm = float(l0_r)
            if preload_f is not None:
                self.vp.preload_f_N = float(preload_f)
                k_sf = getattr(self.vp, 'spring_rate_installed_f_Nmm', 59.5)
                self.vp.spring_preload_f_mm = float(preload_f) / max(k_sf, 0.1)
                st_f = self.vp.calc_damper_strokes_from_preload('f')
                self.vp.bumpstop_gap_f_m = float(st_f['wheel_bumpstop_gap_mm'] / 1000.0)
            if preload_r is not None:
                self.vp.preload_r_N = float(preload_r)
                k_sr = getattr(self.vp, 'spring_rate_installed_r_Nmm', 61.0)
                self.vp.spring_preload_r_mm = float(preload_r) / max(k_sr, 0.1)
                st_r = self.vp.calc_damper_strokes_from_preload('r')
                self.vp.bumpstop_gap_r_m = float(st_r['wheel_bumpstop_gap_mm'] / 1000.0)
            if rh_f is not None and rh_r is not None:
                self.vp.update_static_cg_from_ride_height(float(rh_f) / 1000.0, float(rh_r) / 1000.0)
            if aero_bal is not None:
                self.vp.aero_balance = float(aero_bal)
            if Cz is not None:
                self.vp.Cz = float(Cz)
                self.vp.aero_Cz = float(Cz)
            if Cd is not None:
                self.vp.Cd = float(Cd)
                self.vp.aero_Cd = float(Cd)
            if press_f is not None:
                self.vp.tire_pressure_f_bar = float(press_f)
            if press_r is not None:
                self.vp.tire_pressure_r_bar = float(press_r)
            if k_phi_f is not None:
                self.vp.K_phi_f = float(k_phi_f)
            if k_phi_r is not None:
                self.vp.K_phi_r = float(k_phi_r)
            if lltd is not None:
                self.vp.lltd = float(lltd)
            elif k_phi_f is not None or k_phi_r is not None:
                ktot = self.vp.K_phi_f + self.vp.K_phi_r
                if ktot > 1.0:
                    self.vp.lltd = float(self.vp.K_phi_f / ktot)
            if arb_model is not None:
                self.vp.arb_diamond_model = str(arb_model)

            rebuild_ode = False
            if diff_mode is not None:
                self.vp.diff_mode = str(diff_mode)
                self.vp.diff_type = str(diff_mode)
            if damper_mode is not None:
                self.vp.damper_mode = str(damper_mode)
                rebuild_ode = True
            if lsc_f is not None:
                self.vp.damper_lsc_clicks_f = float(lsc_f)
                rebuild_ode = True
            if hsc_f is not None:
                self.vp.damper_hsc_turns_f = float(hsc_f)
                rebuild_ode = True
            if lsr_f is not None:
                self.vp.damper_lsr_clicks_f = float(lsr_f)
                rebuild_ode = True
            if hsr_f is not None:
                self.vp.damper_hsr_turns_f = float(hsr_f)
                rebuild_ode = True
            if lsc_r is not None:
                self.vp.damper_lsc_clicks_r = float(lsc_r)
                rebuild_ode = True
            if hsc_r is not None:
                self.vp.damper_hsc_turns_r = float(hsc_r)
                rebuild_ode = True
            if lsr_r is not None:
                self.vp.damper_lsr_clicks_r = float(lsr_r)
                rebuild_ode = True
            if hsr_r is not None:
                self.vp.damper_hsr_turns_r = float(hsr_r)
                rebuild_ode = True

            if rebuild_ode:
                self.model = Vehicle14DOF(self.vp, self.tp, self.tp)
                self.ode = self.model.get_fast_ode()
                self._params_dirty = True

            self._tuning_dirty = True

    def set_physics_hz(self, hz):
        """
        Allows dynamically updating vehicle physics frequency (e.g. 500 Hz, 1000 Hz, 1500 Hz).

        Valutazione Stabilità vs Overhead CPU:
        - 500 Hz (dt = 2.0 ms): Il polo del gruppo ruota (m_u ≈ 8-16 kg, K_tire ≈ 75-185 kN/m,
          frequenza propria omega_n ≈ 107-152 rad/s ~ 17-24 Hz) si trova entro il dominio di stabilità
          di RK4 esplicito (omega_n * dt ≈ 0.22-0.30 con fattore smorzamento zeta ≈ 0.35); ideale
          per minimizzare i carichi su sistemi con risorse CPU limitate.
        - 1000 Hz (dt = 1.0 ms): Riduce omega_n * dt a ~0.11-0.15 << 2.8, garantendo convergenza
          asintotica incondizionata, rapida estinzione dei disturbi e piena reiezione del chattering
          durante i transitori bruschi di coppia motrice (100% throttle launch).
        - 1500 Hz (dt ≈ 0.667 ms): Integrazione ad altissima fedeltà per transitori estremi non-lineari.
          Poiché il solutore C compilato (ode_14dof) richiede ~19 µs per valutazione (~0.11 ms per step
          completo RK4 a 4 stadi, corrispondente a oltre 8500 Hz di calcolo continuo), 1500 Hz occupa
          meno del 18% di un singolo core CPU moderno, offrendo il miglior compromesso qualità/prestazioni.
        """
        new_hz = float(np.clip(hz, 100.0, 1500.0))
        self.target_hz = new_hz
        self.dt = 1.0 / new_hz

    def set_target_fps(self, fps: float):
        """Allows setting the target UI frame rate (e.g. 60, 120, 144, 240, or 0 for unlocked)."""
        fps_val = float(fps)
        self.target_fps = fps_val
        if fps_val <= 0 or fps_val >= 240:
            self.target_emit_dt = 0.0035  # emit every physics substep at 250 Hz (~240-250 FPS)
        elif fps_val >= 144:
            self.target_emit_dt = 0.0065  # ~144-150 FPS
        elif fps_val >= 120:
            self.target_emit_dt = 0.0075  # ~125 FPS
        elif fps_val >= 90:
            self.target_emit_dt = 0.010   # ~100 FPS
        elif fps_val >= 60:
            self.target_emit_dt = 0.015   # 60 FPS rock-solid
        elif fps_val >= 30:
            self.target_emit_dt = 0.031   # ~30 FPS
        else:
            self.target_emit_dt = 1.0 / max(fps_val, 1.0)

    def set_thermal_hz(self, hz):
        """Allows setting separate integration frequency for tire thermal degradation (0 = synchronous)."""
        self.thermal_hz = float(max(hz, 0.0))

    def set_tire_t_start(self, t_start):
        """Update initial tire starting temperature [°C]."""
        self.tire_T_start = float(t_start)

    def set_tire_wear_scale(self, scale):
        """Update tire wear acceleration multiplier (1.0 = real endurance, 2.0, 5.0, 10.0 = quick test)."""
        if hasattr(self, 'tire_deg_model'):
            self.tire_deg_model.set_wear_rate_scale(float(scale))

    def set_road_profile_config(self, enable_macro=False, enable_micro=False,
                                macro_elevation_amp=1.5, macro_bank_max_deg=3.5,
                                micro_roughness_amp_mm=8.0):
        """Update 3D road profile parameters and recalculate road surface ribbons."""
        self.road_config.enable_macro = bool(enable_macro)
        self.road_config.enable_micro = bool(enable_micro)
        self.road_config.macro_elevation_amp = float(macro_elevation_amp)
        self.road_config.macro_bank_max_deg = float(macro_bank_max_deg)
        self.road_config.micro_roughness_amp_mm = float(micro_roughness_amp_mm)
        self._update_road_profile_for_track()

    def _update_road_profile_for_track(self):
        """Generates or updates 3D road profile coordinates for current track."""
        if self.track_data is None:
            return
        xc = self.track_data.get('x_center', [])
        if len(xc) < 2:
            return
        s_arr = self.track_data.get('s', None)
        if s_arr is None or len(s_arr) != len(xc):
            dx = np.diff(xc)
            dy = np.diff(self.track_data.get('y_center', []))
            ds_seg = np.hypot(dx, dy)
            s_arr = np.concatenate([[0.0], np.cumsum(ds_seg)])
            self.track_data['s'] = s_arr
        curv = self.track_data.get('kappa', None)
        if getattr(self.road_config, 'is_custom_user', False) and getattr(self.road_config, 'custom_stations', None):
            old_stations = self.road_config.custom_stations
            L_track = float(s_arr[-1])
            max_s = max((st['s'] if isinstance(st, dict) else st[0]) for st in old_stations)
            if max_s > L_track * 1.05 or max_s < L_track * 0.6:
                scale = L_track / max(max_s, 1.0)
                adapted_stations = []
                for st in old_stations:
                    if isinstance(st, dict):
                        adapted_stations.append({'s': round(st['s'] * scale, 1), 'z': st['z'], 'bank_deg': st.get('bank_deg', 0.0)})
                    else:
                        adapted_stations.append({'s': round(st[0] * scale, 1), 'z': st[1], 'bank_deg': st[2] if len(st) > 2 else 0.0})
                self.road_config.custom_stations = adapted_stations
            self.road_profile = RoadProfile.from_user_stations(
                s_arr, self.road_config.custom_stations, is_closed=self.track_data.get('is_closed', True),
                config=self.road_config
            )
        self.road_profile.apply_to_track_data(self.track_data)
        
        extrude_verge_geometry(self.track_data, w_verge=8.0, h_drop=0.15)
        L_track = float(s_arr[-1] - s_arr[0]) if len(s_arr) > 1 else 100.0
        w_track = float(self.track_data.get('track_width_m', self.track_data.get('track_width', 3.8)))
        if getattr(self.road_config, 'enable_micro', False) or getattr(self, 'road_surface_2d', None) is None:
            self.road_surface_2d = RoadSurface2D(length=L_track, width=w_track, ds=0.05, dd=0.05, L_patch=0.12)

    def set_custom_road_profile(self, stations, enable_macro=True):
        """Applies user-defined 3D stations [s, z, bank_deg] to the road profile."""
        self.road_config.is_custom_user = True
        self.road_config.custom_stations = list(stations)
        self.road_config.enable_macro = bool(enable_macro)
        self._update_road_profile_for_track()

    def get_telemetry_snapshot(self):
        """Thread-safe snapshot copy of telemetry history to prevent concurrent mutation errors."""
        with self._history_lock:
            return list(self.telemetry_history)

    def set_track_grip(self, grip_mult):
        """Dynamically update track surface grip multiplier."""
        self._sim_opts_track_grip = float(max(grip_mult, 0.05))

    def set_tire_degradation_mode(self, mode_str):
        """Set tire degradation mode ('off', 'light', 'detailed')."""
        if hasattr(self, 'tire_deg_model'):
            self.tire_deg_model.set_mode(mode_str)

    def set_tire_wear_scale(self, scale: float):
        """Set tire wear rate scale multiplier (1.0 = real endurance, 2.0, 5.0, 10.0)."""
        if hasattr(self, 'tire_deg_model'):
            self.tire_deg_model.set_wear_rate_scale(scale)

    def _cache_best_lap_arrays(self):
        """Precomputes contiguous NumPy 1D arrays for best-lap trajectory interpolation to avoid 250 Hz allocations."""
        if len(self.best_lap_trajectory) >= 5:
            arr = np.asarray(self.best_lap_trajectory, dtype=np.float64)
            self._ghost_s = np.ascontiguousarray(arr[:, 0])
            self._ghost_x = np.ascontiguousarray(arr[:, 1])
            self._ghost_y = np.ascontiguousarray(arr[:, 2])
            self._ghost_z = np.ascontiguousarray(arr[:, 3])
            self._ghost_psi = np.ascontiguousarray(arr[:, 4])
            self._ghost_t = np.ascontiguousarray(arr[:, 5])
        else:
            self._ghost_s = None
            self._ghost_x = None
            self._ghost_y = None
            self._ghost_z = None
            self._ghost_psi = None
            self._ghost_t = None

    def _update_road_contacts(self, p_vec):
        """
        Evaluates 3D road profile and updates contact patch elevations, rates, and normals in p_vec.
        Uses fast local window search for track centerline to eliminate full array allocations.
        """
        L_m = float(getattr(self.vp, 'L', 1.530))
        wd_f = float(getattr(self.vp, 'wd_f', 0.53))
        a_dist = L_m * (1.0 - wd_f)
        b_dist = L_m * wd_f
        tf = max(float(getattr(self.vp, 't_f', 1.250)), 0.5)
        tr = max(float(getattr(self.vp, 't_r', 1.200)), 0.5)

        z_road = 0.0
        grade_road = 0.0
        bank_road = 0.0
        wh_heights = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        wh_rates = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        wh_normals = {
            'fl': np.array([0.0, 0.0, 1.0], dtype=np.float64), 'fr': np.array([0.0, 0.0, 1.0], dtype=np.float64),
            'rl': np.array([0.0, 0.0, 1.0], dtype=np.float64), 'rr': np.array([0.0, 0.0, 1.0], dtype=np.float64)
        }
        wh_contacts = {
            'fl': np.array([self.pos_x, self.pos_y, 0.0], dtype=np.float64), 'fr': np.array([self.pos_x, self.pos_y, 0.0], dtype=np.float64),
            'rl': np.array([self.pos_x, self.pos_y, 0.0], dtype=np.float64), 'rr': np.array([self.pos_x, self.pos_y, 0.0], dtype=np.float64)
        }
        wh_r2d = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}

        if self.track_data is not None:
            xc_tr = self.track_data.get('x_center', np.array([]))
            yc_tr = self.track_data.get('y_center', np.array([]))
            s_tr = self.track_data.get('s', None)
            N_tr = len(xc_tr)
            if N_tr > 2 and s_tr is not None:
                # Fast local window search around previous closest index
                c_idx = getattr(self, '_closest_track_idx', 0)
                win_start = max(0, c_idx - 15)
                win_end = min(N_tr, c_idx + 25)

                dx_w = xc_tr[win_start:win_end] - self.pos_x
                dy_w = yc_tr[win_start:win_end] - self.pos_y
                d_sq_w = dx_w * dx_w + dy_w * dy_w
                min_w = int(np.argmin(d_sq_w))

                if d_sq_w[min_w] < 36.0:  # within 6m
                    idx_near = win_start + min_w
                else:
                    dx_vec = xc_tr - self.pos_x
                    dy_vec = yc_tr - self.pos_y
                    idx_near = int(np.argmin(dx_vec * dx_vec + dy_vec * dy_vec))
                self._closest_track_idx = idx_near
                s_curr = float(s_tr[idx_near])

                z_center_macro, grade_road, bank_road = self.road_profile.interpolate_at(s_curr)

                theta_tr = float(self.track_data['theta'][idx_near]) if 'theta' in self.track_data else 0.0
                sin_theta = math.sin(theta_tr)
                cos_theta = math.cos(theta_tr)
                d_car = -(self.pos_x - xc_tr[idx_near]) * sin_theta + (self.pos_y - yc_tr[idx_near]) * cos_theta
                sin_bank = math.sin(bank_road)

                z_road = z_center_macro + d_car * sin_bank

                wh_lateral_rel = {'fl': tf / 2.0, 'fr': -tf / 2.0, 'rl': tr / 2.0, 'rr': -tr / 2.0}
                wh_long_rel = {'fl': a_dist, 'fr': a_dist, 'rl': -b_dist, 'rr': -b_dist}

                cos_psi = math.cos(self.psi)
                sin_psi = math.sin(self.psi)
                d_ray = np.array([0.0, 0.0, -1.0])

                for corner in ('fl', 'fr', 'rl', 'rr'):
                    dx_l = wh_long_rel[corner]
                    dy_l = wh_lateral_rel[corner]
                    xw = self.pos_x + dx_l * cos_psi - dy_l * sin_psi
                    yw = self.pos_y + dx_l * sin_psi + dy_l * cos_psi

                    rw_hub = np.array([xw, yw, z_road + 1.5])
                    r_contact, n_road, stype = self.track_mesh.raycast_down(rw_hub, d_ray)

                    wh_contacts[corner] = r_contact
                    wh_normals[corner] = n_road
                    wh_heights[corner] = float(r_contact[2])
                    wh_rates[corner] = 0.0

                self.model.set_road_elevations(
                    wh_heights['fl'], wh_heights['fr'], wh_heights['rl'], wh_heights['rr'],
                    wh_rates['fl'], wh_rates['fr'], wh_rates['rl'], wh_rates['rr']
                )
                while len(p_vec) < 37:
                    p_vec.append(0.0)
                p_vec[17] = wh_heights['fl']
                p_vec[18] = wh_heights['fr']
                p_vec[19] = wh_heights['rl']
                p_vec[20] = wh_heights['rr']
                p_vec[21] = wh_rates['fl']
                p_vec[22] = wh_rates['fr']
                p_vec[23] = wh_rates['rl']
                p_vec[24] = wh_rates['rr']
                for c_idx, c_key in enumerate(('fl', 'fr', 'rl', 'rr')):
                    n_road = wh_normals[c_key]
                    nx_c = n_road[0] * cos_psi + n_road[1] * sin_psi
                    ny_c = -n_road[0] * sin_psi + n_road[1] * cos_psi
                    nz_c = n_road[2]
                    p_vec[25 + c_idx * 3] = nx_c
                    p_vec[26 + c_idx * 3] = ny_c
                    p_vec[27 + c_idx * 3] = nz_c

        return z_road, grade_road, bank_road, wh_heights, wh_rates, wh_normals, wh_contacts, wh_r2d

    def set_track_data(self, track_data):
        """Dynamically update track data in physics thread, reset S/F detector, and update 3D road."""
        self.track_data = track_data
        self.track_mesh = TrackCollisionMesh(track_data, road_surface_2d=getattr(self, 'road_surface_2d', None))
        self.best_lap_trajectory = []
        self._cache_best_lap_arrays()
        self.current_lap_trajectory = []
        self.live_delta_lap = 0.0
        self.ghost_car_state = None
        self._sf_dot_prev = -1.0
        self._sf_cooldown = 0.0
        self._sf_armed = False
        self._update_road_profile_for_track()

    def reset_state(self, x=None, y=None, psi=None, vx=0.0, T_start=None):
        """
        Resets vehicle state.
        Stages the car 10.0 meters ahead on the starting grid, centered on the track
        centerline, aligned with the tangent vector t(s0), resting directly on the asphalt.
        """
        s0 = 10.0
        z_road_s0 = 0.0
        heading = 0.0

        if self.track_data is not None and len(self.track_data.get('x_center', [])) > 1:
            xc = np.asarray(self.track_data['x_center'], dtype=float)
            yc = np.asarray(self.track_data['y_center'], dtype=float)
            s_arr = self.track_data.get('s', None)
            N_tr = len(xc)

            if s_arr is not None and len(s_arr) == N_tr:
                s_arr = np.asarray(s_arr, dtype=float)
                L_total = float(s_arr[-1] - s_arr[0]) if len(s_arr) > 1 else 100.0
                s0_eff = min(10.0, L_total * 0.15) if L_total > 12.0 else 10.0
                x0 = float(np.interp(s0_eff, s_arr, xc))
                y0 = float(np.interp(s0_eff, s_arr, yc))
                ds = 0.5
                s_fwd = (s0_eff + ds) % L_total
                s_bwd = (s0_eff - ds) % L_total
                dx0 = float(np.interp(s_fwd, s_arr, xc) - np.interp(s_bwd, s_arr, xc))
                dy0 = float(np.interp(s_fwd, s_arr, yc) - np.interp(s_bwd, s_arr, yc))
                self.pos_s = s0_eff
            else:
                dx0 = float(xc[1] - xc[0])
                dy0 = float(yc[1] - yc[0])
                L0 = math.hypot(dx0, dy0)
                ux0 = (dx0 / L0) if L0 > 1e-4 else 1.0
                uy0 = (dy0 / L0) if L0 > 1e-4 else 0.0
                x0 = float(xc[0] + s0 * ux0)
                y0 = float(yc[0] + s0 * uy0)
                self.pos_s = s0

            heading = math.atan2(dy0, dx0)

            if hasattr(self, 'road_profile') and self.road_profile is not None:
                z_road_s0, _, _ = self.road_profile.interpolate_at(self.pos_s)
            elif 'z_center' in self.track_data:
                zc = np.asarray(self.track_data['z_center'], dtype=float)
                if s_arr is not None and len(s_arr) == len(zc):
                    z_road_s0 = float(np.interp(self.pos_s, s_arr, zc))
                else:
                    z_road_s0 = float(zc[0])

            if x is None or y is None:
                self.pos_x = x0
                self.pos_y = y0
                self.psi = float(heading) if psi is None else float(psi)
            else:
                self.pos_x = float(x)
                self.pos_y = float(y)
                self.psi = float(psi) if psi is not None else float(heading)
        else:
            self.pos_x = float(x) if x is not None else 0.0
            self.pos_y = float(y) if y is not None else 0.0
            self.psi = float(psi) if psi is not None else 0.0
            self.pos_s = 0.0

        self.filtered_steer_rad = 0.0

        Rw = getattr(self.vp, 'R_w', 0.203)
        w0 = max(vx, 0.0) / Rw
        z_init = float(z_road_s0)

        # Set model ground contact elevations so initial tire deflection is exact
        if hasattr(self, 'model') and hasattr(self.model, 'set_road_elevations'):
            self.model.set_road_elevations(z_init, z_init, z_init, z_init)

        if hasattr(self, '_p_vec') and self._p_vec is not None and len(self._p_vec) >= 21:
            self._p_vec[17] = z_init
            self._p_vec[18] = z_init
            self._p_vec[19] = z_init
            self._p_vec[20] = z_init

        self.x_state = np.array([
            vx, 0.0, 0.0,               # vx, vy, r         (0..2)
            0.0, 0.0,                  # phi, phi_dot      (3..4)
            0.0, 0.0,                  # theta, theta_dot  (5..6)
            0.0,                       # w_heave           (7)
            0.0, 0.0, 0.0, 0.0,        # z_fl, z_fr, z_rl, z_rr (8..11) (SSOT: 0.0 at static 1g ground equilibrium)
            0.0, 0.0, 0.0, 0.0,        # vz_fl, vz_fr, vz_rl, vz_rr (12..15)
            w0, w0, w0, w0,            # w_fl, w_fr, w_rl, w_rr (16..19)
            0.0, 0.0,                  # ux_fl, uy_fl      (20..21)
            0.0, 0.0,                  # ux_fr, uy_fr      (22..23)
            0.0, 0.0,                  # ux_rl, uy_rl      (24..25)
            0.0, 0.0                   # ux_rr, uy_rr      (26..27)
        ], dtype=float)

        self.time_sim = 0.0
        self.lap_timer = 0.0
        self.dist_total = float(self.pos_s)
        self._closest_track_idx = 0

        # --- Lap Timer (real S/F gate detection) ---
        self.lap_count = 0
        self.lap_time_current = 0.0
        self.lap_time_last = 0.0
        self.lap_time_best = float('inf')
        self.lap_times_list = []
        self._sf_dot_prev = -1.0         # Start behind the line (negative dot product)
        self._sf_armed = False           # Starts disarmed; armed upon crossing start line
        self._sf_cooldown = 0.0          # Debounce timer [s]

        # --- Torque Vectoring & Ghost Trajectory State ---
        self._tv_integral = 0.0
        self.current_lap_trajectory = []
        self.live_delta_lap = 0.0
        self.ghost_car_state = None

        # --- Tire Degradation ---
        if T_start is not None:
            self.tire_T_start = float(T_start)
        t_init = getattr(self, 'tire_T_start', 25.0)
        if not hasattr(self, 'tire_deg_model'):
            self.tire_deg_model = TireDegradationModel(mode="off")
        self.tire_deg_model.update_from_vehicle_params(self.vp)
        self.tire_states = [TireState() for _ in range(4)]  # FL, FR, RL, RR
        for ts in self.tire_states:
            self.tire_deg_model.reset(ts, T_start=t_init)

        # --- Wheel Spin Angle Accumulators ---
        self._wheel_angle = [0.0, 0.0, 0.0, 0.0]

        # --- Lap Invalidation State ---
        self._lap_invalid = False
        self._offtrack_4wheel_timer = 0.0

        # --- Dynamic Grip Factor ---
        self._grip_multiplier = 1.0

        with self._history_lock:
            if self.telemetry_history:
                self.last_stint_telemetry = list(self.telemetry_history)
            self.telemetry_history.clear()
        import gc
        gc.collect()

    def run(self):
        self.running = True
        last_emit_time = time.perf_counter()
        step_times = []
        last_hz_calc_time = time.time()
        steps_count = 0

        # --- Real-Time OS Timing, Affinity & Priority Setup ---
        is_windows = (sys.platform == "win32")
        if is_windows:
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass

        try:
            import psutil
            proc = psutil.Process()
            cores = list(range(psutil.cpu_count(logical=True)))
            # Dedicate core 2 or highest core, keeping core 0 for OS DPC/ISR and GUI
            target_core = 2 if 2 in cores else cores[-1]
            proc.cpu_affinity([target_core])
            if is_windows:
                # HIGH_PRIORITY_CLASS (0x00000080) guarantees deterministic scheduling
                ctypes.windll.kernel32.SetPriorityClass(
                    ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
                )
        except Exception:
            pass

        # Disabilita il Garbage Collector durante la sessione fisica attiva
        import gc
        gc.disable()

        target_dt_ns = int(self.dt * 1e9)
        next_tick_ns = time.perf_counter_ns()
        
        z_s0_init = float(getattr(self, 'pos_s', 0.0))
        z_r0 = 0.0
        if hasattr(self, 'road_profile') and self.road_profile is not None:
            z_r0, _, _ = self.road_profile.interpolate_at(z_s0_init)
        elif self.track_data is not None and 'z_center' in self.track_data:
            zc = np.asarray(self.track_data['z_center'], dtype=float)
            z_r0 = float(zc[0]) if len(zc) > 0 else 0.0

        p_vec = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            float(getattr(self.vp, 'LMUY', 1.0)),
            float(getattr(self.vp, 'LMUX', 1.0)),
            float(getattr(self.vp, 'Cd', 1.20)),
            z_r0, z_r0, z_r0, z_r0,  # 17..20: z_road_fl, z_road_fr, z_road_rl, z_road_rr
            0.0, 0.0, 0.0, 0.0,      # 21..24: dz_road_fl, dz_road_fr, dz_road_rl, dz_road_rr
            0.0, 0.0, 1.0,           # 25..27: nx_fl, ny_fl, nz_fl
            0.0, 0.0, 1.0,           # 28..30: nx_fr, ny_fr, nz_fr
            0.0, 0.0, 1.0,           # 31..33: nx_rl, ny_rl, nz_rl
            0.0, 0.0, 1.0            # 34..36: nx_rr, ny_rr, nz_rr
        ]
        self._p_vec = p_vec

        # --- Load grip & degradation settings from SimulationOptions ---
        try:
            from src.core.simulation_options import get_simulation_options
            sim_opts = get_simulation_options()
            self._sim_opts_track_grip = float(getattr(sim_opts, 'track_grip_multiplier', 1.0))
            self._sim_opts_offtrack_grip = float(getattr(sim_opts, 'offtrack_grip_multiplier', 0.35))
            self._sim_opts_kerb_grip = float(getattr(sim_opts, 'kerb_grip_multiplier', 0.85))
            deg_mode = str(getattr(sim_opts, 'tire_degradation_mode', 'off'))
            self.tire_deg_model.set_mode(deg_mode)
            self.tire_deg_model.params.T_ambient = float(getattr(sim_opts, 'tire_T_ambient', 25.0))
            self.tire_deg_model.params.T_start = float(getattr(sim_opts, 'tire_T_start', 25.0))
            self.tire_deg_model.params.T_optimal = float(getattr(sim_opts, 'tire_T_optimal', 56.2))
            self.tire_deg_model.params.T_overheat = float(getattr(sim_opts, 'tire_T_overheat', 85.0))
            self.tire_deg_model.set_wear_rate_scale(float(getattr(sim_opts, 'tire_wear_rate_scale', 1.0)))
        except Exception:
            self._sim_opts_track_grip = 1.0
            self._sim_opts_offtrack_grip = 0.35
            self._sim_opts_kerb_grip = 0.85

        t_max_motor = getattr(self.vp, 'T_max_motor', 100.0)
        final_drive = getattr(self.vp, 'final_drive', 4.0)
        is_dual_motor = getattr(self.vp, 'diff_mode', 'DUAL_MOTOR_OPEN').upper().startswith('DUAL') or getattr(self.vp, 'drivetrain_type', 'rwd') == 'rwd'
        t_max_wheel = (t_max_motor * final_drive) if is_dual_motor else (t_max_motor * final_drive) / 2.0
        t_brake_max_total = float(getattr(self.vp, 'T_brake_max_total', 1400.0))
        pt = getattr(self.vp, 'get_powertrain', lambda: None)()
        if pt is None:
            from src.models.powertrain_electric import get_default_powertrain
            pt = get_default_powertrain()
        self._powertrain_telemetry = {}
        if hasattr(self, 'tire_deg_model') and self.tire_deg_model is not None:
            self.tire_deg_model.update_from_vehicle_params(self.vp)

        def eval_poly3(coeffs, val):
            a3, a2, a1, a0 = coeffs
            return a3*(val**3) + a2*(val**2) + a1*val + a0

        accumulator = 0.0
        prev_wall = time.perf_counter()
        MAX_SUBSTEPS = 8

        while self.running:
            now_wall = time.perf_counter()
            dt_real = min(now_wall - prev_wall, MAX_SUBSTEPS * self.dt)
            prev_wall = now_wall
            accumulator += dt_real
            
            cmd = self.gamepad.poll_inputs(self.key_states)
            if cmd['reset']:
                self.reset_state()
                time.sleep(0.05)
                continue

            # Direct 1:1 Steer command in vehicle frame (full lock authority at any speed)
            steer_norm = float(cmd['steer'])
            vx_curr = max(float(self.x_state[0]), 0.5)
            is_wheel_dev = self.gamepad.is_wheel()
            if getattr(self, 'speed_sensitive_steer', False) and not is_wheel_dev:
                # Optional assist (only for gamepad thumbsticks when enabled)
                speed_factor = 1.0 / (1.0 + (vx_curr / 15.0)**1.2)
            else:
                # Direct 1:1 racing steering (100% steering authority at any speed)
                speed_factor = 1.0
            target_steer_deg = steer_norm * self.gamepad.max_steer_deg * speed_factor
            target_steer_rad = np.deg2rad(target_steer_deg)
            
            # Steering rate limiter: instantaneous 1:1 tracking for physical wheels
            if is_wheel_dev:
                self.filtered_steer_rad = target_steer_rad
            else:
                max_rate_rad_s = np.deg2rad(720.0)
                max_delta_step = max_rate_rad_s * dt_real
                delta_diff = target_steer_rad - self.filtered_steer_rad
                delta_step = np.clip(delta_diff, -max_delta_step, max_delta_step)
                self.filtered_steer_rad += delta_step
            
            delta_vis_rad = self.filtered_steer_rad
            delta_physics_rad = -delta_vis_rad
            
            t_drive_total = cmd['throttle'] * (t_max_wheel * 2.0)
            drivetrain = getattr(self.vp, 'drivetrain_type', 'rwd')
            diff_mode = getattr(self.vp, 'diff_mode', getattr(self.vp, 'diff_type', 'dual_motor_open')).lower()

            w_fl = float(self.x_state[16])
            w_fr = float(self.x_state[17])
            w_rl = float(self.x_state[18])
            w_rr = float(self.x_state[19])
            Rw = self.vp.R_w

            delta_Mz = 0.0
            r_target = float(self.x_state[2])
            r_error = 0.0

            if drivetrain == 'fwd':
                # Front-Wheel Drive:
                # Open Differential distributes torque 50/50: T_fl = T_fr = T_drive_total / 2.0
                t_fl = t_drive_total / 2.0
                t_fr = t_drive_total / 2.0

                # Open Differential physical slip limitation:
                # If one wheel breaks traction (slips), the open differential cannot support higher reaction torque,
                # so the torque transmitted to BOTH wheels is limited by the traction of the slipping wheel.
                kappa_fl = (w_fl * Rw - vx_curr) / max(vx_curr, 1.0)
                kappa_fr = (w_fr * Rw - vx_curr) / max(vx_curr, 1.0)
                slip_diff = abs(kappa_fl - kappa_fr)

                if ('open' in diff_mode) and slip_diff > 0.12:
                    # Slip-limited torque clamping for open differential
                    slip_clamp = max(1.0 - (slip_diff - 0.12) * 4.0, 0.25)
                    t_fl *= slip_clamp
                    t_fr *= slip_clamp

                # Traction Control System for FWD
                if getattr(self, 'tcs_enabled', True) and (t_fl > 2.0 or t_fr > 2.0):
                    base_target_slip = float(getattr(self, 'tcs_target_slip', 0.095))
                    steer_demand = abs(delta_physics_rad) / max(np.deg2rad(15.0), 1e-4)
                    target_slip = float(max(base_target_slip - 0.045 * min(steer_demand, 1.0), 0.04))

                    k_fl_val = (w_fl * Rw - vx_curr) / max(vx_curr, 0.40)
                    k_fr_val = (w_fr * Rw - vx_curr) / max(vx_curr, 0.40)
                    excess_fl = max(k_fl_val - target_slip, 0.0)
                    excess_fr = max(k_fr_val - target_slip, 0.0)
                    cut_fl = float(np.clip(1.0 - 6.0 * excess_fl, 0.15, 1.0))
                    cut_fr = float(np.clip(1.0 - 6.0 * excess_fr, 0.15, 1.0))
                    t_fl *= cut_fl
                    t_fr *= cut_fr
                    min_cut = min(cut_fl, cut_fr)
                    self.tcs_active = bool(min_cut < 0.96)
                    self.tcs_cut_pct = float((1.0 - min_cut) * 100.0)
                elif drivetrain == 'fwd':
                    self.tcs_active = False
                    self.tcs_cut_pct = 0.0

                t_drive_l = t_fl
                t_drive_r = t_fr
                t_drv_fl_curr = float(t_fl)
                t_drv_fr_curr = float(t_fr)
                t_drv_rl_curr = 0.0
                t_drv_rr_curr = 0.0
            else:
                # Rear-Wheel Drive: 4 Dual-Motor Differential Modes
                t_half = t_drive_total / 2.0
                delta_w = w_rl - w_rr
                L_wb = float(getattr(self.vp, 'L', 1.530))
                tr_w = float(getattr(self.vp, 't_r', 1.200))
                r_curr = float(self.x_state[2])

                # Ideal kinematic/linear target yaw rate in vehicle chassis frame:
                # delta_physics_rad has the same sign convention as vehicle yaw rate r:
                # delta_physics_rad < 0 for right turn -> r < 0
                # delta_physics_rad > 0 for left turn  -> r > 0
                kus = float(getattr(self.vp, 'tv_kus', 0.0018))
                r_lin = (vx_curr / (L_wb + kus * (vx_curr**2))) * delta_physics_rad
                mu_pdy = float(getattr(self.tp, 'PDY1', 1.60)) * getattr(self, '_grip_multiplier', 1.0)
                r_max = (mu_pdy * 9.81) / max(vx_curr, 1.0)
                r_target = float(np.clip(r_lin, -r_max, r_max))
                r_error = r_target - r_curr

                if t_drive_total < 1.0:
                    # Coasting / Zero throttle: Electric motors do not apply positive drive torque
                    t_rl = 0.0
                    t_rr = 0.0
                    delta_Mz = 0.0
                elif 'spool' in diff_mode:
                    # 100% locked spool: dual motors deliver equal drive torque
                    t_rl = t_half
                    t_rr = t_half
                    delta_Mz = 0.0
                elif 'salisbury' in diff_mode or 'lsd' in diff_mode:
                    # Salisbury Multi-Plate clutch LSD with static preload and power ramp locking
                    t_pre = float(getattr(self.vp, 'diff_lsd_preload', 35.0))
                    mu_pow = float(getattr(self.vp, 'diff_lsd_mu_power', 0.40))
                    t_lock = (t_pre + mu_pow * abs(t_drive_total)) * math.tanh(3.0 * delta_w)
                    t_lock = float(np.clip(t_lock, -t_half, t_half))
                    t_rl = t_half - t_lock
                    t_rr = t_half + t_lock
                    delta_Mz = -t_lock * (tr_w / max(Rw, 0.1))
                elif 'torque_vectoring' in diff_mode or 'tv' in diff_mode:
                    # Active Torque Vectoring closed-loop yaw control
                    if vx_curr < 2.0:
                        t_rl = t_half
                        t_rr = t_half
                        delta_Mz = 0.0
                        self._tv_integral = 0.0
                    else:
                        kp = float(getattr(self.vp, 'tv_kp', 650.0))
                        ki = float(getattr(self.vp, 'tv_ki', 120.0))
                        max_mz = float(getattr(self.vp, 'tv_max_delta_mz', 500.0))

                        p_term = kp * r_error
                        i_cand = self._tv_integral + ki * r_error * dt_real
                        mz_cand = p_term + i_cand

                        if mz_cand > max_mz:
                            delta_Mz = max_mz
                            if r_error < 0.0:
                                self._tv_integral = i_cand
                        elif mz_cand < -max_mz:
                            delta_Mz = -max_mz
                            if r_error > 0.0:
                                self._tv_integral = i_cand
                        else:
                            delta_Mz = mz_cand
                            self._tv_integral = i_cand

                        delta_T = delta_Mz * (Rw / max(tr_w, 0.5))
                        delta_T = float(np.clip(delta_T, -t_half, t_half))
                        t_rl = t_half - delta_T
                        t_rr = t_half + delta_T
                else:
                    # DUAL_MOTOR_OPEN: 50/50 torque split
                    t_rl = t_half
                    t_rr = t_half
                    delta_Mz = 0.0

                # Physical torque clamping to wheel motor limits [0, t_max_wheel]
                t_rl = float(np.clip(t_rl, 0.0, t_max_wheel))
                t_rr = float(np.clip(t_rr, 0.0, t_max_wheel))

                # -------------------------------------------------------------
                # TRACTION CONTROL SYSTEM (TCS) — Switchable Closed-Loop Slip & Lateral Stability Regulation
                # -------------------------------------------------------------
                if getattr(self, 'tcs_enabled', True) and (t_rl > 2.0 or t_rr > 2.0):
                    base_target_slip = float(getattr(self, 'tcs_target_slip', 0.095))
                    steer_demand = abs(delta_physics_rad) / max(np.deg2rad(15.0), 1e-4)
                    target_slip = float(max(base_target_slip - 0.045 * min(steer_demand, 1.0), 0.04))

                    k_rl_val = (w_rl * Rw - vx_curr) / max(vx_curr, 0.40)
                    k_rr_val = (w_rr * Rw - vx_curr) / max(vx_curr, 0.40)

                    excess_rl = max(k_rl_val - target_slip, 0.0)
                    excess_rr = max(k_rr_val - target_slip, 0.0)

                    # Fast responsive closed-loop cut down to 0.0 torque (full cut)
                    cut_rl = float(np.clip(1.0 - 6.0 * excess_rl, 0.0, 1.0))
                    cut_rr = float(np.clip(1.0 - 6.0 * excess_rr, 0.0, 1.0))

                    t_rl *= cut_rl
                    t_rr *= cut_rr

                    min_cut = min(cut_rl, cut_rr)
                    self.tcs_active = bool(min_cut < 0.96)
                    self.tcs_cut_pct = float((1.0 - min_cut) * 100.0)
                else:
                    self.tcs_active = False
                    self.tcs_cut_pct = 0.0

                # Apply EMRAX 188HV motor curves, 4:1 gear reduction, and 80 kW accumulator electrical limit
                # Supports asymmetric Torque Vectoring (e.g. 50 kW outer, 30 kW inner)
                t_rl, t_rr, pt_telem = pt.allocate_dual_motor_torques(w_rl, w_rr, t_rl, t_rr)
                self._powertrain_telemetry = pt_telem

                t_drive_l = t_rl
                t_drive_r = t_rr
                t_drv_fl_curr = 0.0
                t_drv_fr_curr = 0.0
                t_drv_rl_curr = float(t_rl)
                t_drv_rr_curr = float(t_rr)

            t_brake_total = cmd['brake'] * t_brake_max_total
            u_input = [delta_physics_rad, t_drive_l, t_drive_r, t_brake_total]

            if getattr(self, '_params_dirty', False):
                with self._tuning_lock:
                    old_tail = list(p_vec[17:]) if len(p_vec) >= 37 else [0.0]*8 + [0.0, 0.0, 1.0]*4
                    p_vec = [
                        self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
                        self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
                        self.vp.camber_f_static, self.vp.camber_r_static,
                        self.vp.toe_f_static, self.vp.toe_r_static,
                        self.vp.L, self.vp.t_f, self.vp.t_r,
                        float(getattr(self.vp, 'LMUY', 1.0)),
                        float(getattr(self.vp, 'LMUX', 1.0)),
                        float(getattr(self.vp, 'Cd', 1.20)),
                    ] + old_tail
                    self._p_vec = p_vec
                    t_max_motor = getattr(self.vp, 'T_max_motor', 100.0)
                    final_drive = getattr(self.vp, 'final_drive', 4.0)
                    is_dual_motor = getattr(self.vp, 'diff_mode', 'DUAL_MOTOR_OPEN').upper().startswith('DUAL') or getattr(self.vp, 'drivetrain_type', 'rwd') == 'rwd'
                    t_max_wheel = (t_max_motor * final_drive) if is_dual_motor else (t_max_motor * final_drive) / 2.0
                    t_brake_max_total = float(getattr(self.vp, 'T_brake_max_total', 1400.0))
                    pt = getattr(self.vp, 'get_powertrain', lambda: None)()
                    if pt is None:
                        from src.models.powertrain_electric import get_default_powertrain
                        pt = get_default_powertrain()
                    if hasattr(self, 'tire_deg_model') and self.tire_deg_model is not None:
                        self.tire_deg_model.update_from_vehicle_params(self.vp)
                    self._params_dirty = False
                    self._tuning_dirty = False
            elif getattr(self, '_tuning_dirty', False):
                with self._tuning_lock:
                    p_vec[1] = self.vp.lltd
                    p_vec[2] = self.vp.K_phi_f
                    p_vec[3] = self.vp.K_phi_r
                    p_vec[4] = self.vp.K_z_f
                    p_vec[5] = self.vp.K_z_r
                    p_vec[6] = self.vp.aero_balance
                    p_vec[7] = self.vp.camber_f_static
                    p_vec[8] = self.vp.camber_r_static
                    p_vec[9] = self.vp.toe_f_static
                    p_vec[10] = self.vp.toe_r_static
                    if len(p_vec) > 16:
                        p_vec[16] = float(getattr(self.vp, 'Cd', 1.20))
                    else:
                        p_vec.append(float(getattr(self.vp, 'Cd', 1.20)))
                    while len(p_vec) < 37:
                        p_vec.append(0.0)
                    self._tuning_dirty = False

            # 3D Road Profile Evaluation & Contact Patches (Synchronized before RK4)
            z_road, grade_road, bank_road, wh_heights, wh_rates, wh_normals, wh_contacts, wh_r2d = self._update_road_contacts(p_vec)

            steps_done = 0
            while accumulator >= self.dt and steps_done < MAX_SUBSTEPS:
                t_loop_start = time.perf_counter()
                t_step = self.dt
                
                # RK4 Integration of 14-DOF ODE (Zero-Allocation In-Place)
                try:
                    self._k1[:] = np.asarray(self.ode(self.x_state, u_input, p_vec)).flatten()
                    np.add(self.x_state, 0.5 * t_step * self._k1, out=self._x_tmp)
                    self._k2[:] = np.asarray(self.ode(self._x_tmp, u_input, p_vec)).flatten()
                    np.add(self.x_state, 0.5 * t_step * self._k2, out=self._x_tmp)
                    self._k3[:] = np.asarray(self.ode(self._x_tmp, u_input, p_vec)).flatten()
                    np.add(self.x_state, t_step * self._k3, out=self._x_tmp)
                    self._k4[:] = np.asarray(self.ode(self._x_tmp, u_input, p_vec)).flatten()

                    np.add(self._k2, self._k3, out=self._x_tmp)
                    self._x_tmp *= 2.0
                    self._x_tmp += self._k1
                    self._x_tmp += self._k4
                    self._x_tmp *= (t_step / 6.0)
                    np.add(self.x_state, self._x_tmp, out=self._x_cand)

                    if not np.all(np.isfinite(self._x_cand)):
                        now_pc = time.perf_counter()
                        if getattr(self, '_last_nan_print', 0.0) + 1.0 < now_pc:
                            print(f"[PHYSICS WARN] Non-finite state detected: vx={self._x_cand[0]:.1f}, z_fl={self._x_cand[8]*1000:.1f}mm. Recovering.", flush=True)
                            self._last_nan_print = now_pc
                        self.reset_state(vx=0.0)
                    else:
                        # Defensive safety clamping against unphysical micro-divergences
                        self._x_cand[8:12] = np.clip(self._x_cand[8:12], -0.065, 0.065)  # max suspension stroke +/- 65mm
                        self._x_cand[3] = np.clip(self._x_cand[3], -1.05, 1.05)          # max roll ~60 deg
                        self._x_cand[5] = np.clip(self._x_cand[5], -0.87, 0.87)          # max pitch ~50 deg
                        self._x_cand[20:28] = np.clip(self._x_cand[20:28], -0.20, 0.20)  # carcass deflections
                        self.x_state[:] = self._x_cand
                except Exception:
                    pass

                if 'spool' in diff_mode:
                    w_spool = 0.5 * (self.x_state[18] + self.x_state[19])
                    self.x_state[18] = w_spool
                    self.x_state[19] = w_spool

                # Physical standstill stabilization:
                # With relaxation length, carcass elasticity naturally stabilizes the car at rest.
                v_planar = math.hypot(self.x_state[0], self.x_state[1])
                r_abs = abs(self.x_state[2])
                v_yaw_edge = r_abs * (self.vp.L / 2.0)
                v_motion = math.hypot(v_planar, v_yaw_edge)
                w_stopped = all(abs(self.x_state[16+i]) < 0.5 for i in range(4))

                if cmd['throttle'] < 0.01:
                    # Pitch angle (theta) provides gravity acceleration on slopes
                    theta_pitch = self.x_state[5]
                    gravity_accel = 9.81 * math.sin(theta_pitch) if abs(theta_pitch) > 0.001 else 0.0

                    # 1. Definite stop condition (braking at low speed, or stationary and not yawing on flat ground)
                    if (v_motion < 0.025 and (cmd['brake'] > 0.05 or w_stopped)) or (v_motion < 0.008 and r_abs < 0.02 and abs(gravity_accel) < 0.1):
                        self.x_state[0] = 0.0     # vx
                        self.x_state[1] = 0.0     # vy
                        self.x_state[2] = 0.0     # r
                        self.x_state[7] = 0.0     # w_heave
                        self.x_state[12:16] = 0.0 # vz_fl..vz_rr
                        self.x_state[16:20] = 0.0 # w_fl..w_rr
                        self.x_state[20:28] = 0.0 # Clear stored elastic carcass deflection (eliminates ghost launch)
                    # 2. Free rolling at sub-creeping speeds without throttle (and not spinning)
                    elif v_motion < 0.08 and abs(gravity_accel) < 0.1:
                        # Heavy numerical dissipation on relaxation lengths, vertical rates and wheel speed ringing only at true creeping speeds
                        self.x_state[12:16] *= 0.60
                        self.x_state[16:20] *= 0.85
                        self.x_state[20:28] *= 0.80
                
                vx = self.x_state[0]
                vy = self.x_state[1]
                r = self.x_state[2]
                phi = self.x_state[3]
                theta = self.x_state[5]
                w_heave = self.x_state[7]
                z_fl = self.x_state[8]
                z_fr = self.x_state[9]
                z_rl = self.x_state[10]
                z_rr = self.x_state[11]
                vz_fl = self.x_state[12]
                vz_fr = self.x_state[13]
                vz_rl = self.x_state[14]
                vz_rr = self.x_state[15]
                w_fl = self.x_state[16]
                w_fr = self.x_state[17]
                w_rl = self.x_state[18]
                w_rr = self.x_state[19]

                # Global Position (World +X forward, +Y left)
                self.psi += self.x_state[2] * t_step
                self.pos_x += (self.x_state[0] * np.cos(self.psi) - self.x_state[1] * np.sin(self.psi)) * t_step
                self.pos_y += (self.x_state[0] * np.sin(self.psi) + self.x_state[1] * np.cos(self.psi)) * t_step
                self.time_sim += t_step
                self.lap_timer += t_step
                self.dist_total += float(math.hypot(self.x_state[0], self.x_state[1]) * t_step)
                if self._sf_armed:
                    self.lap_time_current += t_step

                # Wheel spin angle integration [rad] for 3D rendering (wraps at 2π)
                _TWO_PI = 6.283185307179586
                self._wheel_angle[0] = (self._wheel_angle[0] + float(w_fl) * t_step) % _TWO_PI
                self._wheel_angle[1] = (self._wheel_angle[1] + float(w_fr) * t_step) % _TWO_PI
                self._wheel_angle[2] = (self._wheel_angle[2] + float(w_rl) * t_step) % _TWO_PI
                self._wheel_angle[3] = (self._wheel_angle[3] + float(w_rr) * t_step) % _TWO_PI

                
                # --- S/F Photocell Gate Crossing Detection ---
                if self._sf_cooldown > 0:
                    self._sf_cooldown -= t_step

                if self.track_data is not None:
                    x_cnt = self.track_data.get('x_center', np.array([]))
                    y_cnt = self.track_data.get('y_center', np.array([]))
                    N_pts = len(x_cnt)
                    if N_pts > 2:
                        x_sf = float(x_cnt[0])
                        y_sf = float(y_cnt[0])
                        dx_sf = float(x_cnt[1] - x_sf)
                        dy_sf = float(y_cnt[1] - y_sf)
                        L_sf = math.hypot(dx_sf, dy_sf)
                        tx = (dx_sf / L_sf) if L_sf > 1e-4 else 1.0
                        ty = (dy_sf / L_sf) if L_sf > 1e-4 else 0.0
                        nx = -ty
                        ny = tx

                        # Signed forward distance across S/F line plane along track tangent [m]
                        s_fwd = (self.pos_x - x_sf) * tx + (self.pos_y - y_sf) * ty
                        # Lateral offset from centerline [m]
                        d_lat = abs((self.pos_x - x_sf) * nx + (self.pos_y - y_sf) * ny)
                        dist_sf = math.hypot(self.pos_x - x_sf, self.pos_y - y_sf)

                        tw = float(self.track_data.get('track_width', 3.8))
                        w_gate_half = tw / 2.0 + 3.0  # gate extends 3m beyond kerbs for wide lines

                        is_closed_circuit = math.hypot(x_cnt[-1] - x_sf, y_cnt[-1] - y_sf) < 15.0

                        # S/F Gate forward crossing event
                        crossed_sf = (self._sf_dot_prev < 0.0 and s_fwd >= 0.0 and d_lat <= w_gate_half and dist_sf <= w_gate_half * 1.5)

                        if crossed_sf and self._sf_cooldown <= 0.0:
                            if not self._sf_armed:
                                # Crossing from grid: arm timer and start lap 1
                                self._sf_armed = True
                                self.lap_count = 1
                                self.lap_time_current = 0.0
                                self._lap_invalid = False
                                self._offtrack_4wheel_timer = 0.0
                                self._sf_cooldown = 1.5
                            elif is_closed_circuit and self.lap_time_current > 3.0:
                                # Completed a full lap on closed circuit
                                self.lap_time_last = self.lap_time_current
                                if not self._lap_invalid:
                                    # Valid lap: archive as best
                                    if self.lap_time_last < self.lap_time_best:
                                        self.lap_time_best = self.lap_time_last
                                        if len(self.current_lap_trajectory) > 10:
                                            self.best_lap_trajectory = list(self.current_lap_trajectory)
                                            self._cache_best_lap_arrays()
                                    self.lap_times_list.append(self.lap_time_last)
                                else:
                                    # Invalidated lap (4-wheel off-track): record as NaN, skip best
                                    self.lap_times_list.append(float('nan'))
                                self.lap_count += 1
                                self.lap_time_current = 0.0
                                self.current_lap_trajectory = []
                                self._lap_invalid = False
                                self._offtrack_4wheel_timer = 0.0
                                self._sf_cooldown = 1.5

                        # Handle finish line for OPEN tracks (at index N-1)
                        if not is_closed_circuit and self._sf_armed and self.lap_time_current > 2.0 and self._sf_cooldown <= 0.0:
                            x_fin = float(x_cnt[-1])
                            y_fin = float(y_cnt[-1])
                            dx_fin = float(x_cnt[-1] - x_cnt[-2])
                            dy_fin = float(y_cnt[-1] - y_cnt[-2])
                            L_fin = math.hypot(dx_fin, dy_fin)
                            tx_f = (dx_fin / L_fin) if L_fin > 1e-4 else 1.0
                            ty_f = (dy_fin / L_fin) if L_fin > 1e-4 else 0.0
                            s_fin = (self.pos_x - x_fin) * tx_f + (self.pos_y - y_fin) * ty_f
                            dist_fin = math.hypot(self.pos_x - x_fin, self.pos_y - y_fin)
                            if s_fin >= 0.0 and dist_fin <= w_gate_half * 1.5:
                                self.lap_time_last = self.lap_time_current
                                if not self._lap_invalid:
                                    if self.lap_time_last < self.lap_time_best:
                                        self.lap_time_best = self.lap_time_last
                                        if len(self.current_lap_trajectory) > 10:
                                            self.best_lap_trajectory = list(self.current_lap_trajectory)
                                            self._cache_best_lap_arrays()
                                    self.lap_times_list.append(self.lap_time_last)
                                else:
                                    self.lap_times_list.append(float('nan'))
                                self._sf_armed = False
                                self.current_lap_trajectory = []
                                self._lap_invalid = False
                                self._offtrack_4wheel_timer = 0.0
                                self._sf_cooldown = 2.0


                        # Always track continuous forward projection
                        self._sf_dot_prev = s_fwd

                        # Curvilinear track distance s_lap projection
                        s_cnt = self.track_data.get('s', None)
                        if s_cnt is not None and len(s_cnt) == N_pts:
                            c_idx = getattr(self, '_closest_track_idx', 0)
                            win = 30
                            idx_start = max(0, c_idx - win)
                            idx_end = min(N_pts, c_idx + win + 1)
                            dists_sq = (x_cnt[idx_start:idx_end] - self.pos_x)**2 + (y_cnt[idx_start:idx_end] - self.pos_y)**2
                            best_idx = idx_start + int(np.argmin(dists_sq))
                            self._closest_track_idx = best_idx
                            next_idx = (best_idx + 1) % N_pts
                            dx_seg = float(x_cnt[next_idx] - x_cnt[best_idx])
                            dy_seg = float(y_cnt[next_idx] - y_cnt[best_idx])
                            L_seg_sq = dx_seg * dx_seg + dy_seg * dy_seg
                            if L_seg_sq > 1e-4:
                                proj = ((self.pos_x - x_cnt[best_idx]) * dx_seg + (self.pos_y - y_cnt[best_idx]) * dy_seg) / L_seg_sq
                                proj_clamped = max(0.0, min(proj, 1.0))
                                L_seg = math.sqrt(L_seg_sq)
                                self.pos_s = float(s_cnt[best_idx] + proj_clamped * L_seg)
                            else:
                                self.pos_s = float(s_cnt[best_idx])
                        else:
                            self.pos_s = float(self.dist_total)

                        # Trajectory recording along active timed lap
                        if self._sf_armed:
                            if (not self.current_lap_trajectory or 
                                (self.pos_s - self.current_lap_trajectory[-1][0]) >= 0.5 or 
                                (self.lap_time_current - self.current_lap_trajectory[-1][5]) >= 0.02):
                                z_body = float(self.x_state[8] + self.x_state[9] + self.x_state[10] + self.x_state[11]) / 4.0
                                self.current_lap_trajectory.append((
                                    float(self.pos_s),
                                    float(self.pos_x),
                                    float(self.pos_y),
                                    float(z_body),
                                    float(self.psi),
                                    float(self.lap_time_current)
                                ))
                
                steps_count += 1
                
                t_step_calc = (time.perf_counter() - t_loop_start) * 1000.0
                self.physics_step_ms = 0.95 * getattr(self, 'physics_step_ms', 1.0) + 0.05 * t_step_calc

                accumulator -= self.dt
                steps_done += 1

            # --- PERFORMANCE MONITOR (250 ms) ---
            now_wall_2 = time.time()
            if now_wall_2 - last_hz_calc_time >= 0.25:
                dt_wall = now_wall_2 - last_hz_calc_time
                self.actual_physics_hz = float(steps_count / max(dt_wall, 0.001))
                self.real_time_ratio = float((steps_count * self.dt) / max(dt_wall, 0.001))
                steps_count = 0
                last_hz_calc_time = now_wall_2

            # --- GATED TELEMETRY, PACEJKA EVALUATION & UI SIGNAL EMISSION (60 Hz) ---
            now_pc = time.perf_counter()
            if now_pc - last_emit_time < getattr(self, 'target_emit_dt', 0.015):
                # Precision Hybrid Pacing for intermediate non-telemetry physics ticks
                next_tick_ns += target_dt_ns
                now_ns = time.perf_counter_ns()
                rem_ns = next_tick_ns - now_ns

                if rem_ns < -50_000_000:
                    next_tick_ns = time.perf_counter_ns()
                else:
                    if rem_ns > 1_500_000:
                        time.sleep((rem_ns - 1_200_000) / 1e9)
                    while time.perf_counter_ns() < next_tick_ns:
                        pass
                continue

            last_emit_time = now_pc

            # --- AFTER SUB-STEPS (TELEMETRY FRAME) ---
            vx = self.x_state[0]
            vy = self.x_state[1]
            v_planar = float(math.hypot(vx, vy))
            r = self.x_state[2]
            phi = self.x_state[3]
            theta = self.x_state[5]
            w_heave = self.x_state[7]
            z_fl = self.x_state[8]
            z_fr = self.x_state[9]
            z_rl = self.x_state[10]
            z_rr = self.x_state[11]
            vz_fl = self.x_state[12]
            vz_fr = self.x_state[13]
            vz_rl = self.x_state[14]
            vz_rr = self.x_state[15]
            w_fl = self.x_state[16]
            w_fr = self.x_state[17]
            w_rl = self.x_state[18]
            w_rr = self.x_state[19]
            ux_fl = self.x_state[20]
            uy_fl = self.x_state[21]
            ux_fr = self.x_state[22]
            uy_fr = self.x_state[23]
            ux_rl = self.x_state[24]
            uy_rl = self.x_state[25]
            ux_rr = self.x_state[26]
            uy_rr = self.x_state[27]

            L_m = self.vp.L
            a_dist = L_m * (1.0 - self.vp.wd_f)
            b_dist = L_m * self.vp.wd_f
            tf = max(self.vp.t_f, 0.5)
            tr = max(self.vp.t_r, 0.5)
            
            # Project road gravity acceleration (ISO 8855: +X fwd, +Y left, +Z up; roll > 0 tilts right down)
            ax_road_grav = 0.0
            ay_road_grav = 0.0

            if vx > 0.3:
                ay_ms2 = float(np.clip(vx * r + ay_road_grav, -25.0, 25.0))
                ax_raw = float(self._k1[0]) if len(self._k1) > 0 else 0.0
                ax_ms2 = float(np.clip(ax_raw + ax_road_grav, -25.0, 25.0))
            else:
                ay_ms2 = 0.0
                ax_ms2 = 0.0
            
            nu_caster = np.deg2rad(getattr(self.vp, 'caster_f', 5.5))
            sigma_kpi = np.deg2rad(getattr(self.vp, 'kpi_f', 6.5))
            d_spindle = getattr(self.vp, 'spindle_offset_f', 0.12)
            r_scrub = getattr(self.vp, 'scrub_radius_f', 0.018)
            
            # Decoupled Caster (differential) and KPI (heave) jacking
            dz_jack_diff = -d_spindle * np.sin(nu_caster) * np.sin(delta_physics_rad)
            dFz_jack_diff = self.vp.K_z_f * dz_jack_diff
            dFz_jack = dFz_jack_diff
            dz_jack_heave = r_scrub * np.sin(sigma_kpi) * (1.0 - np.cos(delta_physics_rad))
            dFz_jack_heave = self.vp.K_z_f * dz_jack_heave
            diag_split = (self.vp.K_z_r * tr) / (self.vp.K_z_f * tf + self.vp.K_z_r * tr)

            # Road Wheel Steer Angles (Adams Car direct curve with bilateral mirror symmetry)
            w_steer_coeffs = getattr(self.vp, 'wheel_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
            gr_steer = getattr(self.vp, 'steering_ratio', 5.0)
            delta_sw_deg = float(np.degrees(delta_physics_rad) * gr_steer)
            if any(abs(c) > 1e-6 for c in w_steer_coeffs[:3]):
                delta_fl_deg = float(eval_poly3(w_steer_coeffs, delta_sw_deg))
                delta_fr_deg = float(-eval_poly3(w_steer_coeffs, -delta_sw_deg))
            else:
                ack_pct = getattr(self.vp, 'ackermann_pct', 0.51)
                tan_d = math.tan(delta_physics_rad)
                L_m = self.vp.L
                tf_m = self.vp.t_f
                ack_fl_rad = math.atan2(L_m * tan_d, L_m - (tf_m / 2.0) * tan_d)
                ack_fr_rad = math.atan2(L_m * tan_d, L_m + (tf_m / 2.0) * tan_d)
                delta_fl_rad = delta_physics_rad + ack_pct * (ack_fl_rad - delta_physics_rad)
                delta_fr_rad = delta_physics_rad + ack_pct * (ack_fr_rad - delta_physics_rad)
                delta_fl_deg = float(np.degrees(delta_fl_rad))
                delta_fr_deg = float(np.degrees(delta_fr_rad))

            # Camber Steer Delta from steering angle [deg]
            camber_steer_coeffs = getattr(self.vp, 'camber_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
            dcam_steer_fl_deg = float(eval_poly3(camber_steer_coeffs, delta_fl_deg * gr_steer))
            dcam_steer_fr_deg = float(eval_poly3(camber_steer_coeffs, -delta_fr_deg * gr_steer))

            cam_fl_deg = float(eval_poly3(self.vp.camber_gain_f_coeffs, z_fl) + self.vp.camber_f_static + dcam_steer_fl_deg)
            cam_fr_deg = float(eval_poly3(self.vp.camber_gain_f_coeffs, z_fr) + self.vp.camber_f_static + dcam_steer_fr_deg)
            cam_rl_deg = float(eval_poly3(self.vp.camber_gain_r_coeffs, z_rl) + self.vp.camber_r_static)
            cam_rr_deg = float(eval_poly3(self.vp.camber_gain_r_coeffs, z_rr) + self.vp.camber_r_static)
            
            toe_fl_deg = float(eval_poly3(self.vp.toe_gain_f_coeffs, z_fl) + self.vp.toe_f_static)
            toe_fr_deg = float(eval_poly3(self.vp.toe_gain_f_coeffs, z_fr) + self.vp.toe_f_static)
            toe_rl_deg = float(eval_poly3(self.vp.toe_gain_r_coeffs, z_rl) + self.vp.toe_r_static)
            toe_rr_deg = float(eval_poly3(self.vp.toe_gain_r_coeffs, z_rr) + self.vp.toe_r_static)
            
            phi_deg = float(np.degrees(phi))
            h_rc_f_mm = float(eval_poly3(self.vp.rollcenter_f_coeffs, phi_deg) * 1000.0)
            h_rc_r_mm = float(eval_poly3(self.vp.rollcenter_r_coeffs, phi_deg) * 1000.0)
            y_rc_f_mm = float(eval_poly3(getattr(self.vp, 'rollcenter_lat_f_coeffs', [0.0, 0.0, 0.06453, 0.0]), phi_deg) * 1000.0)
            y_rc_r_mm = float(eval_poly3(getattr(self.vp, 'rollcenter_lat_r_coeffs', [0.0, 0.0, 0.04213, 0.0]), phi_deg) * 1000.0)
            
            # Dynamic CG height including chassis heave, roll and pitch
            z_f_avg = (z_fl + z_fr) / 2.0
            z_r_avg = (z_rl + z_rr) / 2.0
            z_cg = self.vp.wd_f * z_f_avg + (1.0 - self.vp.wd_f) * z_r_avg
            h_cg_dyn = max(0.05, self.vp.h_cg - z_cg)
            h_rc_avg_m = (h_rc_f_mm + h_rc_r_mm) * 0.0005
            y_rc_avg_m = (y_rc_f_mm + y_rc_r_mm) * 0.0005
            h_roll_arm = (h_cg_dyn - h_rc_avg_m) * np.cos(phi) - y_rc_avg_m * np.sin(phi)
            h_pitch_arm = h_cg_dyn * np.cos(theta)

            # Static 1g loads on front and rear axle (Reference z_i = 0 at static 1g ground equilibrium)
            Fz_0_f = self.vp.m * 9.81 * self.vp.wd_f
            Fz_0_r = self.vp.m * 9.81 * (1.0 - self.vp.wd_f)
            Fz_0_fl = Fz_0_f / 2.0
            Fz_0_fr = Fz_0_f / 2.0
            Fz_0_rl = Fz_0_r / 2.0
            Fz_0_rr = Fz_0_r / 2.0

            # Emergent vertical loads from radial tire spring and damping:
            # Fz_i = Fz_0_i + K_tz * defl_i + C_tz * (dz_road_i - vz_i) (+ jacking)
            K_tz_f = float(self.tp.get('VERTICAL_STIFFNESS', 75831.641)) if hasattr(self, 'tp') and self.tp else 75831.641
            K_tz_r = float(self.tp.get('VERTICAL_STIFFNESS', 75831.641)) if hasattr(self, 'tp') and self.tp else 75831.641
            C_tz = float(getattr(self.vp, 'C_tire_z', 500.0))
            Fz_max = self.vp.m * 9.81 * 1.50

            defl_0_fl = Fz_0_fl / K_tz_f
            defl_0_fr = Fz_0_fr / K_tz_f
            defl_0_rl = Fz_0_rl / K_tz_r
            defl_0_rr = Fz_0_rr / K_tz_r

            # Road Mean Plane Kinematics & Relative Tire Deflection
            L_wb = self.vp.L
            tf_m = self.vp.t_f
            tr_m = self.vp.t_r
            t_avg = (tf_m + tr_m) / 2.0
            z_road_f = (wh_heights['fl'] + wh_heights['fr']) / 2.0
            z_road_r = (wh_heights['rl'] + wh_heights['rr']) / 2.0
            z_mid = (z_road_f + z_road_r) / 2.0

            theta_road = - (z_road_f - z_road_r) / max(L_wb, 0.5)
            phi_road = ((wh_heights['fl'] + wh_heights['rl']) - (wh_heights['fr'] + wh_heights['rr'])) / (2.0 * max(t_avg, 0.5))

            z_plane_fl = z_mid - theta_road * (L_wb / 2.0) + phi_road * (tf_m / 2.0)
            z_plane_fr = z_mid - theta_road * (L_wb / 2.0) - phi_road * (tf_m / 2.0)
            z_plane_rl = z_mid + theta_road * (L_wb / 2.0) + phi_road * (tr_m / 2.0)
            z_plane_rr = z_mid + theta_road * (L_wb / 2.0) - phi_road * (tr_m / 2.0)

            dz_micro_fl = wh_heights['fl'] - z_plane_fl
            dz_micro_fr = wh_heights['fr'] - z_plane_fr
            dz_micro_rl = wh_heights['rl'] - z_plane_rl
            dz_micro_rr = wh_heights['rr'] - z_plane_rr

            defl_fl = defl_0_fl - z_fl + dz_micro_fl
            defl_fr = defl_0_fr - z_fr + dz_micro_fr
            defl_rl = defl_0_rl - z_rl + dz_micro_rl
            defl_rr = defl_0_rr - z_rr + dz_micro_rr

            gate_fl = 0.5 * (1.0 + np.tanh(defl_fl * 500.0))
            gate_fr = 0.5 * (1.0 + np.tanh(defl_fr * 500.0))
            gate_rl = 0.5 * (1.0 + np.tanh(defl_rl * 500.0))
            gate_rr = 0.5 * (1.0 + np.tanh(defl_rr * 500.0))

            Fz_fl = float(np.clip(gate_fl * max(0.0, K_tz_f * defl_fl + C_tz * (wh_rates['fl'] - vz_fl) + dFz_jack_diff + dFz_jack_heave), 0.0, Fz_max))
            Fz_fr = float(np.clip(gate_fr * max(0.0, K_tz_f * defl_fr + C_tz * (wh_rates['fr'] - vz_fr) - dFz_jack_diff + dFz_jack_heave), 0.0, Fz_max))
            Fz_rl = float(np.clip(gate_rl * max(0.0, K_tz_r * defl_rl + C_tz * (wh_rates['rl'] - vz_rl) - dFz_jack_diff * diag_split), 0.0, Fz_max))
            Fz_rr = float(np.clip(gate_rr * max(0.0, K_tz_r * defl_rr + C_tz * (wh_rates['rr'] - vz_rr) + dFz_jack_diff * diag_split), 0.0, Fz_max))

            # Lift-off detection: delta_z,i <= 0 or Fz,i <= 1.0 N
            liftoff_fl = bool(defl_fl <= 0.0 or Fz_fl <= 1.0)
            liftoff_fr = bool(defl_fr <= 0.0 or Fz_fr <= 1.0)
            liftoff_rl = bool(defl_rl <= 0.0 or Fz_rl <= 1.0)
            liftoff_rr = bool(defl_rr <= 0.0 or Fz_rr <= 1.0)
            
            # Contact Patch Relaxation Length Dynamic Slips
            sigma_x = float(getattr(self.vp, 'sigma_x', 0.15))
            sigma_y = float(getattr(self.vp, 'sigma_y', 0.22))

            kappa_dyn_fl = float(1.5 * np.tanh(ux_fl / (1.5 * sigma_x)))
            kappa_dyn_fr = float(1.5 * np.tanh(ux_fr / (1.5 * sigma_x)))
            kappa_dyn_rl = float(1.5 * np.tanh(ux_rl / (1.5 * sigma_x)))
            kappa_dyn_rr = float(1.5 * np.tanh(ux_rr / (1.5 * sigma_x)))

            alpha_dyn_fl_rad = float(np.arctan(1.5 * np.tanh(uy_fl / (1.5 * sigma_y)))) + np.deg2rad(toe_fl_deg)
            alpha_dyn_fr_rad = float(np.arctan(1.5 * np.tanh(uy_fr / (1.5 * sigma_y)))) - np.deg2rad(toe_fr_deg)
            alpha_dyn_rl_rad = float(np.arctan(1.5 * np.tanh(uy_rl / (1.5 * sigma_y)))) - np.deg2rad(toe_rl_deg)
            alpha_dyn_rr_rad = float(np.arctan(1.5 * np.tanh(uy_rr / (1.5 * sigma_y)))) + np.deg2rad(toe_rr_deg)

            alpha_fl = float(np.rad2deg(alpha_dyn_fl_rad))
            alpha_fr = float(np.rad2deg(alpha_dyn_fr_rad))
            alpha_rl = float(np.rad2deg(alpha_dyn_rl_rad))
            alpha_rr = float(np.rad2deg(alpha_dyn_rr_rad))

            p_press_f = getattr(self.vp, 'tire_pressure_f_bar', 0.85) * 1e5
            p_press_r = getattr(self.vp, 'tire_pressure_r_bar', 0.85) * 1e5

            tyreside = getattr(self.tp, 'get_tyreside', lambda: 'RIGHT')()
            is_right = (tyreside == 'RIGHT')

            # Camber angle in tire coordinate frame
            gamma_fl_rad = np.deg2rad(-cam_fl_deg if is_right else cam_fl_deg)
            gamma_fr_rad = np.deg2rad(cam_fr_deg if is_right else -cam_fr_deg)
            gamma_rl_rad = np.deg2rad(-cam_rl_deg if is_right else cam_rl_deg)
            gamma_rr_rad = np.deg2rad(cam_rr_deg if is_right else -cam_rr_deg)

            lvy_fl = -1.0 if is_right else 1.0
            lvy_fr =  1.0 if is_right else -1.0
            lvy_rl = -1.0 if is_right else 1.0
            lvy_rr =  1.0 if is_right else -1.0

            lhy_fl = -1.0 if is_right else 1.0
            lhy_fr =  1.0 if is_right else -1.0
            lhy_rl = -1.0 if is_right else 1.0
            lhy_rr =  1.0 if is_right else -1.0

            # Pure lateral contact patch forces (conforming to ISO chassis convention -mf61)
            Fy_fl_p = -float(mf61_lateral_force(alpha_dyn_fl_rad, Fz_fl, gamma_fl_rad, self.tp, LMUY=self.vp.LMUY, LVY=lvy_fl, LHY=lhy_fl, pressure=p_press_f))
            Fy_fr_p = -float(mf61_lateral_force(alpha_dyn_fr_rad, Fz_fr, gamma_fr_rad, self.tp, LMUY=self.vp.LMUY, LVY=lvy_fr, LHY=lhy_fr, pressure=p_press_f))
            Fy_rl_p = -float(mf61_lateral_force(alpha_dyn_rl_rad, Fz_rl, gamma_rl_rad, self.tp, LMUY=self.vp.LMUY, LVY=lvy_rl, LHY=lhy_rl, pressure=p_press_r))
            Fy_rr_p = -float(mf61_lateral_force(alpha_dyn_rr_rad, Fz_rr, gamma_rr_rad, self.tp, LMUY=self.vp.LMUY, LVY=lvy_rr, LHY=lhy_rr, pressure=p_press_r))

            # Longitudinal Contact Patch Forces Fx [N] (Pure & Combined MF6.1)
            Fx_fl_p = float(mf61_longitudinal_force(kappa_dyn_fl, Fz_fl, gamma_fl_rad, self.tp, LMUX=self.vp.LMUX, pressure=p_press_f))
            Fx_fr_p = float(mf61_longitudinal_force(kappa_dyn_fr, Fz_fr, gamma_fr_rad, self.tp, LMUX=self.vp.LMUX, pressure=p_press_f))
            Fx_rl_p = float(mf61_longitudinal_force(kappa_dyn_rl, Fz_rl, gamma_rl_rad, self.tp, LMUX=self.vp.LMUX, pressure=p_press_r))
            Fx_rr_p = float(mf61_longitudinal_force(kappa_dyn_rr, Fz_rr, gamma_rr_rad, self.tp, LMUX=self.vp.LMUX, pressure=p_press_r))

            # Combined slip forces in wheel frame (with TYRESIDE-consistent ply-steer and combined shift mirroring)
            Fxw_fl = float(mf61_combined_longitudinal_force(Fx_fl_p, kappa_dyn_fl, alpha_dyn_fl_rad, Fz_fl, gamma_fl_rad, self.tp, LHY=lhy_fl, pressure=p_press_f))
            Fyw_fl = float(mf61_combined_lateral_force(Fy_fl_p, alpha_dyn_fl_rad, kappa_dyn_fl, Fz_fl, gamma_fl_rad, self.tp, LVY=lvy_fl, LHY=lhy_fl, pressure=p_press_f))
            Fxw_fr = float(mf61_combined_longitudinal_force(Fx_fr_p, kappa_dyn_fr, alpha_dyn_fr_rad, Fz_fr, gamma_fr_rad, self.tp, LHY=lhy_fr, pressure=p_press_f))
            Fyw_fr = float(mf61_combined_lateral_force(Fy_fr_p, alpha_dyn_fr_rad, kappa_dyn_fr, Fz_fr, gamma_fr_rad, self.tp, LVY=lvy_fr, LHY=lhy_fr, pressure=p_press_f))

            Fxw_rl = float(mf61_combined_longitudinal_force(Fx_rl_p, kappa_dyn_rl, alpha_dyn_rl_rad, Fz_rl, gamma_rl_rad, self.tp, LHY=lhy_rl, pressure=p_press_r))
            Fyw_rl = float(mf61_combined_lateral_force(Fy_rl_p, alpha_dyn_rl_rad, kappa_dyn_rl, Fz_rl, gamma_rl_rad, self.tp, LVY=lvy_rl, LHY=lhy_rl, pressure=p_press_r))
            Fxw_rr = float(mf61_combined_longitudinal_force(Fx_rr_p, kappa_dyn_rr, alpha_dyn_rr_rad, Fz_rr, gamma_rr_rad, self.tp, LHY=lhy_rr, pressure=p_press_r))
            Fyw_rr = float(mf61_combined_lateral_force(Fy_rr_p, alpha_dyn_rr_rad, kappa_dyn_rr, Fz_rr, gamma_rr_rad, self.tp, LVY=lvy_rr, LHY=lhy_rr, pressure=p_press_r))

            # Rotate front steered wheel forces into chassis frame
            delta_fl_rad = np.deg2rad(delta_fl_deg)
            delta_fr_rad = np.deg2rad(delta_fr_deg)
            Fx_fl = float(Fxw_fl * np.cos(delta_fl_rad) - Fyw_fl * np.sin(delta_fl_rad))
            Fy_fl = float(Fxw_fl * np.sin(delta_fl_rad) + Fyw_fl * np.cos(delta_fl_rad))
            Fx_fr = float(Fxw_fr * np.cos(delta_fr_rad) - Fyw_fr * np.sin(delta_fr_rad))
            Fy_fr = float(Fxw_fr * np.sin(delta_fr_rad) + Fyw_fr * np.cos(delta_fr_rad))

            Fx_rl = Fxw_rl
            Fy_rl = Fyw_rl
            Fx_rr = Fxw_rr
            Fy_rr = Fyw_rr

            # Standstill deadband: when car is stopped at rest without throttle or braking,
            # zero out the raw Pacejka test bench offset (S_Vx) and static toe projection
            if v_planar < 0.05 and cmd.get('throttle', 0.0) < 0.01 and cmd.get('brake', 0.0) < 0.01:
                Fx_fl = 0.0
                Fx_fr = 0.0
                Fx_rl = 0.0
                Fx_rr = 0.0

            # Resultant Triaxial Vector Magnitude |F| = sqrt(Fx^2 + Fy^2 + Fz^2) [N]
            F_tot_fl = float(math.sqrt(Fx_fl**2 + Fy_fl**2 + Fz_fl**2))
            F_tot_fr = float(math.sqrt(Fx_fr**2 + Fy_fr**2 + Fz_fr**2))
            F_tot_rl = float(math.sqrt(Fx_rl**2 + Fy_rl**2 + Fz_rl**2))
            F_tot_rr = float(math.sqrt(Fx_rr**2 + Fy_rr**2 + Fz_rr**2))

            # Per-wheel Drive and Brake Torques [Nm]
            t_drv_fl = t_drv_fl_curr
            t_drv_fr = t_drv_fr_curr
            t_drv_rl = t_drv_rl_curr
            t_drv_rr = t_drv_rr_curr

            brk_bias = float(getattr(self.vp, 'brake_bias', 0.60))
            t_brk_fl = float((t_brake_total * brk_bias) / 2.0)
            t_brk_fr = float((t_brake_total * brk_bias) / 2.0)
            t_brk_rl = float((t_brake_total * (1.0 - brk_bias)) / 2.0)
            t_brk_rr = float((t_brake_total * (1.0 - brk_bias)) / 2.0)

            t_net_fl = float(t_drv_fl - t_brk_fl)
            t_net_fr = float(t_drv_fr - t_brk_fr)
            t_net_rl = float(t_drv_rl - t_brk_rl)
            t_net_rr = float(t_drv_rr - t_brk_rr)

            # Self-aligning torque & trail for steering FFB (driven by dynamic relaxed slips)
            Mz_fl = float(mf61_aligning_torque(alpha_dyn_fl_rad, Fz_fl, gamma_fl_rad, self.tp, LVY=lvy_fl, LHY=lhy_fl, pressure=p_press_f))
            Mz_fr = float(mf61_aligning_torque(alpha_dyn_fr_rad, Fz_fr, gamma_fr_rad, self.tp, LVY=lvy_fr, LHY=lhy_fr, pressure=p_press_f))

            t_p_fl = abs(Mz_fl / Fy_fl) if abs(Fy_fl) > 15.0 else 0.022
            t_p_fr = abs(Mz_fr / Fy_fr) if abs(Fy_fr) > 15.0 else 0.022
            t_m_m = getattr(self.vp, 'mech_trail_m', (self.vp.R_w * np.tan(nu_caster)) / max(np.cos(sigma_kpi), 0.1))

            # Kingpin Steering Torque at handwheel [Nm]
            M_kp_fl = Fy_fl * (t_p_fl + t_m_m) * np.cos(nu_caster) * np.cos(sigma_kpi)
            M_kp_fr = Fy_fr * (t_p_fr + t_m_m) * np.cos(nu_caster) * np.cos(sigma_kpi)
            steer_ratio = getattr(self.vp, 'steering_ratio', 4.5)
            rack_eff = 0.90
            T_sw_phys = abs(M_kp_fl + M_kp_fr) / (steer_ratio * rack_eff)
            T_sw_signed = float((M_kp_fl + M_kp_fr) / (steer_ratio * rack_eff))

            mu_peak_tir = getattr(self.tp, 'PDY1', 1.6918) * getattr(self.vp, 'LMUY', 1.0)
            util_fl = min((abs(Fy_fl) / max(mu_peak_tir * Fz_fl, 1.0)) * 100.0, 100.0)
            util_fr = min((abs(Fy_fr) / max(mu_peak_tir * Fz_fr, 1.0)) * 100.0, 100.0)
            util_rl = min((abs(Fy_rl) / max(mu_peak_tir * Fz_rl, 1.0)) * 100.0, 100.0)
            util_rr = min((abs(Fy_rr) / max(mu_peak_tir * Fz_rr, 1.0)) * 100.0, 100.0)

            Rw = getattr(self.vp, 'R_w', 0.203)
            kappa_fl = float(kappa_dyn_fl * 100.0)
            kappa_fr = float(kappa_dyn_fr * 100.0)
            kappa_rl = float(kappa_dyn_rl * 100.0)
            kappa_rr = float(kappa_dyn_rr * 100.0)

            # Traction Loss & Wheelspin (Right Motor Buzz / Throttle RT) with hysteresis
            if not hasattr(self, '_tc_active'):
                self._tc_active = False
                self._tc_hold_until = 0.0
                self._abs_active = False
                self._abs_hold_until = 0.0

            if drivetrain == 'fwd':
                kappa_driven_max = max(kappa_fl, kappa_fr)
                w_driven_max = max(w_fl, w_fr)
            else:
                kappa_driven_max = max(kappa_rl, kappa_rr)
                w_driven_max = max(w_rl, w_rr)

            # Slip ratio hysteresis: ON at kappa >= 0.12 (12%), OFF at kappa <= 0.08 (8%), min hold time 80 ms
            kappa_driven_norm = max(kappa_driven_max / 100.0, 0.0)
            if self._tc_active:
                if kappa_driven_norm < 0.08 and self.time_sim > self._tc_hold_until:
                    self._tc_active = False
            else:
                if kappa_driven_norm >= 0.12:
                    self._tc_active = True
                    self._tc_hold_until = self.time_sim + 0.080

            throttle_factor = min(cmd['throttle'] * 1.4, 1.0)
            is_spinning = (vx > 0.1) or (cmd['throttle'] > 0.15 and w_driven_max * Rw > 0.5)
            if self._tc_active and is_spinning:
                rt_carrier = 0.65 + 0.35 * math.sin(2.0 * math.pi * 38.0 * self.time_sim)
                haptic_throt_rt = float(np.clip(rt_carrier * min(kappa_driven_norm / 0.20, 1.0) * throttle_factor, 0.20, 1.0))
            else:
                haptic_throt_rt = 0.0

            # Mechanical Brake Lockup / ABS (Left Motor Rumble / Brake LT) with hysteresis
            max_brake_slip = max(-kappa_fl, -kappa_fr, -kappa_rl, -kappa_rr)
            brake_slip_norm = max(max_brake_slip / 100.0, 0.0)
            if self._abs_active:
                if brake_slip_norm < 0.08 and self.time_sim > self._abs_hold_until:
                    self._abs_active = False
            else:
                if brake_slip_norm >= 0.12:
                    self._abs_active = True
                    self._abs_hold_until = self.time_sim + 0.080

            brake_factor = min(cmd['brake'] * 1.4, 1.0)
            if self._abs_active and vx > 0.2:
                lt_carrier = 0.55 + 0.45 * math.sin(2.0 * math.pi * 16.0 * self.time_sim)
                haptic_brake_lt = float(np.clip(lt_carrier * min(brake_slip_norm / 0.20, 1.0) * brake_factor, 0.20, 1.0))
            else:
                haptic_brake_lt = 0.0

            # Force Feedback & Tactile Haptics (Physical, Responsive & Smooth)
            road_gain = getattr(self.gamepad, 'road_feel_gain', 0.20)
            grip_gain = getattr(self.gamepad, 'grip_loss_gain', 0.75)

            # 1. Front Grip Loss / Understeer (Light Motor)
            # Communicates tire scrub energy past peak slip angle without violent rattle
            alpha_peak_deg = 7.5
            front_slip_excess = max(abs(alpha_fl) - alpha_peak_deg, abs(alpha_fr) - alpha_peak_deg, 0.0)
            front_util_excess = max(util_fl - 84.0, util_fr - 84.0, 0.0) / 16.0
            scrub_envelope = float(np.clip(max(front_slip_excess / 3.0, front_util_excess), 0.0, 1.0))
            understeer_ratio = scrub_envelope
            scrub_carrier = 0.70 + 0.30 * math.sin(2.0 * math.pi * 28.0 * self.time_sim)
            scrub_pulse = float(scrub_envelope * scrub_carrier * 0.45 * grip_gain) if vx > 1.0 else 0.0

            # 2. Road Surface Texture & 3D Micro-Roughness (Light Motor)
            # Continuous, tactile feel from tire contact patch vertical displacements & pavement grain
            vz_fl = self.x_state[12]
            vz_fr = self.x_state[13]
            K_haptic = 15.0
            vib_raw = (abs(vz_fl) + abs(vz_fr)) * K_haptic
            alpha_vib = 1.0 - math.exp(-2.0 * math.pi * 40.0 * self.dt)
            self._vib_road_ema = getattr(self, '_vib_road_ema', 0.0)
            self._vib_road_ema = alpha_vib * vib_raw + (1.0 - alpha_vib) * self._vib_road_ema
            road_texture = float(np.clip(self._vib_road_ema * road_gain, 0.0, 0.45)) if vx > 0.4 else 0.0

            # 3. Rear Grip Loss & Chassis Oversteer Snap (Heavy Motor)
            # Responsive warning when rear tires slide or vehicle sideslip diverges
            rear_slip_excess = max(abs(alpha_rl) - alpha_peak_deg, abs(alpha_rr) - alpha_peak_deg, 0.0)
            beta_deg = float(np.rad2deg(np.arctan2(vy, max(vx, 0.5))))
            drift_envelope = float(np.clip((abs(beta_deg) - 2.5) / 4.0, 0.0, 1.0))
            rear_util_excess = max(util_rl - 84.0, util_rr - 84.0, 0.0) / 16.0
            oversteer_envelope = float(np.clip(max(rear_slip_excess / 3.0, drift_envelope, rear_util_excess), 0.0, 1.0))
            oversteer_ratio = oversteer_envelope
            oversteer_carrier = 0.70 + 0.30 * math.sin(2.0 * math.pi * 20.0 * self.time_sim)
            oversteer_pulse = float(oversteer_envelope * oversteer_carrier * 0.50 * grip_gain) if vx > 1.0 else 0.0

            # 4. Suspension Bumpstop bottoming impact (contacts in last 5mm of shock stroke)
            gap_f = getattr(self.vp, 'bumpstop_gap_f_m', None)
            if gap_f is None:
                gap_f = (getattr(self.vp, 'damper_stroke_total_mm', 57.0) - 5.0) / (getattr(self.vp, 'motion_ratio_f', 0.8) * 1000.0)
            gap_r = getattr(self.vp, 'bumpstop_gap_r_m', None)
            if gap_r is None:
                gap_r = (getattr(self.vp, 'damper_stroke_total_mm', 57.0) - 5.0) / (getattr(self.vp, 'motion_ratio_r', 0.8) * 1000.0)
            max_comp = max(z_fl - gap_f, z_fr - gap_f, z_rl - gap_r, z_rr - gap_r, 0.0)
            bumpstop_thud = float(np.clip(max_comp / 0.005, 0.0, 0.50))

            # 5. Kerbs & Off-Track Vibration + Progressive Smooth Grip + Lap Invalidation
            kerb_rumble = 0.0
            offtrack_rumble = 0.0
            surface_grip_mult = 1.0  # Full grip on tarmac
            full_offtrack_4wheel = False  # All 4 wheels outside track limits
            if self.track_data is not None:
                xc_tr = self.track_data.get('x_center', np.array([]))
                yc_tr = self.track_data.get('y_center', np.array([]))
                if len(xc_tr) > 0 and vx > 1.0:
                    dx_tr = xc_tr - self.pos_x
                    dy_tr = yc_tr - self.pos_y
                    dist_sq = dx_tr * dx_tr + dy_tr * dy_tr
                    min_d = float(np.sqrt(np.min(dist_sq)))
                    half_w = float(getattr(self.vp, 'track_width_m', 3.8) / 2.0)
                    if (half_w - 0.45) <= min_d <= (half_w + 0.50):
                        kerb_freq = max(vx * 3.6, 28.0)
                        kerb_rumble = float((0.50 + 0.50 * math.sin(2.0 * math.pi * kerb_freq * self.time_sim)) * 0.40)
                    elif min_d > (half_w + 0.50):
                        offtrack_rumble = float((0.50 + 0.50 * math.sin(2.0 * math.pi * 18.0 * self.time_sim)) * 0.35 * min(vx / 6.0, 1.0))
                        # Gentle progressive transition only when deep in the grass (> 2.0m off track)
                        if min_d > (half_w + 2.0):
                            excess = min_d - (half_w + 2.0)
                            surface_grip_mult = max(1.0 - 0.20 * min(excess / 2.0, 1.0), 0.80)

                    # 4-wheel off-track check: car center > half_w + half_car_track (all 4 wheels clear)
                    car_half_track = float(getattr(self.vp, 't_f', 1.25)) / 2.0 + 0.03
                    full_offtrack_4wheel = (min_d > (half_w + car_half_track))

            # Lap invalidation: accumulate 4-wheel off-track time and flag if threshold exceeded
            if full_offtrack_4wheel:
                self._offtrack_4wheel_timer += dt_real
                if self._offtrack_4wheel_timer >= self._offtrack_4wheel_thresh:
                    self._lap_invalid = True
            else:
                self._offtrack_4wheel_timer = max(self._offtrack_4wheel_timer - dt_real * 2.0, 0.0)


            # Step tire degradation model per-wheel if enabled (with separate integration rate)
            tire_deg_mult = 1.0
            tire_grip_fl = 1.0; tire_grip_fr = 1.0; tire_grip_rl = 1.0; tire_grip_rr = 1.0
            if self.tire_deg_model.mode != "off":
                self._thermal_accumulator += dt_real
                dt_thermal_target = (1.0 / self.thermal_hz) if self.thermal_hz > 0.0 else 0.0
                if self.thermal_hz <= 0.0 or self._thermal_accumulator >= dt_thermal_target:
                    step_dt = self._thermal_accumulator if self.thermal_hz > 0.0 else dt_real
                    corners = [
                        (self.tire_states[0], Fy_fl, kappa_fl / 100.0, alpha_fl, Fz_fl),
                        (self.tire_states[1], Fy_fr, kappa_fr / 100.0, alpha_fr, Fz_fr),
                        (self.tire_states[2], Fy_rl, kappa_rl / 100.0, alpha_rl, Fz_rl),
                        (self.tire_states[3], Fy_rr, kappa_rr / 100.0, alpha_rr, Fz_rr),
                    ]
                    for ts, Fy_i, k_i, a_deg, Fz_i in corners:
                        v_slip_long = abs(vx * k_i)
                        v_slip_lat = abs(vx * math.tan(math.radians(a_deg)))
                        v_slip_tot = math.hypot(v_slip_long, v_slip_lat)
                        F_tot = math.hypot(Fy_i, Fz_i * abs(k_i))
                        self.tire_deg_model.step(ts, F_tot, v_slip_tot, Fz_i, vx, step_dt)
                    self._thermal_accumulator = 0.0

                tire_grip_fl = float(self.tire_deg_model.grip_modifier(self.tire_states[0]))
                tire_grip_fr = float(self.tire_deg_model.grip_modifier(self.tire_states[1]))
                tire_grip_rl = float(self.tire_deg_model.grip_modifier(self.tire_states[2]))
                tire_grip_rr = float(self.tire_deg_model.grip_modifier(self.tire_states[3]))
                tire_deg_mult = float(0.25 * (tire_grip_fl + tire_grip_fr + tire_grip_rl + tire_grip_rr))

            # Apply dynamic grip: track_grip × surface × tire_degradation
            base_lmuy = getattr(self.vp, 'LMUY', 1.0)
            base_lmux = getattr(self.vp, 'LMUX', 1.0)
            track_grip = self._sim_opts_track_grip
            combined_grip = track_grip * surface_grip_mult * tire_deg_mult
            self._grip_multiplier = combined_grip
            p_vec[14] = base_lmuy * combined_grip   # LMUY
            p_vec[15] = base_lmux * combined_grip   # LMUX

            # Composite Force Feedback Motors (Smooth, Responsive & Anti-Jerk)
            rumble_heavy = float(np.clip(oversteer_pulse + bumpstop_thud + offtrack_rumble * 0.65, 0.0, 1.0))
            rumble_light = float(np.clip(scrub_pulse + road_texture + kerb_rumble * 0.65, 0.0, 1.0))

            # Superimposed Procedural Tactile Oscillations for PC Racing Wheels (30-60 Hz kerbs, texture, slip)
            kerb_freq = max(vx * 3.6, 28.0) if vx > 1.0 else 30.0
            wheel_kerb_vibe = math.sin(2.0 * math.pi * kerb_freq * self.time_sim) * 0.35 if kerb_rumble > 0.02 else 0.0
            wheel_road_vibe = math.sin(2.0 * math.pi * 26.0 * self.time_sim) * (0.04 + getattr(self, '_vib_road_ema', 0.0) * 0.35) * road_gain if vx > 0.4 else 0.0
            wheel_scrub_vibe = math.sin(2.0 * math.pi * 32.0 * self.time_sim) * scrub_envelope * 0.30 * grip_gain if vx > 1.0 else 0.0
            wheel_oversteer_vibe = math.sin(2.0 * math.pi * 20.0 * self.time_sim) * oversteer_envelope * 0.30 * grip_gain if vx > 1.0 else 0.0
            wheel_tactile_vibe = float(np.clip(wheel_kerb_vibe + wheel_road_vibe + wheel_scrub_vibe + wheel_oversteer_vibe, -0.60, 0.60))

            self.gamepad.set_vibration(
                rumble_heavy=rumble_heavy,
                rumble_light=rumble_light,
                trigger_left=haptic_brake_lt,
                trigger_right=haptic_throt_rt,
                steer_torque=T_sw_signed,
                tactile_vibe=wheel_tactile_vibe
            )

            # Continuous Spatial Delta-Lap & Ghost Car Interpolation (Zero-Allocation Fast C-Interp)
            if self._sf_armed and self._ghost_s is not None and len(self._ghost_s) >= 5:
                try:
                    t_best_at_s = float(np.interp(self.pos_s, self._ghost_s, self._ghost_t))
                    self.live_delta_lap = float(self.lap_time_current - t_best_at_s)

                    if self.lap_time_current <= self._ghost_t[-1]:
                        x_g = float(np.interp(self.lap_time_current, self._ghost_t, self._ghost_x))
                        y_g = float(np.interp(self.lap_time_current, self._ghost_t, self._ghost_y))
                        z_g = float(np.interp(self.lap_time_current, self._ghost_t, self._ghost_z))
                        psi_g = float(np.interp(self.lap_time_current, self._ghost_t, self._ghost_psi))
                        self.ghost_car_state = {'x': x_g, 'y': y_g, 'z': z_g, 'psi': psi_g}
                    else:
                        self.ghost_car_state = None
                except Exception:
                    self.live_delta_lap = 0.0
                    self.ghost_car_state = None
            else:
                self.live_delta_lap = 0.0
                self.ghost_car_state = None

            gear_ratio = float(getattr(self.vp, 'final_drive', 4.0))
            w_rl_live = float(self.x_state[18])
            w_rr_live = float(self.x_state[19])
            rpm_live_rl = (abs(w_rl_live) * gear_ratio * (60.0 / (2.0 * np.pi))) if (v_planar >= 0.01 or abs(w_rl_live) >= 0.1) else 0.0
            rpm_live_rr = (abs(w_rr_live) * gear_ratio * (60.0 / (2.0 * np.pi))) if (v_planar >= 0.01 or abs(w_rr_live) >= 0.1) else 0.0

            telem_data = {
                't': float(self.time_sim),
                'vehicle_type': str(getattr(self.vp, 'vehicle_type', 'formula')),
                'x': float(self.pos_x),
                'y': float(self.pos_y),
                'psi_deg': float(np.rad2deg(self.psi)),
                'vx_kmh': float(vx * 3.6),
                'vy_kmh': float(vy * 3.6),
                'beta_deg': float(beta_deg),
                'r_deg_s': float(np.rad2deg(r)),
                'ay_g': float(ay_ms2 / 9.81),
                'ax_g': float(ax_ms2 / 9.81),
                'phi_deg': float(np.rad2deg(phi)),
                'theta_deg': float(np.rad2deg(theta)),
                'steer_deg': float(np.rad2deg(delta_vis_rad)),
                'steer_fl_deg': float(delta_fl_deg),
                'steer_fr_deg': float(delta_fr_deg),
                'throttle_pct': float(cmd.get('throttle', 0.0) * 100.0),
                'brake_pct': float(cmd.get('brake', 0.0) * 100.0),
                'raw_steer': float(cmd.get('raw_steer', cmd.get('steer', 0.0))),
                'ctrl_connected': bool(cmd.get('connected', False)),
                'ctrl_device': str(cmd.get('device_name', 'Keyboard')),
                'is_wheel': bool(cmd.get('is_wheel', False)),
                'T_sw_signed_Nm': float(T_sw_signed),
                'haptic_brake_lt': float(haptic_brake_lt * 100.0),
                'haptic_throt_rt': float(haptic_throt_rt * 100.0),
                'rumble_heavy': float(rumble_heavy * 100.0),
                'rumble_light': float(rumble_light * 100.0),
                'understeer_pct': float(understeer_ratio * 100.0),
                'oversteer_pct': float(oversteer_ratio * 100.0),
                'traction_loss_pct': float(haptic_throt_rt * 100.0),
                'lockup_pct': float(haptic_brake_lt * 100.0),
                'tcs_enabled': bool(getattr(self, 'tcs_enabled', True)),
                'tcs_active': bool(getattr(self, 'tcs_active', False)),
                'tcs_cut_pct': float(getattr(self, 'tcs_cut_pct', 0.0)),
                'z_fl_mm': float(z_fl * 1000.0), 'z_fr_mm': float(z_fr * 1000.0),
                'z_rl_mm': float(z_rl * 1000.0), 'z_rr_mm': float(z_rr * 1000.0),
                'defl_fl_mm': float(defl_fl * 1000.0), 'defl_fr_mm': float(defl_fr * 1000.0),
                'defl_rl_mm': float(defl_rl * 1000.0), 'defl_rr_mm': float(defl_rr * 1000.0),
                'z_road_fl_mm': float(wh_heights['fl'] * 1000.0), 'z_road_fr_mm': float(wh_heights['fr'] * 1000.0),
                'z_road_rl_mm': float(wh_heights['rl'] * 1000.0), 'z_road_rr_mm': float(wh_heights['rr'] * 1000.0),
                'liftoff_fl': bool(liftoff_fl), 'liftoff_fr': bool(liftoff_fr),
                'liftoff_rl': bool(liftoff_rl), 'liftoff_rr': bool(liftoff_rr),
                'ux_fl_mm': float(ux_fl * 1000.0), 'ux_fr_mm': float(ux_fr * 1000.0),
                'ux_rl_mm': float(ux_rl * 1000.0), 'ux_rr_mm': float(ux_rr * 1000.0),
                'uy_fl_mm': float(uy_fl * 1000.0), 'uy_fr_mm': float(uy_fr * 1000.0),
                'uy_rl_mm': float(uy_rl * 1000.0), 'uy_rr_mm': float(uy_rr * 1000.0),
                'camber_fl_deg': float(cam_fl_deg), 'camber_fr_deg': float(cam_fr_deg),
                'camber_rl_deg': float(cam_rl_deg), 'camber_rr_deg': float(cam_rr_deg),
                'toe_fl_deg': float(toe_fl_deg), 'toe_fr_deg': float(toe_fr_deg),
                'toe_rl_deg': float(toe_rl_deg), 'toe_rr_deg': float(toe_rr_deg),
                'h_rc_f_mm': float(h_rc_f_mm), 'h_rc_r_mm': float(h_rc_r_mm),
                'y_rc_f_mm': float(y_rc_f_mm), 'y_rc_r_mm': float(y_rc_r_mm),
                'dFz_jack_N': float(dFz_jack),
                'T_sw_Nm': float(T_sw_phys),
                't_p_fl_mm': float(t_p_fl * 1000.0),
                't_p_fr_mm': float(t_p_fr * 1000.0),
                'kerb_active': bool(kerb_rumble > 0.05),
                'offtrack_active': bool(offtrack_rumble > 0.05),
                'offtrack_4wheel': bool(full_offtrack_4wheel),
                'offtrack_4wheel_timer': float(self._offtrack_4wheel_timer),
                'lap_invalid': bool(self._lap_invalid),
                'wheel_angle_fl': float(self._wheel_angle[0]),
                'wheel_angle_fr': float(self._wheel_angle[1]),
                'wheel_angle_rl': float(self._wheel_angle[2]),
                'wheel_angle_rr': float(self._wheel_angle[3]),
                'Fz_fl': float(Fz_fl), 'Fz_fr': float(Fz_fr),
                'Fz_rl': float(Fz_rl), 'Fz_rr': float(Fz_rr),
                'Fy_fl': float(Fy_fl), 'Fy_fr': float(Fy_fr),
                'Fy_rl': float(Fy_rl), 'Fy_rr': float(Fy_rr),
                'Fx_fl': float(Fx_fl), 'Fx_fr': float(Fx_fr),
                'Fx_rl': float(Fx_rl), 'Fx_rr': float(Fx_rr),
                'F_tot_fl': float(F_tot_fl), 'F_tot_fr': float(F_tot_fr),
                'F_tot_rl': float(F_tot_rl), 'F_tot_rr': float(F_tot_rr),
                'r_contact_fl_x': float(wh_contacts['fl'][0]), 'r_contact_fl_y': float(wh_contacts['fl'][1]), 'r_contact_fl_z': float(wh_contacts['fl'][2]),
                'r_contact_fr_x': float(wh_contacts['fr'][0]), 'r_contact_fr_y': float(wh_contacts['fr'][1]), 'r_contact_fr_z': float(wh_contacts['fr'][2]),
                'r_contact_rl_x': float(wh_contacts['rl'][0]), 'r_contact_rl_y': float(wh_contacts['rl'][1]), 'r_contact_rl_z': float(wh_contacts['rl'][2]),
                'r_contact_rr_x': float(wh_contacts['rr'][0]), 'r_contact_rr_y': float(wh_contacts['rr'][1]), 'r_contact_rr_z': float(wh_contacts['rr'][2]),
                'n_road_fl_x': float(wh_normals['fl'][0]), 'n_road_fl_y': float(wh_normals['fl'][1]), 'n_road_fl_z': float(wh_normals['fl'][2]),
                'n_road_fr_x': float(wh_normals['fr'][0]), 'n_road_fr_y': float(wh_normals['fr'][1]), 'n_road_fr_z': float(wh_normals['fr'][2]),
                'n_road_rl_x': float(wh_normals['rl'][0]), 'n_road_rl_y': float(wh_normals['rl'][1]), 'n_road_rl_z': float(wh_normals['rl'][2]),
                'n_road_rr_x': float(wh_normals['rr'][0]), 'n_road_rr_y': float(wh_normals['rr'][1]), 'n_road_rr_z': float(wh_normals['rr'][2]),
                'T_drive_fl': float(t_drv_fl), 'T_drive_fr': float(t_drv_fr),
                'T_drive_rl': float(t_drv_rl), 'T_drive_rr': float(t_drv_rr),
                'p_elec_total_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_elec_total_kw', 0.0)),
                'p_elec_rl_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_elec_rl_kw', 0.0)),
                'p_elec_rr_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_elec_rr_kw', 0.0)),
                'p_mech_total_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_mech_total_kw', 0.0)),
                'p_mech_rl_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_mech_rl_kw', 0.0)),
                'p_mech_rr_kw': float(getattr(self, '_powertrain_telemetry', {}).get('p_mech_rr_kw', 0.0)),
                't_motor_rl_nm': float(getattr(self, '_powertrain_telemetry', {}).get('t_motor_rl_nm', 0.0)),
                't_motor_rr_nm': float(getattr(self, '_powertrain_telemetry', {}).get('t_motor_rr_nm', 0.0)),
                'rpm_motor_rl': float(rpm_live_rl),
                'rpm_motor_rr': float(rpm_live_rr),
                'w_fl_rad_s': float(self.x_state[16]),
                'w_fr_rad_s': float(self.x_state[17]),
                'w_rl_rad_s': float(w_rl_live),
                'w_rr_rad_s': float(w_rr_live),
                'v_wheel_rl_kmh': float(w_rl_live * Rw * 3.6),
                'v_wheel_rr_kmh': float(w_rr_live * Rw * 3.6),
                'eta_motor_rl': float(getattr(self, '_powertrain_telemetry', {}).get('eta_rl', 0.95)),
                'eta_motor_rr': float(getattr(self, '_powertrain_telemetry', {}).get('eta_rr', 0.95)),
                'is_power_limited': bool(getattr(self, '_powertrain_telemetry', {}).get('is_power_limited', False)),
                'T_brake_fl': float(t_brk_fl), 'T_brake_fr': float(t_brk_fr),
                'T_brake_rl': float(t_brk_rl), 'T_brake_rr': float(t_brk_rr),
                'T_net_fl': float(t_net_fl), 'T_net_fr': float(t_net_fr),
                'T_net_rl': float(t_net_rl), 'T_net_rr': float(t_net_rr),
                'alpha_fl': float(alpha_fl), 'alpha_fr': float(alpha_fr),
                'alpha_rl': float(alpha_rl), 'alpha_rr': float(alpha_rr),
                'kappa_fl': float(kappa_fl), 'kappa_fr': float(kappa_fr),
                'kappa_rl': float(kappa_rl), 'kappa_rr': float(kappa_rr),
                'util_fl': float(util_fl), 'util_fr': float(util_fr),
                'util_rl': float(util_rl), 'util_rr': float(util_rr),
                'lap_time': float(self.lap_timer),
                'lap_count': int(self.lap_count),
                'lap_idx': int(self.lap_count),
                's_lap': float(self.pos_s),
                'dist_total': float(self.dist_total),
                'lap_time_current': float(self.lap_time_current),
                'lap_time_last': float(self.lap_time_last),
                'lap_time_best': float(self.lap_time_best) if self.lap_time_best < 1e6 else 0.0,
                'lap_delta': float(self.lap_time_current - self.lap_time_best) if self.lap_time_best < 1e6 and self._sf_armed else 0.0,
                'live_delta_lap': float(self.live_delta_lap),
                'diff_mode': str(diff_mode),
                'Delta_Mz_Nm': float(delta_Mz),
                'r_target_degs': float(np.rad2deg(r_target)),
                'r_error_degs': float(np.rad2deg(r_error)),
                'ghost_car': self.ghost_car_state,
                'sf_armed': bool(self._sf_armed),
                'tire_T_fl': float(self.tire_deg_model.get_display_temperature(self.tire_states[0])),
                'tire_T_fr': float(self.tire_deg_model.get_display_temperature(self.tire_states[1])),
                'tire_T_rl': float(self.tire_deg_model.get_display_temperature(self.tire_states[2])),
                'tire_T_rr': float(self.tire_deg_model.get_display_temperature(self.tire_states[3])),
                # 1-Zone Bulk Temperatures
                'tire_T_bulk_fl': float(self.tire_states[0].T_bulk),
                'tire_T_bulk_fr': float(self.tire_states[1].T_bulk),
                'tire_T_bulk_rl': float(self.tire_states[2].T_bulk),
                'tire_T_bulk_rr': float(self.tire_states[3].T_bulk),
                # 3-Zone Detailed Layer Temperatures
                'tire_T_surf_fl': float(self.tire_states[0].T_surface),
                'tire_T_surf_fr': float(self.tire_states[1].T_surface),
                'tire_T_surf_rl': float(self.tire_states[2].T_surface),
                'tire_T_surf_rr': float(self.tire_states[3].T_surface),
                'tire_T_tread_fl': float(self.tire_states[0].T_tread),
                'tire_T_tread_fr': float(self.tire_states[1].T_tread),
                'tire_T_tread_rl': float(self.tire_states[2].T_tread),
                'tire_T_tread_rr': float(self.tire_states[3].T_tread),
                'tire_T_carc_fl': float(self.tire_states[0].T_carcass),
                'tire_T_carc_fr': float(self.tire_states[1].T_carcass),
                'tire_T_carc_rl': float(self.tire_states[2].T_carcass),
                'tire_T_carc_rr': float(self.tire_states[3].T_carcass),
                # Tire Wear & Per-Corner Grip Multipliers
                'tire_wear_fl': float(self.tire_states[0].wear_fraction * 100.0),
                'tire_wear_fr': float(self.tire_states[1].wear_fraction * 100.0),
                'tire_wear_rl': float(self.tire_states[2].wear_fraction * 100.0),
                'tire_wear_rr': float(self.tire_states[3].wear_fraction * 100.0),
                'tire_grip_fl': float(tire_grip_fl),
                'tire_grip_fr': float(tire_grip_fr),
                'tire_grip_rl': float(tire_grip_rl),
                'tire_grip_rr': float(tire_grip_rr),
                'grip_multiplier': float(self._grip_multiplier),
                # 3D Road Profile Telemetry
                'z_road': float(z_road),
                'roughness_fl_mm': float(wh_r2d.get('fl', 0.0) * 1000.0),
                'grade_road': float(grade_road),
                'bank_road_deg': float(np.rad2deg(bank_road)),
                'z_ground_fl': float(wh_heights['fl']),
                'z_ground_fr': float(wh_heights['fr']),
                'z_ground_rl': float(wh_heights['rl']),
                'z_ground_rr': float(wh_heights['rr']),
                'physics_hz': float(self.actual_physics_hz),
                'thermal_hz': float(self.thermal_hz),
                'step_ms': float(self.physics_step_ms),
                'rt_ratio': float(self.real_time_ratio),
                # 28-State 2nd-Order Heave & Suspension Vertical Velocities
                'w_heave_ms': float(self.x_state[7]),
                'vz_fl_ms': float(self.x_state[12]),
                'vz_fr_ms': float(self.x_state[13]),
                'vz_rl_ms': float(self.x_state[14]),
                'vz_rr_ms': float(self.x_state[15])
            }

            with self._history_lock:
                if len(self.telemetry_history) < self.max_history:
                    self.telemetry_history.append(telem_data)

            self.state_updated.emit(telem_data)

            # Precision Hybrid Pacing: Sleep per chunk grandi + Spinloop per jitter sub-microsecondo
            next_tick_ns += target_dt_ns
            now_ns = time.perf_counter_ns()
            rem_ns = next_tick_ns - now_ns

            if rem_ns < -50_000_000:
                next_tick_ns = time.perf_counter_ns()
            else:
                if rem_ns > 1_500_000:
                    time.sleep((rem_ns - 1_200_000) / 1e9)
                while time.perf_counter_ns() < next_tick_ns:
                    pass

        # Cleanup al termine del ciclo di guida
        if is_windows:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
        import gc
        gc.enable()
        gc.collect()

    def stop(self):
        self.running = False
        try:
            self.gamepad.set_vibration(0.0, 0.0, 0.0, 0.0, steer_torque=0.0, tactile_vibe=0.0)
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
        self.wait()


# =============================================================================
# 4. 3D & 2D VIEWPORT WITH ACCURATE PERSPECTIVE & STEERING KINEMATICS
# =============================================================================







