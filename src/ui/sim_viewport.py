from src.utils.track_manager import TrackCollisionMesh, extrude_verge_geometry
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
except Exception as e:
    print('HAS_OPENGL FAILED:', e)
    HAS_OPENGL = False
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
from src.core.physics_thread import PhysicsThread14DOF

class OpenGLViewport3D(QOpenGLWidget):
    """Hardware-Accelerated OpenGL 3D & 2D Viewport with Volumetric 3D Race Car & Smooth Skid Marks."""
    
    GRASS_COLOR_RGB = (72, 170, 68)     # #48aa44
    BASE_TERRAIN_RGB = (56, 138, 52)    # #388a34
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if HAS_OPENGL:
            try:
                fmt = QSurfaceFormat()
                fmt.setDepthBufferSize(24)
                fmt.setStencilBufferSize(8)
                fmt.setSwapInterval(0)
                self.setFormat(fmt)
            except Exception:
                pass
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.car_state = None
        self.trail = []
        self.max_trail = 300
        self.view_mode = "3D"
        self.is_fullscreen_view = False
        self.show_minimap = True
        
        from src.mdo.mdo_direct_collocation import build_track_geometry
        self.track_data = build_track_geometry(track_length_m=850.0, ds=2.0, track_width_m=3.8)
        self.track_mesh = TrackCollisionMesh(self.track_data, road_surface_2d=getattr(self, 'road_surface_2d', None))
        
        self.cam_dist = 6.8
        self.cam_height = 3.6
        self.fov = 420.0
        self.pitch_offset = 0.0
        self.camera_mode = "CHASE"
        
        self.smooth_heading = 0.0
        self.smooth_x = 0.0
        self.smooth_y = 0.0

        # Onboard First-Person Camera Smoothing & Damping States
        self.onboard_cam_pos = None
        self.onboard_fwd_vec = None
        self.onboard_up_vec = None
        self.onboard_theta_smooth = None
        self.onboard_phi_smooth = None
        self.onboard_psi_smooth = None
        self.onboard_z_smooth = None

        # Fast precomputed trigonometric tables (12 segments)
        self.n_seg_fast = 12
        _angles = np.linspace(0, 2 * np.pi, self.n_seg_fast + 1)
        self.cos_tab = np.cos(_angles)
        self.sin_tab = np.sin(_angles)

        # Continuous Skid Mark Ribbons
        self.skid_ribbons = []
        self.prev_wheel_patches = {'fl': None, 'fr': None, 'rl': None, 'rr': None}
        self.max_skid_age = 4.0  # seconds to persist (optimized for smooth framerate)
        self.max_skid_ribbons = 300

        # Rendering FPS performance tracking & smoothing
        self._fps_last_time = time.perf_counter()
        self.fps_smooth = 60.0
        self._frame_count = 0

        # 3D Vehicle Models (FSAE Single-Seater and B-Segment Compact SUV)
        from src.models.vehicle_3d_models import FSAECar3DModel, SUVCar3DModel
        self.fsae_model = FSAECar3DModel()
        self.suv_model = SUVCar3DModel()
        self.vp = None

        # Precompiled OpenGL Display Lists for static track ribbon & kerbs
        self.track_display_list = None
        self.track_display_list_valid = False
        self._last_compiled_track_id = None
        self._last_compiled_track_version = None

    def invalidate_track_display_list(self):
        """Forces OpenGL to recompile the 3D track ribbon display list on the next paintGL frame."""
        self.track_display_list_valid = False
        self._last_compiled_track_id = None
        self._last_compiled_track_version = None

    def reset_camera_smoothing(self):
        """Resets camera low-pass filter states to prevent spatial jumps or transitions from old locations."""
        for attr in ('_cam_ema_x', '_cam_ema_y', '_cam_ema_z', '_look_ema_x', '_look_ema_y', '_look_ema_z'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass
        self.onboard_cam_pos = None
        self.onboard_fwd_vec = None
        self.onboard_up_vec = None
        self.onboard_theta_smooth = None
        self.onboard_phi_smooth = None
        self.onboard_psi_smooth = None
        self.onboard_z_smooth = None
        if self.car_state:
            self.smooth_x = float(self.car_state.get('x', 0.0))
            self.smooth_y = float(self.car_state.get('y', 0.0))
            self.smooth_heading = np.deg2rad(float(self.car_state.get('psi_deg', 0.0)))
        self.trail.clear()
        self.skid_ribbons.clear()

    def set_vehicle_params(self, vp):
        self.vp = vp
        if hasattr(self, 'fsae_model'):
            self.fsae_model.invalidate_display_list()
        if hasattr(self, 'suv_model'):
            self.suv_model.invalidate_display_list()

    def set_track_data(self, track_data):
        self.track_data = track_data
        self.track_mesh = TrackCollisionMesh(track_data, road_surface_2d=getattr(self, 'road_surface_2d', None))
        self.invalidate_track_display_list()

    def keyPressEvent(self, event):
        win = self.window()
        if win and hasattr(win, 'keyPressEvent'):
            win.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        win = self.window()
        if win and hasattr(win, 'keyReleaseEvent'):
            win.keyReleaseEvent(event)
        else:
            super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        win = self.window()
        if win and hasattr(win, 'toggle_fullscreen'):
            win.toggle_fullscreen()
        else:
            super().mouseDoubleClickEvent(event)

    def update_vehicle_state(self, state):
        self.car_state = state
        self.trail.append((state['x'], state['y']))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
            
        rad = np.deg2rad(state['psi_deg'])
        if getattr(self, 'camera_mode', 'CHASE') in ('COCKPIT', 'NOSE', 'BUMPER'):
            # Strict lockstep for onboard cameras to eliminate spatial lag
            self.smooth_x = state['x']
            self.smooth_y = state['y']
            self.smooth_heading = rad
        elif math.hypot(state['x'] - self.smooth_x, state['y'] - self.smooth_y) > 8.0:
            self.smooth_x = state['x']
            self.smooth_y = state['y']
            self.smooth_heading = rad
            self.trail.clear()
            self.skid_ribbons.clear()
        else:
            # Circular shortest-path angle unwrapping to eliminate 360-deg spins at +/- pi
            d_psi = (rad - self.smooth_heading + np.pi) % (2.0 * np.pi) - np.pi
            self.smooth_heading = (self.smooth_heading + 0.15 * d_psi + np.pi) % (2.0 * np.pi) - np.pi
            self.smooth_x = 0.85 * self.smooth_x + 0.15 * state['x']
            self.smooth_y = 0.85 * self.smooth_y + 0.15 * state['y']
        
        # Update Continuous Smooth Skid Mark Ribbons (Prune expired ribbons every 0.5s)
        now = time.time()
        if now - getattr(self, '_last_skid_prune', 0.0) >= 0.5:
            self.skid_ribbons = [r for r in self.skid_ribbons if now - r['time'] < self.max_skid_age]
            if len(self.skid_ribbons) > getattr(self, 'max_skid_ribbons', 300):
                self.skid_ribbons = self.skid_ribbons[-getattr(self, 'max_skid_ribbons', 300):]
            self._last_skid_prune = now
        
        yaw = np.deg2rad(state['psi_deg'])
        cos_p = np.cos(yaw)
        sin_p = np.sin(yaw)
        steer_rad = np.deg2rad(state.get('steer_deg', 0.0))
        
        L = 1.53; W = 1.25; a = L * 0.53; b = L * 0.47; tf_half = W / 2.0
        w_tire = 0.18
        
        wheel_offsets = {
            'fl': (a, tf_half, steer_rad),
            'fr': (a, -tf_half, steer_rad),
            'rl': (-b, tf_half, 0.0),
            'rr': (-b, -tf_half, 0.0)
        }
        
        for corner, (lx, ly, delta_w) in wheel_offsets.items():
            wx = state['x'] + lx * cos_p - ly * sin_p
            wy = state['y'] + lx * sin_p + ly * cos_p
            psi_w = yaw + delta_w
            nx = -np.sin(psi_w) * (w_tire / 2.0)
            ny = np.cos(psi_w) * (w_tire / 2.0)
            
            rc_x = state.get('r_contact_' + corner + '_x', wx)
            rc_y = state.get('r_contact_' + corner + '_y', wy)
            rc_z = state.get('r_contact_' + corner + '_z', 0.0)
            n_x = state.get('n_road_' + corner + '_x', 0.0)
            n_y = state.get('n_road_' + corner + '_y', 0.0)
            n_z = state.get('n_road_' + corner + '_z', 1.0)
            
            cx = rc_x + 0.003 * n_x
            cy = rc_y + 0.003 * n_y
            cz = rc_z + 0.003 * n_z
            
            p_left = (cx + nx, cy + ny, cz)
            p_right = (cx - nx, cy - ny, cz)
            
            util = state.get('util_' + corner, 0.0)
            kappa_abs = abs(state.get('kappa_' + corner, 0.0))
            is_skidding = (util > 82.0) or (kappa_abs > 14.0)
            
            if is_skidding and (state.get('vx_kmh', 0.0) > 3.0):
                intensity = float(np.clip((util - 80.0) / 20.0 * 0.6 + (kappa_abs / 40.0) * 0.4, 0.25, 1.0))
                prev = self.prev_wheel_patches[corner]
                if prev is not None:
                    dist = math.hypot(wx - prev['center'][0], wy - prev['center'][1])
                    if 0.15 < dist < 2.5:
                        self.skid_ribbons.append({
                            'p1': prev['left'],
                            'p2': prev['right'],
                            'p3': p_right,
                            'p4': p_left,
                            'time': now,
                            'intensity': intensity
                        })
                self.prev_wheel_patches[corner] = {'center': (wx, wy), 'left': p_left, 'right': p_right}
            else:
                self.prev_wheel_patches[corner] = None
        self.update()

    def project_3d_point(self, x_world, y_world, z_world, cam_x, cam_y, cam_z, cam_yaw):
        dx = x_world - cam_x
        dy = y_world - cam_y
        dz = z_world - cam_z
        y_depth = dx * np.cos(cam_yaw) + dy * np.sin(cam_yaw)
        if y_depth <= 0.6: return None
        x_cam_right = dx * np.sin(cam_yaw) - dy * np.cos(cam_yaw)
        z_cam_up = dz
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 + 35.0
        sx = cx + (x_cam_right / y_depth) * self.fov
        sy = cy - (z_cam_up / y_depth) * self.fov
        return QPointF(sx, sy), float(y_depth)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.cam_dist = max(1.0, self.cam_dist - 0.5)
        elif delta < 0:
            self.cam_dist = min(15.0, self.cam_dist + 0.5)
        self.update()

    def initializeGL(self):
        try:
            from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT
            wglSwapIntervalEXT(0)
        except Exception:
            pass
        glClearColor(176/255.0, 218/255.0, 245/255.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_FLAT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_CULL_FACE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        actual_fov = 90.0 - (self.fov - 300) / 500.0 * 60.0 
        aspect = w / (h if h > 0 else 1)
        gluPerspective(actual_fov, aspect, 0.25, 1500.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Measure rendering frame delta time & smoothed FPS
            now_time = time.perf_counter()
            dt_frame = now_time - getattr(self, '_fps_last_time', now_time)
            self._fps_last_time = now_time
            if 0.0005 < dt_frame < 1.0:
                inst_fps = 1.0 / dt_frame
                self.fps_smooth = 0.90 * getattr(self, 'fps_smooth', 60.0) + 0.10 * inst_fps
            self._frame_count = getattr(self, '_frame_count', 0) + 1

            if not self.car_state:
                return

            cam_mode = getattr(self, 'camera_mode', 'CHASE')
            is_onboard = (cam_mode in ('COCKPIT', 'NOSE', 'BUMPER')) or (abs(self.cam_dist) < 0.1 and self.cam_height < 0.9)
            if is_onboard:
                x_car = float(self.car_state['x'])
                y_car = float(self.car_state['y'])
                yaw_car = np.deg2rad(float(self.car_state['psi_deg']))
            else:
                x_car = self.smooth_x
                y_car = self.smooth_y
                yaw_car = self.smooth_heading
            
            if self.view_mode == "3D":
                w = self.width()
                h = self.height()
                aspect = w / (h if h > 0 else 1)
                
                z_car = float(self.car_state.get('z_road', 0.0))
                
                if is_onboard:
                    actual_fov = 75.0 if cam_mode == 'BUMPER' else 72.0
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluPerspective(actual_fov, aspect, 0.25, 1500.0)
                    
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    
                    # Decoupled time-continuous exponential camera filter
                    # Separates physics integration frequency (250-500 Hz) from render frame-rate (60-144 Hz)
                    dt = max(0.001, min(float(dt_frame), 0.05)) if 'dt_frame' in locals() and dt_frame > 0 else 0.016
                    alpha_pos = 1.0 - math.exp(-42.0 * dt)
                    alpha_rot = 1.0 - math.exp(-30.0 * dt)
                    alpha_pitch = 1.0 - math.exp(-16.0 * dt)
                    alpha_z = 1.0 - math.exp(-20.0 * dt)

                    is_suv = getattr(self.vp, 'vehicle_type', 'formula') == 'b_segment_suv'
                    xv = float(self.car_state['x'])
                    yv = float(self.car_state['y'])
                    
                    # Smooth road elevation & vertical suspension compression (absorbs curbs and altimetric jumps)
                    z_target = float(self.car_state.get('z_road', 0.0))
                    if self.onboard_z_smooth is None or abs(z_target - self.onboard_z_smooth) > 3.0:
                        self.onboard_z_smooth = z_target
                    else:
                        self.onboard_z_smooth += alpha_z * (z_target - self.onboard_z_smooth)
                    zv = self.onboard_z_smooth + (0.315 if is_suv else 0.20)

                    # Shortest-path smoothed yaw to eliminate any 360-deg wrapping discontinuities
                    psi_raw = np.deg2rad(float(self.car_state['psi_deg']))
                    if self.onboard_psi_smooth is None:
                        self.onboard_psi_smooth = psi_raw
                    d_psi = (psi_raw - self.onboard_psi_smooth + np.pi) % (2.0 * np.pi) - np.pi
                    self.onboard_psi_smooth = (self.onboard_psi_smooth + alpha_rot * d_psi + np.pi) % (2.0 * np.pi) - np.pi
                    psi_car = self.onboard_psi_smooth

                    # Smooth pitch angle and filter out road grade steps & bumps
                    theta_raw = np.deg2rad(float(self.car_state.get('theta_deg', 0.0)))
                    grade_raw = float(self.car_state.get('grade_road', 0.0))
                    theta_target = theta_raw - math.atan(grade_raw)
                    if self.onboard_theta_smooth is None or abs(theta_target - self.onboard_theta_smooth) > 0.35:
                        self.onboard_theta_smooth = theta_target
                    else:
                        self.onboard_theta_smooth += alpha_pitch * (theta_target - self.onboard_theta_smooth)
                    theta_car = self.onboard_theta_smooth

                    # Smooth roll angle
                    phi_raw = np.deg2rad(float(self.car_state.get('phi_deg', 0.0)) + float(self.car_state.get('bank_road_deg', 0.0)))
                    if self.onboard_phi_smooth is None or abs(phi_raw - self.onboard_phi_smooth) > 0.35:
                        self.onboard_phi_smooth = phi_raw
                    else:
                        self.onboard_phi_smooth += alpha_pitch * (phi_raw - self.onboard_phi_smooth)
                    phi_car = self.onboard_phi_smooth

                    c_psi = np.cos(psi_car); s_psi = np.sin(psi_car)
                    c_th  = np.cos(theta_car); s_th  = np.sin(theta_car)
                    c_phi = np.cos(phi_car); s_phi = np.sin(phi_car)

                    # Exact 3D rotation matrix matching glRotatef(psi, 0,0,1) * glRotatef(theta, 0,1,0) * glRotatef(phi, 1,0,0)
                    R00 = c_psi * c_th
                    R01 = c_psi * s_th * s_phi - s_psi * c_phi
                    R02 = c_psi * s_th * c_phi + s_psi * s_phi

                    R10 = s_psi * c_th
                    R11 = s_psi * s_th * s_phi + c_psi * c_phi
                    R12 = s_psi * s_th * c_phi - c_psi * s_phi

                    R20 = -s_th
                    R21 = c_th * s_phi
                    R22 = c_th * c_phi

                    # Eye coordinates in chassis local frame (+X fwd, +Y left, +Z up):
                    if cam_mode == 'NOSE':
                        loc_x = 1.53; loc_y = 0.0; loc_z = 0.28
                    elif cam_mode == 'BUMPER':
                        loc_x = 1.50; loc_y = 0.0; loc_z = 0.08
                    elif is_suv:
                        loc_x = 0.04; loc_y = 0.35; loc_z = 0.88
                    else:
                        loc_x = 0.22; loc_y = 0.0; loc_z = 0.46

                    target_cam_x = xv + R00 * loc_x + R01 * loc_y + R02 * loc_z
                    target_cam_y = yv + R10 * loc_x + R11 * loc_y + R12 * loc_z
                    target_cam_z = zv + R20 * loc_x + R21 * loc_y + R22 * loc_z
                    target_pos = np.array([target_cam_x, target_cam_y, target_cam_z], dtype=float)

                    if self.onboard_cam_pos is None or np.linalg.norm(target_pos - self.onboard_cam_pos) > 4.0:
                        self.onboard_cam_pos = target_pos.copy()
                    else:
                        self.onboard_cam_pos += alpha_pos * (target_pos - self.onboard_cam_pos)
                    cam_x, cam_y, cam_z = self.onboard_cam_pos[0], self.onboard_cam_pos[1], self.onboard_cam_pos[2]

                    # Filtered forward and up vectors with spherical orthonormalization
                    raw_fwd = np.array([R00, R10, R20], dtype=float)
                    raw_up = np.array([R02, R12, R22], dtype=float)
                    if self.onboard_fwd_vec is None:
                        self.onboard_fwd_vec = raw_fwd.copy()
                        self.onboard_up_vec = raw_up.copy()
                    else:
                        self.onboard_fwd_vec += alpha_rot * (raw_fwd - self.onboard_fwd_vec)
                        self.onboard_up_vec += alpha_rot * (raw_up - self.onboard_up_vec)

                    fwd_mag = np.linalg.norm(self.onboard_fwd_vec)
                    fwd = self.onboard_fwd_vec / (fwd_mag if fwd_mag > 1e-6 else 1.0)
                    up_ortho = self.onboard_up_vec - np.dot(self.onboard_up_vec, fwd) * fwd
                    up_mag = np.linalg.norm(up_ortho)
                    up = up_ortho / (up_mag if up_mag > 1e-6 else 1.0)

                    look_dist = 40.0
                    pitch_tune = (self.pitch_offset * 0.04) - 0.02
                    look_x = cam_x + look_dist * fwd[0] + pitch_tune * up[0]
                    look_y = cam_y + look_dist * fwd[1] + pitch_tune * up[1]
                    look_z = cam_z + look_dist * fwd[2] + pitch_tune * up[2]

                    gluLookAt(cam_x, cam_y, cam_z,
                              look_x, look_y, look_z,
                              up[0], up[1], up[2])
                else:
                    actual_fov = 90.0 - (self.fov - 300) / 500.0 * 60.0
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluPerspective(actual_fov, aspect, 0.25, 1500.0)
                    
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    
                    vp_obj = getattr(self, 'vp', None)
                    v_type = getattr(vp_obj, 'vehicle_type', None) or (self.car_state.get('vehicle_type', 'formula') if self.car_state else 'formula')
                    is_suv = (v_type == 'b_segment_suv')
                    dist_eff = (self.cam_dist * 1.30 + 1.0) if is_suv else self.cam_dist
                    h_eff = (self.cam_height * 1.25 + 0.35) if is_suv else self.cam_height
                    z_target = (z_car + 0.85) if is_suv else (z_car + 0.5)

                    # Compensate road slope grade so camera follows elevation profile naturally
                    grade_rd = float(self.car_state.get('grade_road', 0.0))
                    cam_x = x_car - dist_eff * np.cos(yaw_car)
                    cam_y = y_car - dist_eff * np.sin(yaw_car)
                    cam_z = z_car - dist_eff * grade_rd + h_eff

                    look_x = x_car + 4.5 * np.cos(yaw_car)
                    look_y = y_car + 4.5 * np.sin(yaw_car)
                    look_z = z_target + 4.5 * grade_rd + self.pitch_offset * 0.1

                    gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 0, 1)
                
                # 1. Sky Quad (rendered with depth write disabled so distant terrain and verges are never culled)
                glDepthMask(GL_FALSE)
                glPolygonOffset(4.0, 4.0)
                glColor3f(176/255.0, 218/255.0, 245/255.0)
                glBegin(GL_QUADS)
                glVertex3f(cam_x + 500 * np.cos(yaw_car) - 500 * np.sin(yaw_car), cam_y + 500 * np.sin(yaw_car) + 500 * np.cos(yaw_car), z_car + 150.0)
                glVertex3f(cam_x + 500 * np.cos(yaw_car) + 500 * np.sin(yaw_car), cam_y + 500 * np.sin(yaw_car) - 500 * np.cos(yaw_car), z_car + 150.0)
                glVertex3f(cam_x + 500 * np.cos(yaw_car) + 500 * np.sin(yaw_car), cam_y + 500 * np.sin(yaw_car) - 500 * np.cos(yaw_car), z_car - 20.0)
                glVertex3f(cam_x + 500 * np.cos(yaw_car) - 500 * np.sin(yaw_car), cam_y + 500 * np.sin(yaw_car) + 500 * np.cos(yaw_car), z_car - 20.0)
                glEnd()
                glDepthMask(GL_TRUE)

                # 2. Rigid Ground Plane removed in favor of 3D Extruded Verge Mesh Skirt (Assetto Corsa style)

                # 3. Dynamic Track Rendering (Hardware Accelerated via Precompiled Display List)
                if self.track_data is not None:
                    track_id = id(self.track_data)
                    track_version = self.track_data.get('_version', 0) if isinstance(self.track_data, dict) else 0
                    if (track_id != getattr(self, '_last_compiled_track_id', None) or
                            track_version != getattr(self, '_last_compiled_track_version', None)):
                        self.track_display_list_valid = False

                    if HAS_OPENGL:
                        if not getattr(self, 'track_display_list_valid', False) or self.track_display_list is None:
                            if self.track_display_list is None:
                                try:
                                    self.track_display_list = glGenLists(1)
                                except Exception:
                                    self.track_display_list = None
                            if self.track_display_list is not None:
                                try:
                                    glNewList(self.track_display_list, GL_COMPILE)
                                    self._draw_track_geometry()
                                    glEndList()
                                    self.track_display_list_valid = True
                                    self._last_compiled_track_id = track_id
                                    self._last_compiled_track_version = track_version
                                except Exception:
                                    self.track_display_list_valid = False

                        if getattr(self, 'track_display_list_valid', False) and self.track_display_list is not None:
                            glCallList(self.track_display_list)
                        else:
                            self._draw_track_geometry()
                    else:
                        self._draw_track_geometry()

                # 4. Smooth Alpha-Blended Skid Marks (Batched single draw call)
                if self.skid_ribbons:
                    glPolygonOffset(-1.5, -1.5)
                    now_t = time.time()
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glBegin(GL_QUADS)
                    for sm in self.skid_ribbons:
                        age = now_t - sm['time']
                        if age < self.max_skid_age:
                            alpha = float(sm['intensity'] * 0.45 * max(0.0, 1.0 - (age / self.max_skid_age)))
                            glColor4f(0.08, 0.08, 0.08, alpha)
                            glVertex3f(sm['p1'][0], sm['p1'][1], sm['p1'][2])
                            glVertex3f(sm['p2'][0], sm['p2'][1], sm['p2'][2])
                            glVertex3f(sm['p3'][0], sm['p3'][1], sm['p3'][2])
                            glVertex3f(sm['p4'][0], sm['p4'][1], sm['p4'][2])
                    glEnd()

                # 5. Fast Ground Contact AO Shadow & Thermal Halos
                self._draw_car_ao_shadows(x_car, y_car, yaw_car)

                # 6. Render Solid 3D Formula Student / SUV Car (Optimized)
                self._draw_3d_car()

                # 6b. Render 3D Translucent Ghost Car (Personal Best Lap)
                self._draw_ghost_car(cam_pos=(cam_x, cam_y, cam_z), cam_dir=(look_x - cam_x, look_y - cam_y, look_z - cam_z))
                
                # 7. Mini HUD Overlay
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                self._draw_hud(painter, w, h)
                painter.end()

            else:
                # 2D Map View
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                w = self.width()
                h = self.height()
                
                sky_grad = QLinearGradient(0, 0, 0, int(h * 0.44))
                sky_grad.setColorAt(0.0, QColor(92, 160, 211))
                sky_grad.setColorAt(1.0, QColor(190, 220, 240))
                painter.fillRect(0, 0, w, int(h * 0.44), sky_grad)
                
                grass_grad = QLinearGradient(0, int(h * 0.44), 0, h)
                grass_grad.setColorAt(0.0, QColor(46, 125, 50))
                grass_grad.setColorAt(1.0, QColor(34, 94, 38))
                painter.fillRect(0, int(h * 0.44), w, int(h * 0.56), grass_grad)
                
                self.paint_2d_view(painter, w, h, x_car, y_car, yaw_car)
                painter.end()

        except Exception as e:
            # Safe rendering fallback: keep running without exiting process
            pass
    def _draw_cone(self, cx, cy, cz, r, g, b):
        glColor3f(r, g, b)
        base_r, top_r, h = 0.16, 0.04, 0.32
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(cx, cy, cz + h)
        for i in range(len(self.cos_tab)):
            glVertex3f(cx + top_r*self.cos_tab[i], cy + top_r*self.sin_tab[i], cz + h)
        glEnd()
        glBegin(GL_QUAD_STRIP)
        for i in range(len(self.cos_tab)):
            glVertex3f(cx + base_r*self.cos_tab[i], cy + base_r*self.sin_tab[i], cz)
            glVertex3f(cx + top_r*self.cos_tab[i], cy + top_r*self.sin_tab[i], cz + h)
        glEnd()

    def _draw_car_ao_shadows(self, xc, yc, yaw):
        """Draws soft contact Ambient Occlusion (AO) shadows and dynamic thermal friction halos under the car."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)

        z_road = float(self.car_state.get('z_road', 0.0)) if self.car_state else 0.0
        grade_rd = float(self.car_state.get('grade_road', 0.0)) if self.car_state else 0.0
        bank_rd_deg = float(self.car_state.get('bank_road_deg', 0.0)) if self.car_state else 0.0

        glPushMatrix()
        glTranslatef(xc, yc, z_road + 0.005)
        glRotatef(np.rad2deg(yaw), 0, 0, 1)
        glRotatef(bank_rd_deg, 1, 0, 0)
        glRotatef(-np.rad2deg(math.atan(grade_rd)), 0, 1, 0)

        vp_obj = getattr(self, 'vp', None)
        v_type = getattr(vp_obj, 'vehicle_type', None) or (self.car_state.get('vehicle_type', 'formula') if self.car_state else 'formula')
        is_suv = (v_type == 'b_segment_suv')
        if is_suv:
            shadow_pts = [
                (1.65, 0.0), (1.50, 0.60), (1.10, 0.92), (0.0, 0.95),
                (-1.40, 0.95), (-2.10, 0.80), (-2.25, 0.0), (-2.10, -0.80),
                (-1.40, -0.95), (0.0, -0.95), (1.10, -0.92), (1.50, -0.60),
                (1.65, 0.0)
            ]
            a = getattr(vp_obj, 'a', 1.024) if vp_obj else 1.024
            b = getattr(vp_obj, 'b', 1.536) if vp_obj else 1.536
            tf = (getattr(vp_obj, 't_f', 1.54) / 2.0) if vp_obj else (1.54 / 2.0)
            rx_patch, ry_patch = 0.30, 0.20
        else:
            shadow_pts = [
                (1.30, 0.0), (1.10, 0.42), (0.60, 0.68), (0.0, 0.70),
                (-0.60, 0.70), (-1.05, 0.58), (-1.18, 0.0), (-1.05, -0.58),
                (-0.60, -0.70), (0.0, -0.70), (0.60, -0.68), (1.10, -0.42),
                (1.30, 0.0)
            ]
            a = getattr(vp_obj, 'a', 0.81) if vp_obj else 0.81
            b = getattr(vp_obj, 'b', 0.72) if vp_obj else 0.72
            tf = (getattr(vp_obj, 't_f', 1.25) / 2.0) if vp_obj else (1.25 / 2.0)
            rx_patch, ry_patch = 0.22, 0.16

        # Main Chassis Contact Shadow Decal
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.02, 0.03, 0.04, 0.75)
        glVertex3f(0.0, 0.0, 0.0)
        glColor4f(0.05, 0.06, 0.08, 0.0)
        for sx, sy in shadow_pts:
            glVertex3f(sx, sy, 0.0)
        glEnd()

        # 4 Tire Contact Patch AO Shadows & Dynamic Thermal Friction Halo
        steer_rad = np.deg2rad(self.car_state.get('steer_deg', 0.0))
        
        def draw_patch_ao(px, py, s_rad, util):
            glPushMatrix()
            glTranslatef(px, py, 0.0)
            glRotatef(np.rad2deg(s_rad), 0, 0, 1)
            
            # Dark Central Core
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(0.01, 0.01, 0.02, 0.85)
            glVertex3f(0.0, 0.0, 0.0)
            glColor4f(0.05, 0.06, 0.08, 0.0)
            for i in range(len(self.cos_tab)):
                glVertex3f(rx_patch * self.cos_tab[i], ry_patch * self.sin_tab[i], 0.0)
            glEnd()

            # Dynamic Thermal Friction Halo
            if util < 60:
                h_col = (0.18, 0.85, 0.44, 0.40)
            elif util < 88:
                h_col = (0.98, 0.80, 0.05, 0.65)
            else:
                h_col = (0.95, 0.15, 0.15, 0.85)

            glBegin(GL_TRIANGLE_FAN)
            glColor4f(*h_col)
            glVertex3f(0.0, 0.0, 0.001)
            glColor4f(h_col[0], h_col[1], h_col[2], 0.0)
            rx_halo = rx_patch * 1.25 + (util / 100.0) * 0.12
            ry_halo = ry_patch * 1.25 + (util / 100.0) * 0.10
            for i in range(len(self.cos_tab)):
                glVertex3f(rx_halo * self.cos_tab[i], ry_halo * self.sin_tab[i], 0.001)
            glEnd()

            glPopMatrix()

        steer_fl_ao = np.radians(float(self.car_state.get('steer_fl_deg', 0.0)))
        steer_fr_ao = np.radians(float(self.car_state.get('steer_fr_deg', 0.0)))
        draw_patch_ao(a, tf, steer_fl_ao, self.car_state.get('util_fl', 0))
        draw_patch_ao(a, -tf, steer_fr_ao, self.car_state.get('util_fr', 0))
        draw_patch_ao(-b, tf, 0.0, self.car_state.get('util_rl', 0))
        draw_patch_ao(-b, -tf, 0.0, self.car_state.get('util_rr', 0))

        glPopMatrix()
        glDepthMask(GL_TRUE)

    def _init_asphalt_texture(self):
        """Generates a hardware 2D micro-roughness asphalt texture tile (ISO 8608 grain) on GPU."""
        if not HAS_OPENGL:
            return None
        if getattr(self, '_asphalt_tex_id', None) is not None:
            return self._asphalt_tex_id
        try:
            tex_size = 64
            # Deterministic noise coherent with ISO 8608 asphalt grain
            rng = np.random.default_rng(12345)
            noise = rng.normal(loc=0.0, scale=0.18, size=(tex_size, tex_size))
            # Fast 2D box filter in pure NumPy (zero scipy import overhead or thread contention)
            blurred = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
            tex_lum = np.clip(185 + blurred * 70, 130, 240).astype(np.uint8)
            tex_rgb = np.stack([tex_lum, tex_lum, tex_lum], axis=-1)

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_size, tex_size, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_rgb.tobytes())
            glBindTexture(GL_TEXTURE_2D, 0)
            self._asphalt_tex_id = tex_id
            return tex_id
        except Exception:
            self._asphalt_tex_id = None
            return None

    def _draw_track_geometry(self):
        """Draws the complete 3D track ribbon, verge skirt, kerbs, centerline, and S/F photocell gate geometry."""
        if self.track_data is None:
            return
        x_cnt = self.track_data.get('x_center', np.array([]))
        y_cnt = self.track_data.get('y_center', np.array([]))
        x_left = self.track_data.get('x_left', np.array([]))
        y_left = self.track_data.get('y_left', np.array([]))
        x_right = self.track_data.get('x_right', np.array([]))
        y_right = self.track_data.get('y_right', np.array([]))

        N_pts = len(x_cnt)
        if N_pts <= 2:
            return

        z_cnt_arr = self.track_data.get('z_center', np.zeros(N_pts))
        z_left_arr = self.track_data.get('z_left', np.zeros(N_pts))
        z_right_arr = self.track_data.get('z_right', np.zeros(N_pts))

        is_closed = bool(self.track_data.get('is_closed', math.hypot(x_cnt[-1] - x_cnt[0], y_cnt[-1] - y_cnt[0]) < 10.0))

        # Ensure 3D verge mesh skirt is extruded
        verge = self.track_data.get('verge', None)
        if verge is None:
            from src.utils.track_manager import extrude_verge_geometry
            verge = extrude_verge_geometry(self.track_data, w_verge=8.0, h_drop=0.15)

        xv_l = verge['x_verge_l']
        yv_l = verge['y_verge_l']
        zv_l = verge['z_verge_l']
        xv_r = verge['x_verge_r']
        yv_r = verge['y_verge_r']
        zv_r = verge['z_verge_r']

        r_base, g_base, b_base = self.BASE_TERRAIN_RGB[0]/255.0, self.BASE_TERRAIN_RGB[1]/255.0, self.BASE_TERRAIN_RGB[2]/255.0
        r_grass, g_grass, b_grass = self.GRASS_COLOR_RGB[0]/255.0, self.GRASS_COLOR_RGB[1]/255.0, self.GRASS_COLOR_RGB[2]/255.0

        # 0. Bottom Terrain Floor
        x_min = float(np.min(x_cnt)) - 100.0
        x_max = float(np.max(x_cnt)) + 100.0
        y_min = float(np.min(y_cnt)) - 100.0
        y_max = float(np.max(y_cnt)) + 100.0
        z_min_track = float(min(np.min(z_left_arr), np.min(z_right_arr), np.min(z_cnt_arr)))
        z_base = float(verge.get('z_base', z_min_track - 1.5))

        glPolygonOffset(4.0, 4.0)
        glColor3f(r_base, g_base, b_base)
        glBegin(GL_QUADS)
        glVertex3f(x_min, y_min, z_base)
        glVertex3f(x_max, y_min, z_base)
        glVertex3f(x_max, y_max, z_base)
        glVertex3f(x_min, y_max, z_base)
        glEnd()

        # 1a. Left Vertical Sealing Skirt
        glPolygonOffset(3.0, 3.0)
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(float(xv_l[i]), float(yv_l[i]), float(zv_l[i]))
            glColor3f(r_base, g_base, b_base)
            glVertex3f(float(xv_l[i]), float(yv_l[i]), z_base)
        if is_closed:
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(float(xv_l[0]), float(yv_l[0]), float(zv_l[0]))
            glColor3f(r_base, g_base, b_base)
            glVertex3f(float(xv_l[0]), float(yv_l[0]), z_base)
        glEnd()

        # 1b. Right Vertical Sealing Skirt
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(float(xv_r[i]), float(yv_r[i]), float(zv_r[i]))
            glColor3f(r_base, g_base, b_base)
            glVertex3f(float(xv_r[i]), float(yv_r[i]), z_base)
        if is_closed:
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(float(xv_r[0]), float(yv_r[0]), float(zv_r[0]))
            glColor3f(r_base, g_base, b_base)
            glVertex3f(float(xv_r[0]), float(yv_r[0]), z_base)
        glEnd()

        # 2a. Extruded Left Verge Surface
        glPolygonOffset(2.0, 2.0)
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(x_left[i], y_left[i], float(z_left_arr[i]))
            glVertex3f(float(xv_l[i]), float(yv_l[i]), float(zv_l[i]))
        if is_closed:
            glVertex3f(x_left[0], y_left[0], float(z_left_arr[0]))
            glVertex3f(float(xv_l[0]), float(yv_l[0]), float(zv_l[0]))
        glEnd()

        # 2b. Extruded Right Verge Surface
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            glColor3f(r_grass, g_grass, b_grass)
            glVertex3f(x_right[i], y_right[i], float(z_right_arr[i]))
            glVertex3f(float(xv_r[i]), float(yv_r[i]), float(zv_r[i]))
        if is_closed:
            glVertex3f(x_right[0], y_right[0], float(z_right_arr[0]))
            glVertex3f(float(xv_r[0]), float(yv_r[0]), float(zv_r[0]))
        glEnd()

        # 3a. Asphalt Ribbon with Hardware Micro-Texture Tiling (u = s * 1.5, v = d * 1.5)
        tex_id = self._init_asphalt_texture()
        has_tex = (tex_id is not None)
        if has_tex:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glPolygonOffset(2.0, 2.0)
        glColor3f(68/255.0, 75/255.0, 88/255.0)
        glBegin(GL_QUAD_STRIP)
        s_arr = self.track_data.get('s', None)
        w_trk = float(self.track_data.get('track_width_m', 3.8))
        v_l = 0.0
        v_r = w_trk * 1.5
        for i in range(N_pts):
            s_i = float(s_arr[i]) if s_arr is not None and i < len(s_arr) else float(i * 2.0)
            u_i = s_i * 1.5
            if has_tex:
                glTexCoord2f(u_i, v_l)
            glVertex3f(x_left[i], y_left[i], float(z_left_arr[i]))
            if has_tex:
                glTexCoord2f(u_i, v_r)
            glVertex3f(x_right[i], y_right[i], float(z_right_arr[i]))
        if is_closed:
            s_0 = float(s_arr[-1]) if s_arr is not None and len(s_arr) > 0 else float(N_pts * 2.0)
            u_0 = s_0 * 1.5
            if has_tex:
                glTexCoord2f(u_0, v_l)
            glVertex3f(x_left[0], y_left[0], float(z_left_arr[0]))
            if has_tex:
                glTexCoord2f(u_0, v_r)
            glVertex3f(x_right[0], y_right[0], float(z_right_arr[0]))
        glEnd()

        if has_tex:
            glDisable(GL_TEXTURE_2D)

        # 3b. Left Kerb
        glPolygonOffset(-1.0, -1.0)
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            if (i // 3) % 2 == 0:
                glColor3f(231/255.0, 76/255.0, 60/255.0)
            else:
                glColor3f(245/255.0, 246/255.0, 250/255.0)
            dx = x_left[i] - x_cnt[i]; dy = y_left[i] - y_cnt[i]
            L_norm = math.hypot(dx, dy)
            nx, ny = (dx / L_norm, dy / L_norm) if L_norm > 1e-4 else (0, 0)
            zl_i = float(z_left_arr[i])
            glVertex3f(x_left[i], y_left[i], zl_i + 0.010)
            glVertex3f(x_left[i] + nx * 0.35, y_left[i] + ny * 0.35, zl_i + 0.015)
        if is_closed:
            dx = x_left[0] - x_cnt[0]; dy = y_left[0] - y_cnt[0]
            L_norm = math.hypot(dx, dy)
            nx, ny = (dx / L_norm, dy / L_norm) if L_norm > 1e-4 else (0, 0)
            zl_0 = float(z_left_arr[0])
            glVertex3f(x_left[0], y_left[0], zl_0 + 0.010)
            glVertex3f(x_left[0] + nx * 0.35, y_left[0] + ny * 0.35, zl_0 + 0.015)
        glEnd()

        # 3c. Right Kerb
        glBegin(GL_QUAD_STRIP)
        for i in range(N_pts):
            if (i // 3) % 2 == 0:
                glColor3f(241/255.0, 196/255.0, 15/255.0)
            else:
                glColor3f(245/255.0, 246/255.0, 250/255.0)
            dx = x_right[i] - x_cnt[i]; dy = y_right[i] - y_cnt[i]
            L_norm = math.hypot(dx, dy)
            nx, ny = (dx / L_norm, dy / L_norm) if L_norm > 1e-4 else (0, 0)
            zr_i = float(z_right_arr[i])
            glVertex3f(x_right[i], y_right[i], zr_i + 0.010)
            glVertex3f(x_right[i] + nx * 0.35, y_right[i] + ny * 0.35, zr_i + 0.015)
        if is_closed:
            dx = x_right[0] - x_cnt[0]; dy = y_right[0] - y_cnt[0]
            L_norm = math.hypot(dx, dy)
            nx, ny = (dx / L_norm, dy / L_norm) if L_norm > 1e-4 else (0, 0)
            zr_0 = float(z_right_arr[0])
            glVertex3f(x_right[0], y_right[0], zr_0 + 0.010)
            glVertex3f(x_right[0] + nx * 0.35, y_right[0] + ny * 0.35, zr_0 + 0.015)
        glEnd()

        # 3d. Centerline
        glPolygonOffset(-2.0, -2.0)
        glColor3f(245/255.0, 246/255.0, 250/255.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i in range(0, N_pts - 1, 2):
            glVertex3f(x_cnt[i], y_cnt[i], float(z_cnt_arr[i]) + 0.014)
            glVertex3f(x_cnt[i+1], y_cnt[i+1], float(z_cnt_arr[i+1]) + 0.014)
        glEnd()

        # 3e. Checkered Start / Finish Photocell Line at index 0
        glPolygonOffset(-3.0, -3.0)
        xl0, yl0 = float(x_left[0]), float(y_left[0])
        xr0, yr0 = float(x_right[0]), float(y_right[0])
        zl0, zr0 = float(z_left_arr[0]), float(z_right_arr[0])
        dx_fwd = float(x_cnt[1] - x_cnt[0])
        dy_fwd = float(y_cnt[1] - y_cnt[0])
        len_fwd = math.hypot(dx_fwd, dy_fwd)
        ux, uy = (dx_fwd / len_fwd, dy_fwd / len_fwd) if len_fwd > 1e-4 else (1.0, 0.0)

        n_checks = 8
        line_d = 0.8  # 80cm stripe
        for c in range(n_checks):
            f0 = c / float(n_checks)
            f1 = (c + 1) / float(n_checks)
            if c % 2 == 0:
                glColor3f(1.0, 1.0, 1.0)
            else:
                glColor3f(0.12, 0.12, 0.12)
            p0x = xl0 + f0 * (xr0 - xl0)
            p0y = yl0 + f0 * (yr0 - yl0)
            p0z = zl0 + f0 * (zr0 - zl0) + 0.015
            p1x = xl0 + f1 * (xr0 - xl0)
            p1y = yl0 + f1 * (yr0 - yl0)
            p1z = zl0 + f1 * (zr0 - zl0) + 0.015
            glBegin(GL_QUADS)
            glVertex3f(p0x, p0y, p0z)
            glVertex3f(p1x, p1y, p1z)
            glVertex3f(p1x + ux * line_d, p1y + uy * line_d, p1z)
            glVertex3f(p0x + ux * line_d, p0y + uy * line_d, p0z)
            glEnd()

        # 3f. Finish Line for Open Tracks (Sprint / Acceleration / Open AutoX) at index N-1
        if not is_closed:
            glPolygonOffset(-3.0, -3.0)
            xl_end, yl_end = float(x_left[-1]), float(y_left[-1])
            xr_end, yr_end = float(x_right[-1]), float(y_right[-1])
            zl_end, zr_end = float(z_left_arr[-1]), float(z_right_arr[-1])
            dx_fin = float(x_cnt[-1] - x_cnt[-2])
            dy_fin = float(y_cnt[-1] - y_cnt[-2])
            len_fin = math.hypot(dx_fin, dy_fin)
            ux_f, uy_f = (dx_fin / len_fin, dy_fin / len_fin) if len_fin > 1e-4 else (1.0, 0.0)

            for c in range(n_checks):
                f0 = c / float(n_checks)
                f1 = (c + 1) / float(n_checks)
                if c % 2 == 0:
                    col_fin = (0.95, 0.15, 0.15)
                else:
                    col_fin = (1.0, 1.0, 1.0)
                glColor3f(*col_fin)
                p0x = xl_end + f0 * (xr_end - xl_end)
                p0y = yl_end + f0 * (yr_end - yl_end)
                p0z = zl_end + f0 * (zr_end - zl_end) + 0.015
                p1x = xl_end + f1 * (xr_end - xl_end)
                p1y = yl_end + f1 * (yr_end - yl_end)
                p1z = zl_end + f1 * (zr_end - zl_end) + 0.015
                glBegin(GL_QUADS)
                glVertex3f(p0x, p0y, p0z)
                glVertex3f(p1x, p1y, p1z)
                glVertex3f(p1x + ux_f * line_d, p1y + uy_f * line_d, p1z)
                glVertex3f(p0x + ux_f * line_d, p0y + uy_f * line_d, p0z)
                glEnd()

            glLineWidth(3.5)
            glColor3f(1.0, 0.75, 0.0)
            glBegin(GL_LINES)
            glVertex3f(xl_end - uy_f * 0.3, yl_end + ux_f * 0.3, zl_end)
            glVertex3f(xl_end - uy_f * 0.3, yl_end + ux_f * 0.3, zl_end + 1.6)
            glVertex3f(xr_end + uy_f * 0.3, yr_end - ux_f * 0.3, zr_end)
            glVertex3f(xr_end + uy_f * 0.3, yr_end - ux_f * 0.3, zr_end + 1.6)
            glEnd()

    def _draw_3d_car(self):
        """Renders the appropriate solid volumetric 3D car (Formula Student or Segment B SUV) with dynamic wheels, steering, and driver."""
        xv = float(self.car_state['x'])
        yv = float(self.car_state['y'])
        z_road = float(self.car_state.get('z_road', 0.0))
        psi_car = np.deg2rad(float(self.car_state['psi_deg']))
        grade_rd = float(self.car_state.get('grade_road', 0.0))
        bank_rd_deg = float(self.car_state.get('bank_road_deg', 0.0))
        phi_car = np.deg2rad(float(self.car_state['phi_deg']) + bank_rd_deg)
        theta_car = np.deg2rad(float(self.car_state['theta_deg'])) - math.atan(grade_rd)

        is_cockpit = (getattr(self, 'camera_mode', 'CHASE') == 'COCKPIT') or (abs(self.cam_dist) < 0.1 and self.cam_height < 0.9)
        vp_obj = getattr(self, 'vp', None)
        v_type = getattr(vp_obj, 'vehicle_type', None) or (self.car_state.get('vehicle_type', 'formula') if self.car_state else 'formula')

        # Elevation offset: FSAE axle is at 0.20m, SUV axle is at 0.315m
        zv = z_road + (0.315 if v_type == 'b_segment_suv' else 0.20)

        glPushMatrix()
        glTranslatef(xv, yv, zv)
        glRotatef(np.rad2deg(psi_car), 0, 0, 1)
        glRotatef(np.rad2deg(theta_car), 0, 1, 0)
        glRotatef(np.rad2deg(phi_car), 1, 0, 0)

        if v_type == 'b_segment_suv':
            self.suv_model.draw(self.car_state, is_cockpit=is_cockpit, skip_ao=True)
        else:
            self.fsae_model.draw(self.car_state, is_cockpit=is_cockpit, skip_ao=True)

        glPopMatrix()

    def _draw_ghost_car(self, cam_pos=None, cam_dir=None):
        """Renders translucent satin cyan/white ghost car representing personal best lap with distance culling & fading."""
        if not self.car_state:
            return
        ghost = self.car_state.get('ghost_car', None)
        if not ghost or not isinstance(ghost, dict):
            return

        xv = float(ghost.get('x', 0.0))
        yv = float(ghost.get('y', 0.0))
        zv_base = float(ghost.get('z', 0.0))
        psi_car = float(ghost.get('psi', 0.0))

        vp_obj = getattr(self, 'vp', None)
        v_type = getattr(vp_obj, 'vehicle_type', None) or (self.car_state.get('vehicle_type', 'formula') if self.car_state else 'formula')
        zv = zv_base + (0.315 if v_type == 'b_segment_suv' else 0.20)

        # Distance & Camera Frustum Culling / Proximity Fade
        # Prevents fillrate explosion and near-plane polygon clipping when ghost is right behind the player in chase cam
        alpha_ghost = 0.35
        if cam_pos is not None:
            cx, cy, cz = cam_pos
            dx = xv - cx
            dy = yv - cy
            dz = zv - cz
            if cam_dir is not None:
                cdx, cdy, cdz = cam_dir
                clen = math.hypot(cdx, cdy, cdz)
                if clen > 1e-4:
                    d_forward = (dx * cdx + dy * cdy + dz * cdz) / clen
                else:
                    d_forward = math.hypot(dx, dy)
            else:
                d_forward = math.hypot(dx, dy)

            # Cull completely if behind camera or within near clipping plane (< 0.6m)
            if d_forward < 0.6:
                return

            # Smoothly fade transparency if between 0.6m and 2.6m to prevent screen blockage and overdraw spikes
            if d_forward < 2.6:
                fade = (d_forward - 0.6) / 2.0
                alpha_ghost = float(np.clip(0.35 * fade, 0.0, 0.35))
                if alpha_ghost < 0.015:
                    return

        glPushMatrix()
        glTranslatef(xv, yv, zv)
        glRotatef(np.rad2deg(psi_car), 0, 0, 1)

        # Enable Back-Face Culling to discard 50% of the fragments instantly
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        glEnable(GL_BLEND)
        try:
            glBlendColor(0.25, 0.80, 0.98, alpha_ghost)
            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
        except Exception:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)

        if v_type == 'b_segment_suv':
            self.suv_model.draw_ghost(ghost)
        else:
            self.fsae_model.draw_ghost(ghost)

        glDepthMask(GL_TRUE)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPopMatrix()

    def _draw_hud(self, painter, w, h):
        """Draws a professional, high-contrast digital racing dashboard and tachometer HUD overlay."""
        if not self.car_state:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        vx_kmh = float(self.car_state.get('vx_kmh', 0.0))
        lap_time = float(self.car_state.get('lap_time', 0.0))
        steer_deg = float(self.car_state.get('steer_deg', 0.0))
        throttle_pct = float(self.car_state.get('throttle_pct', 0.0))
        brake_pct = float(self.car_state.get('brake_pct', 0.0))
        ax_g = float(self.car_state.get('ax_g', 0.0))
        ay_g = float(self.car_state.get('ay_g', 0.0))
        tot_g = math.hypot(ax_g, ay_g)
        beta_deg = float(self.car_state.get('beta_deg', 0.0))
        r_deg_s = float(self.car_state.get('r_deg_s', 0.0))
        t_sw = float(self.car_state.get('T_sw_Nm', 0.0))
        
        lockup_pct = float(self.car_state.get('lockup_pct', 0.0))
        traction_loss_pct = float(self.car_state.get('traction_loss_pct', 0.0))
        understeer_pct = float(self.car_state.get('understeer_pct', 0.0))
        oversteer_pct = float(self.car_state.get('oversteer_pct', 0.0))
        lap_invalid = bool(self.car_state.get('lap_invalid', False))
        offtrack_4wheel = bool(self.car_state.get('offtrack_4wheel', False))
        offtrack_timer = float(self.car_state.get('offtrack_4wheel_timer', 0.0))

        # -------------------------------------------------------------
        # 1. TOP RACING STATUS & DYNAMICS ALERT BANNER
        # -------------------------------------------------------------
        banner_w, banner_h = min(460, w - 440), 30
        banner_x = (w - banner_w) // 2
        banner_y = 12

        if lap_invalid:
            b_text = "  GIRO ANNULLATO — 4 Ruote Fuori Pista"
            b_bg = QColor(243, 156, 18, 245)   # FIA amber
        elif offtrack_4wheel and offtrack_timer > 0.0:
            b_text = f"  FUORI PISTA — {offtrack_timer:.2f} s / 0.15 s"
            b_bg = QColor(230, 126, 34, 230)   # Orange warning
        elif lockup_pct > 12.0:
            b_text = f"[BLOCCAGGIO FRENI] LOCKUP RUOTE {lockup_pct:.0f}% (NO ABS)"
            b_bg = QColor(192, 57, 43, 230)
        elif traction_loss_pct > 15.0:
            b_text = f"[TRAZIONE] PATTINAMENTO POSTERIORE {traction_loss_pct:.0f}%"
            b_bg = QColor(243, 156, 18, 230)
        elif understeer_pct > 15.0:
            b_text = f"[SOTTOSTERZO] SCRUB GOMME ANTERIORI {understeer_pct:.0f}%"
            b_bg = QColor(230, 126, 34, 230)
        elif oversteer_pct > 15.0:
            b_text = f"[SOVRASTERZO] SNAP CODA (BETA = {beta_deg:+.1f} deg)"
            b_bg = QColor(231, 76, 60, 230)
        else:
            b_text = "[ADERENZA OTTIMALE] Zona Lineare"
            b_bg = QColor(39, 174, 96, 210)


        if banner_w > 200:
            painter.setBrush(QBrush(b_bg))
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
            banner_rect = QRectF(banner_x, banner_y, banner_w, banner_h)
            painter.drawRoundedRect(banner_rect, 5, 5)
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(Qt.white)
            painter.drawText(banner_rect, Qt.AlignCenter, b_text)

        # -------------------------------------------------------------
        # 2. TOP LEFT LAP TIMING CARD (Real Lap Timer)
        # -------------------------------------------------------------
        lap_count = int(self.car_state.get('lap_count', 0))
        lap_current = float(self.car_state.get('lap_time_current', 0.0))
        lap_last = float(self.car_state.get('lap_time_last', 0.0))
        lap_best = float(self.car_state.get('lap_time_best', 0.0))
        lap_delta = float(self.car_state.get('lap_delta', 0.0))
        sf_armed = bool(self.car_state.get('sf_armed', False))

        tl_w, tl_h = 210, 120
        painter.setBrush(QBrush(QColor(15, 20, 26, 220)))
        painter.setPen(QPen(QColor(53, 59, 72), 1.2))
        painter.drawRoundedRect(14, 12, tl_w, tl_h, 6, 6)

        y_off = 30
        # Lap count header
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        if sf_armed:
            painter.setPen(QColor(160, 170, 180))
            lap_label = f"LAP {lap_count}"
        elif lap_last > 0.1:
            painter.setPen(QColor(46, 204, 113))
            lap_label = "FINISH"
        else:
            painter.setPen(QColor(241, 196, 15))
            lap_label = "GRID / READY"
        painter.drawText(22, y_off, lap_label)
        painter.setPen(QColor(160, 170, 180))
        painter.drawText(130, y_off, f"{vx_kmh:.0f} km/h")

        # Current lap time (large, cyan; amber when lap is invalidated)
        y_off += 22
        mins_c = int(lap_current // 60)
        secs_c = lap_current % 60
        painter.setFont(QFont("Consolas", 16, QFont.Bold))
        if lap_invalid:
            painter.setPen(QColor(243, 156, 18))    # FIA amber = lap invalid
        else:
            painter.setPen(QColor(0, 206, 201))     # Cyan = lap valid
        painter.drawText(22, y_off, f"{mins_c:02d}:{secs_c:05.2f}")

        # Delta vs best (green = faster, red = slower; suppressed on invalid lap)
        if sf_armed and lap_best > 0.1 and not lap_invalid:
            delta_val = float(self.car_state.get('live_delta_lap', lap_delta))
            delta_color = QColor(46, 204, 113) if delta_val <= 0 else QColor(231, 76, 60)
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.setPen(delta_color)
            painter.drawText(145, y_off, f"{delta_val:+.2f}")

        # Last lap time
        y_off += 20
        painter.setFont(QFont("Segoe UI", 9, QFont.Normal))
        painter.setPen(QColor(180, 185, 195))
        if lap_last > 0.1:
            mins_l = int(lap_last // 60)
            secs_l = lap_last % 60
            painter.drawText(22, y_off, f"LAST  {mins_l:02d}:{secs_l:05.2f}")
        else:
            painter.drawText(22, y_off, "LAST  --:--.--")

        # Best lap time (magenta/purple)
        y_off += 18
        if lap_best > 0.1:
            mins_b = int(lap_best // 60)
            secs_b = lap_best % 60
            painter.setPen(QColor(155, 89, 182))
            painter.drawText(22, y_off, f"BEST  {mins_b:02d}:{secs_b:05.2f}")
        else:
            painter.setPen(QColor(120, 125, 135))
            painter.drawText(22, y_off, "BEST  --:--.--")

        # Yaw rate (small, bottom of card)
        y_off += 18
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QColor(140, 145, 155))
        painter.drawText(22, y_off, f"ψ̇ {r_deg_s:+.1f} °/s  |  β {beta_deg:+.1f}°")

        # -------------------------------------------------------------
        # 2.5. CIRCUIT MINIMAP CARD (TOP LEFT, BELOW LAP TIMING)
        # -------------------------------------------------------------
        if getattr(self, 'show_minimap', True) and self.track_data is not None and h > 460:
            mm_w, mm_h = 210, 180
            mm_x, mm_y = 14, 140
            self._draw_minimap(painter, mm_x, mm_y, mm_w, mm_h)

        # -------------------------------------------------------------
        # 3. TOP RIGHT FULLSCREEN HINT & BADGE + FPS COUNTER
        # -------------------------------------------------------------
        fps_val = getattr(self, 'fps_smooth', 60.0)
        if fps_val >= 55.0:
            fps_col = QColor(46, 204, 113)   # emerald green (60+ FPS)
        elif fps_val >= 30.0:
            fps_col = QColor(241, 196, 15)   # golden yellow (30-54 FPS)
        else:
            fps_col = QColor(231, 76, 60)    # alert red (< 30 FPS)

        tr_w, tr_h = 175, 28
        tr_x = w - tr_w - 14

        # Standalone High-Visibility FPS Counter Badge
        fps_b_w, fps_b_h = 80, 28
        fps_b_x = tr_x - fps_b_w - 8
        painter.setBrush(QBrush(QColor(15, 20, 26, 210)))
        painter.setPen(QPen(fps_col, 1.2))
        fps_badge_rect = QRectF(fps_b_x, 12, fps_b_w, fps_b_h)
        painter.drawRoundedRect(fps_badge_rect, 5, 5)
        painter.setPen(fps_col)
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(fps_badge_rect, Qt.AlignCenter, f"{fps_val:.0f} FPS")

        # Fullscreen Toggle Badge
        painter.setBrush(QBrush(QColor(15, 20, 26, 190)))
        painter.setPen(QPen(QColor(72, 84, 96), 1))
        badge_rect = QRectF(tr_x, 12, tr_w, tr_h)
        painter.drawRoundedRect(badge_rect, 5, 5)
        painter.setPen(QColor(245, 246, 250))
        painter.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        is_fs = getattr(self, 'is_fullscreen_view', False) or (w > 1200 and h > 750)
        btn_txt = "⛶ Esci Fullscreen [F11]" if is_fs else "⛶ Fullscreen [F11]"
        painter.drawText(badge_rect, Qt.AlignCenter, btn_txt)

        # Performance & Calculation Frequency Monitor (Always visible in fullscreen & driving view)
        perf_w, perf_h = 230, 68
        perf_x = w - perf_w - 14
        perf_y = 46
        
        phys_hz = float(self.car_state.get('physics_hz', 250.0))
        thermal_hz = float(self.car_state.get('thermal_hz', 50.0))
        step_ms = float(self.car_state.get('step_ms', 0.35))
        rt_ratio = float(self.car_state.get('rt_ratio', 1.0))
        
        # Real-time lag / slowdown check
        is_lagging = (rt_ratio < 0.88) or (phys_hz < 80.0) or (fps_val < 25.0)
        is_heavy = (rt_ratio < 0.95) or (fps_val < 45.0)
        
        if is_lagging:
            perf_border = QColor(231, 76, 60, 240)
            status_dot_col = QColor(231, 76, 60)
            status_txt = "⚠️ RALLENTAMENTO / LAG"
        elif is_heavy:
            perf_border = QColor(241, 196, 15, 220)
            status_dot_col = QColor(241, 196, 15)
            status_txt = "▲ CARICO ELEVATO"
        else:
            perf_border = QColor(46, 204, 113, 200)
            status_dot_col = QColor(46, 204, 113)
            status_txt = "● TEMPO REALE OTTIMALE"
            
        painter.setBrush(QBrush(QColor(15, 20, 26, 210)))
        painter.setPen(QPen(perf_border, 1.2))
        perf_rect = QRectF(perf_x, perf_y, perf_w, perf_h)
        painter.drawRoundedRect(perf_rect, 6, 6)
        
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.setPen(QColor(0, 206, 201))
        painter.drawText(int(perf_x + 10), int(perf_y + 18), f"⚡ RK4: {phys_hz:.0f} Hz ({step_ms:.2f} ms)")
        painter.setPen(fps_col)
        painter.drawText(int(perf_x + perf_w - 74), int(perf_y + 18), f"{fps_val:.0f} FPS")
        
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QColor(180, 185, 195))
        therm_str = f"{thermal_hz:.0f} Hz" if thermal_hz > 0 else "Sincrono"
        painter.drawText(int(perf_x + 10), int(perf_y + 36), f"Gomme: {therm_str} | Carico: {int(min(rt_ratio * 100, 100))}%")
        
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(status_dot_col)
        painter.drawText(int(perf_x + 10), int(perf_y + 54), status_txt)

        # -------------------------------------------------------------
        # 4. CENTRAL TACHOMETER & SPEEDOMETER CLUSTER (BOTTOM CENTER)
        # -------------------------------------------------------------
        dash_w, dash_h = 320, 160
        dash_x = (w - dash_w) // 2
        dash_y = h - dash_h - 12
        
        # Dark bezel background
        painter.setBrush(QBrush(QColor(12, 16, 22, 230)))
        painter.setPen(QPen(QColor(0, 206, 201, 90), 1.5))
        painter.drawRoundedRect(dash_x, dash_y, dash_w, dash_h, 8, 8)

        # TCS Status Badge inside speedometer cluster
        tcs_en = bool(self.car_state.get('tcs_enabled', True))
        tcs_act = bool(self.car_state.get('tcs_active', False))
        tcs_cut = float(self.car_state.get('tcs_cut_pct', 0.0))

        tcs_bw, tcs_bh = 110, 18
        tcs_bx = dash_x + (dash_w - tcs_bw) // 2
        tcs_by = dash_y + 8
        tcs_badge_rect = QRectF(tcs_bx, tcs_by, tcs_bw, tcs_bh)

        if not tcs_en:
            painter.setBrush(QBrush(QColor(40, 20, 20, 210)))
            painter.setPen(QPen(QColor(231, 76, 60), 1.0))
            painter.drawRoundedRect(tcs_badge_rect, 4, 4)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(231, 76, 60))
            painter.drawText(tcs_badge_rect, Qt.AlignCenter, "TCS: OFF")
        elif tcs_act:
            blink = (int(time.time() * 8) % 2 == 0)
            tcs_bg = QColor(80, 60, 10, 240) if blink else QColor(50, 35, 8, 220)
            tcs_border = QColor(241, 196, 15) if blink else QColor(230, 126, 34)
            painter.setBrush(QBrush(tcs_bg))
            painter.setPen(QPen(tcs_border, 1.2))
            painter.drawRoundedRect(tcs_badge_rect, 4, 4)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(241, 196, 15))
            painter.drawText(tcs_badge_rect, Qt.AlignCenter, f"TCS: -{tcs_cut:.0f}%")
        else:
            painter.setBrush(QBrush(QColor(15, 35, 25, 200)))
            painter.setPen(QPen(QColor(46, 204, 113, 140), 1.0))
            painter.drawRoundedRect(tcs_badge_rect, 4, 4)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(46, 204, 113))
            painter.drawText(tcs_badge_rect, Qt.AlignCenter, "TCS: ATTIVO")

        # Speedometer Arc
        arc_cx = dash_x + dash_w // 2
        arc_cy = dash_y + 80
        arc_r = 62
        arc_rect = QRectF(arc_cx - arc_r, arc_cy - arc_r, arc_r * 2, arc_r * 2)

        # Background track arc (-220 deg to +40 deg, span 260 deg in Qt 1/16th deg)
        painter.setPen(QPen(QColor(45, 52, 60), 8, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(arc_rect, -40 * 16, 260 * 16)

        # Active speed gradient arc
        v_max_gauge = 140.0
        v_ratio = min(max(vx_kmh / v_max_gauge, 0.0), 1.0)
        span_active = int(v_ratio * 260.0 * 16)

        # Dynamic arc color
        if vx_kmh < 75.0:
            arc_col = QColor(0, 206, 201)
        elif vx_kmh < 110.0:
            arc_col = QColor(241, 196, 15)
        else:
            arc_col = QColor(231, 76, 60)

        painter.setPen(QPen(arc_col, 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(arc_rect, (220 - int(v_ratio * 260.0)) * 16, span_active)

        # Big Digital Speed Readout
        painter.setFont(QFont("Consolas", 28, QFont.Bold))
        painter.setPen(Qt.white)
        speed_str = f"{int(vx_kmh)}"
        painter.drawText(QRectF(arc_cx - 55, arc_cy - 26, 110, 36), Qt.AlignCenter, speed_str)

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(160, 175, 190))
        painter.drawText(QRectF(arc_cx - 40, arc_cy + 10, 80, 16), Qt.AlignCenter, "KM/H")

        # Steer Angle mini-readout at bottom of tachometer
        painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        steer_fl_deg = float(self.car_state.get('steer_fl_deg', steer_deg))
        steer_fr_deg = float(self.car_state.get('steer_fr_deg', steer_deg))
        steer_col = QColor(46, 204, 113) if abs(steer_deg) < 12.0 else QColor(241, 196, 15)
        painter.setPen(steer_col)
        painter.drawText(QRectF(dash_x + 10, dash_y + 138, dash_w - 20, 18), Qt.AlignCenter, f"STERZO: FL {steer_fl_deg:+.1f}° | FR {steer_fr_deg:+.1f}° | SAT: {t_sw:.1f} Nm")

        # Flanking Throttle (Right) and Brake (Left) Vertical Bars
        # Brake Pedal Bar (Left)
        brk_bx = dash_x + 16
        brk_by = dash_y + 24
        brk_bw, brk_bh = 14, 100
        painter.setBrush(QBrush(QColor(30, 36, 44)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(brk_bx, brk_by, brk_bw, brk_bh, 3, 3)
        b_fill = int((brake_pct / 100.0) * brk_bh)
        b_col = QColor(231, 76, 60) if lockup_pct < 10 else QColor(255, 71, 87)
        painter.setBrush(QBrush(b_col))
        painter.drawRoundedRect(brk_bx, brk_by + (brk_bh - b_fill), brk_bw, b_fill, 3, 3)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(231, 76, 60))
        painter.drawText(brk_bx - 4, brk_by + brk_bh + 11, "BRK")

        # Throttle Pedal Bar (Right)
        thr_bx = dash_x + dash_w - 30
        thr_by = dash_y + 24
        thr_bw, thr_bh = 14, 100
        painter.setBrush(QBrush(QColor(30, 36, 44)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(thr_bx, thr_by, thr_bw, thr_bh, 3, 3)
        t_fill = int((throttle_pct / 100.0) * thr_bh)
        t_col = QColor(46, 204, 113) if traction_loss_pct < 10 else QColor(241, 196, 15)
        painter.setBrush(QBrush(t_col))
        painter.drawRoundedRect(thr_bx, thr_by + (thr_bh - t_fill), thr_bw, t_fill, 3, 3)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(46, 204, 113))
        painter.drawText(thr_bx - 4, thr_by + thr_bh + 11, "THR")

        # -------------------------------------------------------------
        # 5. BOTTOM LEFT: 4-WHEEL ASSETTO, GOMME & ADERENZA MATRIX
        # -------------------------------------------------------------
        mat_w, mat_h = 295, 185
        mat_x = 14
        mat_y = h - mat_h - 12
        painter.setBrush(QBrush(QColor(15, 20, 26, 225)))
        painter.setPen(QPen(QColor(53, 59, 72), 1.2))
        painter.drawRoundedRect(mat_x, mat_y, mat_w, mat_h, 8, 8)

        grip_pct = float(self.car_state.get('grip_multiplier', 1.0)) * 100.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(0, 206, 201))
        painter.drawText(mat_x + 10, mat_y + 15, f"ASSETTO & GOMME [Grip: {grip_pct:.0f}%]")

        # Dividers between quadrants
        painter.setPen(QPen(QColor(53, 59, 72, 160), 1))
        painter.drawLine(mat_x + 145, mat_y + 22, mat_x + 145, mat_y + 175)
        painter.drawLine(mat_x + 10, mat_y + 97, mat_x + mat_w - 10, mat_y + 97)

        wheel_data = [
            ("FL", self.car_state.get('util_fl', 0), self.car_state.get('Fz_fl', 0), self.car_state.get('z_fl_mm', 0),
             self.car_state.get('tire_T_fl', 25.0), self.car_state.get('tire_wear_fl', 0.0),
             float(self.car_state.get('defl_fl_mm', 0.0)), float(self.car_state.get('z_road_fl_mm', 0.0)),
             bool(self.car_state.get('liftoff_fl', False)), mat_x + 10, mat_y + 24),
            ("FR", self.car_state.get('util_fr', 0), self.car_state.get('Fz_fr', 0), self.car_state.get('z_fr_mm', 0),
             self.car_state.get('tire_T_fr', 25.0), self.car_state.get('tire_wear_fr', 0.0),
             float(self.car_state.get('defl_fr_mm', 0.0)), float(self.car_state.get('z_road_fr_mm', 0.0)),
             bool(self.car_state.get('liftoff_fr', False)), mat_x + 152, mat_y + 24),
            ("RL", self.car_state.get('util_rl', 0), self.car_state.get('Fz_rl', 0), self.car_state.get('z_rl_mm', 0),
             self.car_state.get('tire_T_rl', 25.0), self.car_state.get('tire_wear_rl', 0.0),
             float(self.car_state.get('defl_rl_mm', 0.0)), float(self.car_state.get('z_road_rl_mm', 0.0)),
             bool(self.car_state.get('liftoff_rl', False)), mat_x + 10, mat_y + 103),
            ("RR", self.car_state.get('util_rr', 0), self.car_state.get('Fz_rr', 0), self.car_state.get('z_rr_mm', 0),
             self.car_state.get('tire_T_rr', 25.0), self.car_state.get('tire_wear_rr', 0.0),
             float(self.car_state.get('defl_rr_mm', 0.0)), float(self.car_state.get('z_road_rr_mm', 0.0)),
             bool(self.car_state.get('liftoff_rr', False)), mat_x + 152, mat_y + 103),
        ]

        for name, u_val, fz_val, z_val, t_val, w_val, defl_val, z_road_val, liftoff_flag, wx, wy in wheel_data:
            is_liftoff = liftoff_flag or (fz_val <= 1.0) or (defl_val <= 0.0)

            # Grip load color
            t_col = QColor(46, 204, 113) if u_val < 65 else (QColor(241, 196, 15) if u_val < 88 else QColor(231, 76, 60))

            if is_liftoff:
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QColor(231, 76, 60))
                painter.drawText(wx, wy + 11, f"{name}: {fz_val:.0f}N")

                # Highlighted LIFT-OFF badge
                badge_rect = QRectF(wx + 66, wy, 60, 13)
                painter.setBrush(QBrush(QColor(231, 76, 60, 230)))
                painter.setPen(QPen(QColor(241, 196, 15), 1.0))
                painter.drawRoundedRect(badge_rect, 3, 3)
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                painter.setPen(Qt.white)
                painter.drawText(badge_rect, Qt.AlignCenter, "LIFT-OFF")
            else:
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QColor(230, 235, 240))
                painter.drawText(wx, wy + 11, f"{name}: {fz_val:.0f}N ({u_val:.0f}%)")

            # Mini Grip bar
            painter.setBrush(QBrush(QColor(35, 42, 50)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(wx, wy + 16, 125, 3, 1.5, 1.5)
            g_fill = int(min(max(u_val / 100.0, 0.0), 1.0) * 125)
            painter.setBrush(QBrush(t_col))
            painter.drawRoundedRect(wx, wy + 16, g_fill, 3, 1.5, 1.5)

            # Tire Thermal & Wear status
            if t_val < 46.0:
                temp_col = QColor(52, 152, 219)
            elif t_val <= 65.0:
                temp_col = QColor(46, 204, 113)
            elif t_val <= 78.0:
                temp_col = QColor(241, 196, 15)
            else:
                temp_col = QColor(231, 76, 60)

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(temp_col)
            painter.drawText(wx, wy + 29, f"T: {t_val:.0f}°C")
            painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
            painter.setPen(QColor(170, 180, 190))
            painter.drawText(wx + 52, wy + 29, f"Usura: {w_val:.1f}%")

            # Suspension Travel
            z_col = QColor(46, 204, 113) if abs(z_val) < 1.0 else (QColor(0, 206, 201) if z_val < 0 else QColor(230, 126, 34))
            painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
            painter.setPen(z_col)
            painter.drawText(wx, wy + 42, f"Corsa: {z_val:+.1f} mm")

            # Tire Deflection
            defl_col = QColor(231, 76, 60) if is_liftoff else (QColor(46, 204, 113) if defl_val > 4.0 else QColor(241, 196, 15))
            painter.setPen(defl_col)
            painter.drawText(wx, wy + 55, f"Defl Gomma: {defl_val:.1f} mm")

            # Effective Ground Elevation
            painter.setPen(QColor(160, 175, 190))
            painter.drawText(wx, wy + 68, f"Z Suolo Eff: {z_road_val:+.1f} mm")

        # -------------------------------------------------------------
        # 6. BOTTOM RIGHT: G-G FRICTION CIRCLE & FFB MONITOR
        # -------------------------------------------------------------
        gg_w, gg_h = 240, 160
        gg_x = w - gg_w - 14
        gg_y = h - gg_h - 12
        painter.setBrush(QBrush(QColor(15, 20, 26, 220)))
        painter.setPen(QPen(QColor(53, 59, 72), 1.2))
        painter.drawRoundedRect(gg_x, gg_y, gg_w, gg_h, 8, 8)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(0, 206, 201))
        painter.drawText(gg_x + 10, gg_y + 16, "ACCELEROMETRO G-G & FFB")

        # Friction Circle
        fc_cx = gg_x + 60
        fc_cy = gg_y + 85
        fc_r = 45

        painter.setBrush(QBrush(QColor(25, 30, 38)))
        painter.setPen(QPen(QColor(70, 80, 92), 1))
        painter.drawEllipse(fc_cx - fc_r, fc_cy - fc_r, fc_r * 2, fc_r * 2)

        # 1G & 2G inner rings
        painter.setPen(QPen(QColor(50, 60, 72, 160), 1, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(fc_cx - int(fc_r * 0.33), fc_cy - int(fc_r * 0.33), int(fc_r * 0.66), int(fc_r * 0.66))
        painter.drawEllipse(fc_cx - int(fc_r * 0.66), fc_cy - int(fc_r * 0.66), int(fc_r * 1.32), int(fc_r * 1.32))

        # Dynamic G-ball
        bx = max(fc_cx - fc_r, min(fc_cx + fc_r, fc_cx - (ay_g / 3.0) * fc_r))
        by = max(fc_cy - fc_r, min(fc_cy + fc_r, fc_cy + (ax_g / 3.0) * fc_r))
        painter.setBrush(QBrush(QColor(231, 76, 60)))
        painter.setPen(QPen(Qt.white, 1.2))
        painter.drawEllipse(int(bx) - 5, int(by) - 5, 10, 10)

        # Digital G readouts
        painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        painter.setPen(QColor(241, 196, 15))
        painter.drawText(gg_x + 115, gg_y + 45, f"LAT:  {ay_g:+.2f} g")
        painter.drawText(gg_x + 115, gg_y + 65, f"LONG: {ax_g:+.2f} g")
        painter.drawText(gg_x + 115, gg_y + 85, f"TOT:  {tot_g:.2f} g")
        
        # FFB Info
        painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
        painter.setPen(QColor(180, 190, 200))
        painter.drawText(gg_x + 115, gg_y + 115, f"Coppia SAT: {t_sw:.1f}Nm")
        
        if self.car_state.get('kerb_active', False):
            painter.setPen(QColor(241, 196, 15))
            painter.drawText(gg_x + 115, gg_y + 135, "[CORDOLO ATTIVO]")
        elif self.car_state.get('offtrack_active', False):
            painter.setPen(QColor(231, 76, 60))
            painter.drawText(gg_x + 115, gg_y + 135, "[FUORI PISTA]")
        else:
            painter.setPen(QColor(46, 204, 113))
            painter.drawText(gg_x + 115, gg_y + 135, "[ASFALTO]")

        # -------------------------------------------------------------
        # 6. BOTTOM LEFT: DYNAMIC ROUGHNESS HUD (Micro-Rugosità)
        # -------------------------------------------------------------
        rough_mm = float(self.car_state.get('roughness_fl_mm', 0.0))
        rough_w, rough_h = 210, 40
        rough_x = 14
        rough_y = h - rough_h - 14

        painter.setBrush(QBrush(QColor(15, 20, 26, 210)))
        painter.setPen(QPen(QColor(72, 84, 96), 1.2))
        painter.drawRoundedRect(rough_x, rough_y, rough_w, rough_h, 4, 4)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(160, 170, 180))
        painter.drawText(rough_x + 8, rough_y + 16, "ROAD ROUGHNESS (μ-2D)")

        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        val_color = QColor(46, 204, 113) if abs(rough_mm) < 1.0 else (QColor(241, 196, 15) if abs(rough_mm) < 3.0 else QColor(231, 76, 60))
        painter.setPen(val_color)
        painter.drawText(rough_x + 130, rough_y + 16, f"{rough_mm:+.1f} mm")

        # Oscilloscope / Bar
        bar_x = rough_x + 8
        bar_y = rough_y + 24
        bar_w = rough_w - 16
        painter.setPen(QPen(QColor(100, 100, 100, 120), 1.5))
        painter.drawLine(bar_x, bar_y + 4, bar_x + bar_w, bar_y + 4)
        painter.drawLine(bar_x + bar_w // 2, bar_y, bar_x + bar_w // 2, bar_y + 8)

        # Draw pulse dot
        # Scale: max +- 5 mm
        pulse_x = bar_x + bar_w // 2 + (np.clip(rough_mm, -5.0, 5.0) / 5.0) * (bar_w // 2)
        painter.setBrush(QBrush(val_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(pulse_x, bar_y + 4.0), 4.0, 4.0)

        painter.restore()

    def _draw_minimap(self, painter, mm_x, mm_y, mm_w, mm_h):
        """Renders a high-clarity circuit minimap with real-time vehicle position and heading indicator."""
        if self.track_data is None:
            return

        xc = self.track_data.get('x_center', None)
        yc = self.track_data.get('y_center', None)
        if xc is None or yc is None or len(xc) < 3:
            return

        # Check / update cached minimap geometry
        track_id = id(self.track_data)
        cache_key = (track_id, mm_w, mm_h)
        if getattr(self, '_cached_mm_key', None) != cache_key:
            min_x = float(np.min(xc))
            max_x = float(np.max(xc))
            min_y = float(np.min(yc))
            max_y = float(np.max(yc))
            span_x = max(max_x - min_x, 10.0)
            span_y = max(max_y - min_y, 10.0)
            mid_x = 0.5 * (min_x + max_x)
            mid_y = 0.5 * (min_y + max_y)

            # Available inner draw area (with 16px margins)
            draw_w = mm_w - 32
            draw_h = mm_h - 44  # space for header
            scale = min(draw_w / span_x, draw_h / span_y)

            cx_screen = mm_x + mm_w / 2.0
            cy_screen = mm_y + 26 + draw_h / 2.0

            # Precalculate track polyline
            pts = []
            for xi, yi in zip(xc, yc):
                sx = cx_screen + (xi - mid_x) * scale
                sy = cy_screen - (yi - mid_y) * scale  # Invert Y for screen
                pts.append(QPointF(sx, sy))

            self._cached_mm_key = cache_key
            self._cached_mm_poly = QPolygonF(pts)
            self._cached_mm_scale = scale
            self._cached_mm_mid_x = mid_x
            self._cached_mm_mid_y = mid_y
            self._cached_mm_cx = cx_screen
            self._cached_mm_cy = cy_screen

        # 1. Dark translucent card background
        painter.setBrush(QBrush(QColor(15, 20, 26, 220)))
        painter.setPen(QPen(QColor(53, 59, 72), 1.2))
        painter.drawRoundedRect(mm_x, mm_y, mm_w, mm_h, 6, 6)

        # 2. Card Header: Track progress & distance
        s_lap = float(self.car_state.get('s_lap', 0.0))
        s_arr = self.track_data.get('s', None)
        track_len = float(s_arr[-1]) if s_arr is not None and len(s_arr) > 0 else 850.0
        s_pct = min(max((s_lap / max(track_len, 1.0)) * 100.0, 0.0), 100.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(0, 206, 201))
        painter.drawText(mm_x + 10, mm_y + 18, f"CIRCUITO [{s_pct:.0f}%]")

        painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
        painter.setPen(QColor(160, 170, 180))
        painter.drawText(mm_x + mm_w - 75, mm_y + 18, f"{s_lap:.0f}m / {track_len:.0f}m")

        # 3. Track Ribbon & Centerline
        poly = getattr(self, '_cached_mm_poly', None)
        if poly is not None and len(poly) > 1:
            # Outer gray track base
            painter.setPen(QPen(QColor(53, 59, 72, 220), 6.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(poly)

            # Inner cyan track centerline
            painter.setPen(QPen(QColor(113, 128, 147, 240), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPolyline(poly)

            # Start / Finish line marker (Amber/Gold bar)
            p0 = poly[0]
            p1 = poly[1]
            dx = p1.x() - p0.x()
            dy = p1.y() - p0.y()
            L_seg = math.hypot(dx, dy)
            if L_seg > 1e-3:
                nx = -dy / L_seg * 6.0
                ny = dx / L_seg * 6.0
                painter.setPen(QPen(QColor(241, 196, 15), 2.5))
                painter.drawLine(QPointF(p0.x() - nx, p0.y() - ny), QPointF(p0.x() + nx, p0.y() + ny))

        # 4. Vehicle Marker: Heading Arrow & Glowing Pulse Dot
        xv = float(self.car_state.get('x', 0.0))
        yv = float(self.car_state.get('y', 0.0))
        psi_car = np.deg2rad(float(self.car_state.get('psi_deg', 0.0)))

        scale = getattr(self, '_cached_mm_scale', 1.0)
        cx_screen = getattr(self, '_cached_mm_cx', mm_x + mm_w / 2.0)
        cy_screen = getattr(self, '_cached_mm_cy', mm_y + mm_h / 2.0)
        mid_x = getattr(self, '_cached_mm_mid_x', 0.0)
        mid_y = getattr(self, '_cached_mm_mid_y', 0.0)

        car_sx = cx_screen + (xv - mid_x) * scale
        car_sy = cy_screen - (yv - mid_y) * scale

        # Clamp to card bounds
        car_sx = min(max(car_sx, mm_x + 8), mm_x + mm_w - 8)
        car_sy = min(max(car_sy, mm_y + 26), mm_y + mm_h - 8)

        # Glowing outer halo
        painter.setBrush(QBrush(QColor(0, 206, 201, 50)))
        painter.setPen(QPen(QColor(0, 206, 201, 180), 1.5))
        painter.drawEllipse(QPointF(car_sx, car_sy), 6.5, 6.5)

        # Directional Arrow (Nose + Wings) oriented along yaw psi_car
        fwd_x = math.cos(psi_car)
        fwd_y = -math.sin(psi_car)
        lat_x = -fwd_y
        lat_y = fwd_x

        p_nose = QPointF(car_sx + fwd_x * 7.5, car_sy + fwd_y * 7.5)
        p_left = QPointF(car_sx - fwd_x * 4.5 + lat_x * 4.5, car_sy - fwd_y * 4.5 + lat_y * 4.5)
        p_right = QPointF(car_sx - fwd_x * 4.5 - lat_x * 4.5, car_sy - fwd_y * 4.5 - lat_y * 4.5)

        car_arrow = QPolygonF([p_nose, p_left, p_right])
        painter.setBrush(QBrush(QColor(231, 76, 60)))  # High-contrast Ferrari red
        painter.setPen(QPen(Qt.white, 1.2))
        painter.drawPolygon(car_arrow)

    def paint_2d_view(self, painter, w, h, x_car, y_car, yaw_car):
        zoom = 5.0
        cx = w / 2.0
        cy = h / 2.0
        def to_screen(xm, ym):
            return QPointF(cx + (xm - x_car) * zoom, cy - (ym - y_car) * zoom)

        xc = self.track_data['x_center']
        yc = self.track_data['y_center']
        xl = self.track_data['x_left']
        yl = self.track_data['y_left']
        xr = self.track_data['x_right']
        yr = self.track_data['y_right']

        # Track Edges
        painter.setPen(QPen(QColor(231, 76, 60, 200), 2.5))
        for i in range(1, len(xl)):
            painter.drawLine(to_screen(xl[i-1], yl[i-1]), to_screen(xl[i], yl[i]))

        painter.setPen(QPen(QColor(241, 196, 15, 200), 2.5))
        for i in range(1, len(xr)):
            painter.drawLine(to_screen(xr[i-1], yr[i-1]), to_screen(xr[i], yr[i]))

        # Centerline
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1.8, Qt.DashLine))
        for i in range(1, len(xc)):
            painter.drawLine(to_screen(xc[i-1], yc[i-1]), to_screen(xc[i], yc[i]))

        cos_p = np.cos(yaw_car)
        sin_p = np.sin(yaw_car)
        steer_rad = np.deg2rad(self.car_state.get('steer_deg', 0.0))

        def car_2d_pt(lx, ly):
            wx = x_car + lx * cos_p - ly * sin_p
            wy = y_car + lx * sin_p + ly * cos_p
            return to_screen(wx, wy)

        poly_body = QPolygonF([
            car_2d_pt(1.15, 0.0),
            car_2d_pt(0.70, -0.32),
            car_2d_pt(-0.70, -0.35),
            car_2d_pt(-0.70, 0.35),
            car_2d_pt(0.70, 0.32)
        ])
        painter.setBrush(QBrush(QColor(6, 82, 221, 230)))
        painter.setPen(QPen(QColor(236, 240, 241), 1.5))
        painter.drawPolygon(poly_body)

        def draw_2d_wheel(wx_c, wy_c, d_ang, util_pct):
            c_d = np.cos(d_ang); s_d = np.sin(d_ang)
            col = QColor(46, 204, 113) if util_pct < 60 else (QColor(241, 196, 15) if util_pct < 88 else QColor(231, 76, 60))
            def w_pt(dx, dy):
                lx = wx_c + dx * c_d + dy * s_d
                ly = wy_c - dx * s_d + dy * c_d
                return car_2d_pt(lx, ly)
            w_poly = QPolygonF([w_pt(0.22, 0.09), w_pt(0.22, -0.09), w_pt(-0.22, -0.09), w_pt(-0.22, 0.09)])
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(Qt.black, 1.2))
            painter.drawPolygon(w_poly)

        steer_fl_2d = np.radians(float(self.car_state.get('steer_fl_deg', np.degrees(steer_rad))))
        steer_fr_2d = np.radians(float(self.car_state.get('steer_fr_deg', np.degrees(steer_rad))))
        draw_2d_wheel(0.81, 0.62, steer_fl_2d, self.car_state.get('util_fl', 0))
        draw_2d_wheel(0.81, -0.62, steer_fr_2d, self.car_state.get('util_fr', 0))
        draw_2d_wheel(-0.72, 0.62, 0.0, self.car_state.get('util_rl', 0))
        draw_2d_wheel(-0.72, -0.62, 0.0, self.car_state.get('util_rr', 0))


def get_viewport_class():
    return OpenGLViewport3D if HAS_OPENGL else _LegacyViewport3D


class _LegacyViewport3D(QWidget):
    """3D Perspective Chase Camera & 2D Track Viewport with Exact Right-Handed Steering Kinematics."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #27ae60; border: 2px solid #2d3436; border-radius: 6px;")
        
        self.car_state = None
        self.trail = []
        self.max_trail = 300
        self.view_mode = "3D"
        self.show_minimap = True
        
        self.track_data = build_track_geometry(track_length_m=850.0, ds=2.0, track_width_m=3.8)
        self.track_mesh = TrackCollisionMesh(self.track_data, road_surface_2d=getattr(self, 'road_surface_2d', None))
        
        self.cam_dist = 6.8
        self.cam_height = 3.6
        self.fov = 420.0
        self.smooth_heading = 0.0
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.vp = None

    def set_vehicle_params(self, vp):
        self.vp = vp

    def update_vehicle_state(self, state):
        self.car_state = state
        self.trail.append((state['x'], state['y']))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
            
        rad = np.deg2rad(state['psi_deg'])
        self.smooth_heading = 0.88 * self.smooth_heading + 0.12 * rad
        self.smooth_x = 0.85 * self.smooth_x + 0.15 * state['x']
        self.smooth_y = 0.85 * self.smooth_y + 0.15 * state['y']
        self.update()

    def project_3d_point(self, x_world, y_world, z_world, cam_x, cam_y, cam_z, cam_yaw):
        """
        Projects world (X, Y, Z) point onto screen viewport.
        World coordinates: +X forward, +Y LEFT, +Z UP.
        Camera looks along cam_yaw vector (cos(yaw), sin(yaw)).
        Screen X is perpendicular to cam_yaw: RIGHT is positive (sx > cx).
        Screen Y is vertical: DOWN is positive in Qt (sy > cy for negative Z).
        """
        dx = x_world - cam_x
        dy = y_world - cam_y
        dz = z_world - cam_z
        
        # Forward depth along camera line of sight:
        y_depth = dx * np.cos(cam_yaw) + dy * np.sin(cam_yaw)
        
        if y_depth <= 0.6:
            return None
            
        # Lateral offset to the RIGHT of camera line of sight:
        # Since world +Y is LEFT, the unit vector to the RIGHT is (sin(cam_yaw), -cos(cam_yaw)):
        x_cam_right = dx * np.sin(cam_yaw) - dy * np.cos(cam_yaw)
        z_cam_up = dz
        
        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0 + 35.0
        
        sx = cx + (x_cam_right / y_depth) * self.fov
        sy = cy - (z_cam_up / y_depth) * self.fov
        return QPointF(sx, sy), float(y_depth)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()

        sky_grad = QLinearGradient(0, 0, 0, int(h * 0.44))
        sky_grad.setColorAt(0.0, QColor(92, 160, 211))
        sky_grad.setColorAt(1.0, QColor(190, 220, 240))
        painter.fillRect(0, 0, w, int(h * 0.44), sky_grad)

        grass_grad = QLinearGradient(0, int(h * 0.44), 0, h)
        grass_grad.setColorAt(0.0, QColor(46, 125, 50))
        grass_grad.setColorAt(1.0, QColor(34, 94, 38))
        painter.fillRect(0, int(h * 0.44), w, int(h * 0.56), grass_grad)

        if not self.car_state:
            return

        x_car = self.smooth_x
        y_car = self.smooth_y
        yaw_car = self.smooth_heading
        
        if self.view_mode == "3D":
            self.paint_3d_view(painter, w, h, x_car, y_car, yaw_car)
        else:
            self.paint_2d_view(painter, w, h, x_car, y_car, yaw_car)

    def paint_3d_view(self, painter, w, h, x_car, y_car, yaw_car):
        z_car = float(self.car_state.get('z_road', 0.0))
        cam_x = x_car - self.cam_dist * np.cos(yaw_car)
        cam_y = y_car - self.cam_dist * np.sin(yaw_car)
        cam_z = z_car + self.cam_height
        
        x_cnt = self.track_data['x_center']
        y_cnt = self.track_data['y_center']
        x_l = self.track_data['x_left']
        y_l = self.track_data['y_left']
        x_r = self.track_data['x_right']
        y_r = self.track_data['y_right']
        N_pts = len(x_cnt)
        z_l_arr = self.track_data.get('z_left', np.zeros(N_pts))
        z_r_arr = self.track_data.get('z_right', np.zeros(N_pts))
        
        dists = (x_cnt - x_car)**2 + (y_cnt - y_car)**2
        idx_near = int(np.argmin(dists))
        idx_start = max(0, idx_near - 20)
        idx_end = min(N_pts - 1, idx_near + 80)

        segments = []
        for i in range(idx_start, idx_end):
            zl1 = float(z_l_arr[i]) if i < len(z_l_arr) else 0.0
            zr1 = float(z_r_arr[i]) if i < len(z_r_arr) else 0.0
            zl2 = float(z_l_arr[i+1]) if (i+1) < len(z_l_arr) else 0.0
            zr2 = float(z_r_arr[i+1]) if (i+1) < len(z_r_arr) else 0.0
            p_l1 = self.project_3d_point(x_l[i], y_l[i], zl1, cam_x, cam_y, cam_z, yaw_car)
            p_r1 = self.project_3d_point(x_r[i], y_r[i], zr1, cam_x, cam_y, cam_z, yaw_car)
            p_l2 = self.project_3d_point(x_l[i+1], y_l[i+1], zl2, cam_x, cam_y, cam_z, yaw_car)
            p_r2 = self.project_3d_point(x_r[i+1], y_r[i+1], zr2, cam_x, cam_y, cam_z, yaw_car)
            
            if p_l1 and p_r1 and p_l2 and p_r2:
                avg_depth = (p_l1[1] + p_r1[1] + p_l2[1] + p_r2[1]) / 4.0
                segments.append({
                    'depth': avg_depth,
                    'idx': i,
                    'p_l1': p_l1[0], 'p_r1': p_r1[0],
                    'p_l2': p_l2[0], 'p_r2': p_r2[0]
                })

        segments.sort(key=lambda s: s['depth'], reverse=True)

        road_brush = QBrush(QColor(52, 73, 94))
        curb_pen_l = QPen(QColor(231, 76, 60), 3)
        curb_pen_r = QPen(QColor(241, 196, 15), 3)

        for seg in segments:
            poly = QPolygonF([seg['p_l1'], seg['p_r1'], seg['p_r2'], seg['p_l2']])
            painter.setPen(Qt.NoPen)
            painter.setBrush(road_brush)
            painter.drawPolygon(poly)
            
            painter.setPen(curb_pen_l)
            painter.drawLine(seg['p_l1'], seg['p_l2'])
            painter.setPen(curb_pen_r)
            painter.drawLine(seg['p_r1'], seg['p_r2'])

        self.draw_3d_car(painter, cam_x, cam_y, cam_z, yaw_car)

    def draw_3d_car(self, painter, cam_x, cam_y, cam_z, cam_yaw):
        xv = self.car_state['x']
        yv = self.car_state['y']
        z_road = float(self.car_state.get('z_road', 0.0))
        psi_car = np.deg2rad(self.car_state['psi_deg'])
        phi_car = np.deg2rad(self.car_state['phi_deg'])
        theta_car = np.deg2rad(self.car_state['theta_deg'])
        steer_rad = np.deg2rad(self.car_state.get('steer_deg', 0.0))

        L = 1.53
        W = 1.25
        a = L * 0.53
        b = L * 0.47
        tf_half = W / 2.0
        
        # Car local frame: +lx forward, +ly to the LEFT, +lz up
        def car_to_world(lx, ly, lz):
            lz_eff = lz + ly * np.sin(phi_car) - lx * np.sin(theta_car)
            ly_eff = ly * np.cos(phi_car)
            lx_eff = lx * np.cos(theta_car)
            
            # In world coordinates: heading psi has +X fwd (cos(psi), sin(psi)), +Y left (-sin(psi), cos(psi))
            cos_p = np.cos(psi_car)
            sin_p = np.sin(psi_car)
            wx = xv + lx_eff * cos_p - ly_eff * sin_p
            wy = yv + lx_eff * sin_p + ly_eff * cos_p
            wz = max(z_road + 0.12 + lz_eff, z_road + 0.02)
            return wx, wy, wz

        def prj(lx, ly, lz):
            wx, wy, wz = car_to_world(lx, ly, lz)
            return self.project_3d_point(wx, wy, wz, cam_x, cam_y, cam_z, cam_yaw)

        # Ground Shadow
        s1 = prj(1.1, 0.65, 0.0); s2 = prj(1.1, -0.65, 0.0)
        s3 = prj(-0.9, -0.65, 0.0); s4 = prj(-0.9, 0.65, 0.0)
        if s1 and s2 and s3 and s4:
            painter.setBrush(QBrush(QColor(15, 15, 20, 160)))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygonF([s1[0], s2[0], s3[0], s4[0]]))

        # Chassis Monocoque
        c_front_l = prj(0.7, 0.32, 0.35); c_front_r = prj(0.7, -0.32, 0.35)
        c_rear_l = prj(-0.7, 0.38, 0.40); c_rear_r = prj(-0.7, -0.38, 0.40)
        c_nose = prj(1.05, 0.0, 0.22)
        if c_front_l and c_front_r and c_rear_l and c_rear_r and c_nose:
            painter.setBrush(QBrush(QColor(41, 128, 185)))
            painter.setPen(QPen(QColor(236, 240, 241), 1.5))
            painter.drawPolygon(QPolygonF([c_nose[0], c_front_r[0], c_rear_r[0], c_rear_l[0], c_front_l[0]]))
            
            cp1 = prj(0.25, 0.22, 0.37); cp2 = prj(0.25, -0.22, 0.37)
            cp3 = prj(-0.25, -0.22, 0.39); cp4 = prj(-0.25, 0.22, 0.39)
            if cp1 and cp2 and cp3 and cp4:
                painter.setBrush(QBrush(QColor(20, 20, 25)))
                painter.drawPolygon(QPolygonF([cp1[0], cp2[0], cp3[0], cp4[0]]))

        # Front & Rear Wings
        fw_l = prj(1.15, 0.75, 0.12); fw_r = prj(1.15, -0.75, 0.12)
        fw_bl = prj(0.85, 0.75, 0.12); fw_br = prj(0.85, -0.75, 0.12)
        if fw_l and fw_r and fw_bl and fw_br:
            painter.setBrush(QBrush(QColor(231, 76, 60)))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawPolygon(QPolygonF([fw_l[0], fw_r[0], fw_br[0], fw_bl[0]]))

        rw_l = prj(-0.85, 0.72, 0.82); rw_r = prj(-0.85, -0.72, 0.82)
        rw_bl = prj(-0.65, 0.72, 0.82); rw_br = prj(-0.65, -0.72, 0.82)
        if rw_l and rw_r and rw_bl and rw_br:
            painter.setBrush(QBrush(QColor(30, 39, 46)))
            painter.setPen(QPen(QColor(0, 206, 201), 1.2))
            painter.drawPolygon(QPolygonF([rw_l[0], rw_r[0], rw_br[0], rw_bl[0]]))

        # 4 Wheels (Exact wheel orientation: steer_rad > 0 turns wheel to the RIGHT)
        def draw_3d_wheel(wheel_x, wheel_y, steer_a, util_pct):
            if util_pct < 60:
                col = QColor(46, 204, 113)
            elif util_pct < 88:
                col = QColor(241, 196, 15)
            else:
                col = QColor(231, 76, 60)
                
            w_rad = 0.20
            w_wid = 0.16
            cos_s = np.cos(steer_a)
            sin_s = np.sin(steer_a)
            
            # Local wheel coordinate transformation with steering rotation:
            # steer_a > 0 (turning right) rotates front of wheel (dx > 0) towards -ly (Right)
            def w_pt(dx, dy, dz):
                lx = wheel_x + dx * cos_s + dy * sin_s
                ly = wheel_y - dx * sin_s + dy * cos_s
                return prj(lx, ly, dz)
                
            w1 = w_pt(w_rad, w_wid/2, 0.22); w2 = w_pt(w_rad, -w_wid/2, 0.22)
            w3 = w_pt(-w_rad, -w_wid/2, 0.22); w4 = w_pt(-w_rad, w_wid/2, 0.22)
            if w1 and w2 and w3 and w4:
                painter.setBrush(QBrush(col))
                painter.setPen(QPen(Qt.black, 1.2))
                painter.drawPolygon(QPolygonF([w1[0], w2[0], w3[0], w4[0]]))

        # Front Wheels steered with steer_rad (+tf_half is Left, -tf_half is Right)
        draw_3d_wheel(a, tf_half, steer_rad, self.car_state.get('util_fl', 0))
        draw_3d_wheel(a, -tf_half, steer_rad, self.car_state.get('util_fr', 0))
        # Rear Wheels
        draw_3d_wheel(-b, tf_half, 0.0, self.car_state.get('util_rl', 0))
        draw_3d_wheel(-b, -tf_half, 0.0, self.car_state.get('util_rr', 0))

        self._draw_hud(painter, w, h)

    def paint_2d_view(self, painter, w, h, x_car, y_car, yaw_car):
        zoom = 5.0
        cx = w / 2.0
        cy = h / 2.0
        def to_screen(xm, ym):
            return QPointF(cx + (xm - x_car) * zoom, cy - (ym - y_car) * zoom)

        xc = self.track_data['x_center']
        yc = self.track_data['y_center']
        xl = self.track_data['x_left']
        yl = self.track_data['y_left']
        xr = self.track_data['x_right']
        yr = self.track_data['y_right']

        # Track Edges
        painter.setPen(QPen(QColor(231, 76, 60, 200), 2))
        for i in range(1, len(xl)):
            painter.drawLine(to_screen(xl[i-1], yl[i-1]), to_screen(xl[i], yl[i]))

        painter.setPen(QPen(QColor(241, 196, 15, 200), 2))
        for i in range(1, len(xr)):
            painter.drawLine(to_screen(xr[i-1], yr[i-1]), to_screen(xr[i], yr[i]))

        # Centerline
        painter.setPen(QPen(QColor(255, 255, 255, 140), 1.5, Qt.DashLine))
        for i in range(1, len(xc)):
            painter.drawLine(to_screen(xc[i-1], yc[i-1]), to_screen(xc[i], yc[i]))

        # Vehicle Body in 2D
        cos_p = np.cos(yaw_car)
        sin_p = np.sin(yaw_car)
        steer_rad = np.deg2rad(self.car_state.get('steer_deg', 0.0))

        def car_2d_pt(lx, ly):
            wx = x_car + lx * cos_p - ly * sin_p
            wy = y_car + lx * sin_p + ly * cos_p
            return to_screen(wx, wy)

        # Body outline
        poly_body = QPolygonF([
            car_2d_pt(1.0, 0.0),
            car_2d_pt(0.7, -0.35),
            car_2d_pt(-0.7, -0.40),
            car_2d_pt(-0.7, 0.40),
            car_2d_pt(0.7, 0.35)
        ])
        painter.setBrush(QBrush(QColor(41, 128, 185, 220)))
        painter.setPen(QPen(QColor(236, 240, 241), 1.5))
        painter.drawPolygon(poly_body)

        # 4 Wheels in 2D
        def draw_2d_wheel(wx_c, wy_c, d_ang, util_pct):
            c_d = np.cos(d_ang); s_d = np.sin(d_ang)
            col = QColor(46, 204, 113) if util_pct < 60 else (QColor(241, 196, 15) if util_pct < 88 else QColor(231, 76, 60))
            def w_pt(dx, dy):
                lx = wx_c + dx * c_d + dy * s_d
                ly = wy_c - dx * s_d + dy * c_d
                return car_2d_pt(lx, ly)
            w_poly = QPolygonF([w_pt(0.22, 0.08), w_pt(0.22, -0.08), w_pt(-0.22, -0.08), w_pt(-0.22, 0.08)])
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(Qt.black, 1.2))
            painter.drawPolygon(w_poly)

        steer_fl_2d = np.radians(float(self.car_state.get('steer_fl_deg', np.degrees(steer_rad))))
        steer_fr_2d = np.radians(float(self.car_state.get('steer_fr_deg', np.degrees(steer_rad))))
        draw_2d_wheel(0.8, 0.6, steer_fl_2d, self.car_state.get('util_fl', 0))
        draw_2d_wheel(0.8, -0.6, steer_fr_2d, self.car_state.get('util_fr', 0))
        draw_2d_wheel(-0.7, 0.6, 0.0, self.car_state.get('util_rl', 0))
        if getattr(self, 'show_minimap', True) and self.track_data is not None and h > 460:
            mm_w = 260
            mm_h = 240
            mm_x = w - mm_w - 20
            mm_y = 20
            self._draw_minimap(painter, mm_x, mm_y, mm_w, mm_h)

        self._draw_hud(painter, w, h)

    _draw_minimap = OpenGLViewport3D._draw_minimap





