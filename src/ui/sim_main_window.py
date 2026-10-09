from src.core.gamepad_manager import FFB_PRESETS
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
from src.core.physics_thread import PhysicsThread14DOF
from src.ui.sim_viewport import get_viewport_class
from src.ui.sim_telemetry import FullTelemetryWindow, TelemetryPopoutDialog, LapSelectionDialog, CustomTelemetryLayoutDialog
from src.ui.sim_controls import TrackDesignerDialog, RoadProfileEditorDialog
from src.core.gamepad_manager import GamepadManager


class Interactive14DOFDrivingWindow(QMainWindow):
    setup_changed = pyqtSignal(object)

    def __init__(self, vehicle_params=None, tir_file=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("14-DOF Real-Time Driving Simulator (FSAE)")
        self.resize(1280, 840)
        self.setMinimumSize(960, 600)
        self.setStyleSheet("background-color: #242b35; color: #ecf0f1; font-family: Segoe UI, sans-serif;")

        self.vp = vehicle_params if vehicle_params is not None else VehicleParams.nominal_fsae_params()
        self.tir_file = tir_file
        self.is_fullscreen_mode = False
        self._skip_reset_prompt = False
        
        self.physics_thread = PhysicsThread14DOF(vehicle_params=self.vp, tir_file=self.tir_file, target_hz=250.0)
        self.physics_thread.state_updated.connect(self.on_physics_frame)

        self.init_ui()
        self.physics_thread.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #34495e;
                width: 6px;
            }
            QSplitter::handle:hover {
                background-color: #00cec9;
            }
        """)
        main_layout.addWidget(self.main_splitter)

        ViewportClass = get_viewport_class()
        self.viewport = ViewportClass(self)
        if hasattr(self.viewport, 'set_vehicle_params'):
            self.viewport.set_vehicle_params(self.vp)
        else:
            self.viewport.vp = self.vp
        self.main_splitter.addWidget(self.viewport)

        self.right_panel = QWidget()
        self.right_panel.setMinimumWidth(360)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.main_splitter.addWidget(self.right_panel)

        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setSizes([850, 480])

        tabs_right = QTabWidget()
        right_layout.addWidget(tabs_right)

        # --- TAB 1: HUD DASHBOARD & TELEMETRY ---
        tab_hud = QWidget()
        hud_layout = QVBoxLayout(tab_hud)

        # Monitor Real-Time & Physics Performance
        rt_group = QGroupBox("Monitor Real-Time e Prestazioni Fisiche")
        rt_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        rt_layout = QGridLayout(rt_group)
        
        self.lbl_hz = QLabel("Frequenza Fisica: 250 Hz")
        self.lbl_hz.setStyleSheet("font-size: 11px; font-weight: bold; color: #2ecc71;")
        self.lbl_step = QLabel("Tempo Step RK4: 0.25 ms")
        self.lbl_step.setStyleSheet("font-size: 11px; color: #ecf0f1;")
        self.lbl_ratio = QLabel("Ratio Real-Time: 1.00x (OK)")
        self.lbl_ratio.setStyleSheet("font-size: 11px; font-weight: bold; color: #00cec9;")
        self.lbl_fps = QLabel("Frame Rate Video: 60 FPS")
        self.lbl_fps.setStyleSheet("font-size: 11px; font-weight: bold; color: #2ecc71;")
        
        rt_layout.addWidget(self.lbl_hz, 0, 0)
        rt_layout.addWidget(self.lbl_step, 0, 1)
        rt_layout.addWidget(self.lbl_ratio, 1, 0)
        rt_layout.addWidget(self.lbl_fps, 1, 1)
        hud_layout.addWidget(rt_group)

        # Speed & Timer
        speed_group = QGroupBox("Strumentazione Pilota")
        speed_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        speed_layout = QVBoxLayout(speed_group)
        
        self.lbl_speed = QLabel("0.0 km/h")
        self.lbl_speed.setStyleSheet("font-size: 34px; font-weight: bold; color: #2ecc71;")
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.lbl_speed)
        
        self.lbl_timer = QLabel("Lap Time: 00:00.00")
        self.lbl_timer.setStyleSheet("font-size: 15px; font-weight: bold; color: #f1c40f;")
        self.lbl_timer.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.lbl_timer)
        hud_layout.addWidget(speed_group)

        # 4-Wheel Grip & Corner Weights
        grip_group = QGroupBox("Corner Weights e Carichi 4 Ruote (MF6.1 + Jacking)")
        grip_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        grip_layout = QGridLayout(grip_group)
        
        self.pb_fl = QProgressBar(); self.pb_fr = QProgressBar()
        self.pb_rl = QProgressBar(); self.pb_rr = QProgressBar()
        self.lbl_fl = QLabel("FL: 0N"); self.lbl_fr = QLabel("FR: 0N")
        self.lbl_rl = QLabel("RL: 0N"); self.lbl_rr = QLabel("RR: 0N")
        
        grip_layout.addWidget(QLabel("<b>FL:</b>"), 0, 0); grip_layout.addWidget(self.pb_fl, 0, 1); grip_layout.addWidget(self.lbl_fl, 0, 2)
        grip_layout.addWidget(QLabel("<b>FR:</b>"), 1, 0); grip_layout.addWidget(self.pb_fr, 1, 1); grip_layout.addWidget(self.lbl_fr, 1, 2)
        grip_layout.addWidget(QLabel("<b>RL:</b>"), 2, 0); grip_layout.addWidget(self.pb_rl, 2, 1); grip_layout.addWidget(self.lbl_rl, 2, 2)
        grip_layout.addWidget(QLabel("<b>RR:</b>"), 3, 0); grip_layout.addWidget(self.pb_rr, 3, 1); grip_layout.addWidget(self.lbl_rr, 3, 2)
        hud_layout.addWidget(grip_group)

        # Dynamics & Kinematics Summary
        dyn_group = QGroupBox("Cinematica ed Escursioni 14-DOF Live")
        dyn_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        dyn_layout = QGridLayout(dyn_group)
        
        self.lbl_ay = QLabel("Acc. Laterale Ay: 0.00 g")
        self.lbl_ax = QLabel("Acc. Longitudinale Ax: 0.00 g")
        self.lbl_phi = QLabel("Rollio: 0.00 deg")
        self.lbl_theta = QLabel("Beccheggio: 0.00 deg")
        
        self.lbl_z_front = QLabel("Escursione Ant: 0.0 / 0.0 mm")
        self.lbl_z_rear = QLabel("Escursione Post: 0.0 / 0.0 mm")
        self.lbl_cam_live = QLabel("Camber Ant: -2.5 deg | Post: -1.5 deg")
        self.lbl_jack_live = QLabel("Jacking Caster/KPI: Delta Fz=0 N")
        
        dyn_layout.addWidget(self.lbl_ay, 0, 0); dyn_layout.addWidget(self.lbl_ax, 0, 1)
        dyn_layout.addWidget(self.lbl_phi, 1, 0); dyn_layout.addWidget(self.lbl_theta, 1, 1)
        dyn_layout.addWidget(self.lbl_z_front, 2, 0); dyn_layout.addWidget(self.lbl_z_rear, 2, 1)
        dyn_layout.addWidget(self.lbl_cam_live, 3, 0); dyn_layout.addWidget(self.lbl_jack_live, 3, 1)
        
        self.lbl_rc_live = QLabel("Centri Rollio [f(φ)]: Ant: h=46.3mm, y=0.0mm | Post: h=76.9mm, y=0.0mm")
        self.lbl_rc_live.setStyleSheet("font-size: 10px; color: #74b9ff; font-weight: bold;")
        self.lbl_rc_live.setWordWrap(True)
        dyn_layout.addWidget(self.lbl_rc_live, 4, 0, 1, 2)
        hud_layout.addWidget(dyn_group)

        # -------------------------------------------------------------
        # 1.5. TERNA FORZE RUOTA TRIASSIALI (Fx, Fy, Fz, |F|) [N]
        # -------------------------------------------------------------
        self.peak_loads = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}

        loads_group = QGroupBox("Terna Forze Ruote Triassiali (Fx, Fy, Fz) [N] — Load Cases")
        loads_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        loads_layout = QVBoxLayout(loads_group)

        loads_top_h = QHBoxLayout()
        loads_top_h.addWidget(QLabel("<b>Vista:</b>"))
        self.combo_loads_view = QComboBox()
        self.combo_loads_view.addItems([
            "Tabella Completa (4 Ruote)",
            "Ruota Anteriore SX (FL)",
            "Ruota Anteriore DX (FR)",
            "Ruota Posteriore SX (RL)",
            "Ruota Posteriore DX (RR)"
        ])
        self.combo_loads_view.setStyleSheet("background-color: #2d3436; color: white; padding: 3px; font-weight: bold;")
        self.combo_loads_view.currentIndexChanged.connect(self._on_loads_view_changed)
        loads_top_h.addWidget(self.combo_loads_view, 1)

        self.btn_reset_load_peaks = QPushButton("↺ Reset Picchi")
        self.btn_reset_load_peaks.setStyleSheet("font-size: 10px; padding: 3px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        self.btn_reset_load_peaks.clicked.connect(self._reset_load_peaks)
        loads_top_h.addWidget(self.btn_reset_load_peaks)
        loads_layout.addLayout(loads_top_h)

        # Container A: 4-Wheel Table
        self.widget_loads_table = QWidget()
        t_layout = QGridLayout(self.widget_loads_table)
        t_layout.setContentsMargins(0, 4, 0, 0)
        t_layout.setHorizontalSpacing(6)
        t_layout.setVerticalSpacing(3)

        hdr_style = "font-size: 10px; font-weight: bold; color: #81ecec;"
        t_layout.addWidget(QLabel("<b>Ruota</b>"), 0, 0)
        lbl_h_fx = QLabel("Fx [N]"); lbl_h_fx.setStyleSheet(hdr_style)
        lbl_h_fy = QLabel("Fy [N]"); lbl_h_fy.setStyleSheet(hdr_style)
        lbl_h_fz = QLabel("Fz [N]"); lbl_h_fz.setStyleSheet(hdr_style)
        lbl_h_ft = QLabel("|F| [N]"); lbl_h_ft.setStyleSheet("font-size: 10px; font-weight: bold; color: #f1c40f;")
        t_layout.addWidget(lbl_h_fx, 0, 1)
        t_layout.addWidget(lbl_h_fy, 0, 2)
        t_layout.addWidget(lbl_h_fz, 0, 3)
        t_layout.addWidget(lbl_h_ft, 0, 4)

        cell_style = "font-size: 11px; font-family: Consolas;"
        self.lbl_fx_fl = QLabel("+0 N"); self.lbl_fx_fl.setStyleSheet(cell_style)
        self.lbl_fy_fl = QLabel("+0 N"); self.lbl_fy_fl.setStyleSheet(cell_style)
        self.lbl_fz_fl = QLabel("0 N");  self.lbl_fz_fl.setStyleSheet(cell_style)
        self.lbl_ftot_fl = QLabel("0 N"); self.lbl_ftot_fl.setStyleSheet(cell_style + "font-weight: bold; color: #f1c40f;")

        self.lbl_fx_fr = QLabel("+0 N"); self.lbl_fx_fr.setStyleSheet(cell_style)
        self.lbl_fy_fr = QLabel("+0 N"); self.lbl_fy_fr.setStyleSheet(cell_style)
        self.lbl_fz_fr = QLabel("0 N");  self.lbl_fz_fr.setStyleSheet(cell_style)
        self.lbl_ftot_fr = QLabel("0 N"); self.lbl_ftot_fr.setStyleSheet(cell_style + "font-weight: bold; color: #f1c40f;")

        self.lbl_fx_rl = QLabel("+0 N"); self.lbl_fx_rl.setStyleSheet(cell_style)
        self.lbl_fy_rl = QLabel("+0 N"); self.lbl_fy_rl.setStyleSheet(cell_style)
        self.lbl_fz_rl = QLabel("0 N");  self.lbl_fz_rl.setStyleSheet(cell_style)
        self.lbl_ftot_rl = QLabel("0 N"); self.lbl_ftot_rl.setStyleSheet(cell_style + "font-weight: bold; color: #f1c40f;")

        self.lbl_fx_rr = QLabel("+0 N"); self.lbl_fx_rr.setStyleSheet(cell_style)
        self.lbl_fy_rr = QLabel("+0 N"); self.lbl_fy_rr.setStyleSheet(cell_style)
        self.lbl_fz_rr = QLabel("0 N");  self.lbl_fz_rr.setStyleSheet(cell_style)
        self.lbl_ftot_rr = QLabel("0 N"); self.lbl_ftot_rr.setStyleSheet(cell_style + "font-weight: bold; color: #f1c40f;")

        t_layout.addWidget(QLabel("<b>FL:</b>"), 1, 0)
        t_layout.addWidget(self.lbl_fx_fl, 1, 1); t_layout.addWidget(self.lbl_fy_fl, 1, 2); t_layout.addWidget(self.lbl_fz_fl, 1, 3); t_layout.addWidget(self.lbl_ftot_fl, 1, 4)

        t_layout.addWidget(QLabel("<b>FR:</b>"), 2, 0)
        t_layout.addWidget(self.lbl_fx_fr, 2, 1); t_layout.addWidget(self.lbl_fy_fr, 2, 2); t_layout.addWidget(self.lbl_fz_fr, 2, 3); t_layout.addWidget(self.lbl_ftot_fr, 2, 4)

        t_layout.addWidget(QLabel("<b>RL:</b>"), 3, 0)
        t_layout.addWidget(self.lbl_fx_rl, 3, 1); t_layout.addWidget(self.lbl_fy_rl, 3, 2); t_layout.addWidget(self.lbl_fz_rl, 3, 3); t_layout.addWidget(self.lbl_ftot_rl, 3, 4)

        t_layout.addWidget(QLabel("<b>RR:</b>"), 4, 0)
        t_layout.addWidget(self.lbl_fx_rr, 4, 1); t_layout.addWidget(self.lbl_fy_rr, 4, 2); t_layout.addWidget(self.lbl_fz_rr, 4, 3); t_layout.addWidget(self.lbl_ftot_rr, 4, 4)

        loads_layout.addWidget(self.widget_loads_table)

        # Container B: Single Wheel Detailed Card
        self.widget_loads_single = QWidget()
        s_layout = QVBoxLayout(self.widget_loads_single)
        s_layout.setContentsMargins(0, 4, 0, 0)
        
        self.lbl_single_wheel_title = QLabel("<b>Dettaglio Carichi: FL (Ant SX)</b>")
        self.lbl_single_wheel_title.setStyleSheet("color: #00cec9; font-size: 11px;")
        s_layout.addWidget(self.lbl_single_wheel_title)

        card_style = "font-size: 11px; font-family: Consolas; padding: 2px;"
        self.lbl_single_fx = QLabel("Fx (Longitudinale Spinta/Freno): +0 N")
        self.lbl_single_fx.setStyleSheet(card_style + "color: #3498db;")
        self.lbl_single_fy = QLabel("Fy (Laterale Aderenza Curva): +0 N")
        self.lbl_single_fy.setStyleSheet(card_style + "color: #e74c3c;")
        self.lbl_single_fz = QLabel("Fz (Carico Verticale Normale): 0 N")
        self.lbl_single_fz.setStyleSheet(card_style + "color: #2ecc71;")
        self.lbl_single_ftot = QLabel("|F| (Risultante 3D Vettoriale): 0 N")
        self.lbl_single_ftot.setStyleSheet(card_style + "font-weight: bold; color: #f1c40f;")
        self.lbl_single_fpeak = QLabel("|F|_max (Picco Sessione): 0 N")
        self.lbl_single_fpeak.setStyleSheet(card_style + "font-weight: bold; color: #e17055;")

        s_layout.addWidget(self.lbl_single_fx)
        s_layout.addWidget(self.lbl_single_fy)
        s_layout.addWidget(self.lbl_single_fz)
        s_layout.addWidget(self.lbl_single_ftot)
        s_layout.addWidget(self.lbl_single_fpeak)

        loads_layout.addWidget(self.widget_loads_single)
        self.widget_loads_single.setVisible(False)  # Default: table is visible

        self.btn_open_wheel_forces = QPushButton("Apri Grafici Forze Ruote (Fy, Fx, Fz 4 Ruote)...")
        self.btn_open_wheel_forces.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 5px; border-radius: 3px;")
        self.btn_open_wheel_forces.setToolTip("Apre i grafici di telemetria focalizzati sul confronto a 4 ruote di Fy, Fx, Fz e coppie")
        self.btn_open_wheel_forces.clicked.connect(lambda: self.open_full_telemetry_dialog(initial_preset="2. Forze Ruote 4 Angoli (Fy, Fx, Fz, Coppie)"))
        loads_layout.addWidget(self.btn_open_wheel_forces)

        hud_layout.addWidget(loads_group)

        # -------------------------------------------------------------
        # 1.6. COPPIE RUOTE: TRAZIONE & FRENATA [Nm]
        # -------------------------------------------------------------
        torques_group = QGroupBox("Coppie Ruote: Trazione & Frenata [Nm]")
        torques_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        torques_layout = QVBoxLayout(torques_group)

        tq_grid = QGridLayout()
        tq_grid.setHorizontalSpacing(8)
        tq_grid.setVerticalSpacing(3)

        tq_grid.addWidget(QLabel("<b>Ruota</b>"), 0, 0)
        tq_grid.addWidget(QLabel("<b style='color:#0984e3;'>Trazione [Nm]</b>"), 0, 1)
        tq_grid.addWidget(QLabel("<b style='color:#e74c3c;'>Frenata [Nm]</b>"), 0, 2)
        tq_grid.addWidget(QLabel("<b style='color:#f1c40f;'>Netta [Nm]</b>"), 0, 3)

        tq_style = "font-size: 11px; font-family: Consolas;"
        self.lbl_tdrv_fl = QLabel("0.0 Nm"); self.lbl_tdrv_fl.setStyleSheet(tq_style + "color: #7f8c8d;")
        self.lbl_tbrk_fl = QLabel("0.0 Nm"); self.lbl_tbrk_fl.setStyleSheet(tq_style)
        self.lbl_tnet_fl = QLabel("0.0 Nm"); self.lbl_tnet_fl.setStyleSheet(tq_style)

        self.lbl_tdrv_fr = QLabel("0.0 Nm"); self.lbl_tdrv_fr.setStyleSheet(tq_style + "color: #7f8c8d;")
        self.lbl_tbrk_fr = QLabel("0.0 Nm"); self.lbl_tbrk_fr.setStyleSheet(tq_style)
        self.lbl_tnet_fr = QLabel("0.0 Nm"); self.lbl_tnet_fr.setStyleSheet(tq_style)

        self.lbl_tdrv_rl = QLabel("0.0 Nm"); self.lbl_tdrv_rl.setStyleSheet(tq_style + "color: #0984e3; font-weight: bold;")
        self.lbl_tbrk_rl = QLabel("0.0 Nm"); self.lbl_tbrk_rl.setStyleSheet(tq_style)
        self.lbl_tnet_rl = QLabel("0.0 Nm"); self.lbl_tnet_rl.setStyleSheet(tq_style)

        self.lbl_tdrv_rr = QLabel("0.0 Nm"); self.lbl_tdrv_rr.setStyleSheet(tq_style + "color: #0984e3; font-weight: bold;")
        self.lbl_tbrk_rr = QLabel("0.0 Nm"); self.lbl_tbrk_rr.setStyleSheet(tq_style)
        self.lbl_tnet_rr = QLabel("0.0 Nm"); self.lbl_tnet_rr.setStyleSheet(tq_style)

        tq_grid.addWidget(QLabel("<b>FL:</b>"), 1, 0); tq_grid.addWidget(self.lbl_tdrv_fl, 1, 1); tq_grid.addWidget(self.lbl_tbrk_fl, 1, 2); tq_grid.addWidget(self.lbl_tnet_fl, 1, 3)
        tq_grid.addWidget(QLabel("<b>FR:</b>"), 2, 0); tq_grid.addWidget(self.lbl_tdrv_fr, 2, 1); tq_grid.addWidget(self.lbl_tbrk_fr, 2, 2); tq_grid.addWidget(self.lbl_tnet_fr, 2, 3)
        tq_grid.addWidget(QLabel("<b>RL:</b>"), 3, 0); tq_grid.addWidget(self.lbl_tdrv_rl, 3, 1); tq_grid.addWidget(self.lbl_tbrk_rl, 3, 2); tq_grid.addWidget(self.lbl_tnet_rl, 3, 3)
        tq_grid.addWidget(QLabel("<b>RR:</b>"), 4, 0); tq_grid.addWidget(self.lbl_tdrv_rr, 4, 1); tq_grid.addWidget(self.lbl_tbrk_rr, 4, 2); tq_grid.addWidget(self.lbl_tnet_rr, 4, 3)

        torques_layout.addLayout(tq_grid)

        self.lbl_torque_summary = QLabel("Trazione Tot: 0.0 Nm (50/50) | Freno Tot: 0.0 Nm (Bias 60/40%)")
        self.lbl_torque_summary.setStyleSheet("font-size: 10px; color: #81ecec; padding-top: 4px;")
        torques_layout.addWidget(self.lbl_torque_summary)

        hud_layout.addWidget(torques_group)

        # Powertrain Elettrico EMRAX 188 & Batteria 80 kW (FSAE EV)
        pt_group = QGroupBox("Powertrain Elettrico (EMRAX 188HV - Limite 80 kW)")
        pt_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        pt_layout = QVBoxLayout(pt_group)

        pt_pwr_h = QHBoxLayout()
        self.lbl_pt_pwr_total = QLabel("Potenza Elettrica Accumulatore: <b>0.0 / 80.0 kW</b>")
        self.lbl_pt_pwr_total.setStyleSheet("font-size: 11px; color: #ecf0f1;")
        pt_pwr_h.addWidget(self.lbl_pt_pwr_total)

        self.lbl_pt_pwr_status = QLabel("[OK] Sotto Limite 80kW")
        self.lbl_pt_pwr_status.setStyleSheet("font-size: 10px; font-weight: bold; color: #2ecc71; background-color: #2d3436; padding: 2px 6px; border-radius: 3px;")
        pt_pwr_h.addWidget(self.lbl_pt_pwr_status, 0, Qt.AlignRight)
        pt_layout.addLayout(pt_pwr_h)

        self.pb_pt_pwr_total = QProgressBar()
        self.pb_pt_pwr_total.setRange(0, 800)
        self.pb_pt_pwr_total.setValue(0)
        self.pb_pt_pwr_total.setTextVisible(False)
        self.pb_pt_pwr_total.setStyleSheet("QProgressBar { height: 12px; background-color: #2d3436; border-radius: 3px; } QProgressBar::chunk { background-color: #00cec9; }")
        pt_layout.addWidget(self.pb_pt_pwr_total)

        pt_motors_grid = QGridLayout()
        pt_motors_grid.setHorizontalSpacing(8)
        pt_motors_grid.setVerticalSpacing(2)

        pt_hdr_style = "font-size: 10px; font-weight: bold; color: #81ecec;"
        pt_motors_grid.addWidget(QLabel("<b>Motore Posteriore</b>"), 0, 0)
        lbl_h_rpm = QLabel("Regime"); lbl_h_rpm.setStyleSheet(pt_hdr_style)
        lbl_h_tq = QLabel("Coppia"); lbl_h_tq.setStyleSheet(pt_hdr_style)
        lbl_h_pw = QLabel("P_elec / P_mech"); lbl_h_pw.setStyleSheet(pt_hdr_style)
        lbl_h_eta = QLabel("Rend. η"); lbl_h_eta.setStyleSheet(pt_hdr_style)
        pt_motors_grid.addWidget(lbl_h_rpm, 0, 1)
        pt_motors_grid.addWidget(lbl_h_tq, 0, 2)
        pt_motors_grid.addWidget(lbl_h_pw, 0, 3)
        pt_motors_grid.addWidget(lbl_h_eta, 0, 4)

        pt_cell_style = "font-size: 10px; font-family: Consolas;"
        self.lbl_pt_rpm_l = QLabel("0 RPM"); self.lbl_pt_rpm_l.setStyleSheet(pt_cell_style)
        self.lbl_pt_tq_l = QLabel("0.0 Nm"); self.lbl_pt_tq_l.setStyleSheet(pt_cell_style)
        self.lbl_pt_pw_l = QLabel("0.0 / 0.0 kW"); self.lbl_pt_pw_l.setStyleSheet(pt_cell_style + "color: #0984e3; font-weight: bold;")
        self.lbl_pt_eta_l = QLabel("95.0 %"); self.lbl_pt_eta_l.setStyleSheet(pt_cell_style + "color: #2ecc71;")

        self.lbl_pt_rpm_r = QLabel("0 RPM"); self.lbl_pt_rpm_r.setStyleSheet(pt_cell_style)
        self.lbl_pt_tq_r = QLabel("0.0 Nm"); self.lbl_pt_tq_r.setStyleSheet(pt_cell_style)
        self.lbl_pt_pw_r = QLabel("0.0 / 0.0 kW"); self.lbl_pt_pw_r.setStyleSheet(pt_cell_style + "color: #0984e3; font-weight: bold;")
        self.lbl_pt_eta_r = QLabel("95.0 %"); self.lbl_pt_eta_r.setStyleSheet(pt_cell_style + "color: #2ecc71;")

        pt_motors_grid.addWidget(QLabel("<b>SX (RL):</b>"), 1, 0)
        pt_motors_grid.addWidget(self.lbl_pt_rpm_l, 1, 1)
        pt_motors_grid.addWidget(self.lbl_pt_tq_l, 1, 2)
        pt_motors_grid.addWidget(self.lbl_pt_pw_l, 1, 3)
        pt_motors_grid.addWidget(self.lbl_pt_eta_l, 1, 4)

        pt_motors_grid.addWidget(QLabel("<b>DX (RR):</b>"), 2, 0)
        pt_motors_grid.addWidget(self.lbl_pt_rpm_r, 2, 1)
        pt_motors_grid.addWidget(self.lbl_pt_tq_r, 2, 2)
        pt_motors_grid.addWidget(self.lbl_pt_pw_r, 2, 3)
        pt_motors_grid.addWidget(self.lbl_pt_eta_r, 2, 4)

        pt_layout.addLayout(pt_motors_grid)

        self.lbl_pt_tv_delta = QLabel("Torque Vectoring: ΔP=0.0 kW | Riduzione 4:1 (Coppia Ruota = 4 × Motore)")
        self.lbl_pt_tv_delta.setStyleSheet("font-size: 10px; color: #74b9ff; padding-top: 3px;")
        pt_layout.addWidget(self.lbl_pt_tv_delta)

        hud_layout.addWidget(pt_group)

        # 4-Channel Haptic & Force Feedback Monitor Group
        ffb_group = QGroupBox("Feedback Aptico Grilletti e Force Feedback Live")
        ffb_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        ffb_layout = QGridLayout(ffb_group)
        
        self.lbl_ffb_alert = QLabel("[OK] Aderenza Ottimale (Zona Lineare)")
        self.lbl_ffb_alert.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 4px; border-radius: 3px;")
        self.lbl_ffb_alert.setAlignment(Qt.AlignCenter)
        ffb_layout.addWidget(self.lbl_ffb_alert, 0, 0, 1, 2)
        
        ffb_layout.addWidget(QLabel("Grilletto Freno LT (Bloccaggio / Lockup):"), 1, 0)
        self.pb_haptic_lt = QProgressBar()
        self.pb_haptic_lt.setRange(0, 100); self.pb_haptic_lt.setValue(0)
        self.pb_haptic_lt.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
        ffb_layout.addWidget(self.pb_haptic_lt, 1, 1)
        
        ffb_layout.addWidget(QLabel("Grilletto Gas RT (Trazione/Wheelspin):"), 2, 0)
        self.pb_haptic_rt = QProgressBar()
        self.pb_haptic_rt.setRange(0, 100); self.pb_haptic_rt.setValue(0)
        self.pb_haptic_rt.setStyleSheet("QProgressBar::chunk { background-color: #f1c40f; }")
        ffb_layout.addWidget(self.pb_haptic_rt, 2, 1)

        ffb_layout.addWidget(QLabel("Rumble Telaio (Sovrasterzo/Snap):"), 3, 0)
        self.pb_rumble_h = QProgressBar()
        self.pb_rumble_h.setRange(0, 100); self.pb_rumble_h.setValue(0)
        self.pb_rumble_h.setStyleSheet("QProgressBar::chunk { background-color: #9b59b6; }")
        ffb_layout.addWidget(self.pb_rumble_h, 3, 1)

        ffb_layout.addWidget(QLabel("Rumble Gomme (Sottosterzo/Scrub):"), 4, 0)
        self.pb_rumble_l = QProgressBar()
        self.pb_rumble_l.setRange(0, 100); self.pb_rumble_l.setValue(0)
        self.pb_rumble_l.setStyleSheet("QProgressBar::chunk { background-color: #e67e22; }")
        ffb_layout.addWidget(self.pb_rumble_l, 4, 1)

        hud_layout.addWidget(ffb_group)
        hud_layout.addStretch()

        scroll_hud = QScrollArea()
        scroll_hud.setWidgetResizable(True)
        scroll_hud.setFrameShape(QFrame.NoFrame)
        scroll_hud.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_hud.setWidget(tab_hud)
        tabs_right.addTab(scroll_hud, "Telemetria Live")

        # --- TAB 2: ASSETTO RAPIDO & AERODINAMICA ---
        scroll_setup = self._build_setup_and_aero_tab()
        tabs_right.addTab(scroll_setup, "Assetto & Aerodinamica")

        # --- TAB 3: GAMEPAD & PHYSICS SETTINGS ---
        tab_ctrl = QWidget()
        ctrl_layout = QVBoxLayout(tab_ctrl)

        # Setup Veicolo & Mappe Cinematiche 14-DOF
        setup_group = QGroupBox("Profilo Setup Veicolo e Cinematiche 14-DOF")
        setup_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        setup_layout = QVBoxLayout(setup_group)

        setup_top_h = QHBoxLayout()
        setup_top_h.addWidget(QLabel("Setup:"))
        self.combo_sim_setup = QComboBox()
        self.combo_sim_setup.setStyleSheet("background-color: #2d3436; color: white; padding: 4px; font-weight: bold;")
        setup_top_h.addWidget(self.combo_sim_setup, 1)
        setup_layout.addLayout(setup_top_h)

        self.lbl_active_kinematics = QLabel()
        self.lbl_active_kinematics.setWordWrap(True)
        self.lbl_active_kinematics.setStyleSheet("""
            QLabel {
                background-color: #1e272e;
                color: #81ecec;
                border: 1px solid #00cec9;
                border-radius: 4px;
                padding: 6px;
                font-size: 10px;
            }
        """)
        setup_layout.addWidget(self.lbl_active_kinematics)

        self.btn_inspect_sim_params = QPushButton("Live-Tuning & Ispezione Telemetria...")
        self.btn_inspect_sim_params.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 6px; border-radius: 4px; font-size: 11px;")
        self.btn_inspect_sim_params.setToolTip("Apre la finestra non-bloccante fluttuante per live-tuning assetto e monitoraggio telemetrico a 25 Hz")
        self.btn_inspect_sim_params.clicked.connect(self.show_simulator_parameters_dialog)
        setup_layout.addWidget(self.btn_inspect_sim_params)

        self.btn_open_3d_studio = QPushButton("Studio Anteprima 3D Veicolo (Orbit 360)...")
        self.btn_open_3d_studio.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 6px; border-radius: 4px; font-size: 11px;")
        self.btn_open_3d_studio.setToolTip("Apre lo studio 3D dedicato con telecamera orbitale a 360°, anteprima FSAE e SUV compatto")
        self.btn_open_3d_studio.clicked.connect(self.open_3d_preview_studio)
        setup_layout.addWidget(self.btn_open_3d_studio)

        self._populate_sim_setups()
        self.combo_sim_setup.currentIndexChanged.connect(self._on_sim_setup_changed)
        self._update_sim_kinematics_label()

        ctrl_layout.addWidget(setup_group)

        ctrl_group = QGroupBox("Personalizzazione Controlli e Rilevamento")
        ctrl_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        cg_layout = QGridLayout(ctrl_group)

        cg_layout.addWidget(QLabel("Periferica:"), 0, 0)
        self.combo_device_mode = QComboBox()
        self.combo_device_mode.addItem("Auto (Rilevamento)", "auto")
        self.combo_device_mode.addItem("Gamepad", "gamepad")
        self.combo_device_mode.addItem("Volante PC", "wheel")
        self.combo_device_mode.setStyleSheet("background-color: #2d3436; color: white; font-weight: bold;")
        self.combo_device_mode.currentIndexChanged.connect(self.on_device_mode_changed)
        cg_layout.addWidget(self.combo_device_mode, 0, 1)

        cg_layout.addWidget(QLabel("Stato Dispositivo:"), 1, 0)
        self.lbl_ctrl_name = QLabel("In scansione...")
        self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #2ecc71; font-size: 11px;")
        cg_layout.addWidget(self.lbl_ctrl_name, 1, 1)

        btn_rescan = QPushButton("Rileva Controller / Volante Adesso")
        btn_rescan.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 4px;")
        btn_rescan.clicked.connect(self.force_rescan_controller)
        cg_layout.addWidget(btn_rescan, 2, 0, 1, 2)

        self.cb_invert_steer = QCheckBox("Inverti Direzione Sterzo SX / DX")
        self.cb_invert_steer.setChecked(False)
        self.cb_invert_steer.toggled.connect(self.update_controller_settings)
        cg_layout.addWidget(self.cb_invert_steer, 3, 0, 1, 2)

        self.cb_speed_steer = QCheckBox("Assistenza Attenuazione Sterzo con Velocità")
        self.cb_speed_steer.setChecked(False)
        self.cb_speed_steer.setToolTip(
            "Se DISABILITATO (consigliato / 1:1 racing): lo sterzo ha autorità completa al 100% a qualsiasi velocità.\n"
            "Se ABILITATO (assist per gamepad arcade): riduce progressivamente l'angolo massimo ad alta velocità."
        )
        self.cb_speed_steer.toggled.connect(self.update_controller_settings)
        cg_layout.addWidget(self.cb_speed_steer, 4, 0, 1, 2)

        self.cb_enable_ffb = QCheckBox("Abilita Force Feedback / Vibrazione (FFB)")
        self.cb_enable_ffb.setChecked(True)
        self.cb_enable_ffb.toggled.connect(self.update_controller_settings)
        cg_layout.addWidget(self.cb_enable_ffb, 5, 0, 1, 2)

        # Vibration Test Button
        vib_test_layout = QHBoxLayout()
        self.btn_test_vibration = QPushButton("⚡ Prova Vibrazione (Chassis + Freno/Gas)")
        self.btn_test_vibration.setStyleSheet("font-size: 10px; font-weight: bold; padding: 4px 6px; background-color: #27ae60; color: white; border-radius: 3px;")
        self.btn_test_vibration.clicked.connect(lambda: self.physics_thread.gamepad.test_vibration())
        self.btn_test_triggers = self.btn_test_vibration
        vib_test_layout.addWidget(self.btn_test_vibration)
        cg_layout.addLayout(vib_test_layout, 6, 0, 1, 2)

        # Force Feedback Presets
        cg_layout.addWidget(QLabel("Preset Force Feedback:"), 7, 0)
        self.combo_ffb_presets = QComboBox()
        for p_name in FFB_PRESETS.keys():
            self.combo_ffb_presets.addItem(p_name)
        self.combo_ffb_presets.addItem("Personalizzato (Manuale)")
        self.combo_ffb_presets.setCurrentIndex(0)
        self.combo_ffb_presets.currentIndexChanged.connect(self.on_ffb_preset_changed)
        cg_layout.addWidget(self.combo_ffb_presets, 7, 1)

        ffb_btn_layout = QHBoxLayout()
        self.btn_save_ffb = QPushButton("💾 Salva Preset...")
        self.btn_save_ffb.setStyleSheet("font-size: 10px; padding: 3px 6px; background-color: #2c3e50; color: white;")
        self.btn_save_ffb.clicked.connect(self.save_ffb_preset)
        self.btn_load_ffb = QPushButton("📂 Carica Preset...")
        self.btn_load_ffb.setStyleSheet("font-size: 10px; padding: 3px 6px; background-color: #2c3e50; color: white;")
        self.btn_load_ffb.clicked.connect(self.load_ffb_preset)
        ffb_btn_layout.addWidget(self.btn_save_ffb)
        ffb_btn_layout.addWidget(self.btn_load_ffb)
        cg_layout.addLayout(ffb_btn_layout, 8, 0, 1, 2)

        cg_layout.addWidget(QLabel("Intensità Force Feedback Globale (%):"), 9, 0)
        self.sb_ffb_gain = QDoubleSpinBox()
        self.sb_ffb_gain.setRange(0.0, 200.0)
        self.sb_ffb_gain.setValue(80.0)
        self.sb_ffb_gain.setSingleStep(10.0)
        self.sb_ffb_gain.setSuffix(" %")
        self.sb_ffb_gain.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_ffb_gain, 9, 1)

        cg_layout.addWidget(QLabel("Feedback di Fondo / Texture Asfalto (%):"), 10, 0)
        self.sb_road_feel = QDoubleSpinBox()
        self.sb_road_feel.setRange(0.0, 100.0)
        self.sb_road_feel.setValue(20.0)
        self.sb_road_feel.setSingleStep(5.0)
        self.sb_road_feel.setSuffix(" %")
        self.sb_road_feel.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_road_feel, 10, 1)

        cg_layout.addWidget(QLabel("Sensibilità Perdita Aderenza (%):"), 11, 0)
        self.sb_grip_loss = QDoubleSpinBox()
        self.sb_grip_loss.setRange(0.0, 200.0)
        self.sb_grip_loss.setValue(75.0)
        self.sb_grip_loss.setSingleStep(5.0)
        self.sb_grip_loss.setSuffix(" %")
        self.sb_grip_loss.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_grip_loss, 11, 1)

        self.cb_enable_triggers = QCheckBox("Abilita Feedback Freno e Trazione (Grilletti / Rumble)")
        self.cb_enable_triggers.setChecked(True)
        self.cb_enable_triggers.setToolTip("Abilita la vibrazione fisica del bloccaggio ruote (freno) al lato sinistro e lo slittamento di trazione (gas) al lato destro del controller (funzionante su tutti i gamepad).")
        self.cb_enable_triggers.toggled.connect(self.update_controller_settings)
        cg_layout.addWidget(self.cb_enable_triggers, 12, 0, 1, 2)

        cg_layout.addWidget(QLabel("Intensità Grilletti LT/RT (%):"), 13, 0)
        self.sb_trigger_gain = QDoubleSpinBox()
        self.sb_trigger_gain.setRange(0.0, 200.0)
        self.sb_trigger_gain.setValue(100.0)
        self.sb_trigger_gain.setSingleStep(10.0)
        self.sb_trigger_gain.setSuffix(" %")
        self.sb_trigger_gain.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_trigger_gain, 13, 1)


        cg_layout.addWidget(QLabel("Zona Morta Sterzo (Deadzone %):"), 14, 0)
        self.sb_deadzone = QDoubleSpinBox()
        self.sb_deadzone.setRange(0.0, 30.0)
        self.sb_deadzone.setValue(4.0)
        self.sb_deadzone.setSingleStep(0.5)
        self.sb_deadzone.setSuffix(" %")
        self.sb_deadzone.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_deadzone, 14, 1)

        cg_layout.addWidget(QLabel("Linearità / Sensibilità Sterzo:"), 15, 0)
        self.sb_sens = QDoubleSpinBox()
        self.sb_sens.setRange(0.5, 3.0)
        self.sb_sens.setValue(1.25)
        self.sb_sens.setSingleStep(0.05)
        self.sb_sens.valueChanged.connect(self.update_controller_settings)
        cg_layout.addWidget(self.sb_sens, 15, 1)

        cg_layout.addWidget(QLabel("Sterzo Effettivo:"), 16, 0)
        self.pb_steer_live = QProgressBar()
        self.pb_steer_live.setRange(-100, 100)
        self.pb_steer_live.setValue(0)
        cg_layout.addWidget(self.pb_steer_live, 16, 1)

        cg_layout.addWidget(QLabel("Acceleratore (RT/W):"), 17, 0)
        self.pb_throt_live = QProgressBar()
        self.pb_throt_live.setRange(0, 100)
        self.pb_throt_live.setValue(0)
        cg_layout.addWidget(self.pb_throt_live, 17, 1)

        cg_layout.addWidget(QLabel("Freno (LT/S):"), 18, 0)
        self.pb_brake_live = QProgressBar()
        self.pb_brake_live.setRange(0, 100)
        self.pb_brake_live.setValue(0)
        cg_layout.addWidget(self.pb_brake_live, 18, 1)

        cg_layout.addWidget(QLabel("Modalità Telecamera:"), 19, 0)
        self.combo_cam = QComboBox()
        self.combo_cam.addItems([
            "3D Volo d'Uccello (Inseguimento)",
            "3D Prima Persona / Cockpit (POV Pilota)",
            "3D Musetto Anteriore (Hood Cam)",
            "3D Bassa Stradale (Bumper Cam)",
            "3D Panoramica Elicottero (Heli Cam)",
            "2D Mappa Pista (Top-Down)"
        ])
        self.combo_cam.currentIndexChanged.connect(self.on_cam_changed)
        cg_layout.addWidget(self.combo_cam, 19, 1)

        # Simulator Track Layout Selector
        cg_layout.addWidget(QLabel("Tracciato Simulatore:"), 20, 0)
        self.combo_sim_track = QComboBox()
        self.combo_sim_track.addItems([
            "FSAE AutoX (850m)",
            "FSAE Skidpad (Figure-8)",
            "FSAE Endurance (1250m)",
            "FSAE Acceleration (200m)",
            "Carica File Tracciato (.csv/.json)...",
            "Disegna Nuovo Tracciato (Editor 2D)..."
        ])
        self.combo_sim_track.currentIndexChanged.connect(self.on_sim_track_changed)
        cg_layout.addWidget(self.combo_sim_track, 20, 1)

        self.btn_draw_track_ui = QPushButton("Disegna / Modifica Tracciato (Spline 2D)")
        self.btn_draw_track_ui.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 6px; border-radius: 3px;")
        self.btn_draw_track_ui.clicked.connect(self.open_track_designer)
        cg_layout.addWidget(self.btn_draw_track_ui, 21, 0, 1, 2)

        ctrl_layout.addWidget(ctrl_group)

        # Dedicated Calculation Frequencies Group (RK4 Physics & Tire Thermal Model)
        freq_group = QGroupBox("Frequenze di Calcolo (Fisica 14-DOF e Termica)")
        freq_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        freq_layout = QGridLayout(freq_group)

        freq_layout.addWidget(QLabel("<b>Frequenza Fisica Veicolo (RK4):</b>"), 0, 0)
        self.combo_hz = QComboBox()
        self.combo_hz.addItem("250 Hz (Efficienza CPU - 4.0 ms)", 250.0)
        self.combo_hz.addItem("500 Hz (Standard - 2.0 ms)", 500.0)
        self.combo_hz.addItem("1000 Hz (Alta Stabilità - 1.0 ms)", 1000.0)
        self.combo_hz.addItem("1500 Hz (Ultra-Precisione - 0.67 ms)", 1500.0)
        self.combo_hz.setCurrentIndex(1)  # Default: 500 Hz
        self.combo_hz.currentIndexChanged.connect(self.on_physics_hz_changed)
        freq_layout.addWidget(self.combo_hz, 0, 1)

        freq_layout.addWidget(QLabel("<b>Frequenza Modello Termico Gomme:</b>"), 1, 0)
        self.combo_thermal_hz = QComboBox()
        self.combo_thermal_hz.addItem("10 Hz (Molto Leggera)", 10.0)
        self.combo_thermal_hz.addItem("25 Hz (Efficiente)", 25.0)
        self.combo_thermal_hz.addItem("50 Hz (Consigliata)", 50.0)
        self.combo_thermal_hz.addItem("100 Hz (Alta Risoluzione)", 100.0)
        self.combo_thermal_hz.addItem("Sincrona con RK4", 0.0)
        self.combo_thermal_hz.setCurrentIndex(2)
        self.combo_thermal_hz.currentIndexChanged.connect(self.on_thermal_hz_changed)
        freq_layout.addWidget(self.combo_thermal_hz, 1, 1)

        freq_layout.addWidget(QLabel("<b>Target Frame Rate (FPS Video):</b>"), 2, 0)
        self.combo_target_fps = QComboBox()
        self.combo_target_fps.addItem("60 FPS (Fisso - Granitico)", 60.0)
        self.combo_target_fps.addItem("120 FPS (Monitor 120Hz)", 120.0)
        self.combo_target_fps.addItem("144 FPS (Monitor 144Hz)", 144.0)
        self.combo_target_fps.addItem("240 FPS (Monitor 240Hz)", 240.0)
        self.combo_target_fps.addItem("Illimitato (Max Hz)", 0.0)
        self.combo_target_fps.setCurrentIndex(0)
        self.combo_target_fps.currentIndexChanged.connect(self.on_target_fps_changed)
        freq_layout.addWidget(self.combo_target_fps, 2, 1)

        self.lbl_freq_status = QLabel("⚡ Fisica: 250 Hz | Video: 60 FPS | Termica: 50 Hz")
        self.lbl_freq_status.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 11px;")
        self.lbl_freq_status.setWordWrap(True)
        freq_layout.addWidget(self.lbl_freq_status, 3, 0, 1, 2)

        ctrl_layout.addWidget(freq_group)

        # Track Grip & Tire Degradation Controls
        env_group = QGroupBox("Condizioni Tracciato e Degrado Gomme")
        env_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        env_layout = QGridLayout(env_group)

        env_layout.addWidget(QLabel("Grip Asfalto Pista:"), 0, 0)
        self.lbl_grip_val = QLabel("100% (Asciutto)")
        self.lbl_grip_val.setStyleSheet("font-weight: bold; color: #2ecc71;")
        env_layout.addWidget(self.lbl_grip_val, 0, 1)

        self.sl_grip = QSlider(Qt.Horizontal)
        self.sl_grip.setRange(10, 150)
        self.sl_grip.setValue(100)
        self.sl_grip.valueChanged.connect(self._on_grip_slider_changed)
        env_layout.addWidget(self.sl_grip, 1, 0, 1, 2)

        preset_layout = QHBoxLayout()
        btn_dry = QPushButton("Asciutto")
        btn_dry.setStyleSheet("padding: 2px 6px; font-size: 10px; background-color: #27ae60; color: white;")
        btn_dry.clicked.connect(lambda: self.sl_grip.setValue(100))
        btn_damp = QPushButton("Umido")
        btn_damp.setStyleSheet("padding: 2px 6px; font-size: 10px; background-color: #2980b9; color: white;")
        btn_damp.clicked.connect(lambda: self.sl_grip.setValue(75))
        btn_wet = QPushButton("Bagnato")
        btn_wet.setStyleSheet("padding: 2px 6px; font-size: 10px; background-color: #8e44ad; color: white;")
        btn_wet.clicked.connect(lambda: self.sl_grip.setValue(50))
        btn_ice = QPushButton("Ghiaccio")
        btn_ice.setStyleSheet("padding: 2px 6px; font-size: 10px; background-color: #7f8c8d; color: white;")
        btn_ice.clicked.connect(lambda: self.sl_grip.setValue(15))
        preset_layout.addWidget(btn_dry)
        preset_layout.addWidget(btn_damp)
        preset_layout.addWidget(btn_wet)
        preset_layout.addWidget(btn_ice)
        env_layout.addLayout(preset_layout, 2, 0, 1, 2)

        env_layout.addWidget(QLabel("Degrado Gomme:"), 3, 0)
        self.combo_tire_deg = QComboBox()
        self.combo_tire_deg.addItems([
            "Disattivato (Off)",
            "Leggero (1-Zona Bulk)",
            "Dettagliato (3-Zone)"
        ])
        self.combo_tire_deg.currentIndexChanged.connect(self._on_tire_deg_changed)
        # Default to Light (1-Zone) mode so thermal & wear degradation are active by default
        try:
            from src.core.simulation_options import get_simulation_options
            _so = get_simulation_options()
            _mode_map = {"off": 0, "light": 1, "detailed": 2}
            init_deg_idx = _mode_map.get(getattr(_so, 'tire_degradation_mode', 'light'), 1)
        except Exception:
            init_deg_idx = 1
        self.combo_tire_deg.setCurrentIndex(init_deg_idx)
        env_layout.addWidget(self.combo_tire_deg, 3, 1)

        # Initial Tire Temperature (T_start)
        env_layout.addWidget(QLabel("Temp. Partenza Gomme:"), 4, 0)
        self.spn_tire_t_start = QDoubleSpinBox()
        self.spn_tire_t_start.setRange(10.0, 95.0)
        self.spn_tire_t_start.setValue(25.0)
        self.spn_tire_t_start.setSingleStep(1.0)
        self.spn_tire_t_start.setSuffix(" °C")
        self.spn_tire_t_start.valueChanged.connect(self._on_tire_t_start_changed)
        env_layout.addWidget(self.spn_tire_t_start, 4, 1)

        t_preset_layout = QHBoxLayout()
        btn_cold = QPushButton("25°C Fredda")
        btn_cold.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #34495e; color: white;")
        btn_cold.clicked.connect(lambda: self.spn_tire_t_start.setValue(25.0))
        btn_warm = QPushButton("45°C Calda")
        btn_warm.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #d35400; color: white;")
        btn_warm.clicked.connect(lambda: self.spn_tire_t_start.setValue(45.0))
        btn_opt = QPushButton("56°C TTC")
        btn_opt.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #c0392b; color: white;")
        btn_opt.clicked.connect(lambda: self.spn_tire_t_start.setValue(56.2))
        t_preset_layout.addWidget(btn_cold)
        t_preset_layout.addWidget(btn_warm)
        t_preset_layout.addWidget(btn_opt)
        env_layout.addLayout(t_preset_layout, 5, 0, 1, 2)

        # Tire Wear Rate Acceleration Factor
        env_layout.addWidget(QLabel("Velocità Usura:"), 6, 0)
        self.combo_wear_scale = QComboBox()
        self.combo_wear_scale.addItems([
            "1x (Reale Endurance)",
            "2x (Medio)",
            "5x (Test Rapido)",
            "10x (Sprint Usura)"
        ])
        self.combo_wear_scale.currentIndexChanged.connect(self._on_wear_scale_changed)
        env_layout.addWidget(self.combo_wear_scale, 6, 1)

        self.cb_standstill_start = QCheckBox("Partenza da Fermo (0 km/h sulla griglia)")
        self.cb_standstill_start.setChecked(True)
        self.cb_standstill_start.setStyleSheet("font-weight: bold; color: #f1c40f;")
        env_layout.addWidget(self.cb_standstill_start, 7, 0, 1, 2)

        ctrl_layout.addWidget(env_group)

        # 3D Road Profile Controls (Optional: Macro Altimetry & Micro Roughness)
        road_group = QGroupBox("Profilo Stradale 3D (Opzionale)")
        road_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a29bfe; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        road_layout = QGridLayout(road_group)

        self.cb_road_macro = QCheckBox("Altimetria e Banking 3D (Macro)")
        self.cb_road_macro.setChecked(False)
        self.cb_road_macro.setStyleSheet("font-weight: bold; color: #a29bfe;")
        self.cb_road_macro.toggled.connect(self._on_road_profile_toggled)
        road_layout.addWidget(self.cb_road_macro, 0, 0, 1, 2)

        self.cb_road_micro = QCheckBox("Micro-Ondulazioni e Rugosità (Micro)")
        self.cb_road_micro.setChecked(False)
        self.cb_road_micro.setStyleSheet("font-weight: bold; color: #fd79a8;")
        self.cb_road_micro.toggled.connect(self._on_road_profile_toggled)
        road_layout.addWidget(self.cb_road_micro, 1, 0, 1, 2)

        road_layout.addWidget(QLabel("Dislivelli (0-4m):"), 2, 0)
        self.sl_road_elev = QSlider(Qt.Horizontal)
        self.sl_road_elev.setRange(0, 40)
        self.sl_road_elev.setValue(15)
        self.sl_road_elev.valueChanged.connect(self._on_road_profile_params_changed)
        road_layout.addWidget(self.sl_road_elev, 2, 1)

        road_layout.addWidget(QLabel("Banking (0-6°):"), 3, 0)
        self.sl_road_bank = QSlider(Qt.Horizontal)
        self.sl_road_bank.setRange(0, 60)
        self.sl_road_bank.setValue(35)
        self.sl_road_bank.valueChanged.connect(self._on_road_profile_params_changed)
        road_layout.addWidget(self.sl_road_bank, 3, 1)

        road_layout.addWidget(QLabel("Rugosità (0-25mm):"), 4, 0)
        self.sl_road_rough = QSlider(Qt.Horizontal)
        self.sl_road_rough.setRange(0, 25)
        self.sl_road_rough.setValue(8)
        self.sl_road_rough.valueChanged.connect(self._on_road_profile_params_changed)
        road_layout.addWidget(self.sl_road_rough, 4, 1)

        self.btn_edit_3d_road = QPushButton("✏ Modifica Profilo 3D (Quote & Banking)...")
        self.btn_edit_3d_road.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 7px; border-radius: 4px; font-size: 12px;")
        self.btn_edit_3d_road.clicked.connect(self.open_road_profile_editor)
        road_layout.addWidget(self.btn_edit_3d_road, 5, 0, 1, 2)

        ctrl_layout.addWidget(road_group)

        # Camera Settings
        cam_group = QGroupBox("Regolazioni Telecamera")
        cam_group.setStyleSheet("QGroupBox { font-weight: bold; color: #f39c12; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        cam_layout = QGridLayout(cam_group)
        
        cam_layout.addWidget(QLabel("FOV:"), 0, 0)
        self.sl_fov = QSlider(Qt.Horizontal)
        self.sl_fov.setRange(300, 800)
        self.sl_fov.setValue(420)
        self.sl_fov.valueChanged.connect(self._update_cam)
        cam_layout.addWidget(self.sl_fov, 0, 1)
        
        cam_layout.addWidget(QLabel("Distanza:"), 1, 0)
        self.sl_dist = QSlider(Qt.Horizontal)
        self.sl_dist.setRange(0, 150)
        self.sl_dist.setValue(68)
        self.sl_dist.valueChanged.connect(self._update_cam)
        cam_layout.addWidget(self.sl_dist, 1, 1)
        
        cam_layout.addWidget(QLabel("Altezza:"), 2, 0)
        self.sl_height = QSlider(Qt.Horizontal)
        self.sl_height.setRange(0, 100)
        self.sl_height.setValue(36)
        self.sl_height.valueChanged.connect(self._update_cam)
        cam_layout.addWidget(self.sl_height, 2, 1)

        cam_layout.addWidget(QLabel("Smoothing:"), 3, 0)
        self.sl_smooth = QSlider(Qt.Horizontal)
        self.sl_smooth.setRange(5, 100)
        self.sl_smooth.setValue(12)
        self.sl_smooth.valueChanged.connect(self._update_cam)
        cam_layout.addWidget(self.sl_smooth, 3, 1)
        
        btn_reset_cam = QPushButton("Reset Camera")
        btn_reset_cam.clicked.connect(self._reset_cam)
        cam_layout.addWidget(btn_reset_cam, 4, 0, 1, 2)
        
        ctrl_layout.addWidget(cam_group)
        ctrl_layout.addStretch()

        scroll_ctrl = QScrollArea()
        scroll_ctrl.setWidgetResizable(True)
        scroll_ctrl.setFrameShape(QFrame.NoFrame)
        scroll_ctrl.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_ctrl.setWidget(tab_ctrl)
        tabs_right.addTab(scroll_ctrl, "Calibrazione Controller e Fisica")

        # Bottom Buttons
        btn_grid = QGridLayout()
        btn_grid.setContentsMargins(0, 4, 0, 0)
        btn_grid.setSpacing(4)

        self.btn_reset = QPushButton("Reset (R)")
        self.btn_reset.setStyleSheet("font-weight: bold; background-color: #e74c3c; color: white; padding: 6px; font-size: 11px;")
        self.btn_reset.clicked.connect(self._handle_user_reset)
        
        self.btn_fullscreen = QPushButton("Schermo Intero (F11)")
        self.btn_fullscreen.setStyleSheet("font-weight: bold; background-color: #f39c12; color: white; padding: 6px; font-size: 11px;")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        self.btn_full_telem = QPushButton("Telemetria Grafici")
        self.btn_full_telem.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 6px; font-size: 11px;")
        self.btn_full_telem.clicked.connect(self.open_full_telemetry_dialog)

        self.btn_editor = QPushButton("Disegna Pista")
        self.btn_editor.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 6px; font-size: 11px;")
        self.btn_editor.clicked.connect(self.open_track_designer)

        self.btn_export = QPushButton("Esporta CSV")
        self.btn_export.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 6px; font-size: 11px;")
        self.btn_export.clicked.connect(self.export_telemetry)

        self.btn_load_telem = QPushButton("Carica Telemetria")
        self.btn_load_telem.setStyleSheet("font-weight: bold; background-color: #2980b9; color: white; padding: 6px; font-size: 11px;")
        self.btn_load_telem.setToolTip("Carica un file telemetria salvato (.csv o .json) e apre l'analisi grafici")
        self.btn_load_telem.clicked.connect(lambda: self.load_telemetry())

        self.btn_export_last = QPushButton("Recupera Stint")
        self.btn_export_last.setStyleSheet("font-weight: bold; background-color: #16a085; color: white; padding: 6px; font-size: 11px;")
        self.btn_export_last.setToolTip("Esporta la telemetria CSV dell'ultimo stint azzerato con il reset")
        self.btn_export_last.clicked.connect(self.export_last_stint_telemetry)

        btn_grid.addWidget(self.btn_reset, 0, 0)
        btn_grid.addWidget(self.btn_fullscreen, 0, 1)
        btn_grid.addWidget(self.btn_full_telem, 1, 0)
        btn_grid.addWidget(self.btn_editor, 1, 1)
        btn_grid.addWidget(self.btn_export, 2, 0)
        btn_grid.addWidget(self.btn_load_telem, 2, 1)
        btn_grid.addWidget(self.btn_export_last, 3, 0, 1, 2)
        right_layout.addLayout(btn_grid)

    def _build_setup_and_aero_tab(self):
        """Builds Tab 2: Live In-Simulator Setup Tuning, Pressures, Aerodynamics and MDO Sync."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Title / Description
        lbl_info = QLabel("<b>Regolazioni Rapide Setup & Aerodinamica</b><br><span style='color:#b2bec3; font-size:10px;'>Modifiche live alla fisica 14-DOF in tempo reale con link a DTS MDO.</span>")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("padding-bottom: 4px;")
        layout.addWidget(lbl_info)

        # 1. Static Alignment Geometry
        geo_group = QGroupBox("Geometria Statica Angoli Ruote")
        geo_group.setStyleSheet("QGroupBox { font-weight: bold; color: #74b9ff; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        geo_layout = QGridLayout(geo_group)

        # Camber Front
        geo_layout.addWidget(QLabel("Camber Anteriore:"), 0, 0)
        self.spn_camber_f = QDoubleSpinBox()
        self.spn_camber_f.setRange(-6.0, 1.0)
        self.spn_camber_f.setSingleStep(0.05)
        self.spn_camber_f.setDecimals(2)
        self.spn_camber_f.setSuffix(" °")
        self.spn_camber_f.setValue(float(getattr(self.vp, 'camber_f_static_deg', getattr(self.vp, 'camber_f_static', -2.5))))
        self.spn_camber_f.valueChanged.connect(self._on_geometry_changed)
        geo_layout.addWidget(self.spn_camber_f, 0, 1)

        # Camber Rear
        geo_layout.addWidget(QLabel("Camber Posteriore:"), 1, 0)
        self.spn_camber_r = QDoubleSpinBox()
        self.spn_camber_r.setRange(-5.0, 1.0)
        self.spn_camber_r.setSingleStep(0.05)
        self.spn_camber_r.setDecimals(2)
        self.spn_camber_r.setSuffix(" °")
        self.spn_camber_r.setValue(float(getattr(self.vp, 'camber_r_static_deg', getattr(self.vp, 'camber_r_static', -1.5))))
        self.spn_camber_r.valueChanged.connect(self._on_geometry_changed)
        geo_layout.addWidget(self.spn_camber_r, 1, 1)

        # Toe Front (deg) - negative is toe-out, positive is toe-in
        geo_layout.addWidget(QLabel("Convergenza Ant. (Toe):"), 2, 0)
        self.spn_toe_f = QDoubleSpinBox()
        self.spn_toe_f.setRange(-3.0, 3.0)
        self.spn_toe_f.setSingleStep(0.05)
        self.spn_toe_f.setDecimals(2)
        self.spn_toe_f.setSuffix(" °")
        self.spn_toe_f.setValue(float(getattr(self.vp, 'toe_f_static_deg', getattr(self.vp, 'toe_f_static', -0.5))))
        self.spn_toe_f.valueChanged.connect(self._on_geometry_changed)
        geo_layout.addWidget(self.spn_toe_f, 2, 1)

        # Toe Rear (deg)
        geo_layout.addWidget(QLabel("Convergenza Post. (Toe):"), 3, 0)
        self.spn_toe_r = QDoubleSpinBox()
        self.spn_toe_r.setRange(-3.0, 3.0)
        self.spn_toe_r.setSingleStep(0.05)
        self.spn_toe_r.setDecimals(2)
        self.spn_toe_r.setSuffix(" °")
        self.spn_toe_r.setValue(float(getattr(self.vp, 'toe_r_static_deg', getattr(self.vp, 'toe_r_static', 0.5))))
        self.spn_toe_r.valueChanged.connect(self._on_geometry_changed)
        geo_layout.addWidget(self.spn_toe_r, 3, 1)

        layout.addWidget(geo_group)

        # 2. Cold Tire Pressures (Pacejka MF6.1)
        press_group = QGroupBox("Pressioni Gomme a Freddo (Pacejka MF6.1)")
        press_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        press_layout = QGridLayout(press_group)

        press_layout.addWidget(QLabel("Pressione Ant. (bar):"), 0, 0)
        self.spn_press_f = QDoubleSpinBox()
        self.spn_press_f.setRange(0.50, 1.80)
        self.spn_press_f.setSingleStep(0.02)
        self.spn_press_f.setDecimals(2)
        self.spn_press_f.setSuffix(" bar")
        init_p_f = float(getattr(self.vp, 'tire_pressure_f_bar', 0.85))
        self.spn_press_f.setValue(init_p_f)
        self.spn_press_f.valueChanged.connect(self._on_pressure_changed)
        press_layout.addWidget(self.spn_press_f, 0, 1)

        self.lbl_press_f_psi = QLabel(f"({init_p_f * 14.5038:.1f} psi)")
        self.lbl_press_f_psi.setStyleSheet("color: #81ecec; font-size: 11px;")
        press_layout.addWidget(self.lbl_press_f_psi, 0, 2)

        press_layout.addWidget(QLabel("Pressione Post. (bar):"), 1, 0)
        self.spn_press_r = QDoubleSpinBox()
        self.spn_press_r.setRange(0.50, 1.80)
        self.spn_press_r.setSingleStep(0.02)
        self.spn_press_r.setDecimals(2)
        self.spn_press_r.setSuffix(" bar")
        init_p_r = float(getattr(self.vp, 'tire_pressure_r_bar', 0.85))
        self.spn_press_r.setValue(init_p_r)
        self.spn_press_r.valueChanged.connect(self._on_pressure_changed)
        press_layout.addWidget(self.spn_press_r, 1, 1)

        self.lbl_press_r_psi = QLabel(f"({init_p_r * 14.5038:.1f} psi)")
        self.lbl_press_r_psi.setStyleSheet("color: #81ecec; font-size: 11px;")
        press_layout.addWidget(self.lbl_press_r_psi, 1, 2)

        press_layout.addWidget(QLabel("K_tz a 12 Psi (N/m):"), 2, 0)
        self.spn_tire_k_z = QDoubleSpinBox()
        self.spn_tire_k_z.setRange(20000.0, 200000.0)
        self.spn_tire_k_z.setSingleStep(1000.0)
        self.spn_tire_k_z.setDecimals(1)
        self.spn_tire_k_z.setSuffix(" N/m")
        init_k_z = float(getattr(self.vp, 'tire_vertical_stiffness_12psi', 75831.6))
        self.spn_tire_k_z.setValue(init_k_z)
        self.spn_tire_k_z.valueChanged.connect(self._on_pressure_changed)
        press_layout.addWidget(self.spn_tire_k_z, 2, 1, 1, 2)

        # Quick Pressure Presets
        p_preset_layout = QHBoxLayout()
        btn_p1 = QPushButton("0.70 bar (10.1 psi)")
        btn_p1.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #2c3e50; color: white;")
        btn_p1.clicked.connect(lambda: self._set_pressures(0.70, 0.70))
        btn_p2 = QPushButton("0.85 bar (12.3 psi)")
        btn_p2.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #16a085; color: white;")
        btn_p2.clicked.connect(lambda: self._set_pressures(0.85, 0.85))
        btn_p3 = QPushButton("1.00 bar (14.5 psi)")
        btn_p3.setStyleSheet("padding: 2px 4px; font-size: 10px; background-color: #2c3e50; color: white;")
        btn_p3.clicked.connect(lambda: self._set_pressures(1.00, 1.00))
        p_preset_layout.addWidget(btn_p1)
        p_preset_layout.addWidget(btn_p2)
        p_preset_layout.addWidget(btn_p3)
        press_layout.addLayout(p_preset_layout, 2, 0, 1, 3)

        layout.addWidget(press_group)

        # 3. Decoupled Suspension Tuning: Spring Rate K, Preload, Pushrod Offset L0 & Ride Height
        preload_group = QGroupBox("Sospensioni: Rigidezza K, Precarico, Puntone L0 & Ride Height")
        preload_group.setStyleSheet("QGroupBox { font-weight: bold; color: #fdcb6e; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        preload_layout = QGridLayout(preload_group)

        # Row 0: Spring Rate K [N/mm]
        preload_layout.addWidget(QLabel("Rigidezza K Ant.:"), 0, 0)
        self.spn_k_f = QDoubleSpinBox()
        self.spn_k_f.setRange(20.0, 150.0)
        self.spn_k_f.setSingleStep(1.0)
        self.spn_k_f.setDecimals(1)
        self.spn_k_f.setSuffix(" N/mm")
        self.spn_k_f.setValue(float(getattr(self.vp, 'spring_rate_installed_f_Nmm', 59.5)))
        self.spn_k_f.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_k_f, 0, 1)

        preload_layout.addWidget(QLabel("Rigidezza K Post.:"), 0, 2)
        self.spn_k_r = QDoubleSpinBox()
        self.spn_k_r.setRange(20.0, 150.0)
        self.spn_k_r.setSingleStep(1.0)
        self.spn_k_r.setDecimals(1)
        self.spn_k_r.setSuffix(" N/mm")
        self.spn_k_r.setValue(float(getattr(self.vp, 'spring_rate_installed_r_Nmm', 61.0)))
        self.spn_k_r.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_k_r, 0, 3)

        # Row 1: Spring Preload P [N]
        preload_layout.addWidget(QLabel("Precarico Ant.:"), 1, 0)
        self.spn_preload_f = QSpinBox()
        self.spn_preload_f.setRange(0, 1500)
        self.spn_preload_f.setSingleStep(10)
        self.spn_preload_f.setSuffix(" N")
        self.spn_preload_f.setValue(int(getattr(self.vp, 'preload_f_N', 200)))
        self.spn_preload_f.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_preload_f, 1, 1)

        preload_layout.addWidget(QLabel("Precarico Post.:"), 1, 2)
        self.spn_preload_r = QSpinBox()
        self.spn_preload_r.setRange(0, 1500)
        self.spn_preload_r.setSingleStep(10)
        self.spn_preload_r.setSuffix(" N")
        self.spn_preload_r.setValue(int(getattr(self.vp, 'preload_r_N', 250)))
        self.spn_preload_r.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_preload_r, 1, 3)

        # Row 2: Pushrod Offset L0 [mm] (2cm total range: +/-10 mm)
        preload_layout.addWidget(QLabel("Puntone L0 Ant.:"), 2, 0)
        self.spn_l0_f = QDoubleSpinBox()
        self.spn_l0_f.setRange(-10.0, 10.0)
        self.spn_l0_f.setSingleStep(0.5)
        self.spn_l0_f.setDecimals(1)
        self.spn_l0_f.setSuffix(" mm")
        self.spn_l0_f.setValue(float(getattr(self.vp, 'pushrod_offset_f_mm', 0.0)))
        self.spn_l0_f.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_l0_f, 2, 1)

        preload_layout.addWidget(QLabel("Puntone L0 Post.:"), 2, 2)
        self.spn_l0_r = QDoubleSpinBox()
        self.spn_l0_r.setRange(-10.0, 10.0)
        self.spn_l0_r.setSingleStep(0.5)
        self.spn_l0_r.setDecimals(1)
        self.spn_l0_r.setSuffix(" mm")
        self.spn_l0_r.setValue(float(getattr(self.vp, 'pushrod_offset_r_mm', 0.0)))
        self.spn_l0_r.valueChanged.connect(self._on_suspension_tuning_changed)
        preload_layout.addWidget(self.spn_l0_r, 2, 3)

        # Row 3: Target Ride Height [mm]
        preload_layout.addWidget(QLabel("Ride Height Ant.:"), 3, 0)
        self.spn_rh_f = QDoubleSpinBox()
        self.spn_rh_f.setRange(20.0, 90.0)
        self.spn_rh_f.setSingleStep(0.5)
        self.spn_rh_f.setDecimals(1)
        self.spn_rh_f.setSuffix(" mm")
        self.spn_rh_f.setValue(float(getattr(self.vp, 'static_ride_height_f_m', 0.050) * 1000.0))
        self.spn_rh_f.valueChanged.connect(self._on_rh_changed)
        preload_layout.addWidget(self.spn_rh_f, 3, 1)

        preload_layout.addWidget(QLabel("Ride Height Post.:"), 3, 2)
        self.spn_rh_r = QDoubleSpinBox()
        self.spn_rh_r.setRange(20.0, 90.0)
        self.spn_rh_r.setSingleStep(0.5)
        self.spn_rh_r.setDecimals(1)
        self.spn_rh_r.setSuffix(" mm")
        self.spn_rh_r.setValue(float(getattr(self.vp, 'static_ride_height_r_m', 0.050) * 1000.0))
        self.spn_rh_r.valueChanged.connect(self._on_rh_changed)
        preload_layout.addWidget(self.spn_rh_r, 3, 3)

        # Row 4: Live Badge
        self.lbl_preload_badge = QLabel()
        self.lbl_preload_badge.setStyleSheet("background-color: #1e272e; border: 1px solid #636e72; border-radius: 4px; padding: 6px; font-size: 10px;")
        preload_layout.addWidget(self.lbl_preload_badge, 4, 0, 1, 4)

        layout.addWidget(preload_group)

        # 4. Anti-Roll Bars (ARB) & Mechanical Roll Balance (from Digitized Curves 05 & 06)
        arb_group = QGroupBox("Barre Antirollio (ARB) & Bilanciamento Meccanico")
        arb_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #00b894; border-radius: 4px; padding-top: 10px; }")
        arb_layout = QGridLayout(arb_group)

        arb_layout.addWidget(QLabel("Dominio di Calcolo:"), 0, 0)
        self.combo_arb_domain = QComboBox()
        self.combo_arb_domain.addItems([
            "Ideale (05: Telaio Rigido)",
            "Reale (06: Torsione Telaio Inclusa)"
        ])
        arb_domain_init = getattr(self.vp, 'arb_diamond_model', 'ideal')
        if 'real' in str(arb_domain_init).lower() or 'reale' in str(arb_domain_init).lower() or '06' in str(arb_domain_init):
            self.combo_arb_domain.setCurrentIndex(1)
        else:
            self.combo_arb_domain.setCurrentIndex(0)
        self.combo_arb_domain.currentIndexChanged.connect(self._on_arb_domain_changed)
        arb_layout.addWidget(self.combo_arb_domain, 0, 1)

        arb_layout.addWidget(QLabel("Preset Taratura Coltelli:"), 1, 0)
        self.combo_arb_preset = QComboBox()
        self.combo_arb_preset.currentIndexChanged.connect(self._on_arb_preset_changed)
        arb_layout.addWidget(self.combo_arb_preset, 1, 1)

        arb_layout.addWidget(QLabel("Rigidezza Rollio k_phi:"), 2, 0)
        self.spn_arb_kphi = QDoubleSpinBox()
        self.spn_arb_kphi.setRange(800.0, 2000.0)
        self.spn_arb_kphi.setSingleStep(25.0)
        self.spn_arb_kphi.setDecimals(1)
        self.spn_arb_kphi.setSuffix(" Nm/deg")
        init_kphi_deg = (getattr(self.vp, 'K_phi_f', 34377.0) + getattr(self.vp, 'K_phi_r', 34377.0)) * (np.pi / 180.0)
        self.spn_arb_kphi.setValue(float(init_kphi_deg))
        self.spn_arb_kphi.valueChanged.connect(self._on_arb_direct_changed)
        arb_layout.addWidget(self.spn_arb_kphi, 2, 1)

        arb_layout.addWidget(QLabel("Ripartizione Anteriore MS_t (LLTD):"), 3, 0)
        self.spn_arb_ms = QDoubleSpinBox()
        self.spn_arb_ms.setRange(30.0, 70.0)
        self.spn_arb_ms.setSingleStep(0.5)
        self.spn_arb_ms.setDecimals(1)
        self.spn_arb_ms.setSuffix(" %")
        init_ms = (getattr(self.vp, 'K_phi_f', 34377.0) / max(getattr(self.vp, 'K_phi_f', 34377.0) + getattr(self.vp, 'K_phi_r', 34377.0), 1.0)) * 100.0
        self.spn_arb_ms.setValue(float(init_ms))
        self.spn_arb_ms.valueChanged.connect(self._on_arb_direct_changed)
        arb_layout.addWidget(self.spn_arb_ms, 3, 1)

        self.lbl_arb_badge = QLabel()
        self.lbl_arb_badge.setStyleSheet("background-color: #1e272e; border: 1px solid #00cec9; border-radius: 4px; padding: 6px; font-size: 10px; color: #81ecec;")
        arb_layout.addWidget(self.lbl_arb_badge, 4, 0, 1, 2)

        layout.addWidget(arb_group)

        # 5. Aerodynamic Settings & Presets
        aero_group = QGroupBox("Configurazione Aerodinamica & Ali")
        aero_group.setStyleSheet("QGroupBox { font-weight: bold; color: #fab1a0; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        aero_layout = QGridLayout(aero_group)

        aero_layout.addWidget(QLabel("Preset Aerodinamico:"), 0, 0)
        self.combo_aero_preset = QComboBox()
        self.combo_aero_preset.addItems([
            "ET18 Baseline (Cz=3.421, Cx=1.200, Bal=50.9%)",
            "Basso Carico / Low Drag (Cz=1.850, Cx=0.720, Bal=46.0%)",
            "Carico Medio / Balanced (Cz=2.600, Cx=0.950, Bal=48.5%)",
            "Personalizzato (Manuale)"
        ])
        self.combo_aero_preset.currentIndexChanged.connect(self._on_aero_preset_changed)
        aero_layout.addWidget(self.combo_aero_preset, 0, 1)

        # Wing Incidence Flap Notches
        aero_layout.addWidget(QLabel("Ala Anteriore (Incidenza):"), 1, 0)
        self.combo_wing_f = QComboBox()
        self.combo_wing_f.addItems([
            "Tacca 1 (-15% carico ant.)",
            "Tacca 2 (-5% carico ant.)",
            "Tacca 3 (Nominale base)",
            "Tacca 4 (+10% carico ant.)"
        ])
        self.combo_wing_f.setCurrentIndex(2)
        self.combo_wing_f.currentIndexChanged.connect(self._on_wing_notch_changed)
        aero_layout.addWidget(self.combo_wing_f, 1, 1)

        aero_layout.addWidget(QLabel("Ala Posteriore (Incidenza):"), 2, 0)
        self.combo_wing_r = QComboBox()
        self.combo_wing_r.addItems([
            "Tacca 1 (-15% carico post.)",
            "Tacca 2 (-5% carico post.)",
            "Tacca 3 (Nominale base)",
            "Tacca 4 (+10% carico post.)"
        ])
        self.combo_wing_r.setCurrentIndex(2)
        self.combo_wing_r.currentIndexChanged.connect(self._on_wing_notch_changed)
        aero_layout.addWidget(self.combo_wing_r, 2, 1)

        # Direct values
        aero_layout.addWidget(QLabel("Cz Totale (Carico):"), 3, 0)
        self.spn_aero_cz = QDoubleSpinBox()
        self.spn_aero_cz.setRange(0.40, 6.00)
        self.spn_aero_cz.setSingleStep(0.01)
        self.spn_aero_cz.setDecimals(3)
        self.spn_aero_cz.setValue(float(getattr(self.vp, 'Cz', 3.421)))
        self.spn_aero_cz.valueChanged.connect(self._on_aero_direct_changed)
        aero_layout.addWidget(self.spn_aero_cz, 3, 1)

        aero_layout.addWidget(QLabel("Cx Totale (Resistenza):"), 4, 0)
        self.spn_aero_cx = QDoubleSpinBox()
        self.spn_aero_cx.setRange(0.20, 3.00)
        self.spn_aero_cx.setSingleStep(0.01)
        self.spn_aero_cx.setDecimals(3)
        self.spn_aero_cx.setValue(float(getattr(self.vp, 'Cd', 1.200)))
        self.spn_aero_cx.valueChanged.connect(self._on_aero_direct_changed)
        aero_layout.addWidget(self.spn_aero_cx, 4, 1)

        aero_layout.addWidget(QLabel("Aero Balance (% Ant.):"), 5, 0)
        self.spn_aero_bal = QDoubleSpinBox()
        self.spn_aero_bal.setRange(30.0, 70.0)
        self.spn_aero_bal.setSingleStep(0.1)
        self.spn_aero_bal.setDecimals(1)
        self.spn_aero_bal.setSuffix(" %")
        self.spn_aero_bal.setValue(float(getattr(self.vp, 'aero_balance', 0.509) * 100.0))
        self.spn_aero_bal.valueChanged.connect(self._on_aero_direct_changed)
        aero_layout.addWidget(self.spn_aero_bal, 5, 1)

        # Technical Aero Badge
        self.lbl_aero_badge = QLabel()
        self.lbl_aero_badge.setStyleSheet("background-color: #1e272e; border: 1px solid #fab1a0; border-radius: 4px; padding: 6px; font-size: 10px; color: #ffeaa7;")
        aero_layout.addWidget(self.lbl_aero_badge, 6, 0, 1, 2)

        layout.addWidget(aero_group)

        # 5. Powertrain RWD Differentials & Active Torque Vectoring
        diff_group = QGroupBox("Differenziale RWD & Active Torque Vectoring")
        diff_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a29bfe; border: 1px solid #6c5ce7; border-radius: 4px; padding-top: 10px; }")
        diff_layout = QGridLayout(diff_group)

        diff_layout.addWidget(QLabel("Modalità Differenziale RWD:"), 0, 0)
        self.combo_diff_mode = QComboBox()
        self.combo_diff_mode.addItems([
            "DUAL_MOTOR_OPEN (Ripartizione 50/50 Libera)",
            "VIRTUAL_SPOOL (Differenziale Bloccato 100%)",
            "VIRTUAL_SALISBURY_LSD (Pacco Frizioni Tiro/Rilascio)",
            "ACTIVE_TORQUE_VECTORING (Controllo Imbardata Closed-Loop)"
        ])
        cur_diff = str(getattr(self.vp, 'diff_mode', 'dual_motor_open')).lower()
        if 'spool' in cur_diff:
            self.combo_diff_mode.setCurrentIndex(1)
        elif 'salisbury' in cur_diff or 'lsd' in cur_diff:
            self.combo_diff_mode.setCurrentIndex(2)
        elif 'torque_vectoring' in cur_diff or 'tv' in cur_diff:
            self.combo_diff_mode.setCurrentIndex(3)
        else:
            self.combo_diff_mode.setCurrentIndex(0)
        self.combo_diff_mode.currentIndexChanged.connect(self._on_diff_mode_changed)
        diff_layout.addWidget(self.combo_diff_mode, 0, 1)

        self.lbl_diff_badge = QLabel()
        self.lbl_diff_badge.setStyleSheet("background-color: #1e272e; border: 1px solid #6c5ce7; border-radius: 4px; padding: 6px; font-size: 10px; color: #dfe6e9;")
        diff_layout.addWidget(self.lbl_diff_badge, 1, 0, 1, 2)

        # Traction Control System (TCS) Controls
        tcs_box = QHBoxLayout()
        self.cb_tcs = QCheckBox("Controllo di Trazione TCS [T]")
        self.cb_tcs.setChecked(True)
        self.cb_tcs.setStyleSheet("font-weight: bold; color: #2ecc71; font-size: 11px;")
        self.cb_tcs.stateChanged.connect(self._on_tcs_checkbox_changed)
        tcs_box.addWidget(self.cb_tcs)

        self.lbl_tcs_status = QLabel("ON (Slip Target 9.5%)")
        self.lbl_tcs_status.setStyleSheet("font-weight: bold; color: #2ecc71; font-size: 10px; background-color: #1e272e; padding: 2px 6px; border-radius: 3px; border: 1px solid #2ecc71;")
        tcs_box.addWidget(self.lbl_tcs_status)
        diff_layout.addLayout(tcs_box, 2, 0, 1, 2)

        layout.addWidget(diff_group)

        # 6. Suspension Shock Absorbers (Öhlins TTX25 4-Way Registers)
        damper_group = QGroupBox("Ammortizzatori Öhlins TTX25 MkII (Registri 4-Way)")
        damper_group.setStyleSheet("QGroupBox { font-weight: bold; color: #fd79a8; border: 1px solid #e84393; border-radius: 4px; padding-top: 10px; }")
        damper_layout = QGridLayout(damper_group)

        damper_layout.addWidget(QLabel("Modello Smorzamento:"), 0, 0)
        self.combo_damper_mode = QComboBox()
        self.combo_damper_mode.addItems([
            "Lineare (Equivalente C_bump / C_reb)",
            "Öhlins TTX25 Non-Lineare (F17 / R17)",
            "Custom Dyno Curve"
        ])
        cur_dmode = str(getattr(self.vp, 'damper_mode', 'linear')).lower()
        if 'ttx25' in cur_dmode:
            self.combo_damper_mode.setCurrentIndex(1)
        elif 'custom' in cur_dmode:
            self.combo_damper_mode.setCurrentIndex(2)
        else:
            self.combo_damper_mode.setCurrentIndex(0)
        self.combo_damper_mode.currentIndexChanged.connect(self._on_damper_mode_changed)
        damper_layout.addWidget(self.combo_damper_mode, 0, 1, 1, 3)

        # Front 4-way registers
        lbl_reg_f = QLabel("<b>Registri Asse Anteriore:</b>")
        lbl_reg_f.setStyleSheet("color: #74b9ff; font-size: 10px;")
        damper_layout.addWidget(lbl_reg_f, 1, 0, 1, 4)

        damper_layout.addWidget(QLabel("LSC (Low Speed Comp):"), 2, 0)
        self.spn_lsc_f = QSpinBox()
        self.spn_lsc_f.setRange(0, 25)
        self.spn_lsc_f.setSuffix(" clicks")
        self.spn_lsc_f.setValue(int(getattr(self.vp, 'damper_lsc_clicks_f', 10)))
        self.spn_lsc_f.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_lsc_f, 2, 1)

        damper_layout.addWidget(QLabel("HSC (High Speed Comp):"), 2, 2)
        self.spn_hsc_f = QDoubleSpinBox()
        self.spn_hsc_f.setRange(0.0, 4.0)
        self.spn_hsc_f.setSingleStep(0.25)
        self.spn_hsc_f.setDecimals(2)
        self.spn_hsc_f.setSuffix(" turns")
        self.spn_hsc_f.setValue(float(getattr(self.vp, 'damper_hsc_turns_f', 2.0)))
        self.spn_hsc_f.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_hsc_f, 2, 3)

        damper_layout.addWidget(QLabel("LSR (Low Speed Reb):"), 3, 0)
        self.spn_lsr_f = QSpinBox()
        self.spn_lsr_f.setRange(0, 25)
        self.spn_lsr_f.setSuffix(" clicks")
        self.spn_lsr_f.setValue(int(getattr(self.vp, 'damper_lsr_clicks_f', 10)))
        self.spn_lsr_f.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_lsr_f, 3, 1)

        damper_layout.addWidget(QLabel("HSR (High Speed Reb):"), 3, 2)
        self.spn_hsr_f = QDoubleSpinBox()
        self.spn_hsr_f.setRange(0.0, 4.0)
        self.spn_hsr_f.setSingleStep(0.25)
        self.spn_hsr_f.setDecimals(2)
        self.spn_hsr_f.setSuffix(" turns")
        self.spn_hsr_f.setValue(float(getattr(self.vp, 'damper_hsr_turns_f', 2.0)))
        self.spn_hsr_f.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_hsr_f, 3, 3)

        # Rear 4-way registers
        lbl_reg_r = QLabel("<b>Registri Asse Posteriore:</b>")
        lbl_reg_r.setStyleSheet("color: #e17055; font-size: 10px;")
        damper_layout.addWidget(lbl_reg_r, 4, 0, 1, 4)

        damper_layout.addWidget(QLabel("LSC (Low Speed Comp):"), 5, 0)
        self.spn_lsc_r = QSpinBox()
        self.spn_lsc_r.setRange(0, 25)
        self.spn_lsc_r.setSuffix(" clicks")
        self.spn_lsc_r.setValue(int(getattr(self.vp, 'damper_lsc_clicks_r', 10)))
        self.spn_lsc_r.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_lsc_r, 5, 1)

        damper_layout.addWidget(QLabel("HSC (High Speed Comp):"), 5, 2)
        self.spn_hsc_r = QDoubleSpinBox()
        self.spn_hsc_r.setRange(0.0, 4.0)
        self.spn_hsc_r.setSingleStep(0.25)
        self.spn_hsc_r.setDecimals(2)
        self.spn_hsc_r.setSuffix(" turns")
        self.spn_hsc_r.setValue(float(getattr(self.vp, 'damper_hsc_turns_r', 2.0)))
        self.spn_hsc_r.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_hsc_r, 5, 3)

        damper_layout.addWidget(QLabel("LSR (Low Speed Reb):"), 6, 0)
        self.spn_lsr_r = QSpinBox()
        self.spn_lsr_r.setRange(0, 25)
        self.spn_lsr_r.setSuffix(" clicks")
        self.spn_lsr_r.setValue(int(getattr(self.vp, 'damper_lsr_clicks_r', 10)))
        self.spn_lsr_r.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_lsr_r, 6, 1)

        damper_layout.addWidget(QLabel("HSR (High Speed Reb):"), 6, 2)
        self.spn_hsr_r = QDoubleSpinBox()
        self.spn_hsr_r.setRange(0.0, 4.0)
        self.spn_hsr_r.setSingleStep(0.25)
        self.spn_hsr_r.setDecimals(2)
        self.spn_hsr_r.setSuffix(" turns")
        self.spn_hsr_r.setValue(float(getattr(self.vp, 'damper_hsr_turns_r', 2.0)))
        self.spn_hsr_r.valueChanged.connect(self._on_damper_registers_changed)
        damper_layout.addWidget(self.spn_hsr_r, 6, 3)

        self.lbl_damper_badge = QLabel()
        self.lbl_damper_badge.setStyleSheet("background-color: #1e272e; border: 1px solid #e84393; border-radius: 4px; padding: 6px; font-size: 10px; color: #ffeaa7;")
        damper_layout.addWidget(self.lbl_damper_badge, 7, 0, 1, 4)
        layout.addWidget(damper_group)

        # 7. Setup Action Buttons & MDO Sync
        act_group = QGroupBox("Gestione Setup & Sincronizzazione DTS MDO")
        act_group.setStyleSheet("QGroupBox { font-weight: bold; color: #55efc4; border: 1px solid #485460; border-radius: 4px; padding-top: 10px; }")
        act_layout = QVBoxLayout(act_group)

        btn_row1 = QHBoxLayout()
        btn_reset_et18 = QPushButton("Ripristina ET18 Baseline")
        btn_reset_et18.setStyleSheet("font-weight: bold; background-color: #d63031; color: white; padding: 5px; font-size: 11px;")
        btn_reset_et18.clicked.connect(self._reset_to_et18_baseline)
        btn_sync_mdo = QPushButton("Sincronizza DTS MDO")
        btn_sync_mdo.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 5px; font-size: 11px;")
        btn_sync_mdo.clicked.connect(self._manual_sync_mdo)
        btn_row1.addWidget(btn_reset_et18)
        btn_row1.addWidget(btn_sync_mdo)
        act_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_save_setup = QPushButton("Salva Setup (.json)...")
        btn_save_setup.setStyleSheet("background-color: #2d3436; color: white; padding: 4px; font-size: 10px;")
        btn_save_setup.clicked.connect(self._save_current_setup_json)
        btn_load_setup = QPushButton("Carica Setup (.json)...")
        btn_load_setup.setStyleSheet("background-color: #2d3436; color: white; padding: 4px; font-size: 10px;")
        btn_load_setup.clicked.connect(self._load_setup_json_dialog)
        btn_row2.addWidget(btn_save_setup)
        btn_row2.addWidget(btn_load_setup)
        act_layout.addLayout(btn_row2)

        layout.addWidget(act_group)
        layout.addStretch()

        self._populate_arb_presets()
        self._update_preload_badge()
        self._update_arb_badge()
        self._update_aero_badge()
        self._update_diff_badge()
        self._update_damper_badge()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(tab_widget)
        return scroll

    def _set_pressures(self, pf, pr):
        self.spn_press_f.setValue(pf)
        self.spn_press_r.setValue(pr)

    def _populate_arb_presets(self):
        if not hasattr(self, 'combo_arb_domain') or not hasattr(self, 'combo_arb_preset'):
            return
        domain_idx = self.combo_arb_domain.currentIndex()
        self.combo_arb_preset.blockSignals(True)
        self.combo_arb_preset.clear()
        if domain_idx == 0:
            # Ideale (05: Telaio Rigido)
            self.combo_arb_preset.addItems([
                "ET18 Baseline Ideale (k_phi=1200 Nm/deg, MS=50.0%)",
                "Blade 1/1: Soft-Soft (k_phi=1000 Nm/deg, MS=49.0%)",
                "Blade 2/1: Stiff Ant. / Sottosterzo (k_phi=1370 Nm/deg, MS=59.5%)",
                "Blade 1/2: Stiff Post. / Sovrasterzo (k_phi=1420 Nm/deg, MS=38.0%)",
                "Blade 2/2: Stiff-Stiff (k_phi=1650 Nm/deg, MS=50.0%)",
                "Personalizzato (Manuale)"
            ])
        else:
            # Reale (06: Torsione Telaio Inclusa)
            self.combo_arb_preset.addItems([
                "ET18 Baseline Reale (k_phi=1180 Nm/deg, MS=48.3%)",
                "Blade 1/1: Soft-Soft (k_phi=1000 Nm/deg, MS=48.4%)",
                "Blade 2/1: Stiff Ant. / Max Front (k_phi=1300 Nm/deg, MS=49.8%)",
                "Blade 1/2: Stiff Post. / Max Rear (k_phi=1350 Nm/deg, MS=46.7%)",
                "Blade 2/2: Stiff-Stiff / Max Rigidezza (k_phi=1600 Nm/deg, MS=48.2%)",
                "Personalizzato (Manuale)"
            ])
        self.combo_arb_preset.blockSignals(False)

    def _on_arb_domain_changed(self, idx):
        self._populate_arb_presets()
        self._on_arb_preset_changed(0)

    def _on_arb_preset_changed(self, idx):
        if not hasattr(self, 'spn_arb_kphi') or not hasattr(self, 'spn_arb_ms'):
            return
        domain_idx = self.combo_arb_domain.currentIndex()
        self.spn_arb_kphi.blockSignals(True)
        self.spn_arb_ms.blockSignals(True)

        if domain_idx == 0:
            # Ideale (05: da 05_arb_blade_setups__kphi_vs_ms.png)
            setups = [
                (1200.0, 50.0),  # ET18 Baseline
                (1000.0, 49.0),  # Soft-Soft
                (1370.0, 59.5),  # Stiff Ant. / Sottosterzo
                (1420.0, 38.0),  # Stiff Post. / Sovrasterzo
                (1650.0, 50.0),  # Stiff-Stiff
            ]
        else:
            # Reale (06: da 06_arb_blade_setups_kphi_vs_ms.png)
            setups = [
                (1180.0, 48.3),  # ET18 Baseline Reale
                (1000.0, 48.4),  # Soft-Soft
                (1300.0, 49.8),  # Stiff Ant. / Max Front Bias
                (1350.0, 46.7),  # Stiff Post. / Max Rear Bias
                (1600.0, 48.2),  # Stiff-Stiff / Max Rigidezza
            ]

        if idx < len(setups):
            kphi, ms = setups[idx]
            self.spn_arb_kphi.setValue(kphi)
            self.spn_arb_ms.setValue(ms)

        self.spn_arb_kphi.blockSignals(False)
        self.spn_arb_ms.blockSignals(False)
        self._apply_arb_changes()

    def _on_arb_direct_changed(self):
        if hasattr(self, 'combo_arb_preset'):
            self.combo_arb_preset.blockSignals(True)
            self.combo_arb_preset.setCurrentIndex(5)  # Personalizzato
            self.combo_arb_preset.blockSignals(False)
        self._apply_arb_changes()

    def _apply_arb_changes(self):
        if not hasattr(self, 'spn_arb_kphi') or not hasattr(self, 'spn_arb_ms'):
            return
        kphi_deg = self.spn_arb_kphi.value()
        ms_pct = self.spn_arb_ms.value()
        K_phi_tot = kphi_deg * (180.0 / np.pi)
        K_phi_f = (ms_pct / 100.0) * K_phi_tot
        K_phi_r = K_phi_tot - K_phi_f
        lltd = ms_pct / 100.0
        arb_mod = "ideal" if (hasattr(self, 'combo_arb_domain') and self.combo_arb_domain.currentIndex() == 0) else "real"

        self.vp.K_phi_f = float(K_phi_f)
        self.vp.K_phi_r = float(K_phi_r)
        self.vp.lltd = float(lltd)
        self.vp.arb_diamond_model = arb_mod

        self._update_arb_badge()
        if hasattr(self, 'physics_thread') and self.physics_thread:
            self.physics_thread.update_live_tuning(
                k_phi_f=K_phi_f, k_phi_r=K_phi_r, lltd=lltd, arb_model=arb_mod
            )
            self.physics_thread.set_vehicle_params(self.vp)
        if hasattr(self, 'viewport'):
            if hasattr(self.viewport, 'set_vehicle_params'):
                self.viewport.set_vehicle_params(self.vp)
            else:
                self.viewport.vp = self.vp
        self.notify_setup_changed()

    def _update_arb_badge(self):
        if not hasattr(self, 'lbl_arb_badge') or not hasattr(self, 'spn_arb_kphi'):
            return
        try:
            kphi_deg = self.spn_arb_kphi.value()
            ms_pct = self.spn_arb_ms.value()
            K_phi_tot = kphi_deg * (180.0 / np.pi)
            K_phi_f = (ms_pct / 100.0) * K_phi_tot
            K_phi_r = K_phi_tot - K_phi_f

            m_tot = float(getattr(self.vp, 'm', 292.0))
            g = getattr(self.vp, 'g', 9.81)
            h_cg = getattr(self.vp, 'h_cg', 0.285)
            h_rc_avg = (getattr(self.vp, 'h_rc_f', 0.040) + getattr(self.vp, 'h_rc_r', 0.050)) / 2.0
            h_arm = max(h_cg - h_rc_avg, 0.05)
            denom = max(K_phi_tot - m_tot * g * h_arm, 1000.0)
            roll_grad_deg_g = (m_tot * g * h_arm / denom) * (180.0 / np.pi)

            dom_name = "Ideale (05)" if (hasattr(self, 'combo_arb_domain') and self.combo_arb_domain.currentIndex() == 0) else "Reale (06)"
            txt = (
                f"<b>Assetto ARB [{dom_name}]:</b> K_phi,tot=<b>{K_phi_tot:.0f} Nm/rad</b> ({kphi_deg:.0f} Nm/deg)<br>"
                f"• Ant: <b>{K_phi_f:.0f} Nm/rad</b> ({ms_pct:.1f}%) | Post: <b>{K_phi_r:.0f} Nm/rad</b> ({(100.0-ms_pct):.1f}%)<br>"
                f"• LLTD Meccanico: <b>{ms_pct:.1f}%</b> | Gradiente Rollio Stimato: <b>{roll_grad_deg_g:.2f}°/g</b>"
            )
            self.lbl_arb_badge.setText(txt)
        except Exception:
            pass

    def _on_diff_mode_changed(self, idx):
        modes = ["dual_motor_open", "virtual_spool", "virtual_salisbury_lsd", "active_torque_vectoring"]
        m = modes[idx] if idx < len(modes) else "dual_motor_open"
        if hasattr(self, 'physics_thread') and self.physics_thread:
            self.physics_thread.update_live_tuning(diff_mode=m)
        self._update_diff_badge()
        self.notify_setup_changed()

    def _update_diff_badge(self):
        if not hasattr(self, 'lbl_diff_badge'):
            return
        idx = self.combo_diff_mode.currentIndex() if hasattr(self, 'combo_diff_mode') else 0
        if idx == 1:
            desc = "<b>VIRTUAL_SPOOL:</b> 100% bloccato | T_lock = K_spool · (ω_RL - ω_RR) | Stabilizzante (sottosterzo)"
        elif idx == 2:
            desc = "<b>VIRTUAL_SALISBURY_LSD:</b> Pacco frizioni con precarico 35 Nm | Rampa tiro 40% / rilascio 20% | Lock dinamico"
        elif idx == 3:
            desc = "<b>ACTIVE_TORQUE_VECTORING:</b> Closed-Loop Yaw PI (Kp=650, Ki=120) | Target da bicicletta lineare saturo a μg/vx"
        else:
            desc = "<b>DUAL_MOTOR_OPEN:</b> Differenziale aperto | Ripartizione coppia simmetrica 50/50 RL/RR"
        self.lbl_diff_badge.setText(desc)

    def _on_tcs_checkbox_changed(self, state):
        enabled = bool(state == Qt.Checked)
        if hasattr(self, 'physics_thread') and self.physics_thread:
            self.physics_thread.set_tcs_enabled(enabled)
        self._update_tcs_status_badge()

    def _update_tcs_status_badge(self):
        enabled = getattr(self.physics_thread, 'tcs_enabled', True) if hasattr(self, 'physics_thread') else True
        if hasattr(self, 'lbl_tcs_status') and hasattr(self, 'cb_tcs'):
            if enabled:
                self.cb_tcs.setStyleSheet("font-weight: bold; color: #2ecc71; font-size: 11px;")
                self.lbl_tcs_status.setText("ON (Slip Target 9.5%)")
                self.lbl_tcs_status.setStyleSheet("font-weight: bold; color: #2ecc71; font-size: 10px; background-color: #1e272e; padding: 2px 6px; border-radius: 3px; border: 1px solid #2ecc71;")
            else:
                self.cb_tcs.setStyleSheet("font-weight: bold; color: #e74c3c; font-size: 11px;")
                self.lbl_tcs_status.setText("OFF (Manuale)")
                self.lbl_tcs_status.setStyleSheet("font-weight: bold; color: #e74c3c; font-size: 10px; background-color: #1e272e; padding: 2px 6px; border-radius: 3px; border: 1px solid #e74c3c;")

    def toggle_tcs(self):
        """Toggles the Traction Control System on/off and synchronizes the UI checkbox."""
        if hasattr(self, 'physics_thread') and self.physics_thread:
            new_state = self.physics_thread.toggle_tcs()
            if hasattr(self, 'cb_tcs'):
                self.cb_tcs.blockSignals(True)
                self.cb_tcs.setChecked(new_state)
                self.cb_tcs.blockSignals(False)
            self._update_tcs_status_badge()

    def _on_damper_mode_changed(self, idx):
        d_modes = ["linear", "ttx25_nonlinear", "custom_dyno"]
        d_m = d_modes[idx] if idx < len(d_modes) else "linear"
        is_ttx = (d_m != "linear")
        for sp in [getattr(self, 'spn_lsc_f', None), getattr(self, 'spn_hsc_f', None),
                   getattr(self, 'spn_lsr_f', None), getattr(self, 'spn_hsr_f', None),
                   getattr(self, 'spn_lsc_r', None), getattr(self, 'spn_hsc_r', None),
                   getattr(self, 'spn_lsr_r', None), getattr(self, 'spn_hsr_r', None)]:
            if sp is not None:
                sp.setEnabled(is_ttx)
        if hasattr(self, 'physics_thread') and self.physics_thread:
            self.physics_thread.update_live_tuning(
                damper_mode=d_m,
                lsc_f=self.spn_lsc_f.value() if hasattr(self, 'spn_lsc_f') else None,
                hsc_f=self.spn_hsc_f.value() if hasattr(self, 'spn_hsc_f') else None,
                lsr_f=self.spn_lsr_f.value() if hasattr(self, 'spn_lsr_f') else None,
                hsr_f=self.spn_hsr_f.value() if hasattr(self, 'spn_hsr_f') else None,
                lsc_r=self.spn_lsc_r.value() if hasattr(self, 'spn_lsc_r') else None,
                hsc_r=self.spn_hsc_r.value() if hasattr(self, 'spn_hsc_r') else None,
                lsr_r=self.spn_lsr_r.value() if hasattr(self, 'spn_lsr_r') else None,
                hsr_r=self.spn_hsr_r.value() if hasattr(self, 'spn_hsr_r') else None
            )
        self._update_damper_badge()
        self.notify_setup_changed()

    def _on_damper_registers_changed(self):
        if hasattr(self, 'physics_thread') and self.physics_thread:
            self.physics_thread.update_live_tuning(
                lsc_f=self.spn_lsc_f.value() if hasattr(self, 'spn_lsc_f') else None,
                hsc_f=self.spn_hsc_f.value() if hasattr(self, 'spn_hsc_f') else None,
                lsr_f=self.spn_lsr_f.value() if hasattr(self, 'spn_lsr_f') else None,
                hsr_f=self.spn_hsr_f.value() if hasattr(self, 'spn_hsr_f') else None,
                lsc_r=self.spn_lsc_r.value() if hasattr(self, 'spn_lsc_r') else None,
                hsc_r=self.spn_hsc_r.value() if hasattr(self, 'spn_hsc_r') else None,
                lsr_r=self.spn_lsr_r.value() if hasattr(self, 'spn_lsr_r') else None,
                hsr_r=self.spn_hsr_r.value() if hasattr(self, 'spn_hsr_r') else None
            )
        self._update_damper_badge()
        self.notify_setup_changed()

    def _update_damper_badge(self):
        if not hasattr(self, 'lbl_damper_badge'):
            return
        idx = self.combo_damper_mode.currentIndex() if hasattr(self, 'combo_damper_mode') else 0
        if idx == 0:
            c_f_bump = getattr(self.vp, 'c_f_bump', 1400.0)
            c_f_reb = getattr(self.vp, 'c_f_rebound', 2200.0)
            c_r_bump = getattr(self.vp, 'c_r_bump', 1500.0)
            c_r_reb = getattr(self.vp, 'c_r_rebound', 2400.0)
            txt = (f"<b>Lineare Costante:</b> Ant: C_bump={c_f_bump:.0f} Ns/m, C_reb={c_f_reb:.0f} Ns/m | "
                   f"Post: C_bump={c_r_bump:.0f} Ns/m, C_reb={c_r_reb:.0f} Ns/m")
        else:
            try:
                from src.models import damper_model_ttx25 as dmp
                m_total = getattr(self.vp, 'm', 300.0)
                wd_f = getattr(self.vp, 'wd_f', 0.50)
                m_f = (m_total * wd_f) / 2.0
                m_r = (m_total * (1.0 - wd_f)) / 2.0
                k_f = getattr(self.vp, 'K_z_f', 45000.0)
                k_r = getattr(self.vp, 'K_z_r', 50000.0)
                mr_f = getattr(self.vp, 'motion_ratio_f', 0.85)
                mr_r = getattr(self.vp, 'motion_ratio_r', 0.85)
                valv_f = 'CUSTOM_DYNO' if idx == 2 else getattr(self.vp, 'damper_valving_f', 'C12/R12')
                valv_r = 'CUSTOM_DYNO' if idx == 2 else getattr(self.vp, 'damper_valving_r', 'C12/R12')

                rf = dmp.compute_modal_damping_ratios(
                    m_f, k_f, valving=valv_f,
                    lsc=self.spn_lsc_f.value(), hsc=self.spn_hsc_f.value(),
                    lsr=self.spn_lsr_f.value(), hsr=self.spn_hsr_f.value(),
                    motion_ratio=mr_f
                )
                rr = dmp.compute_modal_damping_ratios(
                    m_r, k_r, valving=valv_r,
                    lsc=self.spn_lsc_r.value(), hsc=self.spn_hsc_r.value(),
                    lsr=self.spn_lsr_r.value(), hsr=self.spn_hsr_r.value(),
                    motion_ratio=mr_r
                )
                txt = (f"<b>TTX25 4-Way ({'Custom Dyno' if idx == 2 else 'Öhlins Formula'}):</b><br>"
                       f"• Ant: ζ_bump={rf['zeta_bump_ls']:.2f}, ζ_reb={rf['zeta_reb_ls']:.2f}, ζ_avg={rf['zeta_avg_ls']:.2f} (C_eq={rf['c_wheel_avg']:.0f} Ns/m)<br>"
                       f"• Post: ζ_bump={rr['zeta_bump_ls']:.2f}, ζ_reb={rr['zeta_reb_ls']:.2f}, ζ_avg={rr['zeta_avg_ls']:.2f} (C_eq={rr['c_wheel_avg']:.0f} Ns/m)")
            except Exception as e:
                txt = f"<b>TTX25 4-Way Attivo:</b> Registri modificati live ({e})"
        self.lbl_damper_badge.setText(txt)

    def _on_geometry_changed(self):
        cam_f = self.spn_camber_f.value()
        cam_r = self.spn_camber_r.value()
        toe_f = self.spn_toe_f.value()
        toe_r = self.spn_toe_r.value()
        self.vp.camber_f_static = cam_f
        self.vp.camber_f_static_deg = cam_f
        self.vp.camber_r_static = cam_r
        self.vp.camber_r_static_deg = cam_r
        self.vp.toe_f_static = toe_f
        self.vp.toe_f_static_deg = toe_f
        self.vp.toe_r_static = toe_r
        self.vp.toe_r_static_deg = toe_r
        self.physics_thread.update_live_tuning(
            camber_f=cam_f, camber_r=cam_r, toe_f=toe_f, toe_r=toe_r
        )
        self.notify_setup_changed()

    def _on_pressure_changed(self):
        pf = self.spn_press_f.value()
        pr = self.spn_press_r.value()
        kz = self.spn_tire_k_z.value()
        self.lbl_press_f_psi.setText(f"({pf * 14.5038:.1f} psi)")
        self.lbl_press_r_psi.setText(f"({pr * 14.5038:.1f} psi)")
        self.vp.tire_pressure_f_bar = pf
        self.vp.tire_pressure_r_bar = pr
        self.vp.tire_vertical_stiffness_12psi = kz
        self.physics_thread.update_live_tuning(press_f=pf, press_r=pr, k_tire_z=kz)
        self.notify_setup_changed()

    def _on_suspension_tuning_changed(self):
        if getattr(self, '_syncing_preload_rh', False):
            return
        self._syncing_preload_rh = True
        try:
            kf = float(self.spn_k_f.value())
            kr = float(self.spn_k_r.value())
            pf = float(self.spn_preload_f.value())
            pr = float(self.spn_preload_r.value())
            l0_f = float(self.spn_l0_f.value())
            l0_r = float(self.spn_l0_r.value())

            self.vp.spring_rate_installed_f_Nmm = kf
            self.vp.spring_rate_installed_r_Nmm = kr
            MR_f = max(getattr(self.vp, 'motion_ratio_f', 0.8), 0.01)
            MR_r = max(getattr(self.vp, 'motion_ratio_r', 0.8), 0.01)
            self.vp.K_z_f = float(kf * 1000.0 * (MR_f ** 2))
            self.vp.K_z_r = float(kr * 1000.0 * (MR_r ** 2))
            self.vp.preload_f_N = pf
            self.vp.preload_r_N = pr
            self.vp.spring_preload_f_mm = pf / max(kf, 0.1)
            self.vp.spring_preload_r_mm = pr / max(kr, 0.1)
            self.vp.pushrod_offset_f_mm = l0_f
            self.vp.pushrod_offset_r_mm = l0_r
            self.vp.update_suspension_limits()

            # Dynamic Ride Height calculation: nominal height + pushrod L0 offset - delta_sag
            st_f = self.vp.calc_damper_strokes_from_preload('f')
            st_r = self.vp.calc_damper_strokes_from_preload('r')

            P_nom_f = getattr(self.vp, 'preload_f_nom_N', 200.0)
            P_nom_r = getattr(self.vp, 'preload_r_nom_N', 250.0)
            rh_nom = getattr(self.vp, 'nominal_ride_height_m', 0.050) * 1000.0

            m_tot = float(getattr(self.vp, 'm', 292.0))
            g = getattr(self.vp, 'g', 9.81)
            wdf = getattr(self.vp, 'wd_f', 0.485)
            m_sprung = m_tot * (1.0 - 0.12)
            Fz_stat_f = 0.5 * m_sprung * g * wdf
            Fz_stat_r = 0.5 * m_sprung * g * (1.0 - wdf)
            f_spring_static_f = Fz_stat_f / MR_f
            f_spring_static_r = Fz_stat_r / MR_r

            nom_sag_wheel_f = max(0.0, (f_spring_static_f - P_nom_f) / 59.5) / MR_f
            nom_sag_wheel_r = max(0.0, (f_spring_static_r - P_nom_r) / 61.0) / MR_r

            rh_f = rh_nom + l0_f - (st_f['wheel_sag_mm'] - nom_sag_wheel_f)
            rh_r = rh_nom + l0_r - (st_r['wheel_sag_mm'] - nom_sag_wheel_r)

            if hasattr(self, 'spn_rh_f') and hasattr(self, 'spn_rh_r'):
                self.spn_rh_f.blockSignals(True)
                self.spn_rh_r.blockSignals(True)
                self.spn_rh_f.setValue(float(rh_f))
                self.spn_rh_r.setValue(float(rh_r))
                self.spn_rh_f.blockSignals(False)
                self.spn_rh_r.blockSignals(False)

            self.vp.update_static_cg_from_ride_height(rh_f / 1000.0, rh_r / 1000.0)
            self._update_preload_badge()
            self.physics_thread.update_live_tuning(
                preload_f=pf, preload_r=pr, rh_f=rh_f, rh_r=rh_r,
                k_spring_f=kf, k_spring_r=kr, l0_f=l0_f, l0_r=l0_r
            )
            self.notify_setup_changed()
        finally:
            self._syncing_preload_rh = False

    def _on_preload_changed(self):
        """Backward compatibility alias for _on_suspension_tuning_changed."""
        self._on_suspension_tuning_changed()

    def _on_rh_changed(self):
        if getattr(self, '_syncing_preload_rh', False):
            return
        self._syncing_preload_rh = True
        try:
            rh_f_target = float(self.spn_rh_f.value())
            rh_r_target = float(self.spn_rh_r.value())

            MR_f = max(getattr(self.vp, 'motion_ratio_f', 0.8), 0.01)
            MR_r = max(getattr(self.vp, 'motion_ratio_r', 0.8), 0.01)
            P_nom_f = getattr(self.vp, 'preload_f_nom_N', 200.0)
            P_nom_r = getattr(self.vp, 'preload_r_nom_N', 250.0)
            rh_nom = getattr(self.vp, 'nominal_ride_height_m', 0.050) * 1000.0

            m_tot = float(getattr(self.vp, 'm', 292.0))
            g = getattr(self.vp, 'g', 9.81)
            wdf = getattr(self.vp, 'wd_f', 0.485)
            m_sprung = m_tot * (1.0 - 0.12)
            Fz_stat_f = 0.5 * m_sprung * g * wdf
            Fz_stat_r = 0.5 * m_sprung * g * (1.0 - wdf)
            f_spring_static_f = Fz_stat_f / MR_f
            f_spring_static_r = Fz_stat_r / MR_r

            st_f = self.vp.calc_damper_strokes_from_preload('f')
            st_r = self.vp.calc_damper_strokes_from_preload('r')

            nom_sag_wheel_f = max(0.0, (f_spring_static_f - P_nom_f) / 59.5) / MR_f
            nom_sag_wheel_r = max(0.0, (f_spring_static_r - P_nom_r) / 61.0) / MR_r

            # Decoupled: changing Ride Height adjusts pushrod L0 offset (+-10mm range, 2cm total)
            l0_f = float(np.clip(rh_f_target - rh_nom + (st_f['wheel_sag_mm'] - nom_sag_wheel_f), -10.0, 10.0))
            l0_r = float(np.clip(rh_r_target - rh_nom + (st_r['wheel_sag_mm'] - nom_sag_wheel_r), -10.0, 10.0))

            rh_f_actual = rh_nom + l0_f - (st_f['wheel_sag_mm'] - nom_sag_wheel_f)
            rh_r_actual = rh_nom + l0_r - (st_r['wheel_sag_mm'] - nom_sag_wheel_r)

            if hasattr(self, 'spn_l0_f') and hasattr(self, 'spn_l0_r'):
                self.spn_l0_f.blockSignals(True)
                self.spn_l0_r.blockSignals(True)
                self.spn_l0_f.setValue(l0_f)
                self.spn_l0_r.setValue(l0_r)
                self.spn_l0_f.blockSignals(False)
                self.spn_l0_r.blockSignals(False)

            self.spn_rh_f.blockSignals(True)
            self.spn_rh_r.blockSignals(True)
            self.spn_rh_f.setValue(float(rh_f_actual))
            self.spn_rh_r.setValue(float(rh_r_actual))
            self.spn_rh_f.blockSignals(False)
            self.spn_rh_r.blockSignals(False)

            self.vp.pushrod_offset_f_mm = l0_f
            self.vp.pushrod_offset_r_mm = l0_r
            self.vp.update_static_cg_from_ride_height(rh_f_actual / 1000.0, rh_r_actual / 1000.0)
            self._update_preload_badge()
            self.physics_thread.update_live_tuning(
                rh_f=rh_f_actual, rh_r=rh_r_actual, l0_f=l0_f, l0_r=l0_r
            )
            self.notify_setup_changed()
        finally:
            self._syncing_preload_rh = False

    def _update_preload_badge(self):
        try:
            m_tot = float(getattr(self.vp, 'm', 292.0))
            g = getattr(self.vp, 'g', 9.81)
            wdf = getattr(self.vp, 'wd_f', 0.485)
            m_sprung = m_tot * (1.0 - 0.12)
            Fz_stat_f = 0.5 * m_sprung * g * wdf
            Fz_stat_r = 0.5 * m_sprung * g * (1.0 - wdf)

            MR_f = max(getattr(self.vp, 'motion_ratio_f', 0.7938), 0.01)
            MR_r = max(getattr(self.vp, 'motion_ratio_r', 0.7995), 0.01)
            s_tot_mm = getattr(self.vp, 'damper_stroke_total_mm', 57.0)
            z_wheel_tot_f = s_tot_mm / MR_f
            z_wheel_tot_r = s_tot_mm / MR_r

            st_f = self.vp.calc_damper_strokes_from_preload('f')
            st_r = self.vp.calc_damper_strokes_from_preload('r')

            wheel_sag_f = st_f['wheel_sag_mm']
            wheel_sag_r = st_r['wheel_sag_mm']
            bump_free_f = st_f['wheel_bumpstop_gap_mm']
            bump_free_r = st_r['wheel_bumpstop_gap_mm']
            bump_tot_f = st_f['wheel_bump_available_mm']
            bump_tot_r = st_r['wheel_bump_available_mm']

            # FSAE compliance: verifies kinematic travel capability (z_wheel_tot >= 50 mm)
            # Eliminates non-physical warning on static 1g elastic sag
            ok_fsae_f = (z_wheel_tot_f >= 50.0)
            ok_fsae_r = (z_wheel_tot_r >= 50.0)
            warn_f = f" <span style='color:#2ecc71; font-weight:bold;'>✓ [OK FSAE ({z_wheel_tot_f:.1f}mm &ge; 50mm)]</span>" if ok_fsae_f else " <span style='color:#e74c3c; font-weight:bold;'>⚠️ [&lt;50mm FSAE]</span>"
            warn_r = f" <span style='color:#2ecc71; font-weight:bold;'>✓ [OK FSAE ({z_wheel_tot_r:.1f}mm &ge; 50mm)]</span>" if ok_fsae_r else " <span style='color:#e74c3c; font-weight:bold;'>⚠️ [&lt;50mm FSAE]</span>"

            h_cg_val = getattr(self.vp, 'h_cg', 0.280)
            h_cg_nom = getattr(self.vp, 'h_cg_nominal', 0.280)
            l0_f = getattr(self.vp, 'pushrod_offset_f_mm', 0.0)
            l0_r = getattr(self.vp, 'pushrod_offset_r_mm', 0.0)

            txt = (
                f"<b>Assetto Statico 1g &amp; Corsa Sospensioni (FSAE &ge; 50mm tot):</b><br>"
                f"• Ant: Corsa Totale=<b>{z_wheel_tot_f:.1f} mm</b>{warn_f} | "
                f"Sag 1g=<b>{wheel_sag_f:.1f} mm</b> | "
                f"Bump pre-tampone=<b>{bump_free_f:.1f} mm</b> (tot: {bump_tot_f:.1f} mm)<br>"
                f"• Post: Corsa Totale=<b>{z_wheel_tot_r:.1f} mm</b>{warn_r} | "
                f"Sag 1g=<b>{wheel_sag_r:.1f} mm</b> | "
                f"Bump pre-tampone=<b>{bump_free_r:.1f} mm</b> (tot: {bump_tot_r:.1f} mm)<br>"
                f"• <i>Offset L0 puntone:</i> Ant=<b>{l0_f:+.1f} mm</b>, Post=<b>{l0_r:+.1f} mm</b> | "
                f"<i>h_cg statico:</i> <b>{h_cg_val*1000.0:.1f} mm</b> (&Delta;h={(h_cg_val - h_cg_nom)*1000.0:+.1f} mm)"
            )
            self.lbl_preload_badge.setText(txt)
        except Exception:
            pass

    def _update_aero_badge(self):
        try:
            cz = self.spn_aero_cz.value()
            cx = self.spn_aero_cx.value()
            bal = self.spn_aero_bal.value() / 100.0
            eff = cz / max(cx, 1e-3)

            v_60 = 60.0 / 3.6  # 16.667 m/s
            rho = getattr(self.vp, 'rho', 1.225)
            S = getattr(self.vp, 'S', 1.139)
            q_60 = 0.5 * rho * S * (v_60 ** 2)
            F_down_60 = q_60 * cz
            F_drag_60 = q_60 * cx
            kg_down = F_down_60 / 9.81
            kg_drag = F_drag_60 / 9.81

            txt = (
                f"<b>Cz:</b> {cz:.3f} | <b>Cx:</b> {cx:.3f} | <b>L/D (Efficienza):</b> {eff:.2f}<br>"
                f"<b>Ripartizione:</b> {bal*100.0:.1f}% Ant. / {(1.0-bal)*100.0:.1f}% Post.<br>"
                f"<b>Carico @ 60 km/h:</b> {F_down_60:.0f} N (<b>{kg_down:.1f} kg</b>)<br>"
                f"<b>Resistenza @ 60 km/h:</b> {F_drag_60:.0f} N ({kg_drag:.1f} kg)"
            )
            self.lbl_aero_badge.setText(txt)
        except Exception:
            pass

    def _on_aero_preset_changed(self, idx):
        self.spn_aero_cz.blockSignals(True)
        self.spn_aero_cx.blockSignals(True)
        self.spn_aero_bal.blockSignals(True)

        if idx == 0:  # ET18 Baseline
            self.spn_aero_cz.setValue(3.421)
            self.spn_aero_cx.setValue(1.200)
            self.spn_aero_bal.setValue(50.9)
        elif idx == 1:  # Low Drag / Low Downforce
            self.spn_aero_cz.setValue(1.850)
            self.spn_aero_cx.setValue(0.720)
            self.spn_aero_bal.setValue(46.0)
        elif idx == 2:  # Balanced / Medium Downforce
            self.spn_aero_cz.setValue(2.600)
            self.spn_aero_cx.setValue(0.950)
            self.spn_aero_bal.setValue(48.5)
        # idx == 3: Manual / Personalizzato -> keep current

        self.spn_aero_cz.blockSignals(False)
        self.spn_aero_cx.blockSignals(False)
        self.spn_aero_bal.blockSignals(False)

        self._apply_aero_changes()

    def _on_wing_notch_changed(self):
        f_mults = [0.85, 0.95, 1.00, 1.10]
        r_mults = [0.85, 0.95, 1.00, 1.10]

        mf = f_mults[self.combo_wing_f.currentIndex()]
        mr = r_mults[self.combo_wing_r.currentIndex()]

        base_cz_f = 3.421 * 0.509
        base_cz_r = 3.421 * (1.0 - 0.509)
        base_cx = 1.200

        new_cz_f = base_cz_f * mf
        new_cz_r = base_cz_r * mr
        new_cz_tot = new_cz_f + new_cz_r
        new_bal = (new_cz_f / max(new_cz_tot, 1e-4)) * 100.0
        cz_ratio = new_cz_tot / 3.421
        new_cx = base_cx * (0.4 + 0.6 * cz_ratio)

        self.combo_aero_preset.blockSignals(True)
        self.combo_aero_preset.setCurrentIndex(3)  # Personalizzato
        self.combo_aero_preset.blockSignals(False)

        self.spn_aero_cz.blockSignals(True)
        self.spn_aero_cx.blockSignals(True)
        self.spn_aero_bal.blockSignals(True)

        self.spn_aero_cz.setValue(new_cz_tot)
        self.spn_aero_cx.setValue(new_cx)
        self.spn_aero_bal.setValue(new_bal)

        self.spn_aero_cz.blockSignals(False)
        self.spn_aero_cx.blockSignals(False)
        self.spn_aero_bal.blockSignals(False)

        self._apply_aero_changes()

    def _on_aero_direct_changed(self):
        self.combo_aero_preset.blockSignals(True)
        self.combo_aero_preset.setCurrentIndex(3)  # Personalizzato
        self.combo_aero_preset.blockSignals(False)
        self._apply_aero_changes()

    def _apply_aero_changes(self):
        cz = self.spn_aero_cz.value()
        cx = self.spn_aero_cx.value()
        bal = self.spn_aero_bal.value() / 100.0

        self.vp.Cz = cz
        self.vp.Cd = cx
        self.vp.aero_Cz = cz
        self.vp.aero_Cd = cx
        self.vp.aero_balance = bal

        self._update_aero_badge()
        self.physics_thread.update_live_tuning(aero_bal=bal, Cz=cz, Cd=cx)
        self.physics_thread.set_vehicle_params(self.vp)
        if hasattr(self, 'viewport'):
            if hasattr(self.viewport, 'set_vehicle_params'):
                self.viewport.set_vehicle_params(self.vp)
            else:
                self.viewport.vp = self.vp
        self.notify_setup_changed()

    def _reset_to_et18_baseline(self, show_dialog=True):
        self.spn_camber_f.setValue(-2.5)
        self.spn_camber_r.setValue(-1.5)
        self.spn_toe_f.setValue(-0.5)
        self.spn_toe_r.setValue(0.5)
        self.spn_press_f.setValue(0.85)
        self.spn_press_r.setValue(0.85)
        self.spn_preload_f.setValue(200)
        self.spn_preload_r.setValue(250)
        if hasattr(self, 'spn_rh_f'):
            self.spn_rh_f.setValue(50.0)
        if hasattr(self, 'spn_rh_r'):
            self.spn_rh_r.setValue(50.0)
        self.combo_wing_f.setCurrentIndex(2)
        self.combo_wing_r.setCurrentIndex(2)
        self.combo_aero_preset.setCurrentIndex(0)  # triggers ET18 Baseline
        if hasattr(self, 'combo_arb_domain'):
            self.combo_arb_domain.setCurrentIndex(0)
            self._on_arb_preset_changed(0)
        if show_dialog:
            QMessageBox.information(self, "ET18 Baseline Ripristinato", "Tutti i parametri di assetto e aerodinamica sono stati ripristinati al setup ET18 nominale (Aero Balance: 50.9%, Cz=3.421, Cx=1.200, ARB Baseline: 1200 Nm/deg, MS=50%).")

    def _manual_sync_mdo(self):
        self.notify_setup_changed()
        QMessageBox.information(self, "DTS MDO Sincronizzato", f"Setup sincronizzato con DTS MDO:\n- Cz: {self.vp.Cz:.3f} | Cx: {self.vp.Cd:.3f}\n- Aero Balance: {self.vp.aero_balance*100.0:.1f}%\n- Camber F/R: {self.vp.camber_f_static:.2f}° / {self.vp.camber_r_static:.2f}°\n- ARB F/R: {self.vp.K_phi_f:.0f} / {self.vp.K_phi_r:.0f} Nm/rad (LLTD: {self.vp.lltd*100:.1f}%)\n- Pressioni: {getattr(self.vp, 'tire_pressure_f_bar', 0.85):.2f} / {getattr(self.vp, 'tire_pressure_r_bar', 0.85):.2f} bar")

    def _save_current_setup_json(self):
        setups_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setups")
        os.makedirs(setups_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Salva Setup Veicolo JSON", setups_dir, "JSON Setup Files (*.json)")
        if path:
            if not path.endswith('.json'):
                path += '.json'
            try:
                self.vp.save_json(path)
                QMessageBox.information(self, "Setup Salvato", f"Setup veicolo salvato con successo in:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il setup:\n{e}")

    def _load_setup_json_dialog(self):
        setups_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setups")
        path, _ = QFileDialog.getOpenFileName(self, "Carica Setup Veicolo JSON", setups_dir, "JSON Setup Files (*.json);;All Files (*)")
        if path and os.path.exists(path):
            try:
                new_vp = VehicleParams.load_json(path)
                self.vp = new_vp
                self._sync_ui_from_vp()
                self.physics_thread.set_vehicle_params(self.vp)
                if hasattr(self, 'viewport'):
                    if hasattr(self.viewport, 'set_vehicle_params'):
                        self.viewport.set_vehicle_params(self.vp)
                    else:
                        self.viewport.vp = self.vp
                self.notify_setup_changed()
                QMessageBox.information(self, "Setup Caricato", f"Setup caricato con successo da:\n{os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Caricamento", f"Impossibile caricare il file setup:\n{e}")

    def _sync_ui_from_vp(self):
        if not hasattr(self, 'spn_camber_f'):
            return
        self.spn_camber_f.blockSignals(True)
        self.spn_camber_r.blockSignals(True)
        self.spn_toe_f.blockSignals(True)
        self.spn_toe_r.blockSignals(True)
        self.spn_press_f.blockSignals(True)
        self.spn_press_r.blockSignals(True)
        self.spn_preload_f.blockSignals(True)
        self.spn_preload_r.blockSignals(True)
        if hasattr(self, 'spn_rh_f'):
            self.spn_rh_f.blockSignals(True)
            self.spn_rh_r.blockSignals(True)
        self.spn_aero_cz.blockSignals(True)
        self.spn_aero_cx.blockSignals(True)
        self.spn_aero_bal.blockSignals(True)
        if hasattr(self, 'spn_arb_kphi'):
            self.spn_arb_kphi.blockSignals(True)
            self.spn_arb_ms.blockSignals(True)

        self.spn_camber_f.setValue(float(getattr(self.vp, 'camber_f_static_deg', getattr(self.vp, 'camber_f_static', -2.5))))
        self.spn_camber_r.setValue(float(getattr(self.vp, 'camber_r_static_deg', getattr(self.vp, 'camber_r_static', -1.5))))
        self.spn_toe_f.setValue(float(getattr(self.vp, 'toe_f_static_deg', getattr(self.vp, 'toe_f_static', -0.5))))
        self.spn_toe_r.setValue(float(getattr(self.vp, 'toe_r_static_deg', getattr(self.vp, 'toe_r_static', 0.5))))
        pf = float(getattr(self.vp, 'tire_pressure_f_bar', 0.85))
        pr = float(getattr(self.vp, 'tire_pressure_r_bar', 0.85))
        self.spn_press_f.setValue(pf)
        self.spn_press_r.setValue(pr)
        self.lbl_press_f_psi.setText(f"({pf * 14.5038:.1f} psi)")
        self.lbl_press_r_psi.setText(f"({pr * 14.5038:.1f} psi)")
        self.spn_preload_f.setValue(int(getattr(self.vp, 'preload_f_N', 200)))
        self.spn_preload_r.setValue(int(getattr(self.vp, 'preload_r_N', 250)))
        if hasattr(self, 'spn_rh_f'):
            self.spn_rh_f.setValue(float(getattr(self.vp, 'static_ride_height_f_m', 0.050) * 1000.0))
        if hasattr(self, 'spn_rh_r'):
            self.spn_rh_r.setValue(float(getattr(self.vp, 'static_ride_height_r_m', 0.050) * 1000.0))
        cz = float(getattr(self.vp, 'Cz', 3.421))
        cx = float(getattr(self.vp, 'Cd', 1.200))
        bal = float(getattr(self.vp, 'aero_balance', 0.509) * 100.0)
        self.spn_aero_cz.setValue(cz)
        self.spn_aero_cx.setValue(cx)
        self.spn_aero_bal.setValue(bal)

        if hasattr(self, 'spn_arb_kphi') and hasattr(self, 'spn_arb_ms'):
            kphi_tot = (getattr(self.vp, 'K_phi_f', 34377.0) + getattr(self.vp, 'K_phi_r', 34377.0)) * (np.pi / 180.0)
            ms_pct = (getattr(self.vp, 'K_phi_f', 34377.0) / max(getattr(self.vp, 'K_phi_f', 34377.0) + getattr(self.vp, 'K_phi_r', 34377.0), 1.0)) * 100.0
            self.spn_arb_kphi.setValue(kphi_tot)
            self.spn_arb_ms.setValue(ms_pct)
            self.spn_arb_kphi.blockSignals(False)
            self.spn_arb_ms.blockSignals(False)

        self.spn_camber_f.blockSignals(False)
        self.spn_camber_r.blockSignals(False)
        self.spn_toe_f.blockSignals(False)
        self.spn_toe_r.blockSignals(False)
        self.spn_press_f.blockSignals(False)
        self.spn_press_r.blockSignals(False)
        self.spn_preload_f.blockSignals(False)
        self.spn_preload_r.blockSignals(False)
        if hasattr(self, 'spn_rh_f'):
            self.spn_rh_f.blockSignals(False)
            self.spn_rh_r.blockSignals(False)
        self.spn_aero_cz.blockSignals(False)
        self.spn_aero_cx.blockSignals(False)
        self.spn_aero_bal.blockSignals(False)

        self._update_preload_badge()
        self._update_arb_badge()
        self._update_aero_badge()

    def notify_setup_changed(self):
        try:
            self.setup_changed.emit(self.vp)
        except Exception:
            pass

    def toggle_fullscreen(self):
        """Toggles true Game-Only Fullscreen: collapses sidebar and expands 3D driving viewport to 100% of display."""
        if getattr(self, 'is_fullscreen_mode', False):
            self.is_fullscreen_mode = False
            self.right_panel.setVisible(True)
            self.centralWidget().layout().setContentsMargins(6, 6, 6, 6)
            self.showNormal()
            self.btn_fullscreen.setText("Schermo Intero (F11)")
            if hasattr(self.viewport, 'is_fullscreen_view'):
                self.viewport.is_fullscreen_view = False
        else:
            self.is_fullscreen_mode = True
            self.right_panel.setVisible(False)
            self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
            self.showFullScreen()
            self.btn_fullscreen.setText("Esci Fullscreen (F11)")
            if hasattr(self.viewport, 'is_fullscreen_view'):
                self.viewport.is_fullscreen_view = True

    def open_track_designer(self):
        """Opens the 2D interactive track designer and updates the simulator in real time."""
        dlg = TrackDesignerDialog(current_track=self.viewport.track_data, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.generated_track is not None:
            self.viewport.track_data = dlg.generated_track
            self.viewport.invalidate_track_display_list()
            self.physics_thread.set_track_data(dlg.generated_track)
            if 'custom_stations' in dlg.generated_track:
                self.physics_thread.set_custom_road_profile(dlg.generated_track['custom_stations'], enable_macro=True)
                self.cb_road_macro.setChecked(True)
            self._handle_user_reset()
            self.combo_sim_track.blockSignals(True)
            self.combo_sim_track.setCurrentIndex(5)
            self.combo_sim_track.blockSignals(False)

    def open_road_profile_editor(self):
        """Opens the interactive 3D Road Profile Editor dialog."""
        current_stations = getattr(self.physics_thread.road_config, 'custom_stations', None)
        dlg = RoadProfileEditorDialog(
            track_data=self.physics_thread.track_data,
            current_stations=current_stations,
            parent=self
        )
        if dlg.exec_() == QDialog.Accepted:
            stations = dlg.get_stations()
            enable_macro = dlg.cb_enable_macro.isChecked()
            self.apply_custom_road_profile(stations, enable_macro=enable_macro)

    def apply_custom_road_profile(self, stations, enable_macro=True):
        """Applies custom user elevation and banking stations to the running simulator."""
        self.physics_thread.set_custom_road_profile(stations, enable_macro=enable_macro)
        self.cb_road_macro.setChecked(enable_macro)
        if hasattr(self, 'viewport'):
            if hasattr(self.physics_thread, 'track_data'):
                self.viewport.track_data = self.physics_thread.track_data
            if hasattr(self.viewport, 'invalidate_track_display_list'):
                self.viewport.invalidate_track_display_list()
            self.viewport.update()
        QMessageBox.information(
            self, "Profilo 3D Applicato",
            f"Profilo 3D personalizzato applicato con successo al tracciato ({len(stations)} stazioni altimetriche)!"
        )

    def _update_cam(self):
        self.viewport.fov = float(self.sl_fov.value())
        self.viewport.cam_dist = self.sl_dist.value() / 10.0
        self.viewport.cam_height = self.sl_height.value() / 10.0
        
    def _reset_cam(self):
        self.sl_fov.setValue(420)
        self.sl_dist.setValue(68)
        self.sl_height.setValue(36)
        if hasattr(self, 'sl_smooth'):
            self.sl_smooth.setValue(12)
        self.viewport.pitch_offset = 0
        if hasattr(self.viewport, 'reset_camera_smoothing'):
            self.viewport.reset_camera_smoothing()
        self.viewport.update()

    def set_camera_preset(self, dist, height, fov, pitch, mode="CHASE"):
        self.viewport.camera_mode = mode
        if hasattr(self.viewport, 'reset_camera_smoothing'):
            self.viewport.reset_camera_smoothing()
        self.sl_dist.blockSignals(True)
        self.sl_height.blockSignals(True)
        self.sl_fov.blockSignals(True)
        
        self.sl_dist.setValue(int(dist * 10))
        self.sl_height.setValue(int(height * 10))
        self.sl_fov.setValue(int(fov))
        
        self.viewport.cam_dist = dist
        self.viewport.cam_height = height
        self.viewport.fov = fov
        self.viewport.pitch_offset = pitch
        
        self.sl_dist.blockSignals(False)
        self.sl_height.blockSignals(False)
        self.sl_fov.blockSignals(False)
        self.viewport.update()

    def _on_grip_slider_changed(self, val):
        mult = val / 100.0
        self.physics_thread.set_track_grip(mult)
        if mult >= 0.95:
            desc = "Asciutto"
            col = "#2ecc71"
        elif mult >= 0.65:
            desc = "Umido"
            col = "#3498db"
        elif mult >= 0.35:
            desc = "Bagnato"
            col = "#9b59b6"
        else:
            desc = "Ghiaccio"
            col = "#e74c3c"
        self.lbl_grip_val.setText(f"{val}% ({desc})")
        self.lbl_grip_val.setStyleSheet(f"font-weight: bold; color: {col};")

    def _on_tire_deg_changed(self, idx):
        modes = ["off", "light", "detailed"]
        if idx < len(modes):
            self.physics_thread.set_tire_degradation_mode(modes[idx])

    def _on_wear_scale_changed(self, idx):
        scales = [1.0, 2.0, 5.0, 10.0]
        if idx < len(scales):
            self.physics_thread.set_tire_wear_scale(scales[idx])

    def _prompt_save_telemetry_if_needed(self) -> bool:
        """Prompts the user to optionally export CSV telemetry before wiping data. Returns False if cancelled."""
        hist_len = len(self.physics_thread.telemetry_history)
        if hist_len > 100 and not getattr(self, '_skip_reset_prompt', False):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Reset Simulatore - Protezione Telemetria")
            msg_box.setText(
                f"La sessione corrente contiene {hist_len} campioni di telemetria registrati.\n\n"
                "Desideri salvare i dati su file CSV prima di azzerare la pista?"
            )
            msg_box.setIcon(QMessageBox.Question)
            btn_save = msg_box.addButton("Salva ed Azzera", QMessageBox.AcceptRole)
            btn_reset_nosave = msg_box.addButton("Azzera senza Salvare", QMessageBox.DestructiveRole)
            btn_cancel = msg_box.addButton("Annulla", QMessageBox.RejectRole)
            cb_remember = QCheckBox("Non chiedere piu in questa sessione")
            msg_box.setCheckBox(cb_remember)
            msg_box.exec_()

            if cb_remember.isChecked():
                self._skip_reset_prompt = True

            clicked = msg_box.clickedButton()
            if clicked == btn_cancel:
                return False
            elif clicked == btn_save:
                self.export_telemetry()
        return True

    def _handle_user_reset(self):
        if not self._prompt_save_telemetry_if_needed():
            return
        vx_start = 0.0 if getattr(self, 'cb_standstill_start', None) and self.cb_standstill_start.isChecked() else 6.0
        t_start = self.spn_tire_t_start.value() if hasattr(self, 'spn_tire_t_start') else 25.0
        self.physics_thread.reset_state(vx=vx_start, T_start=t_start)
        if hasattr(self, 'viewport') and hasattr(self.viewport, 'reset_camera_smoothing'):
            self.viewport.reset_camera_smoothing()

    def on_sim_track_changed(self, idx):
        if idx == 5:
            self.open_track_designer()
            return

        if not self._prompt_save_telemetry_if_needed():
            return

        try:
            from src.mdo.mdo_direct_collocation import build_track_geometry, load_custom_track_file
            track_keys = ["fsae_autocross", "fsae_skidpad", "fsae_endurance", "fsae_acceleration"]
            if idx < len(track_keys):
                t_type = track_keys[idx]
                new_track = build_track_geometry(track_type=t_type)
                self.track_mesh = TrackCollisionMesh(new_track, road_surface_2d=getattr(self, 'road_surface_2d', None))
            elif idx == 4:
                path, _ = QFileDialog.getOpenFileName(self, "Carica File Tracciato", "tracks", "Track Files (*.csv *.json);;All Files (*)")
                if not path:
                    self.combo_sim_track.blockSignals(True)
                    self.combo_sim_track.setCurrentIndex(0)
                    self.combo_sim_track.blockSignals(False)
                    return
                new_track = load_custom_track_file(path)
            else:
                return

            self.physics_thread.set_track_data(new_track)
            self.viewport.track_data = self.physics_thread.track_data
            if hasattr(self.viewport, 'invalidate_track_display_list'):
                self.viewport.invalidate_track_display_list()
            if hasattr(self.viewport, 'reset_camera_smoothing'):
                self.viewport.reset_camera_smoothing()
            vx_start = 0.0 if getattr(self, 'cb_standstill_start', None) and self.cb_standstill_start.isChecked() else 6.0
            t_start = self.spn_tire_t_start.value() if hasattr(self, 'spn_tire_t_start') else 25.0
            self.physics_thread.reset_state(vx=vx_start, T_start=t_start)
        except Exception as e:
            QMessageBox.warning(self, "Errore Caricamento Tracciato", f"Impossibile caricare il tracciato: {e}")

    def export_last_stint_telemetry(self):
        """Allows exporting the previous stint's telemetry even after reset."""
        try:
            hist = self.physics_thread.get_last_stint_snapshot()
            if not hist:
                QMessageBox.warning(self, "Recupera Stint", "Nessun dato di telemetria archiviato dallo stint precedente.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Stint Precedente", "stint_precedente_14dof.csv", "CSV Files (*.csv)")
            if path:
                df = pd.DataFrame(hist)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Recupera Stint", f"Telemetria stint precedente salvata con successo ({len(df)} campioni x {len(df.columns)} canali):\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Esportazione", f"Impossibile esportare lo stint precedente:\n{e}")

    def on_ffb_preset_changed(self, idx):
        preset_name = self.combo_ffb_presets.currentText()
        if preset_name in FFB_PRESETS:
            cfg = FFB_PRESETS[preset_name]
            self.sb_ffb_gain.blockSignals(True)
            self.sb_road_feel.blockSignals(True)
            self.sb_grip_loss.blockSignals(True)
            self.sb_deadzone.blockSignals(True)
            self.sb_sens.blockSignals(True)
            self.cb_enable_triggers.blockSignals(True)

            self.sb_ffb_gain.setValue(float(cfg.get("ffb_gain", 80.0)))
            self.sb_road_feel.setValue(float(cfg.get("road_feel", 20.0)))
            self.sb_grip_loss.setValue(float(cfg.get("grip_loss", 75.0)))
            self.sb_deadzone.setValue(float(cfg.get("deadzone", 4.0)))
            self.sb_sens.setValue(float(cfg.get("sensitivity", 1.25)))
            self.cb_enable_triggers.setChecked(bool(cfg.get("trigger_haptics", True)))

            self.sb_ffb_gain.blockSignals(False)
            self.sb_road_feel.blockSignals(False)
            self.sb_grip_loss.blockSignals(False)
            self.sb_deadzone.blockSignals(False)
            self.sb_sens.blockSignals(False)
            self.cb_enable_triggers.blockSignals(False)

            self.update_controller_settings(from_preset=True)

    def save_ffb_preset(self):
        cfg = {
            "ffb_gain": self.sb_ffb_gain.value(),
            "road_feel": self.sb_road_feel.value(),
            "grip_loss": self.sb_grip_loss.value(),
            "deadzone": self.sb_deadzone.value(),
            "sensitivity": self.sb_sens.value(),
            "trigger_haptics": self.cb_enable_triggers.isChecked(),
        }
        path, _ = QFileDialog.getSaveFileName(self, "Salva Preset Force Feedback", "ffb_preset.json", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                QMessageBox.information(self, "Preset Salvato", f"Preset salvato con successo in:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Errore Salvataggio", f"Impossibile salvare il preset:\n{e}")

    def load_ffb_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Carica Preset Force Feedback", "", "JSON Files (*.json);;All Files (*)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.sb_ffb_gain.setValue(float(cfg.get("ffb_gain", self.sb_ffb_gain.value())))
                self.sb_road_feel.setValue(float(cfg.get("road_feel", self.sb_road_feel.value())))
                self.sb_grip_loss.setValue(float(cfg.get("grip_loss", self.sb_grip_loss.value())))
                self.sb_deadzone.setValue(float(cfg.get("deadzone", self.sb_deadzone.value())))
                self.sb_sens.setValue(float(cfg.get("sensitivity", self.sb_sens.value())))
                if "trigger_haptics" in cfg:
                    self.cb_enable_triggers.setChecked(bool(cfg["trigger_haptics"]))
                self.combo_ffb_presets.blockSignals(True)
                self.combo_ffb_presets.setCurrentIndex(self.combo_ffb_presets.count() - 1)  # Personalizzato
                self.combo_ffb_presets.blockSignals(False)
                self.update_controller_settings(from_preset=True)
                QMessageBox.information(self, "Preset Caricato", f"Preset caricato con successo da:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Errore Caricamento", f"Impossibile caricare il preset:\n{e}")

    def _get_setups_dir(self):
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "setups")),
            os.path.abspath(os.path.join(os.getcwd(), "setups")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "setups")),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _populate_sim_setups(self):
        self.combo_sim_setup.blockSignals(True)
        self.combo_sim_setup.clear()
        setups_dir = self._get_setups_dir()
        selected_idx = 0
        if os.path.exists(setups_dir):
            json_files = [f for f in os.listdir(setups_dir) if f.endswith('.json')]
            for idx, fname in enumerate(sorted(json_files)):
                fpath = os.path.join(setups_dir, fname)
                self.combo_sim_setup.addItem(fname, fpath)
                # Auto-select et18_2026_kinematics if current vp has 2026 coeffs
                if hasattr(self, 'vp') and hasattr(self.vp, 'camber_gain_f_coeffs') and len(self.vp.camber_gain_f_coeffs) > 2:
                    if abs(self.vp.camber_gain_f_coeffs[2] - (-33.8889)) < 1.0 and "2026" in fname:
                        selected_idx = idx
        if self.combo_sim_setup.count() > 0:
            self.combo_sim_setup.setCurrentIndex(selected_idx)
        self.combo_sim_setup.blockSignals(False)

    def _on_sim_setup_changed(self, index):
        path = self.combo_sim_setup.currentData()
        if path and os.path.exists(path):
            try:
                from src.models.params_vehicle import VehicleParams
                new_vp = VehicleParams.load_json(path)
                self.vp = new_vp
                self.physics_thread.set_vehicle_params(self.vp)
                if hasattr(self, 'viewport'):
                    if hasattr(self.viewport, 'set_vehicle_params'):
                        self.viewport.set_vehicle_params(self.vp)
                    else:
                        self.viewport.vp = self.vp
                self.physics_thread.reset_state(vx=0.0)
                self._update_sim_kinematics_label()
                self._sync_ui_from_vp()
                self.notify_setup_changed()
            except Exception as e:
                print(f"[Sim Setup] Errore caricamento setup: {e}")

    def _update_sim_kinematics_label(self):
        if not hasattr(self, 'lbl_active_kinematics') or not hasattr(self, 'physics_thread'):
            return
        vp = self.physics_thread.vp
        rc_f_mm = vp.rollcenter_f_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_f_coeffs') and len(vp.rollcenter_f_coeffs) > 3 else 0.0
        rc_r_mm = vp.rollcenter_r_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_r_coeffs') and len(vp.rollcenter_r_coeffs) > 3 else 0.0
        rc_lat_f_grad = vp.rollcenter_lat_f_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_f_coeffs') and len(vp.rollcenter_lat_f_coeffs) > 2 else 0.0
        rc_lat_r_grad = vp.rollcenter_lat_r_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_r_coeffs') and len(vp.rollcenter_lat_r_coeffs) > 2 else 0.0
        cg_f_a1 = vp.camber_gain_f_coeffs[2] if hasattr(vp, 'camber_gain_f_coeffs') and len(vp.camber_gain_f_coeffs) > 2 else 0.0
        cg_r_a1 = vp.camber_gain_r_coeffs[2] if hasattr(vp, 'camber_gain_r_coeffs') and len(vp.camber_gain_r_coeffs) > 2 else 0.0
        fname = self.combo_sim_setup.currentText() if hasattr(self, 'combo_sim_setup') else "Attivo"

        self.lbl_active_kinematics.setText(
            f"🟢 <b>Profilo:</b> {fname} (15 Curve Adams Attive)\n"
            f"• <b>Centri Rollio [f(φ)]:</b> Ant={rc_f_mm:.1f} mm ({rc_lat_f_grad:+.1f} mm/°), Post={rc_r_mm:.1f} mm ({rc_lat_r_grad:+.1f} mm/°)\n"
            f"• <b>Campanatura:</b> Stat: Ant={vp.camber_f_static:+.1f}°, Post={vp.camber_r_static:+.1f}° | Gain a1: Ant={cg_f_a1:.1f}°/m, Post={cg_r_a1:.1f}°/m\n"
            f"• <b>Rigidezza Rollio:</b> Kφ,f={vp.K_phi_f:.0f} Nm/rad, Kφ,r={vp.K_phi_r:.0f} Nm/rad | LLTD={vp.lltd*100:.1f}%"
        )

    def show_simulator_parameters_dialog(self):
        """Opens the LiveVehicleParametersInspector floating non-modal window for real-time inspection and live tuning."""
        try:
            from src.ui.vehicle_parameters_inspector_dialog import LiveVehicleParametersInspector
            if not hasattr(self, '_inspector_window') or self._inspector_window is None:
                self._inspector_window = LiveVehicleParametersInspector(
                    physics_thread=self.physics_thread,
                    tp=getattr(self, 'tp', None),
                    title="Live-Tuning Assetto & Ispezione Telemetria Real-Time",
                    parent=None
                )
            else:
                self._inspector_window.set_physics_thread(self.physics_thread)

            self._inspector_window.show()
            self._inspector_window.raise_()
            self._inspector_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Errore Ispezione", f"Impossibile aprire l'ispezione parametri:\n{e}")

    def open_3d_preview_studio(self):
        """Apre lo Studio 3D dedicato con telecamera orbitale a 360° per il veicolo corrente."""
        try:
            v_type = getattr(self.physics_thread.vp, 'vehicle_type', 'formula')
            try:
                from scripts.preview_car_3d import CarPreviewStudioWindow
            except ImportError:
                from preview_car_3d import CarPreviewStudioWindow
            if not hasattr(self, '_preview_studio_win') or self._preview_studio_win is None:
                self._preview_studio_win = CarPreviewStudioWindow(initial_vehicle=v_type)
            else:
                self._preview_studio_win.viewport.vehicle_type = v_type
                self._preview_studio_win.combo_vehicle.setCurrentIndex(1 if v_type == 'b_segment_suv' else 0)
            self._preview_studio_win.show()
            self._preview_studio_win.raise_()
            self._preview_studio_win.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Errore Studio 3D", f"Impossibile avviare lo Studio 3D:\n{e}")

    def on_physics_hz_changed(self, idx):
        hz_val = self.combo_hz.currentData()
        if hz_val:
            self.physics_thread.set_physics_hz(hz_val)
            self._update_freq_status_label()

    def on_thermal_hz_changed(self, idx):
        hz_val = self.combo_thermal_hz.currentData()
        if hz_val is not None:
            self.physics_thread.set_thermal_hz(hz_val)
            self._update_freq_status_label()

    def on_target_fps_changed(self, idx):
        fps_val = self.combo_target_fps.currentData()
        if fps_val is not None:
            self.physics_thread.set_target_fps(fps_val)
            self._update_freq_status_label()

    def _update_freq_status_label(self):
        if hasattr(self, 'lbl_freq_status'):
            phys_hz = self.combo_hz.currentData() if hasattr(self, 'combo_hz') else 250.0
            therm_hz = self.combo_thermal_hz.currentData() if hasattr(self, 'combo_thermal_hz') else 50.0
            fps_target = self.combo_target_fps.currentData() if hasattr(self, 'combo_target_fps') else 60.0
            fps_str = f"{fps_target:.0f} FPS" if fps_target > 0 else "Max Hz"
            therm_str = f"{therm_hz:.0f} Hz (Disaccoppiata)" if therm_hz > 0.0 else "Sincrona con RK4"
            self.lbl_freq_status.setText(f"⚡ Fisica: {phys_hz:.0f} Hz | Video: {fps_str} | Termica: {therm_str}")

    def _on_tire_t_start_changed(self, val):
        self.physics_thread.set_tire_t_start(float(val))

    def _on_road_profile_toggled(self):
        self._on_road_profile_params_changed()

    def _on_road_profile_params_changed(self):
        enable_macro = self.cb_road_macro.isChecked() if hasattr(self, 'cb_road_macro') else False
        enable_micro = self.cb_road_micro.isChecked() if hasattr(self, 'cb_road_micro') else False
        elev_amp = (self.sl_road_elev.value() * 0.1) if hasattr(self, 'sl_road_elev') else 1.5
        bank_deg = (self.sl_road_bank.value() * 0.1) if hasattr(self, 'sl_road_bank') else 3.5
        rough_mm = float(self.sl_road_rough.value()) if hasattr(self, 'sl_road_rough') else 8.0

        self.physics_thread.set_road_profile_config(
            enable_macro=enable_macro,
            enable_micro=enable_micro,
            macro_elevation_amp=elev_amp,
            macro_bank_max_deg=bank_deg,
            micro_roughness_amp_mm=rough_mm
        )
        if hasattr(self, 'viewport') and hasattr(self.physics_thread, 'track_data'):
            self.viewport.track_data = self.physics_thread.track_data
            if hasattr(self.viewport, 'invalidate_track_display_list'):
                self.viewport.invalidate_track_display_list()
            self.viewport.update()

    def open_full_telemetry_dialog(self, initial_preset=None):
        try:
            hist = self.physics_thread.get_telemetry_snapshot()
            if not hist or len(hist) < 5:
                QMessageBox.warning(
                    self,
                    "Telemetria 14-DOF",
                    "Dati di telemetria insufficienti. Guida per qualche secondo prima di aprire i grafici oppure carica una sessione salvata tramite 'Carica Telemetria'."
                )
                return
            dlg = FullTelemetryWindow(hist, self, initial_preset=initial_preset)
            dlg.exec_()
        except Exception as e:
            err_msg = f"Errore durante l'apertura dei grafici di telemetria:\n{e}\n\n{traceback.format_exc()}"
            try:
                with open("crash_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- TELEMETRY CRASH [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n{err_msg}\n")
            except Exception:
                pass
            QMessageBox.critical(self, "Errore Telemetria", err_msg)

    def on_device_mode_changed(self, idx):
        mode = self.combo_device_mode.currentData() if hasattr(self, 'combo_device_mode') else "auto"
        if hasattr(self, 'physics_thread') and hasattr(self.physics_thread, 'gamepad'):
            self.physics_thread.gamepad.set_device_mode(mode)
            self.physics_thread.gamepad.scan_controllers()
            name = self.physics_thread.gamepad.gamepad_name
            is_w = self.physics_thread.gamepad.is_wheel()
            badge = "[Volante PC]" if is_w else "[Connesso]"
            if self.physics_thread.gamepad.connected:
                self.lbl_ctrl_name.setText(f"{badge} {name}")
                self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #2ecc71;")
            else:
                self.lbl_ctrl_name.setText(f"[Tastiera] {name}")
                self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #f1c40f;")

    def force_rescan_controller(self):
        found = self.physics_thread.gamepad.scan_controllers()
        name = self.physics_thread.gamepad.gamepad_name
        is_w = self.physics_thread.gamepad.is_wheel()
        badge = "[Volante PC]" if is_w else "[Connesso]"
        if found:
            self.lbl_ctrl_name.setText(f"{badge} {name}")
            self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #2ecc71;")
            dev_type_str = "Volante PC con Force Feedback" if is_w else "Controller Gamepad"
            QMessageBox.information(self, "Periferica Rilevata", f"{dev_type_str} connesso con successo!\n\nDispositivo: {name}")
        else:
            self.lbl_ctrl_name.setText(f"[Assente] {name} (Nessuna periferica rilevata)")
            self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #f1c40f;")
            QMessageBox.warning(self, "Nessuna Periferica", "Nessun controller o volante rilevato.\n\nAssicurati che la periferica sia collegata via USB o Bluetooth. Puoi comunque guidare con Tastiera (WASD o Frecce).")

    def update_controller_settings(self, from_preset=False):
        gp = self.physics_thread.gamepad
        gp.invert_steer = self.cb_invert_steer.isChecked()
        gp.enable_ffb = self.cb_enable_ffb.isChecked()
        gp.enable_trigger_haptics = self.cb_enable_triggers.isChecked()
        gp.ffb_gain = self.sb_ffb_gain.value() / 100.0
        if hasattr(self, 'sb_trigger_gain'):
            gp.trigger_gain = self.sb_trigger_gain.value() / 100.0
        gp.road_feel_gain = self.sb_road_feel.value() / 100.0
        if hasattr(self, 'sb_grip_loss'):
            gp.grip_loss_gain = self.sb_grip_loss.value() / 100.0
        gp.steer_deadzone = self.sb_deadzone.value() / 100.0
        gp.steer_sensitivity = self.sb_sens.value()
        if hasattr(self, 'cb_speed_steer') and hasattr(self, 'physics_thread'):
            self.physics_thread.speed_sensitive_steer = self.cb_speed_steer.isChecked()

        if not from_preset and hasattr(self, 'combo_ffb_presets'):
            self.combo_ffb_presets.blockSignals(True)
            self.combo_ffb_presets.setCurrentIndex(self.combo_ffb_presets.count() - 1)  # Personalizzato
            self.combo_ffb_presets.blockSignals(False)

    def on_cam_changed(self, idx):
        if idx == 5:
            self.viewport.view_mode = "2D"
        else:
            self.viewport.view_mode = "3D"
            if idx == 0:
                self.viewport.camera_mode = "CHASE"
                self.set_camera_preset(6.8, 3.6, 420, 0, mode="CHASE")
            elif idx == 1:
                self.viewport.camera_mode = "COCKPIT"
                self.set_camera_preset(0.0, 0.56, 720, 0, mode="COCKPIT")
            elif idx == 2:
                self.viewport.camera_mode = "NOSE"
                self.set_camera_preset(0.0, 0.28, 720, 0, mode="NOSE")
            elif idx == 3:
                self.viewport.camera_mode = "BUMPER"
                self.set_camera_preset(0.0, 0.08, 750, 0, mode="BUMPER")
            elif idx == 4:
                self.viewport.camera_mode = "CHASE"
                self.set_camera_preset(14.0, 9.0, 350, -15, mode="CHASE")

    def keyPressEvent(self, event):
        focus_widget = QApplication.focusWidget()
        if focus_widget and (
            isinstance(focus_widget, (QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox))
            or (focus_widget.parent() and isinstance(focus_widget.parent(), (QAbstractSpinBox, QComboBox)))
        ):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            return
        if event.key() == Qt.Key_Escape and getattr(self, 'is_fullscreen_mode', False):
            self.toggle_fullscreen()
            return
            
        if event.key() == Qt.Key_1:
            self.combo_cam.setCurrentIndex(0)
        elif event.key() == Qt.Key_2:
            self.combo_cam.setCurrentIndex(1)
        elif event.key() == Qt.Key_3:
            self.combo_cam.setCurrentIndex(2)
        elif event.key() == Qt.Key_4:
            self.combo_cam.setCurrentIndex(3)
        elif event.key() == Qt.Key_5:
            self.combo_cam.setCurrentIndex(4)
        elif event.key() == Qt.Key_6:
            self.combo_cam.setCurrentIndex(5)
            
        if event.key() == Qt.Key_R:
            self._handle_user_reset()
            return

        if event.key() == Qt.Key_M:
            if hasattr(self, 'viewport'):
                self.viewport.show_minimap = not getattr(self.viewport, 'show_minimap', True)
                self.viewport.update()
            return

        if event.key() == Qt.Key_T:
            self.toggle_tcs()
            return

        self.physics_thread.key_states[event.key()] = True
        if event.key() == Qt.Key_V:
            cur_idx = self.combo_cam.currentIndex()
            next_idx = (cur_idx + 1) % self.combo_cam.count()
            self.combo_cam.setCurrentIndex(next_idx)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        focus_widget = QApplication.focusWidget()
        if focus_widget and (
            isinstance(focus_widget, (QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit, QComboBox))
            or (focus_widget.parent() and isinstance(focus_widget.parent(), (QAbstractSpinBox, QComboBox)))
        ):
            super().keyReleaseEvent(event)
            return
        self.physics_thread.key_states[event.key()] = False
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        """Reset all key states when window loses focus to prevent stuck steering/throttle."""
        self.physics_thread.key_states.clear()
        super().focusOutEvent(event)

    def changeEvent(self, event):
        """Reset key states on window state change (minimize, alt-tab, deactivation)."""
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.physics_thread.key_states.clear()
        super().changeEvent(event)

    def on_physics_frame(self, state):
        self.viewport.update_vehicle_state(state)
        
        # Throttle UI labels, numbers, and status styling to 15 Hz to eliminate GUI thread starvation
        now_time = time.time()
        if now_time - getattr(self, '_last_ui_label_time', 0.0) < 0.066:
            return
        self._last_ui_label_time = now_time

        self.lbl_hz.setText(f"Frequenza Fisica: {state['physics_hz']:.1f} Hz (RK4)")
        self.lbl_step.setText(f"Tempo Step RK4: {state['step_ms']:.2f} ms")
        ratio_str = f"{state['rt_ratio']:.2f}x"
        ratio_fluid = (state['rt_ratio'] >= 0.96)
        if getattr(self, '_last_ratio_fluid', None) != ratio_fluid:
            self._last_ratio_fluid = ratio_fluid
            self.lbl_ratio.setStyleSheet("font-size: 11px; font-weight: bold; color: #2ecc71;" if ratio_fluid else "font-size: 11px; font-weight: bold; color: #e74c3c;")
        self.lbl_ratio.setText(f"Ratio Real-Time: {ratio_str} (" + ("Fluido)" if ratio_fluid else "Rallentamento)"))

        fps_val = getattr(self.viewport, 'fps_smooth', 60.0)
        fps_bucket = 2 if fps_val >= 55.0 else (1 if fps_val >= 30.0 else 0)
        if getattr(self, '_last_fps_bucket', None) != fps_bucket:
            self._last_fps_bucket = fps_bucket
            color_fps = "#2ecc71" if fps_bucket == 2 else ("#f1c40f" if fps_bucket == 1 else "#e74c3c")
            self.lbl_fps.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color_fps};")
        self.lbl_fps.setText(f"Frame Rate: {fps_val:.0f} FPS")

        self.lbl_speed.setText(f"{state['vx_kmh']:.1f} km/h")
        # Lap Timer (sidebar - uses new real lap timer data)
        lap_count = state.get('lap_count', 0)
        lap_current = state.get('lap_time_current', state['lap_time'])
        lap_last = state.get('lap_time_last', 0.0)
        lap_best = state.get('lap_time_best', 0.0)
        mins_c = int(lap_current // 60)
        secs_c = lap_current % 60
        timer_text = f"LAP {lap_count} | {mins_c:02d}:{secs_c:05.2f}"
        if lap_best > 0.1:
            mins_b = int(lap_best // 60)
            secs_b = lap_best % 60
            timer_text += f"  BEST {mins_b:02d}:{secs_b:05.2f}"
        self.lbl_timer.setText(timer_text)
        
        self.lbl_ay.setText(f"Acc. Lat. Ay: {state['ay_g']:+.2f} g")
        self.lbl_ax.setText(f"Acc. Long. Ax: {state['ax_g']:+.2f} g")
        self.lbl_phi.setText(f"Rollio: {state['phi_deg']:+.2f} deg")
        self.lbl_theta.setText(f"Beccheggio: {state['theta_deg']:+.2f} deg")
        
        # Escursioni, Camber, Jacking e Centri di Rollio Live
        self.lbl_z_front.setText(f"Escursione Ant: FL={state['z_fl_mm']:+.1f}mm | FR={state['z_fr_mm']:+.1f}mm")
        self.lbl_z_rear.setText(f"Escursione Post: RL={state['z_rl_mm']:+.1f}mm | RR={state['z_rr_mm']:+.1f}mm")
        self.lbl_cam_live.setText(f"Camber Dinamico: FL={state['camber_fl_deg']:+.2f} deg | FR={state['camber_fr_deg']:+.2f} deg")
        self.lbl_jack_live.setText(f"Jacking Caster/KPI: Delta Fz={state['dFz_jack_N']:+.0f} N")
        if hasattr(self, 'lbl_rc_live'):
            rc_f_str = f"h={state.get('h_rc_f_mm', 0.0):.1f}mm, y={state.get('y_rc_f_mm', 0.0):+.1f}mm"
            rc_r_str = f"h={state.get('h_rc_r_mm', 0.0):.1f}mm, y={state.get('y_rc_r_mm', 0.0):+.1f}mm"
            self.lbl_rc_live.setText(f"Centri Rollio [f(φ)]: Ant({rc_f_str}) | Post({rc_r_str})")
        
        # 4-Channel Haptics & Alert Feedback
        self.pb_haptic_lt.setValue(int(state.get('haptic_brake_lt', 0)))
        self.pb_haptic_rt.setValue(int(state.get('haptic_throt_rt', 0)))
        self.pb_rumble_h.setValue(int(state.get('rumble_heavy', 0)))
        self.pb_rumble_l.setValue(int(state.get('rumble_light', 0)))
        
        u_pct = state.get('understeer_pct', 0)
        o_pct = state.get('oversteer_pct', 0)
        trac_pct = state.get('haptic_throt_rt', 0)
        lock_pct = state.get('haptic_brake_lt', 0)
        b_deg = state.get('beta_deg', 0)
        
        if lock_pct > 25:
            alert_type = 'lock'
            alert_text = f"[FRENO] Bloccaggio Ruote ({lock_pct:.0f}%)"
            alert_css = "background-color: #c0392b; color: white; font-weight: bold; padding: 4px; border-radius: 3px;"
        elif trac_pct > 25:
            alert_type = 'trac'
            alert_text = f"[TRAZIONE] Slittamento Ruote Posteriori ({trac_pct:.0f}%)"
            alert_css = "background-color: #f39c12; color: white; font-weight: bold; padding: 4px; border-radius: 3px;"
        elif u_pct > 15:
            alert_type = 'under'
            alert_text = f"[SOTTOSTERZO] Perdita Aderenza Anteriore ({u_pct:.0f}%)"
            alert_css = "background-color: #e67e22; color: white; font-weight: bold; padding: 4px; border-radius: 3px;"
        elif o_pct > 15 or abs(b_deg) > 5.0:
            alert_type = 'over'
            alert_text = f"[SOVRASTERZO] Coda Instabile ({o_pct:.0f}%, beta={b_deg:+.1f} deg)"
            alert_css = "background-color: #e74c3c; color: white; font-weight: bold; padding: 4px; border-radius: 3px;"
        else:
            alert_type = 'ok'
            alert_text = "[ADERENZA OTTIMALE] Zona Lineare"
            alert_css = "background-color: #27ae60; color: white; font-weight: bold; padding: 4px; border-radius: 3px;"

        self.lbl_ffb_alert.setText(alert_text)
        if getattr(self, '_last_alert_type', None) != alert_type:
            self._last_alert_type = alert_type
            self.lbl_ffb_alert.setStyleSheet(alert_css)
        
        ctrl_conn = state.get('ctrl_connected', False)
        if getattr(self, '_last_ctrl_conn', None) != ctrl_conn:
            self._last_ctrl_conn = ctrl_conn
            if ctrl_conn:
                self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #2ecc71;")
            else:
                self.lbl_ctrl_name.setStyleSheet("font-weight: bold; color: #f1c40f;")
        if ctrl_conn:
            is_w = state.get('is_wheel', False)
            badge = "[Volante PC]" if is_w else "[Connesso]"
            self.lbl_ctrl_name.setText(f"{badge} {state.get('ctrl_device', 'Controller')}")
        else:
            self.lbl_ctrl_name.setText(f"[Tastiera] {state.get('ctrl_device', 'WASD / Frecce')}")

        steer_pct = int(state['steer_deg'] / self.physics_thread.gamepad.max_steer_deg * 100.0)
        self.pb_steer_live.setValue(steer_pct)
        self.pb_throt_live.setValue(int(state['throttle_pct']))
        self.pb_brake_live.setValue(int(state['brake_pct']))

        # Update corner weights & progress bars
        self.pb_fl.setValue(int(state['util_fl']))
        self.pb_fr.setValue(int(state['util_fr']))
        self.pb_rl.setValue(int(state['util_rl']))
        self.pb_rr.setValue(int(state['util_rr']))
        
        self.lbl_fl.setText(f"{state['Fz_fl']:.0f}N ({state['util_fl']:.0f}%)")
        self.lbl_fr.setText(f"{state['Fz_fr']:.0f}N ({state['util_fr']:.0f}%)")
        self.lbl_rl.setText(f"{state['Fz_rl']:.0f}N ({state['util_rl']:.0f}%)")
        self.lbl_rr.setText(f"{state['Fz_rr']:.0f}N ({state['util_rr']:.0f}%)")

        # -------------------------------------------------------------
        # Live Update: Triaxial Contact Patch Forces (Fx, Fy, Fz, |F|)
        # -------------------------------------------------------------
        fx_fl = state.get('Fx_fl', 0.0)
        fx_fr = state.get('Fx_fr', 0.0)
        fx_rl = state.get('Fx_rl', 0.0)
        fx_rr = state.get('Fx_rr', 0.0)

        fy_fl = state.get('Fy_fl', 0.0)
        fy_fr = state.get('Fy_fr', 0.0)
        fy_rl = state.get('Fy_rl', 0.0)
        fy_rr = state.get('Fy_rr', 0.0)

        fz_fl = state.get('Fz_fl', 0.0)
        fz_fr = state.get('Fz_fr', 0.0)
        fz_rl = state.get('Fz_rl', 0.0)
        fz_rr = state.get('Fz_rr', 0.0)

        ft_fl = state.get('F_tot_fl', math.sqrt(fx_fl**2 + fy_fl**2 + fz_fl**2))
        ft_fr = state.get('F_tot_fr', math.sqrt(fx_fr**2 + fy_fr**2 + fz_fr**2))
        ft_rl = state.get('F_tot_rl', math.sqrt(fx_rl**2 + fy_rl**2 + fz_rl**2))
        ft_rr = state.get('F_tot_rr', math.sqrt(fx_rr**2 + fy_rr**2 + fz_rr**2))

        if not hasattr(self, 'peak_loads'):
            self.peak_loads = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.peak_loads['fl'] = max(self.peak_loads['fl'], abs(ft_fl))
        self.peak_loads['fr'] = max(self.peak_loads['fr'], abs(ft_fr))
        self.peak_loads['rl'] = max(self.peak_loads['rl'], abs(ft_rl))
        self.peak_loads['rr'] = max(self.peak_loads['rr'], abs(ft_rr))

        if hasattr(self, 'lbl_fx_fl'):
            self.lbl_fx_fl.setText(f"{fx_fl:+.0f} N")
            self.lbl_fy_fl.setText(f"{fy_fl:+.0f} N")
            self.lbl_fz_fl.setText(f"{fz_fl:.0f} N")
            self.lbl_ftot_fl.setText(f"{ft_fl:.0f} N")

            self.lbl_fx_fr.setText(f"{fx_fr:+.0f} N")
            self.lbl_fy_fr.setText(f"{fy_fr:+.0f} N")
            self.lbl_fz_fr.setText(f"{fz_fr:.0f} N")
            self.lbl_ftot_fr.setText(f"{ft_fr:.0f} N")

            self.lbl_fx_rl.setText(f"{fx_rl:+.0f} N")
            self.lbl_fy_rl.setText(f"{fy_rl:+.0f} N")
            self.lbl_fz_rl.setText(f"{fz_rl:.0f} N")
            self.lbl_ftot_rl.setText(f"{ft_rl:.0f} N")

            self.lbl_fx_rr.setText(f"{fx_rr:+.0f} N")
            self.lbl_fy_rr.setText(f"{fy_rr:+.0f} N")
            self.lbl_fz_rr.setText(f"{fz_rr:.0f} N")
            self.lbl_ftot_rr.setText(f"{ft_rr:.0f} N")

        if hasattr(self, 'combo_loads_view') and hasattr(self, 'lbl_single_fx'):
            c_idx = self.combo_loads_view.currentIndex()
            if c_idx > 0:
                c_map = {1: ('fl', 'Anteriore SX (FL)'),
                         2: ('fr', 'Anteriore DX (FR)'),
                         3: ('rl', 'Posteriore SX (RL)'),
                         4: ('rr', 'Posteriore DX (RR)')}
                c_key, c_title = c_map.get(c_idx, ('fl', 'Anteriore SX (FL)'))
                c_fx = state.get(f'Fx_{c_key}', 0.0)
                c_fy = state.get(f'Fy_{c_key}', 0.0)
                c_fz = state.get(f'Fz_{c_key}', 0.0)
                c_ft = state.get(f'F_tot_{c_key}', math.sqrt(c_fx**2 + c_fy**2 + c_fz**2))
                self.lbl_single_wheel_title.setText(f"<b>Dettaglio Carichi: {c_title}</b>")
                self.lbl_single_fx.setText(f"Fx (Longitudinale Spinta/Freno): <b>{c_fx:+.0f} N</b>")
                self.lbl_single_fy.setText(f"Fy (Laterale Aderenza Curva): <b>{c_fy:+.0f} N</b>")
                self.lbl_single_fz.setText(f"Fz (Carico Verticale Normale): <b>{c_fz:.0f} N</b>")
                self.lbl_single_ftot.setText(f"|F| (Risultante 3D Vettoriale): <b>{c_ft:.0f} N</b>")
                self.lbl_single_fpeak.setText(f"|F|_max (Picco Sessione): <b>{self.peak_loads[c_key]:.0f} N</b>")

        # -------------------------------------------------------------
        # Live Update: Per-Wheel Drive & Brake Torques [Nm]
        # -------------------------------------------------------------
        if hasattr(self, 'lbl_tbrk_fl'):
            td_fl = state.get('T_drive_fl', 0.0)
            td_fr = state.get('T_drive_fr', 0.0)
            td_rl = state.get('T_drive_rl', 0.0)
            td_rr = state.get('T_drive_rr', 0.0)

            tb_fl = state.get('T_brake_fl', 0.0)
            tb_fr = state.get('T_brake_fr', 0.0)
            tb_rl = state.get('T_brake_rl', 0.0)
            tb_rr = state.get('T_brake_rr', 0.0)

            tn_fl = state.get('T_net_fl', -tb_fl)
            tn_fr = state.get('T_net_fr', -tb_fr)
            tn_rl = state.get('T_net_rl', td_rl - tb_rl)
            tn_rr = state.get('T_net_rr', td_rr - tb_rr)

            self.lbl_tbrk_fl.setText(f"{tb_fl:.1f} Nm")
            self.lbl_tnet_fl.setText(f"{tn_fl:+.1f} Nm")

            self.lbl_tbrk_fr.setText(f"{tb_fr:.1f} Nm")
            self.lbl_tnet_fr.setText(f"{tn_fr:+.1f} Nm")

            self.lbl_tdrv_rl.setText(f"{td_rl:.1f} Nm")
            self.lbl_tbrk_rl.setText(f"{tb_rl:.1f} Nm")
            self.lbl_tnet_rl.setText(f"{tn_rl:+.1f} Nm")

            self.lbl_tdrv_rr.setText(f"{td_rr:.1f} Nm")
            self.lbl_tbrk_rr.setText(f"{tb_rr:.1f} Nm")
            self.lbl_tnet_rr.setText(f"{tn_rr:+.1f} Nm")

            td_tot = td_rl + td_rr
            tb_tot = tb_fl + tb_fr + tb_rl + tb_rr
            bb_pct = float(getattr(self.vp, 'brake_bias', 0.60)) * 100.0
            self.lbl_torque_summary.setText(f"Trazione Tot: <b>{td_tot:.1f} Nm</b> (50/50) | Freno Tot: <b>{tb_tot:.1f} Nm</b> (Bias {bb_pct:.0f}/{(100.0 - bb_pct):.0f}%)")

        # Live Update: Powertrain Elettrico EMRAX 188 & Batteria 80 kW
        if hasattr(self, 'lbl_pt_pwr_total'):
            p_tot_elec = state.get('p_elec_total_kw', 0.0)
            p_l_elec = state.get('p_elec_rl_kw', 0.0)
            p_r_elec = state.get('p_elec_rr_kw', 0.0)
            p_l_mech = state.get('p_mech_rl_kw', 0.0)
            p_r_mech = state.get('p_mech_rr_kw', 0.0)
            rpm_l = state.get('rpm_motor_rl', 0.0)
            rpm_r = state.get('rpm_motor_rr', 0.0)
            tq_l = state.get('t_motor_rl_nm', 0.0)
            tq_r = state.get('t_motor_rr_nm', 0.0)
            eta_l = state.get('eta_motor_rl', 0.95) * 100.0
            eta_r = state.get('eta_motor_rr', 0.95) * 100.0
            is_lim = state.get('is_power_limited', False)

            self.lbl_pt_pwr_total.setText(f"Potenza Elettrica Accumulatore: <b>{p_tot_elec:.1f} / 80.0 kW</b>")
            self.pb_pt_pwr_total.setValue(min(int(p_tot_elec * 10.0), 800))
            if p_tot_elec >= 79.5 or is_lim:
                self.pb_pt_pwr_total.setStyleSheet("QProgressBar { height: 12px; background-color: #2d3436; border-radius: 3px; } QProgressBar::chunk { background-color: #e74c3c; }")
                self.lbl_pt_pwr_status.setText("[LIMIT] Taglio Potenza 80kW")
                self.lbl_pt_pwr_status.setStyleSheet("font-size: 10px; font-weight: bold; color: white; background-color: #e74c3c; padding: 2px 6px; border-radius: 3px;")
            elif p_tot_elec >= 60.0:
                self.pb_pt_pwr_total.setStyleSheet("QProgressBar { height: 12px; background-color: #2d3436; border-radius: 3px; } QProgressBar::chunk { background-color: #f39c12; }")
                self.lbl_pt_pwr_status.setText("[ELEVATA] Potenza >60kW")
                self.lbl_pt_pwr_status.setStyleSheet("font-size: 10px; font-weight: bold; color: white; background-color: #f39c12; padding: 2px 6px; border-radius: 3px;")
            else:
                self.pb_pt_pwr_total.setStyleSheet("QProgressBar { height: 12px; background-color: #2d3436; border-radius: 3px; } QProgressBar::chunk { background-color: #00cec9; }")
                self.lbl_pt_pwr_status.setText("[OK] Sotto Limite 80kW")
                self.lbl_pt_pwr_status.setStyleSheet("font-size: 10px; font-weight: bold; color: #2ecc71; background-color: #2d3436; padding: 2px 6px; border-radius: 3px;")

            self.lbl_pt_rpm_l.setText(f"{rpm_l:.0f} RPM")
            self.lbl_pt_tq_l.setText(f"{tq_l:.1f} Nm")
            self.lbl_pt_pw_l.setText(f"{p_l_elec:.1f} / {p_l_mech:.1f} kW")
            self.lbl_pt_eta_l.setText(f"{eta_l:.1f} %")

            self.lbl_pt_rpm_r.setText(f"{rpm_r:.0f} RPM")
            self.lbl_pt_tq_r.setText(f"{tq_r:.1f} Nm")
            self.lbl_pt_pw_r.setText(f"{p_r_elec:.1f} / {p_r_mech:.1f} kW")
            self.lbl_pt_eta_r.setText(f"{eta_r:.1f} %")

            delta_p = p_r_elec - p_l_elec
            self.lbl_pt_tv_delta.setText(f"Torque Vectoring: ΔP={delta_p:+.1f} kW | Riduzione 4:1 (Coppia Ruota = 4 × Motore)")

    def _on_loads_view_changed(self, index):
        if hasattr(self, 'widget_loads_table') and hasattr(self, 'widget_loads_single'):
            self.widget_loads_table.setVisible(index == 0)
            self.widget_loads_single.setVisible(index > 0)

    def _reset_load_peaks(self):
        self.peak_loads = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        if hasattr(self, 'lbl_single_fpeak'):
            self.lbl_single_fpeak.setText("|F|_max (Picco Sessione): 0 N")


    def export_telemetry(self):
        try:
            hist = self.physics_thread.get_telemetry_snapshot()
            if not hist:
                QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria registrato.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Guida 14-DOF", "driver_telemetry_14dof.csv", "CSV Files (*.csv)")
            if path:
                df = pd.DataFrame(hist)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Esporta CSV", f"Telemetria salvata con successo ({len(df)} campioni x {len(df.columns)} canali):\n{path}")
        except Exception as e:
            err_msg = f"Errore durante l'esportazione della telemetria:\n{e}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Errore Esportazione", err_msg)

    def load_telemetry(self, file_path=None):
        """Loads saved telemetry data from a CSV or JSON file into the simulator and opens analysis."""
        try:
            if not file_path:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Carica File Telemetria 14-DOF",
                    "",
                    "File Telemetria (*.csv *.json);;File CSV (*.csv);;File JSON (*.json);;Tutti i File (*.*)"
                )
                if not path:
                    return
                file_path = path

            if not os.path.exists(file_path):
                QMessageBox.warning(self, "File Non Trovato", f"Il file specificato non esiste:\n{file_path}")
                return

            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    df = pd.DataFrame(raw)
                elif isinstance(raw, dict):
                    df = pd.DataFrame(raw)
                else:
                    raise ValueError("Formato JSON non valido: attesa lista di record o dizionario di serie.")
            else:
                df = pd.read_csv(file_path)

            if df.empty:
                QMessageBox.warning(self, "File Vuoto", "Il file selezionato non contiene record di dati validi.")
                return

            records = df.to_dict('records')
            with self.physics_thread._history_lock:
                self.physics_thread.telemetry_history.clear()
                self.physics_thread.telemetry_history.extend(records)

            QMessageBox.information(
                self,
                "Telemetria Caricata",
                f"Telemetria caricata con successo!\n"
                f"Campioni: {len(df)} | Canali: {len(df.columns)}\n"
                f"File: {os.path.basename(file_path)}\n\n"
                "Apertura della finestra di analisi telemetrica..."
            )

            dlg = FullTelemetryWindow(records, parent=self)
            dlg.setWindowTitle(f"Analisi Telemetria: {os.path.basename(file_path)} ({len(df)} campioni)")
            dlg.exec_()

        except Exception as e:
            err_msg = f"Errore durante il caricamento della telemetria:\n{e}\n\n{traceback.format_exc()}"
            try:
                with open("crash_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- LOAD TELEMETRY ERROR [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n{err_msg}\n")
            except Exception:
                pass
            QMessageBox.critical(self, "Errore Caricamento Telemetria", err_msg)

    def closeEvent(self, event):
        self.physics_thread.stop()
        super().closeEvent(event)





