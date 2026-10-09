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
from src.ui.sim_viewport import get_viewport_class

class TrackEditorCanvas(QWidget):
    """Interactive 2D Canvas for drawing, modifying, and generating smooth spline race tracks."""
    track_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1e272e; border: 2px solid #485460; border-radius: 6px;")
        self.setMouseTracking(True)

        self.control_points = [
            [0.0, 0.0],
            [100.0, 30.0],
            [160.0, 120.0],
            [110.0, 210.0],
            [-30.0, 180.0],
            [-100.0, 80.0]
        ]
        self.track_width = 3.8
        self.ds = 2.0
        self.is_closed = True
        
        self.scale = 1.1
        self.center_x = 30.0
        self.center_y = 100.0
        
        self.selected_node = -1
        self.hover_node = -1
        self.is_dragging = False
        self.is_panning = False
        self.pan_start = QPointF(0, 0)
        
        self.generated_track = None
        self.total_length_m = 0.0
        self.min_radius_m = 0.0
        self.recompute_track()

    def world_to_screen(self, wx, wy):
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        sx = cx + (wx - self.center_x) * self.scale
        sy = cy - (wy - self.center_y) * self.scale
        return QPointF(sx, sy)

    def screen_to_world(self, sx, sy):
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        wx = self.center_x + (sx - cx) / self.scale
        wy = self.center_y - (sy - cy) / self.scale
        return wx, wy

    def recompute_track(self):
        if len(self.control_points) < 3:
            self.generated_track = None
            self.track_changed.emit()
            self.update()
            return
            
        pts = np.array(self.control_points, dtype=float)
        if self.is_closed:
            pts_spline = np.vstack([pts, pts[0]])
        else:
            pts_spline = pts

        try:
            from scipy.interpolate import splprep, splev
            diffs = np.diff(pts_spline, axis=0)
            seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
            total_len = max(float(np.sum(seg_lens)), 10.0)
            
            k_deg = min(3, len(pts) - 1)
            tck, u = splprep([pts_spline[:, 0], pts_spline[:, 1]], s=0.0, per=self.is_closed, k=k_deg)
            num_dense = max(60, int(total_len / max(0.5, self.ds)))
            u_dense = np.linspace(0.0, 1.0, num_dense)
            x_dense, y_dense = splev(u_dense, tck)
            
            dx, dy = splev(u_dense, tck, der=1)
            ddx, ddy = splev(u_dense, tck, der=2)
            
            curv = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-6)
            psi = np.arctan2(dy, dx)
            
            nx = -np.sin(psi) * (self.track_width / 2.0)
            ny = np.cos(psi) * (self.track_width / 2.0)
            
            x_left = x_dense + nx
            y_left = y_dense + ny
            x_right = x_dense - nx
            y_right = y_dense - ny
            
            d_xy = np.hypot(np.diff(x_dense, prepend=x_dense[0]), np.diff(y_dense, prepend=y_dense[0]))
            s_arr = np.cumsum(d_xy)
            self.total_length_m = float(s_arr[-1])
            
            max_k = np.max(np.abs(curv))
            self.min_radius_m = float(1.0 / max(max_k, 1e-4))
            
            self.generated_track = {
                'x_center': np.array(x_dense, dtype=float),
                'y_center': np.array(y_dense, dtype=float),
                'x_left': np.array(x_left, dtype=float),
                'y_left': np.array(y_left, dtype=float),
                'x_right': np.array(x_right, dtype=float),
                'y_right': np.array(y_right, dtype=float),
                's': np.array(s_arr, dtype=float),
                'psi': np.array(psi, dtype=float),
                'curvature': np.array(curv, dtype=float),
                'w_left': np.full_like(x_dense, self.track_width / 2.0),
                'w_right': np.full_like(x_dense, self.track_width / 2.0)
            }
        except Exception:
            self.generated_track = {
                'x_center': pts[:, 0],
                'y_center': pts[:, 1],
                'x_left': pts[:, 0] + self.track_width / 2.0,
                'y_left': pts[:, 1],
                'x_right': pts[:, 0] - self.track_width / 2.0,
                'y_right': pts[:, 1],
                's': np.linspace(0, 100, len(pts)),
                'psi': np.zeros(len(pts)),
                'curvature': np.zeros(len(pts))
            }
        
        self.track_changed.emit()
        self.update()

    def mousePressEvent(self, event):
        sx, sy = event.pos().x(), event.pos().y()
        if event.button() == Qt.LeftButton:
            # Check if clicked on existing node
            clicked_idx = -1
            for idx, pt in enumerate(self.control_points):
                p_scr = self.world_to_screen(pt[0], pt[1])
                if math.hypot(sx - p_scr.x(), sy - p_scr.y()) < 14:
                    clicked_idx = idx
                    break
                    
            if clicked_idx >= 0:
                self.selected_node = clicked_idx
                self.is_dragging = True
            else:
                # Add new node
                wx, wy = self.screen_to_world(sx, sy)
                self.control_points.append([float(wx), float(wy)])
                self.selected_node = len(self.control_points) - 1
                self.is_dragging = True
                self.recompute_track()
        elif event.button() == Qt.RightButton:
            # Delete node under cursor
            for idx, pt in enumerate(self.control_points):
                p_scr = self.world_to_screen(pt[0], pt[1])
                if math.hypot(sx - p_scr.x(), sy - p_scr.y()) < 14:
                    if len(self.control_points) > 3:
                        self.control_points.pop(idx)
                        self.selected_node = -1
                        self.recompute_track()
                    break
        elif event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = event.pos()

    def mouseMoveEvent(self, event):
        sx, sy = event.pos().x(), event.pos().y()
        if self.is_dragging and self.selected_node >= 0:
            wx, wy = self.screen_to_world(sx, sy)
            self.control_points[self.selected_node] = [float(wx), float(wy)]
            self.recompute_track()
        elif self.is_panning:
            dx = (sx - self.pan_start.x()) / self.scale
            dy = -(sy - self.pan_start.y()) / self.scale
            self.center_x -= dx
            self.center_y -= dy
            self.pan_start = event.pos()
            self.update()
        else:
            # Update hover state
            prev_hover = self.hover_node
            self.hover_node = -1
            for idx, pt in enumerate(self.control_points):
                p_scr = self.world_to_screen(pt[0], pt[1])
                if math.hypot(sx - p_scr.x(), sy - p_scr.y()) < 14:
                    self.hover_node = idx
                    break
            if prev_hover != self.hover_node:
                self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.is_panning = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.scale = max(0.1, min(10.0, self.scale * factor))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        # 1. Background Grid & Coordinate Lines
        painter.fillRect(0, 0, w, h, QColor(30, 39, 46))
        painter.setPen(QPen(QColor(53, 59, 72), 1, Qt.DotLine))
        grid_step = 25.0
        
        # Grid lines in world coordinates
        min_wx, min_wy = self.screen_to_world(0, h)
        max_wx, max_wy = self.screen_to_world(w, 0)
        
        start_x = math.floor(min_wx / grid_step) * grid_step
        start_y = math.floor(min_wy / grid_step) * grid_step
        
        gx = start_x
        while gx < max_wx:
            p1 = self.world_to_screen(gx, min_wy)
            p2 = self.world_to_screen(gx, max_wy)
            painter.drawLine(p1, p2)
            gx += grid_step
            
        gy = start_y
        while gy < max_wy:
            p1 = self.world_to_screen(min_wx, gy)
            p2 = self.world_to_screen(max_wx, gy)
            painter.drawLine(p1, p2)
            gy += grid_step

        # 2. Render Generated Track Ribbon
        if self.generated_track:
            xl = self.generated_track['x_left']
            yl = self.generated_track['y_left']
            xr = self.generated_track['x_right']
            yr = self.generated_track['y_right']
            xc = self.generated_track['x_center']
            yc = self.generated_track['y_center']
            
            # Asphalt Surface Fill
            poly_pts = []
            for i in range(len(xl)):
                poly_pts.append(self.world_to_screen(xl[i], yl[i]))
            for i in range(len(xr)-1, -1, -1):
                poly_pts.append(self.world_to_screen(xr[i], yr[i]))
                
            painter.setBrush(QBrush(QColor(52, 73, 94, 210)))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygonF(poly_pts))
            
            # Left Edge Kerb (Red/White)
            painter.setPen(QPen(QColor(231, 76, 60), 2.5))
            for i in range(1, len(xl)):
                painter.drawLine(self.world_to_screen(xl[i-1], yl[i-1]), self.world_to_screen(xl[i], yl[i]))
                
            # Right Edge Kerb (Yellow)
            painter.setPen(QPen(QColor(241, 196, 15), 2.5))
            for i in range(1, len(xr)):
                painter.drawLine(self.world_to_screen(xr[i-1], yr[i-1]), self.world_to_screen(xr[i], yr[i]))
                
            # Centerline (White Dashes)
            painter.setPen(QPen(QColor(245, 246, 250, 180), 1.5, Qt.DashLine))
            for i in range(1, len(xc)):
                painter.drawLine(self.world_to_screen(xc[i-1], yc[i-1]), self.world_to_screen(xc[i], yc[i]))

            # Start/Finish Line Checker & Direction Arrow
            p_start = self.world_to_screen(xc[0], yc[0])
            p_start_l = self.world_to_screen(xl[0], yl[0])
            p_start_r = self.world_to_screen(xr[0], yr[0])
            painter.setPen(QPen(QColor(46, 204, 113), 3.5))
            painter.drawLine(p_start_l, p_start_r)

        # 3. Render Control Points & Polygon Line
        painter.setPen(QPen(QColor(0, 206, 201, 140), 1.5, Qt.DashLine))
        for i in range(len(self.control_points)):
            p1 = self.world_to_screen(self.control_points[i][0], self.control_points[i][1])
            next_idx = (i + 1) if not self.is_closed else ((i + 1) % len(self.control_points))
            if next_idx < len(self.control_points):
                p2 = self.world_to_screen(self.control_points[next_idx][0], self.control_points[next_idx][1])
                painter.drawLine(p1, p2)

        # Draw Nodes as Circular Badges
        for i, pt in enumerate(self.control_points):
            p = self.world_to_screen(pt[0], pt[1])
            is_sel = (i == self.selected_node)
            is_hov = (i == self.hover_node)
            r = 10 if (is_sel or is_hov) else 8
            
            if is_sel:
                painter.setBrush(QBrush(QColor(231, 76, 60)))
                painter.setPen(QPen(Qt.white, 2.5))
            elif is_hov:
                painter.setBrush(QBrush(QColor(241, 196, 15)))
                painter.setPen(QPen(Qt.white, 2.0))
            else:
                painter.setBrush(QBrush(QColor(9, 132, 227)))
                painter.setPen(QPen(QColor(223, 230, 233), 1.5))
                
            painter.drawEllipse(p, r, r)
            
            # Node Number
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(p.x() - 10, p.y() - 10, 20, 20), Qt.AlignCenter, str(i + 1))

        # Instructions Hint
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(178, 190, 195))
        painter.drawText(12, h - 14, "Click Sinistro: Aggiungi/Trascina Nodo | Click Destro: Rimuovi Nodo | Rotella: Zoom | Tasto Centrale: Pan")



class RoadProfileEditorDialog(QDialog):
    """
    User-defined 3D Road Profile Editor (Altimetry Z & Lateral Banking).
    Allows interactive station table editing, live dual-axis Matplotlib profile preview,
    terrain presets, and CSV/JSON file import/export.
    """
    def __init__(self, track_data=None, current_stations=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editor Profilo Stradale 3D (Dislivelli Z & Inclinazione Banking)")
        self.resize(1120, 720)
        self.setStyleSheet("""
            QDialog { background-color: #242b35; color: #ecf0f1; font-family: 'Segoe UI', sans-serif; }
            QLabel { color: #dfe6e9; font-size: 12px; }
            QGroupBox { font-weight: bold; color: #a29bfe; border: 1px solid #485460; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QTableWidget { background-color: #1e272e; color: #ecf0f1; gridline-color: #485460; border: 1px solid #485460; border-radius: 4px; selection-background-color: #0984e3; }
            QHeaderView::section { background-color: #2d3436; color: #00cec9; font-weight: bold; padding: 4px; border: 1px solid #485460; }
            QPushButton { background-color: #34495e; color: white; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #4b6584; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2d3436; color: white; border: 1px solid #485460; border-radius: 4px; padding: 4px; }
        """)

        self.track_data = track_data if track_data is not None else {}
        self.s_arr = self.track_data.get('s', None)
        if self.s_arr is None or len(self.s_arr) < 2:
            xc = self.track_data.get('x_center', [])
            yc = self.track_data.get('y_center', [])
            if len(xc) > 1:
                ds_arr = np.hypot(np.diff(xc), np.diff(yc))
                self.s_arr = np.concatenate([[0.0], np.cumsum(ds_arr)])
            else:
                self.s_arr = np.linspace(0.0, 600.0, 300)
        self.total_length = float(self.s_arr[-1] - self.s_arr[0]) if len(self.s_arr) > 1 else 600.0
        self.curvature = self.track_data.get('kappa', self.track_data.get('curvature', None))
        self.is_closed = bool(self.track_data.get('is_closed', True))

        if current_stations and len(current_stations) >= 2:
            self.stations = []
            for st in current_stations:
                if isinstance(st, dict):
                    self.stations.append({
                        's': round(float(st['s']), 1),
                        'z': round(float(st['z']), 2),
                        'bank_deg': round(float(st.get('bank_deg', 0.0)), 1)
                    })
                elif isinstance(st, (list, tuple)) and len(st) >= 2:
                    self.stations.append({
                        's': round(float(st[0]), 1),
                        'z': round(float(st[1]), 2),
                        'bank_deg': round(float(st[2]) if len(st) > 2 else 0.0, 1)
                    })
        else:
            self.stations = RoadProfile.get_preset_stations("Collinare (Onde & Banking)", self.total_length, self.s_arr, self.curvature)

        self._updating_table = False
        self.init_ui()
        self.populate_table()
        self.update_plots()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # Header Info Banner
        header = QHBoxLayout()
        lbl_title = QLabel("<b>Editor Profilo 3D del Circuito</b> — Dislivelli Quota <i>Z(s)</i> e Inclinazione Trasversale <i>Banking(s)</i>")
        lbl_title.setStyleSheet("font-size: 14px; color: #a29bfe; font-weight: bold;")
        header.addWidget(lbl_title)
        header.addStretch()
        self.lbl_track_info = QLabel(f"Lunghezza Circuito: <b>{self.total_length:.1f} m</b> | Modalità: <b>{'Chiuso (Loop)' if self.is_closed else 'Aperto'}</b>")
        self.lbl_track_info.setStyleSheet("color: #00cec9; font-size: 12px;")
        header.addWidget(self.lbl_track_info)
        main_layout.addLayout(header)

        # Preset & Toolbar Bar
        tool_bar = QHBoxLayout()
        tool_bar.addWidget(QLabel("<b>Preset Altimetrico:</b>"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "Personalizzato (Manuale)",
            "Piatto (2D)",
            "Collinare (Onde & Banking)",
            "Circuito di Montagna (Salita/Discesa +-15m)",
            "Curve Sopraelevate (Banked Corners 10°)",
            "Rollercoaster (Dossi & Avvallamenti)"
        ])
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)
        tool_bar.addWidget(self.combo_presets)

        tool_bar.addSpacing(15)
        tool_bar.addWidget(QLabel("Punti:"))
        self.sb_n_stations = QSpinBox()
        self.sb_n_stations.setRange(4, 32)
        self.sb_n_stations.setValue(len(self.stations) if len(self.stations) >= 4 else 8)
        tool_bar.addWidget(self.sb_n_stations)

        btn_uniform = QPushButton("Genera Griglia")
        btn_uniform.clicked.connect(self.generate_uniform_stations)
        tool_bar.addWidget(btn_uniform)

        btn_flat = QPushButton("Azzera (Piatto 2D)")
        btn_flat.clicked.connect(self.reset_to_flat)
        tool_bar.addWidget(btn_flat)

        self.btn_autobank = QPushButton("⚡ Auto-Banking da Curvatura")
        self.btn_autobank.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_autobank.setToolTip("Calcola automaticamente il banking ottimale per ciascuna curva del tracciato attuale")
        self.btn_autobank.clicked.connect(self.auto_bank_from_curvature)
        tool_bar.addWidget(self.btn_autobank)

        tool_bar.addStretch()
        main_layout.addLayout(tool_bar)

        # Central Splitter (Left: Table, Right: Matplotlib Figure)
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Table & Controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Progressiva s [m]", "Quota Z [m]", "Pendenza [%]", "Banking [°]"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellChanged.connect(self.on_table_cell_changed)
        left_layout.addWidget(self.table)

        # Table Row Action Buttons
        row_btn_layout = QHBoxLayout()
        btn_add = QPushButton("➕ Aggiungi Stazione")
        btn_add.clicked.connect(self.add_station)
        row_btn_layout.addWidget(btn_add)

        btn_del = QPushButton("➖ Elimina Selezionata")
        btn_del.clicked.connect(self.delete_selected_station)
        row_btn_layout.addWidget(btn_del)

        btn_sort = QPushButton("🔄 Ordina per s")
        btn_sort.clicked.connect(self.sort_stations)
        row_btn_layout.addWidget(btn_sort)
        left_layout.addLayout(row_btn_layout)

        # File Import/Export
        io_layout = QHBoxLayout()
        btn_import = QPushButton("📂 Carica CSV/JSON...")
        btn_import.clicked.connect(self.import_profile)
        io_layout.addWidget(btn_import)

        btn_export = QPushButton("💾 Salva CSV/JSON...")
        btn_export.clicked.connect(self.export_profile)
        io_layout.addWidget(btn_export)
        left_layout.addLayout(io_layout)

        splitter.addWidget(left_widget)

        # Right Panel (Matplotlib Canvas with 2D Track Map & 3D Profiles)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(7.5, 8.0), facecolor="#1e272e")
        self.canvas_fig = FigureCanvas(self.fig)
        self.ax_map = self.fig.add_subplot(311)
        self.ax_z = self.fig.add_subplot(312)
        self.ax_bank = self.fig.add_subplot(313)
        right_layout.addWidget(self.canvas_fig)

        splitter.addWidget(right_widget)
        splitter.setSizes([440, 680])
        main_layout.addWidget(splitter, stretch=1)

        # Bottom Action Bar
        bot_layout = QHBoxLayout()
        self.cb_enable_macro = QCheckBox("Abilita Altimetria e Banking 3D nel Simulatore")
        self.cb_enable_macro.setChecked(True)
        self.cb_enable_macro.setStyleSheet("font-weight: bold; color: #a29bfe; font-size: 13px;")
        bot_layout.addWidget(self.cb_enable_macro)

        bot_layout.addSpacing(20)
        self.lbl_stats = QLabel("Statistiche:")
        self.lbl_stats.setStyleSheet("color: #74b9ff; font-weight: bold;")
        bot_layout.addWidget(self.lbl_stats)

        bot_layout.addStretch()

        btn_cancel = QPushButton("Annulla")
        btn_cancel.clicked.connect(self.reject)
        bot_layout.addWidget(btn_cancel)

        btn_apply = QPushButton("Applica al Simulatore")
        btn_apply.setStyleSheet("background-color: #27ae60; color: white; font-size: 13px; font-weight: bold; padding: 8px 18px; border-radius: 4px;")
        btn_apply.clicked.connect(self.apply_and_accept)
        bot_layout.addWidget(btn_apply)

        main_layout.addLayout(bot_layout)

    def populate_table(self):
        self._updating_table = True
        self.table.setRowCount(len(self.stations))
        for r, st in enumerate(self.stations):
            item_s = QTableWidgetItem(f"{st['s']:.1f}")
            item_z = QTableWidgetItem(f"{st['z']:.2f}")
            item_b = QTableWidgetItem(f"{st['bank_deg']:.1f}")
            
            # dz/ds calculated later in update_plots
            item_grade = QTableWidgetItem("0.0 %")
            item_grade.setFlags(item_grade.flags() & ~Qt.ItemIsEditable)

            self.table.setItem(r, 0, item_s)
            self.table.setItem(r, 1, item_z)
            self.table.setItem(r, 2, item_grade)
            self.table.setItem(r, 3, item_b)
        self._updating_table = False

    def on_table_cell_changed(self, row, col):
        if self._updating_table or row >= len(self.stations):
            return
        item = self.table.item(row, col)
        if not item:
            return
        try:
            val = float(item.text().replace('%', '').strip())
            if col == 0:
                self.stations[row]['s'] = min(max(val, 0.0), self.total_length)
            elif col == 1:
                self.stations[row]['z'] = float(val)
            elif col == 3:
                self.stations[row]['bank_deg'] = min(max(val, -30.0), 30.0)
            self.update_plots()
        except ValueError:
            pass

    def update_plots(self):
        if len(self.stations) < 2:
            return

        # Sort stations temporarily for spline evaluation
        st_sorted = sorted(self.stations, key=lambda x: x['s'])
        rp = RoadProfile.from_user_stations(self.s_arr, st_sorted, is_closed=self.is_closed)

        # Update grade column in table
        self._updating_table = True
        for r, st in enumerate(self.stations):
            _, gr, _ = rp.interpolate_at(st['s'])
            it = self.table.item(r, 2)
            if it:
                it.setText(f"{gr * 100.0:+.1f} %")
        self._updating_table = False

        # Clear subplots
        self.ax_map.clear()
        self.ax_z.clear()
        self.ax_bank.clear()

        # Styling
        for ax in (self.ax_map, self.ax_z, self.ax_bank):
            ax.set_facecolor("#242b35")
            ax.tick_params(colors="#b2bec3", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.35, color="#636e72")
            for spine in ax.spines.values():
                spine.set_color("#485460")

        # Top Plot: 2D Track Map with Station Locations
        xc = self.track_data.get('x_center', [])
        yc = self.track_data.get('y_center', [])
        s_pts = [p['s'] for p in self.stations]
        z_pts = [p['z'] for p in self.stations]
        b_pts = [p['bank_deg'] for p in self.stations]

        if len(xc) > 1:
            self.ax_map.plot(xc, yc, color="#00cec9", linewidth=1.8, label="Tracciato 2D")
            self.ax_map.scatter([xc[0]], [yc[0]], color="#2ecc71", s=50, marker="s", zorder=5, label="Start/Finish")
            st_x = np.interp(s_pts, self.s_arr, xc)
            st_y = np.interp(s_pts, self.s_arr, yc)
            self.ax_map.scatter(st_x, st_y, color="#ff7675", s=40, zorder=6, edgecolors="white", linewidths=1.0, label="Stazioni 3D")
            for i, (sx, sy) in enumerate(zip(st_x, st_y)):
                self.ax_map.annotate(f"#{i+1}", (sx, sy), textcoords="offset points", xytext=(0, 4),
                                     ha='center', fontsize=7, color="#fab1a0", fontweight="bold")
            self.ax_map.set_aspect('equal', 'datalim')
            self.ax_map.set_title("Mappa 2D Tracciato & Posizione Stazioni 3D", color="#dfe6e9", fontsize=9.5, fontweight="bold", pad=3)
            self.ax_map.legend(loc="upper right", facecolor="#2d3436", edgecolor="#485460", labelcolor="#dfe6e9", fontsize=7)

        # Middle Plot: Elevation Z(s)
        self.ax_z.plot(self.s_arr, rp.z_macro, color="#00cec9", linewidth=2.0, label="Profilo Quota Z(s)")
        self.ax_z.fill_between(self.s_arr, 0.0, rp.z_macro, color="#00cec9", alpha=0.15)
        self.ax_z.scatter(s_pts, z_pts, color="#ff7675", s=40, zorder=5, edgecolors="white", linewidths=1.0, label="Stazioni Utente")
        for i, (sp, zp) in enumerate(zip(s_pts, z_pts)):
            self.ax_z.annotate(f"#{i+1}", (sp, zp), textcoords="offset points", xytext=(0, 6),
                               ha='center', fontsize=7.5, color="#fab1a0", fontweight="bold")

        self.ax_z.set_ylabel("Quota Z [m]", color="#00cec9", fontsize=9, fontweight="bold")
        self.ax_z.set_title("Profilo Altimetrico Z(s)", color="#dfe6e9", fontsize=9.5, fontweight="bold", pad=3)
        self.ax_z.set_xlim(0, self.total_length)
        self.ax_z.legend(loc="upper right", facecolor="#2d3436", edgecolor="#485460", labelcolor="#dfe6e9", fontsize=7)

        # Bottom Plot: Banking(s) & Track Curvature Reference
        bank_deg_arr = np.rad2deg(rp.bank_rad)
        self.ax_bank.plot(self.s_arr, bank_deg_arr, color="#fdcb6e", linewidth=1.8, label="Inclinazione Banking [°]")
        self.ax_bank.axhline(0.0, color="#636e72", linestyle=":", linewidth=1.0)
        self.ax_bank.scatter(s_pts, b_pts, color="#e17055", s=40, zorder=5, edgecolors="white", linewidths=1.0, label="Punti Banking")
        if self.curvature is not None and len(self.curvature) == len(self.s_arr):
            k_scaled = np.rad2deg(self.curvature) * 0.4
            self.ax_bank.plot(self.s_arr, k_scaled, color="#74b9ff", linestyle="--", linewidth=1.1, alpha=0.70, label="Curvatura κ [x0.4]")
        self.ax_bank.set_xlabel("Distanza / Progressiva s [m]", color="#dfe6e9", fontsize=9)
        self.ax_bank.set_ylabel("Banking [°]", color="#fdcb6e", fontsize=9, fontweight="bold")
        self.ax_bank.set_title("Inclinazione Trasversale Banking(s) & Curvatura Tracciato", color="#dfe6e9", fontsize=9.5, fontweight="bold", pad=3)
        self.ax_bank.set_xlim(0, self.total_length)
        self.ax_bank.legend(loc="upper right", facecolor="#2d3436", edgecolor="#485460", labelcolor="#dfe6e9", fontsize=7)

        self.fig.tight_layout()
        self.canvas_fig.draw()

        # Update stats
        z_min = float(np.min(rp.z_macro))
        z_max = float(np.max(rp.z_macro))
        b_max = float(np.max(np.abs(bank_deg_arr)))
        gr_max = float(np.max(np.abs(rp.grade)) * 100.0)
        self.lbl_stats.setText(
            f"Z: [{z_min:+.1f} m, {z_max:+.1f} m] | Dislivello: {z_max - z_min:.1f} m | "
            f"Pendenza Max: {gr_max:.1f}% | Banking Max: ±{b_max:.1f}°"
        )

    def auto_bank_from_curvature(self):
        """Calculates ideal banking based on track curvature: bank_deg = clip(gain * kappa, -12.0, 12.0)."""
        if self.curvature is None or len(self.curvature) < 2:
            xc = self.track_data.get('x_center', [])
            yc = self.track_data.get('y_center', [])
            if len(xc) > 3:
                dx = np.gradient(xc)
                dy = np.gradient(yc)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                denom = np.maximum((dx**2 + dy**2)**1.5, 1e-6)
                self.curvature = (dx * ddy - dy * ddx) / denom
            else:
                return

        k_max = float(np.max(np.abs(self.curvature)))
        if k_max < 1e-4:
            QMessageBox.information(self, "Auto-Banking", "Il tracciato attuale non ha curve sufficienti per calcolare banking.")
            return
        gain = 8.5 / k_max
        for st in self.stations:
            s_val = st['s']
            kap = float(np.interp(s_val, self.s_arr, self.curvature))
            st['bank_deg'] = round(float(np.clip(kap * gain, -12.0, 12.0)), 1)
        self.populate_table()
        self.update_plots()
        if os.environ.get('QT_QPA_PLATFORM') != 'offscreen':
            QMessageBox.information(self, "Auto-Banking Completato", f"Banking calcolato automaticamente su {len(self.stations)} stazioni in base alla geometria reale del circuito!")

    def on_preset_changed(self, idx):
        preset_name = self.combo_presets.currentText()
        if preset_name == "Personalizzato (Manuale)":
            return
        self.stations = RoadProfile.get_preset_stations(preset_name, self.total_length, self.s_arr, self.curvature)
        self.populate_table()
        self.update_plots()

    def generate_uniform_stations(self):
        n = self.sb_n_stations.value()
        self.stations = RoadProfile.get_default_stations(self.total_length, n)
        self.populate_table()
        self.update_plots()

    def reset_to_flat(self):
        for st in self.stations:
            st['z'] = 0.0
            st['bank_deg'] = 0.0
        self.populate_table()
        self.update_plots()

    def add_station(self):
        cur_row = self.table.currentRow()
        if cur_row >= 0 and cur_row < len(self.stations) - 1:
            s_new = (self.stations[cur_row]['s'] + self.stations[cur_row + 1]['s']) / 2.0
            z_new = (self.stations[cur_row]['z'] + self.stations[cur_row + 1]['z']) / 2.0
            b_new = (self.stations[cur_row]['bank_deg'] + self.stations[cur_row + 1]['bank_deg']) / 2.0
            self.stations.insert(cur_row + 1, {'s': round(s_new, 1), 'z': round(z_new, 2), 'bank_deg': round(b_new, 1)})
        else:
            s_new = min(self.stations[-1]['s'] + 50.0, self.total_length) if self.stations else 0.0
            self.stations.append({'s': round(s_new, 1), 'z': 0.0, 'bank_deg': 0.0})
        self.populate_table()
        self.update_plots()

    def delete_selected_station(self):
        if len(self.stations) <= 2:
            QMessageBox.warning(self, "Attenzione", "Sono richieste almeno 2 stazioni per definire un profilo continuo.")
            return
        cur_row = self.table.currentRow()
        if cur_row >= 0 and cur_row < len(self.stations):
            self.stations.pop(cur_row)
            self.populate_table()
            self.update_plots()

    def sort_stations(self):
        self.stations.sort(key=lambda x: x['s'])
        self.populate_table()
        self.update_plots()

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importa Profilo 3D", "tracks", "Profile Files (*.csv *.json);;All Files (*)")
        if not path:
            return
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_stations = data
                    elif isinstance(data, dict) and 'stations' in data:
                        raw_stations = data['stations']
                    else:
                        raise ValueError("Formato JSON non riconosciuto: atteso array di stazioni o dict con chiave 'stations'.")
                st_parsed = []
                for item in raw_stations:
                    st_parsed.append({
                        's': float(item['s']),
                        'z': float(item['z']),
                        'bank_deg': float(item.get('bank_deg', item.get('bank', 0.0)))
                    })
            else:
                df = pd.read_csv(path)
                s_col = 's' if 's' in df.columns else df.columns[0]
                z_col = 'z' if 'z' in df.columns else ('z_center' if 'z_center' in df.columns else df.columns[1])
                bank_col = 'bank_deg' if 'bank_deg' in df.columns else ('bank' if 'bank' in df.columns else (df.columns[2] if len(df.columns) > 2 else None))
                
                st_parsed = []
                step = max(1, len(df) // 16)
                for idx in range(0, len(df), step):
                    s_val = float(df.iloc[idx][s_col])
                    z_val = float(df.iloc[idx][z_col])
                    b_val = float(df.iloc[idx][bank_col]) if bank_col else 0.0
                    st_parsed.append({'s': round(s_val, 1), 'z': round(z_val, 2), 'bank_deg': round(b_val, 1)})
            
            if len(st_parsed) >= 2:
                self.stations = st_parsed
                self.populate_table()
                self.update_plots()
                QMessageBox.information(self, "Profilo Caricato", f"Caricate con successo {len(self.stations)} stazioni da:\n{path}")
            else:
                QMessageBox.warning(self, "Attenzione", "Il file deve contenere almeno 2 stazioni.")
        except Exception as e:
            QMessageBox.critical(self, "Errore Caricamento", f"Impossibile leggere il file:\n{e}")

    def export_profile(self):
        if len(self.stations) < 2:
            QMessageBox.warning(self, "Attenzione", "Definisci almeno 2 stazioni prima di esportare.")
            return
        path, sel_filter = QFileDialog.getSaveFileName(self, "Salva Profilo 3D", "tracks/custom_road_profile.json", "JSON Files (*.json);;CSV Files (*.csv)")
        if not path:
            return
        try:
            if path.endswith('.csv'):
                df = pd.DataFrame(self.stations)
                df.to_csv(path, index=False)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({'stations': self.stations, 'total_length': self.total_length, 'is_closed': self.is_closed}, f, indent=2)
            QMessageBox.information(self, "Profilo Salvato", f"Profilo 3D salvato con successo:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il profilo:\n{e}")

    def apply_and_accept(self):
        if len(self.stations) < 2:
            QMessageBox.warning(self, "Attenzione", "Sono necessarie almeno 2 stazioni.")
            return
        self.stations.sort(key=lambda x: x['s'])
        self.accept()

    def get_stations(self):
        return list(self.stations)



class TrackDesignerDialog(QDialog):
    """Full Interactive 2D Track Designer & Spline Editor Dialog."""
    def __init__(self, current_track=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editor & Disegnatore Tracciato 2D (Spline FSAE)")
        self.resize(1020, 680)
        self.setStyleSheet("background-color: #242b35; color: #ecf0f1; font-family: Segoe UI, sans-serif;")

        self.generated_track = None
        self.init_ui(current_track)

    def init_ui(self, current_track):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top Control Bar
        top_bar = QHBoxLayout()
        
        top_bar.addWidget(QLabel("<b>Template / Preset:</b>"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "FSAE Autocross Custom",
            "FSAE Skidpad (Figure-8)",
            "Circuito Tecnico (Misto Hairpin)",
            "Slalom Test Track",
            "Anello Ovale Veloce",
            "Tracciato Libero"
        ])
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)
        top_bar.addWidget(self.combo_presets)
        
        top_bar.addSpacing(15)
        top_bar.addWidget(QLabel("Larghezza Pista (m):"))
        self.sb_width = QDoubleSpinBox()
        self.sb_width.setRange(2.0, 8.0)
        self.sb_width.setValue(3.8)
        self.sb_width.setSingleStep(0.2)
        self.sb_width.valueChanged.connect(self.on_params_changed)
        top_bar.addWidget(self.sb_width)
        
        top_bar.addSpacing(15)
        self.cb_closed = QCheckBox("Circuito Chiuso (Loop)")
        self.cb_closed.setChecked(True)
        self.cb_closed.toggled.connect(self.on_params_changed)
        top_bar.addWidget(self.cb_closed)
        
        top_bar.addStretch()
        
        self.btn_clear = QPushButton("Pulisci Canvas")
        self.btn_clear.clicked.connect(self.clear_canvas)
        top_bar.addWidget(self.btn_clear)
        
        layout.addLayout(top_bar)

        # Center Canvas
        self.canvas = TrackEditorCanvas(self)
        self.canvas.track_changed.connect(self.update_stats)
        layout.addWidget(self.canvas, stretch=1)

        # Bottom Stats Card & Action Buttons
        bot_layout = QHBoxLayout()
        
        self.lbl_stats = QLabel("Lunghezza: 0.0 m | Raggio Min: 0.0 m | Nodi: 0")
        self.lbl_stats.setStyleSheet("font-size: 13px; font-weight: bold; color: #00cec9; padding: 4px;")
        bot_layout.addWidget(self.lbl_stats)
        bot_layout.addStretch()

        self.btn_import_csv = QPushButton("Carica da CSV...")
        self.btn_import_csv.clicked.connect(self.import_csv)
        bot_layout.addWidget(self.btn_import_csv)

        self.btn_export_csv = QPushButton("Salva su CSV...")
        self.btn_export_csv.clicked.connect(self.export_csv)
        bot_layout.addWidget(self.btn_export_csv)

        self.btn_3d_profile = QPushButton("Profilo 3D (Quote & Banking)...")
        self.btn_3d_profile.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_3d_profile.clicked.connect(self.open_3d_profile_editor)
        bot_layout.addWidget(self.btn_3d_profile)

        self.btn_apply = QPushButton("Carica nel Simulatore")
        self.btn_apply.setStyleSheet("font-weight: bold; font-size: 13px; background-color: #27ae60; color: white; padding: 8px 16px; border-radius: 4px;")
        self.btn_apply.clicked.connect(self.apply_and_accept)
        bot_layout.addWidget(self.btn_apply)

        layout.addLayout(bot_layout)
        self.update_stats()

    def open_3d_profile_editor(self):
        if not self.canvas.generated_track:
            QMessageBox.warning(self, "Tracciato Non Valido", "Disegna o genera prima un tracciato valido con almeno 3 punti.")
            return
        dlg = RoadProfileEditorDialog(
            track_data=self.canvas.generated_track,
            current_stations=self.canvas.generated_track.get('custom_stations', None),
            parent=self
        )
        if dlg.exec_() == QDialog.Accepted:
            st = dlg.get_stations()
            self.canvas.generated_track['custom_stations'] = st
            rp = RoadProfile.from_user_stations(self.canvas.generated_track['s'], st, is_closed=self.canvas.is_closed)
            rp.apply_to_track_data(self.canvas.generated_track)
            QMessageBox.information(self, "Profilo 3D Configurato", f"Configurate con successo {len(st)} stazioni 3D per questo tracciato!")

    def on_params_changed(self):
        self.canvas.track_width = self.sb_width.value()
        self.canvas.is_closed = self.cb_closed.isChecked()
        self.canvas.recompute_track()

    def update_stats(self):
        pts_count = len(self.canvas.control_points)
        L = self.canvas.total_length_m
        R_min = self.canvas.min_radius_m
        self.lbl_stats.setText(f"Lunghezza: <b>{L:.1f} m</b> | Raggio Minimo: <b>{R_min:.1f} m</b> | Nodi di Controllo: <b>{pts_count}</b>")

    def on_preset_changed(self, idx):
        if idx == 0:  # FSAE Autocross Custom
            self.canvas.control_points = [
                [0.0, 0.0], [80.0, 20.0], [140.0, 90.0], [180.0, 160.0],
                [130.0, 220.0], [50.0, 200.0], [-20.0, 140.0], [-80.0, 80.0]
            ]
            self.cb_closed.setChecked(True)
        elif idx == 1:  # Skidpad Figure-8
            r = 15.25
            self.canvas.control_points = [
                [0.0, -r], [r, 0.0], [0.0, r], [-r, 0.0],
                [0.0, 0.0],
                [0.0, 2*r - r], [r, 2*r], [0.0, 3*r], [-r, 2*r]
            ]
            self.cb_closed.setChecked(True)
        elif idx == 2:  # Circuito Tecnico
            self.canvas.control_points = [
                [0.0, 0.0], [120.0, 0.0], [180.0, 50.0], [160.0, 120.0],
                [100.0, 100.0], [60.0, 160.0], [-20.0, 180.0], [-100.0, 100.0],
                [-60.0, 30.0]
            ]
            self.cb_closed.setChecked(True)
        elif idx == 3:  # Slalom Test Track
            self.canvas.control_points = [
                [0.0, 0.0], [40.0, 15.0], [80.0, -15.0], [120.0, 15.0],
                [160.0, -15.0], [200.0, 15.0], [240.0, 0.0]
            ]
            self.cb_closed.setChecked(False)
        elif idx == 4:  # Anello Ovale Veloce
            self.canvas.control_points = [
                [0.0, 0.0], [150.0, 0.0], [200.0, 60.0], [150.0, 120.0],
                [0.0, 120.0], [-50.0, 60.0]
            ]
            self.cb_closed.setChecked(True)
            
        self.canvas.recompute_track()

    def clear_canvas(self):
        self.canvas.control_points = []
        self.canvas.recompute_track()

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Carica Tracciato CSV", "tracks", "CSV Files (*.csv);;All Files (*)")
        if path:
            try:
                df = pd.read_csv(path)
                if 'x_center' in df.columns and 'y_center' in df.columns:
                    step = max(1, len(df) // 12)
                    pts = df[['x_center', 'y_center']].iloc[::step].values.tolist()
                    self.canvas.control_points = pts
                    self.canvas.recompute_track()
                    QMessageBox.information(self, "Tracciato Caricato", f"Caricati con successo {len(pts)} nodi di controllo da:\n{path}")
                else:
                    QMessageBox.warning(self, "Formato Non Valido", "Il file CSV deve contenere le colonne 'x_center' e 'y_center'.")
            except Exception as e:
                QMessageBox.critical(self, "Errore Caricamento", f"Impossibile leggere il file CSV:\n{e}")

    def export_csv(self):
        if not self.canvas.generated_track:
            QMessageBox.warning(self, "Tracciato Vuoto", "Disegna prima un tracciato valido con almeno 3 punti.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salva Tracciato CSV", "tracks/custom_drawn_track.csv", "CSV Files (*.csv)")
        if path:
            try:
                df = pd.DataFrame(self.canvas.generated_track)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Tracciato Salvato", f"File salvato con successo:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file CSV:\n{e}")

    def apply_and_accept(self):
        if not self.canvas.generated_track:
            QMessageBox.warning(self, "Tracciato Non Valido", "Disegna prima un tracciato con almeno 3 punti.")
            return
        self.generated_track = self.canvas.generated_track
        self.accept()


# =============================================================================
# 6. MAIN SIMULATOR WINDOW WITH DYNAMIC FREQUENCY & HAPTIC CONTROLS
# =============================================================================

