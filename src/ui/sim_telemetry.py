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


class TelemetryPopoutDialog(QDialog):
    """
    Independent resizable & fullscreen-capable dialog to view, inspect, and export
    any telemetry figure or single chart in high definition with interactive mouse zoom.
    """
    def __init__(self, fig, title="Grafico Telemetria Ingrandito", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} - Dynamic Target Suite Telemetria")
        self.resize(1150, 780)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet("background-color: #1e272e; color: #ecf0f1;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Action Bar
        top_bar = QHBoxLayout()
        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_title.setStyleSheet("font-size: 13px; color: #ecf0f1;")
        top_bar.addWidget(lbl_title)
        
        top_bar.addSpacing(15)
        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        btn_reset_zoom.clicked.connect(self._reset_zoom)
        top_bar.addWidget(btn_reset_zoom)

        btn_z10 = QPushButton("10s")
        btn_z10.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        btn_z10.clicked.connect(lambda: self._set_time_window(10.0))
        top_bar.addWidget(btn_z10)

        btn_z30 = QPushButton("30s")
        btn_z30.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        btn_z30.clicked.connect(lambda: self._set_time_window(30.0))
        top_bar.addWidget(btn_z30)
        
        top_bar.addStretch()

        self.btn_fullscreen = QPushButton("[ ] Schermo Intero (F11)")
        self.btn_fullscreen.setStyleSheet("font-weight: bold; padding: 5px 12px; background-color: #0984e3; color: white; border-radius: 4px;")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)

        btn_save = QPushButton("Salva Immagine HD")
        btn_save.setStyleSheet("font-weight: bold; padding: 5px 12px; background-color: #27ae60; color: white; border-radius: 4px;")
        btn_save.clicked.connect(self.save_hd_image)
        top_bar.addWidget(btn_save)

        btn_close = QPushButton("Chiudi (Esc)")
        btn_close.setStyleSheet("padding: 5px 12px; border-radius: 4px; font-weight: bold; background-color: #7f8c8d; color: white;")
        btn_close.clicked.connect(self.accept)
        top_bar.addWidget(btn_close)
        layout.addLayout(top_bar)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._is_fullscreen = False

        # Connect interactive mouse wheel zoom
        self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll_zoom)

    def _on_mouse_scroll_zoom(self, event):
        """Smoothly zooms the X-axis (and Y-axis for G-G) around mouse cursor."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 0.8 if event.button == 'up' else 1.25
        
        cur_xlim = ax.get_xlim()
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_x = (event.xdata - cur_xlim[0]) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
        ax.set_xlim([event.xdata - new_w * rel_x, event.xdata + new_w * (1.0 - rel_x)])
        
        # If G-G diagram (square aspect), zoom Y synchronously
        if getattr(ax, 'name', '') == 'rectilinear' and ax.get_aspect() == 1.0:
            cur_ylim = ax.get_ylim()
            new_h = (cur_ylim[1] - cur_ylim[0]) * scale
            rel_y = (event.ydata - cur_ylim[0]) / max(cur_ylim[1] - cur_ylim[0], 1e-6)
            ax.set_ylim([event.ydata - new_h * rel_y, event.ydata + new_h * (1.0 - rel_y)])
            
        self.canvas.draw_idle()

    def _reset_zoom(self):
        for ax in self.canvas.figure.axes:
            ax.autoscale()
            if hasattr(ax, '_is_gg') and ax._is_gg:
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.set_aspect('equal', 'box')
        self.canvas.draw_idle()

    def _set_time_window(self, win_sec):
        for ax in self.canvas.figure.axes:
            if not (hasattr(ax, '_is_gg') and ax._is_gg):
                lines = ax.get_lines()
                if lines:
                    xdata = lines[0].get_xdata()
                    if len(xdata) > 0:
                        max_x = xdata[-1]
                        min_x = max(0.0, max_x - win_sec)
                        ax.set_xlim(min_x, max_x)
        self.canvas.draw_idle()

    def toggle_fullscreen(self):
        if not self._is_fullscreen:
            self.showFullScreen()
            self.btn_fullscreen.setText("[X] Esci da Schermo Intero (F11)")
            self._is_fullscreen = True
        else:
            self.showNormal()
            self.btn_fullscreen.setText("[ ] Schermo Intero (F11)")
            self._is_fullscreen = False

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
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
            
        if event.key() == Qt.Key_1:
            self.set_camera_preset(6.8, 3.6, 420, 0)
        elif event.key() == Qt.Key_2:
            self.set_camera_preset(0.2, 1.1, 550, -5)
        elif event.key() == Qt.Key_3:
            self.set_camera_preset(-0.5, 0.4, 500, -2)
        elif event.key() == Qt.Key_4:
            self.set_camera_preset(12.0, 8.0, 350, -15)
        elif event.key() == Qt.Key_5:
            self.combo_cam.setCurrentIndex(1)
            
        if event.key() == Qt.Key_M:
            if hasattr(self, 'viewport'):
                self.viewport.show_minimap = not getattr(self.viewport, 'show_minimap', True)
                self.viewport.update()
            return

        self.physics_thread.key_states[event.key()] = True
        if event.key() == Qt.Key_V:
            new_idx = 1 if self.viewport.view_mode == "3D" else 0
            self.combo_cam.setCurrentIndex(new_idx)
        super().keyPressEvent(event)

    def save_hd_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salva Grafico HD", "telemetry_chart_hd.png", "PNG Image (*.png);;PDF Vector (*.pdf);;SVG Vector (*.svg)")
        if path:
            self.canvas.figure.savefig(path, dpi=300, bbox_inches='tight')



class CustomTelemetryLayoutDialog(QDialog):
    """Dialog to configure custom subplots grid and channel mapping."""
    def __init__(self, current_layout, available_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurazione Layout Telemetria Personalizzato")
        self.resize(560, 420)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        
        self.available_channels = available_channels
        self.selected_layout = list(current_layout)
        
        layout = QVBoxLayout(self)
        
        # Grid format selection
        fmt_box = QGroupBox("Formato Griglia Grafici")
        fmt_layout = QHBoxLayout(fmt_box)
        fmt_layout.addWidget(QLabel("Numero Grafici:"))
        self.combo_count = QComboBox()
        self.combo_count.addItems(["1 Grafico (1x1)", "2 Grafici (2x1)", "4 Grafici (2x2)", "6 Grafici (2x3)", "8 Grafici (4x2)"])
        
        count_map = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
        cur_cnt = len(self.selected_layout)
        self.combo_count.setCurrentIndex(count_map.get(cur_cnt, 3))
        self.combo_count.currentIndexChanged.connect(self._rebuild_channel_selectors)
        fmt_layout.addWidget(self.combo_count)
        layout.addWidget(fmt_box)
        
        # Channel assignments
        self.channels_box = QGroupBox("Assegnazione Canali / Grafici per Ciascun Slot")
        self.channels_layout = QVBoxLayout(self.channels_box)
        layout.addWidget(self.channels_box)
        
        self.slot_combos = []
        self._rebuild_channel_selectors()
        
        # Buttons
        btn_bar = QHBoxLayout()
        btn_ok = QPushButton("Applica Layout")
        btn_ok.setStyleSheet("font-weight: bold; padding: 6px 14px; background-color: #27ae60; color: white; border-radius: 4px;")
        btn_ok.clicked.connect(self._on_apply)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Annulla")
        btn_cancel.setStyleSheet("padding: 6px 14px; border-radius: 4px; background-color: #7f8c8d; color: white;")
        btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(btn_cancel)
        layout.addLayout(btn_bar)
        
    def _rebuild_channel_selectors(self):
        # Clear existing
        while self.channels_layout.count():
            item = self.channels_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.slot_combos = []
        cnt_text = self.combo_count.currentText()
        count = int(cnt_text.split()[0])
        
        gl = QGridLayout()
        ch_keys = list(self.available_channels.keys())
        
        for i in range(count):
            row = i // 2
            col = (i % 2) * 2
            gl.addWidget(QLabel(f"<b>Slot {i+1}:</b>"), row, col)
            cb = QComboBox()
            for k in ch_keys:
                cb.addItem(self.available_channels[k]['name'], k)
            # Set current or default
            if i < len(self.selected_layout):
                idx = ch_keys.index(self.selected_layout[i]) if self.selected_layout[i] in ch_keys else 0
                cb.setCurrentIndex(idx)
            else:
                cb.setCurrentIndex(i % len(ch_keys))
            gl.addWidget(cb, row, col + 1)
            self.slot_combos.append(cb)
            
        self.channels_layout.addLayout(gl)
        
    def _on_apply(self):
        self.selected_layout = [cb.currentData() for cb in self.slot_combos]
        self.accept()


# =============================================================================
# 5.35 INTERACTIVE LAP SELECTION DIALOG (MULTI-LAP OVERLAY & COMPARISON)
# =============================================================================

class LapSelectionDialog(QDialog):
    """
    Dialog providing multi-select checkboxes for laps recorded in the session,
    enabling custom overlay comparisons (e.g. Lap 1 vs Lap 3 vs Best Lap).
    """
    def __init__(self, laps_dict, best_lap_idx, lap_stats, initially_selected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selezione Giri per Telemetria & Confronto Overlay")
        self.resize(480, 420)
        self.setStyleSheet("background-color: #1e272e; color: #ecf0f1;")

        self.laps_dict = laps_dict
        self.best_lap_idx = best_lap_idx
        self.lap_stats = lap_stats
        self.selected_laps = set(initially_selected if initially_selected is not None else laps_dict.keys())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        lbl_desc = QLabel("<b>Seleziona i giri da visualizzare e sovrapporre nei grafici:</b>")
        lbl_desc.setStyleSheet("font-size: 12px; color: #00cec9; margin-bottom: 4px;")
        layout.addWidget(lbl_desc)

        # Quick selection buttons
        btn_bar = QHBoxLayout()
        btn_all = QPushButton("Tutti")
        btn_all.setStyleSheet("background-color: #2c3e50; color: white; padding: 4px 8px; font-weight: bold;")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Deseleziona")
        btn_none.setStyleSheet("background-color: #2c3e50; color: white; padding: 4px 8px;")
        btn_none.clicked.connect(self._deselect_all)
        btn_best_last = QPushButton("Best vs Ultimo")
        btn_best_last.setStyleSheet("background-color: #f39c12; color: black; font-weight: bold; padding: 4px 8px;")
        btn_best_last.clicked.connect(self._select_best_and_last)
        btn_invert = QPushButton("Inverti")
        btn_invert.setStyleSheet("background-color: #2c3e50; color: white; padding: 4px 8px;")
        btn_invert.clicked.connect(self._invert_selection)

        btn_bar.addWidget(btn_all)
        btn_bar.addWidget(btn_none)
        btn_bar.addWidget(btn_best_last)
        btn_bar.addWidget(btn_invert)
        layout.addLayout(btn_bar)

        # Scroll area with lap checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #485460; background-color: #242b35; }")
        container = QWidget()
        self.cb_layout = QVBoxLayout(container)
        self.checkboxes = {}

        for l_id in sorted(self.laps_dict.keys()):
            st = self.lap_stats.get(l_id, {})
            dt = st.get('dt', 0.0)
            dist = st.get('dist', 0.0)
            v_avg = st.get('v_avg', 0.0)
            is_best = (l_id == self.best_lap_idx)

            if is_best:
                tag = f"★ BEST LAP - Giro {l_id + 1}: {dt:.2f} s ({dist:.0f} m, media {v_avg:.1f} km/h)"
                style = "color: #f1c40f; font-weight: bold; font-size: 11px;"
            else:
                tag = f"Giro {l_id + 1}: {dt:.2f} s ({dist:.0f} m, media {v_avg:.1f} km/h)"
                style = "color: #ecf0f1; font-size: 11px;"

            cb = QCheckBox(tag)
            cb.setStyleSheet(f"QCheckBox {{ {style} padding: 4px; }} QCheckBox::indicator {{ width: 16px; height: 16px; }}")
            cb.setChecked(l_id in self.selected_laps)
            self.checkboxes[l_id] = cb
            self.cb_layout.addWidget(cb)

        self.cb_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Bottom action buttons
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        btn_cancel = QPushButton("Annulla")
        btn_cancel.setStyleSheet("background-color: #7f8c8d; color: white; padding: 5px 12px; font-weight: bold;")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Applica Selezione")
        btn_ok.setStyleSheet("background-color: #00cec9; color: black; padding: 5px 14px; font-weight: bold;")
        btn_ok.clicked.connect(self._on_apply)
        bottom_bar.addWidget(btn_cancel)
        bottom_bar.addWidget(btn_ok)
        layout.addLayout(bottom_bar)

    def _select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def _invert_selection(self):
        for cb in self.checkboxes.values():
            cb.setChecked(not cb.isChecked())

    def _select_best_and_last(self):
        last_id = sorted(self.laps_dict.keys())[-1] if self.laps_dict else 0
        for l_id, cb in self.checkboxes.items():
            cb.setChecked(l_id == self.best_lap_idx or l_id == last_id)

    def _on_apply(self):
        self.selected_laps = {l_id for l_id, cb in self.checkboxes.items() if cb.isChecked()}
        if not self.selected_laps:
            QMessageBox.warning(self, "Attenzione", "Seleziona almeno un giro prima di confermare.")
            return
        self.accept()


# =============================================================================
# 5.4 FULL TELEMETRY DATA VIEWER (SUSPENSION, CAMBER, FORCES, G-G, WEAR)
# =============================================================================

class FullTelemetryWindow(QDialog):
    """
    Comprehensive interactive engineering telemetry analysis window.
    Features:
    - Distance-based (s [m]) and Time-based (t [s]) horizontal axis.
    - Lap-by-Lap Comparison (Giro vs Giro Overlay, Best Lap, Last Lap).
    - Multi-plot grid & focused single plot views with smooth zoom/pan.
    - Fullscreen (F11) and pop-out analysis.
    """
    AVAILABLE_PLOTS = {
        'suspension_travel': {'name': "1. Escursioni Sospensioni z [mm]",     'title': "1. Escursioni Sospensioni z [mm]"},
        'camber':            {'name': "2. Campanatura Dinamica gamma [deg]",    'title': "2. Campanatura Dinamica gamma [deg]"},
        'toe':               {'name': "3. Convergenza Dinamica Toe [deg]",     'title': "3. Convergenza Dinamica Toe [deg]"},
        'vertical_loads':    {'name': "4. Carichi Verticali Fz (4 Ruote) [N]", 'title': "4. Carichi Verticali Fz (FL, FR, RL, RR) [N]"},
        'lateral_forces':    {'name': "5. Forze Laterali Fy (4 Ruote) [N]",    'title': "5. Forze Laterali Fy (FL, FR, RL, RR) [N]"},
        'slip_angles':       {'name': "6. Angoli Deriva Pneumatici alpha [deg]", 'title': "6. Angoli di Deriva Pneumatici alpha [deg]"},
        'slip_ratios':       {'name': "7. Rapporti Slittamento kappa [%]",     'title': "7. Slittamento Longitudinale kappa [%]"},
        'grip_util':         {'name': "8. Utilizzo Aderenza Pneumatici [%]",   'title': "8. Grip Utilization 4 Ruote [%]"},
        'gg_diagram':        {'name': "9. Diagramma G-G (Ay vs Ax) [1:1 Equal]", 'title': "9. Diagramma G-G (Ay vs Ax)"},
        'chassis_attitude':  {'name': "10. Assetto Cassa (Rollio, Beccheggio)", 'title': "10. Assetto Cassa (Roll, Pitch, Yaw)"},
        'chassis_velocities':{'name': "11. Velocita Chassis (vx, vy, Deriva beta)", 'title': "11. Velocita Chassis & Deriva beta"},
        'driver_inputs':     {'name': "12. Input Pilota (Sterzo, Gas, Freno)", 'title': "12. Input Pilota (Sterzo, Gas, Freno)"},
        'haptics':           {'name': "13. Feedback Aptico & Rumble [%]",      'title': "13. Feedback Aptico Grilletti & Rumble"},
        'roll_centers':      {'name': "14. Centri di Rollio Dinamici h_rc [mm]", 'title': "14. Centri di Rollio Dinamici h_rc [mm]"},
        'tire_temps_bulk':   {'name': "15. Temp. Gomme Bulk (4 Ruote) [°C]",   'title': "15. Temperature Gomme Bulk (FL, FR, RL, RR) [°C]"},
        'tire_temps_detailed': {'name': "16. Strati Termici 3-Zone (Surf, Tread, Carc) [°C]", 'title': "16. Strati Termici Gomme (3 Strati) [°C]"},
        'tire_wear_grip':    {'name': "17. Usura Pneumatici & Moltiplicatori Grip [%]", 'title': "17. Usura Gomme [%] & Grip Multiplier"},
        'wheel_torques':        {'name': "18. Coppie Ruote (Trazione e Frenata) [Nm]", 'title': "18. Coppie Ruote (Trazione e Frenata) [Nm]"},
        'longitudinal_forces':  {'name': "19. Forze Longitudinali Fx (4 Ruote) [N]", 'title': "19. Forze Longitudinali Fx (FL, FR, RL, RR) [N]"},
        'triaxial_forces_fl':   {'name': "20. Terna Forze Ruota FL (Fx, Fy, Fz, |F|) [N]", 'title': "20. Terna Forze Ruota FL (Fx, Fy, Fz, |F|) [N]"},
        'triaxial_forces_fr':   {'name': "21. Terna Forze Ruota FR (Fx, Fy, Fz, |F|) [N]", 'title': "21. Terna Forze Ruota FR (Fx, Fy, Fz, |F|) [N]"},
        'triaxial_forces_rl':   {'name': "22. Terna Forze Ruota RL (Fx, Fy, Fz, |F|) [N]", 'title': "22. Terna Forze Ruota RL (Fx, Fy, Fz, |F|) [N]"},
        'triaxial_forces_rr':   {'name': "23. Terna Forze Ruota RR (Fx, Fy, Fz, |F|) [N]", 'title': "23. Terna Forze Ruota RR (Fx, Fy, Fz, |F|) [N]"},
        'torque_vectoring':     {'name': "24. Torque Vectoring (Delta Mz, r_target)", 'title': "24. Active TV & Imbardata (Delta Mz [Nm], r_target vs r [deg/s])"}
    }
    
    PRESETS = {
        "1. Setup & Dinamica Veicolo": ['suspension_travel', 'camber', 'vertical_loads', 'gg_diagram', 'haptics', 'driver_inputs'],
        "2. Forze Ruote 4 Angoli (Fy, Fx, Fz, Coppie)": ['lateral_forces', 'longitudinal_forces', 'vertical_loads', 'wheel_torques', 'slip_angles', 'slip_ratios'],
        "3. Pneumatici & Aderenza 4 Ruote": ['lateral_forces', 'slip_angles', 'vertical_loads', 'grip_util', 'slip_ratios', 'gg_diagram'],
        "4. Geometria, Assetto & Cassa": ['suspension_travel', 'camber', 'toe', 'roll_centers', 'chassis_attitude', 'chassis_velocities'],
        "5. Guida & Performance Pilota": ['driver_inputs', 'gg_diagram', 'chassis_velocities', 'haptics', 'grip_util', 'chassis_attitude'],
        "6. Termica & Degrado Pneumatici": ['tire_temps_bulk', 'tire_temps_detailed', 'tire_wear_grip', 'grip_util', 'lateral_forces', 'gg_diagram'],
        "7. Carichi Sospensioni & Forze Triassiali (FEA)": ['longitudinal_forces', 'lateral_forces', 'vertical_loads', 'wheel_torques', 'triaxial_forces_fl', 'triaxial_forces_rl']
    }

    LAP_COLORS = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c', '#fdcb6e']

    PLOT_CONFIGS = {
        'suspension_travel': {
            'title': "1. Escursioni Sospensioni z [mm]", 'ylabel': "Corsa [mm]",
            'signals': [('z_fl_mm', 'z_FL [mm]', '#0984e3', '-'), ('z_fr_mm', 'z_FR [mm]', '#00cec9', '-'),
                        ('z_rl_mm', 'z_RL [mm]', '#e17055', '-'), ('z_rr_mm', 'z_RR [mm]', '#d63031', '-')],
            'overlay_primary': ('z_fl_mm', 'z_FL')
        },
        'camber': {
            'title': "2. Campanatura Dinamica gamma [deg]", 'ylabel': "Camber [deg]",
            'signals': [('camber_fl_deg', 'Camber FL [deg]', '#0984e3', '-'), ('camber_fr_deg', 'Camber FR [deg]', '#00cec9', '-'),
                        ('camber_rl_deg', 'Camber RL [deg]', '#e17055', '-'), ('camber_rr_deg', 'Camber RR [deg]', '#d63031', '-')],
            'overlay_primary': ('camber_fl_deg', 'Camber FL')
        },
        'toe': {
            'title': "3. Convergenza Dinamica Toe [deg]", 'ylabel': "Toe [deg]",
            'signals': [('toe_fl_deg', 'Toe FL [deg]', '#0984e3', '-'), ('toe_fr_deg', 'Toe FR [deg]', '#00cec9', '-'),
                        ('toe_rl_deg', 'Toe RL [deg]', '#e17055', '-'), ('toe_rr_deg', 'Toe RR [deg]', '#d63031', '-')],
            'overlay_primary': ('toe_fl_deg', 'Toe FL')
        },
        'vertical_loads': {
            'title': "4. Carichi Verticali Fz (4 Ruote) [N]", 'ylabel': "Carico Fz [N]",
            'signals': [('Fz_fl', 'Fz FL [N]', '#00cec9', '-'), ('Fz_fr', 'Fz FR [N]', '#0984e3', '-'),
                        ('Fz_rl', 'Fz RL [N]', '#e17055', '-'), ('Fz_rr', 'Fz RR [N]', '#d63031', '-'),
                        ('dFz_jack_N', 'Jacking dFz [N]', '#f1c40f', '--')],
            'overlay_primary': ('Fz_fl', 'Fz FL')
        },
        'lateral_forces': {
            'title': "5. Forze Laterali Fy (4 Ruote) [N]", 'ylabel': "Forza Fy [N]",
            'signals': [('Fy_fl', 'Fy FL [N]', '#00cec9', '-'), ('Fy_fr', 'Fy FR [N]', '#0984e3', '-'),
                        ('Fy_rl', 'Fy RL [N]', '#e17055', '-'), ('Fy_rr', 'Fy RR [N]', '#d63031', '-')],
            'overlay_primary': ('Fy_fl', 'Fy FL')
        },
        'slip_angles': {
            'title': "6. Angoli di Deriva Pneumatici alpha [deg]", 'ylabel': "Deriva alpha [deg]",
            'signals': [('alpha_fl', 'alpha FL [deg]', '#0984e3', '-'), ('alpha_fr', 'alpha FR [deg]', '#00cec9', '-'),
                        ('alpha_rl', 'alpha RL [deg]', '#e17055', '-'), ('alpha_rr', 'alpha RR [deg]', '#d63031', '-')],
            'overlay_primary': ('alpha_fl', 'alpha FL')
        },
        'slip_ratios': {
            'title': "7. Slittamento Longitudinale kappa [%]", 'ylabel': "Slittamento kappa [%]",
            'signals': [('kappa_fl', 'kappa FL [%]', '#0984e3', '-'), ('kappa_fr', 'kappa FR [%]', '#00cec9', '-'),
                        ('kappa_rl', 'kappa RL [%]', '#e17055', '-'), ('kappa_rr', 'kappa RR [%]', '#d63031', '-')],
            'overlay_primary': ('kappa_fl', 'kappa FL')
        },
        'grip_util': {
            'title': "8. Grip Utilization 4 Ruote [%]", 'ylabel': "Utilizzo Aderenza [%]",
            'signals': [('util_fl', 'Util FL [%]', '#0984e3', '-'), ('util_fr', 'Util FR [%]', '#00cec9', '-'),
                        ('util_rl', 'Util RL [%]', '#e17055', '-'), ('util_rr', 'Util RR [%]', '#d63031', '-')],
            'overlay_primary': ('util_fl', 'Util FL')
        },
        'chassis_attitude': {
            'title': "10. Assetto Cassa (Roll, Pitch, Yaw)", 'ylabel': "Angolo [deg] / [deg/s]",
            'signals': [('phi_deg', 'Rollio phi [deg]', '#e74c3c', '-'), ('theta_deg', 'Beccheggio theta [deg]', '#f39c12', '-'),
                        ('r_deg_s', 'Yaw Rate r [deg/s]', '#9b59b6', '--')],
            'overlay_primary': ('phi_deg', 'Rollio phi')
        },
        'chassis_velocities': {
            'title': "11. Velocita Chassis & Deriva beta", 'ylabel': "Velocita [km/h] / beta [deg]",
            'signals': [('vx_kmh', 'vx [km/h]', '#2ecc71', '-'), ('vy_kmh', 'vy [km/h]', '#3498db', '-'),
                        ('beta_deg', 'Deriva beta [deg]', '#e67e22', '--')],
            'overlay_primary': ('vx_kmh', 'vx')
        },
        'driver_inputs': {
            'title': "12. Input Pilota (Sterzo, Gas, Freno)", 'ylabel': "Comando [% / deg]",
            'signals': [('steer_deg', 'Sterzo delta [deg]', '#00cec9', '-'), ('throttle_pct', 'Gas [%]', '#2ecc71', '-'),
                        ('brake_pct', 'Freno [%]', '#e74c3c', '-')],
            'overlay_primary': ('steer_deg', 'Sterzo delta')
        },
        'haptics': {
            'title': "13. Feedback Aptico Grilletti & Rumble", 'ylabel': "Intensita [%]",
            'signals': [('haptic_throt_rt', 'Haptic Gas RT [%]', '#f1c40f', '-'), ('haptic_brake_lt', 'Haptic Freno LT [%]', '#e74c3c', '-'),
                        ('rumble_heavy', 'Rumble Telaio [%]', '#9b59b6', ':'), ('rumble_light', 'Rumble Gomme [%]', '#1abc9c', ':')],
            'overlay_primary': ('rumble_heavy', 'Rumble')
        },
        'roll_centers': {
            'title': "14. Centri di Rollio Dinamici (Altezza & Migrazione Lat.) [mm]", 'ylabel': "Posizione RC [mm]",
            'signals': [
                ('h_rc_f_mm', 'h_RC Ant. [mm]', '#0984e3', '-'),
                ('h_rc_r_mm', 'h_RC Post. [mm]', '#e17055', '-'),
                ('y_rc_f_mm', 'y_RC Lat Ant. [mm]', '#74b9ff', '--'),
                ('y_rc_r_mm', 'y_RC Lat Post. [mm]', '#fab1a0', '--')
            ],
            'overlay_primary': ('h_rc_f_mm', 'h_RC Ant')
        },
        'tire_temps_bulk': {
            'title': "15. Temperature Gomme Bulk (4 Ruote) [°C]", 'ylabel': "Temp [°C]",
            'signals': [('tire_T_bulk_fl', 'T_bulk FL [°C]', '#0984e3', '-'), ('tire_T_bulk_fr', 'T_bulk FR [°C]', '#00cec9', '-'),
                        ('tire_T_bulk_rl', 'T_bulk RL [°C]', '#e84393', '-'), ('tire_T_bulk_rr', 'T_bulk RR [°C]', '#e17055', '-')],
            'overlay_primary': ('tire_T_bulk_fl', 'T_bulk FL')
        },
        'tire_temps_detailed': {
            'title': "16. Strati Termici 3-Zone (Surf, Tread, Carc) [°C]", 'ylabel': "Temp [°C]",
            'signals': [('tire_T_surf_fl', 'FL Superficie [°C]', '#e74c3c', '-'), ('tire_T_tread_fl', 'FL Battistrada [°C]', '#e67e22', '--'),
                        ('tire_T_carc_fl', 'FL Carcassa [°C]', '#3498db', ':'), ('tire_T_surf_rr', 'RR Superficie [°C]', '#9b59b6', '-')],
            'overlay_primary': ('tire_T_surf_fl', 'T_surf FL')
        },
        'tire_wear_grip': {
            'title': "17. Usura Gomme [%] & Grip Multiplier", 'ylabel': "Valore [%]",
            'signals': [('tire_grip_fl_pct', 'Grip FL [%]', '#2ecc71', '-'), ('tire_grip_rr_pct', 'Grip RR [%]', '#27ae60', '-'),
                        ('tire_wear_fl', 'Usura FL [%]', '#e74c3c', '--'), ('tire_wear_rr', 'Usura RR [%]', '#e67e22', '--')],
            'overlay_primary': ('tire_grip_fl_pct', 'Grip FL')
        },
        'wheel_torques': {
            'title': "18. Coppie Ruote (Trazione e Frenata) [Nm]", 'ylabel': "Coppia [Nm]",
            'signals': [('T_drive_fl', 'T_drv FL [Nm]', '#00cec9', '-'),
                        ('T_drive_fr', 'T_drv FR [Nm]', '#0984e3', '-'),
                        ('T_drive_rl', 'T_drv RL [Nm]', '#74b9ff', '-'),
                        ('T_drive_rr', 'T_drv RR [Nm]', '#a29bfe', '-'),
                        ('T_brake_fl', 'T_brk FL [Nm]', '#e74c3c', '--'),
                        ('T_brake_fr', 'T_brk FR [Nm]', '#e17055', '--'),
                        ('T_brake_rl', 'T_brk RL [Nm]', '#d63031', ':'),
                        ('T_brake_rr', 'T_brk RR [Nm]', '#c0392b', ':')],
            'overlay_primary': ('T_brake_fl', 'T_brk FL')
        },
        'longitudinal_forces': {
            'title': "19. Forze Longitudinali Fx (4 Ruote) [N]", 'ylabel': "Forza Longitudinale Fx [N]",
            'signals': [('Fx_fl', 'Fx FL [N]', '#00cec9', '-'), ('Fx_fr', 'Fx FR [N]', '#0984e3', '-'),
                        ('Fx_rl', 'Fx RL [N]', '#e17055', '-'), ('Fx_rr', 'Fx RR [N]', '#d63031', '-')],
            'overlay_primary': ('Fx_rl', 'Fx RL')
        },
        'triaxial_forces_fl': {
            'title': "20. Terna Forze Ruota FL (Fx, Fy, Fz, |F|) [N]", 'ylabel': "Forza [N]",
            'signals': [('Fx_fl', 'Fx FL (Long) [N]', '#3498db', '-'),
                        ('Fy_fl', 'Fy FL (Lat) [N]', '#e74c3c', '-'),
                        ('Fz_fl', 'Fz FL (Vert) [N]', '#2ecc71', '-'),
                        ('F_tot_fl', '|F| FL (Risultante) [N]', '#f1c40f', '--')],
            'overlay_primary': ('F_tot_fl', '|F| FL')
        },
        'triaxial_forces_fr': {
            'title': "21. Terna Forze Ruota FR (Fx, Fy, Fz, |F|) [N]", 'ylabel': "Forza [N]",
            'signals': [('Fx_fr', 'Fx FR (Long) [N]', '#3498db', '-'),
                        ('Fy_fr', 'Fy FR (Lat) [N]', '#e74c3c', '-'),
                        ('Fz_fr', 'Fz FR (Vert) [N]', '#2ecc71', '-'),
                        ('F_tot_fr', '|F| FR (Risultante) [N]', '#f1c40f', '--')],
            'overlay_primary': ('F_tot_fr', '|F| FR')
        },
        'triaxial_forces_rl': {
            'title': "22. Terna Forze Ruota RL (Fx, Fy, Fz, |F|) [N]", 'ylabel': "Forza [N]",
            'signals': [('Fx_rl', 'Fx RL (Long) [N]', '#3498db', '-'),
                        ('Fy_rl', 'Fy RL (Lat) [N]', '#e74c3c', '-'),
                        ('Fz_rl', 'Fz RL (Vert) [N]', '#2ecc71', '-'),
                        ('F_tot_rl', '|F| RL (Risultante) [N]', '#f1c40f', '--')],
            'overlay_primary': ('F_tot_rl', '|F| RL')
        },
        'triaxial_forces_rr': {
            'title': "23. Terna Forze Ruota RR (Fx, Fy, Fz, |F|) [N]", 'ylabel': "Forza [N]",
            'signals': [('Fx_rr', 'Fx RR (Long) [N]', '#3498db', '-'),
                        ('Fy_rr', 'Fy RR (Lat) [N]', '#e74c3c', '-'),
                        ('Fz_rr', 'Fz RR (Vert) [N]', '#2ecc71', '-'),
                        ('F_tot_rr', '|F| RR (Risultante) [N]', '#f1c40f', '--')],
            'overlay_primary': ('F_tot_rr', '|F| RR')
        },
        'torque_vectoring': {
            'title': "24. Active Torque Vectoring & Differenziale", 'ylabel': "Delta Mz [Nm] / r [deg/s]",
            'signals': [('Delta_Mz_Nm', 'Delta Mz [Nm]', '#9b59b6', '-'),
                        ('r_target_degs', 'r_target [deg/s]', '#2ecc71', '--'),
                        ('r_deg_s', 'r_effettiva [deg/s]', '#e74c3c', '-'),
                        ('r_error_degs', 'r_error [deg/s]', '#f39c12', ':')],
            'overlay_primary': ('Delta_Mz_Nm', 'Delta Mz')
        }
    }

    def __init__(self, history_data, parent=None, initial_preset=None):
        super().__init__(parent)
        self.setWindowTitle("Analisi Telemetria 14-DOF Completa (Distanza, Tempo & Giro vs Giro)")
        self.resize(1260, 820)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet("background-color: #1e272e; color: #ecf0f1;")
        
        self.history_data = history_data or []
        self.df = pd.DataFrame(self.history_data) if len(self.history_data) > 0 else pd.DataFrame()
        self.t = self.df.get('t', np.arange(len(self.df))) if not self.df.empty else np.array([])
        
        def_preset = initial_preset if (initial_preset and initial_preset in self.PRESETS) else "1. Setup & Dinamica Veicolo"
        self.current_layout_keys = list(self.PRESETS[def_preset])
        self.custom_layout_keys = list(self.current_layout_keys)
        
        # Extract laps and stats
        self.laps, self.best_lap_idx, self.lap_stats = self._extract_laps()
        self.selected_laps = set(self.laps.keys())
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # --- Top Action Bar ---
        top_bar = QHBoxLayout()
        
        top_bar.addWidget(QLabel("<b>Layout:</b>"))
        self.combo_preset = QComboBox()
        for p_name in self.PRESETS.keys():
            self.combo_preset.addItem(p_name)
        self.combo_preset.addItem("Layout Personalizzato...")
        self.combo_preset.setStyleSheet("font-size: 11px; padding: 3px 8px; background-color: #2c3e50; color: white;")
        if def_preset in self.PRESETS:
            self.combo_preset.setCurrentText(def_preset)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        top_bar.addWidget(self.combo_preset)
        
        self.btn_config_layout = QPushButton("Configura Layout...")
        self.btn_config_layout.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_config_layout.clicked.connect(self._open_custom_layout_dialog)
        top_bar.addWidget(self.btn_config_layout)
        
        top_bar.addSpacing(15)
        top_bar.addWidget(QLabel("<b>Focus:</b>"))
        self.combo_focus = QComboBox()
        self.combo_focus.addItem("Tutti i Grafici (Griglia)")
        for k, info in self.AVAILABLE_PLOTS.items():
            self.combo_focus.addItem(info['name'], k)
        self.combo_focus.setStyleSheet("font-size: 11px; padding: 3px 8px; background-color: #2c3e50; color: white;")
        self.combo_focus.currentIndexChanged.connect(self._on_focus_changed)
        top_bar.addWidget(self.combo_focus)
        
        self.btn_popout_selected = QPushButton("Pop-out Ingrandito")
        self.btn_popout_selected.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_popout_selected.clicked.connect(self._popout_current_plot)
        top_bar.addWidget(self.btn_popout_selected)
        
        top_bar.addSpacing(10)
        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        btn_reset_zoom.clicked.connect(self._reset_zoom)
        top_bar.addWidget(btn_reset_zoom)

        self.btn_z1 = QPushButton("Finestra Breve")
        self.btn_z1.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        self.btn_z1.clicked.connect(lambda: self._set_window_span(1))
        top_bar.addWidget(self.btn_z1)

        self.btn_z2 = QPushButton("Finestra Media")
        self.btn_z2.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        self.btn_z2.clicked.connect(lambda: self._set_window_span(2))
        top_bar.addWidget(self.btn_z2)
        
        top_bar.addStretch()

        # Load Telemetry File Button
        self.btn_load_file = QPushButton("Carica Telemetria...")
        self.btn_load_file.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #27ae60; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_load_file.setToolTip("Carica una sessione salvata in formato CSV o JSON per visualizzarla nei grafici")
        self.btn_load_file.clicked.connect(self._load_telemetry_file)
        top_bar.addWidget(self.btn_load_file)

        # Data Export Dropdown Button
        self.btn_export_menu = QToolButton()
        self.btn_export_menu.setText("Esporta Dati...")
        self.btn_export_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_export_menu.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #e67e22; color: white; padding: 4px 10px; border-radius: 3px;")
        export_qmenu = QMenu(self.btn_export_menu)
        act_csv_all = export_qmenu.addAction("Esporta CSV Completo (Tutti i Giri)")
        act_csv_all.triggered.connect(lambda: self._export_csv(selected_only=False))
        act_csv_sel = export_qmenu.addAction("Esporta CSV Giri Selezionati (Overlay MoTeC)")
        act_csv_sel.triggered.connect(lambda: self._export_csv(selected_only=True))
        export_qmenu.addSeparator()
        act_json = export_qmenu.addAction("Esporta JSON Dati Sessione")
        act_json.triggered.connect(self._export_json)
        act_fig = export_qmenu.addAction("Salva Grafico Attuale (PNG / PDF 300 DPI)")
        act_fig.triggered.connect(self._export_figure)
        self.btn_export_menu.setMenu(export_qmenu)
        top_bar.addWidget(self.btn_export_menu)
        
        self.btn_fullscreen = QPushButton("Fullscreen (F11)")
        self.btn_fullscreen.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #00b894; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)
        
        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet("font-size: 11px; padding: 4px 10px; background-color: #7f8c8d; color: white; border-radius: 3px;")
        btn_close.clicked.connect(self.accept)
        top_bar.addWidget(btn_close)
        
        main_layout.addLayout(top_bar)

        # --- Secondary Action Bar: Asse X & Analisi Giri ---
        analysis_bar = QHBoxLayout()
        analysis_bar.addWidget(QLabel("<b>Asse X:</b>"))
        self.combo_x_axis = QComboBox()
        self.combo_x_axis.addItem("Distanza Pista s [m]", "s_lap")
        self.combo_x_axis.addItem("Tempo t [s]", "t")
        self.combo_x_axis.addItem("Distanza Totale [m]", "dist_total")
        self.combo_x_axis.setStyleSheet("font-size: 11px; font-weight: bold; padding: 3px 8px; background-color: #2c3e50; color: #00cec9;")
        self.combo_x_axis.currentIndexChanged.connect(self._on_analysis_mode_changed)
        analysis_bar.addWidget(self.combo_x_axis)

        analysis_bar.addSpacing(15)
        analysis_bar.addWidget(QLabel("<b>Modalità Giri:</b>"))
        self.combo_lap_filter = QComboBox()
        self.combo_lap_filter.addItem("Tutti i Dati (Sequenziale)", "all")
        self.combo_lap_filter.addItem("Confronto Giri Selezionati (Overlay)", "overlay_selected")
        self.combo_lap_filter.addItem("Tutti i Giri (Overlay Completo)", "overlay")
        self.combo_lap_filter.addItem("Solo Best Lap", "best")
        self.combo_lap_filter.addItem("Solo Ultimo Giro", "last")
        self.combo_lap_filter.setStyleSheet("font-size: 11px; font-weight: bold; padding: 3px 8px; background-color: #2c3e50; color: #f1c40f;")
        self.combo_lap_filter.currentIndexChanged.connect(self._on_analysis_mode_changed)
        analysis_bar.addWidget(self.combo_lap_filter)

        self.btn_select_laps = QPushButton("Seleziona Giri...")
        self.btn_select_laps.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 3px 10px; border-radius: 3px;")
        self.btn_select_laps.clicked.connect(self._open_lap_selection_dialog)
        analysis_bar.addWidget(self.btn_select_laps)

        self.lbl_lap_stats = QLabel("")
        self.lbl_lap_stats.setStyleSheet("font-size: 11px; color: #f39c12; font-style: italic; padding-left: 10px;")
        analysis_bar.addWidget(self.lbl_lap_stats)

        analysis_bar.addStretch()
        main_layout.addLayout(analysis_bar)
        
        # --- Data Inspector (Lateral Tree Panel) & Canvas Splitter ---
        self.disabled_signals = set()
        self._updating_channel_tree = False
        
        self.splitter_telemetry = QSplitter(Qt.Horizontal)
        
        # Lateral Inspector Panel
        inspector_widget = QWidget()
        insp_layout = QVBoxLayout(inspector_widget)
        insp_layout.setContentsMargins(0, 0, 0, 0)
        insp_layout.setSpacing(4)
        
        lbl_insp = QLabel("<b>Data Inspector (Canali)</b>")
        lbl_insp.setStyleSheet("font-size: 11px; color: #00cec9; padding: 2px;")
        insp_layout.addWidget(lbl_insp)
        
        btn_bar = QHBoxLayout()
        btn_all = QPushButton("Tutti")
        btn_all.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #34495e; color: white; border-radius: 2px;")
        btn_all.clicked.connect(self._select_all_channels)
        btn_none = QPushButton("Nessuno")
        btn_none.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #34495e; color: white; border-radius: 2px;")
        btn_none.clicked.connect(self._deselect_all_channels)
        btn_reset_ch = QPushButton("Reset")
        btn_reset_ch.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #34495e; color: white; border-radius: 2px;")
        btn_reset_ch.clicked.connect(self._reset_channel_selection)
        btn_bar.addWidget(btn_all)
        btn_bar.addWidget(btn_none)
        btn_bar.addWidget(btn_reset_ch)
        insp_layout.addLayout(btn_bar)
        
        self.tree_channels = QTreeWidget()
        self.tree_channels.setHeaderHidden(True)
        self.tree_channels.setStyleSheet("font-size: 10px; background-color: #1a252f; color: #ecf0f1;")
        self.tree_channels.itemChanged.connect(self._on_channel_item_changed)
        insp_layout.addWidget(self.tree_channels)
        
        self.splitter_telemetry.addWidget(inspector_widget)
        
        # Canvas & Toolbar Container
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig = Figure(figsize=(12, 7.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        self.splitter_telemetry.addWidget(plot_container)
        self.splitter_telemetry.setSizes([260, 1000])
        main_layout.addWidget(self.splitter_telemetry)
        
        # Connect interactive mouse wheel zoom
        self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll_zoom)
        
        self._is_fullscreen = False
        self._update_lap_stats_label()
        self._update_channel_tree()
        self._render_current_view()

    def _extract_laps(self):
        """Groups telemetry points by lap, returns dictionary {lap_idx: sub_df}, best_lap_idx, and lap_stats."""
        if self.df.empty:
            return {}, None, {}
            
        laps = {}
        lap_col = 'lap_idx' if 'lap_idx' in self.df.columns else ('lap_count' if 'lap_count' in self.df.columns else None)
        if lap_col:
            for l_id, g in self.df.groupby(lap_col):
                if len(g) >= 5:
                    laps[int(l_id)] = g.copy()
        else:
            laps[0] = self.df.copy()
            
        if not laps:
            laps[0] = self.df.copy()

        lap_stats = {}
        best_id = None
        best_time = 1e9
        
        for l_id, g in sorted(laps.items()):
            dt = float(g['t'].iloc[-1] - g['t'].iloc[0]) if 't' in g.columns and len(g) > 1 else 0.0
            dist = float(g['dist_total'].iloc[-1] - g['dist_total'].iloc[0]) if 'dist_total' in g.columns and len(g) > 1 else 0.0
            v_avg = float(g['vx_kmh'].mean()) if 'vx_kmh' in g.columns else 0.0
            rec_lap_time = float(g['lap_time_last'].iloc[-1]) if 'lap_time_last' in g.columns and g['lap_time_last'].iloc[-1] > 0 else dt
            lap_stats[l_id] = {'dt': dt, 'dist': dist, 'v_avg': v_avg, 'rec_time': rec_lap_time}
            
            if (dist > 50.0 or dt > 5.0) and dt < best_time:
                best_time = dt
                best_id = l_id
                
        if best_id is None:
            best_id = list(laps.keys())[0]
            
        return laps, best_id, lap_stats

    def _update_lap_stats_label(self):
        n_laps = len(self.laps)
        n_sel = len(self.selected_laps)
        if n_laps <= 1:
            self.lbl_lap_stats.setText("1 giro registrato (completa ulteriori giri sul circuito per il confronto overlay)")
        else:
            best_info = self.lap_stats.get(self.best_lap_idx, {})
            best_t = best_info.get('dt', 0.0)
            self.lbl_lap_stats.setText(f"Giri totali: {n_laps} | Selezionati: {n_sel} | Best Lap: Giro {self.best_lap_idx + 1} ({best_t:.2f} s)")

    def _open_lap_selection_dialog(self):
        dlg = LapSelectionDialog(self.laps, self.best_lap_idx, self.lap_stats, initially_selected=self.selected_laps, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.selected_laps = set(dlg.selected_laps)
            # Switch to overlay_selected automatically
            idx = self.combo_lap_filter.findData("overlay_selected")
            if idx >= 0:
                self.combo_lap_filter.blockSignals(True)
                self.combo_lap_filter.setCurrentIndex(idx)
                self.combo_lap_filter.blockSignals(False)
            self._update_lap_stats_label()
            self._render_current_view()

    def _export_csv(self, selected_only=False):
        try:
            if selected_only:
                laps_to_export = [self.laps[lid] for lid in sorted(self.selected_laps) if lid in self.laps]
                if not laps_to_export:
                    QMessageBox.warning(self, "Esporta CSV", "Nessun giro selezionato per l'esportazione.")
                    return
                df_exp = pd.concat(laps_to_export, ignore_index=True)
                default_name = f"telemetry_selected_{len(laps_to_export)}_laps.csv"
            else:
                df_exp = self.df.copy()
                default_name = "telemetry_all_session.csv"

            if df_exp.empty:
                QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria disponibile.")
                return

            path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria CSV", default_name, "CSV Files (*.csv);;All Files (*)")
            if path:
                df_exp.to_csv(path, index=False)
                QMessageBox.information(self, "Esportazione Riuscita",
                                        f"Dati esportati con successo!\n\nFile: {path}\nRighe: {len(df_exp)} | Canali: {len(df_exp.columns)}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Esportazione", f"Impossibile esportare in CSV:\n{e}\n\n{traceback.format_exc()}")

    def _export_json(self):
        try:
            if self.df.empty:
                QMessageBox.warning(self, "Esporta JSON", "Nessun dato di telemetria disponibile.")
                return

            path, _ = QFileDialog.getSaveFileName(self, "Salva Sessione JSON", "telemetry_session.json", "JSON Files (*.json);;All Files (*)")
            if path:
                export_dict = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_points": len(self.df),
                    "total_laps": len(self.laps),
                    "best_lap_idx": self.best_lap_idx,
                    "lap_stats": {int(k): v for k, v in self.lap_stats.items()},
                    "selected_laps": list(self.selected_laps),
                    "data": self.df.to_dict(orient="records")
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(export_dict, f, indent=2)
                QMessageBox.information(self, "Esportazione Riuscita",
                                        f"Sessione esportata in JSON con successo!\n\nFile: {path}\nCampioni: {len(self.df)}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Esportazione", f"Impossibile esportare in JSON:\n{e}\n\n{traceback.format_exc()}")

    def _export_figure(self):
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Salva Grafico Attuale", "telemetry_plot_300dpi.png",
                                                  "PNG Image (*.png);;PDF Vector (*.pdf);;SVG Vector (*.svg)")
            if path:
                self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=self.fig.get_facecolor())
                QMessageBox.information(self, "Grafico Salvato", f"Grafico salvato in alta definizione (300 DPI):\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio Grafico", f"Impossibile salvare il grafico:\n{e}\n\n{traceback.format_exc()}")

    def _load_telemetry_file(self):
        """Loads a telemetry dataset from CSV or JSON directly into this telemetry viewer."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Carica Telemetria Salvata in Grafici",
            "",
            "File Telemetria (*.csv *.json);;File CSV (*.csv);;File JSON (*.json);;Tutti i File (*.*)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                df = pd.DataFrame(raw)
            else:
                df = pd.read_csv(path)

            if df.empty:
                QMessageBox.warning(self, "File Vuoto", "Il file selezionato non contiene record di dati validi.")
                return

            self.df = df
            self.history_data = df.to_dict('records')
            self.t = self.df.get('t', np.arange(len(self.df)))
            self.laps, self.best_lap_idx, self.lap_stats = self._extract_laps()
            self.selected_laps = set(self.laps.keys())
            self.setWindowTitle(f"Analisi Telemetria 14-DOF: {os.path.basename(path)} ({len(df)} campioni)")
            self._update_lap_checkboxes()
            self._update_lap_stats_label()
            self._render_current_view()
            QMessageBox.information(
                self,
                "Telemetria Caricata",
                f"Caricati con successo {len(df)} campioni e {len(df.columns)} canali da:\n{os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore Caricamento", f"Impossibile caricare il file di telemetria:\n{e}\n\n{traceback.format_exc()}")

    def _on_analysis_mode_changed(self):
        self._update_lap_stats_label()
        self._render_current_view()

    def _on_mouse_scroll_zoom(self, event):
        """Smoothly zooms the X-axis (and Y-axis for G-G) around mouse cursor."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 0.8 if event.button == 'up' else 1.25
        
        cur_xlim = ax.get_xlim()
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_x = (event.xdata - cur_xlim[0]) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
        ax.set_xlim([event.xdata - new_w * rel_x, event.xdata + new_w * (1.0 - rel_x)])
        
        if getattr(ax, 'name', '') == 'rectilinear' and ax.get_aspect() == 1.0:
            cur_ylim = ax.get_ylim()
            new_h = (cur_ylim[1] - cur_ylim[0]) * scale
            rel_y = (event.ydata - cur_ylim[0]) / max(cur_ylim[1] - cur_ylim[0], 1e-6)
            ax.set_ylim([event.ydata - new_h * rel_y, event.ydata + new_h * (1.0 - rel_y)])
            
        self.canvas.draw_idle()

    def _reset_zoom(self):
        for ax in self.fig.axes:
            ax.autoscale()
            if hasattr(ax, '_is_gg') and ax._is_gg:
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.set_aspect('equal', 'box')
        self.canvas.draw_idle()

    def _set_window_span(self, mode_idx):
        x_mode = self.combo_x_axis.currentData() if hasattr(self, 'combo_x_axis') else 's_lap'
        if x_mode == 's_lap':
            span = 250.0 if mode_idx == 1 else 600.0
        elif x_mode == 'dist_total':
            span = 500.0 if mode_idx == 1 else 1500.0
        else:
            span = 10.0 if mode_idx == 1 else 30.0

        for ax in self.fig.axes:
            if not (hasattr(ax, '_is_gg') and ax._is_gg):
                lines = ax.get_lines()
                if lines:
                    xdata = lines[0].get_xdata()
                    if len(xdata) > 0:
                        max_x = xdata[-1]
                        min_x = max(0.0, max_x - span)
                        ax.set_xlim(min_x, max_x)
        self.canvas.draw_idle()
        
    def _on_focus_changed(self, idx):
        self._update_channel_tree()
        self._render_current_view()

    def _on_preset_changed(self, index):
        p_name = self.combo_preset.currentText()
        if p_name in self.PRESETS:
            self.current_layout_keys = list(self.PRESETS[p_name])
            self.combo_focus.setCurrentIndex(0)
            self._update_channel_tree()
            self._render_current_view()
        else:
            self._open_custom_layout_dialog()

    def _open_custom_layout_dialog(self):
        dlg = CustomTelemetryLayoutDialog(self.custom_layout_keys, self.AVAILABLE_PLOTS, self)
        if dlg.exec_() == QDialog.Accepted:
            self.custom_layout_keys = list(dlg.selected_layout)
            self.current_layout_keys = list(self.custom_layout_keys)
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(self.combo_preset.count() - 1)
            self.combo_preset.blockSignals(False)
            self.combo_focus.setCurrentIndex(0)
            self._update_channel_tree()
            self._render_current_view()

    def _update_channel_tree(self):
        """Rebuilds the Data Inspector tree with active subplots and individual signal checkboxes."""
        if not hasattr(self, 'tree_channels') or self.tree_channels is None:
            return
            
        self._updating_channel_tree = True
        self.tree_channels.blockSignals(True)
        self.tree_channels.clear()
        
        focus_idx = self.combo_focus.currentIndex() if hasattr(self, 'combo_focus') else 0
        if focus_idx == 0:
            active_keys = self.current_layout_keys
        else:
            focus_key = self.combo_focus.currentData()
            active_keys = [focus_key] if focus_key in self.AVAILABLE_PLOTS else self.current_layout_keys
            
        for k in active_keys:
            cfg = self.PLOT_CONFIGS.get(k)
            if not cfg and k == 'gg_diagram':
                # Special G-G plot item
                plot_item = QTreeWidgetItem([self.AVAILABLE_PLOTS.get(k, {}).get('name', '9. Diagramma G-G')])
                plot_item.setFlags(plot_item.flags() | Qt.ItemIsUserCheckable)
                plot_item.setData(0, Qt.UserRole, ('plot', k))
                is_on = (k, 'gg') not in self.disabled_signals
                plot_item.setCheckState(0, Qt.Checked if is_on else Qt.Unchecked)
                self.tree_channels.addTopLevelItem(plot_item)
                continue
            elif not cfg:
                continue
                
            plot_item = QTreeWidgetItem([cfg.get('title', k)])
            plot_item.setFlags(plot_item.flags() | Qt.ItemIsUserCheckable)
            plot_item.setData(0, Qt.UserRole, ('plot', k))
            
            signals = cfg.get('signals', [])
            all_checked = True
            any_checked = False
            for col, lbl, color, ls in signals:
                sig_item = QTreeWidgetItem([lbl])
                sig_item.setFlags(sig_item.flags() | Qt.ItemIsUserCheckable)
                sig_item.setData(0, Qt.UserRole, ('signal', k, col))
                is_sig_on = (k, col) not in self.disabled_signals
                sig_item.setCheckState(0, Qt.Checked if is_sig_on else Qt.Unchecked)
                if is_sig_on:
                    any_checked = True
                else:
                    all_checked = False
                plot_item.addChild(sig_item)
                
            if all_checked:
                plot_item.setCheckState(0, Qt.Checked)
            elif any_checked:
                plot_item.setCheckState(0, Qt.PartiallyChecked)
            else:
                plot_item.setCheckState(0, Qt.Unchecked)
                
            self.tree_channels.addTopLevelItem(plot_item)
            plot_item.setExpanded(True)
            
        self.tree_channels.blockSignals(False)
        self._updating_channel_tree = False

    def _on_channel_item_changed(self, item, column):
        if getattr(self, '_updating_channel_tree', False):
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        self._updating_channel_tree = True
        try:
            if data[0] == 'plot':
                plot_key = data[1]
                new_state = item.checkState(0)
                if plot_key == 'gg_diagram':
                    if new_state == Qt.Checked:
                        self.disabled_signals.discard((plot_key, 'gg'))
                    else:
                        self.disabled_signals.add((plot_key, 'gg'))
                else:
                    for i in range(item.childCount()):
                        child = item.child(i)
                        child.setCheckState(0, new_state)
                        cdata = child.data(0, Qt.UserRole)
                        if cdata and cdata[0] == 'signal':
                            if new_state == Qt.Checked:
                                self.disabled_signals.discard((cdata[1], cdata[2]))
                            else:
                                self.disabled_signals.add((cdata[1], cdata[2]))
            elif data[0] == 'signal':
                plot_key, col = data[1], data[2]
                if item.checkState(0) == Qt.Checked:
                    self.disabled_signals.discard((plot_key, col))
                else:
                    self.disabled_signals.add((plot_key, col))
                parent = item.parent()
                if parent:
                    n_checked = sum(1 for i in range(parent.childCount()) if parent.child(i).checkState(0) == Qt.Checked)
                    if n_checked == parent.childCount():
                        parent.setCheckState(0, Qt.Checked)
                    elif n_checked == 0:
                        parent.setCheckState(0, Qt.Unchecked)
                    else:
                        parent.setCheckState(0, Qt.PartiallyChecked)
        finally:
            self._updating_channel_tree = False
            
        self._render_current_view()

    def _select_all_channels(self):
        self.disabled_signals.clear()
        self._update_channel_tree()
        self._render_current_view()

    def _deselect_all_channels(self):
        active_keys = self.current_layout_keys if self.combo_focus.currentIndex() == 0 else [self.combo_focus.currentData()]
        for pk in active_keys:
            if pk == 'gg_diagram':
                self.disabled_signals.add((pk, 'gg'))
            else:
                for col, _, _, _ in self.PLOT_CONFIGS.get(pk, {}).get('signals', []):
                    self.disabled_signals.add((pk, col))
        self._update_channel_tree()
        self._render_current_view()

    def _reset_channel_selection(self):
        self.disabled_signals.clear()
        self._update_channel_tree()
        self._render_current_view()

    def toggle_fullscreen(self):
        if not self._is_fullscreen:
            self.showFullScreen()
            self.btn_fullscreen.setText("Esci Fullscreen (F11)")
            self._is_fullscreen = True
        else:
            self.showNormal()
            self.btn_fullscreen.setText("Fullscreen (F11)")
            self._is_fullscreen = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            return
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
                return
            else:
                self.accept()
                return
        super().keyPressEvent(event)

    def _render_current_view(self):
        self.fig.clear()
        if self.df.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Nessun dato di telemetria disponibile.\nGuida l'auto per raccogliere dati.",
                    ha='center', va='center', fontsize=12, color='white')
            self.canvas.draw()
            return
            
        focus_idx = self.combo_focus.currentIndex()
        if focus_idx == 0:
            keys = self.current_layout_keys
            n_plots = len(keys)
            
            if n_plots == 1:
                rows, cols = 1, 1
            elif n_plots == 2:
                rows, cols = 2, 1
            elif n_plots <= 4:
                rows, cols = 2, 2
            elif n_plots <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 4, 2
                
            for i, k in enumerate(keys):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                self._draw_single_plot(ax, k, is_grid=True)
        else:
            key = self.combo_focus.currentData()
            ax = self.fig.add_subplot(111)
            self._draw_single_plot(ax, key, is_grid=False)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def _popout_current_plot(self):
        focus_idx = self.combo_focus.currentIndex()
        fig_pop = Figure(figsize=(10, 6), dpi=100)
        
        if focus_idx == 0:
            title = f"Griglia Telemetria ({self.combo_preset.currentText()})"
            keys = self.current_layout_keys
            n_plots = len(keys)
            rows = 2 if n_plots <= 6 else 4
            cols = 3 if n_plots == 6 else (2 if n_plots <= 4 else 2)
            if n_plots == 1: rows, cols = 1, 1
            elif n_plots == 2: rows, cols = 2, 1
            
            for i, k in enumerate(keys):
                ax = fig_pop.add_subplot(rows, cols, i + 1)
                self._draw_single_plot(ax, k, is_grid=True, fig=fig_pop)
        else:
            key = self.combo_focus.currentData()
            title = self.AVAILABLE_PLOTS.get(key, {}).get('title', "Grafico Telemetria")
            ax = fig_pop.add_subplot(111)
            self._draw_single_plot(ax, key, is_grid=False, fig=fig_pop)
            
        fig_pop.tight_layout()
        dlg = TelemetryPopoutDialog(fig_pop, title=title, parent=self)
        dlg.exec_()

    def _draw_single_plot(self, ax, plot_key, is_grid=False, fig=None):
        target_fig = fig if fig is not None else self.fig
        fs_title = 8.5 if is_grid else 11.0
        fs_lbl = 8.0 if is_grid else 10.0
        fs_leg = 7.0 if is_grid else 9.0

        x_mode = self.combo_x_axis.currentData() if hasattr(self, 'combo_x_axis') else 's_lap'
        lap_mode = self.combo_lap_filter.currentData() if hasattr(self, 'combo_lap_filter') else 'all'

        if x_mode == 's_lap':
            x_axis_label = "Distanza Pista s [m]"
        elif x_mode == 'dist_total':
            x_axis_label = "Distanza Totale [m]"
        else:
            x_axis_label = "Tempo [s]"

        # Special Case: G-G Diagram
        if plot_key == 'gg_diagram':
            if ('gg_diagram', 'gg') in self.disabled_signals:
                ax.text(0.5, 0.5, "Diagramma G-G (Disattivato in Data Inspector)",
                        ha='center', va='center', fontsize=fs_title, color='#7f8c8d')
                return
            ax._is_gg = True
            is_overlay = lap_mode in ['overlay', 'overlay_selected'] and len(self.laps) > 1
            if is_overlay:
                active_laps = {lid: df for lid, df in self.laps.items() if (lap_mode == 'overlay' or lid in self.selected_laps)}
                if not active_laps:
                    active_laps = self.laps
                for l_id, lap_df in sorted(active_laps.items()):
                    ay = lap_df.get('ay_g', np.zeros(len(lap_df)))
                    ax_g = lap_df.get('ax_g', np.zeros(len(lap_df)))
                    is_best = (l_id == self.best_lap_idx)
                    c = '#f1c40f' if is_best else self.LAP_COLORS[l_id % len(self.LAP_COLORS)]
                    lbl = f"★ Best Lap (G{l_id+1})" if is_best else f"Giro {l_id+1}"
                    ax.scatter(ay, ax_g, color=c, s=12 if is_grid else 22, alpha=0.75 if is_best else 0.45, label=lbl)
            else:
                if lap_mode == 'best' and self.best_lap_idx in self.laps:
                    sub_df = self.laps[self.best_lap_idx]
                elif lap_mode == 'last' and self.laps:
                    sub_df = self.laps[sorted(self.laps.keys())[-1]]
                else:
                    sub_df = self.df
                    
                ay = sub_df.get('ay_g', np.zeros(len(sub_df)))
                ax_g = sub_df.get('ax_g', np.zeros(len(sub_df)))
                v = sub_df.get('vx_kmh', np.zeros(len(sub_df)))
                if len(ay) > 0:
                    sc = ax.scatter(ay, ax_g, c=v, cmap='turbo', s=12 if is_grid else 20, alpha=0.8, edgecolors='none')
                    cbar = target_fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('v [km/h]', fontsize=fs_lbl)
                    cbar.ax.tick_params(labelsize=fs_leg)
                    
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal', 'box')
            ax.axhline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.axvline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.set_title("9. Diagramma G-G (Ay vs Ax) [1:1 Equal]", fontsize=fs_title, fontweight='bold')
            ax.set_xlabel("Acc. Lat. Ay [g]", fontsize=fs_lbl)
            ax.set_ylabel("Acc. Long. Ax [g]", fontsize=fs_lbl)
            ax.grid(True, linestyle=':', alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=fs_leg, loc='best')
            return

        plot_configs = self.PLOT_CONFIGS

        if plot_key not in plot_configs:
            return

        cfg = plot_configs[plot_key]
        title = cfg['title']
        ylabel = cfg['ylabel']

        # --- LAP OVERLAY MODE ---
        if lap_mode in ['overlay', 'overlay_selected'] and len(self.laps) > 1:
            col, short_lbl = cfg['overlay_primary']
            if (plot_key, col) in self.disabled_signals:
                ax.text(0.5, 0.5, f"{title}\n(Disattivato in Data Inspector)", ha='center', va='center', fontsize=fs_title, color='#7f8c8d')
                return
            active_laps = {lid: df for lid, df in self.laps.items() if (lap_mode == 'overlay' or lid in self.selected_laps)}
            if not active_laps:
                active_laps = self.laps
            for l_id, lap_df in sorted(active_laps.items()):
                if x_mode == 's_lap' and 's_lap' in lap_df.columns:
                    lap_df_sorted = lap_df.sort_values('s_lap')
                    x_lap = lap_df_sorted['s_lap']
                    x_axis_label = "Distanza Pista s [m]"
                elif x_mode == 't':
                    lap_df_sorted = lap_df
                    x_lap = lap_df['t'] - lap_df['t'].iloc[0]
                    x_axis_label = "Tempo Trascorso nel Giro dt [s]"
                else:
                    lap_df_sorted = lap_df
                    x_lap = lap_df.get('dist_total', lap_df['t'])
                    x_axis_label = "Distanza Totale [m]"

                if col == 'tire_grip_fl_pct' and 'tire_grip_fl' in lap_df_sorted.columns:
                    y_lap = lap_df_sorted['tire_grip_fl'] * 100.0
                elif col in lap_df_sorted.columns:
                    y_lap = lap_df_sorted[col]
                else:
                    continue

                is_best = (l_id == self.best_lap_idx)
                lap_dt = self.lap_stats.get(l_id, {}).get('dt', 0.0)
                lap_t_str = f"{lap_dt:.2f}s"
                color = '#f1c40f' if is_best else self.LAP_COLORS[l_id % len(self.LAP_COLORS)]
                lw = 2.4 if is_best else 1.3
                lbl = f"★ Best Lap (G{l_id+1}: {lap_t_str})" if is_best else f"Giro {l_id+1} ({lap_t_str})"
                ax.plot(x_lap, y_lap, label=lbl, color=color, linewidth=lw)

                # In focused view, add auxiliary secondary channels per lap if applicable
                if not is_grid and plot_key == 'chassis_velocities' and 'vy_kmh' in lap_df_sorted.columns:
                    if (plot_key, 'vy_kmh') not in self.disabled_signals:
                        ax.plot(x_lap, lap_df_sorted['vy_kmh'], color=color, linewidth=1.0, linestyle='--', alpha=0.55)
                if not is_grid and plot_key == 'driver_inputs' and 'throttle_pct' in lap_df_sorted.columns:
                    if (plot_key, 'throttle_pct') not in self.disabled_signals:
                        ax.plot(x_lap, lap_df_sorted['throttle_pct'], color=color, linewidth=1.0, linestyle=':', alpha=0.55)

            if plot_key in ['tire_temps_bulk', 'tire_temps_detailed']:
                ax.axhline(56.2, color='#2ecc71', linestyle=':', label='Ottimale TTC (56.2°C)')

            mode_tag = "Giri Selezionati" if lap_mode == 'overlay_selected' else "Giro vs Giro"
            ax.set_title(f"{title} [Overlay {mode_tag}]", fontsize=fs_title, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=fs_lbl)
            ax.set_xlabel(x_axis_label, fontsize=fs_lbl)

        # --- STANDARD / FILTERED VIEW (ALL, BEST, LAST) ---
        else:
            if lap_mode == 'best' and self.best_lap_idx in self.laps:
                sub_df = self.laps[self.best_lap_idx].copy()
                suffix = f" [Best Lap: Giro {self.best_lap_idx + 1}]"
            elif lap_mode == 'last' and self.laps:
                last_id = sorted(self.laps.keys())[-1]
                sub_df = self.laps[last_id].copy()
                suffix = f" [Ultimo Giro: Giro {last_id + 1}]"
            else:
                sub_df = self.df.copy()
                suffix = "" if len(self.laps) > 1 else " (Giro 1)"

            if x_mode == 's_lap' and 's_lap' in sub_df.columns:
                if lap_mode in ['best', 'last']:
                    sub_df = sub_df.sort_values('s_lap')
                x_vals = sub_df['s_lap']
                x_axis_label = "Distanza Pista s [m]"
            elif x_mode == 'dist_total' and 'dist_total' in sub_df.columns:
                x_vals = sub_df['dist_total']
                x_axis_label = "Distanza Totale [m]"
            else:
                x_vals = sub_df['t'] if 't' in sub_df.columns else np.arange(len(sub_df))
                x_axis_label = "Tempo [s]"

            for col, lbl, color, ls in cfg['signals']:
                if (plot_key, col) in self.disabled_signals:
                    continue
                if col == 'tire_grip_fl_pct' and 'tire_grip_fl' in sub_df.columns:
                    y_vals = sub_df['tire_grip_fl'] * 100.0
                elif col == 'tire_grip_rr_pct' and 'tire_grip_rr' in sub_df.columns:
                    y_vals = sub_df['tire_grip_rr'] * 100.0
                elif col in sub_df.columns:
                    y_vals = sub_df[col]
                else:
                    continue
                ax.plot(x_vals, y_vals, label=lbl, color=color, linestyle=ls)

            if plot_key in ['tire_temps_bulk', 'tire_temps_detailed']:
                ax.axhline(56.2, color='#2ecc71', linestyle=':', label='Ottimale TTC (56.2°C)')

            ax.set_title(f"{title}{suffix}", fontsize=fs_title, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=fs_lbl)
            ax.set_xlabel(x_axis_label, fontsize=fs_lbl)

        ax.grid(True, linestyle=':', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=fs_leg, loc='best')



# =============================================================================
# 5.5 INTERACTIVE 2D TRACK DESIGNER & SPLINE EDITOR
# =============================================================================

