#!/usr/bin/env python3
"""
preview_car_3d.py
=================
Standalone 3D Preview & Modeling Studio for Formula Student (FSAE) and B-Segment SUV.
Features:
- Full 360° Orbit Camera (Left Click Drag: Rotate, Right Click / Wheel: Zoom, Middle Click: Pan)
- Multi-Vehicle Selection: Formula Student (FSAE) or Piccolo SUV Segmento B (FWD)
- Ambient Occlusion (AO) Ground Contact Shadows & Diffuser Shadow Decals
- Multi-Source Studio Lighting (Key Light, Fill Sky Light, Ground Bounce)
- Solid Volumetric 3D Wheels & Chassis
- Live Controls: Steering, Suspension Travel (z_fl..z_rr), Grip Glow, Livery Presets, Wireframe & Auto-Rotate
"""

import os
import sys
import math
import argparse
import numpy as np

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox,
    QPushButton, QComboBox, QCheckBox, QColorDialog
)
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush

try:
    from PyQt5.QtWidgets import QOpenGLWidget
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("ERRORE: PyOpenGL o PyQt5.QOpenGLWidget non trovato nel venv!")

from src.models.vehicle_3d_models import FSAECar3DModel, SUVCar3DModel


# =============================================================================
# 1. INTERACTIVE 3D OPENGL ORBIT STUDIO VIEWPORT
# =============================================================================
class CarStudioViewport(QOpenGLWidget):
    """Interactive OpenGL Canvas with 360 Orbit Camera and Dynamic Studio Lighting."""
    def __init__(self, parent=None, initial_vehicle="formula"):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.fsae_model = FSAECar3DModel()
        self.suv_model = SUVCar3DModel()
        self.vehicle_type = initial_vehicle

        # Orbit Camera Parameters
        self.cam_dist = 6.2 if self.vehicle_type == 'b_segment_suv' else 4.2
        self.cam_pitch = 20.0 if self.vehicle_type == 'b_segment_suv' else 22.0
        self.cam_yaw = 45.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.last_mouse_pos = QPointF(0, 0)
        self.is_rotating = False
        self.is_panning = False
        self.auto_rotate = False
        self.wireframe = False

        # Live Car States
        self.steer_deg = 0.0
        self.z_fl = 0.0
        self.z_fr = 0.0
        self.z_rl = 0.0
        self.z_rr = 0.0
        self.util_pct = 50.0

        # Auto-rotate timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(16)  # ~60 FPS

    @property
    def car_model(self):
        return self.suv_model if self.vehicle_type == 'b_segment_suv' else self.fsae_model

    def set_vehicle_type(self, v_type: str):
        self.vehicle_type = v_type
        if v_type == 'b_segment_suv':
            self.cam_dist = 6.2
            self.cam_pitch = 20.0
            self.cam_yaw = 45.0
            self.pan_x = 0.0
            self.pan_y = 0.0
        else:
            self.cam_dist = 4.2
            self.cam_pitch = 22.0
            self.cam_yaw = 45.0
            self.pan_x = 0.0
            self.pan_y = 0.0
        self.update()

    def _on_timer(self):
        if self.auto_rotate:
            self.cam_yaw = (self.cam_yaw + 0.6) % 360.0
            self.update()

    def initializeGL(self):
        glClearColor(26/255.0, 32/255.0, 42/255.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Disable face culling so double-sided geometry and rims are 100% solid
        glDisable(GL_CULL_FACE)

        # Multi-Source Studio Lighting Setup
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)  # Main Sun Key Light
        glEnable(GL_LIGHT1)  # Soft Sky Fill Light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

        # Key Light (Top-Front-Right)
        glLightfv(GL_LIGHT0, GL_POSITION, [4.0, 5.0, 8.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.98, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.8, 1.0])

        # Fill Light (Low Left Ambient Fill)
        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0, -4.0, 3.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.35, 0.40, 0.48, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.28, 0.30, 0.35, 1.0])

        # Material Specular Properties for Automotive Gloss
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / (h if h > 0 else 1)
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Compute Orbit Camera Position
        pitch_rad = np.deg2rad(self.cam_pitch)
        yaw_rad = np.deg2rad(self.cam_yaw)

        cx = self.cam_dist * np.cos(pitch_rad) * np.cos(yaw_rad)
        cy = self.cam_dist * np.cos(pitch_rad) * np.sin(yaw_rad)
        cz = self.cam_dist * np.sin(pitch_rad)

        z_target = 0.45 if self.vehicle_type == 'b_segment_suv' else 0.15

        gluLookAt(
            cx + self.pan_x, cy + self.pan_y, cz,
            self.pan_x, self.pan_y, z_target,
            0.0, 0.0, 1.0
        )

        # Update Key Light in Camera/World coordinates
        glLightfv(GL_LIGHT0, GL_POSITION, [4.0, 5.0, 8.0, 1.0])

        # Optional Wireframe Mode
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # 1. Draw Studio Ground Grid & Origin Pedestal
        self._draw_ground_pedestal()

        # 2. Draw 3D Car with AO and Solid Geometry
        state = {
            'steer_deg': self.steer_deg,
            'z_fl': self.z_fl, 'z_fr': self.z_fr,
            'z_rl': self.z_rl, 'z_rr': self.z_rr,
            'util_fl': self.util_pct, 'util_fr': self.util_pct,
            'util_rl': self.util_pct, 'util_rr': self.util_pct
        }
        self.car_model.draw(state)

        # 3. Render 2D Floating Grip Telemetry HUD in Top-Left Corner
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_telemetry_hud(painter)
        painter.end()

    def _draw_telemetry_hud(self, painter):
        """Draws an ultra-crisp 4-wheel tire grip telemetry HUD widget."""
        w_box, h_box = 240, 140
        x0, y0 = 16, 16

        # Background Card
        painter.setBrush(QColor(18, 22, 28, 220))
        painter.setPen(QPen(QColor(53, 59, 72), 1.5))
        painter.drawRoundedRect(x0, y0, w_box, h_box, 6, 6)

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(0, 206, 201))
        painter.drawText(x0 + 10, y0 + 20, "UTILIZZO GRIP GOMME")

        # 4 Wheel Indicators
        wheels = [
            ("FL (Ant. SX)", self.util_pct, x0 + 10, y0 + 42),
            ("FR (Ant. DX)", self.util_pct, x0 + 125, y0 + 42),
            ("RL (Post. SX)", self.util_pct, x0 + 10, y0 + 88),
            ("RR (Post. DX)", self.util_pct, x0 + 125, y0 + 88),
        ]

        for lbl, u_val, bx, by in wheels:
            col = QColor(46, 204, 113) if u_val < 60 else (QColor(241, 196, 15) if u_val < 88 else QColor(231, 76, 60))
            
            # Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(220, 225, 230))
            painter.drawText(bx, by, f"{lbl}: {u_val:.0f}%")

            # Progress Bar Track
            painter.setBrush(QColor(35, 42, 50))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bx, by + 6, 100, 12, 3, 3)

            # Active Bar
            fill_w = int(np.clip(u_val / 100.0, 0.0, 1.0) * 100)
            painter.setBrush(col)
            painter.drawRoundedRect(bx, by + 6, fill_w, 12, 3, 3)

    def _draw_ground_pedestal(self):
        # Pedestal Circular Floor (Dark Studio Turntable)
        glColor3f(0.14, 0.17, 0.22)
        r_ped = 3.6 if self.vehicle_type == 'b_segment_suv' else 2.4
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, -0.10)
        for i in range(33):
            ang = i * 2.0 * math.pi / 32.0
            glVertex3f(r_ped * math.cos(ang), r_ped * math.sin(ang), -0.10)
        glEnd()

        # Pedestal Outer Ring (Cyan Glow Accent)
        glColor3f(0.0, 0.85, 0.80)
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            ang = i * 2.0 * math.pi / 32.0
            glVertex3f(r_ped * math.cos(ang), r_ped * math.sin(ang), -0.098)
        glEnd()

        # Studio Radial Coordinate Rings
        glColor3f(0.20, 0.24, 0.30)
        glLineWidth(1.2)
        rings = [1.2, 2.4] if self.vehicle_type == 'b_segment_suv' else [0.8, 1.6]
        for r_ring in rings:
            glBegin(GL_LINE_LOOP)
            for i in range(32):
                ang = i * 2.0 * math.pi / 32.0
                glVertex3f(r_ring * math.cos(ang), r_ring * math.sin(ang), -0.099)
            glEnd()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.LeftButton:
            self.is_rotating = True
        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self.is_panning = True

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()

        if self.is_rotating:
            self.cam_yaw = (self.cam_yaw + dx * 0.4) % 360.0
            self.cam_pitch = np.clip(self.cam_pitch - dy * 0.3, -80.0, 85.0)
            self.update()
        elif self.is_panning:
            yaw_rad = np.deg2rad(self.cam_yaw)
            self.pan_x += (-dx * np.sin(yaw_rad) - dy * np.cos(yaw_rad)) * 0.005
            self.pan_y += (dx * np.cos(yaw_rad) - dy * np.sin(yaw_rad)) * 0.005
            self.update()

    def mouseReleaseEvent(self, event):
        self.is_rotating = False
        self.is_panning = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.cam_dist = np.clip(self.cam_dist * zoom_factor, 1.2, 25.0)
        self.update()


# =============================================================================
# 2. STANDALONE STUDIO MAIN APPLICATION WINDOW
# =============================================================================
class CarPreviewStudioWindow(QMainWindow):
    """Main Window of the Standalone 3D Preview Studio."""
    def __init__(self, initial_vehicle="formula"):
        super().__init__()
        self.initial_vehicle = initial_vehicle
        title = "Studio 3D — Piccolo SUV Segmento B (FWD)" if initial_vehicle == 'b_segment_suv' else "Studio 3D — Formula Student (FSAE)"
        self.setWindowTitle(title)
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #1a1e24; color: #ecf0f1; font-family: Segoe UI, sans-serif;")

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Left 3D Viewport
        self.viewport = CarStudioViewport(self, initial_vehicle=self.initial_vehicle)
        main_layout.addWidget(self.viewport, stretch=7)

        # Right Control Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(right_panel, stretch=3)

        # --- GROUP 0: VEHICLE SELECTION ---
        veh_group = QGroupBox("Selezione Modello Veicolo 3D")
        veh_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #3d4a58; border-radius: 4px; padding-top: 10px; }")
        veh_layout = QVBoxLayout(veh_group)

        self.combo_vehicle = QComboBox()
        self.combo_vehicle.addItem("Formula Student (FSAE Single-Seater)", "formula")
        self.combo_vehicle.addItem("Piccolo SUV Segmento B (FWD)", "b_segment_suv")
        idx_init = 1 if self.initial_vehicle == 'b_segment_suv' else 0
        self.combo_vehicle.setCurrentIndex(idx_init)
        self.combo_vehicle.currentIndexChanged.connect(self._on_vehicle_changed)
        veh_layout.addWidget(self.combo_vehicle)
        right_layout.addWidget(veh_group)

        # --- GROUP 1: STEERING & SUSPENSIONS ---
        kin_group = QGroupBox("Sterzo & Sospensioni Live")
        kin_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00cec9; border: 1px solid #3d4a58; border-radius: 4px; padding-top: 10px; }")
        kin_layout = QGridLayout(kin_group)

        kin_layout.addWidget(QLabel("Angolo Volante / Sterzo:"), 0, 0)
        self.sl_steer = QSlider(Qt.Horizontal)
        steer_max = 35 if self.initial_vehicle == 'b_segment_suv' else 25
        self.sl_steer.setRange(-steer_max, steer_max)
        self.sl_steer.setValue(0)
        self.sl_steer.valueChanged.connect(self._on_steer_changed)
        kin_layout.addWidget(self.sl_steer, 0, 1)

        self.lbl_steer_val = QLabel("0.0 deg")
        kin_layout.addWidget(self.lbl_steer_val, 0, 2)

        # Suspension Travel (z_fl, z_fr, z_rl, z_rr) in mm
        susp_rng = 80.0 if self.initial_vehicle == 'b_segment_suv' else 40.0
        kin_layout.addWidget(QLabel("Escursione Ant. SX (z_fl):"), 1, 0)
        self.sb_z_fl = QDoubleSpinBox()
        self.sb_z_fl.setRange(-susp_rng, susp_rng)
        self.sb_z_fl.setValue(0.0)
        self.sb_z_fl.setSuffix(" mm")
        self.sb_z_fl.valueChanged.connect(self._on_susp_changed)
        kin_layout.addWidget(self.sb_z_fl, 1, 1, 1, 2)

        kin_layout.addWidget(QLabel("Escursione Ant. DX (z_fr):"), 2, 0)
        self.sb_z_fr = QDoubleSpinBox()
        self.sb_z_fr.setRange(-susp_rng, susp_rng)
        self.sb_z_fr.setValue(0.0)
        self.sb_z_fr.setSuffix(" mm")
        self.sb_z_fr.valueChanged.connect(self._on_susp_changed)
        kin_layout.addWidget(self.sb_z_fr, 2, 1, 1, 2)

        kin_layout.addWidget(QLabel("Escursione Post. SX (z_rl):"), 3, 0)
        self.sb_z_rl = QDoubleSpinBox()
        self.sb_z_rl.setRange(-susp_rng, susp_rng)
        self.sb_z_rl.setValue(0.0)
        self.sb_z_rl.setSuffix(" mm")
        self.sb_z_rl.valueChanged.connect(self._on_susp_changed)
        kin_layout.addWidget(self.sb_z_rl, 3, 1, 1, 2)

        kin_layout.addWidget(QLabel("Escursione Post. DX (z_rr):"), 4, 0)
        self.sb_z_rr = QDoubleSpinBox()
        self.sb_z_rr.setRange(-susp_rng, susp_rng)
        self.sb_z_rr.setValue(0.0)
        self.sb_z_rr.setSuffix(" mm")
        self.sb_z_rr.valueChanged.connect(self._on_susp_changed)
        kin_layout.addWidget(self.sb_z_rr, 4, 1, 1, 2)

        btn_reset_susp = QPushButton("Azzera Sospensioni")
        btn_reset_susp.clicked.connect(self._reset_susp)
        kin_layout.addWidget(btn_reset_susp, 5, 0, 1, 3)

        right_layout.addWidget(kin_group)

        # --- GROUP 2: TIRE GRIP SIMULATION ---
        grip_group = QGroupBox("Simulazione Grip & Cerchi")
        grip_group.setStyleSheet("QGroupBox { font-weight: bold; color: #f1c40f; border: 1px solid #3d4a58; border-radius: 4px; padding-top: 10px; }")
        grip_layout = QGridLayout(grip_group)

        grip_layout.addWidget(QLabel("Grip Utilization (%):"), 0, 0)
        self.sl_grip = QSlider(Qt.Horizontal)
        self.sl_grip.setRange(0, 100)
        self.sl_grip.setValue(50)
        self.sl_grip.valueChanged.connect(self._on_grip_changed)
        grip_layout.addWidget(self.sl_grip, 0, 1)

        self.lbl_grip_val = QLabel("50 % (Verde)")
        grip_layout.addWidget(self.lbl_grip_val, 0, 2)

        right_layout.addWidget(grip_group)

        # --- GROUP 3: LIVERY & COLOR CUSTOMIZATION ---
        liv_group = QGroupBox("Livrea & Personalizzazione Colori")
        liv_group.setStyleSheet("QGroupBox { font-weight: bold; color: #e74c3c; border: 1px solid #3d4a58; border-radius: 4px; padding-top: 10px; }")
        liv_layout = QGridLayout(liv_group)

        liv_layout.addWidget(QLabel("Preset Livrea:"), 0, 0)
        self.combo_livery = QComboBox()
        self.combo_livery.addItems([
            "Formula Blue / Compact Azure",
            "Rosso Corsa (Scuderia)",
            "British Racing Green",
            "Stealth Carbon / Urban Grey",
            "Modena Yellow / Solar Orange"
        ])
        self.combo_livery.currentIndexChanged.connect(self._on_livery_preset)
        liv_layout.addWidget(self.combo_livery, 0, 1)

        btn_col_body = QPushButton("Colore Carrozzeria...")
        btn_col_body.clicked.connect(self._pick_body_color)
        liv_layout.addWidget(btn_col_body, 1, 0)

        btn_col_accent = QPushButton("Colore Dettagli/Ali...")
        btn_col_accent.clicked.connect(self._pick_accent_color)
        liv_layout.addWidget(btn_col_accent, 1, 1)

        right_layout.addWidget(liv_group)

        # --- GROUP 4: STUDIO CAMERA & VISUAL OPTIONS ---
        cam_group = QGroupBox("Opzioni Camera Studio")
        cam_group.setStyleSheet("QGroupBox { font-weight: bold; color: #9b59b6; border: 1px solid #3d4a58; border-radius: 4px; padding-top: 10px; }")
        cam_layout = QVBoxLayout(cam_group)

        self.cb_autorotate = QCheckBox("Rotazione Automatica 360 deg")
        self.cb_autorotate.toggled.connect(lambda v: setattr(self.viewport, 'auto_rotate', v))
        cam_layout.addWidget(self.cb_autorotate)

        self.cb_wireframe = QCheckBox("Modalita Wireframe (Mesh)")
        self.cb_wireframe.toggled.connect(self._toggle_wireframe)
        cam_layout.addWidget(self.cb_wireframe)

        btn_reset_cam = QPushButton("Centra Camera")
        btn_reset_cam.clicked.connect(self._reset_camera)
        cam_layout.addWidget(btn_reset_cam)

        right_layout.addWidget(cam_group)
        right_layout.addStretch()

        # Instructions Footer
        lbl_hint = QLabel("Mouse SX Drag: Ruota 360 deg | Mouse DX/Wheel: Zoom | Tasto Centrale: Pan")
        lbl_hint.setStyleSheet("font-size: 11px; color: #8395a7;")
        right_layout.addWidget(lbl_hint)

    def _on_vehicle_changed(self, idx):
        v_type = self.combo_vehicle.itemData(idx)
        self.viewport.set_vehicle_type(v_type)
        if v_type == 'b_segment_suv':
            self.setWindowTitle("Studio 3D — Piccolo SUV Segmento B (FWD, Differenziale Aperto)")
            self.sb_z_fl.setRange(-80.0, 80.0)
            self.sb_z_fr.setRange(-80.0, 80.0)
            self.sb_z_rl.setRange(-80.0, 80.0)
            self.sb_z_rr.setRange(-80.0, 80.0)
            self.sl_steer.setRange(-35, 35)
        else:
            self.setWindowTitle("Studio 3D — Formula Student (FSAE Single-Seater)")
            self.sb_z_fl.setRange(-40.0, 40.0)
            self.sb_z_fr.setRange(-40.0, 40.0)
            self.sb_z_rl.setRange(-40.0, 40.0)
            self.sb_z_rr.setRange(-40.0, 40.0)
            self.sl_steer.setRange(-25, 25)

    def _on_steer_changed(self, val):
        self.viewport.steer_deg = float(val)
        self.lbl_steer_val.setText(f"{val:.1f}°")
        self.viewport.update()

    def _on_susp_changed(self):
        self.viewport.z_fl = self.sb_z_fl.value()
        self.viewport.z_fr = self.sb_z_fr.value()
        self.viewport.z_rl = self.sb_z_rl.value()
        self.viewport.z_rr = self.sb_z_rr.value()
        self.viewport.update()

    def _reset_susp(self):
        self.sb_z_fl.setValue(0.0)
        self.sb_z_fr.setValue(0.0)
        self.sb_z_rl.setValue(0.0)
        self.sb_z_rr.setValue(0.0)

    def _on_grip_changed(self, val):
        self.viewport.util_pct = float(val)
        tag = "Verde" if val < 60 else ("Giallo" if val < 88 else "Rosso Limit")
        self.lbl_grip_val.setText(f"{val} % ({tag})")
        self.viewport.update()

    def _on_livery_preset(self, idx):
        cm = self.viewport.car_model
        if hasattr(cm, 'set_livery_preset'):
            cm.set_livery_preset(idx)
        else:
            if idx == 0:
                cm.col_body = (0.024, 0.322, 0.867)
                cm.col_accent = (0.906, 0.188, 0.188)
            elif idx == 1:
                cm.col_body = (0.85, 0.12, 0.12)
                cm.col_accent = (0.95, 0.95, 0.95)
            elif idx == 2:
                cm.col_body = (0.08, 0.45, 0.28)
                cm.col_accent = (0.95, 0.77, 0.15)
            elif idx == 3:
                cm.col_body = (0.12, 0.14, 0.16)
                cm.col_accent = (0.95, 0.77, 0.15)
            elif idx == 4:
                cm.col_body = (0.95, 0.77, 0.06)
                cm.col_accent = (0.12, 0.14, 0.16)
        self.viewport.update()

    def _pick_body_color(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.viewport.car_model.col_body = (col.redF(), col.greenF(), col.blueF())
            self.viewport.update()

    def _pick_accent_color(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.viewport.car_model.col_accent = (col.redF(), col.greenF(), col.blueF())
            self.viewport.update()

    def _toggle_wireframe(self, checked):
        self.viewport.wireframe = checked
        self.viewport.update()

    def _reset_camera(self):
        if self.viewport.vehicle_type == 'b_segment_suv':
            self.viewport.cam_dist = 6.2
            self.viewport.cam_pitch = 20.0
        else:
            self.viewport.cam_dist = 4.2
            self.viewport.cam_pitch = 22.0
        self.viewport.cam_yaw = 45.0
        self.viewport.pan_x = 0.0
        self.viewport.pan_y = 0.0
        self.viewport.update()


# =============================================================================
# 3. ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Studio Anteprima 3D Veicolo")
    parser.add_argument('--vehicle', choices=['formula', 'b_segment_suv', 'fsae', 'suv'], default='formula',
                        help="Tipo di veicolo 3D da visualizzare (formula / b_segment_suv)")
    args, _ = parser.parse_known_args()

    v_type = 'b_segment_suv' if args.vehicle in ['b_segment_suv', 'suv'] else 'formula'
    app = QApplication(sys.argv)
    win = CarPreviewStudioWindow(initial_vehicle=v_type)
    win.show()
    sys.exit(app.exec_())
