"""
ui_seven_post_rig.py
--------------------
Interfaccia Grafica Professionale, Post-Processing Avanzato e Banco 2D per il 7-Post Shaker Rig.

Include:
- Pannello impostazioni completo (Chirp, Sine Dwell, Kerb/Obstacle, ISO 8608, Taratura TTX25 e Overrides)
- Impostazioni di visualizzazione post-processing (dB vs Lineare, Scala Frequenze, Decomposizione CPLV)
- Monitor KPI sintetici a 8 Card (Score Setup, CPLV Medio, Lift-off %, Accel RMS, Escursione Ride Height)
- Visualizzatore Analitico Avanzato a 8 Sub-Tab:
  1. Spettro FRF, Fase & Coerenza (Bode con annotazione picchi di risonanza)
  2. Grip & CPLV Multibanda (Confronto Low-Band 0.5-4.5 Hz vs High-Band 8-18 Hz e perdita grip %)
  3. Telemetria 7 Attuatori (4 stradali + 3 aerodinamici + carichi normali Fz)
  4. Diagnostica TTX25 & Bilstrom (Lissajous F-v, istogramma a 5 bande, bilancio di potenza ed energia)
  5. Piattaforma Aerodinamica (Fluttuazione ride height, beccheggio/rollio dinamici, PSD accelerazioni)
  6. Modi Propri & Smorzamento (Tabella linearizzata e diagramma dei poli nel piano s)
  7. Componente Grafica 2D Animata del Banco Prova (Vista Laterale + Frontale con Timeline Play/Scrub)
  8. Report Tecnico Ingegneristico (Export TXT/MD, Copia negli appunti)
- Integrazione asincrona tramite SevenPostRigWorker (QThread)
"""

import os
import math
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSplitter, QTabWidget, QSlider,
    QDoubleSpinBox, QSpinBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QFrame, QScrollArea, QSizePolicy, QTextEdit,
    QCheckBox, QApplication
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.mdo.mdo_seven_post_rig import SevenPostRig, SevenPostConfig, SevenPostResult
from src.ui.gui_workers import SevenPostRigWorker
from src.ui.ui_theme_engine import apply_plot_theme, get_theme_palette


class RigSchematic2DCanvas(FigureCanvas):
    """
    Visualizzatore Cinematico 2D del Banco Prova 7-Post Rig.
    Mostra il telaio, i 4 attuatori stradali, i 3 attuatori aerodinamici e la deformazione
    di molle, ammortizzatori e pneumatici in tempo reale con timeline interattiva.
    """
    def __init__(self, parent=None, width=9, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Due viste affiancate: Vista Laterale (Heave/Pitch) e Vista Posteriore/Frontale (Roll)
        self.ax_side = self.fig.add_subplot(121)
        self.ax_end = self.fig.add_subplot(122)
        apply_plot_theme(self.fig)
        self.fig.tight_layout()

        self.res: SevenPostResult = None
        self.current_frame = 0

    def set_result(self, res: SevenPostResult):
        self.res = res
        self.current_frame = 0
        self.draw_frame(0)

    def draw_frame(self, frame_idx: int):
        if self.res is None or frame_idx >= len(self.res.time):
            return

        self.current_frame = frame_idx
        res = self.res
        t = res.time[frame_idx]

        # Ingrandimento visivo per rendere chiare le oscillazioni dinamiche (scala 3.5x)
        vis_scale = 3.5

        # Dati cinematici istantanei
        zs = res.z_chassis[frame_idx, 0] * vis_scale
        theta = res.z_chassis[frame_idx, 1] * vis_scale
        phi = res.z_chassis[frame_idx, 2] * vis_scale

        z_p = res.z_posts[frame_idx] * vis_scale     # [FL, FR, RL, RR]
        zw = res.z_wheels[frame_idx] * vis_scale     # [FL, FR, RL, RR]
        Fz = res.Fz_tires[frame_idx]                 # [FL, FR, RL, RR]
        Fa = res.F_aero_posts[frame_idx]             # [Front, Rear-Left, Rear-Right]

        # -------------------------------------------------------------
        # 1. Vista Laterale (Side View: Asse X-Z)
        # -------------------------------------------------------------
        ax_s = self.ax_side
        ax_s.clear()
        ax_s.set_title(
            f"Vista Laterale Banco — t = {t:.2f} s | Heave={zs*1000/vis_scale:+.1f} mm, Pitch={math.degrees(theta/vis_scale):+.2f} deg",
            fontsize=9, fontweight='bold', color='#58a6ff'
        )

        L_vis = 1.60
        a_vis = 0.85
        b_vis = 0.75
        x_f = a_vis
        x_r = -b_vis

        # Piano di appoggio fisso del banco
        ax_s.plot([-1.2, 1.2], [-0.08, -0.08], color='#30363d', lw=4)

        # Attuatori stradali (Cilindri idraulici a terra)
        ax_s.plot([x_f, x_f], [-0.08, z_p[0]], color='#8b949e', lw=5)
        ax_s.plot([x_f - 0.08, x_f + 0.08], [z_p[0], z_p[0]], color='#f0883e', lw=4)

        ax_s.plot([x_r, x_r], [-0.08, z_p[2]], color='#8b949e', lw=5)
        ax_s.plot([x_r - 0.08, x_r + 0.08], [z_p[2], z_p[2]], color='#f0883e', lw=4)

        # Ruote e Mozzi
        c_tire_f = '#3fb950' if Fz[0] > 10.0 else '#f85149'
        ax_s.plot([x_f, x_f], [z_p[0], zw[0] + 0.12], color=c_tire_f, lw=8, solid_capstyle='round')
        ax_s.plot(x_f, zw[0] + 0.12, 'o', color='#ffffff', markersize=6)

        c_tire_r = '#3fb950' if Fz[2] > 10.0 else '#f85149'
        ax_s.plot([x_r, x_r], [z_p[2], zw[2] + 0.12], color=c_tire_r, lw=8, solid_capstyle='round')
        ax_s.plot(x_r, zw[2] + 0.12, 'o', color='#ffffff', markersize=6)

        # Punti attacco telaio con beccheggio e heave
        z_c_f = 0.22 + zs - theta * a_vis
        z_c_r = 0.22 + zs + theta * b_vis
        z_c_mid = 0.22 + zs + 0.10

        # Silhouette telaio Formula Student
        body_x = [x_f + 0.25, x_f, 0.0, x_r, x_r - 0.15, x_r, 0.0, x_f, x_f + 0.25]
        body_z = [
            z_c_f - 0.05, z_c_f + 0.05, z_c_mid + 0.18, z_c_r + 0.28,
            z_c_r + 0.05, z_c_r - 0.04, z_c_mid - 0.06, z_c_f - 0.06, z_c_f - 0.05
        ]
        ax_s.plot(body_x, body_z, color='#58a6ff', lw=2.5)
        ax_s.fill(body_x, body_z, color='#1f6feb', alpha=0.25)

        # Sospensioni
        ax_s.plot([x_f, x_f], [zw[0] + 0.12, z_c_f], color='#e3b341', lw=2.5, ls='--')
        ax_s.plot([x_r, x_r], [zw[2] + 0.12, z_c_r], color='#e3b341', lw=2.5, ls='--')

        # Attuatori Aerodinamici Superiori (Post 5 Ant & Post 6/7 Post)
        is_4post = getattr(res, 'rig_architecture', '7_post') == '4_post'
        if not is_4post:
            ax_s.plot([-1.2, 1.2], [0.80, 0.80], color='#30363d', lw=4)
            ax_s.plot([x_f, x_f], [0.80, z_c_f + 0.05], color='#bc8cff', lw=3)
            ax_s.annotate(f"Post 5\n{Fa[0]:.0f} N", xy=(x_f, z_c_f + 0.08), xytext=(x_f + 0.10, 0.65),
                          arrowprops=dict(arrowstyle="->", color='#bc8cff', lw=1.5),
                          fontsize=7, color='#bc8cff', fontweight='bold')

            fa_r_avg = 0.5 * (Fa[1] + Fa[2])
            ax_s.plot([x_r, x_r], [0.80, z_c_r + 0.28], color='#bc8cff', lw=3)
            ax_s.annotate(f"Post 6+7\n{fa_r_avg:.0f} N", xy=(x_r, z_c_r + 0.30), xytext=(x_r - 0.30, 0.65),
                          arrowprops=dict(arrowstyle="->", color='#bc8cff', lw=1.5),
                          fontsize=7, color='#bc8cff', fontweight='bold')
        else:
            ax_s.plot([-1.2, 1.2], [0.80, 0.80], color='#21262d', lw=2, linestyle=':')
            ax_s.text(0.0, 0.76, "[CONFIGURAZIONE 4-POST: TELAIO LIBERO]",
                      color='#8b949e', fontsize=7, ha='center', fontweight='bold')

        ax_s.set_xlim(-1.3, 1.3)
        ax_s.set_ylim(-0.15, 0.90)
        ax_s.set_aspect('equal')
        ax_s.grid(True, color='#21262d', linestyle=':', alpha=0.6)
        ax_s.set_xlabel("X Longitudinale [m]", fontsize=8)
        ax_s.set_ylabel("Z Quota [m]", fontsize=8)

        # -------------------------------------------------------------
        # 2. Vista Posteriore / Frontale (End View: Asse Y-Z)
        # -------------------------------------------------------------
        ax_e = self.ax_end
        ax_e.clear()
        ax_e.set_title(
            f"Vista Trasversale — Roll={math.degrees(phi/vis_scale):+.2f} deg | Fz_RL={Fz[2]:.0f} N, Fz_RR={Fz[3]:.0f} N",
            fontsize=9, fontweight='bold', color='#3fb950'
        )

        t_vis = 0.60
        y_l = t_vis
        y_r = -t_vis

        ax_e.plot([-0.9, 0.9], [-0.08, -0.08], color='#30363d', lw=4)

        # Attuatori stradali Sx e Dx (Post 3 e Post 4)
        ax_e.plot([y_l, y_l], [-0.08, z_p[2]], color='#8b949e', lw=5)
        ax_e.plot([y_l - 0.08, y_l + 0.08], [z_p[2], z_p[2]], color='#f0883e', lw=4)

        ax_e.plot([y_r, y_r], [-0.08, z_p[3]], color='#8b949e', lw=5)
        ax_e.plot([y_r - 0.08, y_r + 0.08], [z_p[3], z_p[3]], color='#f0883e', lw=4)

        # Pneumatici Sx e Dx
        c_tire_rl = '#3fb950' if Fz[2] > 10.0 else '#f85149'
        ax_e.plot([y_l, y_l], [z_p[2], zw[2] + 0.12], color=c_tire_rl, lw=9, solid_capstyle='round')
        ax_e.plot(y_l, zw[2] + 0.12, 'o', color='#ffffff', markersize=6)

        c_tire_rr = '#3fb950' if Fz[3] > 10.0 else '#f85149'
        ax_e.plot([y_r, y_r], [z_p[3], zw[3] + 0.12], color=c_tire_rr, lw=9, solid_capstyle='round')
        ax_e.plot(y_r, zw[3] + 0.12, 'o', color='#ffffff', markersize=6)

        # Punti attacco telaio con rollio
        z_c_l = 0.24 + zs + phi * t_vis
        z_c_r = 0.24 + zs - phi * t_vis
        z_c_rollhoop = 0.52 + zs

        chassis_x = [y_l, y_l * 0.5, 0.0, y_r * 0.5, y_r, y_r * 0.7, y_l * 0.7, y_l]
        chassis_z = [z_c_l, z_c_l + 0.10, z_c_rollhoop, z_c_r + 0.10, z_c_r, z_c_r - 0.05, z_c_l - 0.05, z_c_l]
        ax_e.plot(chassis_x, chassis_z, color='#58a6ff', lw=2.5)
        ax_e.fill(chassis_x, chassis_z, color='#1f6feb', alpha=0.25)

        ax_e.plot([y_l, y_l], [zw[2] + 0.12, z_c_l], color='#e3b341', lw=2.5, ls='--')
        ax_e.plot([y_r, y_r], [zw[3] + 0.12, z_c_r], color='#e3b341', lw=2.5, ls='--')
        ax_e.plot([y_l * 0.7, y_r * 0.7], [z_c_l - 0.03, z_c_r - 0.03], color='#f0883e', lw=3)

        if not is_4post:
            # Attuatori Aerodinamici Superiori Post 6 (RL) & Post 7 (RR)
            ax_e.plot([-0.9, 0.9], [0.80, 0.80], color='#30363d', lw=4)
            ax_e.plot([y_l, y_l], [0.80, z_c_l + 0.10], color='#bc8cff', lw=3)
            ax_e.plot([y_r, y_r], [0.80, z_c_r + 0.10], color='#bc8cff', lw=3)
        else:
            ax_e.plot([-0.9, 0.9], [0.80, 0.80], color='#21262d', lw=2, linestyle=':')
            ax_e.text(0.0, 0.76, "[POST AERO 6 e 7 DISCONNESSI]",
                      color='#8b949e', fontsize=7, ha='center', fontweight='bold')

        ax_e.set_xlim(-1.0, 1.0)
        ax_e.set_ylim(-0.15, 0.90)
        ax_e.set_aspect('equal')
        ax_e.grid(True, color='#21262d', linestyle=':', alpha=0.6)
        ax_e.set_xlabel("Y Trasversale [m]", fontsize=8)

        apply_plot_theme(self.fig)
        self.draw_idle()


class SevenPostRigTab(QWidget):
    """
    Sub-Tab Ingegneristico per il Banco 7-Post Rig con Post-Processing Avanzato.
    Integrabile in main_gui_pro.py (Workspace 2) e main_gui_classic.py.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.mw = main_window
        self.worker: SevenPostRigWorker = None
        self.last_result: SevenPostResult = None
        self.baseline_result: Optional[SevenPostResult] = None
        self._cards: List[QFrame] = []

        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_fps = 30
        self.anim_speed_multiplier = 0.5

        self._build_ui()

    def _get_active_vp(self):
        """Recupera l'istanza VehicleParams attiva dalla finestra principale o genera il nominale."""
        if self.mw is not None:
            if hasattr(self.mw, '_get_current_vehicle_params'):
                try:
                    return self.mw._get_current_vehicle_params()
                except Exception:
                    pass
            vp = getattr(self.mw, 'vp', None)
            if vp is not None:
                return vp
            if hasattr(self.mw, '_active_vehicle_params') and self.mw._active_vehicle_params is not None:
                return self.mw._active_vehicle_params
        from src.models.params_vehicle import VehicleParams
        return VehicleParams()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # -------------------------------------------------------------
        # Header KPI Cards (8 Monitor Cards)
        # -------------------------------------------------------------
        kpi_bar = QHBoxLayout()
        kpi_bar.setSpacing(5)

        self.card_score = self._create_kpi_card("SETUP RIG SCORE", "— / 100", "#58a6ff")
        self.card_grip_score = self._create_kpi_card("GRIP SCORE", "— / 100", "#3fb950")
        self.card_cplv = self._create_kpi_card("CPLV MEDIO (GRIP)", "— %", "#3fb950")
        self.card_liftoff = self._create_kpi_card("LIFT-OFF", "— %", "#f85149")
        self.card_heave = self._create_kpi_card("HEAVE RMS", "— m/s²", "#58a6ff")
        self.card_pitch = self._create_kpi_card("PITCH RMS", "— rad/s²", "#58a6ff")
        self.card_roll = self._create_kpi_card("ROLL RMS", "— rad/s²", "#bc8cff")
        self.card_pkpk_rh = self._create_kpi_card("Δ RIDE HEIGHT", "— mm", "#f0883e")

        kpi_bar.addWidget(self.card_score)
        kpi_bar.addWidget(self.card_grip_score)
        kpi_bar.addWidget(self.card_cplv)
        kpi_bar.addWidget(self.card_liftoff)
        kpi_bar.addWidget(self.card_heave)
        kpi_bar.addWidget(self.card_pitch)
        kpi_bar.addWidget(self.card_roll)
        kpi_bar.addWidget(self.card_pkpk_rh)
        main_layout.addLayout(kpi_bar)

        # -------------------------------------------------------------
        # Splitter Principale: Controlli Sinistra + Visualizzatori Destra
        # -------------------------------------------------------------
        splitter = QSplitter(QtCore.Qt.Horizontal)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_widget = QWidget()
        ctrl_layout = QVBoxLayout(controls_widget)
        ctrl_layout.setContentsMargins(2, 2, 2, 2)
        ctrl_layout.setSpacing(6)

        # 0. Selezione Architettura Banco Prova (7-Post vs 4-Post)
        gb_arch = QGroupBox("Architettura Banco Prova")
        gb_arch_layout = QVBoxLayout(gb_arch)
        self.cb_rig_arch = QComboBox()
        self.cb_rig_arch.addItems([
            "7-Post Shaker Rig (4 Stradali + 3 Aero Telaio)",
            "4-Post Shaker Rig Classico (4 Stradali, Senza Aero)"
        ])
        self.cb_rig_arch.currentIndexChanged.connect(self._on_rig_arch_changed)
        gb_arch_layout.addWidget(self.cb_rig_arch)
        ctrl_layout.addWidget(gb_arch)

        # 1. Configurazione Banco & Modalità di Prova
        gb_mode = QGroupBox("Modo di Prova Shaker Rig")
        gb_mode_layout = QGridLayout(gb_mode)
        gb_mode_layout.setSpacing(4)

        gb_mode_layout.addWidget(QLabel("Modalità:"), 0, 0)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems([
            "Chirp Sweep (Logaritmico)",
            "Chirp Sweep (Lineare)",
            "Sine Dwell (Risonanza Singola)",
            "Gradino / Cordolo (Step & Obstacle)",
            "Rugosità Stradale ISO 8608"
        ])
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        gb_mode_layout.addWidget(self.cb_mode, 0, 1)

        gb_mode_layout.addWidget(QLabel("Fasatura Ruote:"), 1, 0)
        self.cb_phasing = QComboBox()
        self.cb_phasing.addItems([
            "Heave Puro [ + + + + ]",
            "Pitch Puro [ + + - - ]",
            "Roll Puro [ + - + - ]",
            "Warp / Twist [ + - - + ]",
            "Solo Ant. Sx (FL) [ + 0 0 0 ]",
            "Solo Ant. Dx (FR) [ 0 + 0 0 ]",
            "Solo Post. Sx (RL) [ 0 0 + 0 ]",
            "Solo Post. Dx (RR) [ 0 0 0 + ]"
        ])
        gb_mode_layout.addWidget(self.cb_phasing, 1, 1)

        # Frequenze Chirp
        self.lbl_freq_range = QLabel("Freq Inizio / Fine [Hz]:")
        gb_mode_layout.addWidget(self.lbl_freq_range, 2, 0)
        f_row = QHBoxLayout()
        self.sp_fstart = QDoubleSpinBox()
        self.sp_fstart.setRange(0.1, 10.0); self.sp_fstart.setValue(0.5); self.sp_fstart.setSingleStep(0.5)
        self.sp_fend = QDoubleSpinBox()
        self.sp_fend.setRange(5.0, 40.0); self.sp_fend.setValue(30.0); self.sp_fend.setSingleStep(5.0)
        f_row.addWidget(self.sp_fstart); f_row.addWidget(self.sp_fend)
        self.widget_f_row = QWidget(); self.widget_f_row.setLayout(f_row)
        gb_mode_layout.addWidget(self.widget_f_row, 2, 1)

        # Sine Dwell Frequenza
        self.lbl_dwell_freq = QLabel("Frequenza Dwell [Hz]:")
        self.sp_dwell_freq = QDoubleSpinBox()
        self.sp_dwell_freq.setRange(0.2, 35.0); self.sp_dwell_freq.setValue(2.5); self.sp_dwell_freq.setSingleStep(0.5)
        gb_mode_layout.addWidget(self.lbl_dwell_freq, 3, 0)
        gb_mode_layout.addWidget(self.sp_dwell_freq, 3, 1)

        # Step Obstacle Controlli
        self.lbl_step_type = QLabel("Tipo Ostacolo:")
        self.cb_step_type = QComboBox()
        self.cb_step_type.addItems(["Cordolo (Kerb)", "Buca (Pothole)", "Dosso Sinusoidale (Sine Bump)", "Gradino Salita (Step)"])
        gb_mode_layout.addWidget(self.lbl_step_type, 4, 0)
        gb_mode_layout.addWidget(self.cb_step_type, 4, 1)

        self.lbl_step_geom = QLabel("Quota [mm] / Durata [s]:")
        step_row = QHBoxLayout()
        self.sp_step_height = QDoubleSpinBox()
        self.sp_step_height.setRange(-50.0, 50.0); self.sp_step_height.setValue(10.0); self.sp_step_height.setSingleStep(2.0)
        self.sp_step_dur = QDoubleSpinBox()
        self.sp_step_dur.setRange(0.02, 0.50); self.sp_step_dur.setValue(0.08); self.sp_step_dur.setSingleStep(0.02)
        step_row.addWidget(self.sp_step_height); step_row.addWidget(self.sp_step_dur)
        self.widget_step_row = QWidget(); self.widget_step_row.setLayout(step_row)
        gb_mode_layout.addWidget(self.lbl_step_geom, 5, 0)
        gb_mode_layout.addWidget(self.widget_step_row, 5, 1)

        # ISO 8608 Controlli
        self.lbl_iso_class = QLabel("Classe Stradale ISO:")
        self.cb_iso_class = QComboBox()
        self.cb_iso_class.addItems(["Classe A (Autodromo)", "Classe B (Medio Bump)", "Classe C (Sconnesso)", "Classe D (Severo)"])
        self.cb_iso_class.setCurrentIndex(1)
        gb_mode_layout.addWidget(self.lbl_iso_class, 6, 0)
        gb_mode_layout.addWidget(self.cb_iso_class, 6, 1)

        self.lbl_iso_dur = QLabel("Durata Prova ISO [s]:")
        self.sp_iso_dur = QDoubleSpinBox()
        self.sp_iso_dur.setRange(2.0, 30.0); self.sp_iso_dur.setValue(8.0); self.sp_iso_dur.setSingleStep(1.0)
        gb_mode_layout.addWidget(self.lbl_iso_dur, 7, 0)
        gb_mode_layout.addWidget(self.sp_iso_dur, 7, 1)

        # Ampiezza e Durata Generali
        self.lbl_amp_dur = QLabel("Ampiezza [mm] / Durata [s]:")
        amp_row = QHBoxLayout()
        self.sp_amp = QDoubleSpinBox()
        self.sp_amp.setRange(0.5, 30.0); self.sp_amp.setValue(6.0); self.sp_amp.setSingleStep(1.0)
        self.sp_dur = QDoubleSpinBox()
        self.sp_dur.setRange(1.0, 30.0); self.sp_dur.setValue(12.0); self.sp_dur.setSingleStep(1.0)
        amp_row.addWidget(self.sp_amp); amp_row.addWidget(self.sp_dur)
        self.widget_amp_row = QWidget(); self.widget_amp_row.setLayout(amp_row)
        gb_mode_layout.addWidget(self.lbl_amp_dur, 8, 0)
        gb_mode_layout.addWidget(self.widget_amp_row, 8, 1)

        # Limite Velocità Attuatore
        gb_mode_layout.addWidget(QLabel("V_max Attuatore [m/s]:"), 9, 0)
        self.sp_vmax = QDoubleSpinBox()
        self.sp_vmax.setRange(0.10, 1.50); self.sp_vmax.setValue(0.25); self.sp_vmax.setSingleStep(0.05)
        gb_mode_layout.addWidget(self.sp_vmax, 9, 1)

        ctrl_layout.addWidget(gb_mode)

        # 2. Carichi Aerodinamici & Velocità di Pista
        self.gb_aero = QGroupBox("Attuatori Telaio & Carico Aerodinamico (Post 5-7)")
        gb_aero_layout = QGridLayout(self.gb_aero)
        gb_aero_layout.setSpacing(4)

        gb_aero_layout.addWidget(QLabel("Modalità Aero:"), 0, 0)
        self.cb_aero_mode = QComboBox()
        self.cb_aero_mode.addItems([
            "Dinamico (Proporzionale a V²)",
            "Carico Costante (Precarico N)",
            "Disattivato (Solo Massa Gravitazionale)"
        ])
        self.cb_aero_mode.currentIndexChanged.connect(self._on_aero_mode_changed)
        gb_aero_layout.addWidget(self.cb_aero_mode, 0, 1)

        gb_aero_layout.addWidget(QLabel("Velocità Virtuale [km/h]:"), 1, 0)
        self.sp_speed_kmh = QDoubleSpinBox()
        self.sp_speed_kmh.setRange(10.0, 160.0); self.sp_speed_kmh.setValue(72.0); self.sp_speed_kmh.setSingleStep(5.0)
        gb_aero_layout.addWidget(self.sp_speed_kmh, 1, 1)

        self.lbl_const_downforce = QLabel("Downforce Costante [N]:")
        self.sp_const_downforce = QDoubleSpinBox()
        self.sp_const_downforce.setRange(50.0, 4000.0); self.sp_const_downforce.setValue(800.0); self.sp_const_downforce.setSingleStep(50.0)
        gb_aero_layout.addWidget(self.lbl_const_downforce, 2, 0)
        gb_aero_layout.addWidget(self.sp_const_downforce, 2, 1)

        gb_aero_layout.addWidget(QLabel("Ripartizione Aero Ant [%]:"), 3, 0)
        self.sp_aero_bal = QDoubleSpinBox()
        self.sp_aero_bal.setRange(30.0, 65.0); self.sp_aero_bal.setValue(45.0); self.sp_aero_bal.setSingleStep(1.0)
        gb_aero_layout.addWidget(self.sp_aero_bal, 3, 1)

        ctrl_layout.addWidget(self.gb_aero)

        # 3. Setup Sospensioni & Taratura TTX25 MkII (Overrides)
        gb_setup = QGroupBox("Setup Sospensioni & TTX25 (Overrides Banco)")
        gb_setup_layout = QGridLayout(gb_setup)
        gb_setup_layout.setSpacing(4)

        self.btn_load_setup = QPushButton("Sincronizza Setup da Veicolo")
        self.btn_load_setup.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 4px;")
        self.btn_load_setup.clicked.connect(self._on_load_setup_clicked)
        gb_setup_layout.addWidget(self.btn_load_setup, 0, 0, 1, 3)

        gb_setup_layout.addWidget(QLabel("<b>Parametro</b>"), 1, 0)
        gb_setup_layout.addWidget(QLabel("<b>Ant.</b>"), 1, 1)
        gb_setup_layout.addWidget(QLabel("<b>Post.</b>"), 1, 2)

        # Molle [N/mm]
        gb_setup_layout.addWidget(QLabel("Rigidità Molla [N/mm]:"), 2, 0)
        self.sp_kz_f = QDoubleSpinBox(); self.sp_kz_f.setRange(15.0, 120.0); self.sp_kz_f.setValue(40.0); self.sp_kz_f.setSingleStep(2.5)
        self.sp_kz_r = QDoubleSpinBox(); self.sp_kz_r.setRange(15.0, 120.0); self.sp_kz_r.setValue(45.0); self.sp_kz_r.setSingleStep(2.5)
        gb_setup_layout.addWidget(self.sp_kz_f, 2, 1); gb_setup_layout.addWidget(self.sp_kz_r, 2, 2)

        # Barre ARB [Nm/rad]
        gb_setup_layout.addWidget(QLabel("Rigidità ARB [Nm/rad]:"), 3, 0)
        self.sp_kphi_f = QDoubleSpinBox(); self.sp_kphi_f.setRange(0.0, 2000.0); self.sp_kphi_f.setValue(600.0); self.sp_kphi_f.setSingleStep(50.0)
        self.sp_kphi_r = QDoubleSpinBox(); self.sp_kphi_r.setRange(0.0, 2000.0); self.sp_kphi_r.setValue(450.0); self.sp_kphi_r.setSingleStep(50.0)
        gb_setup_layout.addWidget(self.sp_kphi_f, 3, 1); gb_setup_layout.addWidget(self.sp_kphi_r, 3, 2)

        # Luce Bumpstop [mm]
        gb_setup_layout.addWidget(QLabel("Luce Bumpstop [mm]:"), 4, 0)
        self.sp_gap_f = QDoubleSpinBox(); self.sp_gap_f.setRange(2.0, 50.0); self.sp_gap_f.setValue(18.0); self.sp_gap_f.setSingleStep(2.0)
        self.sp_gap_r = QDoubleSpinBox(); self.sp_gap_r.setRange(2.0, 50.0); self.sp_gap_r.setValue(22.0); self.sp_gap_r.setSingleStep(2.0)
        gb_setup_layout.addWidget(self.sp_gap_f, 4, 1); gb_setup_layout.addWidget(self.sp_gap_r, 4, 2)

        # LSC (clicks from closed: 0 stiff, 25 soft)
        gb_setup_layout.addWidget(QLabel("TTX25 LSC [clic chiuso]:"), 5, 0)
        self.sp_lsc_f = QSpinBox(); self.sp_lsc_f.setRange(0, 25); self.sp_lsc_f.setValue(10)
        self.sp_lsc_r = QSpinBox(); self.sp_lsc_r.setRange(0, 25); self.sp_lsc_r.setValue(10)
        gb_setup_layout.addWidget(self.sp_lsc_f, 5, 1); gb_setup_layout.addWidget(self.sp_lsc_r, 5, 2)

        # HSC (turns from open: 0 soft, 4.3 stiff)
        gb_setup_layout.addWidget(QLabel("TTX25 HSC [giri aperto]:"), 6, 0)
        self.sp_hsc_f = QDoubleSpinBox(); self.sp_hsc_f.setRange(0.0, 4.3); self.sp_hsc_f.setValue(2.0); self.sp_hsc_f.setSingleStep(0.5)
        self.sp_hsc_r = QDoubleSpinBox(); self.sp_hsc_r.setRange(0.0, 4.3); self.sp_hsc_r.setValue(2.0); self.sp_hsc_r.setSingleStep(0.5)
        gb_setup_layout.addWidget(self.sp_hsc_f, 6, 1); gb_setup_layout.addWidget(self.sp_hsc_r, 6, 2)

        # LSR (clicks from closed)
        gb_setup_layout.addWidget(QLabel("TTX25 LSR [clic chiuso]:"), 7, 0)
        self.sp_lsr_f = QSpinBox(); self.sp_lsr_f.setRange(0, 25); self.sp_lsr_f.setValue(10)
        self.sp_lsr_r = QSpinBox(); self.sp_lsr_r.setRange(0, 25); self.sp_lsr_r.setValue(10)
        gb_setup_layout.addWidget(self.sp_lsr_f, 7, 1); gb_setup_layout.addWidget(self.sp_lsr_r, 7, 2)

        # HSR (turns from open)
        gb_setup_layout.addWidget(QLabel("TTX25 HSR [giri aperto]:"), 8, 0)
        self.sp_hsr_f = QDoubleSpinBox(); self.sp_hsr_f.setRange(0.0, 4.3); self.sp_hsr_f.setValue(2.0); self.sp_hsr_f.setSingleStep(0.5)
        self.sp_hsr_r = QDoubleSpinBox(); self.sp_hsr_r.setRange(0.0, 4.3); self.sp_hsr_r.setValue(2.0); self.sp_hsr_r.setSingleStep(0.5)
        gb_setup_layout.addWidget(self.sp_hsr_f, 8, 1); gb_setup_layout.addWidget(self.sp_hsr_r, 8, 2)

        ctrl_layout.addWidget(gb_setup)

        # 4. Impostazioni Selezionabili di Post-Processing
        gb_post = QGroupBox("Opzioni di Visualizzazione & Post-Processing")
        gb_post_layout = QGridLayout(gb_post)
        gb_post_layout.setSpacing(4)

        gb_post_layout.addWidget(QLabel("Unità FRF Bode:"), 0, 0)
        self.cb_frf_unit = QComboBox()
        self.cb_frf_unit.addItems(["Decibel [dB] (20 log10)", "Lineare [-]"])
        self.cb_frf_unit.currentIndexChanged.connect(self._re_render_active_plots)
        gb_post_layout.addWidget(self.cb_frf_unit, 0, 1)

        gb_post_layout.addWidget(QLabel("Scala Frequenze:"), 1, 0)
        self.cb_freq_scale = QComboBox()
        self.cb_freq_scale.addItems(["Logaritmica", "Lineare"])
        self.cb_freq_scale.currentIndexChanged.connect(self._re_render_active_plots)
        gb_post_layout.addWidget(self.cb_freq_scale, 1, 1)

        gb_post_layout.addWidget(QLabel("Decomposizione CPLV:"), 2, 0)
        self.cb_cplv_view = QComboBox()
        self.cb_cplv_view.addItems(["Multibanda (Overall + Low + High)", "Spettro Continuo Ruote"])
        self.cb_cplv_view.currentIndexChanged.connect(self._re_render_active_plots)
        gb_post_layout.addWidget(self.cb_cplv_view, 2, 1)

        self.chk_show_thresholds = QCheckBox("Evidenzia Soglie Critiche (Lift-off & Target)")
        self.chk_show_thresholds.setChecked(True)
        self.chk_show_thresholds.stateChanged.connect(self._re_render_active_plots)
        gb_post_layout.addWidget(self.chk_show_thresholds, 3, 0, 1, 2)

        ctrl_layout.addWidget(gb_post)

        # 5. Pulsanti di Esecuzione & Azioni
        self.btn_run = QPushButton("Avvia Simulazione 7-Post Rig")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #238636; color: white; padding: 7px; font-size: 10pt;")
        self.btn_run.clicked.connect(self._on_run_clicked)
        ctrl_layout.addWidget(self.btn_run)

        self.btn_modal = QPushButton("Analisi Modale Linearizzata")
        self.btn_modal.clicked.connect(self._on_modal_clicked)
        ctrl_layout.addWidget(self.btn_modal)

        self.btn_export = QPushButton("Esporta Dati Telemetria (CSV)")
        self.btn_export.clicked.connect(self._on_export_clicked)
        ctrl_layout.addWidget(self.btn_export)

        self.btn_export_report = QPushButton("Esporta Report Tecnico (TXT)")
        self.btn_export_report.clicked.connect(self._on_export_report_clicked)
        ctrl_layout.addWidget(self.btn_export_report)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        ctrl_layout.addStretch()
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(330)
        controls_scroll.setMaximumWidth(410)
        splitter.addWidget(controls_scroll)

        # -------------------------------------------------------------
        # Area Destra: Visualizzatori Analitici a 8 Sub-Tab
        # -------------------------------------------------------------
        self.tabs_view = QTabWidget()

        # Tab 1: Spettro FRF, Fase & Coerenza (Bode Completo)
        self.fig_bode = Figure(figsize=(8, 6), dpi=100)
        self.canvas_bode = FigureCanvas(self.fig_bode)
        tab_bode = QWidget()
        v_bode = QVBoxLayout(tab_bode)
        v_bode.setContentsMargins(2, 2, 2, 2)
        v_bode.addWidget(NavigationToolbar(self.canvas_bode, tab_bode))

        bench_bar = QHBoxLayout()
        bench_bar.setContentsMargins(4, 2, 4, 2)
        bench_bar.setSpacing(8)

        self.chk_show_model_bench = QCheckBox("Mostra Risonanze Teoriche Modello (Autovalori / Quarter-Car)")
        self.chk_show_model_bench.setChecked(True)
        self.chk_show_model_bench.stateChanged.connect(self._re_render_active_plots)
        bench_bar.addWidget(self.chk_show_model_bench)

        self.chk_compare_baseline = QCheckBox("Overlay Baseline")
        self.chk_compare_baseline.setChecked(True)
        self.chk_compare_baseline.stateChanged.connect(self._re_render_active_plots)
        bench_bar.addWidget(self.chk_compare_baseline)

        self.btn_save_baseline = QPushButton("Fissa come Baseline")
        self.btn_save_baseline.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 2px 8px;")
        self.btn_save_baseline.clicked.connect(self._on_save_baseline_clicked)
        bench_bar.addWidget(self.btn_save_baseline)

        self.btn_clear_baseline = QPushButton("Resetta Baseline")
        self.btn_clear_baseline.setStyleSheet("background-color: #21262d; color: #8b949e; padding: 2px 8px;")
        self.btn_clear_baseline.clicked.connect(self._on_clear_baseline_clicked)
        bench_bar.addWidget(self.btn_clear_baseline)

        self.lbl_baseline_status = QLabel("Nessuna baseline")
        self.lbl_baseline_status.setStyleSheet("color: #8b949e; font-size: 8pt;")
        bench_bar.addWidget(self.lbl_baseline_status)

        bench_bar.addStretch()
        v_bode.addLayout(bench_bar)

        v_bode.addWidget(self.canvas_bode)
        self.tabs_view.addTab(tab_bode, "Spettro FRF & Coerenza")

        # Tab 2: Grip & CPLV Multibanda
        self.fig_cplv = Figure(figsize=(8, 6), dpi=100)
        self.canvas_cplv = FigureCanvas(self.fig_cplv)
        tab_cplv = QWidget()
        v_cplv = QVBoxLayout(tab_cplv)
        v_cplv.setContentsMargins(2, 2, 2, 2)
        v_cplv.addWidget(NavigationToolbar(self.canvas_cplv, tab_cplv))
        v_cplv.addWidget(self.canvas_cplv)
        self.tabs_view.addTab(tab_cplv, "Grip & CPLV Multibanda")

        # Tab 3: Telemetria 7 Attuatori Canali Temporali
        self.fig_telem = Figure(figsize=(8, 6), dpi=100)
        self.canvas_telem = FigureCanvas(self.fig_telem)
        tab_telem = QWidget()
        v_telem = QVBoxLayout(tab_telem)
        v_telem.setContentsMargins(2, 2, 2, 2)
        v_telem.addWidget(NavigationToolbar(self.canvas_telem, tab_telem))
        v_telem.addWidget(self.canvas_telem)
        self.tabs_view.addTab(tab_telem, "Telemetria 7 Attuatori")

        # Tab 4: Diagnostica TTX25 & Curve Lissajous
        self.fig_ttx = Figure(figsize=(8, 6), dpi=100)
        self.canvas_ttx = FigureCanvas(self.fig_ttx)
        tab_ttx = QWidget()
        v_ttx = QVBoxLayout(tab_ttx)
        v_ttx.setContentsMargins(2, 2, 2, 2)
        v_ttx.addWidget(NavigationToolbar(self.canvas_ttx, tab_ttx))
        v_ttx.addWidget(self.canvas_ttx)
        self.tabs_view.addTab(tab_ttx, "Diagnostica TTX25 & Bilstrom")

        # Tab 5: Piattaforma Aerodinamica & Dinamica Telaio
        self.fig_aero = Figure(figsize=(8, 6), dpi=100)
        self.canvas_aero = FigureCanvas(self.fig_aero)
        tab_aero = QWidget()
        v_aero = QVBoxLayout(tab_aero)
        v_aero.setContentsMargins(2, 2, 2, 2)
        v_aero.addWidget(NavigationToolbar(self.canvas_aero, tab_aero))
        v_aero.addWidget(self.canvas_aero)
        self.tabs_view.addTab(tab_aero, "Piattaforma Aerodinamica")

        # Tab 6: Tabella Modi Propri Linearizzati & Piano s
        self.fig_modal = Figure(figsize=(8, 4), dpi=100)
        self.canvas_modal = FigureCanvas(self.fig_modal)
        self.table_modes = QTableWidget()
        self.table_modes.setColumnCount(6)
        self.table_modes.setHorizontalHeaderLabels([
            "Modo Dinamico", "Freq Modello [Hz]", "Freq Banco [Hz]", "Delta %", "Smorzamento ζ [-]", "Correlazione Modello/Banco"
        ])
        self.table_modes.horizontalHeader().setStretchLastSection(True)
        tab_modes = QWidget()
        v_modes = QVBoxLayout(tab_modes)
        v_modes.addWidget(self.table_modes, stretch=2)
        v_modes.addWidget(self.canvas_modal, stretch=3)
        self.tabs_view.addTab(tab_modes, "Modi Propri & Benchmark Modello")

        # Tab 7: Componente Grafica 2D Animata del Banco Prova
        tab_rig2d = QWidget()
        v_rig2d = QVBoxLayout(tab_rig2d)
        v_rig2d.setContentsMargins(2, 2, 2, 2)
        self.canvas_rig2d = RigSchematic2DCanvas(tab_rig2d)

        anim_bar = QHBoxLayout()
        self.btn_anim_play = QPushButton("Play")
        self.btn_anim_play.setFixedWidth(70)
        self.btn_anim_play.clicked.connect(self._toggle_anim_play)
        anim_bar.addWidget(self.btn_anim_play)

        self.slider_anim = QSlider(QtCore.Qt.Horizontal)
        self.slider_anim.setRange(0, 100)
        self.slider_anim.setValue(0)
        self.slider_anim.valueChanged.connect(self._on_slider_scrub)
        anim_bar.addWidget(self.slider_anim)

        self.lbl_anim_time = QLabel("0.00 / 0.00 s")
        self.lbl_anim_time.setFixedWidth(90)
        anim_bar.addWidget(self.lbl_anim_time)

        anim_bar.addWidget(QLabel("Velocità:"))
        self.cb_anim_speed = QComboBox()
        self.cb_anim_speed.addItems(["0.25x (Rallentatore)", "0.50x", "1.00x (Reale)"])
        self.cb_anim_speed.setCurrentIndex(1)
        self.cb_anim_speed.currentIndexChanged.connect(self._on_anim_speed_changed)
        anim_bar.addWidget(self.cb_anim_speed)

        v_rig2d.addWidget(self.canvas_rig2d)
        v_rig2d.addLayout(anim_bar)
        self.tabs_view.addTab(tab_rig2d, "Vista Banco 2D (Cinematica)")

        # Tab 8: Report Tecnico Ingegneristico
        tab_report = QWidget()
        v_rep = QVBoxLayout(tab_report)
        v_rep.setContentsMargins(4, 4, 4, 4)
        rep_btn_bar = QHBoxLayout()
        self.btn_copy_rep = QPushButton("Copia Report negli Appunti")
        self.btn_copy_rep.clicked.connect(self._on_copy_report_clicked)
        self.btn_save_rep = QPushButton("Salva Report su File (.txt)")
        self.btn_save_rep.clicked.connect(self._on_export_report_clicked)
        rep_btn_bar.addWidget(self.btn_copy_rep)
        rep_btn_bar.addWidget(self.btn_save_rep)
        rep_btn_bar.addStretch()
        v_rep.addLayout(rep_btn_bar)

        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setFont(QtGui.QFont("Courier New", 9))
        self.txt_report.setPlaceholderText("Il report tecnico di dettaglio verrà generato automaticamente al termine della simulazione...")
        v_rep.addWidget(self.txt_report)
        self.tabs_view.addTab(tab_report, "Report Tecnico Ingegneristico")

        splitter.addWidget(self.tabs_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 940])

        main_layout.addWidget(splitter, stretch=1)

        # Inizializza visibilità controlli per modalità di default
        self._on_rig_arch_changed()
        self._on_mode_changed()
        self._on_aero_mode_changed()

        # Applica temi iniziali
        for f in [self.fig_bode, self.fig_cplv, self.fig_telem, self.fig_ttx, self.fig_aero, self.fig_modal, self.canvas_rig2d.fig]:
            apply_plot_theme(f)

    def _create_kpi_card(self, title: str, value: str, color_accent: str) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.color_accent = color_accent
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-top: 3px solid {color_accent};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 8pt; color: #8b949e; font-weight: bold;")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet(f"font-size: 11pt; color: {color_accent}; font-weight: bold;")
        lbl_v.setObjectName(f"val_{title.split()[0].lower()}")

        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        card.lbl_title = lbl_t
        card.lbl_val = lbl_v
        if not hasattr(self, '_cards'):
            self._cards = []
        self._cards.append(card)
        return card

    def apply_theme(self, theme_name: Optional[str] = None):
        """Sincronizza i canvas Matplotlib e le card KPI con il tema attivo."""
        for f in [self.fig_bode, self.fig_cplv, self.fig_telem, self.fig_ttx, self.fig_aero, self.fig_modal, self.canvas_rig2d.fig]:
            apply_plot_theme(f, theme_name=theme_name)
        for c in [self.canvas_bode, self.canvas_cplv, self.canvas_telem, self.canvas_ttx, self.canvas_aero, self.canvas_modal, self.canvas_rig2d]:
            try:
                c.draw_idle()
            except Exception:
                pass
        t = get_theme_palette(theme_name or 'dark')
        for card in getattr(self, '_cards', []):
            color_accent = getattr(card, 'color_accent', '#58a6ff')
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {t.get('bg_card', '#161b22')};
                    border: 1px solid {t.get('border', '#30363d')};
                    border-top: 3px solid {color_accent};
                    border-radius: 4px;
                    padding: 4px;
                }}
            """)
            if hasattr(card, 'lbl_title'):
                card.lbl_title.setStyleSheet(f"font-size: 8pt; color: {t.get('text_secondary', '#8b949e')}; font-weight: bold;")

    def _on_rig_arch_changed(self):
        is_4post = getattr(self, 'cb_rig_arch', None) and self.cb_rig_arch.currentIndex() == 1
        if hasattr(self, 'gb_aero'):
            self.gb_aero.setEnabled(not is_4post)
            if is_4post:
                self.gb_aero.setTitle("Attuatori Telaio (DISCONNESSI in modalità 4-Post)")
            else:
                self.gb_aero.setTitle("Attuatori Telaio & Carico Aerodinamico (Post 5-7)")
        if hasattr(self, 'btn_run'):
            if is_4post:
                self.btn_run.setText("Avvia Simulazione 4-Post Rig")
            else:
                self.btn_run.setText("Avvia Simulazione 7-Post Rig")
        if hasattr(self, 'card_score') and hasattr(self.card_score, 'lbl_title'):
            self.card_score.lbl_title.setText("SETUP 4-POST SCORE" if is_4post else "SETUP RIG SCORE")
        if hasattr(self.mw, 'statusBar'):
            arch_label = "4-Post Classico (4 Attuatori Stradali, Senza Aero)" if is_4post else "7-Post Shaker Rig (4 Stradali + 3 Aero)"
            self.mw.statusBar().showMessage(f"Architettura selezionata: {arch_label}", 2500)

    def _on_save_baseline_clicked(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Baseline Rig", "Nessuna simulazione disponibile da impostare come baseline.")
            return
        self.baseline_result = self.last_result
        arch_name = "4-Post" if self.baseline_result.rig_architecture == '4_post' else "7-Post"
        self.lbl_baseline_status.setText(f"Baseline: {arch_name} (Score {self.baseline_result.score_overall:.1f})")
        self.lbl_baseline_status.setStyleSheet("color: #3fb950; font-size: 8pt; font-weight: bold;")
        self._re_render_active_plots()
        if hasattr(self.mw, 'statusBar'):
            self.mw.statusBar().showMessage(f"[{arch_name} Rig] Setup fissato come baseline di confronto.", 3000)

    def _on_clear_baseline_clicked(self):
        self.baseline_result = None
        self.lbl_baseline_status.setText("Nessuna baseline")
        self.lbl_baseline_status.setStyleSheet("color: #8b949e; font-size: 8pt;")
        self._re_render_active_plots()
        if hasattr(self.mw, 'statusBar'):
            self.mw.statusBar().showMessage("[Rig] Baseline di confronto azzerata.", 2000)

    def _on_mode_changed(self):
        idx = self.cb_mode.currentIndex()
        # 0: Chirp Log, 1: Chirp Lin, 2: Sine Dwell, 3: Step & Obstacle, 4: ISO 8608
        is_chirp = idx in (0, 1)
        is_dwell = idx == 2
        is_step = idx == 3
        is_iso = idx == 4

        self.cb_phasing.setEnabled(is_chirp or is_dwell or is_step)

        # Chirp controls
        self.lbl_freq_range.setVisible(is_chirp)
        self.widget_f_row.setVisible(is_chirp)

        # Sine Dwell controls
        self.lbl_dwell_freq.setVisible(is_dwell)
        self.sp_dwell_freq.setVisible(is_dwell)

        # Step Obstacle controls
        self.lbl_step_type.setVisible(is_step)
        self.cb_step_type.setVisible(is_step)
        self.lbl_step_geom.setVisible(is_step)
        self.widget_step_row.setVisible(is_step)

        # ISO controls
        self.lbl_iso_class.setVisible(is_iso)
        self.cb_iso_class.setVisible(is_iso)
        self.lbl_iso_dur.setVisible(is_iso)
        self.sp_iso_dur.setVisible(is_iso)

        # Amp/Dur general controls
        self.lbl_amp_dur.setVisible(not is_iso)
        self.widget_amp_row.setVisible(not is_iso)

    def _on_aero_mode_changed(self):
        idx = self.cb_aero_mode.currentIndex()
        is_dynamic = idx == 0
        is_constant = idx == 1

        self.sp_speed_kmh.setEnabled(is_dynamic)
        self.lbl_const_downforce.setVisible(is_constant)
        self.sp_const_downforce.setVisible(is_constant)

    def _on_load_setup_clicked(self):
        vp = self._get_active_vp()



        self.sp_kz_f.setValue(float(getattr(vp, 'K_z_f', 40000.0)) / 1e3)
        self.sp_kz_r.setValue(float(getattr(vp, 'K_z_r', 45000.0)) / 1e3)
        self.sp_kphi_f.setValue(float(getattr(vp, 'K_phi_f', 600.0)))
        self.sp_kphi_r.setValue(float(getattr(vp, 'K_phi_r', 450.0)))

        self.sp_lsc_f.setValue(int(getattr(vp, 'damper_lsc_clicks_f', 10)))
        self.sp_lsc_r.setValue(int(getattr(vp, 'damper_lsc_clicks_r', 10)))
        self.sp_hsc_f.setValue(float(getattr(vp, 'damper_hsc_turns_f', 2.0)))
        self.sp_hsc_r.setValue(float(getattr(vp, 'damper_hsc_turns_r', 2.0)))
        self.sp_lsr_f.setValue(int(getattr(vp, 'damper_lsr_clicks_f', 10)))
        self.sp_lsr_r.setValue(int(getattr(vp, 'damper_lsr_clicks_r', 10)))
        self.sp_hsr_f.setValue(float(getattr(vp, 'damper_hsr_turns_f', 2.0)))
        self.sp_hsr_r.setValue(float(getattr(vp, 'damper_hsr_turns_r', 2.0)))

        if hasattr(self.mw, 'statusBar'):
            self.mw.statusBar().showMessage("[7-Post Rig] Parametri sospensione sincronizzati con successo dal veicolo.", 3000)

    def _get_active_config(self) -> SevenPostConfig:
        mode_idx = self.cb_mode.currentIndex()
        if mode_idx == 0:
            mode_str = 'chirp'; sweep_type = 'log'
        elif mode_idx == 1:
            mode_str = 'chirp'; sweep_type = 'linear'
        elif mode_idx == 2:
            mode_str = 'sine_dwell'; sweep_type = 'log'
        elif mode_idx == 3:
            mode_str = 'step_bump'; sweep_type = 'log'
        else:
            mode_str = 'iso_road'; sweep_type = 'log'

        phasing_map = {
            0: 'heave', 1: 'pitch', 2: 'roll', 3: 'warp',
            4: 'fl_only', 5: 'fr_only', 6: 'rl_only', 7: 'rr_only'
        }
        phasing_str = phasing_map.get(self.cb_phasing.currentIndex(), 'heave')

        step_types = ['kerb', 'pothole', 'sine_bump', 'step']
        step_type_str = step_types[self.cb_step_type.currentIndex()]

        iso_classes = ['A', 'B', 'C', 'D']
        iso_class_str = iso_classes[self.cb_iso_class.currentIndex()]

        is_4post = getattr(self, 'cb_rig_arch', None) and self.cb_rig_arch.currentIndex() == 1
        rig_arch_str = '4_post' if is_4post else '7_post'

        aero_modes = ['speed_proportional', 'constant', 'none']
        aero_str = 'none' if is_4post else aero_modes[self.cb_aero_mode.currentIndex()]

        speed_mps = self.sp_speed_kmh.value() / 3.6

        # Overrides dictionary
        overrides = {
            'K_z_f': self.sp_kz_f.value() * 1e3,
            'K_z_r': self.sp_kz_r.value() * 1e3,
            'K_phi_f': self.sp_kphi_f.value(),
            'K_phi_r': self.sp_kphi_r.value(),
            'gap_f': self.sp_gap_f.value() * 1e-3,
            'gap_r': self.sp_gap_r.value() * 1e-3,
            'aero_bal': self.sp_aero_bal.value() / 100.0,
            'lsc_f': float(self.sp_lsc_f.value()),
            'hsc_f': float(self.sp_hsc_f.value()),
            'lsr_f': float(self.sp_lsr_f.value()),
            'hsr_f': float(self.sp_hsr_f.value()),
            'lsc_r': float(self.sp_lsc_r.value()),
            'hsc_r': float(self.sp_hsc_r.value()),
            'lsr_r': float(self.sp_lsr_r.value()),
            'hsr_r': float(self.sp_hsr_r.value()),
        }

        dur = self.sp_iso_dur.value() if mode_str == 'iso_road' else self.sp_dur.value()

        return SevenPostConfig(
            rig_architecture=rig_arch_str,
            mode=mode_str,
            phasing=phasing_str,
            f_start=self.sp_fstart.value(),
            f_end=self.sp_fend.value(),
            sweep_duration=dur,
            sweep_type=sweep_type,
            amplitude_mm=self.sp_amp.value(),
            v_max_actuator_mps=self.sp_vmax.value(),
            dwell_frequency=self.sp_dwell_freq.value(),
            step_type=step_type_str,
            step_height_mm=self.sp_step_height.value(),
            step_duration_s=self.sp_step_dur.value(),
            aero_mode=aero_str,
            sim_speed_mps=speed_mps,
            constant_downforce_n=self.sp_const_downforce.value(),
            iso_class=iso_class_str,
            iso_duration=self.sp_iso_dur.value(),
            overrides=overrides,
            dt=0.002
        )

    def _apply_ui_damper_settings(self, vp):
        """Applica i clic modificati dall'utente sull'istanza VehicleParams."""
        vp.damper_mode = 'ttx25_nonlinear'
        vp.damper_lsc_clicks_f = float(self.sp_lsc_f.value())
        vp.damper_lsc_clicks_r = float(self.sp_lsc_r.value())
        vp.damper_hsc_turns_f = float(self.sp_hsc_f.value())
        vp.damper_hsc_turns_r = float(self.sp_hsc_r.value())
        vp.damper_lsr_clicks_f = float(self.sp_lsr_f.value())
        vp.damper_lsr_clicks_r = float(self.sp_lsr_r.value())
        vp.damper_hsr_turns_f = float(self.sp_hsr_f.value())
        vp.damper_hsr_turns_r = float(self.sp_hsr_r.value())

    def _on_run_clicked(self):
        vp = self._get_active_vp()


        self._apply_ui_damper_settings(vp)
        cfg = self._get_active_config()

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Simulazione in corso...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = SevenPostRigWorker(vp, cfg)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        if hasattr(self.mw, 'statusBar'):
            self.mw.statusBar().showMessage(f"[7-Post Rig] {msg}", 2000)

    def _on_worker_finished(self, res: SevenPostResult):
        self.last_result = res
        self.btn_run.setEnabled(True)
        is_4post = getattr(res, 'rig_architecture', '7_post') == '4_post'
        self.btn_run.setText(f"Avvia Simulazione {'4-Post' if is_4post else '7-Post'} Rig")
        self.progress_bar.setVisible(False)

        # Aggiorna KPI cards
        self.card_score.lbl_title.setText("SETUP 4-POST SCORE" if is_4post else "SETUP RIG SCORE")
        self.card_score.lbl_val.setText(f"{res.score_overall:.1f} / 100")
        self.card_grip_score.lbl_val.setText(f"{res.score_grip:.1f} / 100")
        self.card_cplv.lbl_val.setText(f"{res.cplv['avg']*100:.1f} %")
        lift_total = 0.25 * (res.lift_off_duration_pct['fl'] + res.lift_off_duration_pct['fr'] +
                             res.lift_off_duration_pct['rl'] + res.lift_off_duration_pct['rr'])
        self.card_liftoff.lbl_val.setText(f"{lift_total:.1f} %")
        self.card_heave.lbl_val.setText(f"{res.rms_heave_accel:.2f} m/s²")
        self.card_pitch.lbl_val.setText(f"{res.rms_pitch_accel:.2f} rad/s²")
        self.card_roll.lbl_val.setText(f"{res.rms_roll_accel:.2f} rad/s²")
        self.card_pkpk_rh.lbl_val.setText(f"{res.pkpk_ride_height_f_mm:.1f} / {res.pkpk_ride_height_r_mm:.1f} mm")

        # Aggiorna tabella benchmark modale con il modello
        if res.model_comparison:
            self._update_modal_table_from_comparison(res.model_comparison)

        # Rendering grafici analitici
        self._re_render_active_plots()

        # Aggiorna visualizzatore 2D
        self.canvas_rig2d.set_result(res)
        self.slider_anim.setMaximum(len(res.time) - 1)
        self.slider_anim.setValue(0)
        self.lbl_anim_time.setText(f"0.00 / {res.time[-1]:.2f} s")

        # Genera Report Tecnico
        vp = self._get_active_vp()
        rig = SevenPostRig(vp)
        report_text = rig.generate_engineering_report(res)
        self.txt_report.setPlainText(report_text)

    def _on_worker_failed(self, err_msg: str):
        self.btn_run.setEnabled(True)
        is_4post = getattr(self, 'cb_rig_arch', None) and self.cb_rig_arch.currentIndex() == 1
        self.btn_run.setText(f"Avvia Simulazione {'4-Post' if is_4post else '7-Post'} Rig")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Errore Simulazione Shaker Rig", f"Errore durante l'esecuzione:\n{err_msg}")

    def _re_render_active_plots(self):
        """Rigenera tutti i grafici post-processing tenendo conto delle opzioni utente (dB/lin, log/lin)."""
        if self.last_result is None:
            return
        res = self.last_result
        self._render_bode_plots(res)
        self._render_cplv_plots(res)
        self._render_telemetry_plots(res)
        self._render_ttx_diagnostics(res)
        self._render_aero_platform_plots(res)

    def _render_bode_plots(self, res: SevenPostResult):
        self.fig_bode.clear()
        use_db = self.cb_frf_unit.currentIndex() == 0
        use_log_x = self.cb_freq_scale.currentIndex() == 0
        is_4post = getattr(res, 'rig_architecture', '7_post') == '4_post'
        arch_title = "4-Post Classico" if is_4post else "7-Post Rig"

        ax1 = self.fig_bode.add_subplot(311)
        ax2 = self.fig_bode.add_subplot(312, sharex=ax1)
        ax3 = self.fig_bode.add_subplot(313, sharex=ax1)

        f = res.freqs
        valid = (f >= 0.3) & (f <= 35.0)
        f_v = f[valid]

        # 1. Magnitudo Trasmissibilità (dB o Lineare)
        if use_db:
            mag_h = 20.0 * np.log10(np.maximum(res.transmissibility_heave[valid], 1e-4))
            mag_p = 20.0 * np.log10(np.maximum(res.transmissibility_pitch[valid], 1e-4))
            mag_r = 20.0 * np.log10(np.maximum(res.transmissibility_roll[valid], 1e-4))
            ax1.set_ylabel("Magnitudo [dB]", fontsize=8)
            ax1.set_title(f"Bode FRF ({arch_title}) — Magnitudo Trasmissibilità Telaio", fontsize=9, fontweight='bold', color='#f0f6fc')
        else:
            mag_h = res.transmissibility_heave[valid]
            mag_p = res.transmissibility_pitch[valid]
            mag_r = res.transmissibility_roll[valid]
            ax1.set_ylabel("Trasmissibilità [-]", fontsize=8)
            ax1.set_title(f"Bode FRF ({arch_title}) — Trasmissibilità Lineare Telaio", fontsize=9, fontweight='bold', color='#f0f6fc')

        ax1.plot(f_v, mag_h, color='#58a6ff', lw=2, label='Chassis Heave (Banco)')
        ax1.plot(f_v, mag_p, color='#f0883e', lw=1.8, label='Chassis Pitch (Banco)')
        ax1.plot(f_v, mag_r, color='#bc8cff', lw=1.8, label='Chassis Roll (Banco)')

        # Annotazione picchi di risonanza identificati sul banco
        for peak in res.resonance_peaks:
            p_f = peak.get('frequency_hz', peak.get('freq_hz', 0.0))
            if 0.3 <= p_f <= 35.0:
                ax1.axvline(p_f, color='#e3b341', linestyle=':', lw=1.2, alpha=0.7)
                ax1.annotate(f"{p_f:.1f}Hz\n({peak['mode']})", xy=(p_f, ax1.get_ylim()[1] if not use_db else 5.0),
                             xytext=(p_f + 0.3, ax1.get_ylim()[1] * 0.8 if not use_db else 0.0),
                             fontsize=7, color='#e3b341', fontweight='bold')

        # Benchmark con frequenze proprie teoriche del modello (Quarter-Car & Autovalori)
        if getattr(self, 'chk_show_model_bench', None) and self.chk_show_model_bench.isChecked():
            if res.model_comparison and 'analytical' in res.model_comparison:
                ana = res.model_comparison['analytical']
                fn_h = ana.get('heave_approx_hz', 0.0)
                fn_p = ana.get('pitch_approx_hz', 0.0)
                fn_r = ana.get('roll_approx_hz', 0.0)
                fn_hf = ana.get('hop_front_hz', 0.0)

                if 0.3 <= fn_h <= 35.0:
                    ax1.axvline(fn_h, color='#3fb950', linestyle='--', lw=1.3, alpha=0.85, label=f'Modello Heave ({fn_h:.1f} Hz)')
                if 0.3 <= fn_p <= 35.0:
                    ax1.axvline(fn_p, color='#d29922', linestyle='--', lw=1.3, alpha=0.85, label=f'Modello Pitch ({fn_p:.1f} Hz)')
                if 0.3 <= fn_r <= 35.0:
                    ax1.axvline(fn_r, color='#bc8cff', linestyle='--', lw=1.3, alpha=0.85, label=f'Modello Roll ({fn_r:.1f} Hz)')
                if 0.3 <= fn_hf <= 35.0:
                    ax1.axvline(fn_hf, color='#f85149', linestyle='--', lw=1.3, alpha=0.85, label=f'Modello Hop Ant ({fn_hf:.1f} Hz)')

        # Overlay con Baseline (se fissata dall'utente)
        if getattr(self, 'chk_compare_baseline', None) and self.chk_compare_baseline.isChecked() and self.baseline_result is not None:
            b_res = self.baseline_result
            fb = b_res.freqs
            valid_b = (fb >= 0.3) & (fb <= 35.0)
            fb_v = fb[valid_b]
            if len(fb_v) > 0:
                if use_db:
                    b_mag_h = 20.0 * np.log10(np.maximum(b_res.transmissibility_heave[valid_b], 1e-4))
                else:
                    b_mag_h = b_res.transmissibility_heave[valid_b]
                arch_tag = "4-Post" if b_res.rig_architecture == '4_post' else "7-Post"
                ax1.plot(fb_v, b_mag_h, color='#8b949e', lw=1.6, linestyle='-.', label=f'Baseline {arch_tag} Heave')

        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=7, ncol=2)

        # 2. Sfasamento (Phase Lag)
        ax2.plot(f_v, res.phase_heave[valid], color='#58a6ff', lw=1.5, label='Fase Heave')
        ax2.plot(f_v, res.phase_pitch[valid], color='#f0883e', lw=1.5, label='Fase Pitch')
        ax2.plot(f_v, res.phase_roll[valid], color='#bc8cff', lw=1.5, label='Fase Roll')
        ax2.set_ylabel("Fase [deg]", fontsize=8)
        ax2.set_ylim(-180, 180)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=7)

        # 3. Funzione di Coerenza (Coherence gamma^2)
        ax3.plot(f_v, res.coherence_heave[valid], color='#58a6ff', lw=1.5, label='Coerenza Heave (γ²)')
        ax3.plot(f_v, res.coherence_pitch[valid], color='#f0883e', lw=1.5, label='Coerenza Pitch (γ²)')
        ax3.plot(f_v, res.coherence_roll[valid], color='#bc8cff', lw=1.5, label='Coerenza Roll (γ²)')
        if self.chk_show_thresholds.isChecked():
            ax3.axhline(0.80, color='#f85149', linestyle='--', lw=1.2, label='Soglia Lineare γ²=0.8')
        ax3.set_xlabel("Frequenza di Eccitazione [Hz]", fontsize=8)
        ax3.set_ylabel("Coerenza γ² [-]", fontsize=8)
        ax3.set_ylim(0.0, 1.05)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='lower right', fontsize=7)

        if use_log_x:
            xmin = max(0.3, float(f_v[0])) if len(f_v) > 0 else 0.3
            xmax = max(xmin + 0.1, min(35.0, float(f_v[-1]))) if len(f_v) > 0 else 35.0
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax3.set_xscale('log')
            ax1.set_xlim(left=xmin, right=xmax)
        else:
            xmax = max(1.0, min(35.0, float(f_v[-1]))) if len(f_v) > 0 else 35.0
            ax1.set_xlim(left=0.0, right=xmax)

        apply_plot_theme(self.fig_bode)
        self.fig_bode.tight_layout()
        self.canvas_bode.draw()

    def _render_cplv_plots(self, res: SevenPostResult):
        self.fig_cplv.clear()
        ax1 = self.fig_cplv.add_subplot(121)
        ax2 = self.fig_cplv.add_subplot(122)

        # 1. Bar Chart Multibanda CPLV (Overall vs Low-Band vs High-Band)
        corners = ['FL (Ant Sx)', 'FR (Ant Dx)', 'RL (Post Sx)', 'RR (Post Dx)', 'MEDIA']
        keys = ['fl', 'fr', 'rl', 'rr', 'avg']

        cplv_tot = [res.cplv[k] * 100.0 for k in keys]
        cplv_low = [res.cplv_low_band[k] * 100.0 for k in keys]
        cplv_high = [res.cplv_high_band[k] * 100.0 for k in keys]

        x = np.arange(len(corners))
        w = 0.25

        ax1.bar(x - w, cplv_tot, w, label='CPLV Totale', color='#58a6ff', alpha=0.9)
        ax1.bar(x, cplv_low, w, label='Low-Band (0.5-4.5Hz Cassa)', color='#e3b341', alpha=0.9)
        ax1.bar(x + w, cplv_high, w, label='High-Band (8-18Hz Hop)', color='#ff7b72', alpha=0.9)

        if self.chk_show_thresholds.isChecked():
            ax1.axhline(15.0, color='#3fb950', linestyle='--', lw=1.2, label='Target Ottimo (<15%)')

        ax1.set_xticks(x)
        ax1.set_xticklabels(corners, fontsize=8, rotation=15)
        ax1.set_ylabel("CPLV [%]", fontsize=8)
        ax1.set_title("Decomposizione Contact Patch Load Variation (CPLV)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax1.legend(loc='upper right', fontsize=7)

        # 2. Spettro Continuo CPLV o Grafico Perdita di Grip Stimata
        f = res.freqs
        valid = (f >= 0.3) & (f <= 35.0)

        if self.cb_cplv_view.currentIndex() == 1:
            # Spettro continuo
            ax2.plot(f[valid], res.cplv_spectrum[0, valid], color='#58a6ff', lw=1.5, label='FL')
            ax2.plot(f[valid], res.cplv_spectrum[1, valid], color='#39c5cf', lw=1.5, label='FR')
            ax2.plot(f[valid], res.cplv_spectrum[2, valid], color='#f0883e', lw=1.5, label='RL')
            ax2.plot(f[valid], res.cplv_spectrum[3, valid], color='#ff7b72', lw=1.5, label='RR')
            ax2.axvspan(0.5, 4.5, color='#e3b341', alpha=0.12, label='Low-Band (Body)')
            ax2.axvspan(8.0, 18.0, color='#ff7b72', alpha=0.12, label='High-Band (Hop)')
            ax2.set_xlabel("Frequenza [Hz]", fontsize=8)
            ax2.set_ylabel("CPLV Spectrum [N/m]", fontsize=8)
            ax2.set_title("Spettro Continuo CPLV vs Frequenza", fontsize=9, fontweight='bold', color='#f0f6fc')
        else:
            # Perdita di Aderenza % e Lift-off
            loss_pcts = [res.estimated_grip_loss_pct[k] for k in keys]
            bars = ax2.bar(x, loss_pcts, width=0.45, color='#f85149', alpha=0.85, label='Perdita Grip %')
            for b in bars:
                h = b.get_height()
                ax2.annotate(f"-{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                             fontsize=8, fontweight='bold', color='#f85149')
            ax2.set_xticks(x)
            ax2.set_xticklabels(corners, fontsize=8, rotation=15)
            ax2.set_ylabel("Perdita di Aderenza Stimata [%]", fontsize=8)
            ax2.set_title("Stima Perdita Grip Laterale (Degressività Fz)", fontsize=9, fontweight='bold', color='#f0f6fc')

        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=7)

        apply_plot_theme(self.fig_cplv)
        self.fig_cplv.tight_layout()
        self.canvas_cplv.draw()

    def _render_telemetry_plots(self, res: SevenPostResult):
        self.fig_telem.clear()
        ax1 = self.fig_telem.add_subplot(311)
        ax2 = self.fig_telem.add_subplot(312, sharex=ax1)
        ax3 = self.fig_telem.add_subplot(313, sharex=ax1)

        t = res.time

        # 1. Spostamento 4 attuatori stradali
        ax1.plot(t, res.z_posts[:, 0] * 1e3, color='#58a6ff', lw=1.2, label='Post FL')
        ax1.plot(t, res.z_posts[:, 1] * 1e3, color='#39c5cf', lw=1.2, label='Post FR')
        ax1.plot(t, res.z_posts[:, 2] * 1e3, color='#f0883e', lw=1.2, label='Post RL')
        ax1.plot(t, res.z_posts[:, 3] * 1e3, color='#ff7b72', lw=1.2, label='Post RR')
        ax1.set_ylabel("Corsa Post [mm]", fontsize=8)
        ax1.set_title("Attuatori Stradali Idraulici (Post 1-4)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=7, ncol=4)

        # 2. Carichi verticali normali pneumatici Fz
        ax2.plot(t, res.Fz_tires[:, 0], color='#58a6ff', lw=1.2, label='Fz FL')
        ax2.plot(t, res.Fz_tires[:, 1], color='#39c5cf', lw=1.2, label='Fz FR')
        ax2.plot(t, res.Fz_tires[:, 2], color='#f0883e', lw=1.2, label='Fz RL')
        ax2.plot(t, res.Fz_tires[:, 3], color='#ff7b72', lw=1.2, label='Fz RR')
        if self.chk_show_thresholds.isChecked():
            ax2.axhline(0.0, color='#f85149', linestyle='--', lw=1.2, label='Soglia Lift-off (0N)')
        ax2.set_ylabel("Carico Fz [N]", fontsize=8)
        ax2.set_title("Carichi Verticali Contact Patch (Dinamica Impronta a Terra)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=7, ncol=5)

        # 3. Attuatori Telaio (Aero Downforce)
        is_4post = getattr(res, 'rig_architecture', '7_post') == '4_post'
        if not is_4post:
            ax3.plot(t, res.F_aero_posts[:, 0], color='#bc8cff', lw=1.5, label='Post 5 (Aero Ant)')
            ax3.plot(t, res.F_aero_posts[:, 1], color='#d29922', lw=1.2, label='Post 6 (Aero Post Sx)')
            ax3.plot(t, res.F_aero_posts[:, 2], color='#e3b341', lw=1.2, label='Post 7 (Aero Post Dx)')
            ax3.set_ylabel("Forza Aero [N]", fontsize=8)
            ax3.set_title("Attuatori Aerodinamici Superiore Telaio (Post 5-7)", fontsize=9, fontweight='bold', color='#f0f6fc')
            ax3.legend(loc='upper right', fontsize=7, ncol=3)
        else:
            ax3.plot(t, np.zeros_like(t), color='#8b949e', lw=1.2, linestyle=':')
            ax3.set_ylabel("Forza Aero [N]", fontsize=8)
            ax3.set_title("Attuatori Telaio Post 5-7 (DISCONNESSI in modalità 4-Post)", fontsize=9, fontweight='bold', color='#8b949e')
            ax3.text(0.5, 0.5, "MODALITA' 4-POST CLASSICO: NESSUN CARICO APPLICATO AL TELAIO (0 N)",
                     transform=ax3.transAxes, ha='center', va='center', color='#58a6ff', fontsize=8, fontweight='bold')
        ax3.set_xlabel("Tempo [s]", fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.6)

        apply_plot_theme(self.fig_telem)
        self.fig_telem.tight_layout()
        self.canvas_telem.draw()

    def _render_ttx_diagnostics(self, res: SevenPostResult):
        self.fig_ttx.clear()
        ax1 = self.fig_ttx.add_subplot(121)
        ax2 = self.fig_ttx.add_subplot(122)

        # 1. Cicli Lissajous Forza-Velocità
        v_damper_fl = res.v_damper[:, 0] * 1e3
        F_damper_fl = res.F_damper[:, 0]
        v_damper_rl = res.v_damper[:, 2] * 1e3
        F_damper_rl = res.F_damper[:, 2]

        ax1.plot(v_damper_fl, F_damper_fl, color='#58a6ff', lw=1.2, alpha=0.7, label='TTX25 Ant (FL)')
        ax1.plot(v_damper_rl, F_damper_rl, color='#f0883e', lw=1.2, alpha=0.7, label='TTX25 Post (RL)')
        ax1.axhline(0.0, color='#8b949e', linestyle=':', lw=0.8)
        ax1.axvline(0.0, color='#8b949e', linestyle=':', lw=0.8)
        ax1.set_xlabel("Velocità Stelo [mm/s]", fontsize=8)
        ax1.set_ylabel("Forza Smorzamento [N]", fontsize=8)
        ax1.set_title("Cicli di Isteresi Forza-Velocità (Lissajous)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', fontsize=8)

        # 2. Istogramma Bilstrom a 5 Regimi di Velocità
        b_fl = res.bilstrom_5bin['fl']
        b_rl = res.bilstrom_5bin['rl']
        labels = ['<10mm/s\n(Ultra-Low)', '10-25\n(Low)', '25-50\n(Mid)', '50-100\n(Mid-High)', '>100mm/s\n(High)']
        vals_f = [b_fl['ultra_low'], b_fl['low'], b_fl['mid'], b_fl['mid_high'], b_fl['high']]
        vals_r = [b_rl['ultra_low'], b_rl['low'], b_rl['mid'], b_rl['mid_high'], b_rl['high']]

        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width/2, vals_f, width, label='Ant. (FL)', color='#58a6ff')
        ax2.bar(x + width/2, vals_r, width, label='Post. (RL)', color='#f0883e')

        # Annotazione Box Energetico
        info_str = (
            f"Energia Dissipata:\n FL: {res.damper_energy_dissipated_j['fl']:.0f} J | RL: {res.damper_energy_dissipated_j['rl']:.0f} J\n"
            f"Potenza RMS:\n FL: {res.damper_power_rms_w['fl']:.1f} W | RL: {res.damper_power_rms_w['rl']:.1f} W\n"
            f"Asimmetria Comp/Reb:\n FL: {res.damper_asymmetry_ratio['fl']:.2f} | RL: {res.damper_asymmetry_ratio['rl']:.2f}"
        )
        ax2.text(0.97, 0.95, info_str, transform=ax2.transAxes, fontsize=7,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', alpha=0.85))

        ax2.set_ylabel("Percentuale di Tempo [%]", fontsize=8)
        ax2.set_title("Istogramma Bilstrom a 5 Regimi di Velocità", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax2.legend(loc='upper left', fontsize=7)

        apply_plot_theme(self.fig_ttx)
        self.fig_ttx.tight_layout()
        self.canvas_ttx.draw()

    def _render_aero_platform_plots(self, res: SevenPostResult):
        self.fig_aero.clear()
        ax1 = self.fig_aero.add_subplot(221)
        ax2 = self.fig_aero.add_subplot(222)
        ax3 = self.fig_aero.add_subplot(212)

        t = res.time

        # 1. Ride Height Fluctuation
        # z_c_f = zs - theta * a_s, z_c_r = zs + theta * b_s
        zs = res.z_chassis[:, 0]
        theta = res.z_chassis[:, 1]
        phi = res.z_chassis[:, 2]

        rh_f = -(zs - theta * 0.85) * 1e3
        rh_r = -(zs + theta * 0.75) * 1e3

        ax1.plot(t, rh_f, color='#58a6ff', lw=1.2, label='Δ Ride Height Ant [mm]')
        ax1.plot(t, rh_r, color='#f0883e', lw=1.2, label='Δ Ride Height Post [mm]')
        ax1.set_ylabel("Δ Quota [mm]", fontsize=8)
        ax1.set_title("Fluttuazione Ride Height Sotto Carico", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=7)

        # 2. Pitch and Roll Angles
        ax2.plot(t, np.degrees(theta), color='#f0883e', lw=1.2, label='Beccheggio θ [deg]')
        ax2.plot(t, np.degrees(phi), color='#bc8cff', lw=1.2, label='Rollio φ [deg]')
        ax2.set_ylabel("Angolo [deg]", fontsize=8)
        ax2.set_title("Assetto Angolare Telaio (Pitch & Roll)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=7)

        # 3. PSD Accelerazioni Telaio
        f = res.freqs
        valid = (f >= 0.3) & (f <= 35.0)
        ax3.semilogy(f[valid], res.psd_heave[valid], color='#58a6ff', lw=1.5, label='PSD Heave [(m/s²)²/Hz]')
        ax3.semilogy(f[valid], res.psd_pitch[valid], color='#f0883e', lw=1.5, label='PSD Pitch [(rad/s²)²/Hz]')
        ax3.semilogy(f[valid], res.psd_roll[valid], color='#bc8cff', lw=1.5, label='PSD Roll [(rad/s²)²/Hz]')
        ax3.set_xlabel("Frequenza [Hz]", fontsize=8)
        ax3.set_ylabel("Densità Spettrale (PSD)", fontsize=8)
        ax3.set_title("Spettro di Densità di Potenza Accelerazioni Telaio", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper right', fontsize=8)

        apply_plot_theme(self.fig_aero)
        self.fig_aero.tight_layout()
        self.canvas_aero.draw()

    def _render_modal_plots(self, modal: Dict[str, Any]):
        self.fig_modal.clear()
        ax = self.fig_modal.add_subplot(111)

        eigvals = modal['eigenvalues']
        real_parts = eigvals.real
        imag_parts = eigvals.imag

        ax.scatter(real_parts, imag_parts, color='#58a6ff', s=50, edgecolors='white', zorder=5, label='Poli Sistema')
        ax.axvline(0.0, color='#f85149', linestyle='--', lw=1.2, alpha=0.7)
        ax.axhline(0.0, color='#8b949e', linestyle=':', lw=0.8)

        # Traccia linee di smorzamento costante zeta = 0.2, 0.4
        for zeta in [0.2, 0.4, 0.7]:
            theta = math.acos(zeta)
            r = np.linspace(0, max(np.abs(imag_parts)) * 1.2, 50)
            ax.plot(-r * math.cos(theta), r * math.sin(theta), color='#8b949e', linestyle=':', lw=0.8)
            ax.plot(-r * math.cos(theta), -r * math.sin(theta), color='#8b949e', linestyle=':', lw=0.8)

        ax.set_xlabel("Parte Reale σ [rad/s]", fontsize=8)
        ax.set_ylabel("Parte Immaginaria jω [rad/s]", fontsize=8)
        ax.set_title("Mappa dei Poli nel Piano Complesso s (Stabilità & Smorzamento)", fontsize=9, fontweight='bold', color='#f0f6fc')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=8)

        apply_plot_theme(self.fig_modal)
        self.fig_modal.tight_layout()
        self.canvas_modal.draw()

    def _update_modal_table_from_comparison(self, comp: Dict[str, Any]):
        corr_table = comp.get('correlation_table', [])
        if not corr_table:
            return
        self.table_modes.setColumnCount(6)
        self.table_modes.setHorizontalHeaderLabels([
            "Modo Dinamico", "Freq Modello [Hz]", "Freq Banco [Hz]", "Delta %", "Smorzamento ζ [-]", "Correlazione Modello/Banco"
        ])
        self.table_modes.setRowCount(len(corr_table))
        for row, r in enumerate(corr_table):
            self.table_modes.setItem(row, 0, QTableWidgetItem(str(r['mode'])))
            self.table_modes.setItem(row, 1, QTableWidgetItem(f"{r['model_hz']:.2f} Hz"))
            f_rig_str = f"{r['rig_hz']:.2f} Hz" if r['rig_hz'] is not None else "—"
            self.table_modes.setItem(row, 2, QTableWidgetItem(f_rig_str))
            delta_str = f"{r['delta_pct']:+.1f} %" if r['delta_pct'] is not None else "—"
            it_d = QTableWidgetItem(delta_str)
            self.table_modes.setItem(row, 3, it_d)
            self.table_modes.setItem(row, 4, QTableWidgetItem(f"{r['damping_ratio']:.3f}"))
            it_st = QTableWidgetItem(str(r['status']))
            status_text = str(r['status'])
            if "< 5%" in status_text:
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#3fb950")))
            elif "< 10%" in status_text:
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#39c5cf")))
            elif "< 20%" in status_text:
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#e3b341")))
            elif "Non eccitato" in status_text:
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#8b949e")))
            else:
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#f85149")))
            self.table_modes.setItem(row, 5, it_st)

    def _on_modal_clicked(self):
        vp = self._get_active_vp()
        self._apply_ui_damper_settings(vp)

        rig = SevenPostRig(vp)
        peaks = self.last_result.resonance_peaks if self.last_result else None
        comp = rig.compute_model_benchmark_comparison(peaks)
        self._update_modal_table_from_comparison(comp)

        modal = rig.compute_linear_modal_analysis()
        self._render_modal_plots(modal)
        self.tabs_view.setCurrentIndex(5)
        is_4post = getattr(self, 'cb_rig_arch', None) and self.cb_rig_arch.currentIndex() == 1
        arch_lbl = "4-Post" if is_4post else "7-Post"
        if hasattr(self.mw, 'statusBar'):
            self.mw.statusBar().showMessage(f"[{arch_lbl} Rig] Analisi modale e benchmark col modello teorico aggiornati.", 3000)

    def _on_export_clicked(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Esportazione Telemetria", "Nessuna simulazione disponibile da esportare.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Esporta Telemetria 7-Post Rig", "telemetria_7post_rig.csv", "File CSV (*.csv)")
        if not path:
            return

        res = self.last_result
        try:
            data = np.column_stack([
                res.time,
                res.z_posts[:, 0], res.z_posts[:, 1], res.z_posts[:, 2], res.z_posts[:, 3],
                res.Fz_tires[:, 0], res.Fz_tires[:, 1], res.Fz_tires[:, 2], res.Fz_tires[:, 3],
                res.z_chassis[:, 0], res.z_chassis[:, 1], res.z_chassis[:, 2],
                res.F_aero_posts[:, 0], res.F_aero_posts[:, 1], res.F_aero_posts[:, 2]
            ])
            header = ("time,z_post_fl,z_post_fr,z_post_rl,z_post_rr,"
                      "Fz_fl,Fz_fr,Fz_rl,Fz_rr,"
                      "z_heave,pitch_rad,roll_rad,"
                      "Faero_front,Faero_rl,Faero_rr")
            np.savetxt(path, data, delimiter=',', header=header, comments='')
            QMessageBox.information(self, "Esportazione Telemetria", f"Telemetria esportata con successo in:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file:\n{str(e)}")

    def _on_export_report_clicked(self):
        if not self.txt_report.toPlainText().strip():
            QMessageBox.warning(self, "Esportazione Report", "Nessun report generato disponibile da salvare.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Salva Report Tecnico", "report_tecnico_7post_rig.txt", "File Testo (*.txt);;File Markdown (*.md)")
        if not path:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.txt_report.toPlainText())
            QMessageBox.information(self, "Salvataggio Report", f"Report salvato con successo in:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il report:\n{str(e)}")

    def _on_copy_report_clicked(self):
        text = self.txt_report.toPlainText()
        if text.strip():
            QApplication.clipboard().setText(text)
            if hasattr(self.mw, 'statusBar'):
                self.mw.statusBar().showMessage("[7-Post Rig] Report tecnico copiato negli appunti.", 3000)

    def _toggle_anim_play(self):
        if self.anim_timer.isActive():
            self.anim_timer.stop()
            self.btn_anim_play.setText("Play")
        else:
            if self.last_result is None:
                return
            interval_ms = int(1000.0 / self.anim_fps)
            self.anim_timer.start(interval_ms)
            self.btn_anim_play.setText("Pause")

    def _on_anim_tick(self):
        if self.last_result is None:
            self.anim_timer.stop()
            return

        dt_sim = self.last_result.time[1] - self.last_result.time[0] if len(self.last_result.time) > 1 else 0.002
        step_advance = max(1, int(round((1.0 / self.anim_fps) * self.anim_speed_multiplier / dt_sim)))

        curr = self.slider_anim.value() + step_advance
        if curr >= len(self.last_result.time):
            curr = 0

        self.slider_anim.setValue(curr)

    def _on_slider_scrub(self, val: int):
        if self.last_result is None:
            return
        t = self.last_result.time[val]
        self.lbl_anim_time.setText(f"{t:.2f} / {self.last_result.time[-1]:.2f} s")
        self.canvas_rig2d.draw_frame(val)

    def _on_anim_speed_changed(self, idx: int):
        speeds = [0.25, 0.50, 1.00]
        self.anim_speed_multiplier = speeds[idx]
