"""
ui_direct_collocation.py
------------------------
Modulo dedicato per l'ottimizzazione di traiettoria (Direct Collocation OCP)
e analisi avanzata dei dati di telemetria.

Caratteristiche:
- CollocationTrajectoryCanvas: Canvas Matplotlib disaccoppiato dal simulatore per la
  panoramica ingegneristica standard (Tracciato 2D & Racing Line, Velocita vs Profiler,
  Accelerazioni ax/ay, Scostamento laterale n(s) & Angolo sterzo).
- DataInspectorCanvas: Strumento avanzato Telemetry Data Inspector per la selezione libera
  e personalizzata dei canali di telemetria, configurazione multi-plot (1x1, 2x1, 3x1, 2x2),
  cursori interattivi sincronizzati e switch asse X (Progressione s [m] o Tempo t [s]).
- TrackConfigPanel: Sezione dedicata alla selezione tracciato da preset ufficiali FSAE
  (Autocross, Skidpad, Endurance, Accelerazione) o file personalizzati (.csv, .json),
  con anteprima geometrica immediata e configurazione parametri (w, ds, schema di collocazione).
- DirectCollocationDialog: Finestra di dialogo completa che eredita rigorosamente il tema
  e i font dell'applicazione (Motorsport Scuro, Ingegneristico Chiaro, Alto Contrasto)
  senza emoji, con log IPOPT asincrono non bloccante.
"""

import os
import time
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QProgressBar, QFileDialog, QMessageBox, QSplitter, QGroupBox,
    QTabWidget, QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup, QScrollArea
)
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.mdo.mdo_direct_collocation import (
    build_track_geometry, load_custom_track_file,
    compute_backward_forward_velocity_profile,
    DirectCollocationCostWeights
)
from src.ui.gui_workers import CollocationWorker
from src.ui.ui_theme_engine import (
    get_stylesheet, get_theme_palette, apply_app_palette, apply_plot_theme
)


# =============================================================================
# 1. CANVAS DEDICATO DISACCOPPIATO - PANORAMICA INGEGNERISTICA 4 SUBPLOT
# =============================================================================
class CollocationTrajectoryCanvas(FigureCanvas):
    """
    Canvas Matplotlib isolato e autonomo, deputato al rendering della panoramica
    standard a 4 subplot di ottimizzazione traiettoria.
    """
    def __init__(self, parent=None, width=10, height=7, dpi=100, theme_name="dark"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.theme_name = theme_name
        self.axes: List[Any] = []
        self._init_layout()

    def _init_layout(self):
        """Inizializza la griglia 2x2 dei subplot."""
        self.fig.clear()
        self.ax_track = self.fig.add_subplot(221)
        self.ax_speed = self.fig.add_subplot(222)
        self.ax_accel = self.fig.add_subplot(223)
        self.ax_lateral = self.fig.add_subplot(224)
        self.ax_steer = self.ax_lateral.twinx()
        self.axes = [self.ax_track, self.ax_speed, self.ax_accel, self.ax_lateral, self.ax_steer]
        self.apply_theme(self.theme_name)
        self.plot_empty_state()

    def apply_theme(self, theme_name: str):
        """Applica la palette tematica dell'applicazione."""
        self.theme_name = theme_name
        t = get_theme_palette(theme_name)
        apply_plot_theme(self.fig, self.axes, theme_name=theme_name)
        self.fig.patch.set_facecolor(t['plot_fig_bg'])
        self.draw_idle()

    def plot_empty_state(self):
        """Visualizza un messaggio di stato vuoto / pronto al calcolo."""
        t = get_theme_palette(self.theme_name)
        for ax in [self.ax_track, self.ax_speed, self.ax_accel, self.ax_lateral]:
            ax.clear()
            ax.set_facecolor(t['plot_ax_bg'])
            ax.grid(True, linestyle=':', alpha=0.5, color=t['plot_grid'])

        self.ax_steer.clear()
        self.ax_track.text(0.5, 0.5, "Tracciato non caricato.\nSeleziona un circuito e premi 'Anteprima Tracciato'.",
                           ha='center', va='center', transform=self.ax_track.transAxes,
                           color=t['text_secondary'], fontsize=9)
        self.ax_speed.text(0.5, 0.5, "In attesa di ottimizzazione traiettoria...",
                           ha='center', va='center', transform=self.ax_speed.transAxes,
                           color=t['text_secondary'], fontsize=9)
        self.ax_accel.text(0.5, 0.5, "Accelerazioni non disponibili.",
                           ha='center', va='center', transform=self.ax_accel.transAxes,
                           color=t['text_secondary'], fontsize=9)
        self.ax_lateral.text(0.5, 0.5, "Scostamento e sterzo non disponibili.",
                             ha='center', va='center', transform=self.ax_lateral.transAxes,
                             color=t['text_secondary'], fontsize=9)
        self.fig.tight_layout()
        self.draw_idle()

    def plot_track_preview(self, track: Dict[str, Any]):
        """Disegna l'anteprima geometrica del tracciato prima dell'ottimizzazione."""
        t = get_theme_palette(self.theme_name)
        self.ax_track.clear()
        self.ax_track.set_facecolor(t['plot_ax_bg'])
        self.ax_track.grid(True, linestyle=':', alpha=0.5, color=t['plot_grid'])

        self.ax_track.plot(track['x_left'], track['y_left'], color='#f85149', linestyle='-', linewidth=1.4, alpha=0.85, label='Limite SX')
        self.ax_track.plot(track['x_right'], track['y_right'], color='#f85149', linestyle='--', linewidth=1.4, alpha=0.85, label='Limite DX')
        self.ax_track.plot(track['x_center'], track['y_center'], color='#8b949e', linestyle=':', linewidth=1.1, label='Mezzeria')

        # Nodi di collocazione
        self.ax_track.scatter(track['x_center'], track['y_center'], s=12, color='#58a6ff', alpha=0.6, zorder=3, label=f'Nodi Collocazione (N={track["N"]})')
        self.ax_track.plot(track['x_center'][0], track['y_center'][0], 'go', markersize=8, label='Partenza', zorder=5)

        self.ax_track.set_xlabel("X [m]", fontsize=8, color=t['plot_labels'])
        self.ax_track.set_ylabel("Y [m]", fontsize=8, color=t['plot_labels'])
        self.ax_track.set_title(f"Anteprima Tracciato: {track.get('track_type', 'Circuit')} (L={track['s'][-1]:.1f} m, N={track['N']})",
                                fontsize=9, fontweight='bold', color=t['plot_title'])
        self.ax_track.set_aspect('equal', 'datalim')

        leg = self.ax_track.legend(fontsize=7, loc='best')
        if leg:
            leg.get_frame().set_facecolor(t['plot_ax_bg'])
            leg.get_frame().set_edgecolor(t['plot_spines'])
            for txt in leg.get_texts():
                txt.set_color(t['text_primary'])

        apply_plot_theme(self.fig, [self.ax_track], theme_name=self.theme_name)
        self.fig.tight_layout()
        self.draw_idle()

    def plot_optimization_results(self, res: Dict[str, Any]):
        """Renderizza la panoramica completa a 4 subplot con i risultati ottimi."""
        t = get_theme_palette(self.theme_name)
        for ax in [self.ax_track, self.ax_speed, self.ax_accel, self.ax_lateral]:
            ax.clear()
            ax.set_facecolor(t['plot_ax_bg'])
            ax.grid(True, linestyle=':', alpha=0.5, color=t['plot_grid'])
        self.ax_steer.clear()

        df = res['telemetry']
        prof = res['profiler']
        track = res['track']
        t_colloc = res['lap_time_s']
        t_prof = prof['lap_time_s']
        d_time = res['delta_time_s']
        mod_str = res.get('model_type', '14-DOF')

        # --- SUBPLOT 1: 2D RACING LINE & VELOCITA ---
        self.ax_track.plot(track['x_left'], track['y_left'], color='#f85149', linestyle='-', linewidth=1.2, alpha=0.8, label='Limite SX')
        self.ax_track.plot(track['x_right'], track['y_right'], color='#f85149', linestyle='--', linewidth=1.2, alpha=0.8, label='Limite DX')
        self.ax_track.plot(track['x_center'], track['y_center'], color='#8b949e', linestyle=':', linewidth=1.0, alpha=0.7, label='Mezzeria')

        sc = self.ax_track.scatter(df['x_opt'], df['y_opt'], c=df['v_kmh'], cmap='turbo', s=16, zorder=4, label='Racing Line Ottima')
        cbar = self.fig.colorbar(sc, ax=self.ax_track, fraction=0.046, pad=0.04)
        cbar.set_label("Velocita [km/h]", fontsize=7, color=t['plot_labels'])
        cbar.ax.tick_params(colors=t['plot_ticks'], labelsize=7)
        cbar.outline.set_edgecolor(t['plot_spines'])

        self.ax_track.plot(df['x_opt'].iloc[0], df['y_opt'].iloc[0], 'go', markersize=7, label='Start / Finish', zorder=5)
        self.ax_track.set_xlabel("X Globale [m]", fontsize=8, color=t['plot_labels'])
        self.ax_track.set_ylabel("Y Globale [m]", fontsize=8, color=t['plot_labels'])
        self.ax_track.set_title(f"1. Traiettoria Ottimale ({mod_str})", fontsize=9, fontweight='bold', color=t['plot_title'])
        self.ax_track.set_aspect('equal', 'datalim')

        # --- SUBPLOT 2: SPEED PROFILE ---
        self.ax_speed.plot(df['s_m'], df['v_kmh'], color='#58a6ff', linewidth=2.0, label=f'Direct Collocation ({t_colloc:.2f} s)')
        self.ax_speed.plot(df['s_m'], df['v_profiler_kmh'], color='#f0883e', linestyle='--', linewidth=1.6, label=f'Quasi-Static Profiler ({t_prof:.2f} s)')
        self.ax_speed.fill_between(df['s_m'], df['v_kmh'], df['v_profiler_kmh'],
                                   where=(df['v_kmh'] >= df['v_profiler_kmh']),
                                   color='#3fb950', alpha=0.25, label='Guadagno Velocita')
        self.ax_speed.set_xlabel("Progressione s [m]", fontsize=8, color=t['plot_labels'])
        self.ax_speed.set_ylabel("Velocita [km/h]", fontsize=8, color=t['plot_labels'])
        self.ax_speed.set_title(f"2. Profilo Velocita: Collocation vs Profiler (Delta: {d_time:+.2f} s)", fontsize=9, fontweight='bold', color=t['plot_title'])

        # --- SUBPLOT 3: ACCELERAZIONI AX E AY ---
        self.ax_accel.plot(df['s_m'], df['ay_g'], color='#bc8cff', linewidth=1.5, label='Collocation a_y [g]')
        self.ax_accel.plot(df['s_m'], df['ay_profiler_g'], color='#bc8cff', linestyle=':', linewidth=1.2, alpha=0.6, label='Profiler a_y [g]')
        self.ax_accel.plot(df['s_m'], df['ax_g'], color='#3fb950', linewidth=1.5, label='Collocation a_x [g]')
        self.ax_accel.plot(df['s_m'], df['ax_profiler_g'], color='#3fb950', linestyle=':', linewidth=1.2, alpha=0.6, label='Profiler a_x [g]')
        self.ax_accel.set_xlabel("Progressione s [m]", fontsize=8, color=t['plot_labels'])
        self.ax_accel.set_ylabel("Accelerazione [g]", fontsize=8, color=t['plot_labels'])
        self.ax_accel.set_title("3. Confronto Accelerazioni a_x (Long.) e a_y (Lat.)", fontsize=9, fontweight='bold', color=t['plot_title'])

        # --- SUBPLOT 4: USO CARREGGIATA & ANGOLO STERZO ---
        self.ax_lateral.plot(df['s_m'], df['n_m'], color='#39c5cf', linewidth=1.8, label='Scostamento n [m]')
        w_h = track['track_width'] / 2.0
        self.ax_lateral.axhline(+w_h, color='#f85149', linestyle='--', linewidth=1.0, alpha=0.7, label='Limite SX (+w/2)')
        self.ax_lateral.axhline(-w_h, color='#f85149', linestyle='--', linewidth=1.0, alpha=0.7, label='Limite DX (-w/2)')
        self.ax_lateral.set_xlabel("Progressione s [m]", fontsize=8, color=t['plot_labels'])
        self.ax_lateral.set_ylabel("Distanza da Mezzeria n [m]", color='#39c5cf', fontsize=8)

        self.ax_steer.plot(df['s_m'], df['steer_deg'], color='#e3b341', linestyle=':', linewidth=1.4, label='Angolo Sterzo delta [deg]')
        self.ax_steer.set_ylabel("Angolo Sterzo delta [deg]", color='#e3b341', fontsize=8)
        self.ax_lateral.set_title("4. Uso della Carreggiata & Angolo Sterzo", fontsize=9, fontweight='bold', color=t['plot_title'])

        apply_plot_theme(self.fig, self.axes, theme_name=self.theme_name)
        for ax in [self.ax_track, self.ax_speed, self.ax_accel, self.ax_lateral]:
            leg = ax.legend(fontsize=7, loc='best')
            if leg:
                leg.get_frame().set_facecolor(t['plot_ax_bg'])
                leg.get_frame().set_edgecolor(t['plot_spines'])
                for txt in leg.get_texts():
                    txt.set_color(t['text_primary'])

        self.fig.tight_layout()
        self.draw_idle()


# =============================================================================
# 2. DATA INSPECTOR CANVAS - TELEMETRY DATA INSPECTOR
# =============================================================================
class DataInspectorCanvas(FigureCanvas):
    """
    Canvas configurabile Telemetry Data Inspector.
    Consente la selezione libera di canali di telemetria, configurazione
    del numero di grafici (1x1, 2x1, 3x1, 2x2) e cursori sincronizzati.
    """
    def __init__(self, parent=None, width=10, height=7, dpi=100, theme_name="dark"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.theme_name = theme_name
        self.layout_mode = "2x1"  # "1x1", "2x1", "3x1", "2x2"
        self.x_mode = "s"         # "s" (progressione [m]) o "t" (tempo [s])
        self.subplots: List[Any] = []
        self.df_telemetry: Optional[pd.DataFrame] = None
        self.active_channels: Dict[int, List[str]] = {0: ['v_kmh', 'v_profiler_kmh'], 1: ['ay_g', 'ax_g']}
        self._cursor_lines: List[Any] = []
        
        # Connetti eventi interattivi per crosshair cursore
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self._rebuild_layout()

    def set_data(self, df: pd.DataFrame):
        """Imposta il dataframe di telemetria e aggiorna i grafici."""
        self.df_telemetry = df.copy()
        self._compute_derived_channels()
        self.update_plots()

    def _compute_derived_channels(self):
        """Calcola canali derivati ingegneristici utili per l'ispezione."""
        if self.df_telemetry is None:
            return
        df = self.df_telemetry
        if 'ax_g' in df.columns and 'ay_g' in df.columns:
            df['atot_g'] = np.sqrt(df['ax_g']**2 + df['ay_g']**2)
        if 'v_kmh' in df.columns and 'v_profiler_kmh' in df.columns:
            df['delta_v_kmh'] = df['v_kmh'] - df['v_profiler_kmh']
        if 't_s' in df.columns and 's_m' in df.columns:
            # Calcolo yaw rate stimato deg/s
            if 'v_mps' in df.columns and 'ay_g' in df.columns:
                g = 9.81
                # ay = v * r  => r = ay / v [rad/s]
                v_safe = np.maximum(df['v_mps'].values, 2.0)
                r_rad = (df['ay_g'].values * g) / v_safe
                df['yaw_rate_degs'] = np.rad2deg(r_rad)
            # Potenza longitudinale stimata [kW] per m=300 kg
            if 'v_mps' in df.columns and 'ax_g' in df.columns:
                m_nom = 300.0
                df['power_long_kw'] = (m_nom * df['ax_g'].values * 9.81 * df['v_mps'].values) / 1000.0

    def set_layout_mode(self, mode: str):
        """Imposta la griglia dei subplot ('1x1', '2x1', '3x1', '2x2')."""
        self.layout_mode = mode
        self._rebuild_layout()
        self.update_plots()

    def set_x_mode(self, x_mode: str):
        """Imposta l'asse X ('s' progressione o 't' tempo)."""
        self.x_mode = x_mode
        self.update_plots()

    def set_plot_channels(self, plot_idx: int, channels: List[str]):
        """Assegna i canali selezionati a un dato subplot."""
        self.active_channels[plot_idx] = list(channels)
        self.update_plots()

    def _rebuild_layout(self):
        """Ricostruisce i subplot Matplotlib e collega gli assi X."""
        self.fig.clear()
        self.subplots.clear()
        self._cursor_lines.clear()

        share_ax = None
        if self.layout_mode == "1x1":
            ax = self.fig.add_subplot(111)
            self.subplots.append(ax)
        elif self.layout_mode == "2x1":
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212, sharex=ax1)
            self.subplots.extend([ax1, ax2])
        elif self.layout_mode == "3x1":
            ax1 = self.fig.add_subplot(311)
            ax2 = self.fig.add_subplot(312, sharex=ax1)
            ax3 = self.fig.add_subplot(313, sharex=ax1)
            self.subplots.extend([ax1, ax2, ax3])
        elif self.layout_mode == "2x2":
            ax1 = self.fig.add_subplot(221)
            ax2 = self.fig.add_subplot(222, sharex=ax1)
            ax3 = self.fig.add_subplot(223, sharex=ax1)
            ax4 = self.fig.add_subplot(224, sharex=ax1)
            self.subplots.extend([ax1, ax2, ax3, ax4])

        apply_plot_theme(self.fig, self.subplots, theme_name=self.theme_name)
        self.fig.tight_layout()
        self.draw_idle()

    def apply_theme(self, theme_name: str):
        """Sincronizza il tema con l'interfaccia centrale."""
        self.theme_name = theme_name
        apply_plot_theme(self.fig, self.subplots, theme_name=theme_name)
        t = get_theme_palette(theme_name)
        self.fig.patch.set_facecolor(t['plot_fig_bg'])
        self.update_plots()

    def update_plots(self):
        """Aggiorna i grafici con i canali correntemente selezionati."""
        t = get_theme_palette(self.theme_name)
        palette = t.get('plot_colors', ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff', '#ff7b72', '#d29922', '#39c5cf', '#e3b341'])

        if self.df_telemetry is None or self.df_telemetry.empty:
            for idx, ax in enumerate(self.subplots):
                ax.clear()
                ax.set_facecolor(t['plot_ax_bg'])
                ax.grid(True, linestyle=':', alpha=0.5, color=t['plot_grid'])
                ax.text(0.5, 0.5, f"Grafico #{idx+1}: Nessun dato telemetrico.\nEsegui l'ottimizzazione per ispezionare i canali.",
                        ha='center', va='center', transform=ax.transAxes, color=t['text_secondary'], fontsize=9)
            self.fig.tight_layout()
            self.draw_idle()
            return

        df = self.df_telemetry
        x_col = 's_m' if self.x_mode == 's' else 't_s'
        x_label = "Progressione Tracciato s [m]" if self.x_mode == 's' else "Tempo Cumulativo t [s]"
        x_data = df[x_col] if x_col in df.columns else np.arange(len(df))

        for idx, ax in enumerate(self.subplots):
            ax.clear()
            ax.set_facecolor(t['plot_ax_bg'])
            ax.grid(True, linestyle=':', alpha=0.5, color=t['plot_grid'])

            channels = self.active_channels.get(idx, [])
            if not channels:
                ax.text(0.5, 0.5, f"Grafico #{idx+1}: Nessun canale selezionato.\nSeleziona uno o piu canali dal pannello a sinistra.",
                        ha='center', va='center', transform=ax.transAxes, color=t['text_secondary'], fontsize=9)
                continue

            for c_i, ch in enumerate(channels):
                if ch in df.columns:
                    col = palette[c_i % len(palette)]
                    line_style = '-' if c_i == 0 else ('--' if c_i == 1 else '-.')
                    ax.plot(x_data, df[ch], label=ch, color=col, linestyle=line_style, linewidth=1.6)

            ax.set_ylabel("Valore", fontsize=8, color=t['plot_labels'])
            if idx == len(self.subplots) - 1 or self.layout_mode == "2x2" and idx in (2, 3):
                ax.set_xlabel(x_label, fontsize=8, color=t['plot_labels'])

            ch_str = ", ".join(channels[:3]) + ("..." if len(channels) > 3 else "")
            ax.set_title(f"Grafico #{idx+1}: {ch_str}", fontsize=8, fontweight='bold', color=t['plot_title'])

            leg = ax.legend(fontsize=7, loc='best')
            if leg:
                leg.get_frame().set_facecolor(t['plot_ax_bg'])
                leg.get_frame().set_edgecolor(t['plot_spines'])
                for txt in leg.get_texts():
                    txt.set_color(t['text_primary'])

        apply_plot_theme(self.fig, self.subplots, theme_name=self.theme_name)
        self.fig.tight_layout()
        self.draw_idle()

    def _on_mouse_move(self, event):
        """Visualizza un cursore crosshair verticale sincronizzato su tutti i subplot."""
        if event.inaxes is None or self.df_telemetry is None:
            return
        x_val = event.xdata
        if x_val is None:
            return

        t = get_theme_palette(self.theme_name)
        # Rimuovi vecchi cursori
        for l in self._cursor_lines:
            try:
                l.remove()
            except Exception:
                pass
        self._cursor_lines.clear()

        # Aggiungi nuova linea verticale
        for ax in self.subplots:
            l = ax.axvline(x=x_val, color=t['accent_warning'], linestyle=':', linewidth=1.0, alpha=0.8)
            self._cursor_lines.append(l)

        self.draw_idle()


# =============================================================================
# 3. PANNELLO CONFIGURAZIONE TRACCIATO (TRACK SELECTOR & CUSTOMIZER)
# =============================================================================
class TrackConfigPanel(QGroupBox):
    """
    Pannello laterale dedicato alla selezione, caricamento e customizzazione
    geometrica del tracciato e delle condizioni al contorno per Direct Collocation.
    """
    track_changed = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None, theme_name="dark"):
        super().__init__("Configurazione Tracciato & Vincoli", parent)
        self.theme_name = theme_name
        self.current_track_dict: Optional[Dict[str, Any]] = None
        self._init_ui()

    def _init_ui(self):
        t = get_theme_palette(self.theme_name)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 1. Preset Selector
        layout.addWidget(QLabel("<b>Tracciato:</b>"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "FSAE Autocross (850 m - Tecnico)",
            "FSAE Endurance (1250 m - Veloce)",
            "FSAE Skidpad (FSG Figure-8)",
            "FSAE Sprint / Accelerazione (150 m)",
            "Carica da File Personalizzato (.csv, .json)..."
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        layout.addWidget(self.combo_presets)

        # File Chooser Row (visibile se custom)
        self.file_row = QWidget()
        f_layout = QHBoxLayout(self.file_row)
        f_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_file_path = QLineEdit()
        self.txt_file_path.setReadOnly(True)
        self.txt_file_path.setPlaceholderText("Seleziona file CSV/JSON...")
        self.btn_browse = QPushButton("Sfoglia...")
        self.btn_browse.clicked.connect(self._browse_track_file)
        f_layout.addWidget(self.txt_file_path, 1)
        f_layout.addWidget(self.btn_browse)
        self.file_row.setVisible(False)
        layout.addWidget(self.file_row)

        # 2. Geometric Parameters
        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Larghezza Corsia w [m]:"), 0, 0)
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(2.0, 15.0)
        self.spin_width.setSingleStep(0.2)
        self.spin_width.setValue(3.8)
        self.spin_width.valueChanged.connect(self._rebuild_track_geometry)
        grid.addWidget(self.spin_width, 0, 1)

        grid.addWidget(QLabel("Passo Discretizz. ds [m]:"), 1, 0)
        self.spin_ds = QDoubleSpinBox()
        self.spin_ds.setRange(0.5, 10.0)
        self.spin_ds.setSingleStep(0.5)
        self.spin_ds.setValue(2.0)
        self.spin_ds.valueChanged.connect(self._rebuild_track_geometry)
        grid.addWidget(self.spin_ds, 1, 1)

        grid.addWidget(QLabel("Schema Collocazione:"), 2, 0)
        self.combo_scheme = QComboBox()
        self.combo_scheme.addItems([
            "Radau IIA 3° Grado (Consigliato, L-stabile)",
            "Gauss-Legendre 3° Grado (O(ds^6))",
            "Radau IIA 2° Grado (Veloce)",
            "Gauss-Legendre 2° Grado"
        ])
        grid.addWidget(self.combo_scheme, 2, 1)

        layout.addLayout(grid)

        # 3. Boundary Conditions
        gb_bc = QGroupBox("Condizioni al Contorno")
        bc_layout = QVBoxLayout(gb_bc)
        self.chk_closed_circuit = QCheckBox("Circuito Chiuso Periodico (Start = Finish)")
        self.chk_closed_circuit.setChecked(True)
        self.chk_closed_circuit.toggled.connect(self._on_closed_toggled)
        bc_layout.addWidget(self.chk_closed_circuit)

        bc_row = QHBoxLayout()
        bc_row.addWidget(QLabel("V_start [km/h]:"))
        self.spin_v0 = QDoubleSpinBox()
        self.spin_v0.setRange(0.0, 150.0)
        self.spin_v0.setValue(35.0)
        self.spin_v0.setEnabled(False)
        bc_row.addWidget(self.spin_v0)

        bc_row.addWidget(QLabel("V_end [km/h]:"))
        self.spin_vf = QDoubleSpinBox()
        self.spin_vf.setRange(0.0, 150.0)
        self.spin_vf.setValue(40.0)
        self.spin_vf.setEnabled(False)
        bc_row.addWidget(self.spin_vf)
        bc_layout.addLayout(bc_row)
        layout.addWidget(gb_bc)

        # 4. Track Information Card
        self.lbl_info = QLabel("Lunghezza: 850 m | Nodi: 425 | R_min: 7.2 m")
        self.lbl_info.setStyleSheet(f"background-color: {t['bg_input']}; color: #58a6ff; padding: 6px; border-radius: 4px; font-size: 11px;")
        layout.addWidget(self.lbl_info)

        # Preview Button
        self.btn_preview = QPushButton("Aggiorna Anteprima Tracciato")
        self.btn_preview.setStyleSheet("font-weight: bold; background-color: #238636; color: white; padding: 6px; border-radius: 4px;")
        self.btn_preview.clicked.connect(self._rebuild_track_geometry)
        layout.addWidget(self.btn_preview)

        layout.addStretch()
        # Genera tracciato iniziale di default
        self._rebuild_track_geometry()

    def _on_closed_toggled(self, checked: bool):
        """Abilita o disabilita campi velocita iniziale/finale se circuito aperto."""
        self.spin_v0.setEnabled(not checked)
        self.spin_vf.setEnabled(not checked)

    def _on_preset_selected(self, index: int):
        """Gestisce il cambio di preset."""
        is_custom = (index == 4)
        self.file_row.setVisible(is_custom)
        if not is_custom:
            if index == 2:  # Skidpad
                self.chk_closed_circuit.setChecked(False)
                self.spin_width.setValue(3.0)
            elif index == 3:  # Acceleration
                self.chk_closed_circuit.setChecked(False)
                self.spin_width.setValue(4.0)
            else:
                self.chk_closed_circuit.setChecked(True)
                self.spin_width.setValue(3.8)
            self._rebuild_track_geometry()

    def _browse_track_file(self):
        """Apre il dialogo di selezione file tracciato personalizzato."""
        data_dir = os.path.join(os.getcwd(), 'data', 'tracks')
        start_dir = data_dir if os.path.exists(data_dir) else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Seleziona Tracciato", start_dir, "File Tracciato (*.csv *.json *.mat);;Tutti i File (*.*)")
        if path:
            self.txt_file_path.setText(path)
            self._rebuild_track_geometry()

    def _rebuild_track_geometry(self):
        """Ricostruisce la geometria del tracciato e notifica il listener."""
        idx = self.combo_presets.currentIndex()
        ds = float(self.spin_ds.value())
        width = float(self.spin_width.value())

        type_map = {
            0: "fsae_autocross",
            1: "fsae_endurance",
            2: "fsae_skidpad",
            3: "fsae_acceleration"
        }

        try:
            if idx == 4:  # Custom file
                filepath = self.txt_file_path.text()
                if not filepath or not os.path.exists(filepath):
                    return
                track = load_custom_track_file(filepath, ds=ds, track_width_m=width)
            else:
                tt = type_map.get(idx, "fsae_autocross")
                track = build_track_geometry(track_type=tt, ds=ds, track_width_m=width)

            self.current_track_dict = track
            tot_len = track['s'][-1]
            n_pts = track['N']
            max_kap = np.max(np.abs(track['kappa']))
            r_min = (1.0 / max_kap) if max_kap > 1e-4 else float('inf')

            r_str = f"{r_min:.1f} m" if r_min != float('inf') else "Rettilineo"
            self.lbl_info.setText(f"Lunghezza: {tot_len:.1f} m | Nodi: {n_pts} | R_min: {r_str} | w: {width:.1f} m")
            self.track_changed.emit(track)
        except Exception as e:
            self.lbl_info.setText(f"[ERR] Errore Geometria: {str(e)[:40]}")

    def get_track_data(self) -> Optional[Dict[str, Any]]:
        """Restituisce il dizionario del tracciato attualmente attivo."""
        return self.current_track_dict

    def get_collocation_scheme(self) -> str:
        """Restituisce la chiave dello schema di collocazione polinomiale selezionato."""
        idx = self.combo_scheme.currentIndex()
        schemes = ["radau_3", "legendre_3", "radau_2", "legendre_2"]
        return schemes[idx] if idx < len(schemes) else "radau_3"


# =============================================================================
# 4. DIALOGO PRINCIPALE DIRECT COLLOCATION INTEGRATO
# =============================================================================
class DirectCollocationDialog(QDialog):
    """
    Finestra professionale per Ottimizzazione di Traiettoria & Racing Line OCP.
    Eredita rigorosamente il tema dell'applicazione, include selettore di tracciato,
    canvas 4-subplot isolato e Telemetry Data Inspector.
    """
    def __init__(self, parent=None, vehicle_params=None, tir_file=None, theme_name="dark", density="standard"):
        super().__init__(parent)
        self.vp = vehicle_params
        self.tir_file = tir_file
        self.theme_name = theme_name
        self.density = density
        self.last_colloc_res: Optional[Dict[str, Any]] = None
        self.active_worker: Optional[CollocationWorker] = None

        self.setWindowTitle("Direct Collocation: Ottimizzazione Traiettoria & Telemetry Data Inspector")
        self.resize(1280, 840)
        self.setStyleSheet(get_stylesheet(self.theme_name, self.density))
        apply_app_palette(self, self.theme_name)

        self._build_ui()

    def _build_ui(self):
        t = get_theme_palette(self.theme_name)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 1. Top Bar: Model Selection & Actions
        top_bar = QHBoxLayout()
        lbl_mod = QLabel("<b>Modello Veicolo:</b>")
        lbl_mod.setStyleSheet(f"color: {t['text_primary']};")
        top_bar.addWidget(lbl_mod)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "14-DOF (Full Non-Linear Transient)",
            "7-DOF (Chassis + 4 Wheels)",
            "3-DOF (Bicycle Model)"
        ])
        top_bar.addWidget(self.model_combo)

        lbl_iter = QLabel("Max Iter IPOPT:")
        lbl_iter.setStyleSheet(f"color: {t['text_primary']};")
        top_bar.addWidget(lbl_iter)
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(20, 1000)
        self.spin_max_iter.setValue(250)
        self.spin_max_iter.setSingleStep(50)
        top_bar.addWidget(self.spin_max_iter)

        self.btn_run = QPushButton("Esegui Ottimizzazione IPOPT")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #1f6feb; color: #ffffff; padding: 7px 18px; border-radius: 4px; border: 1px solid #388bfd;")
        self.btn_run.clicked.connect(self._run_optimization)
        top_bar.addWidget(self.btn_run)

        self.btn_export = QPushButton("Esporta CSV Telemetria")
        self.btn_export.setStyleSheet(f"padding: 7px 14px; font-weight: bold; background-color: {t['bg_input']}; color: {t['text_primary']}; border: 1px solid {t['border']}; border-radius: 4px;")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        top_bar.addWidget(self.btn_export)

        top_bar.addStretch()
        num_cpus = os.cpu_count() or 4
        lbl_cpu = QLabel(f"<b>CPU Multi-Threading:</b> {num_cpus} Thread paralleli attivi (MUMPS + OpenMP)")
        lbl_cpu.setStyleSheet("color: #58a6ff; font-size: 11px;")
        top_bar.addWidget(lbl_cpu)
        main_layout.addLayout(top_bar)

        # 2. Progress Bar & Status
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setTextVisible(True)
        self.prog_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid {t['border']}; border-radius: 3px; text-align: center; height: 18px; color: {t['text_primary']}; background-color: {t['bg_input']}; }} QProgressBar::chunk {{ background-color: #1f6feb; }}")
        main_layout.addWidget(self.prog_bar)

        self.lbl_status = QLabel("Configura il tracciato e avvia l'ottimizzazione per calcolare traiettoria, velocita e forze ottimali.")
        self.lbl_status.setStyleSheet(f"background-color: {t['bg_card']}; color: #58a6ff; border: 1px solid {t['border']}; padding: 6px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        main_layout.addWidget(self.lbl_status)

        # 3. Canvases & Visualization Setup (instantiated early for signal bindings)
        self.canvas_overview = CollocationTrajectoryCanvas(self, theme_name=self.theme_name)
        self.tb_overview = NavigationToolbar(self.canvas_overview, self)
        self.tb_overview.setStyleSheet(f"background-color: {t['bg_card']}; border: none; color: {t['text_secondary']};")

        self.canvas_inspector = DataInspectorCanvas(self, theme_name=self.theme_name)
        self.tb_inspector = NavigationToolbar(self.canvas_inspector, self)
        self.tb_inspector.setStyleSheet(f"background-color: {t['bg_card']}; border: none; color: {t['text_secondary']};")

        # 4. Main Splitter: Left Sidebar (Track & Data Inspector Config) | Right (Tabs View)
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANEL: TABS FOR CONFIGURATION ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.left_tabs = QTabWidget()

        # Tab A: Track Config Panel
        self.track_panel = TrackConfigPanel(theme_name=self.theme_name)
        self.track_panel.track_changed.connect(self._on_track_changed)
        self.left_tabs.addTab(self.track_panel, "Tracciato")

        # Tab B: Data Inspector Signal Selector
        inspector_ctrl = self._build_inspector_controls()
        self.left_tabs.addTab(inspector_ctrl, "Canali Inspector")

        left_layout.addWidget(self.left_tabs)

        # Iteration Log Box (Live IPOPT Console)
        self.txt_iter_log = QTextEdit()
        self.txt_iter_log.setMaximumHeight(110)
        self.txt_iter_log.setReadOnly(True)
        self.txt_iter_log.setStyleSheet(f"background-color: {t['bg_root']}; color: #3fb950; font-family: Consolas, monospace; font-size: 10px; border: 1px solid {t['border']}; border-radius: 4px;")
        self.txt_iter_log.setPlaceholderText("Registro delle iterazioni IPOPT...")
        left_layout.addWidget(self.txt_iter_log)

        splitter.addWidget(left_widget)

        # --- RIGHT PANEL: VISUALIZATION TABS ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.view_tabs = QTabWidget()

        # Tab 1: Engineering Overview 4-Subplots
        w_overview = QWidget()
        ov_layout = QVBoxLayout(w_overview)
        ov_layout.addWidget(self.tb_overview)
        ov_layout.addWidget(self.canvas_overview)
        self.view_tabs.addTab(w_overview, "Panoramica 4 Subplot (Standard)")

        # Tab 2: Telemetry Data Inspector
        w_inspector = QWidget()
        insp_layout = QVBoxLayout(w_inspector)
        insp_layout.addWidget(self.tb_inspector)
        insp_layout.addWidget(self.canvas_inspector)
        self.view_tabs.addTab(w_inspector, "Telemetry Data Inspector")

        # Tab 3: Telemetry Data Table
        self.telemetry_table = QTableWidget()
        self.telemetry_table.setStyleSheet(f"background-color: {t['bg_card']}; color: {t['text_primary']}; gridline-color: {t['border']}; font-size: 10px;")
        self.view_tabs.addTab(self.telemetry_table, "Tabella Numerica Dati")

        right_layout.addWidget(self.view_tabs)
        splitter.addWidget(right_widget)

        splitter.setSizes([340, 940])
        main_layout.addWidget(splitter, 1)

        # Disegna anteprima iniziale del tracciato
        curr_track = self.track_panel.get_track_data()
        if curr_track:
            self.canvas_overview.plot_track_preview(curr_track)

    def _build_inspector_controls(self) -> QWidget:
        """Costruisce il widget dei controlli di selezione canali Telemetry Data Inspector."""
        t = get_theme_palette(self.theme_name)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Layout selector
        layout.addWidget(QLabel("<b>Layout Subplot:</b>"))
        self.combo_layout_mode = QComboBox()
        self.combo_layout_mode.addItems(["2 Subplot Verticali (2x1)", "1 Subplot Singolo (1x1)", "3 Subplot Verticali (3x1)", "4 Subplot Griglia (2x2)"])
        self.combo_layout_mode.currentIndexChanged.connect(self._on_inspector_layout_changed)
        layout.addWidget(self.combo_layout_mode)

        # X-Axis Selector
        layout.addWidget(QLabel("<b>Asse Orizzontale (X):</b>"))
        x_group = QHBoxLayout()
        self.rb_x_s = QRadioButton("Progressione s [m]")
        self.rb_x_s.setChecked(True)
        self.rb_x_s.toggled.connect(lambda chk: self.canvas_inspector.set_x_mode("s") if chk else None)
        self.rb_x_t = QRadioButton("Tempo t [s]")
        self.rb_x_t.toggled.connect(lambda chk: self.canvas_inspector.set_x_mode("t") if chk else None)
        x_group.addWidget(self.rb_x_s)
        x_group.addWidget(self.rb_x_t)
        layout.addLayout(x_group)

        # Subplot selector to assign channels
        layout.addWidget(QLabel("<b>Configura Canali per Subplot:</b>"))
        self.combo_target_subplot = QComboBox()
        self.combo_target_subplot.addItems(["Subplot #1", "Subplot #2", "Subplot #3", "Subplot #4"])
        self.combo_target_subplot.currentIndexChanged.connect(self._sync_channel_checkboxes)
        layout.addWidget(self.combo_target_subplot)

        # Channel checkboxes in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        self.channels_layout = QVBoxLayout(scroll_w)
        self.channels_layout.setSpacing(4)

        self.available_channels = [
            # Velocita
            ("v_kmh", "Velocita Ottimizzata [km/h]"),
            ("v_profiler_kmh", "Velocita Profiler Quasi-Statico [km/h]"),
            ("delta_v_kmh", "Delta Guadagno Velocita (v_colloc - v_prof) [km/h]"),
            ("v_mps", "Velocita [m/s]"),
            ("t_s", "Tempo Cumulativo [s]"),
            # Accelerazioni & Forze
            ("ay_g", "Accelerazione Laterale Collocation a_y [g]"),
            ("ay_profiler_g", "Accelerazione Laterale Profiler a_y [g]"),
            ("ax_g", "Accelerazione Longitudinale Collocation a_x [g]"),
            ("ax_profiler_g", "Accelerazione Longitudinale Profiler a_x [g]"),
            ("atot_g", "Accelerazione Risultante a_tot [g]"),
            ("power_long_kw", "Potenza Longitudinale Stimata [kW]"),
            # Dinamica & Assetto
            ("steer_deg", "Angolo di Sterzo Ruote delta [deg]"),
            ("n_m", "Scostamento Laterale da Mezzeria n [m]"),
            ("xi_deg", "Angolo di Assetto / Yaw Relativo xi [deg]"),
            ("psi_deg", "Angolo di Yaw Globale psi [deg]"),
            ("yaw_rate_degs", "Velocita di Imbardata yaw_rate [deg/s]"),
        ]

        self.channel_checkboxes: Dict[str, QCheckBox] = {}
        for ch_key, ch_name in self.available_channels:
            cb = QCheckBox(f"{ch_name} ({ch_key})")
            cb.stateChanged.connect(self._on_channel_checkbox_toggled)
            self.channel_checkboxes[ch_key] = cb
            self.channels_layout.addWidget(cb)

        scroll_w.setLayout(self.channels_layout)
        scroll.setWidget(scroll_w)
        layout.addWidget(scroll, 1)

        self._sync_channel_checkboxes()
        return w

    def _on_inspector_layout_changed(self, idx: int):
        """Gestisce il cambio layout subplot dell'inspector."""
        modes = ["2x1", "1x1", "3x1", "2x2"]
        mode = modes[idx] if idx < len(modes) else "2x1"
        self.canvas_inspector.set_layout_mode(mode)

    def _sync_channel_checkboxes(self):
        """Sincronizza le checkbox con i canali attivi per il subplot selezionato."""
        sub_idx = self.combo_target_subplot.currentIndex()
        active = self.canvas_inspector.active_channels.get(sub_idx, [])
        for ch_key, cb in self.channel_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(ch_key in active)
            cb.blockSignals(False)

    def _on_channel_checkbox_toggled(self):
        """Aggiorna i canali del subplot attivo quando una checkbox cambia."""
        sub_idx = self.combo_target_subplot.currentIndex()
        selected = [k for k, cb in self.channel_checkboxes.items() if cb.isChecked()]
        self.canvas_inspector.set_plot_channels(sub_idx, selected)

    def _on_track_changed(self, track: Dict[str, Any]):
        """Callback di aggiornamento anteprima quando la geometria del tracciato cambia."""
        self.canvas_overview.plot_track_preview(track)

    def _run_optimization(self):
        """Lancia il solutore IPOPT asincrono tramite CollocationWorker."""
        track = self.track_panel.get_track_data()
        if not track:
            QMessageBox.warning(self, "Attenzione", "Seleziona o carica prima un tracciato valido.")
            return

        scheme = self.track_panel.get_collocation_scheme()
        max_iter = int(self.spin_max_iter.value())
        ds_val = float(self.track_panel.spin_ds.value())
        w_val = float(self.track_panel.spin_width.value())

        mod_idx = self.model_combo.currentIndex()
        model_key = "14-DOF" if mod_idx == 0 else ("7-DOF" if mod_idx == 1 else "3-DOF")

        self.btn_run.setEnabled(False)
        self.prog_bar.setValue(5)
        self.lbl_status.setText(f"Avvio ottimizzazione IPOPT ({model_key}) con schema {scheme}...")
        self.txt_iter_log.append(f"[{time.strftime('%H:%M:%S')}] Avvio OCP: schema={scheme}, ds={ds_val}m, max_iter={max_iter}")

        vp = self.vp
        worker = CollocationWorker(
            vp=vp, ds_val=ds_val, w_val=w_val, model_name=model_key,
            max_it=max_iter, track_data=track, collocation_scheme=scheme
        )
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        self.active_worker = worker
        worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        """Callback per aggiornamento barra di avanzamento e log."""
        self.prog_bar.setValue(pct)
        self.lbl_status.setText(f"[{pct}%] {msg}")
        if "Iterazione" in msg or "Risoluzione" in msg or "IPOPT" in msg:
            self.txt_iter_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.txt_iter_log.verticalScrollBar().setValue(self.txt_iter_log.verticalScrollBar().maximum())

    def _on_worker_finished(self, res: Dict[str, Any]):
        """Callback eseguita al completamento positivo dell'ottimizzazione."""
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(res.get('success', False) or res.get('telemetry') is not None)
        self.last_colloc_res = res

        df = res['telemetry']
        prof = res['profiler']
        t_colloc = res['lap_time_s']
        t_prof = prof['lap_time_s']
        d_time = res['delta_time_s']
        d_pct = res['delta_time_pct']
        v_max = res['v_max_kmh']
        v_avg = res['v_avg_kmh']
        ay_max = res['max_ay_g']
        dev_max = res['max_lateral_dev_m']
        mod_str = res.get('model_type', '14-DOF')

        self.prog_bar.setValue(100)
        self.lbl_status.setText(
            f"[OK] OTTIMIZZATO ({mod_str}) | Lap Collocation: {t_colloc:.2f} s vs Profiler: {t_prof:.2f} s (Delta: {d_time:+.2f}s / {d_pct:+.1f}%) | "
            f"V_avg: {v_avg:.1f} km/h | V_max: {v_max:.1f} km/h | Max Ay: {ay_max:.2f} g | Deviazione n: +/-{dev_max:.2f} m"
        )
        self.txt_iter_log.append(f"[{time.strftime('%H:%M:%S')}] [OK] Soluzione Ottimale Trovata! Lap Time: {t_colloc:.2f} s")
        self.txt_iter_log.verticalScrollBar().setValue(self.txt_iter_log.verticalScrollBar().maximum())

        # 1. Aggiorna Panoramica 4 Subplot
        self.canvas_overview.plot_optimization_results(res)

        # 2. Aggiorna Data Inspector
        self.canvas_inspector.set_data(df)

        # 3. Aggiorna Tabella Numerica Dati
        self._populate_telemetry_table(df)

    def _on_worker_failed(self, err_msg: str):
        """Callback in caso di errore durante l'esecuzione del worker."""
        self.btn_run.setEnabled(True)
        self.prog_bar.setValue(0)
        self.lbl_status.setText("[ERR] Errore durante l'ottimizzazione IPOPT.")
        self.txt_iter_log.append(f"[{time.strftime('%H:%M:%S')}] [ERR] {err_msg}")
        QMessageBox.critical(self, "Errore Collocation", f"Si e verificato un errore:\n{err_msg}")

    def _populate_telemetry_table(self, df: pd.DataFrame):
        """Popola la tabella numerica dei dati con formattazione a 3 decimali."""
        cols = [c for c in df.columns if not c.startswith('_')]
        self.telemetry_table.setColumnCount(len(cols))
        self.telemetry_table.setRowCount(len(df))
        self.telemetry_table.setHorizontalHeaderLabels(cols)

        # Campiona per efficienza se troppe righe (es. > 1500)
        step = max(1, len(df) // 1000)
        row_indices = list(range(0, len(df), step))
        self.telemetry_table.setRowCount(len(row_indices))

        for r_i, df_idx in enumerate(row_indices):
            row_data = df.iloc[df_idx]
            for c_i, col_name in enumerate(cols):
                val = row_data[col_name]
                text = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.telemetry_table.setItem(r_i, c_i, item)

    def _export_csv(self):
        """Esporta il dataframe dei risultati in formato CSV."""
        if not self.last_colloc_res or self.last_colloc_res.get('telemetry') is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Direct Collocation CSV", "direct_collocation_telemetry.csv", "CSV Files (*.csv)")
        if path:
            self.last_colloc_res['telemetry'].to_csv(path, index=False)
            QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
