"""
main_gui_pro.py
---------------
Interfaccia Grafica Professionale di Seconda Generazione per Dynamic Target Suite.
Caratteristiche architetturali:
- Layout a 5 Workspace Ingegneristici (Dinamica Quasistatica, Eventi Dinamici & Telemetria,
  Assetto & Sospensioni, MDO & Sensibilita, Dashboard KPI & Punteggi FSAE).
- Inspector laterale parametrico contestuale e collassabile (modalita Full-Canvas).
- Header tecnico con status badge, selettore profili, selezione modelli e tir.
- Motore di personalizzazione avanzato (Motorsport Scuro, Ingegneristico Chiaro, Alto Contrasto)
  e calibrazione densita (Standard, Compatta).
- Sincronizzazione cromatica automatica dei canvas Matplotlib con il tema attivo.
- Totale eliminazione di emoji e grafica non tecnica; adozione di testo conciso,
  rigoroso e conforme alle normative industriali SAE J670e e ISO 4138.
- Pieno supporto al backup dell'interfaccia classica: avviabile da menu dedicato o da CLI.
"""

import sys
import os
import re
import subprocess
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, QGroupBox,
    QGridLayout, QComboBox, QMessageBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QTextBrowser, QAction, QMenuBar, QScrollArea,
    QSplitter, QDialog, QFileDialog, QProgressBar, QSlider,
    QDoubleSpinBox, QSpinBox, QSizePolicy, QStackedWidget, QFrame
)

from src.ui import main_gui_classic as legacy_gui
from src.ui.ui_theme_engine import (
    get_stylesheet, apply_plot_theme, load_ui_settings,
    save_ui_settings, get_theme_palette, apply_app_palette,
    THEMES, DENSITIES
)

def strip_emojis(text: str) -> str:
    """Rimuove categoricamente simboli ed emoji unicode convertendo gli indicatori in notazioni tecniche ASCII."""
    if not text:
        return ""
    t = str(text)
    replacements = {
        "✔": "[OK]", "✅": "[OK]",
        "❌": "[ERR]", "✖": "[X]",
        "⚠": "[!]",
        "➜": "->", "→": "->", "←": "<-",
        "±": "+/-", "°": " deg",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    cleaned = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27ff\ufe0f\u200d\u20e3\u2300-\u23ff\u2b50-\u2b55\u3030\u303d\u3297\u3299]', '', t)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class LineEditStartFilter(QtCore.QObject):
    """Garantisce che il testo nelle caselle di testo (QLineEdit) sia sempre visualizzato dall'inizio."""
    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.FocusOut, QtCore.QEvent.Show, QtCore.QEvent.Resize):
            if isinstance(obj, QLineEdit) and not obj.hasFocus():
                obj.setCursorPosition(0)
        return super().eventFilter(obj, event)

# Patch idempotente su QLineEdit per far si che setText e init posizionino sempre il cursore all'inizio
_orig_qlineedit_init = QLineEdit.__init__
def _safe_qlineedit_init(self, *args, **kwargs):
    _orig_qlineedit_init(self, *args, **kwargs)
    self.setCursorPosition(0)
QLineEdit.__init__ = _safe_qlineedit_init

_orig_qlineedit_setText = QLineEdit.setText
def _safe_qlineedit_setText(self, text):
    _orig_qlineedit_setText(self, text)
    self.setCursorPosition(0)
QLineEdit.setText = _safe_qlineedit_setText

# Intercettori globali per sanitizzare rigorosamente qualsiasi testo da emoji
_orig_btn_setText = QPushButton.setText
def _safe_btn_setText(self, text):
    _orig_btn_setText(self, strip_emojis(str(text)) if text else "")
QPushButton.setText = _safe_btn_setText

_orig_lbl_setText = QLabel.setText
def _safe_lbl_setText(self, text):
    _orig_lbl_setText(self, strip_emojis(str(text)) if text else "")
QLabel.setText = _safe_lbl_setText

_orig_gb_setTitle = QGroupBox.setTitle
def _safe_gb_setTitle(self, text):
    _orig_gb_setTitle(self, strip_emojis(str(text)) if text else "")
QGroupBox.setTitle = _safe_gb_setTitle

_orig_act_setText = QAction.setText
def _safe_act_setText(self, text):
    _orig_act_setText(self, strip_emojis(str(text)) if text else "")
QAction.setText = _safe_act_setText

_orig_tw_setTabText = QTabWidget.setTabText
def _safe_tw_setTabText(self, index, text):
    _orig_tw_setTabText(self, index, strip_emojis(str(text)) if text else "")
QTabWidget.setTabText = _safe_tw_setTabText

_orig_tw_addTab = QTabWidget.addTab
def _safe_tw_addTab(self, *args):
    if len(args) == 2 and isinstance(args[1], str):
        return _orig_tw_addTab(self, args[0], strip_emojis(args[1]))
    elif len(args) == 3 and isinstance(args[2], str):
        return _orig_tw_addTab(self, args[0], args[1], strip_emojis(args[2]))
    return _orig_tw_addTab(self, *args)
QTabWidget.addTab = _safe_tw_addTab

class MainWindowPro(legacy_gui.MainWindow):
    """Finestra principale dell'interfaccia grafica professionale Dynamic Target Suite Pro."""
    
    def __init__(self):
        # Inizializza prima la classe base per istanziare backend, canvas, input e callback
        super(MainWindowPro, self).__init__()
        
        # PRESERVAZIONE CRUCIALE: Prende ownership del central widget classico di MainWindow
        # evitando che Qt C++ lo marchi per DeferredDelete quando viene installato il nuovo central widget pro.
        # In questo modo tutti i 58 controlli, tab, subtabs e layout di MainWindow rimangono vivi e integri in memoria.
        self._legacy_central_widget = self.takeCentralWidget()
        if self._legacy_central_widget is not None:
            self._legacy_central_widget.hide()
        
        self.line_edit_filter = LineEditStartFilter(self)
        
        # Carica configurazione preferenze utente (predefinito: light)
        self.ui_settings = load_ui_settings()
        self.current_theme = self.ui_settings.get('theme', 'light')
        self.current_density = self.ui_settings.get('density', 'standard')
        self.active_workspace_idx = self.ui_settings.get('active_workspace', 0)
        self.sidebar_is_visible = self.ui_settings.get('sidebar_visible', True)
        
        # Costruisce la nuova interfaccia professionale
        self._build_pro_interface()
        
        # Applica tema iniziale
        self._apply_active_theme()

    def _build_pro_interface(self):
        """Riorganizza layout, menu, inspector laterale e workspace in stile ingegneristico."""
        self.setWindowTitle("DYNAMIC TARGET SUITE PRO | Analisi Dinamica del Veicolo & MDO FSAE")
        self.resize(1440, 920)
        self.setMinimumSize(1024, 700)

        # 1. Ristrutturazione Menu Bar
        self._build_pro_menubar()

        # 2. Creazione Widget Centrale Unificato
        pro_central_widget = QWidget()
        pro_main_layout = QVBoxLayout(pro_central_widget)
        pro_main_layout.setContentsMargins(4, 4, 4, 4)
        pro_main_layout.setSpacing(4)

        # 3. Header di Comando e Stato Tecnico
        header_widget = self._build_pro_header()
        pro_main_layout.addWidget(header_widget)

        # 4. Navigation Rail / Barra Workspace Primari
        workspace_nav_widget = self._build_workspace_navigation_rail()
        pro_main_layout.addWidget(workspace_nav_widget)

        # 5. Splitter Orizzontale Principale (Inspector Laterale + Area Analitica)
        self.pro_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.pro_splitter.setHandleWidth(3)

        # 5a. Inspector Laterale (Parametri, Vincoli, Eventi Dinamici, Sweep)
        self.pro_inspector = self._build_pro_inspector()
        self.pro_splitter.addWidget(self.pro_inspector)

        # 5b. Area Analitica a 5 Workspace
        self.pro_workspace_stack = self._build_analytical_workspaces()
        self.pro_splitter.addWidget(self.pro_workspace_stack)

        # Imposta proporzioni splitter (Inspector ~420px, Workspace ~1020px)
        self.pro_inspector.setMinimumWidth(380)
        self.pro_splitter.setStretchFactor(0, 0)
        self.pro_splitter.setStretchFactor(1, 1)
        self.pro_splitter.setSizes([420, 1020])

        pro_main_layout.addWidget(self.pro_splitter, stretch=1)

        # 6. Barra di Stato & Console Tecnica Inferiore
        self.pro_statusbar = self._build_pro_statusbar()
        pro_main_layout.addWidget(self.pro_statusbar)

        # Sostituisce il central widget
        self.setCentralWidget(pro_central_widget)

        # Routing automatico delle chiamate setCurrentWidget da self.tabs verso i workspace Pro
        def _pro_set_current_tab(widget):
            try:
                legacy_gui.QTabWidget.setCurrentWidget(self.tabs, widget)
            except Exception:
                pass
            self._navigate_to_widget(widget)
            
        self.tabs.setCurrentWidget = _pro_set_current_tab

        # 7. Pulizia Ricorsiva Emoji e Normalizzazione Testo
        self._scrub_all_emojis(self)

        # 8. Reset stili inline legacy e assegnazione ruoli tecnici
        self._reset_and_apply_roles()

        # 9. Sincronizzazione Canvas Matplotlib
        self._sync_all_matplotlib_canvases()

        # 10. Garantisce visibilita iniziale completa su tutte le caselle di testo
        self._setup_line_edits_visibility()

    def _navigate_to_widget(self, widget):
        """Mappa in modo reattivo qualsiasi widget analitico legacy al workspace e subtab Pro corrispondente."""
        if widget is None:
            return
        
        mapping = {
            getattr(self, 'tab_ymd', None): (0, getattr(self, 'tab_ymd_single', None)),
            getattr(self, 'tab_ymd_single', None): (0, getattr(self, 'tab_ymd_single', None)),
            getattr(self, 'tab_ymd_multi', None): (0, getattr(self, 'tab_ymd_multi', None)),
            getattr(self, 'tab_ggv', None): (0, getattr(self, 'tab_ggv', None)),
            getattr(self, 'tab_handling', None): (0, getattr(self, 'tab_hd_diagram', None)),
            getattr(self, 'tab_hd_diagram', None): (0, getattr(self, 'tab_hd_diagram', None)),
            getattr(self, 'tab_kus_analysis', None): (0, getattr(self, 'tab_kus_analysis', None)),
            getattr(self, 'tab_maps', None): (0, getattr(self, 'tab_maps', None)),
            getattr(self, 'tab_telemetry', None): (1, getattr(self, 'tab_telem_skidpad', None)),
            getattr(self, 'tab_telem_skidpad', None): (1, getattr(self, 'tab_telem_skidpad', None)),
            getattr(self, 'tab_telem_accel', None): (1, getattr(self, 'tab_telem_accel', None)),
            getattr(self, 'tab_telem_step', None): (1, getattr(self, 'tab_telem_step', None)),
            getattr(self, 'tab_telem_autox', None): (1, getattr(self, 'tab_telem_autox', None)),
            getattr(self, 'tab_susp_sizing', None): (2, getattr(self, 'tab_susp_sizing', None)),
            getattr(self, 'tab_arb_diamond', None): (2, getattr(self, 'tab_arb_diamond', None)),
            getattr(self, 'tab_damper_dyno', None): (2, getattr(self, 'tab_damper_dyno', None)),
            getattr(self, 'tab_seven_post', None): (2, getattr(self, 'tab_seven_post', None)),
            getattr(self, 'tab_steer_torque', None): (2, getattr(self, 'tab_steer_torque', None)),
            getattr(self, 'tab_kinematics', None): (2, getattr(self, 'tab_kinematics', None)),
            getattr(self, 'tab_sweep_res', None): (3, getattr(self, 'tab_sweep_res', None)),
            getattr(self, 'tab_sens_1d', None): (3, getattr(self, 'tab_sens_1d', None)),
            getattr(self, 'tab_radar_balance', None): (3, getattr(self, 'tab_radar_balance', None)),
            getattr(self, 'tab_transient_sensitivity', None): (3, getattr(self, 'tab_transient_sensitivity', None)),
            getattr(self, 'tab_kpi', None): (4, getattr(self, 'tab_kpi', None)),
            getattr(self, 'tab_sens_scoring', None): (4, getattr(self, 'tab_sens_scoring', None)),
        }
        
        target = mapping.get(widget)
        if target:
            ws_idx, sub_widget = target
            self.switch_workspace(ws_idx)
            if sub_widget is not None and hasattr(self, 'pro_workspace_stack'):
                ws_tab = self.pro_workspace_stack.widget(ws_idx)
                if isinstance(ws_tab, QTabWidget) and sub_widget:
                    ws_tab.setCurrentWidget(sub_widget)

    def _setup_line_edits_visibility(self):
        """Assicura che tutte le caselle di testo mostrino il testo partendo dal primo carattere."""
        for le in self.findChildren(QLineEdit):
            le.setCursorPosition(0)
            if hasattr(self, 'line_edit_filter'):
                le.installEventFilter(self.line_edit_filter)

    def _rebuild_params_panel(self):
        """Ricostruisce il pannello parametri assicurando 100% di spazio e testo visibile dall'inizio."""
        super(MainWindowPro, self)._rebuild_params_panel()
        # Nasconde la guida orfana in params_splitter per dare tutto lo spazio alla tabella parametri
        if hasattr(self, 'guide_widget') and self.guide_widget:
            self.guide_widget.hide()
        if hasattr(self, 'params_splitter') and self.params_splitter:
            self.params_splitter.setSizes([1000, 0])
        self._setup_line_edits_visibility()

    def _reset_and_apply_roles(self):
        """Azzera gli stili inline saturati legacy e applica le classi di ruolo professionali."""
        for w in self.findChildren((QPushButton, QLabel, QComboBox, QGroupBox)):
            if w.objectName() not in ["pro_header", "workspace_nav_bar", "pro_inspector_panel", "pro_statusbar"]:
                w.setStyleSheet("")

        primary_buttons = [
            'run_btn', 'btn_run_skidpad', 'btn_run_accel', 'btn_run_step', 'btn_run_dlc',
            'btn_run_autox', 'btn_run_colloc', 'btn_save_as_setup', 'btn_recalc_ymd',
            'btn_calc_multispeed_ymd', 'btn_calc_susp_sizing', 'btn_calc_damper_dyno',
            'btn_calc_steering_torque', 'btn_calc_balance_range', 'btn_calc_radar',
            'btn_calc_fsae_scores', 'btn_run_sweep',
            'btn_calc_diamond_fast', 'btn_calc_diamond_2d', 'btn_calc_diamond_1d',
            'btn_save_as_diamond_cfg'
        ]
        for attr in primary_buttons:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setProperty("role", "primary")

        success_buttons = [
            'btn_save_setup', 'btn_apply_opt_to_init', 'btn_apply_kin_gains',
            'btn_apply_diamond_sweetspot'
        ]
        for attr in success_buttons:
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setProperty("role", "success")

        # Ruoli Run MDO e Switch Layout
        if hasattr(self, 'run_btn') and self.run_btn:
            self.run_btn.setProperty("role", "primary_run")
        if hasattr(self, 'btn_header_run') and self.btn_header_run:
            self.btn_header_run.setProperty("role", "primary_run")
        if hasattr(self, 'btn_run_constant') and self.btn_run_constant:
            self.btn_run_constant.setProperty("role", "secondary_run")
        if hasattr(self, 'btn_header_eval') and self.btn_header_eval:
            self.btn_header_eval.setProperty("role", "secondary_run")
        if hasattr(self, 'btn_switch_classic') and self.btn_switch_classic:
            self.btn_switch_classic.setProperty("role", "layout_switch")

    def export_history_csv_dialog(self):
        """Esporta lo storico ottimizzazioni in formato CSV."""
        from src.utils.optimized_params_manager import export_history_to_csv
        path, _ = QFileDialog.getSaveFileName(self, "Salva Storico Ottimizzazioni CSV", "optimization_history.csv", "CSV Files (*.csv)")
        if path:
            if export_history_to_csv(path):
                QMessageBox.information(self, "Esporta CSV", f"Storico esportato con successo in:\n{path}")

    def _build_pro_menubar(self):
        """Costruisce un menu bar rigoroso, tecnico e privo di emoji."""
        mb = self.menuBar()
        mb.clear()

        # --- Menu File ---
        file_menu = mb.addMenu("File")
        
        act_save_setup = QAction("Salva Setup Corrente", self)
        act_save_setup.setShortcut("Ctrl+S")
        act_save_setup.triggered.connect(self.save_current_setup_profile)
        file_menu.addAction(act_save_setup)

        act_save_as_setup = QAction("Salva Setup con Nome...", self)
        act_save_as_setup.triggered.connect(self.save_as_new_setup_profile)
        file_menu.addAction(act_save_as_setup)

        act_load_setup = QAction("Carica Setup da File (JSON)...", self)
        act_load_setup.setShortcut("Ctrl+O")
        act_load_setup.triggered.connect(self.browse_load_setup_profile)
        file_menu.addAction(act_load_setup)

        file_menu.addSeparator()

        act_export_history = QAction("Esporta Storico Ottimizzazioni (CSV)...", self)
        act_export_history.triggered.connect(self.export_history_csv_dialog)
        file_menu.addAction(act_export_history)

        file_menu.addSeparator()

        act_exit = QAction("Esci", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # --- Menu Visualizzazione & Temi ---
        view_menu = mb.addMenu("Visualizzazione")

        theme_sub = view_menu.addMenu("Schema Colori")
        self.act_theme_dark = QAction("Motorsport Scuro (Predefinito)", self, checkable=True)
        self.act_theme_dark.setChecked(self.current_theme == 'dark')
        self.act_theme_dark.triggered.connect(lambda: self.set_theme('dark'))
        theme_sub.addAction(self.act_theme_dark)

        self.act_theme_light = QAction("Ingegneristico Chiaro (Laboratorio / Documentazione)", self, checkable=True)
        self.act_theme_light.setChecked(self.current_theme == 'light')
        self.act_theme_light.triggered.connect(lambda: self.set_theme('light'))
        theme_sub.addAction(self.act_theme_light)

        self.act_theme_contrast = QAction("Alto Contrasto Monocromatico", self, checkable=True)
        self.act_theme_contrast.setChecked(self.current_theme == 'contrast')
        self.act_theme_contrast.triggered.connect(lambda: self.set_theme('contrast'))
        theme_sub.addAction(self.act_theme_contrast)

        density_sub = view_menu.addMenu("Densita Interfaccia")
        self.act_density_std = QAction("Standard (Display Desktop 1440p/4K)", self, checkable=True)
        self.act_density_std.setChecked(self.current_density == 'standard')
        self.act_density_std.triggered.connect(lambda: self.set_density('standard'))
        density_sub.addAction(self.act_density_std)

        self.act_density_comp = QAction("Compatta (Laptop 1080p / Alta Densita)", self, checkable=True)
        self.act_density_comp.setChecked(self.current_density == 'compact')
        self.act_density_comp.triggered.connect(lambda: self.set_density('compact'))
        density_sub.addAction(self.act_density_comp)

        view_menu.addSeparator()

        act_toggle_sidebar = QAction("Mostra/Nascondi Inspector Parametrico", self)
        act_toggle_sidebar.setShortcut("Ctrl+B")
        act_toggle_sidebar.triggered.connect(self.toggle_inspector_visibility)
        view_menu.addAction(act_toggle_sidebar)

        act_fullscreen = QAction("Schermo Intero", self)
        act_fullscreen.setShortcut("F11")
        act_fullscreen.triggered.connect(self._toggle_fullscreen_mode)
        view_menu.addAction(act_fullscreen)

        view_menu.addSeparator()

        # VOCE CRUCIALE: AVVIO BACKUP CLASSICO
        act_launch_classic = QAction("Avvia Interfaccia Legacy (Backup)", self)
        act_launch_classic.setToolTip("Avvia l'interfaccia grafica originale di backup in un processo separato.")
        act_launch_classic.triggered.connect(self.launch_classic_backup_gui)
        view_menu.addAction(act_launch_classic)

        # --- Menu Strumenti ---
        tools_menu = mb.addMenu("Strumenti")
        
        act_digitizer = QAction("Curve Digitizer (Cinematica Sospensioni)", self)
        act_digitizer.triggered.connect(self.launch_digitizer)
        tools_menu.addAction(act_digitizer)

        act_colloc = QAction("Collocazione Diretta (Ottimizzazione Traiettoria)", self)
        act_colloc.triggered.connect(self.run_direct_collocation_dialog)
        tools_menu.addAction(act_colloc)

        act_driver = QAction("Simulatore di Guida Real-Time 14-DOF", self)
        act_driver.triggered.connect(self.launch_driving_simulator)
        tools_menu.addAction(act_driver)

        act_preview_3d = QAction("Studio Anteprima 3D Veicolo (FSAE & SUV)", self)
        act_preview_3d.triggered.connect(self.launch_3d_preview_studio)
        tools_menu.addAction(act_preview_3d)

        act_7post = QAction("Banco Dinamico 7-Post Shaker Rig", self)
        act_7post.triggered.connect(self.open_seven_post_rig)
        tools_menu.addAction(act_7post)

        tools_menu.addSeparator()

        act_inspect_params = QAction("Ispezione Parametri Attivi (16 Curve Cinematiche)...", self)
        act_inspect_params.triggered.connect(self.show_active_parameters_dialog)
        tools_menu.addAction(act_inspect_params)

        act_import_curves = QAction("Importa Cartella Curve Cinematiche (.json)...", self)
        act_import_curves.triggered.connect(self.import_digitized_curves_folder)
        tools_menu.addAction(act_import_curves)

        # --- Menu Opzioni ---
        options_menu = mb.addMenu("Opzioni")
        
        act_sim_opt = QAction("Parametri del Solutore & Simulazione", self)
        act_sim_opt.triggered.connect(self.show_simulation_options_dialog)
        options_menu.addAction(act_sim_opt)

        act_veh_opt = QAction("Parametri Fisici Completi del Veicolo", self)
        act_veh_opt.triggered.connect(self.show_options_dialog)
        options_menu.addAction(act_veh_opt)

        act_history = QAction("Storico Sessioni & Setup Salvati", self)
        act_history.triggered.connect(self.show_optimization_history_dialog)
        options_menu.addAction(act_history)

        options_menu.addSeparator()

        self.lang_en_action = QAction("Grafici in Lingua Inglese (Report / Paper SAE)", self, checkable=True)
        self.lang_en_action.triggered.connect(self.on_language_toggled)
        options_menu.addAction(self.lang_en_action)

        # --- Menu Guida ---
        help_menu = mb.addMenu("Guida")
        
        act_user_guide = QAction("Manuale Tecnico & Riferimenti Dinamica del Veicolo", self)
        act_user_guide.setShortcut("F1")
        act_user_guide.triggered.connect(self.show_user_guide_dialog)
        help_menu.addAction(act_user_guide)

    def _build_pro_header(self) -> QWidget:
        """Header tecnico a due livelli con stato solutore, setup rapido e controlli di visualizzazione."""
        header = QFrame()
        header.setObjectName("pro_header")
        header.setFrameShape(QFrame.StyledPanel)
        v_layout = QVBoxLayout(header)
        v_layout.setContentsMargins(8, 5, 8, 5)
        v_layout.setSpacing(4)

        # =========================================================================
        # RIGA 1: Identificativo Suite, Profilo Setup, Modello Veicolo e Pneumatico
        # =========================================================================
        r1 = QHBoxLayout()
        r1.setSpacing(8)

        # 1. Badge Titolo Suite
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(1)
        lbl_title = QLabel("DYNAMIC TARGET SUITE PRO")
        lbl_title.setStyleSheet("font-weight: 800; font-size: 10.5pt; letter-spacing: 0.5px;")
        self.lbl_subtitle = QLabel("FSAE Vehicle Dynamics & MDO Target Optimization")
        self.lbl_subtitle.setStyleSheet("font-size: 7.5pt; font-weight: 500; color: #1e293b;")
        brand_layout.addWidget(lbl_title)
        brand_layout.addWidget(self.lbl_subtitle)
        r1.addLayout(brand_layout)

        # Separatore verticale
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        r1.addWidget(sep1)

        # 2. Setup Profilo Selettore & Azioni Rapide
        setup_box = QHBoxLayout()
        setup_box.setSpacing(5)
        lbl_setup = QLabel("Profilo Setup:")
        lbl_setup.setStyleSheet("font-weight: 600; font-size: 8.5pt;")
        setup_box.addWidget(lbl_setup)

        self.combo_setup_profile.setMinimumWidth(160)
        setup_box.addWidget(self.combo_setup_profile)

        self.btn_save_setup.setText("Salva")
        self.btn_save_setup.setToolTip("Salva le modifiche nel profilo attivo")
        self.btn_save_setup.setMinimumWidth(60)
        setup_box.addWidget(self.btn_save_setup)

        self.btn_save_as_setup.setText("Salva con Nome...")
        self.btn_save_as_setup.setToolTip("Salva la configurazione del veicolo in un nuovo file .json con nome personalizzato")
        self.btn_save_as_setup.setMinimumWidth(125)
        setup_box.addWidget(self.btn_save_as_setup)

        self.btn_load_setup.setText("Sfoglia...")
        self.btn_load_setup.setMinimumWidth(75)
        setup_box.addWidget(self.btn_load_setup)

        r1.addLayout(setup_box)

        # Separatore verticale
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        r1.addWidget(sep2)

        # 3. Modello Veicolo & Pneumatico TIR
        veh_box = QHBoxLayout()
        veh_box.setSpacing(5)

        lbl_model = QLabel("Modello:")
        lbl_model.setStyleSheet("font-weight: 600; font-size: 8.5pt;")
        veh_box.addWidget(lbl_model)

        self.model_combo.clear()
        self.model_combo.addItems([
            "3-DOF Quasistatico (Trim & MDO)",
            "7-DOF Transitorio (Pitch, Roll, 4 Ruote)",
            "14-DOF Multibody Completo (Sospensioni 3D)"
        ])
        self.model_combo.setMinimumWidth(170)
        veh_box.addWidget(self.model_combo)

        lbl_tir = QLabel("Pneumatico:")
        lbl_tir.setStyleSheet("font-weight: 600; font-size: 8.5pt;")
        veh_box.addWidget(lbl_tir)
        self.tir_combo.setMinimumWidth(150)
        veh_box.addWidget(self.tir_combo)

        r1.addLayout(veh_box)
        r1.addStretch(1)
        v_layout.addLayout(r1)

        # =========================================================================
        # RIGA 2: Esecuzione MDO, Banner di Stato Tecnico e Controlli Visualizzazione
        # =========================================================================
        r2 = QHBoxLayout()
        r2.setSpacing(8)

        # 4. Pulsanti Primari Esecuzione MDO (Ampi, chiari, in primo piano)
        run_box = QHBoxLayout()
        run_box.setSpacing(6)

        self.btn_header_run = QPushButton("RUN OTTIMIZZAZIONE MDO")
        self.btn_header_run.setProperty("role", "primary_run")
        self.btn_header_run.setToolTip("Avvia l'ottimizzazione globale target multidisciplinare (SLSQP / OpenMDAO)")
        self.btn_header_run.setMinimumWidth(220)
        self.btn_header_run.clicked.connect(self.run_optimization)
        run_box.addWidget(self.btn_header_run)

        self.btn_header_eval = QPushButton("VALUTA PARAMETRI")
        self.btn_header_eval.setProperty("role", "secondary_run")
        self.btn_header_eval.setToolTip("Calcola istantaneamente tutti i diagrammi (YMD, G-G-V, Handling, MAPs, KPI) sui parametri correnti senza ottimizzare")
        self.btn_header_eval.setMinimumWidth(165)
        self.btn_header_eval.clicked.connect(self.run_constant_parameters)
        run_box.addWidget(self.btn_header_eval)

        r2.addLayout(run_box)

        # Separatore verticale
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        r2.addWidget(sep3)

        # Status informativo testuale
        self.lbl_header_status = QLabel("STATO: PRONTO | FSAE Vehicle Dynamics & Multi-Disciplinary Optimization")
        self.lbl_header_status.setStyleSheet("font-size: 8.5pt; font-weight: 600; color: #1e293b;")
        r2.addWidget(self.lbl_header_status)

        r2.addStretch(1)

        # 5. Selettori Rapidi Visualizzazione (Tema, Densita, Sidebar, Switch Layout Classico)
        quick_view_box = QHBoxLayout()
        quick_view_box.setSpacing(6)

        lbl_theme = QLabel("Tema:")
        lbl_theme.setStyleSheet("font-size: 8.5pt; font-weight: 600;")
        quick_view_box.addWidget(lbl_theme)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Motorsport Scuro", "Ingegneristico Chiaro", "Alto Contrasto"])
        theme_map = {'dark': 0, 'light': 1, 'contrast': 2}
        self.theme_combo.setCurrentIndex(theme_map.get(self.current_theme, 1))
        self.theme_combo.setMinimumWidth(155)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        quick_view_box.addWidget(self.theme_combo)

        lbl_density = QLabel("Densità:")
        lbl_density.setStyleSheet("font-size: 8.5pt; font-weight: 600;")
        quick_view_box.addWidget(lbl_density)

        self.density_combo = QComboBox()
        self.density_combo.addItems(["Standard", "Compatta"])
        density_map = {'standard': 0, 'compact': 1}
        self.density_combo.setCurrentIndex(density_map.get(self.current_density, 0))
        self.density_combo.setMinimumWidth(90)
        self.density_combo.currentIndexChanged.connect(self._on_density_combo_changed)
        quick_view_box.addWidget(self.density_combo)

        self.btn_toggle_sidebar = QPushButton("Inspector On/Off")
        self.btn_toggle_sidebar.setCheckable(True)
        self.btn_toggle_sidebar.setChecked(self.sidebar_is_visible)
        self.btn_toggle_sidebar.setMinimumWidth(110)
        self.btn_toggle_sidebar.setToolTip("Mostra o nasconde la barra laterale dei parametri (Ctrl+B)")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_inspector_visibility)
        quick_view_box.addWidget(self.btn_toggle_sidebar)

        self.btn_inspect_params_pro = QPushButton("Parametri Attivi")
        self.btn_inspect_params_pro.setMinimumWidth(110)
        self.btn_inspect_params_pro.setToolTip("Ispeziona tutti i parametri attivi del veicolo e le 16 curve cinematiche Adams Car caricate")
        self.btn_inspect_params_pro.clicked.connect(self.show_active_parameters_dialog)
        quick_view_box.addWidget(self.btn_inspect_params_pro)

        self.btn_switch_classic = QPushButton("Layout Classico (Backup)")
        self.btn_switch_classic.setProperty("role", "layout_switch")
        self.btn_switch_classic.setMinimumWidth(160)
        self.btn_switch_classic.setToolTip("Apre l'interfaccia grafica classica originale di backup in-process")
        self.btn_switch_classic.clicked.connect(self.launch_classic_backup_gui)
        quick_view_box.addWidget(self.btn_switch_classic)

        r2.addLayout(quick_view_box)
        v_layout.addLayout(r2)

        return header

    def _build_workspace_navigation_rail(self) -> QWidget:
        """Barra di navigazione primaria per i 5 workspace ingegneristici."""
        nav_frame = QFrame()
        nav_frame.setObjectName("workspace_nav_bar")
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(4, 3, 4, 3)
        nav_layout.setSpacing(6)

        self.workspace_buttons = []
        workspace_definitions = [
            ("1. Quasistatica", "1. Dinamica Quasistatica: Trim stazionario, diagrammi YMD, handling, GGV e mappe prestazionali MAPs"),
            ("2. Transitorio", "2. Dinamica Transitoria & Telemetria: Skidpad ISO, accelerazione 75m, step steer, DLC e AutoX"),
            ("3. Sospensioni", "3. Assetto & Componenti Sospensioni: Dimensionamento molle/barre, dyno TTX25, 7-post rig, sterzo e cinematica"),
            ("4. MDO & Sweep", "4. Ottimizzazione MDO & Sensibilità: OpenMDAO, sweep vincoli, tornado plot e radar 4-rapporti"),
            ("5. Dashboard KPI", "5. Dashboard KPI & Punteggi FSAE: Riepilogo prestazioni, delta target e punteggi regolamento 2026")
        ]

        for idx, (title, tip) in enumerate(workspace_definitions):
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setChecked(idx == self.active_workspace_idx)
            btn.setToolTip(tip)
            btn.setProperty("workspace_idx", idx)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setMinimumHeight(28)
            btn.setMinimumWidth(135)
            btn.setStyleSheet("font-weight: 600; font-size: 9.5pt; padding: 4px 10px;")
            btn.clicked.connect(lambda checked, i=idx: self.switch_workspace(i))
            nav_layout.addWidget(btn, stretch=1)
            self.workspace_buttons.append(btn)

        return nav_frame

    def _build_pro_inspector(self) -> QWidget:
        """Pannello laterale collassabile per parametri di progetto, eventi dinamici e sweep."""
        inspector = QFrame()
        inspector.setObjectName("pro_inspector_panel")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(2, 2, 2, 2)
        inspector_layout.setSpacing(4)

        # Tab widget interno dell'inspector
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setUsesScrollButtons(True)
        
        # Tab 1: Parametri & Vincoli MDO (riutilizza self.tab_params)
        self.inspector_tabs.addTab(self.tab_params, "Setup & Vincoli")
        self.inspector_tabs.setTabToolTip(0, "Parametri di Progetto, Valori Iniziali e Vincoli MDO")

        # Tab 2: Eventi Dinamici (riutilizza self.tab_dynamic_scroll o self.tab_dynamic)
        tab_dyn_widget = getattr(self, 'tab_dynamic_scroll', self.tab_dynamic)
        self.inspector_tabs.addTab(tab_dyn_widget, "Eventi Dinamici")
        self.inspector_tabs.setTabToolTip(1, "Configurazione Manovre ed Eventi Transitori")

        # Tab 3: Sweep & Impostazioni (riutilizza self.tab_sweep_scroll o self.tab_sweep)
        tab_sweep_widget = getattr(self, 'tab_sweep_scroll', self.tab_sweep)
        self.inspector_tabs.addTab(tab_sweep_widget, "Sweep Parametri")
        self.inspector_tabs.setTabToolTip(2, "Sweep di Velocità e Diagrammi di Manovrabilità")

        # Rimuove l'ingombro della guida orfana in params_splitter per dare tutto lo spazio alla tabella parametri
        if hasattr(self, 'guide_widget') and self.guide_widget:
            self.guide_widget.hide()
        if hasattr(self, 'params_splitter') and self.params_splitter:
            self.params_splitter.setSizes([1000, 0])

        # Splitter Verticale dell'Inspector (Tabs + Comandi sopra, Console Log sotto)
        self.inspector_splitter = QSplitter(QtCore.Qt.Vertical)
        self.inspector_splitter.setHandleWidth(5)
        self.inspector_splitter.setChildrenCollapsible(False)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(self.inspector_tabs, stretch=1)

        # 4. Riquadro Risoluzione & Esecuzione Rapida nell'Inspector (Sempre Accessibile)
        exec_group = QGroupBox("COMANDI DI RISOLUZIONE & RUN")
        exec_layout = QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(6, 6, 6, 6)
        exec_layout.setSpacing(4)

        self.run_btn.setText("RUN OTTIMIZZAZIONE GLOBALE MDO")
        self.run_btn.setProperty("role", "primary_run")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.setToolTip("Avvia il solutore SLSQP/OpenMDAO per ottimizzare i target veicolo secondo i vincoli impostati")
        exec_layout.addWidget(self.run_btn)

        self.btn_run_constant.setText("VALUTA PARAMETRI CORRENTI")
        self.btn_run_constant.setProperty("role", "secondary_run")
        self.btn_run_constant.setMinimumHeight(26)
        self.btn_run_constant.setToolTip("Valuta i parametri veicolo correnti e aggiorna tutti i diagrammi (YMD, GGV, MAPs, KPI) senza ottimizzare")
        exec_layout.addWidget(self.btn_run_constant)
        top_layout.addWidget(exec_group)
        self.inspector_splitter.addWidget(top_container)

        # 5. Console Output Solutore (Ridimensionabile fluidamente verso l'alto e verso il basso)
        log_group = QGroupBox("CONSOLE OUTPUT SOLUTORE")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(4)

        log_head = QHBoxLayout()
        log_head.setContentsMargins(0, 0, 0, 0)
        lbl_log = QLabel("Log Iterazioni:")
        lbl_log.setStyleSheet("font-weight: 600; font-size: 8.5pt;")
        log_head.addWidget(lbl_log)
        log_head.addStretch()

        btn_popout = QPushButton("Finestra Dedicata")
        btn_popout.setStyleSheet("font-size: 7.5pt; padding: 2px 6px; font-weight: bold; background-color: #2980b9; color: white; border-radius: 3px;")
        btn_popout.setToolTip("Apri la console del solutore in una finestra dedicata e ridimensionabile a tutto schermo")
        btn_popout.clicked.connect(self.show_console_popout_dialog)
        log_head.addWidget(btn_popout)

        btn_clr = QPushButton("Pulisci")
        btn_clr.setMaximumWidth(60)
        btn_clr.setStyleSheet("font-size: 7.5pt; padding: 1px 4px;")
        btn_clr.clicked.connect(self.log_output.clear)
        log_head.addWidget(btn_clr)
        log_layout.addLayout(log_head)

        self.log_output.setMinimumHeight(50)
        self.log_output.setMaximumHeight(16777215)
        log_layout.addWidget(self.log_output)
        self.inspector_splitter.addWidget(log_group)

        top_container.setMinimumHeight(100)
        log_group.setMinimumHeight(60)
        self.inspector_splitter.setStretchFactor(0, 1)
        self.inspector_splitter.setStretchFactor(1, 1)
        self.inspector_splitter.setSizes([450, 300])

        inspector_layout.addWidget(self.inspector_splitter)

        return inspector

    def show_console_popout_dialog(self):
        """Apre la console del solutore in una finestra dedicata separata ad alta risoluzione."""
        from src.ui.gui_dialogs import ConsolePopoutDialog
        dlg = ConsolePopoutDialog(self.log_output, parent=self)
        dlg.exec_()

    def _build_analytical_workspaces(self) -> QStackedWidget:
        """Crea i 5 workspace ingegneristici alloggiati nello stack principale."""
        stack = QStackedWidget()

        # =========================================================================
        # WORKSPACE 0: Dinamica Quasistatica
        # =========================================================================
        ws0 = QTabWidget()
        ws0.setObjectName("ws_quasistatic")
        ws0.setUsesScrollButtons(True)
        ws0.addTab(self.tab_ymd_single, "YMD Singolo")
        ws0.setTabToolTip(0, "Diagramma del Momento d'Imbardata (YMD) Singolo")
        ws0.addTab(self.tab_ymd_multi, "YMD Multi-Velocità")
        ws0.setTabToolTip(1, "Overlay YMD Multi-Velocità & Derivate di Stabilità dMz/dbeta")
        ws0.addTab(self.tab_ggv, "Inviluppo G-G-V")
        ws0.setTabToolTip(2, "Inviluppo di Aderenza G-G-V ad Assetto Variabile")
        ws0.addTab(self.tab_hd_diagram, "Handling Diagram")
        ws0.setTabToolTip(3, "Diagramma di Manovrabilità (Handling) di Pacejka")
        ws0.addTab(self.tab_kus_analysis, "Gradiente K_us")
        ws0.setTabToolTip(4, "Gradiente di Sottosterzo K_us (drho/day)")
        ws0.addTab(self.tab_maps, "MAPs")
        ws0.setTabToolTip(5, "Map of Achievable Performance (MAPs)")

        # Integra barra controlli parametri YMD direttamente nel tab YMD Singolo
        if hasattr(self, 'tab_ymd_single') and self.tab_ymd_single.layout() is not None:
            ymd_ctrl_widget = QWidget()
            ctrl_layout = QHBoxLayout(ymd_ctrl_widget)
            ctrl_layout.setContentsMargins(4, 2, 4, 4)
            ctrl_layout.setSpacing(8)
            
            lbl_vx = QLabel("<b>Velocità vx:</b>")
            ctrl_layout.addWidget(lbl_vx)
            if hasattr(self, 'ymd_vx_input'):
                ctrl_layout.addWidget(self.ymd_vx_input)
            if hasattr(self, 'ymd_vx_slider'):
                ctrl_layout.addWidget(self.ymd_vx_slider)
            ctrl_layout.addWidget(QLabel("m/s"))
            
            if hasattr(self, 'ymd_show_radii_cb'):
                ctrl_layout.addWidget(self.ymd_show_radii_cb)
            if hasattr(self, 'btn_recalc_ymd'):
                ctrl_layout.addWidget(self.btn_recalc_ymd)
            if hasattr(self, 'btn_calc_multispeed_ymd'):
                ctrl_layout.addWidget(self.btn_calc_multispeed_ymd)
            if hasattr(self, 'btn_info_ymd'):
                ctrl_layout.addWidget(self.btn_info_ymd)
                
            ctrl_layout.addStretch()
            self.tab_ymd_single.layout().insertWidget(0, ymd_ctrl_widget)

        # Integra barra controlli per YMD Multi-Velocità
        if hasattr(self, 'tab_ymd_multi') and self.tab_ymd_multi.layout() is not None:
            multi_ctrl_widget = QWidget()
            multi_layout = QHBoxLayout(multi_ctrl_widget)
            multi_layout.setContentsMargins(4, 2, 4, 4)
            multi_layout.setSpacing(8)
            multi_layout.addWidget(QLabel("<b>Velocità Multi [m/s]:</b>"))
            if hasattr(self, 'ymd_multispeeds_input'):
                multi_layout.addWidget(self.ymd_multispeeds_input)
            btn_multi = QPushButton("Genera Overlay Multi-Velocità")
            btn_multi.clicked.connect(self.plot_mmm_multispeed)
            multi_layout.addWidget(btn_multi)
            multi_layout.addStretch()
            self.tab_ymd_multi.layout().insertWidget(0, multi_ctrl_widget)

        stack.addWidget(ws0)

        # =========================================================================
        # WORKSPACE 1: Dinamica Transitoria & Telemetria
        # =========================================================================
        ws1 = QTabWidget()
        ws1.setObjectName("ws_dynamic_events")
        ws1.setUsesScrollButtons(True)
        ws1.addTab(self.tab_telem_skidpad, "Skidpad ISO")
        ws1.setTabToolTip(0, "Telemetria Skidpad (Normativa ISO 4138)")
        ws1.addTab(self.tab_telem_accel, "Accelerazione 75m")
        ws1.setTabToolTip(1, "Telemetria Prova di Accelerazione (75 m)")
        ws1.addTab(self.tab_telem_step, "Step Steer & DLC")
        ws1.setTabToolTip(2, "Risposta al Transitorio: Step Steer & Double Lane Change ISO 3888-2")
        ws1.addTab(self.tab_telem_autox, "AutoX / Endurance")
        ws1.setTabToolTip(3, "Simulazione Percorso Completo (AutoX / Endurance 22km)")
        stack.addWidget(ws1)

        # =========================================================================
        # WORKSPACE 2: Assetto & Componenti Sospensioni
        # =========================================================================
        if not hasattr(self, 'tab_seven_post') or self.tab_seven_post is None:
            try:
                from src.ui.ui_seven_post_rig import SevenPostRigTab
                self.tab_seven_post = SevenPostRigTab(main_window=self)
            except Exception as e:
                self.tab_seven_post = None

        ws2 = QTabWidget()
        ws2.setObjectName("ws_chassis_suspension")
        ws2.setUsesScrollButtons(True)
        ws2.addTab(self.tab_susp_sizing, "Rigidezze & ARB")
        ws2.setTabToolTip(0, "Dimensionamento Molle, Wheel Rate e Barre Antirollio")
        if hasattr(self, 'tab_arb_diamond') and self.tab_arb_diamond:
            ws2.addTab(self.tab_arb_diamond, "Rombo ARB 14-DOF")
            ws2.setTabToolTip(ws2.count() - 1, "Rombo di Setting ARB & Bilanciamento Meccanico con Modello 14-DOF")
        ws2.addTab(self.tab_damper_dyno, "Dyno TTX25")
        ws2.setTabToolTip(ws2.count() - 1, "Caratterizzazione Ammortizzatori Öhlins TTX25 MkII FSAE")
        if hasattr(self, 'tab_seven_post') and self.tab_seven_post:
            ws2.addTab(self.tab_seven_post, "7-Post Shaker Rig")
            ws2.setTabToolTip(ws2.count() - 1, "Banco Dinamico 7-Post Shaker Rig (4 Attuatori Stradali + 3 Aerodinamici)")
        ws2.addTab(self.tab_steer_torque, "Coppia Sterzo")
        ws2.setTabToolTip(ws2.count() - 1, "Geometria di Sterzo, Bracci a Terra e Coppia Pilota")
        ws2.addTab(self.tab_kinematics, "Cinematica Digitizer")
        ws2.setTabToolTip(ws2.count() - 1, "Mappe Cinematiche Ruota: Camber, Toe e Centri di Rollio")
        stack.addWidget(ws2)

        # =========================================================================
        # WORKSPACE 3: Ottimizzazione MDO & Sensibilita
        # =========================================================================
        ws3 = QTabWidget()
        ws3.setObjectName("ws_mdo_sensitivity")
        ws3.setUsesScrollButtons(True)
        ws3.addTab(self.tab_sweep_res, "Sweep OpenMDAO")
        ws3.setTabToolTip(0, "Sweep Parametrico e Convergenza OpenMDAO")
        ws3.addTab(self.tab_sens_1d, "Sensibilità 1D & Tornado")
        ws3.setTabToolTip(1, "Analisi di Sensibilità Parametrica Monovariabile e Tornado Multi-Parametro")
        ws3.addTab(self.tab_radar_balance, "Bilanciamento & Radar")
        ws3.setTabToolTip(2, "Finestra di Bilanciamento Dinamico & Radar 4-Rapporti Meccanici/Aero")
        ws3.addTab(self.tab_transient_sensitivity, "Sensibilità Transitoria & RC")
        ws3.setTabToolTip(3, "Sensibilità ai Centri di Rollio e Risposta Transitoria")
        stack.addWidget(ws3)

        # =========================================================================
        # WORKSPACE 4: Dashboard KPI & Punteggi FSAE
        # =========================================================================
        ws4 = QTabWidget()
        ws4.setObjectName("ws_kpi_scoring")
        ws4.setUsesScrollButtons(True)
        ws4.addTab(self.tab_kpi, "Dashboard KPI")
        ws4.setTabToolTip(0, "Dashboard Prestazioni e Delta di Ottimizzazione Target")
        ws4.addTab(self.tab_sens_scoring, "Punteggi FSAE 2026")
        ws4.setTabToolTip(1, "Calcolo Punteggi Regolamento Ufficiale Formula Student 2026")

        # Documentazione e guida tecnica integrata
        doc_widget = QWidget()
        doc_layout = QVBoxLayout(doc_widget)
        doc_layout.setContentsMargins(4, 4, 4, 4)
        lbl_doc_header = QLabel("Documentazione e Basi Fisiche del Modello Veicolo Attivo")
        lbl_doc_header.setStyleSheet("font-weight: bold; font-size: 10pt;")
        doc_layout.addWidget(lbl_doc_header)
        doc_layout.addWidget(self.guide_browser)
        ws4.addTab(doc_widget, "Guida & Equazioni")
        ws4.setTabToolTip(2, "Documentazione Completa del Modello Veicolo ed Equazioni Matematiche")

        stack.addWidget(ws4)

        # Rinomina in modo compatto i subtab di maps_subtabs se presenti
        if hasattr(self, 'maps_subtabs') and self.maps_subtabs:
            self.maps_subtabs.setUsesScrollButtons(True)
            map_titles = [
                ("Regioni Ingresso (δ, u)", "Regioni di Ingresso Realizzabili nel piano (δ, u, ay)"),
                ("Regioni Uscita (ρ, β)", "Regioni di Uscita e Miste nel piano (ρ, β, δ)"),
                ("Mappe Stabilità DNA", "Mappe del DNA del Veicolo e Derivate di Stabilità (n1...n6)"),
                ("Parametri MAPs", "Parametri di Calcolo, Velocità ed Evidenziazioni per MAPs")
            ]
            for i, (title, tip) in enumerate(map_titles):
                if i < self.maps_subtabs.count():
                    self.maps_subtabs.setTabText(i, title)
                    self.maps_subtabs.setTabToolTip(i, tip)

        stack.setCurrentIndex(self.active_workspace_idx)
        return stack

    def _build_pro_statusbar(self) -> QWidget:
        """Barra di stato concisa e tecnica."""
        bar = QFrame()
        bar.setObjectName("pro_statusbar")
        bar.setFrameShape(QFrame.StyledPanel)
        b_layout = QHBoxLayout(bar)
        b_layout.setContentsMargins(8, 2, 8, 2)
        b_layout.setSpacing(12)

        self.lbl_status_engine = QLabel("STATO: PRONTO")
        self.lbl_status_engine.setStyleSheet("font-weight: 700; color: #3fb950;")
        b_layout.addWidget(self.lbl_status_engine)

        self.lbl_status_setup = QLabel("Profilo: et18_2026_kinematics.json")
        self.lbl_status_setup.setStyleSheet("color: #8b949e;")
        b_layout.addWidget(self.lbl_status_setup)

        b_layout.addStretch(1)

        self.lbl_status_kpi = QLabel("Solutore: CasADi / OpenMDAO SLSQP | Convenzione: rho_y = -K/L")
        self.lbl_status_kpi.setStyleSheet("font-size: 8pt; color: #8b949e;")
        b_layout.addWidget(self.lbl_status_kpi)

        return bar

    # =========================================================================
    # GESTIONE TEMI, DENSITA E LAYOUT
    # =========================================================================

    def switch_workspace(self, idx: int):
        """Passa al workspace selezionato e aggiorna lo stato dei pulsanti di navigazione."""
        self.active_workspace_idx = idx
        self.pro_workspace_stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.workspace_buttons):
            btn.setChecked(i == idx)
        self.ui_settings['active_workspace'] = idx
        save_ui_settings(self.ui_settings)

    def set_theme(self, theme_name: str):
        """Imposta lo schema colori attivo e aggiorna QSS e Matplotlib."""
        if theme_name not in THEMES:
            return
        self.current_theme = theme_name
        self.act_theme_dark.setChecked(theme_name == 'dark')
        self.act_theme_light.setChecked(theme_name == 'light')
        self.act_theme_contrast.setChecked(theme_name == 'contrast')
        
        theme_map = {'dark': 0, 'light': 1, 'contrast': 2}
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentIndex(theme_map.get(theme_name, 0))
        self.theme_combo.blockSignals(False)

        self._apply_active_theme()

    def set_density(self, density_name: str):
        """Imposta la densita dell'interfaccia (standard vs compatta)."""
        if density_name not in DENSITIES:
            return
        self.current_density = density_name
        self.act_density_std.setChecked(density_name == 'standard')
        self.act_density_comp.setChecked(density_name == 'compact')
        
        density_map = {'standard': 0, 'compact': 1}
        self.density_combo.blockSignals(True)
        self.density_combo.setCurrentIndex(density_map.get(density_name, 0))
        self.density_combo.blockSignals(False)

        self._apply_active_theme()

    def _on_theme_combo_changed(self, idx: int):
        keys = ['dark', 'light', 'contrast']
        if 0 <= idx < len(keys):
            self.set_theme(keys[idx])

    def _on_density_combo_changed(self, idx: int):
        keys = ['standard', 'compact']
        if 0 <= idx < len(keys):
            self.set_density(keys[idx])

    def _apply_active_theme(self):
        """Genera e applica il foglio di stile QSS a tutta l'applicazione."""
        qss = get_stylesheet(self.current_theme, self.current_density)
        self.setStyleSheet(qss)

        # Sincronizza la palette dell'applicazione Qt DOPO il foglio di stile
        # affinche Matplotlib e i controlli rilevino correttamente Dark/Light mode
        app_inst = QtWidgets.QApplication.instance()
        if app_inst:
            apply_app_palette(app_inst, self.current_theme)
        apply_app_palette(self, self.current_theme)

        self._sync_all_matplotlib_canvases()
        
        # Aggiornamento dinamico etichette per contrasto elevato costante
        t = get_theme_palette(self.current_theme)
        if hasattr(self, 'lbl_subtitle') and self.lbl_subtitle:
            self.lbl_subtitle.setStyleSheet(f"font-size: 8pt; font-weight: 500; color: {t['text_secondary']};")
        if hasattr(self, 'lbl_status_setup') and self.lbl_status_setup:
            self.lbl_status_setup.setStyleSheet(f"font-weight: 600; color: {t['text_secondary']};")
        if hasattr(self, 'lbl_status_kpi') and self.lbl_status_kpi:
            self.lbl_status_kpi.setStyleSheet(f"font-size: 8.5pt; font-weight: 500; color: {t['text_secondary']};")
        if hasattr(self, 'lbl_status_engine') and self.lbl_status_engine:
            self.lbl_status_engine.setStyleSheet(f"font-weight: 700; color: {t['accent_secondary']};")
        if hasattr(self, 'lbl_header_status') and self.lbl_header_status:
            self.lbl_header_status.setStyleSheet(f"font-size: 8pt; font-weight: 600; color: {t['text_secondary']}; background: {t['bg_card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 2px 8px;")
        
        # Salva preferenze
        self.ui_settings['theme'] = self.current_theme
        self.ui_settings['density'] = self.current_density
        save_ui_settings(self.ui_settings)

    def toggle_inspector_visibility(self):
        """Mostra o nasconde l'inspector laterale per visualizzazione a tutto schermo dei grafici."""
        is_vis = self.pro_inspector.isVisible()
        new_vis = not is_vis
        self.pro_inspector.setVisible(new_vis)
        self.btn_toggle_sidebar.setChecked(new_vis)
        self.ui_settings['sidebar_visible'] = new_vis
        save_ui_settings(self.ui_settings)

    def _toggle_fullscreen_mode(self):
        """Attiva o disattiva la modalita a schermo intero."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def open_seven_post_rig(self):
        """Naviga direttamente al tab 7-Post Shaker Rig nel Workspace 2."""
        self.switch_workspace(2)
        if hasattr(self, 'tab_seven_post') and self.tab_seven_post:
            ws2 = self.pro_workspace_stack.widget(2)
            if isinstance(ws2, QTabWidget):
                ws2.setCurrentWidget(self.tab_seven_post)

    def launch_classic_backup_gui(self):
        """Apre o porta in primo piano l'interfaccia classica di backup in-process in modo affidabile."""
        try:
            self.ui_settings['preferred_gui'] = 'classic'
            save_ui_settings(self.ui_settings)
            
            if not hasattr(self, '_classic_win') or self._classic_win is None:
                self._classic_win = legacy_gui.MainWindow()
            self._classic_win.show()
            self._classic_win.raise_()
            self._classic_win.activateWindow()
            self.hide()
            if hasattr(self, 'lbl_status_engine') and self.lbl_status_engine:
                self.lbl_status_engine.setText("STATO: Interfaccia classica di backup aperta.")
        except Exception as e:
            QMessageBox.critical(self, "Errore Avvio Backup", f"Impossibile aprire l'interfaccia classica:\n{e}")

    def run_optimization(self):
        """Esegue l'ottimizzazione globale target multidisciplinare sincronizzando i controlli."""
        self._set_running_state(True, "OTTIMIZZAZIONE MDO")
        try:
            super(MainWindowPro, self).run_optimization()
        finally:
            self._set_running_state(False)
            self.switch_workspace(4)

    def run_constant_parameters(self):
        """Valuta i parametri veicolo correnti aggiornando grafici e commutando su Dashboard KPI."""
        self._set_running_state(True, "VALUTAZIONE PARAMETRI")
        try:
            super(MainWindowPro, self).run_constant_parameters()
        finally:
            self._set_running_state(False)
            self.switch_workspace(4)

    def _set_running_state(self, is_running: bool, task_name="CALCOLO"):
        """Aggiorna lo stato visivo dei pulsanti di esecuzione e della barra di stato."""
        t = get_theme_palette(self.current_theme)
        if is_running:
            if hasattr(self, 'btn_header_run') and self.btn_header_run:
                self.btn_header_run.setEnabled(False)
                self.btn_header_run.setText(f"{task_name} IN CORSO...")
            if hasattr(self, 'btn_header_eval') and self.btn_header_eval:
                self.btn_header_eval.setEnabled(False)
            if hasattr(self, 'run_btn') and self.run_btn:
                self.run_btn.setEnabled(False)
                self.run_btn.setText(f"{task_name} IN CORSO...")
            if hasattr(self, 'btn_run_constant') and self.btn_run_constant:
                self.btn_run_constant.setEnabled(False)
            if hasattr(self, 'lbl_status_engine') and self.lbl_status_engine:
                self.lbl_status_engine.setText(f"STATO: {task_name} IN CORSO...")
                self.lbl_status_engine.setStyleSheet(f"font-weight: 700; color: {t['accent_warning']};")
            if hasattr(self, 'lbl_header_status') and self.lbl_header_status:
                self.lbl_header_status.setText(f"STATO: {task_name} IN CORSO...")
                self.lbl_header_status.setStyleSheet(f"font-size: 8pt; font-weight: 600; color: {t['accent_warning']}; background: {t['bg_card']}; border: 1px solid {t['accent_warning']}; border-radius: 4px; padding: 2px 8px;")
        else:
            if hasattr(self, 'btn_header_run') and self.btn_header_run:
                self.btn_header_run.setEnabled(True)
                self.btn_header_run.setText("RUN OTTIMIZZAZIONE MDO")
            if hasattr(self, 'btn_header_eval') and self.btn_header_eval:
                self.btn_header_eval.setEnabled(True)
                self.btn_header_eval.setText("VALUTA PARAMETRI")
            if hasattr(self, 'run_btn') and self.run_btn:
                self.run_btn.setEnabled(True)
                self.run_btn.setText("RUN OTTIMIZZAZIONE GLOBALE MDO")
            if hasattr(self, 'btn_run_constant') and self.btn_run_constant:
                self.btn_run_constant.setEnabled(True)
                self.btn_run_constant.setText("VALUTA PARAMETRI CORRENTI")
            if hasattr(self, 'lbl_status_engine') and self.lbl_status_engine:
                self.lbl_status_engine.setText("STATO: PRONTO")
                self.lbl_status_engine.setStyleSheet(f"font-weight: 700; color: {t['accent_secondary']};")
            if hasattr(self, 'lbl_header_status') and self.lbl_header_status:
                self.lbl_header_status.setText("STATO: PRONTO | FSAE Vehicle Dynamics & Multi-Disciplinary Optimization")
                self.lbl_header_status.setStyleSheet(f"font-size: 8pt; font-weight: 600; color: {t['text_secondary']}; background: {t['bg_card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 2px 8px;")

    # =========================================================================
    # SINCRONIZZAZIONE MATPLOTLIB & PULIZIA EMOJI
    # =========================================================================

    def _sync_all_matplotlib_canvases(self):
        """Sincronizza tutti i canvas Matplotlib con i colori del tema attivo e rende i pulsanti di navigazione sempre visibili."""
        canvas_attrs = [
            'canvas_ymd', 'canvas_ymd_multi', 'canvas_ggv', 'canvas_handling', 'canvas_kus',
            'canvas_maps_dna', 'canvas_maps_input', 'canvas_maps_output', 'canvas_sweep',
            'canvas_kpi_delta', 'canvas_telem_skidpad', 'canvas_telem_accel', 'canvas_telem_step',
            'canvas_telem_autox', 'canvas_steer_torque', 'canvas_susp_sizing', 'canvas_damper_dyno',
            'canvas_radar', 'canvas_arb_diamond', 'canvas_sensitivity', 'canvas_kinematics', 'canvas_sens', 'canvas_scores'
        ]
        for attr in canvas_attrs:
            canvas = getattr(self, attr, None)
            if canvas is not None and hasattr(canvas, 'fig'):
                apply_plot_theme(canvas.fig, theme_name=self.current_theme)
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

        if hasattr(self, 'tab_seven_post') and self.tab_seven_post is not None:
            if hasattr(self.tab_seven_post, 'apply_theme'):
                self.tab_seven_post.apply_theme(self.current_theme)

        # Sincronizza tutti i QToolBar e i relativi QToolButton presenti
        for tb in self.findChildren(QtWidgets.QToolBar):
            apply_app_palette(tb, self.current_theme)
            for btn in tb.findChildren(QtWidgets.QToolButton):
                btn.setAutoRaise(False)
                icon = btn.icon()
                if not icon.isNull():
                    btn.setIcon(icon)
                btn.update()
            tb.update()

    def _scrub_all_emojis(self, root_widget):
        """Scansiona ricorsivamente tutti i widget e rimuove qualsiasi emoji e testo informale."""
        for btn in root_widget.findChildren(QPushButton):
            txt = btn.text()
            if txt:
                btn.setText(strip_emojis(txt))
        for lbl in root_widget.findChildren(QLabel):
            txt = lbl.text()
            if txt and not txt.startswith('<') and len(txt) < 120:
                lbl.setText(strip_emojis(txt))
        for gb in root_widget.findChildren(QGroupBox):
            txt = gb.title()
            if txt:
                gb.setTitle(strip_emojis(txt))
        for cb in root_widget.findChildren(QCheckBox):
            txt = cb.text()
            if txt:
                cb.setText(strip_emojis(txt))
        for tw in root_widget.findChildren(QTabWidget):
            for i in range(tw.count()):
                tt = tw.tabText(i)
                if tt:
                    tw.setTabText(i, strip_emojis(tt))

    # =========================================================================
    # AGGIORNAMENTI STATO
    # =========================================================================

    def update_active_setup_banner(self, text):
        """Aggiorna sia il banner classico che la barra di stato professionale."""
        super(MainWindowPro, self).update_active_setup_banner(text)
        clean_t = strip_emojis(text)
        if hasattr(self, 'lbl_status_setup'):
            self.lbl_status_setup.setText(clean_t[:90])
        if hasattr(self, 'lbl_header_status') and self.lbl_header_status:
            self.lbl_header_status.setText(f"SETUP: {clean_t[:70]}")


def main():
    """Entry point principale per Dynamic Target Suite Pro."""
    legacy_gui.setup_global_exception_handler()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Controlla eventuale argomento CLI per avviare il backup classico
    if "--classic" in sys.argv or "--legacy" in sys.argv:
        print("[INFO] Avvio richiesto per l'interfaccia classica di backup.")
        win = legacy_gui.MainWindow()
    else:
        win = MainWindowPro()
        
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
