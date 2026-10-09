import sys
import os
import re
import json

if hasattr(sys, '_MEIPASS'):
    casadi_dir = os.path.join(sys._MEIPASS, 'casadi')
    if os.path.exists(casadi_dir):
        os.add_dll_directory(casadi_dir)

import traceback
import time

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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QTabWidget, QGroupBox, QGridLayout, QComboBox, QMessageBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QTextBrowser, QAction, QMenuBar, 
                             QScrollArea, QSplitter, QDialog, QFileDialog, QProgressBar, QSlider, QDoubleSpinBox, QSpinBox, QSizePolicy,
                             QListWidget, QListWidgetItem)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from src.mdo.mdo_global_optimizer import GlobalOptimizer
from src.mdo import mdo_target_optimization as mdo
from src.mdo.mdo_steady_state import evaluate_kpis_with_openmdao
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_skidpad import simulate_skidpad
from src.mdo.mdo_acceleration import simulate_acceleration
from src.mdo.mdo_dynamic_events import simulate_step_steer, simulate_double_lane_change, simulate_autox_endurance
from src.core.simulation_options import SimulationOptions, get_simulation_options, update_simulation_options
from src.utils.optimized_params_manager import save_optimization_run, load_history, get_last_run, export_history_to_csv
from src.models.tire_model_mf61 import get_equations_markdown as get_tire_eq_mf61
from src.models.steering_dynamics import calculate_steering_torque, generate_steering_torque_curve
from src.utils.suspension_sizing import calculate_suspension_and_arb_sizing, evaluate_4_ratios_balance
from src.models import damper_model_ttx25 as dm_ttx25
from src.utils.transient_sensitivity import (
    SCENARIO_CATALOG, 
    KPI_METADATA, 
    run_sensitivity_sweep_1d, 
    run_sensitivity_sweep_2d
)
from src.utils.vehicle_dna_maps import (
    compute_vehicle_dna_fields,
    plot_vehicle_dna_figures
)
from src.mdo.mdo_balance_range import (
    evaluate_optimal_balance_sweep,
    SWEEP_PARAMETER_CONFIGS
)
from src.mdo.mdo_arb_diamond_balance import (
    ARBDiamondConfig,
    convert_diamond_units,
    stiffness_from_coords,
    evaluate_setup_14dof,
    evaluate_diamond_vertices_and_current,
    evaluate_diamond_2d_mesh,
    evaluate_balance_ratio_1d_sweep
)

def _safe_is_checked(widget, default=False):
    """Verifica lo stato isChecked in modo sicuro proteggendosi da oggetti C++ eliminati o None."""
    if widget is None:
        return default
    try:
        from PyQt5 import sip
        if hasattr(sip, 'isdeleted') and sip.isdeleted(widget):
            return default
        return bool(widget.isChecked())
    except (RuntimeError, AttributeError):
        return default

# ---------------------------------------------------------------------------
# Modularized GUI Components (Canvas, Dialogs, Background Workers)
# Re-exported here for backwards compatibility with MainWindowPro and test suite
# ---------------------------------------------------------------------------
from src.ui.gui_canvas import MplCanvas
from src.ui.gui_dialogs import (
    ChartGuideDialog,
    PopoutPlotDialog,
    FIGURE_IT_TO_EN,
    translate_figure_language,
    CollapsibleGroupBox
)
from src.ui.gui_workers import (
    SensitivityWorker,
    ARBDiamondWorker,
    CollocationWorker,
    _worker_optimize_single_speed
)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("FSAE Global MDO Target Suite")
        self.setMinimumSize(850, 600)
        self.resize(1300, 850)
        
        # Menu Bar
        menu_bar = self.menuBar()
        tools_menu = menu_bar.addMenu("Tools")
        
        digitizer_action = QAction("Curve Digitizer", self)
        digitizer_action.triggered.connect(self.launch_digitizer)
        tools_menu.addAction(digitizer_action)

        import_curves_action = QAction(" Importa Cartella Curve Cinematiche (.json)...", self)
        import_curves_action.triggered.connect(self.import_digitized_curves_folder)
        tools_menu.addAction(import_curves_action)
        
        colloc_action = QAction(" Direct Collocation (Racing Line Optimization)", self)
        colloc_action.triggered.connect(self.run_direct_collocation_dialog)
        tools_menu.addAction(colloc_action)
        
        driver_action = QAction(" 14-DOF Gamepad Driving Simulator", self)
        driver_action.triggered.connect(self.launch_driving_simulator)
        tools_menu.addAction(driver_action)

        preview_3d_action = QAction(" Studio Anteprima 3D Veicolo (FSAE & SUV)", self)
        preview_3d_action.triggered.connect(self.launch_3d_preview_studio)
        tools_menu.addAction(preview_3d_action)
        
        options_menu = menu_bar.addMenu("Options")
        sim_options_action = QAction(" Simulation & Solver Settings", self)
        sim_options_action.triggered.connect(self.show_simulation_options_dialog)
        options_menu.addAction(sim_options_action)
        
        view_options_action = QAction(" Vehicle Physical Parameters", self)
        view_options_action.triggered.connect(self.show_options_dialog)
        options_menu.addAction(view_options_action)
        
        history_action = QAction(" Optimization History & Load Setup", self)
        history_action.triggered.connect(self.show_optimization_history_dialog)
        options_menu.addAction(history_action)

        self.lang_en_action = QAction(" English Mode (Report / Paper Charts)", self, checkable=True)
        self.lang_en_action.triggered.connect(self.on_language_toggled)
        options_menu.addAction(self.lang_en_action)

        # Help & Guides Menu
        help_menu = menu_bar.addMenu("Help & Guides")
        user_guide_action = QAction("Manuale Utente & Guida Dinamica del Veicolo", self)
        user_guide_action.triggered.connect(self.show_user_guide_dialog)
        help_menu.addAction(user_guide_action)

        # Interfaccia Menu
        view_menu = menu_bar.addMenu("Interfaccia")
        act_to_pro = QAction("Passa a Dynamic Target Suite Pro", self)
        act_to_pro.triggered.connect(self.switch_to_pro_gui)
        view_menu.addAction(act_to_pro)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        self.main_splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # --- Left Panel (Controls + Log) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        self.main_splitter.addWidget(left_panel)
        
        # Model Selection
        model_group = QGroupBox("Vehicle Model & Tires")
        model_layout = QVBoxLayout(model_group)

        # Language toggle for report/paper mode & Switch to Pro
        lang_bar = QHBoxLayout()
        self.lang_en_cb = QCheckBox("Grafici in Inglese (Report / Paper)")
        self.lang_en_cb.setToolTip("Abilita grafici in inglese formale accademico / SAE per report, paper scientifici e presentazioni.")
        self.lang_en_cb.stateChanged.connect(self.on_language_toggled)
        lang_bar.addWidget(self.lang_en_cb)
        
        lang_bar.addStretch()
        self.btn_switch_to_pro = QPushButton("Passa a Layout Pro")
        self.btn_switch_to_pro.setStyleSheet("font-weight: bold; font-size: 8.5pt; background-color: #0052cc; color: white; padding: 3px 10px; border-radius: 3px;")
        self.btn_switch_to_pro.setToolTip("Passa alla nuova interfaccia professionale Dynamic Target Suite Pro")
        self.btn_switch_to_pro.clicked.connect(self.switch_to_pro_gui)
        lang_bar.addWidget(self.btn_switch_to_pro)
        
        model_layout.addLayout(lang_bar)
        
        # Veicolo
        veh_layout = QHBoxLayout()
        veh_layout.addWidget(QLabel("Vehicle Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["3-DOF Steady-State MDO", "7-DOF Transient (Tier 2)", "14-DOF Full Vehicle (Tier 2)"])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        veh_layout.addWidget(self.model_combo)
        model_layout.addLayout(veh_layout)
        
        # Setup Profile Management Bar
        prof_layout = QHBoxLayout()
        prof_layout.addWidget(QLabel("Setup Profile:"))
        self.combo_setup_profile = QComboBox()
        prof_layout.addWidget(self.combo_setup_profile, stretch=1)
        
        self.btn_save_setup = QPushButton(" Salva")
        self.btn_save_setup.setToolTip("Salva le modifiche sul file di parametri attivo")
        self.btn_save_setup.setStyleSheet("font-size: 10px; font-weight: bold; background-color: #27ae60; color: white; padding: 3px 6px; border-radius: 3px;")
        self.btn_save_setup.clicked.connect(self.save_current_setup_profile)
        prof_layout.addWidget(self.btn_save_setup)
        
        self.btn_save_as_setup = QPushButton(" Salva con Nome...")
        self.btn_save_as_setup.setToolTip("Salva come nuovo profilo di parametri (.json)")
        self.btn_save_as_setup.setStyleSheet("font-size: 10px; font-weight: bold; background-color: #2980b9; color: white; padding: 3px 6px; border-radius: 3px;")
        self.btn_save_as_setup.clicked.connect(self.save_as_new_setup_profile)
        prof_layout.addWidget(self.btn_save_as_setup)
        
        self.btn_load_setup = QPushButton(" Sfoglia...")
        self.btn_load_setup.setToolTip("Carica un file setup .json personalizzato")
        self.btn_load_setup.setStyleSheet("font-size: 10px; padding: 3px 6px; border-radius: 3px;")
        self.btn_load_setup.clicked.connect(self.browse_load_setup_profile)
        prof_layout.addWidget(self.btn_load_setup)

        self.btn_import_curves = QPushButton(" Importa Curve...")
        self.btn_import_curves.setToolTip("Importa un intero set di curve cinematiche digitalizzate da una cartella JSON (es. 2025 o 2026)")
        self.btn_import_curves.setStyleSheet("font-size: 10px; font-weight: bold; background-color: #8e44ad; color: white; padding: 3px 6px; border-radius: 3px;")
        self.btn_import_curves.clicked.connect(self.import_digitized_curves_folder)
        prof_layout.addWidget(self.btn_import_curves)

        self.btn_inspect_active_params = QPushButton("📋 Parametri Attivi")
        self.btn_inspect_active_params.setToolTip("Ispeziona tutti i parametri attivi del veicolo e le 15 curve Adams Car")
        self.btn_inspect_active_params.setStyleSheet("font-size: 10px; font-weight: bold; background-color: #00cec9; color: black; padding: 3px 6px; border-radius: 3px;")
        self.btn_inspect_active_params.clicked.connect(self.show_active_parameters_dialog)
        prof_layout.addWidget(self.btn_inspect_active_params)

        model_layout.addLayout(prof_layout)
        
        # Pneumatici
        tir_layout = QHBoxLayout()
        tir_layout.addWidget(QLabel("Tire Model (.tir):"))
        self.tir_combo = QComboBox()
        self.populate_tir_files()
        tir_layout.addWidget(self.tir_combo)
        model_layout.addLayout(tir_layout)
        
        left_layout.addWidget(model_group)
        
        # Populate setup profiles
        self.populate_setup_profiles()
        self.combo_setup_profile.currentIndexChanged.connect(self.on_setup_profile_selected)
        
        # Active Vehicle Setup Banner
        self.setup_banner = QLabel(" Setup Attivo: et18_2026_kinematics.json (L=1.529m, tf=1.252m, tr=1.136m, wd_f=48.5%, LLTD=52.5%)")
        self.setup_banner.setStyleSheet("background-color: #2d3436; color: #00cec9; padding: 5px 8px; border-radius: 4px; font-weight: bold; font-size: 10.5px;")
        left_layout.addWidget(self.setup_banner)
        
        # Left Tabs
        self.left_tabs = QTabWidget()
        left_layout.addWidget(self.left_tabs)
        
        # -- Tab: Optimizer Params (Divided by Categories — Dynamic per Model)
        self.tab_params = QWidget()
        tab_params_layout = QVBoxLayout(self.tab_params)
        
        guide_bar = QHBoxLayout()
        self.btn_toggle_guide = QPushButton(" Mostra/Nascondi Model Guide")
        self.btn_toggle_guide.setStyleSheet("font-size: 11px; font-weight: bold; padding: 3px 8px; background-color: #f1f2f6; border: 1px solid #ced6e0; border-radius: 3px;")
        self.btn_toggle_guide.clicked.connect(self.toggle_guide_visibility)
        guide_bar.addWidget(self.btn_toggle_guide)
        
        guide_bar.addWidget(QLabel("<b>Vista Parametri:</b>"))
        self.combo_param_view = QComboBox()
        self.combo_param_view.addItems([" Parametri Modello Attivo", " Mostra Tutti i Parametri (ALL)"])
        self.combo_param_view.setStyleSheet("font-size: 10.5px; padding: 2px 6px; font-weight: bold; background-color: #dfe6e9; color: #2d3436;")
        self.combo_param_view.currentIndexChanged.connect(self._rebuild_params_panel)
        guide_bar.addWidget(self.combo_param_view)
        
        guide_bar.addStretch()
        tab_params_layout.addLayout(guide_bar)
        
        self.params_splitter = QSplitter(QtCore.Qt.Horizontal)
        tab_params_layout.addWidget(self.params_splitter)
        
        # Left: scrollable parameter table
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_splitter.addWidget(self.params_scroll)
        
        # Right: Guide Box
        self.guide_widget = QWidget()
        guide_layout = QVBoxLayout(self.guide_widget)
        guide_layout.setContentsMargins(0, 0, 0, 0)
        guide_layout.addWidget(QLabel("Model Guide:"))
        self.guide_browser = QTextBrowser()
        guide_layout.addWidget(self.guide_browser)
        self.params_splitter.addWidget(self.guide_widget)
        
        self.params_splitter.setSizes([600, 400])
        
        self.bounds_inputs = {}  # populated by _rebuild_params_panel
        self._rebuild_params_panel()
        self.left_tabs.addTab(self.tab_params, "Optimizer Params")

        # -- Tab: Dynamic Events
        self.tab_dynamic_scroll = QScrollArea()
        self.tab_dynamic_scroll.setWidgetResizable(True)
        self.tab_dynamic_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tab_dynamic_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tab_dynamic_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.tab_dynamic = QWidget()
        dynamic_layout = QVBoxLayout(self.tab_dynamic)
        dynamic_layout.setContentsMargins(4, 4, 4, 4)
        dynamic_layout.setSpacing(6)
        
        dyn_hdr = QHBoxLayout()
        self.dyn_setup_banner = QLabel(" Setup Dinamico: Sincronizzato con Setup Attivo")
        self.dyn_setup_banner.setStyleSheet("background-color: #dfe6e9; color: #2d3436; padding: 4px 8px; border-radius: 3px; font-weight: bold; font-size: 10px;")
        btn_hist_dyn = QPushButton(" Storico Setup")
        btn_hist_dyn.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        btn_hist_dyn.clicked.connect(self.show_optimization_history_dialog)
        dyn_hdr.addWidget(self.dyn_setup_banner)
        dyn_hdr.addWidget(btn_hist_dyn)
        dynamic_layout.addLayout(dyn_hdr)
        
        # Skidpad settings
        skidpad_group = CollapsibleGroupBox("Skidpad Event", collapsed=False)
        skid_layout = QGridLayout(skidpad_group.content_widget)
        skid_layout.setContentsMargins(4, 4, 4, 4)
        skid_layout.setHorizontalSpacing(6)
        skid_layout.setVerticalSpacing(4)
        skid_layout.addWidget(QLabel("Radius [m]:"), 0, 0)
        self.skid_radius_input = QLineEdit("9.125")
        skid_layout.addWidget(self.skid_radius_input, 0, 1)
        self.btn_run_skidpad = QPushButton("Run Skidpad")
        self.btn_run_skidpad.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white;")
        self.btn_run_skidpad.clicked.connect(self.run_skidpad_event)
        skid_layout.addWidget(self.btn_run_skidpad, 0, 2)
        dynamic_layout.addWidget(skidpad_group)
        
        # Acceleration settings
        accel_group = CollapsibleGroupBox("Acceleration Event", collapsed=False)
        accel_layout = QGridLayout(accel_group.content_widget)
        accel_layout.setContentsMargins(4, 4, 4, 4)
        accel_layout.setHorizontalSpacing(6)
        accel_layout.setVerticalSpacing(4)
        accel_layout.addWidget(QLabel("Distance [m]:"), 0, 0)
        self.accel_dist_input = QLineEdit("75.0")
        accel_layout.addWidget(self.accel_dist_input, 0, 1)
        self.btn_run_accel = QPushButton("Run Acceleration")
        self.btn_run_accel.setStyleSheet("font-weight: bold; background-color: #e67e22; color: white;")
        self.btn_run_accel.clicked.connect(self.run_accel_event)
        accel_layout.addWidget(self.btn_run_accel, 0, 2)
        dynamic_layout.addWidget(accel_group)
        
        # Step Steer & Double Lane Change Settings
        trans_group = CollapsibleGroupBox("Transient Maneuver (Step Steer / Double Lane Change)", collapsed=False)
        trans_layout = QGridLayout(trans_group.content_widget)
        trans_layout.setContentsMargins(4, 4, 4, 4)
        trans_layout.setHorizontalSpacing(6)
        trans_layout.setVerticalSpacing(4)
        
        trans_layout.addWidget(QLabel("Speed Vx [m/s]:"), 0, 0)
        self.step_vx_input = QLineEdit("15.0")
        trans_layout.addWidget(self.step_vx_input, 0, 1)
        trans_layout.addWidget(QLabel("Steer δ [°]:"), 0, 2)
        self.step_steer_input = QLineEdit("3.0")
        trans_layout.addWidget(self.step_steer_input, 0, 3)
        
        trans_layout.addWidget(QLabel("Step Start t_step [s]:"), 1, 0)
        self.step_tstep_input = QLineEdit("0.50")
        trans_layout.addWidget(self.step_tstep_input, 1, 1)
        trans_layout.addWidget(QLabel("Steer Ramp t_ramp [s]:"), 1, 2)
        self.step_tramp_input = QLineEdit("0.08")
        trans_layout.addWidget(self.step_tramp_input, 1, 3)
        
        trans_layout.addWidget(QLabel("Duration t_end [s]:"), 2, 0)
        self.step_tend_input = QLineEdit("4.0")
        trans_layout.addWidget(self.step_tend_input, 2, 1)
        trans_layout.addWidget(QLabel("ODE Time Step dt [s]:"), 2, 2)
        self.step_dt_input = QLineEdit("0.002")
        trans_layout.addWidget(self.step_dt_input, 2, 3)
        
        # Torque Vectoring with explanation button
        tv_box = QWidget()
        tv_layout = QHBoxLayout(tv_box)
        tv_layout.setContentsMargins(0, 0, 0, 0)
        tv_layout.addWidget(QLabel("Torque Vectoring ΔT [Nm]:"))
        self.btn_tv_help = QPushButton("ℹ Cos'è?")
        self.btn_tv_help.setStyleSheet("font-size: 10px; padding: 1px 4px; background-color: #dfe6e9; border: 1px solid #b2bec3; border-radius: 3px;")
        self.btn_tv_help.clicked.connect(self.show_tv_guide_dialog)
        tv_layout.addWidget(self.btn_tv_help)
        trans_layout.addWidget(tv_box, 3, 0)
        self.step_tv_input = QLineEdit("0.0")
        trans_layout.addWidget(self.step_tv_input, 3, 1)
        
        self.step_maintain_speed_cb = QCheckBox("Mantieni Velocità Vx (Cruise Throttle)")
        self.step_maintain_speed_cb.setChecked(True)
        self.step_maintain_speed_cb.setToolTip("Se attivo, applica coppia motrice controllata per contrastare drag aerodinamico e resistenza pneumatici mantenendo Vx nominale.")
        trans_layout.addWidget(self.step_maintain_speed_cb, 3, 2, 1, 2)
        
        self.btn_run_step = QPushButton(" Run Step Steer (7/14 DOF)")
        self.btn_run_step.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white;")
        self.btn_run_step.clicked.connect(self.run_step_steer_event)
        trans_layout.addWidget(self.btn_run_step, 4, 0, 1, 2)
        
        self.btn_run_dlc = QPushButton(" Run Double Lane Change")
        self.btn_run_dlc.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white;")
        self.btn_run_dlc.clicked.connect(self.run_dlc_event)
        trans_layout.addWidget(self.btn_run_dlc, 4, 2, 1, 2)
        dynamic_layout.addWidget(trans_group)
        
        # AutoX & Endurance Simulation Settings
        autox_group = CollapsibleGroupBox("FSAE AutoX & Endurance (Lap Time & Consumi)", collapsed=True)
        autox_layout = QGridLayout(autox_group.content_widget)
        autox_layout.setContentsMargins(4, 4, 4, 4)
        autox_layout.setHorizontalSpacing(6)
        autox_layout.setVerticalSpacing(4)
        autox_layout.addWidget(QLabel("Evento:"), 0, 0)
        self.autox_event_combo = QComboBox()
        self.autox_event_combo.addItems(["AutoX (1 Lap Sprint)", "Endurance (22 Laps / 22 km)"])
        autox_layout.addWidget(self.autox_event_combo, 0, 1)
        autox_layout.addWidget(QLabel("Track Length [m]:"), 0, 2)
        self.autox_len_input = QLineEdit("850.0")
        autox_layout.addWidget(self.autox_len_input, 0, 3)
        
        autox_layout.addWidget(QLabel("Passo Discretizzazione Spaziale ds [m]:"), 1, 0)
        self.autox_ds_input = QLineEdit("0.50")
        autox_layout.addWidget(self.autox_ds_input, 1, 1)
        
        # Track Layout Selector for AutoX & Endurance
        autox_layout.addWidget(QLabel("Layout Tracciato:"), 2, 0)
        self.combo_autox_track = QComboBox()
        self.combo_autox_track.addItems([
            " FSAE AutoX Circuit (850m)",
            " FSAE Skidpad (Figure-8)",
            " FSAE Endurance (1250m)",
            " FSAE Acceleration (200m)",
            " Carica File Tracciato (.csv/.json)..."
        ])
        self.combo_autox_track.currentIndexChanged.connect(self.on_autox_track_changed)
        autox_layout.addWidget(self.combo_autox_track, 2, 1, 1, 3)

        self.btn_run_autox = QPushButton(" Run AutoX / Endurance (Quasi-Static Instant)")
        self.btn_run_autox.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 5px;")
        self.btn_run_autox.clicked.connect(self.run_autox_endurance_event)
        autox_layout.addWidget(self.btn_run_autox, 3, 0, 1, 2)
        
        self.btn_run_colloc = QPushButton(" Trajectory Direct Collocation")
        self.btn_run_colloc.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 5px;")
        self.btn_run_colloc.clicked.connect(self.run_direct_collocation_dialog)
        autox_layout.addWidget(self.btn_run_colloc, 3, 2, 1, 2)
        
        self.btn_launch_driver = QPushButton(" Avvia Simulatore di Guida 14-DOF (Gamepad / Wheel / Keyboard)")
        self.btn_launch_driver.setStyleSheet("font-weight: bold; background-color: #e17055; color: white; padding: 6px; font-size: 11px;")
        self.btn_launch_driver.clicked.connect(self.launch_driving_simulator)
        autox_layout.addWidget(self.btn_launch_driver, 4, 0, 1, 4)
        
        dynamic_layout.addWidget(autox_group)
        
        # Manual Vehicle Setup Group for Dynamic Tests
        dyn_setup_group = CollapsibleGroupBox("Setup Veicolo per Prove Dinamiche (Parametri Manuali)", collapsed=True)
        dyn_setup_layout = QGridLayout(dyn_setup_group.content_widget)
        dyn_setup_layout.setContentsMargins(4, 4, 4, 4)
        dyn_setup_layout.setHorizontalSpacing(6)
        dyn_setup_layout.setVerticalSpacing(4)
        
        dyn_setup_layout.addWidget(QLabel("Massa m [kg]:"), 0, 0)
        self.dyn_m_input = QLineEdit("314.8")
        dyn_setup_layout.addWidget(self.dyn_m_input, 0, 1)
        
        dyn_setup_layout.addWidget(QLabel("Potenza P_max [kW]:"), 0, 2)
        self.dyn_pmax_input = QLineEdit("80.0")
        dyn_setup_layout.addWidget(self.dyn_pmax_input, 0, 3)
        
        dyn_setup_layout.addWidget(QLabel("Coppia Motore T_max [Nm]:"), 1, 0)
        self.dyn_tmax_input = QLineEdit("120.0")
        dyn_setup_layout.addWidget(self.dyn_tmax_input, 1, 1)
        
        dyn_setup_layout.addWidget(QLabel("Rapporto Trasmissione i_g:"), 1, 2)
        self.dyn_fd_input = QLineEdit("3.50")
        dyn_setup_layout.addWidget(self.dyn_fd_input, 1, 3)
        
        dyn_setup_layout.addWidget(QLabel("Passo L [m]:"), 2, 0)
        self.dyn_L_input = QLineEdit("1.530")
        dyn_setup_layout.addWidget(self.dyn_L_input, 2, 1)
        
        dyn_setup_layout.addWidget(QLabel("Track tf [m]:"), 2, 2)
        self.dyn_tf_input = QLineEdit("1.250")
        dyn_setup_layout.addWidget(self.dyn_tf_input, 2, 3)
        
        dyn_setup_layout.addWidget(QLabel("Track tr [m]:"), 3, 0)
        self.dyn_tr_input = QLineEdit("1.150")
        dyn_setup_layout.addWidget(self.dyn_tr_input, 3, 1)
        
        dyn_setup_layout.addWidget(QLabel("Ripartizione wd_f:"), 3, 2)
        self.dyn_wdf_input = QLineEdit("0.470")
        dyn_setup_layout.addWidget(self.dyn_wdf_input, 3, 3)
        
        dyn_setup_layout.addWidget(QLabel("LLTD (Ant.):"), 4, 0)
        self.dyn_lltd_input = QLineEdit("0.550")
        dyn_setup_layout.addWidget(self.dyn_lltd_input, 4, 1)
        
        dyn_setup_layout.addWidget(QLabel("Aero Balance:"), 4, 2)
        self.dyn_aero_input = QLineEdit("0.460")
        dyn_setup_layout.addWidget(self.dyn_aero_input, 4, 3)
        
        dyn_setup_layout.addWidget(QLabel("Grip LMUX / LMUY:"), 5, 0)
        grip_w = QWidget()
        grip_l = QHBoxLayout(grip_w)
        grip_l.setContentsMargins(0, 0, 0, 0)
        self.dyn_lmux_input = QLineEdit("1.00")
        self.dyn_lmuy_input = QLineEdit("1.00")
        grip_l.addWidget(self.dyn_lmux_input)
        grip_l.addWidget(self.dyn_lmuy_input)
        dyn_setup_layout.addWidget(grip_w, 5, 1)
        
        btn_bar = QHBoxLayout()
        self.btn_sync_dyn = QPushButton(" Copia da Optimizer")
        self.btn_sync_dyn.clicked.connect(self.sync_dyn_params_from_optimizer)
        self.btn_reset_dyn = QPushButton("↺ Ripristina Default")
        self.btn_reset_dyn.clicked.connect(self.reset_dyn_params_default)
        btn_bar.addWidget(self.btn_sync_dyn)
        btn_bar.addWidget(self.btn_reset_dyn)
        dyn_setup_layout.addLayout(btn_bar, 6, 0, 1, 4)
        
        dynamic_layout.addWidget(dyn_setup_group)
        
        # MDO Objective setting
        obj_group = CollapsibleGroupBox("MDO Global Objective", collapsed=False)
        obj_layout = QVBoxLayout(obj_group.content_widget)
        obj_layout.setContentsMargins(4, 4, 4, 4)
        self.obj_combo = QComboBox()
        self.obj_combo.addItems([
            "Max Lateral G (Steady-State)",
            "Assetto Neutro Bilanciato (Max Ay + Target Kus)",
            "Min Skidpad Time",
            "Min Acceleration Time"
        ])
        obj_layout.addWidget(self.obj_combo)
        dynamic_layout.addWidget(obj_group)
        
        dynamic_layout.addStretch()
        self.tab_dynamic_scroll.setWidget(self.tab_dynamic)
        self.left_tabs.addTab(self.tab_dynamic_scroll, "Dynamic Events")
        
        
        # -- Tab: Sweep & Settings
        self.tab_sweep_scroll = QScrollArea()
        self.tab_sweep_scroll.setWidgetResizable(True)
        self.tab_sweep_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tab_sweep_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tab_sweep_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.tab_sweep = QWidget()
        self.tab_sweep_layout = QVBoxLayout(self.tab_sweep)
        
        sweep_group = QGroupBox("Speed Sweep Settings")
        sweep_layout = QGridLayout(sweep_group)
        
        self.enable_sweep = QCheckBox("Enable Speed Sweep")
        sweep_layout.addWidget(self.enable_sweep, 0, 0, 1, 2)
        
        sweep_layout.addWidget(QLabel("V Min (m/s):"), 1, 0)
        self.v_min_input = QLineEdit("5.0")
        sweep_layout.addWidget(self.v_min_input, 1, 1)
        
        sweep_layout.addWidget(QLabel("V Max (m/s):"), 2, 0)
        self.v_max_input = QLineEdit("25.0")
        sweep_layout.addWidget(self.v_max_input, 2, 1)
        
        sweep_layout.addWidget(QLabel("V Steps:"), 3, 0)
        self.v_steps_input = QLineEdit("5")
        sweep_layout.addWidget(self.v_steps_input, 3, 1)
        
        self.tab_sweep_layout.addWidget(sweep_group)
        

        diag_group = QGroupBox("Diagram Settings")
        diag_layout = QGridLayout(diag_group)
        
        diag_layout.addWidget(QLabel("HD Speeds (m/s):"), 0, 0)
        self.hd_speeds_input = QLineEdit("8.0, 12.0, 16.0, 20.0, 25.0")
        diag_layout.addWidget(self.hd_speeds_input, 0, 1)
        
        diag_layout.addWidget(QLabel("HD Points:"), 1, 0)
        self.hd_points_input = QLineEdit("120")
        diag_layout.addWidget(self.hd_points_input, 1, 1)
        
        diag_layout.addWidget(QLabel("HD Method:"), 2, 0)
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        hd_method_widget = QWidget()
        hd_method_layout = QHBoxLayout(hd_method_widget)
        hd_method_layout.setContentsMargins(0, 0, 0, 0)
        self.rb_arclength = QRadioButton("Arc-Length")
        self.rb_peak = QRadioButton("Stop Picco")
        self.rb_classic = QRadioButton("Classic")
        self.rb_arclength.setChecked(True)
        self.hd_method_group = QButtonGroup()
        self.hd_method_group.addButton(self.rb_arclength, 0)
        self.hd_method_group.addButton(self.rb_peak, 1)
        self.hd_method_group.addButton(self.rb_classic, 2)
        hd_method_layout.addWidget(self.rb_arclength)
        hd_method_layout.addWidget(self.rb_peak)
        hd_method_layout.addWidget(self.rb_classic)
        diag_layout.addWidget(hd_method_widget, 2, 1)
        
        self.show_stability_frontier_cb = QCheckBox("Frontiera di Stabilità (Picchi)")
        self.show_stability_frontier_cb.setChecked(True)
        self.show_stability_frontier_cb.setToolTip("Congiunge i punti di picco dell'Handling Diagram a diverse velocità, tracciando il confine di stabilità stazionaria.")
        diag_layout.addWidget(self.show_stability_frontier_cb, 3, 0, 1, 2)
        
        self.btn_plot_handling = QPushButton("Plot Handling")
        self.btn_plot_handling.clicked.connect(self.run_handling_standalone)
        diag_layout.addWidget(self.btn_plot_handling, 4, 0)
        
        self.btn_plot_maps = QPushButton(" Plot Steady-State MAPs")
        self.btn_plot_maps.clicked.connect(self.run_maps_standalone)
        diag_layout.addWidget(self.btn_plot_maps, 4, 1)
        
        diag_layout.addWidget(QLabel("MMM Deltas (deg):"), 5, 0)
        self.mmm_deltas_input = QLineEdit("-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12")
        diag_layout.addWidget(self.mmm_deltas_input, 5, 1)

        diag_layout.addWidget(QLabel("MMM Delta Points:"), 6, 0)
        self.mmm_dpoints_input = QLineEdit("13")
        diag_layout.addWidget(self.mmm_dpoints_input, 6, 1)

        diag_layout.addWidget(QLabel("MMM Beta Range:"), 7, 0)
        self.mmm_beta_input = QLineEdit("-10, 10")
        diag_layout.addWidget(self.mmm_beta_input, 7, 1)
        
        diag_layout.addWidget(QLabel("MMM Beta Points:"), 8, 0)
        self.mmm_bpoints_input = QLineEdit("35")
        diag_layout.addWidget(self.mmm_bpoints_input, 8, 1)

        self.tab_sweep_layout.addWidget(diag_group)

        single_group = QGroupBox("Single Run Settings")
        single_layout = QGridLayout(single_group)
        single_layout.addWidget(QLabel("Target Speed (m/s):"), 0, 0)
        self.v_x_input = QLineEdit("15.0")
        single_layout.addWidget(self.v_x_input, 0, 1)
        
        single_layout.addWidget(QLabel("Track Penalty:"), 1, 0)
        self.penalty_input = QLineEdit("0.0")
        single_layout.addWidget(self.penalty_input, 1, 1)
        
        single_layout.addWidget(QLabel("Tire Uncertainty (± %):"), 2, 0)
        self.tire_unc_input = QLineEdit("10.0")
        single_layout.addWidget(self.tire_unc_input, 2, 1)
        
        self.robust_opt_cb = QCheckBox("Ottimizzazione Robusta all'Incertezza")
        self.robust_opt_cb.setChecked(True)
        single_layout.addWidget(self.robust_opt_cb, 3, 0, 1, 2)
        
        self.auto_maps_cb = QCheckBox("Auto-generate MAPs (Slower)")
        self.auto_maps_cb.setChecked(False)
        single_layout.addWidget(self.auto_maps_cb, 4, 0, 1, 2)
        
        self.enforce_understeer_cb = QCheckBox("Forza Sottosterzo Minimo:")
        self.enforce_understeer_cb.setChecked(True)
        single_layout.addWidget(self.enforce_understeer_cb, 5, 0)

        us_box = QWidget()
        us_box_l = QHBoxLayout(us_box)
        us_box_l.setContentsMargins(0, 0, 0, 0)
        us_box_l.addWidget(QLabel("K_us min:"))
        self.kus_min_input = QLineEdit("0.10")
        self.kus_min_input.setMaximumWidth(40)
        self.kus_min_input.setToolTip("Gradiente minimo di sottosterzo richiesto [deg/g] (default: +0.10 deg/g).")
        us_box_l.addWidget(self.kus_min_input)

        us_box_l.addWidget(QLabel("Peso:"))
        self.kus_weight_input = QLineEdit("1.0")
        self.kus_weight_input.setMaximumWidth(35)
        self.kus_weight_input.setToolTip("Peso / severità della penalità nel solutore (default: 1.0).")
        us_box_l.addWidget(self.kus_weight_input)
        single_layout.addWidget(us_box, 5, 1)
        
        single_layout.addWidget(QLabel("Target K_us (deg/g):"), 6, 0)
        self.kus_target_input = QLineEdit("0.00")
        self.kus_target_input.setMaximumWidth(55)
        self.kus_target_input.setToolTip("Target di gradiente di sottosterzo per la modalità 'Assetto Neutro Bilanciato' [deg/g] (default: 0.00 deg/g per neutro puro).")
        single_layout.addWidget(self.kus_target_input, 6, 1)
        
        self.tab_sweep_layout.addWidget(single_group)
        self.tab_sweep_layout.addStretch()
        self.tab_sweep_scroll.setWidget(self.tab_sweep)
        self.left_tabs.addTab(self.tab_sweep_scroll, "Sweep & Settings")
        
        self.run_btn = QPushButton("Run Global Optimization")
        self.run_btn.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #2e7d32; color: white; padding: 10px;")
        self.run_btn.clicked.connect(self.run_optimization)
        left_layout.addWidget(self.run_btn)
        
        self.btn_run_constant = QPushButton(" Valuta Parametri Costanti (No Ottimizzazione)")
        self.btn_run_constant.setToolTip("Genera ed aggiorna tutti i grafici (YMD, Handling, G-G-V, MAPs, KPI) sui parametri correnti senza eseguire l'ottimizzazione.")
        self.btn_run_constant.setStyleSheet("font-size: 12.5px; font-weight: bold; background-color: #0984e3; color: white; padding: 8px; margin-top: 2px; border-radius: 4px;")
        self.btn_run_constant.clicked.connect(self.run_constant_parameters)
        left_layout.addWidget(self.btn_run_constant)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        left_layout.addWidget(QLabel("Log:"))
        left_layout.addWidget(self.log_output)
        
        # --- Right Panel (Plots in Tabs) ---
        right_panel_widget = QWidget()
        right_panel = QVBoxLayout(right_panel_widget)
        right_panel.setContentsMargins(2, 2, 2, 2)
        self.main_splitter.addWidget(right_panel_widget)
        self.main_splitter.setSizes([460, 740])
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ced6e0;
                background-color: #ffffff;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #f1f2f6;
                color: #2f3542;
                padding: 7px 14px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #ced6e0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0984e3;
                border-bottom: 2px solid #0984e3;
            }
            QTabBar::tab:hover {
                background: #dfe4ea;
                color: #000000;
            }
        """)
        right_panel.addWidget(self.tabs)
        
        # YMD Tab with Sub-Tabs
        self.tab_ymd = QWidget()
        self.tab_ymd_layout = QVBoxLayout(self.tab_ymd)
        
        ymd_top_bar = QHBoxLayout()
        self.ymd_top_bar = ymd_top_bar
        self.btn_info_ymd = QPushButton("ℹ Guida & Interpretazione (YMD / MMM)")
        self.btn_info_ymd.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_ymd.clicked.connect(lambda: self.show_chart_guide("YMD"))
        ymd_top_bar.addWidget(self.btn_info_ymd)
        
        ymd_top_bar.addWidget(QLabel("<b>Velocità vx [m/s]:</b>"))
        self.ymd_vx_input = QLineEdit("15.0")
        self.ymd_vx_input.setMaximumWidth(45)
        ymd_top_bar.addWidget(self.ymd_vx_input)
        
        self.ymd_vx_slider = QSlider(QtCore.Qt.Horizontal)
        self.ymd_vx_slider.setRange(5, 35)
        self.ymd_vx_slider.setValue(15)
        self.ymd_vx_slider.setFixedWidth(110)
        self.ymd_vx_slider.valueChanged.connect(lambda val: self.ymd_vx_input.setText(str(float(val))))
        self.ymd_vx_slider.sliderReleased.connect(self.run_mmm_standalone)
        ymd_top_bar.addWidget(self.ymd_vx_slider)
        
        self.ymd_show_radii_cb = QCheckBox("Mostra Raggi di Curva (Iso-R)")
        self.ymd_show_radii_cb.setChecked(True)
        self.ymd_show_radii_cb.toggled.connect(self.run_mmm_standalone)
        ymd_top_bar.addWidget(self.ymd_show_radii_cb)

        ymd_top_bar.addWidget(QLabel("Speeds Multi [m/s]:"))
        self.ymd_multispeeds_input = QLineEdit("8.0, 12.0, 16.0, 20.0, 25.0")
        self.ymd_multispeeds_input.setMaximumWidth(130)
        ymd_top_bar.addWidget(self.ymd_multispeeds_input)
        
        self.btn_recalc_ymd = QPushButton(" Ricalcola YMD")
        self.btn_recalc_ymd.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_recalc_ymd.clicked.connect(self.run_mmm_standalone)
        ymd_top_bar.addWidget(self.btn_recalc_ymd)
        
        self.btn_calc_multispeed_ymd = QPushButton(" Genera Multi-Speed Overlay")
        self.btn_calc_multispeed_ymd.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #e17055; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_multispeed_ymd.clicked.connect(self.plot_mmm_multispeed)
        ymd_top_bar.addWidget(self.btn_calc_multispeed_ymd)
        
        self.btn_info_ymd = QPushButton("ℹ Guida & Interpretazione (YMD & Stabilità)")
        self.btn_info_ymd.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_ymd.clicked.connect(lambda: self.show_chart_guide("YMD"))
        ymd_top_bar.addWidget(self.btn_info_ymd)
        
        self.canvas_ymd = MplCanvas(self, width=6, height=5, dpi=100)
        self.setup_popout(self.canvas_ymd, "Yaw Moment Diagram (YMD / Moment Method)", ymd_top_bar)
        ymd_top_bar.addStretch()
        self.tab_ymd_layout.addLayout(ymd_top_bar)
        
        self.ymd_subtabs = QTabWidget()
        self.tab_ymd_layout.addWidget(self.ymd_subtabs)
        
        # Sub-Tab 1: Single Speed YMD
        self.tab_ymd_single = QWidget()
        tab_ymd_single_layout = QVBoxLayout(self.tab_ymd_single)
        self.toolbar_ymd = NavigationToolbar(self.canvas_ymd, self.tab_ymd_single)
        tab_ymd_single_layout.addWidget(self.toolbar_ymd)
        tab_ymd_single_layout.addWidget(self.canvas_ymd)
        self.ymd_subtabs.addTab(self.tab_ymd_single, " Yaw Moment Diagram (Iso-Raggi & Griglia δ × β)")
        
        # Sub-Tab 2: Multi-Speed Overlay
        self.tab_ymd_multi = QWidget()
        tab_ymd_multi_layout = QVBoxLayout(self.tab_ymd_multi)
        self.fig_ymd_multi = Figure(figsize=(10, 6), dpi=100)
        self.canvas_ymd_multi = FigureCanvas(self.fig_ymd_multi)
        self.setup_popout(self.canvas_ymd_multi, "Multi-Speed YMD Overlay & Stabilità vs Velocità", None)
        self.toolbar_ymd_multi = NavigationToolbar(self.canvas_ymd_multi, self.tab_ymd_multi)
        tab_ymd_multi_layout.addWidget(self.toolbar_ymd_multi)
        tab_ymd_multi_layout.addWidget(self.canvas_ymd_multi)
        self.ymd_subtabs.addTab(self.tab_ymd_multi, " Multi-Speed YMD Overlay & Derivate di Stabilità (v)")
        
        self.tabs.addTab(self.tab_ymd, "Yaw Moment Diagram")
        
        # GGV Tab
        self.tab_ggv = QWidget()
        self.tab_ggv_layout = QVBoxLayout(self.tab_ggv)
        
        ggv_top_bar = QHBoxLayout()
        self.btn_info_ggv = QPushButton("ℹ Guida & Interpretazione (G-G-V 3D)")
        self.btn_info_ggv.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_ggv.clicked.connect(lambda: self.show_chart_guide("GGV"))
        ggv_top_bar.addWidget(self.btn_info_ggv)
        
        ggv_top_bar.addWidget(QLabel("V min [m/s]:"))
        self.ggv_vmin_input = QLineEdit("8.0")
        self.ggv_vmin_input.setFixedWidth(45)
        ggv_top_bar.addWidget(self.ggv_vmin_input)
        
        ggv_top_bar.addWidget(QLabel("V max [m/s]:"))
        self.ggv_vmax_input = QLineEdit("28.0")
        self.ggv_vmax_input.setFixedWidth(45)
        ggv_top_bar.addWidget(self.ggv_vmax_input)
        
        ggv_top_bar.addWidget(QLabel("N° Vel.:"))
        self.ggv_nslices_input = QLineEdit("6")
        self.ggv_nslices_input.setFixedWidth(35)
        ggv_top_bar.addWidget(self.ggv_nslices_input)
        
        self.btn_run_ggv = QPushButton(" Rigenera G-G-V 3D")
        self.btn_run_ggv.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_run_ggv.clicked.connect(self.run_ggv_standalone)
        ggv_top_bar.addWidget(self.btn_run_ggv)
        
        self.fig_ggv = Figure(figsize=(6, 5), dpi=100)
        self.axes_ggv = self.fig_ggv.add_subplot(111, projection='3d')
        self.canvas_ggv = FigureCanvas(self.fig_ggv)
        self.setup_popout(self.canvas_ggv, "G-G-V 3D Performance Bowl", ggv_top_bar)
        ggv_top_bar.addStretch()
        self.tab_ggv_layout.addLayout(ggv_top_bar)
        
        self.toolbar_ggv = NavigationToolbar(self.canvas_ggv, self.tab_ggv)
        self.tab_ggv_layout.addWidget(self.toolbar_ggv)
        self.tab_ggv_layout.addWidget(self.canvas_ggv)
        self.tabs.addTab(self.tab_ggv, "G-G-V Diagram")
        
        # Handling Diagram Tab with Sub-Tabs
        self.tab_handling = QWidget()
        self.tab_handling_layout = QVBoxLayout(self.tab_handling)
        
        hd_top_bar = QHBoxLayout()
        self.btn_info_handling = QPushButton("ℹ Guida & Interpretazione (Handling Diagram & K_us)")
        self.btn_info_handling.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_handling.clicked.connect(lambda: self.show_chart_guide("HANDLING"))
        hd_top_bar.addWidget(self.btn_info_handling)
        
        self.canvas_handling = MplCanvas(self, width=6, height=5, dpi=100)
        self.setup_popout(self.canvas_handling, "Handling Diagram (ay/g vs L·ρ - δ)", hd_top_bar)
        hd_top_bar.addStretch()
        self.tab_handling_layout.addLayout(hd_top_bar)
        
        self.handling_subtabs = QTabWidget()
        self.tab_handling_layout.addWidget(self.handling_subtabs)
        
        # Sub-Tab 1: Handling Diagram (Cartesian)
        self.tab_hd_diagram = QWidget()
        tab_hd_diagram_layout = QVBoxLayout(self.tab_hd_diagram)
        self.toolbar_handling = NavigationToolbar(self.canvas_handling, self.tab_hd_diagram)
        tab_hd_diagram_layout.addWidget(self.toolbar_handling)
        tab_hd_diagram_layout.addWidget(self.canvas_handling)
        self.handling_subtabs.addTab(self.tab_hd_diagram, " Handling Diagram")
        
        # Sub-Tab 2: Kus Variation Analysis
        self.tab_kus_analysis = QWidget()
        tab_kus_layout = QVBoxLayout(self.tab_kus_analysis)
        self.fig_kus = Figure(figsize=(10, 8), dpi=100)
        self.axes_kus_ay = self.fig_kus.add_subplot(211)
        self.axes_kus_v  = self.fig_kus.add_subplot(212)
        self.canvas_kus = FigureCanvas(self.fig_kus)
        self.setup_popout(self.canvas_kus, "Gradiente di Sottosterzo K_us vs ay", None)
        self.toolbar_kus = NavigationToolbar(self.canvas_kus, self.tab_kus_analysis)
        tab_kus_layout.addWidget(self.toolbar_kus)
        tab_kus_layout.addWidget(self.canvas_kus)
        self.handling_subtabs.addTab(self.tab_kus_analysis, " Analisi Variazione Gradiente K_us")
        
        self.tabs.addTab(self.tab_handling, "Handling & K_us")
        
        # MAPs Tab with sub-tabs for Input Regions & Output/Mixed Regions & Options
        self.tab_maps = QWidget()
        self.tab_maps_layout = QVBoxLayout(self.tab_maps)
        
        maps_top_bar = QHBoxLayout()
        self.btn_gen_maps = QPushButton(" Genera Suite Completa MAPs (5 Regioni)")
        self.btn_gen_maps.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #0984e3; color: white; padding: 6px 12px; border-radius: 4px;")
        self.btn_gen_maps.clicked.connect(self.run_maps_standalone)
        maps_top_bar.addWidget(self.btn_gen_maps)
        
        self.btn_info_maps = QPushButton("ℹ Guida & Interpretazione (MAPs)")
        self.btn_info_maps.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 6px 12px; border-radius: 4px;")
        self.btn_info_maps.clicked.connect(lambda: self.show_chart_guide("MAPS"))
        maps_top_bar.addWidget(self.btn_info_maps)
        
        self.fig_maps_input = Figure(figsize=(12, 5), dpi=100)
        self.axes_map_delta_u  = self.fig_maps_input.add_subplot(131)
        self.axes_map_u_ay     = self.fig_maps_input.add_subplot(132)
        self.axes_map_delta_ay = self.fig_maps_input.add_subplot(133)
        self.canvas_maps_input = FigureCanvas(self.fig_maps_input)
        self.setup_popout(self.canvas_maps_input, "MAPs - Regioni di Ingresso (δ, u, ay)", maps_top_bar)
        
        self.fig_maps_output = Figure(figsize=(10, 5), dpi=100)
        self.axes_map_rho_beta  = self.fig_maps_output.add_subplot(121)
        self.axes_map_delta_rho = self.fig_maps_output.add_subplot(122)
        self.canvas_maps_output = FigureCanvas(self.fig_maps_output)
        self.setup_popout(self.canvas_maps_output, "MAPs - Regioni di Uscita e Miste (ρ, β, δ)", maps_top_bar)

        maps_top_bar.addStretch()
        self.tab_maps_layout.addLayout(maps_top_bar)
        
        self.maps_subtabs = QTabWidget()
        self.tab_maps_layout.addWidget(self.maps_subtabs)
        
        # Sub-Tab 1: Input Achievable Regions (3 subplots: (delta, u), (u, ay), (delta, ay))
        self.tab_maps_input = QWidget()
        self.tab_maps_input_layout = QVBoxLayout(self.tab_maps_input)
        self.toolbar_maps_input = NavigationToolbar(self.canvas_maps_input, self.tab_maps_input)
        self.tab_maps_input_layout.addWidget(self.toolbar_maps_input)
        self.tab_maps_input_layout.addWidget(self.canvas_maps_input)
        self.maps_subtabs.addTab(self.tab_maps_input, "Input Achievable Regions (δ, u, ay)")
        
        # Sub-Tab 2: Output & Mixed Regions (2 subplots: (rho, beta) and (delta, rho))
        self.tab_maps_output = QWidget()
        self.tab_maps_output_layout = QVBoxLayout(self.tab_maps_output)
        self.toolbar_maps_output = NavigationToolbar(self.canvas_maps_output, self.tab_maps_output)
        self.tab_maps_output_layout.addWidget(self.toolbar_maps_output)
        self.tab_maps_output_layout.addWidget(self.canvas_maps_output)
        self.maps_subtabs.addTab(self.tab_maps_output, "Output & Mixed Regions (ρ, β, δ)")
        
        # Sub-Tab 3: MAPs Options (solver + plot parameters)
        self.tab_maps_options = QWidget()
        tab_maps_options_layout = QVBoxLayout(self.tab_maps_options)
        self.maps_opts_scroll = QScrollArea()
        self.maps_opts_scroll.setWidgetResizable(True)
        maps_opts_container = QWidget()
        maps_opts_layout = QVBoxLayout(maps_opts_container)

        # --- Sezione Calcolo ---
        calc_group = QGroupBox("Parametri di Calcolo MAPs")
        calc_layout = QGridLayout(calc_group)

        calc_layout.addWidget(QLabel("V min [m/s]:"), 0, 0)
        self.maps_v_min_input = QLineEdit("4.0")
        calc_layout.addWidget(self.maps_v_min_input, 0, 1)

        calc_layout.addWidget(QLabel("V max [m/s]:"), 1, 0)
        self.maps_v_max_input = QLineEdit("30.0")
        calc_layout.addWidget(self.maps_v_max_input, 1, 1)

        calc_layout.addWidget(QLabel("N° velocità griglia:"), 2, 0)
        self.maps_n_speeds_input = QLineEdit("50")
        calc_layout.addWidget(self.maps_n_speeds_input, 2, 1)

        calc_layout.addWidget(QLabel("N° punti / curva ay:"), 3, 0)
        self.maps_n_points_input = QLineEdit("70")
        calc_layout.addWidget(self.maps_n_points_input, 3, 1)

        calc_layout.addWidget(QLabel("ay max sweep [g]:"), 4, 0)
        self.maps_ay_max_input = QLineEdit("3.5")
        calc_layout.addWidget(self.maps_ay_max_input, 4, 1)

        calc_layout.addWidget(QLabel("Velocità evidenziate [m/s]:"), 5, 0)
        self.maps_highlight_v_input = QLineEdit("10, 15, 20, 25")
        calc_layout.addWidget(self.maps_highlight_v_input, 5, 1)

        maps_opts_layout.addWidget(calc_group)

        # --- Sezione Iso-Livelli ---
        iso_group = QGroupBox("Livelli Iso-Curve")
        iso_layout = QGridLayout(iso_group)

        iso_layout.addWidget(QLabel("Iso-δ [deg]:"), 0, 0)
        self.maps_iso_delta_input = QLineEdit("1, 2, 4, 6, 8, 10, 15, 25")
        iso_layout.addWidget(self.maps_iso_delta_input, 0, 1)

        iso_layout.addWidget(QLabel("Iso-ay [g]:"), 1, 0)
        self.maps_iso_ay_input = QLineEdit("0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8")
        iso_layout.addWidget(self.maps_iso_ay_input, 1, 1)

        iso_layout.addWidget(QLabel("Iso-ρ [1/m]:"), 2, 0)
        self.maps_iso_rho_input = QLineEdit("0.01, 0.02, 0.05, 0.10, 0.20")
        iso_layout.addWidget(self.maps_iso_rho_input, 2, 1)

        iso_layout.addWidget(QLabel("Iso-β [deg]:"), 3, 0)
        self.maps_iso_beta_input = QLineEdit("-10, -5, -3, -1, 0, 1, 3, 5, 10")
        iso_layout.addWidget(self.maps_iso_beta_input, 3, 1)

        maps_opts_layout.addWidget(iso_group)

        # --- Sezione Visibilità Plot ---
        vis_group = QGroupBox("Visibilità Elementi Plot")
        vis_layout = QVBoxLayout(vis_group)

        self.maps_show_iso_ay_cb   = QCheckBox("Mostra iso-ay")
        self.maps_show_iso_ay_cb.setChecked(True)
        self.maps_show_iso_rho_cb  = QCheckBox("Mostra iso-ρ (curvatura)")
        self.maps_show_iso_rho_cb.setChecked(True)
        self.maps_show_iso_beta_cb = QCheckBox("Mostra iso-β (sideslip)")
        self.maps_show_iso_beta_cb.setChecked(True)
        self.maps_show_envelope_cb = QCheckBox("Mostra bordo limite (envelope)")
        self.maps_show_envelope_cb.setChecked(True)
        self.maps_show_iso_delta_cb = QCheckBox("Mostra iso-δ (sterzo fisso)")
        self.maps_show_iso_delta_cb.setChecked(True)
        self.maps_show_highlight_cb = QCheckBox("Evidenzia velocità specifiche")
        self.maps_show_highlight_cb.setChecked(True)

        vis_layout.addWidget(self.maps_show_iso_ay_cb)
        vis_layout.addWidget(self.maps_show_iso_rho_cb)
        vis_layout.addWidget(self.maps_show_iso_beta_cb)
        vis_layout.addWidget(self.maps_show_envelope_cb)
        vis_layout.addWidget(self.maps_show_iso_delta_cb)
        vis_layout.addWidget(self.maps_show_highlight_cb)

        maps_opts_layout.addWidget(vis_group)
        maps_opts_layout.addStretch()

        # Sub-Tab 3: Vehicle Dynamic Stability & Sensitivity Maps
        self.tab_maps_dna = QWidget()
        self.tab_maps_dna_layout = QVBoxLayout(self.tab_maps_dna)
        
        dna_top_bar = QHBoxLayout()
        dna_top_bar.addWidget(QLabel("<b>Visualizzazione DNA:</b>"))
        self.maps_dna_view_combo = QComboBox()
        self.maps_dna_view_combo.addItems([
            "1. Mappe di Sensibilità n₁...n₆ & Modi Oscillatori",
            "2. Gradiente Sottosterzo Kus(δ, ay) & Soglia Stabilità ρ_y = 1/u²"
        ])
        self.maps_dna_view_combo.currentIndexChanged.connect(lambda _: self.plot_dna_maps())
        dna_top_bar.addWidget(self.maps_dna_view_combo)
        
        dna_top_bar.addWidget(QLabel("Velocità v [m/s]:"))
        self.maps_dna_v_input = QLineEdit("15.0")
        self.maps_dna_v_input.setMaximumWidth(45)
        dna_top_bar.addWidget(self.maps_dna_v_input)
        
        self.btn_plot_dna = QPushButton(" Calcola & Traccia Mappe DNA")
        self.btn_plot_dna.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 12px; border-radius: 3px;")
        self.btn_plot_dna.clicked.connect(self.plot_dna_maps)
        dna_top_bar.addWidget(self.btn_plot_dna)
        
        self.btn_guide_dna = QPushButton("ℹ Guida Stabilità Dinamica")
        self.btn_guide_dna.setStyleSheet("font-size: 11px; background-color: #dfe6e9; padding: 4px 8px; border-radius: 3px;")
        self.btn_guide_dna.clicked.connect(lambda: self.show_chart_guide("DNA_MAPS"))
        dna_top_bar.addWidget(self.btn_guide_dna)
        
        self.fig_maps_dna = Figure(figsize=(10, 7.5), dpi=100)
        self.canvas_maps_dna = FigureCanvas(self.fig_maps_dna)
        self.setup_popout(self.canvas_maps_dna, "Mappe di Stabilità Dinamica & Sensibilità (n1-n6)", dna_top_bar)
        
        dna_top_bar.addStretch()
        self.tab_maps_dna_layout.addLayout(dna_top_bar)
        
        self.toolbar_maps_dna = NavigationToolbar(self.canvas_maps_dna, self.tab_maps_dna)
        self.tab_maps_dna_layout.addWidget(self.toolbar_maps_dna)
        self.tab_maps_dna_layout.addWidget(self.canvas_maps_dna)
        
        self.maps_subtabs.addTab(self.tab_maps_dna, " Mappe Stabilità Dinamica & Sensibilità")
        
        self.maps_opts_scroll.setWidget(maps_opts_container)
        tab_maps_options_layout.addWidget(self.maps_opts_scroll)
        self.maps_subtabs.addTab(self.tab_maps_options, " Parametri di Calcolo & Opzioni MAPs")
        
        self.tabs.addTab(self.tab_maps, "Map of Achievable Performance (MAPs)")

        
        # Sweep Results Tab
        self.tab_sweep_res = QWidget()
        self.tab_sweep_res_layout = QVBoxLayout(self.tab_sweep_res)
        
        sweep_top_bar = QHBoxLayout()
        self.btn_info_sweep = QPushButton("ℹ Guida & Interpretazione (Speed Sweep)")
        self.btn_info_sweep.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_sweep.clicked.connect(lambda: self.show_chart_guide("SWEEP"))
        sweep_top_bar.addWidget(self.btn_info_sweep)
        
        self.canvas_sweep = MplCanvas(self, width=6, height=5, dpi=100)
        self.setup_popout(self.canvas_sweep, "Risultati Speed Sweep Ottimizzato", sweep_top_bar)
        sweep_top_bar.addStretch()
        self.tab_sweep_res_layout.addLayout(sweep_top_bar)
        
        # Splitter with interactive parameter/KPI selector on left and Canvas on right
        self.splitter_sweep = QSplitter(Qt.Horizontal)
        
        sweep_side_panel = QWidget()
        sweep_side_layout = QVBoxLayout(sweep_side_panel)
        sweep_side_layout.setContentsMargins(0, 0, 0, 0)
        sweep_side_layout.setSpacing(4)
        
        lbl_sweep_params = QLabel("<b>Canali Sweep & KPI:</b>")
        lbl_sweep_params.setStyleSheet("font-size: 11px; color: #0984e3; padding: 2px;")
        sweep_side_layout.addWidget(lbl_sweep_params)
        
        sweep_btn_box = QHBoxLayout()
        btn_sw_def = QPushButton("Default")
        btn_sw_def.setStyleSheet("font-size: 10px; padding: 2px 5px; background-color: #34495e; color: white;")
        btn_sw_def.clicked.connect(self._reset_sweep_params_default)
        btn_sw_susp = QPushButton("Sospensioni")
        btn_sw_susp.setStyleSheet("font-size: 10px; padding: 2px 5px; background-color: #34495e; color: white;")
        btn_sw_susp.clicked.connect(self._select_sweep_params_suspension)
        btn_sw_all = QPushButton("Tutti")
        btn_sw_all.setStyleSheet("font-size: 10px; padding: 2px 5px; background-color: #34495e; color: white;")
        btn_sw_all.clicked.connect(self._select_all_sweep_params)
        btn_sw_none = QPushButton("Nessuno")
        btn_sw_none.setStyleSheet("font-size: 10px; padding: 2px 5px; background-color: #34495e; color: white;")
        btn_sw_none.clicked.connect(self._deselect_all_sweep_params)
        sweep_btn_box.addWidget(btn_sw_def)
        sweep_btn_box.addWidget(btn_sw_susp)
        sweep_btn_box.addWidget(btn_sw_all)
        sweep_btn_box.addWidget(btn_sw_none)
        sweep_side_layout.addLayout(sweep_btn_box)
        
        self.sweep_param_list = QListWidget()
        self.sweep_param_list.setStyleSheet("font-size: 10px;")
        self.sweep_param_list.itemChanged.connect(self._on_sweep_param_item_changed)
        sweep_side_layout.addWidget(self.sweep_param_list)
        
        self.splitter_sweep.addWidget(sweep_side_panel)
        
        sweep_canvas_container = QWidget()
        sweep_canvas_layout = QVBoxLayout(sweep_canvas_container)
        sweep_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_sweep = NavigationToolbar(self.canvas_sweep, self.tab_sweep_res)
        sweep_canvas_layout.addWidget(self.toolbar_sweep)
        sweep_canvas_layout.addWidget(self.canvas_sweep)
        self.splitter_sweep.addWidget(sweep_canvas_container)
        self.splitter_sweep.setSizes([240, 960])
        
        self.tab_sweep_res_layout.addWidget(self.splitter_sweep)
        self.tabs.addTab(self.tab_sweep_res, "Sweep Results")
        
        # KPI Dashboard Tab with Graphical Parameter Comparison & Delta Chart
        self.tab_kpi = QWidget()
        self.tab_kpi_layout = QVBoxLayout(self.tab_kpi)
        
        kpi_top_bar = QHBoxLayout()
        self.btn_info_kpi = QPushButton("ℹ Guida & Interpretazione (Metriche Stabilità & KPI)")
        self.btn_info_kpi.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_kpi.clicked.connect(lambda: self.show_chart_guide("KPI"))
        kpi_top_bar.addWidget(self.btn_info_kpi)
        
        self.canvas_kpi_delta = MplCanvas(self, width=5, height=5, dpi=100)
        self.setup_popout(self.canvas_kpi_delta, "Confronto Setup Baseline vs Ottimizzato", kpi_top_bar)
        kpi_top_bar.addStretch()
        self.tab_kpi_layout.addLayout(kpi_top_bar)
        
        self.kpi_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.tab_kpi_layout.addWidget(self.kpi_splitter)
        
        # Left: Graphical Optimization Delta & Parameter Comparison Chart
        kpi_chart_widget = QWidget()
        kpi_chart_layout = QVBoxLayout(kpi_chart_widget)
        self.toolbar_kpi_delta = NavigationToolbar(self.canvas_kpi_delta, kpi_chart_widget)
        kpi_chart_layout.addWidget(self.toolbar_kpi_delta)
        kpi_chart_layout.addWidget(self.canvas_kpi_delta)
        self.kpi_splitter.addWidget(kpi_chart_widget)
        
        # Right: KPI Table
        self.kpi_table = QTableWidget(0, 2)
        self.kpi_table.setHorizontalHeaderLabels(["KPI Name", "Value"])
        self.kpi_table.horizontalHeader().setStretchLastSection(True)
        self.kpi_splitter.addWidget(self.kpi_table)
        self.kpi_splitter.setSizes([550, 450])
        
        self.tabs.addTab(self.tab_kpi, "KPI Dashboard & Optimization Deltas")
        
        # Telemetry Tab (Skidpad & Acceleration)
        self.tab_telemetry = QWidget()
        self.tab_telemetry_layout = QVBoxLayout(self.tab_telemetry)
        
        telem_top_bar = QHBoxLayout()
        self.btn_info_telem = QPushButton("ℹ Guida & Interpretazione (Prove Dinamiche)")
        self.btn_info_telem.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_telem.clicked.connect(lambda: self.show_chart_guide("TELEMETRY"))
        telem_top_bar.addWidget(self.btn_info_telem)
        telem_top_bar.addStretch()
        self.tab_telemetry_layout.addLayout(telem_top_bar)
        
        self.telemetry_subtabs = QTabWidget()
        self.tab_telemetry_layout.addWidget(self.telemetry_subtabs)
        
        # Subtab 1: Skidpad Telemetry
        self.tab_telem_skidpad = QWidget()
        self.tab_telem_skidpad_layout = QVBoxLayout(self.tab_telem_skidpad)
        
        skid_btn_bar = QHBoxLayout()
        self.btn_export_skidpad = QPushButton(" Esporta Telemetria Skidpad (CSV)")
        self.btn_export_skidpad.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #27ae60; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_export_skidpad.clicked.connect(self.export_skidpad_csv)
        skid_btn_bar.addWidget(self.btn_export_skidpad)
        
        self.fig_telem_skidpad = Figure(figsize=(10, 8), dpi=100)
        self.canvas_telem_skidpad = FigureCanvas(self.fig_telem_skidpad)
        self.setup_popout(self.canvas_telem_skidpad, "Telemetria Skidpad", skid_btn_bar)
        skid_btn_bar.addStretch()
        self.tab_telem_skidpad_layout.addLayout(skid_btn_bar)
        
        self.toolbar_telem_skidpad = NavigationToolbar(self.canvas_telem_skidpad, self.tab_telem_skidpad)
        self.tab_telem_skidpad_layout.addWidget(self.toolbar_telem_skidpad)
        self.tab_telem_skidpad_layout.addWidget(self.canvas_telem_skidpad)
        self.telemetry_subtabs.addTab(self.tab_telem_skidpad, " Skidpad Telemetria")
        
        # Subtab 2: Acceleration Telemetry
        self.tab_telem_accel = QWidget()
        self.tab_telem_accel_layout = QVBoxLayout(self.tab_telem_accel)
        
        accel_btn_bar = QHBoxLayout()
        self.btn_export_accel = QPushButton(" Esporta Telemetria Acceleration (CSV)")
        self.btn_export_accel.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #e67e22; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_export_accel.clicked.connect(self.export_accel_csv)
        accel_btn_bar.addWidget(self.btn_export_accel)
        
        self.fig_telem_accel = Figure(figsize=(10, 8), dpi=100)
        self.canvas_telem_accel = FigureCanvas(self.fig_telem_accel)
        self.setup_popout(self.canvas_telem_accel, "Telemetria Acceleration", accel_btn_bar)
        accel_btn_bar.addStretch()
        self.tab_telem_accel_layout.addLayout(accel_btn_bar)
        
        self.toolbar_telem_accel = NavigationToolbar(self.canvas_telem_accel, self.tab_telem_accel)
        self.tab_telem_accel_layout.addWidget(self.toolbar_telem_accel)
        self.tab_telem_accel_layout.addWidget(self.canvas_telem_accel)
        self.telemetry_subtabs.addTab(self.tab_telem_accel, " Acceleration Telemetria")
        
        # Subtab 3: Step Steer & DLC Transient Telemetry
        self.tab_telem_step = QWidget()
        self.tab_telem_step_layout = QVBoxLayout(self.tab_telem_step)
        
        step_btn_bar = QHBoxLayout()
        self.btn_export_step = QPushButton(" Esporta Telemetria Step/DLC (CSV)")
        self.btn_export_step.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_export_step.clicked.connect(self.export_step_csv)
        step_btn_bar.addWidget(self.btn_export_step)
        
        self.fig_telem_step = Figure(figsize=(10, 8), dpi=100)
        self.canvas_telem_step = FigureCanvas(self.fig_telem_step)
        self.setup_popout(self.canvas_telem_step, "Telemetria Step Steer & DLC Transitorio", step_btn_bar)
        step_btn_bar.addStretch()
        self.tab_telem_step_layout.addLayout(step_btn_bar)
        
        self.toolbar_telem_step = NavigationToolbar(self.canvas_telem_step, self.tab_telem_step)
        self.tab_telem_step_layout.addWidget(self.toolbar_telem_step)
        self.tab_telem_step_layout.addWidget(self.canvas_telem_step)
        self.telemetry_subtabs.addTab(self.tab_telem_step, " Step Steer & DLC Transitorio")
        
        # Subtab 4: AutoX & Endurance Simulation
        self.tab_telem_autox = QWidget()
        self.tab_telem_autox_layout = QVBoxLayout(self.tab_telem_autox)
        
        autox_btn_bar = QHBoxLayout()
        self.btn_export_autox = QPushButton(" Esporta Telemetria AutoX/Endurance (CSV)")
        self.btn_export_autox.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #00b894; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_export_autox.clicked.connect(self.export_autox_csv)
        autox_btn_bar.addWidget(self.btn_export_autox)
        
        self.fig_telem_autox = Figure(figsize=(10, 8), dpi=100)
        self.canvas_telem_autox = FigureCanvas(self.fig_telem_autox)
        self.setup_popout(self.canvas_telem_autox, "Telemetria AutoX & Endurance", autox_btn_bar)
        autox_btn_bar.addStretch()
        self.tab_telem_autox_layout.addLayout(autox_btn_bar)
        
        self.toolbar_telem_autox = NavigationToolbar(self.canvas_telem_autox, self.tab_telem_autox)
        self.tab_telem_autox_layout.addWidget(self.toolbar_telem_autox)
        self.tab_telem_autox_layout.addWidget(self.canvas_telem_autox)
        self.telemetry_subtabs.addTab(self.tab_telem_autox, " AutoX & Endurance (Consumi)")
        
        self.tabs.addTab(self.tab_telemetry, "Telemetria Prove Dinamiche")
        
        # Setup Tools & 101-Tips Tab
        self.tab_setup_tools = QWidget()
        self.tab_setup_tools_layout = QVBoxLayout(self.tab_setup_tools)
        
        setup_top_bar = QHBoxLayout()
        self.btn_info_setup_tools = QPushButton("ℹ Guida Setup & Principi di Assetto")
        self.btn_info_setup_tools.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_setup_tools.clicked.connect(lambda: self.show_chart_guide("SETUP_TOOLS"))
        setup_top_bar.addWidget(self.btn_info_setup_tools)
        setup_top_bar.addStretch()
        self.tab_setup_tools_layout.addLayout(setup_top_bar)
        
        self.setup_subtabs = QTabWidget()
        self.tab_setup_tools_layout.addWidget(self.setup_subtabs)
        
        # Sub-Tab 1: Steering Torque (Tip #8, #61)
        self.tab_steer_torque = QWidget()
        self.tab_steer_torque_layout = QVBoxLayout(self.tab_steer_torque)
        
        st_controls = QHBoxLayout()
        st_controls.addWidget(QLabel("Caster [°]:"))
        self.st_caster_input = QLineEdit("4.5")
        self.st_caster_input.setMaximumWidth(45)
        st_controls.addWidget(self.st_caster_input)
        
        st_controls.addWidget(QLabel("KPI [°]:"))
        self.st_kpi_input = QLineEdit("8.0")
        self.st_kpi_input.setMaximumWidth(45)
        st_controls.addWidget(self.st_kpi_input)
        
        st_controls.addWidget(QLabel("Mech Trail [mm]:"))
        self.st_mtrail_input = QLineEdit("18.0")
        self.st_mtrail_input.setMaximumWidth(45)
        st_controls.addWidget(self.st_mtrail_input)
        
        st_controls.addWidget(QLabel("Scrub Radius [mm]:"))
        self.st_scrub_input = QLineEdit("12.0")
        self.st_scrub_input.setMaximumWidth(45)
        st_controls.addWidget(self.st_scrub_input)
        
        st_controls.addWidget(QLabel("Steer Ratio (GR):"))
        self.st_ratio_input = QLineEdit("4.5")
        self.st_ratio_input.setMaximumWidth(45)
        st_controls.addWidget(self.st_ratio_input)
        
        self.btn_calc_steering_torque = QPushButton(" Ricalcola Coppia Sterzo")
        self.btn_calc_steering_torque.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_steering_torque.clicked.connect(self.plot_steering_torque)
        st_controls.addWidget(self.btn_calc_steering_torque)
        
        self.fig_steer_torque = Figure(figsize=(10, 6), dpi=100)
        self.canvas_steer_torque = FigureCanvas(self.fig_steer_torque)
        self.setup_popout(self.canvas_steer_torque, "Coppia Sterzo & Sforzo Pilota (Tip #8, #61)", st_controls)
        st_controls.addStretch()
        self.tab_steer_torque_layout.addLayout(st_controls)
        
        self.toolbar_steer_torque = NavigationToolbar(self.canvas_steer_torque, self.tab_steer_torque)
        self.tab_steer_torque_layout.addWidget(self.toolbar_steer_torque)
        self.tab_steer_torque_layout.addWidget(self.canvas_steer_torque)
        self.setup_subtabs.addTab(self.tab_steer_torque, " Coppia Sterzo & Sforzo Pilota (Tip #8, #61)")
        
        # Sub-Tab 2: Suspension & ARB Sizing (Tip #29, #87)
        self.tab_susp_sizing = QWidget()
        self.tab_susp_sizing_layout = QVBoxLayout(self.tab_susp_sizing)
        
        # Row 1: Target Sizing & ARB
        susp_controls = QHBoxLayout()
        susp_controls.addWidget(QLabel("Freq Ant. fn,f [Hz]:"))
        self.susp_fn_f_input = QLineEdit("3.0")
        self.susp_fn_f_input.setMaximumWidth(45)
        susp_controls.addWidget(self.susp_fn_f_input)
        
        susp_controls.addWidget(QLabel("Freq Post. fn,r [Hz]:"))
        self.susp_fn_r_input = QLineEdit("3.3")
        self.susp_fn_r_input.setMaximumWidth(45)
        susp_controls.addWidget(self.susp_fn_r_input)
        
        susp_controls.addWidget(QLabel("MR Damper (f,r):"))
        self.susp_mr_spring_input = QLineEdit("0.80, 0.80")
        self.susp_mr_spring_input.setMaximumWidth(65)
        susp_controls.addWidget(self.susp_mr_spring_input)
        
        susp_controls.addWidget(QLabel("MR ARB (f,r):"))
        self.susp_mr_arb_input = QLineEdit("0.40, 0.40")
        self.susp_mr_arb_input.setMaximumWidth(65)
        susp_controls.addWidget(self.susp_mr_arb_input)
        
        susp_controls.addWidget(QLabel("Materiale ARB:"))
        self.susp_arb_mat_combo = QComboBox()
        self.susp_arb_mat_combo.addItems(["Steel (Acciaio)", "Titanium (Titanio)"])
        susp_controls.addWidget(self.susp_arb_mat_combo)
        
        self.btn_calc_susp_sizing = QPushButton(" Calcola Molle & ARB")
        self.btn_calc_susp_sizing.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_susp_sizing.clicked.connect(self.calculate_and_plot_suspension_sizing)
        susp_controls.addWidget(self.btn_calc_susp_sizing)
        
        self.fig_susp_sizing = Figure(figsize=(7.5, 5), dpi=100)
        self.canvas_susp_sizing = FigureCanvas(self.fig_susp_sizing)
        self.setup_popout(self.canvas_susp_sizing, "Sizing Molle, Barre ARB & Corse Ammortizzatori", susp_controls)
        susp_controls.addStretch()
        self.tab_susp_sizing_layout.addLayout(susp_controls)

        # Row 2: Real Damper & Installed Spring Measurements & Preloads
        susp_real_bar = QHBoxLayout()
        susp_real_bar.addWidget(QLabel("<b>Setup Reale:</b> Molle Inst. F/R [N/mm]:"))
        self.susp_k_inst_f_input = QLineEdit("40.0")
        self.susp_k_inst_f_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_k_inst_f_input)
        self.susp_k_inst_r_input = QLineEdit("45.0")
        self.susp_k_inst_r_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_k_inst_r_input)

        susp_real_bar.addWidget(QLabel("Precarico F/R [mm]:"))
        self.susp_preload_f_input = QLineEdit("5.0")
        self.susp_preload_f_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_preload_f_input)
        self.susp_preload_r_input = QLineEdit("5.0")
        self.susp_preload_r_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_preload_r_input)

        susp_real_bar.addWidget(QLabel("Corsa TTX25 [mm]:"))
        self.susp_stroke_tot_input = QLineEdit("57.0")
        self.susp_stroke_tot_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_stroke_tot_input)

        susp_real_bar.addWidget(QLabel("Tampone [mm]:"))
        self.susp_bumpstop_gap_input = QLineEdit("20.0")
        self.susp_bumpstop_gap_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_bumpstop_gap_input)

        susp_real_bar.addWidget(QLabel("Altezza Terra F/R [mm]:"))
        self.susp_ride_h_f_input = QLineEdit("30.0")
        self.susp_ride_h_f_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_ride_h_f_input)
        self.susp_ride_h_r_input = QLineEdit("35.0")
        self.susp_ride_h_r_input.setMaximumWidth(38)
        susp_real_bar.addWidget(self.susp_ride_h_r_input)
        susp_real_bar.addStretch()
        self.tab_susp_sizing_layout.addLayout(susp_real_bar)
        
        # Results table and chart splitter
        susp_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.susp_results_table = QTableWidget()
        self.susp_results_table.setColumnCount(2)
        self.susp_results_table.setHorizontalHeaderLabels(["Parametro Sospensione", "Valore Calcolato"])
        self.susp_results_table.horizontalHeader().setStretchLastSection(True)
        susp_splitter.addWidget(self.susp_results_table)
        
        self.toolbar_susp_sizing = NavigationToolbar(self.canvas_susp_sizing, self.tab_susp_sizing)
        susp_chart_container = QWidget()
        susp_chart_layout = QVBoxLayout(susp_chart_container)
        susp_chart_layout.addWidget(self.toolbar_susp_sizing)
        susp_chart_layout.addWidget(self.canvas_susp_sizing)
        susp_splitter.addWidget(susp_chart_container)
        susp_splitter.setSizes([450, 550])
        self.tab_susp_sizing_layout.addWidget(susp_splitter)
        
        self.setup_subtabs.addTab(self.tab_susp_sizing, " Sizing Molle & Barre Antirollio (Tip #29, #87)")
        
        # Sub-Tab 3: Öhlins TTX25 MkII Damper Dyno (4-Way Non-Linear & Linear Mode)
        self.tab_damper_dyno = QWidget()
        self.tab_damper_dyno_layout = QVBoxLayout(self.tab_damper_dyno)

        damper_config_group = QGroupBox(" Configurazione Ammortizzatori Öhlins TTX25 MkII FSAE & Modalità di Smorzamento")
        damper_config_layout = QGridLayout(damper_config_group)

        # Mode Selection
        damper_config_layout.addWidget(QLabel("<b>Modalità Smorzamento:</b>"), 0, 0)
        self.damper_mode_combo = QComboBox()
        self.damper_mode_combo.addItem("Costante / Lineare (Default C_phi & C_z)", "linear")
        self.damper_mode_combo.addItem("Öhlins TTX25 MkII Non-Lineare (4-Way Dyno)", "ttx25_nonlinear")
        self.damper_mode_combo.setCurrentIndex(0)
        self.damper_mode_combo.setStyleSheet("font-weight: bold; padding: 2px;")
        damper_config_layout.addWidget(self.damper_mode_combo, 0, 1, 1, 3)

        self.btn_info_ttx25 = QPushButton("ℹ Guida Taratura TTX25")
        self.btn_info_ttx25.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_info_ttx25.clicked.connect(lambda: self.show_chart_guide("TTX25"))
        damper_config_layout.addWidget(self.btn_info_ttx25, 0, 4)

        self.btn_calc_damper_dyno = QPushButton(" Aggiorna Curve Dyno & Ratios")
        self.btn_calc_damper_dyno.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_damper_dyno.clicked.connect(self.plot_damper_dyno)
        damper_config_layout.addWidget(self.btn_calc_damper_dyno, 0, 5)

        # Row 1: Front Damper Adjusters
        valv_opts = ["C11/R11", "C12/R12", "C13/R13", "C21/R21", "C22/R22", "C23/R23"]
        damper_config_layout.addWidget(QLabel("<b>Ammortizzatore Anteriore:</b>"), 1, 0)
        damper_config_layout.addWidget(QLabel("Valvola:"), 1, 1)
        self.damper_valv_f_combo = QComboBox()
        self.damper_valv_f_combo.addItems(valv_opts)
        self.damper_valv_f_combo.setCurrentText("C12/R12")
        damper_config_layout.addWidget(self.damper_valv_f_combo, 1, 2)

        f_adj_row = QHBoxLayout()
        f_adj_row.addWidget(QLabel("LSC:"))
        self.damper_lsc_f_spin = QSpinBox()
        self.damper_lsc_f_spin.setRange(0, 25)
        self.damper_lsc_f_spin.setValue(10)
        self.damper_lsc_f_spin.setToolTip("Low Speed Comp [click da tutto chiuso (0=stiff, 25=soft)]")
        f_adj_row.addWidget(self.damper_lsc_f_spin)

        f_adj_row.addWidget(QLabel("HSC:"))
        self.damper_hsc_f_spin = QDoubleSpinBox()
        self.damper_hsc_f_spin.setRange(0.0, 4.3)
        self.damper_hsc_f_spin.setSingleStep(0.25)
        self.damper_hsc_f_spin.setValue(2.0)
        self.damper_hsc_f_spin.setToolTip("High Speed Comp [giri da tutto aperto (0=soft, 4.3=stiff)]")
        f_adj_row.addWidget(self.damper_hsc_f_spin)

        f_adj_row.addWidget(QLabel("LSR:"))
        self.damper_lsr_f_spin = QSpinBox()
        self.damper_lsr_f_spin.setRange(0, 25)
        self.damper_lsr_f_spin.setValue(10)
        self.damper_lsr_f_spin.setToolTip("Low Speed Reb [click da tutto chiuso (0=stiff, 25=soft)]")
        f_adj_row.addWidget(self.damper_lsr_f_spin)

        f_adj_row.addWidget(QLabel("HSR:"))
        self.damper_hsr_f_spin = QDoubleSpinBox()
        self.damper_hsr_f_spin.setRange(0.0, 4.3)
        self.damper_hsr_f_spin.setSingleStep(0.25)
        self.damper_hsr_f_spin.setValue(2.0)
        self.damper_hsr_f_spin.setToolTip("High Speed Reb [giri da tutto aperto (0=soft, 4.3=stiff)]")
        f_adj_row.addWidget(self.damper_hsr_f_spin)

        f_adj_row.addWidget(QLabel("MR:"))
        self.damper_mr_f_spin = QDoubleSpinBox()
        self.damper_mr_f_spin.setRange(0.4, 1.5)
        self.damper_mr_f_spin.setSingleStep(0.02)
        self.damper_mr_f_spin.setValue(0.85)
        f_adj_row.addWidget(self.damper_mr_f_spin)

        damper_config_layout.addLayout(f_adj_row, 1, 3, 1, 3)

        # Row 2: Rear Damper Adjusters
        damper_config_layout.addWidget(QLabel("<b>Ammortizzatore Posteriore:</b>"), 2, 0)
        damper_config_layout.addWidget(QLabel("Valvola:"), 2, 1)
        self.damper_valv_r_combo = QComboBox()
        self.damper_valv_r_combo.addItems(valv_opts)
        self.damper_valv_r_combo.setCurrentText("C12/R12")
        damper_config_layout.addWidget(self.damper_valv_r_combo, 2, 2)

        r_adj_row = QHBoxLayout()
        r_adj_row.addWidget(QLabel("LSC:"))
        self.damper_lsc_r_spin = QSpinBox()
        self.damper_lsc_r_spin.setRange(0, 25)
        self.damper_lsc_r_spin.setValue(10)
        r_adj_row.addWidget(self.damper_lsc_r_spin)

        r_adj_row.addWidget(QLabel("HSC:"))
        self.damper_hsc_r_spin = QDoubleSpinBox()
        self.damper_hsc_r_spin.setRange(0.0, 4.3)
        self.damper_hsc_r_spin.setSingleStep(0.25)
        self.damper_hsc_r_spin.setValue(2.0)
        r_adj_row.addWidget(self.damper_hsc_r_spin)

        r_adj_row.addWidget(QLabel("LSR:"))
        self.damper_lsr_r_spin = QSpinBox()
        self.damper_lsr_r_spin.setRange(0, 25)
        self.damper_lsr_r_spin.setValue(10)
        r_adj_row.addWidget(self.damper_lsr_r_spin)

        r_adj_row.addWidget(QLabel("HSR:"))
        self.damper_hsr_r_spin = QDoubleSpinBox()
        self.damper_hsr_r_spin.setRange(0.0, 4.3)
        self.damper_hsr_r_spin.setSingleStep(0.25)
        self.damper_hsr_r_spin.setValue(2.0)
        r_adj_row.addWidget(self.damper_hsr_r_spin)

        r_adj_row.addWidget(QLabel("MR:"))
        self.damper_mr_r_spin = QDoubleSpinBox()
        self.damper_mr_r_spin.setRange(0.4, 1.5)
        self.damper_mr_r_spin.setSingleStep(0.02)
        self.damper_mr_r_spin.setValue(0.85)
        r_adj_row.addWidget(self.damper_mr_r_spin)

        damper_config_layout.addLayout(r_adj_row, 2, 3, 1, 3)

        self.tab_damper_dyno_layout.addWidget(damper_config_group)

        # Plot and KPI Splitter
        damper_splitter = QSplitter(QtCore.Qt.Horizontal)
        self.fig_damper_dyno = Figure(figsize=(9, 5), dpi=100)
        self.canvas_damper_dyno = FigureCanvas(self.fig_damper_dyno)
        self.setup_popout(self.canvas_damper_dyno, "Öhlins TTX25 Damper Dyno (F-V & Modal Ratios)", None)
        self.toolbar_damper_dyno = NavigationToolbar(self.canvas_damper_dyno, self.tab_damper_dyno)

        damper_chart_container = QWidget()
        damper_chart_layout = QVBoxLayout(damper_chart_container)
        damper_chart_layout.addWidget(self.toolbar_damper_dyno)
        damper_chart_layout.addWidget(self.canvas_damper_dyno)
        damper_splitter.addWidget(damper_chart_container)

        self.damper_results_table = QTableWidget()
        self.damper_results_table.setColumnCount(2)
        self.damper_results_table.setHorizontalHeaderLabels(["Metrica Smorzamento", "Valore Calcolato"])
        self.damper_results_table.horizontalHeader().setStretchLastSection(True)
        damper_splitter.addWidget(self.damper_results_table)
        damper_splitter.setSizes([600, 400])

        self.tab_damper_dyno_layout.addWidget(damper_splitter)
        self.setup_subtabs.addTab(self.tab_damper_dyno, " Öhlins TTX25 Damper Dyno (4-Way Non-Lineare)")
        
        # Sub-Tab 3: Optimal Balance Range & 4-Ratio Matrix (Tip #82, #84, #85, #89)
        self.tab_radar_balance = QWidget()
        self.tab_radar_balance_layout = QVBoxLayout(self.tab_radar_balance)
        
        # 1. Top Configuration Group for Parameter Sweep & Uncertainties
        bal_sweep_group = QGroupBox(" Configurazione Sweep di Bilanciamento & Incertezze Parametriche")
        bal_sweep_layout = QGridLayout(bal_sweep_group)
        
        bal_sweep_layout.addWidget(QLabel("<b>Parametro da Variare:</b>"), 0, 0)
        self.combo_bal_param = QComboBox()
        self.combo_bal_param.addItem("Ripartizione Trasferimento Carico (LLTD %)", "lltd")
        self.combo_bal_param.addItem("Distribuzione Masse Avantreno (wd_f %)", "wd_f")
        self.combo_bal_param.addItem("Bilanciamento Aerodinamico (aero_bal %)", "aero_balance")
        self.combo_bal_param.addItem("Centro di Rollio Ant. (rc_f mm)", "rc_height_f")
        self.combo_bal_param.addItem("Centro di Rollio Post. (rc_r mm)", "rc_height_r")
        self.combo_bal_param.addItem("Altezza Baricentro (h_cg mm)", "h_cg")
        self.combo_bal_param.addItem("Rigidezza Rollio Ant. (K_phi_f Nm/deg)", "K_phi_f")
        bal_sweep_layout.addWidget(self.combo_bal_param, 0, 1)
        
        bal_sweep_layout.addWidget(QLabel("Min:"), 0, 2)
        self.bal_min_input = QLineEdit("40.0")
        self.bal_min_input.setMaximumWidth(50)
        bal_sweep_layout.addWidget(self.bal_min_input, 0, 3)
        
        bal_sweep_layout.addWidget(QLabel("Max:"), 0, 4)
        self.bal_max_input = QLineEdit("70.0")
        self.bal_max_input.setMaximumWidth(50)
        bal_sweep_layout.addWidget(self.bal_max_input, 0, 5)
        
        bal_sweep_layout.addWidget(QLabel("Punti:"), 0, 6)
        self.bal_pts_input = QLineEdit("16")
        self.bal_pts_input.setMaximumWidth(40)
        bal_sweep_layout.addWidget(self.bal_pts_input, 0, 7)
        
        bal_sweep_layout.addWidget(QLabel("Velocità Test (m/s):"), 0, 8)
        self.bal_vx_input = QLineEdit("12.0")
        self.bal_vx_input.setMaximumWidth(45)
        bal_sweep_layout.addWidget(self.bal_vx_input, 0, 9)

        # Row 1: Incertezze Parametriche & Target Window
        bal_sweep_layout.addWidget(QLabel("<b>Incertezze (±%):</b>"), 1, 0)
        
        unc_row = QHBoxLayout()
        unc_row.addWidget(QLabel("Grip Gomme Δμ:"))
        self.bal_unc_mu_input = QLineEdit("10.0")
        self.bal_unc_mu_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_unc_mu_input)
        
        unc_row.addWidget(QLabel("Massa ΔWD:"))
        self.bal_unc_wd_input = QLineEdit("1.5")
        self.bal_unc_wd_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_unc_wd_input)
        
        unc_row.addWidget(QLabel("Aero ΔAero:"))
        self.bal_unc_aero_input = QLineEdit("2.0")
        self.bal_unc_aero_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_unc_aero_input)

        unc_row.addWidget(QLabel("Rigidezza ΔLLTD:"))
        self.bal_unc_lltd_input = QLineEdit("2.0")
        self.bal_unc_lltd_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_unc_lltd_input)
        
        unc_row.addWidget(QLabel("Target K_us [deg/g]:"))
        self.bal_kus_min_input = QLineEdit("0.10")
        self.bal_kus_min_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_kus_min_input)
        unc_row.addWidget(QLabel("a"))
        self.bal_kus_max_input = QLineEdit("0.80")
        self.bal_kus_max_input.setMaximumWidth(45)
        unc_row.addWidget(self.bal_kus_max_input)
        
        bal_sweep_layout.addLayout(unc_row, 1, 1, 1, 9)
        
        self.combo_bal_param.currentIndexChanged.connect(self._on_bal_param_changed)
        self.tab_radar_balance_layout.addWidget(bal_sweep_group)
        
        # 2. Action Toolbar & Status Banner
        radar_top = QHBoxLayout()
        self.lbl_stability_margin = QLabel("Stabilità & Finestra Ottimale: In attesa di calcolo")
        self.lbl_stability_margin.setStyleSheet("font-weight: bold; font-size: 11px; color: #2d3436; padding: 4px;")
        radar_top.addWidget(self.lbl_stability_margin)
        
        self.btn_calc_balance_range = QPushButton(" Calcola Finestra di Bilanciamento & Incertezza")
        self.btn_calc_balance_range.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 4px 12px; border-radius: 3px;")
        self.btn_calc_balance_range.clicked.connect(self.plot_optimal_balance_sweep)
        radar_top.addWidget(self.btn_calc_balance_range)

        self.btn_calc_radar = QPushButton(" Mostra Radar 4-Rapporti")
        self.btn_calc_radar.setStyleSheet("font-weight: bold; background-color: #e17055; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_radar.clicked.connect(self.plot_4_ratios_radar)
        radar_top.addWidget(self.btn_calc_radar)
        
        self.fig_radar = Figure(figsize=(10, 6), dpi=100)
        self.canvas_radar = FigureCanvas(self.fig_radar)
        self.setup_popout(self.canvas_radar, "Finestra di Bilanciamento Ottimale & 4-Rapporti", radar_top)
        radar_top.addStretch()
        self.tab_radar_balance_layout.addLayout(radar_top)
        
        self.toolbar_radar = NavigationToolbar(self.canvas_radar, self.tab_radar_balance)
        self.tab_radar_balance_layout.addWidget(self.toolbar_radar)
        self.tab_radar_balance_layout.addWidget(self.canvas_radar)
        self.setup_subtabs.addTab(self.tab_radar_balance, " Finestra di Bilanciamento & 4-Rapporti (Tip #82, #84, #89)")
        
        # Sub-Tab 4: Rombo di Setting ARB & Bilanciamento Meccanico (Modello 14-DOF)
        self._init_tab_arb_diamond()
        
        # Sub-Tab 5: Banco Dinamico 7-Post Shaker Rig (4 Stradali + 3 Aero)
        self._init_tab_seven_post()
        
        self.tab_transient_sensitivity = QWidget()
        self.tab_transient_sensitivity_layout = QVBoxLayout(self.tab_transient_sensitivity)
        
        # 0. Quick Presets Toolbar
        sens_presets_bar = QHBoxLayout()
        sens_presets_bar.addWidget(QLabel("<b> Preset Rapidi:</b>"))
        
        btn_pre_rc_f = QPushButton(" RC Front (q₁)")
        btn_pre_rc_f.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_rc_f.clicked.connect(lambda: self._apply_sens_preset("rc_f"))
        sens_presets_bar.addWidget(btn_pre_rc_f)
        
        btn_pre_rc_r = QPushButton(" RC Rear (q₂)")
        btn_pre_rc_r.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_rc_r.clicked.connect(lambda: self._apply_sens_preset("rc_r"))
        sens_presets_bar.addWidget(btn_pre_rc_r)
        
        btn_pre_rc_2d = QPushButton(" 2D Grid (q₁ × q₂)")
        btn_pre_rc_2d.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #ffeaa7; color: #d35400; border-radius: 3px; font-weight: bold;")
        btn_pre_rc_2d.clicked.connect(lambda: self._apply_sens_preset("rc_2d"))
        sens_presets_bar.addWidget(btn_pre_rc_2d)
        
        btn_pre_arb = QPushButton(" Ripartizione ARB")
        btn_pre_arb.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_arb.clicked.connect(lambda: self._apply_sens_preset("arb"))
        sens_presets_bar.addWidget(btn_pre_arb)
        
        btn_pre_damp = QPushButton(" Smorzamento C_φ")
        btn_pre_damp.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_damp.clicked.connect(lambda: self._apply_sens_preset("damp"))
        sens_presets_bar.addWidget(btn_pre_damp)
        
        btn_pre_mig = QPushButton(" Migrazione Lat. RC")
        btn_pre_mig.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_mig.clicked.connect(lambda: self._apply_sens_preset("mig"))
        sens_presets_bar.addWidget(btn_pre_mig)
        
        btn_pre_hcg = QPushButton(" Baricentro (h_cg)")
        btn_pre_hcg.setStyleSheet("font-size: 10px; padding: 2px 6px; background-color: #dfe6e9; border-radius: 3px; font-weight: bold;")
        btn_pre_hcg.clicked.connect(lambda: self._apply_sens_preset("hcg"))
        sens_presets_bar.addWidget(btn_pre_hcg)
        
        sens_presets_bar.addStretch()
        self.tab_transient_sensitivity_layout.addLayout(sens_presets_bar)
        
        # 1. Top Control Bar (Row 1: Mode, Scenarios, Maneuver, Speed, Steer, KPI)
        sens_top1 = QHBoxLayout()
        sens_top1.addWidget(QLabel("<b>Modalità:</b>"))
        self.sens_mode_combo = QComboBox()
        self.sens_mode_combo.addItems(["1D Parameter Sweep (Curve & Time-History)", "2D Heatmap Grid (Bivariata X-Y)"])
        self.sens_mode_combo.currentIndexChanged.connect(self._on_sensitivity_mode_changed)
        sens_top1.addWidget(self.sens_mode_combo)
        
        sens_top1.addWidget(QLabel("<b>Parametro (X):</b>"))
        self.sens_scen_x_combo = QComboBox()
        for k, info in SCENARIO_CATALOG.items():
            self.sens_scen_x_combo.addItem(info['label'], k)
        self.sens_scen_x_combo.currentIndexChanged.connect(self._on_sensitivity_scenario_changed)
        sens_top1.addWidget(self.sens_scen_x_combo)
        
        self.lbl_sens_scen_y = QLabel("<b>Parametro (Y):</b>")
        self.sens_scen_y_combo = QComboBox()
        for k, info in SCENARIO_CATALOG.items():
            self.sens_scen_y_combo.addItem(info['label'], k)
        self.sens_scen_y_combo.setCurrentIndex(1)
        sens_top1.addWidget(self.lbl_sens_scen_y)
        sens_top1.addWidget(self.sens_scen_y_combo)
        self.lbl_sens_scen_y.setVisible(False)
        self.sens_scen_y_combo.setVisible(False)
        
        sens_top1.addWidget(QLabel("<b>Manovra:</b>"))
        self.sens_maneuver_combo = QComboBox()
        self.sens_maneuver_combo.addItems(["Step Steer (Ramp Step ISO 7401)", "Double Lane Change (DLC ISO 3888-2)"])
        sens_top1.addWidget(self.sens_maneuver_combo)
        
        sens_top1.addWidget(QLabel("Velocità [m/s]:"))
        self.sens_vx_input = QLineEdit("15.0")
        self.sens_vx_input.setMaximumWidth(45)
        self.sens_vx_input.textChanged.connect(lambda _: self._update_stability_card())
        sens_top1.addWidget(self.sens_vx_input)
        
        sens_top1.addWidget(QLabel("Sterzo [°]:"))
        self.sens_steer_input = QLineEdit("3.0")
        self.sens_steer_input.setMaximumWidth(45)
        sens_top1.addWidget(self.sens_steer_input)
        
        sens_top1.addWidget(QLabel("<b>KPI Target:</b>"))
        self.sens_kpi_combo = QComboBox()
        for k, lbl, _ in KPI_METADATA:
            self.sens_kpi_combo.addItem(lbl, k)
        sens_top1.addWidget(self.sens_kpi_combo)
        sens_top1.addStretch()
        self.tab_transient_sensitivity_layout.addLayout(sens_top1)
        
        # 2. Control Bar Row 2 (Ranges, N points, Run/Export/Guide Buttons)
        sens_top2 = QHBoxLayout()
        sens_top2.addWidget(QLabel("<b>Range X:</b> Min:"))
        self.sens_min_x_input = QLineEdit("0.02")
        self.sens_min_x_input.setMaximumWidth(50)
        sens_top2.addWidget(self.sens_min_x_input)
        sens_top2.addWidget(QLabel("Max:"))
        self.sens_max_x_input = QLineEdit("0.14")
        self.sens_max_x_input.setMaximumWidth(50)
        sens_top2.addWidget(self.sens_max_x_input)
        sens_top2.addWidget(QLabel("Punti N:"))
        self.sens_pts_x_spin = QSpinBox()
        self.sens_pts_x_spin.setRange(3, 50)
        self.sens_pts_x_spin.setValue(11)
        self.sens_pts_x_spin.setMaximumWidth(50)
        sens_top2.addWidget(self.sens_pts_x_spin)
        
        self.lbl_sens_range_y = QLabel("<b>Range Y:</b> Min:")
        self.sens_min_y_input = QLineEdit("0.02")
        self.sens_min_y_input.setMaximumWidth(50)
        self.lbl_sens_max_y = QLabel("Max:")
        self.sens_max_y_input = QLineEdit("0.16")
        self.sens_max_y_input.setMaximumWidth(50)
        self.lbl_sens_pts_y = QLabel("Punti Y:")
        self.sens_pts_y_spin = QSpinBox()
        self.sens_pts_y_spin.setRange(3, 25)
        self.sens_pts_y_spin.setValue(7)
        self.sens_pts_y_spin.setMaximumWidth(50)
        
        sens_top2.addWidget(self.lbl_sens_range_y)
        sens_top2.addWidget(self.sens_min_y_input)
        sens_top2.addWidget(self.lbl_sens_max_y)
        sens_top2.addWidget(self.sens_max_y_input)
        sens_top2.addWidget(self.lbl_sens_pts_y)
        sens_top2.addWidget(self.sens_pts_y_spin)
        
        self.lbl_sens_range_y.setVisible(False)
        self.sens_min_y_input.setVisible(False)
        self.lbl_sens_max_y.setVisible(False)
        self.sens_max_y_input.setVisible(False)
        self.lbl_sens_pts_y.setVisible(False)
        self.sens_pts_y_spin.setVisible(False)
        
        sens_top2.addSpacing(15)
        self.btn_run_sensitivity = QPushButton(" Esegui Sensitivity Sweep")
        self.btn_run_sensitivity.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 5px 12px; border-radius: 4px;")
        self.btn_run_sensitivity.clicked.connect(self.run_sensitivity_sweep)
        sens_top2.addWidget(self.btn_run_sensitivity)
        
        self.btn_export_sensitivity = QPushButton(" Esporta CSV")
        self.btn_export_sensitivity.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 5px 10px; border-radius: 4px;")
        self.btn_export_sensitivity.clicked.connect(self.export_sensitivity_csv)
        sens_top2.addWidget(self.btn_export_sensitivity)
        
        self.btn_guide_sensitivity = QPushButton("ℹ Guida Fisica")
        self.btn_guide_sensitivity.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 5px 10px; border-radius: 4px;")
        self.btn_guide_sensitivity.clicked.connect(lambda: self.show_chart_guide("SENSITIVITY_TRANSIENT"))
        sens_top2.addWidget(self.btn_guide_sensitivity)
        
        self.fig_sensitivity = Figure(figsize=(10, 8), dpi=100)
        self.canvas_sensitivity = FigureCanvas(self.fig_sensitivity)
        self.setup_popout(self.canvas_sensitivity, "Transient Sensitivity & Roll Centers", sens_top2)
        sens_top2.addStretch()
        self.tab_transient_sensitivity_layout.addLayout(sens_top2)
        
        # Progress bar + status
        sens_prog_layout = QHBoxLayout()
        self.sens_prog_bar = QProgressBar()
        self.sens_prog_bar.setRange(0, 100)
        self.sens_prog_bar.setValue(0)
        self.sens_prog_bar.setMaximumHeight(14)
        self.sens_prog_bar.setTextVisible(True)
        sens_prog_layout.addWidget(self.sens_prog_bar)
        
        self.sens_status_lbl = QLabel("Pronto. Seleziona i parametri o usa un Preset Rapido e clicca su Esegui Sweep.")
        self.sens_status_lbl.setStyleSheet("font-size: 10.5px; color: #636e72; font-style: italic;")
        sens_prog_layout.addWidget(self.sens_status_lbl)
        self.tab_transient_sensitivity_layout.addLayout(sens_prog_layout)
        
        # Splitter: Left Table & Physics Description, Right Matplotlib Plot
        sens_splitter = QSplitter(QtCore.Qt.Horizontal)
        
        # Left container
        sens_left_panel = QWidget()
        sens_left_layout = QVBoxLayout(sens_left_panel)
        sens_left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Stability Derivatives & Modal Analysis Card
        self.sens_stab_box = QGroupBox(" Derivate di Stabilità & Dinamica Modale (Setup Attivo)")
        sens_stab_layout = QVBoxLayout(self.sens_stab_box)
        self.sens_stab_browser = QTextBrowser()
        self.sens_stab_browser.setMaximumHeight(135)
        self.sens_stab_browser.setStyleSheet("background-color: #2d3436; color: #f5f6fa; border: 1px solid #636e72; font-size: 10.5px; padding: 4px;")
        sens_stab_layout.addWidget(self.sens_stab_browser)
        sens_left_layout.addWidget(self.sens_stab_box)
        
        # 2. Physics Description Card
        self.sens_desc_box = QGroupBox(" Fisica dello Scenario & Effetto Dinamico")
        sens_desc_layout = QVBoxLayout(self.sens_desc_box)
        self.sens_desc_browser = QTextBrowser()
        self.sens_desc_browser.setMaximumHeight(90)
        self.sens_desc_browser.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dfe6e9; font-size: 10.5px; padding: 4px;")
        sens_desc_layout.addWidget(self.sens_desc_browser)
        sens_left_layout.addWidget(self.sens_desc_box)
        
        # 3. KPI Results Table
        self.sens_table_box = QGroupBox(" Tabella KPI Sintetici Estratti")
        sens_table_layout = QVBoxLayout(self.sens_table_box)
        self.sens_table = QTableWidget()
        self.sens_table.setStyleSheet("font-size: 9.5px; alternate-background-color: #f1f2f6;")
        self.sens_table.setAlternatingRowColors(True)
        sens_table_layout.addWidget(self.sens_table)
        sens_left_layout.addWidget(self.sens_table_box)
        
        sens_splitter.addWidget(sens_left_panel)
        
        # Right Plot container
        sens_right_panel = QWidget()
        sens_right_layout = QVBoxLayout(sens_right_panel)
        sens_right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.toolbar_sensitivity = NavigationToolbar(self.canvas_sensitivity, self.tab_transient_sensitivity)
        sens_right_layout.addWidget(self.toolbar_sensitivity)
        sens_right_layout.addWidget(self.canvas_sensitivity)
        
        sens_splitter.addWidget(sens_right_panel)
        sens_splitter.setSizes([380, 720])
        self.tab_transient_sensitivity_layout.addWidget(sens_splitter)
        
        self.setup_subtabs.addTab(self.tab_transient_sensitivity, " Transient Sensitivity & Roll Centers")
        
        # Sub-Tab 5: Kinematic Maps vs Wheel Travel (Camber, Toe, Roll Centers, Ackermann)
        self.tab_kinematics = QWidget()
        self.tab_kinematics_layout = QVBoxLayout(self.tab_kinematics)
        
        # Row 1: Preset selection, folder/file loaders, inspection table
        kin_top_row = QHBoxLayout()
        kin_top_row.addWidget(QLabel("<b>Preset Rapido:</b>"))
        self.combo_kin_preset = QComboBox()
        self.combo_kin_preset.setStyleSheet("background-color: #2d3436; color: white; padding: 3px 6px; font-weight: bold;")
        self.combo_kin_preset.addItems([
            "⚡ Seleziona Preset Rapido...",
            "🟢 ET-18 (2026 Adams Digitized)",
            "🟡 ET-17 (2025 Adams Digitized)"
        ])
        self.combo_kin_preset.currentIndexChanged.connect(self._on_kin_preset_combo_changed)
        kin_top_row.addWidget(self.combo_kin_preset)

        self.btn_load_kin_folder = QPushButton("📁 Carica Pack Curve (Cartella)")
        self.btn_load_kin_folder.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_load_kin_folder.setToolTip("Carica un intero set/cartella di curve cinematiche digitalizzate (.json)")
        self.btn_load_kin_folder.clicked.connect(self.import_digitized_curves_folder)
        kin_top_row.addWidget(self.btn_load_kin_folder)

        self.btn_load_kin_json = QPushButton("📄 Carica Singolo JSON")
        self.btn_load_kin_json.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_load_kin_json.setToolTip("Carica un singolo file JSON di curva cinematica digitalizzata")
        self.btn_load_kin_json.clicked.connect(self.load_kinematic_json)
        kin_top_row.addWidget(self.btn_load_kin_json)

        self.btn_inspect_kin_curves = QPushButton("📋 Tabella Sinottica 16 Curve")
        self.btn_inspect_kin_curves.setStyleSheet("font-weight: bold; background-color: #00cec9; color: black; padding: 4px 8px; border-radius: 3px;")
        self.btn_inspect_kin_curves.setToolTip("Visualizza la tabella completa con tutte le 16 curve Adams caricate (incluso Ackermann), polinomi e valori statici")
        self.btn_inspect_kin_curves.clicked.connect(self.show_active_parameters_dialog)
        kin_top_row.addWidget(self.btn_inspect_kin_curves)

        self.btn_replot_kin = QPushButton("🔄 Ricalcola Curve")
        self.btn_replot_kin.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_replot_kin.clicked.connect(self.plot_kinematic_maps)
        kin_top_row.addWidget(self.btn_replot_kin)

        kin_top_row.addWidget(QLabel("<b>Layout Vista:</b>"))
        self.combo_kin_view = QComboBox()
        self.combo_kin_view.setStyleSheet("background-color: #2d3436; color: white; padding: 3px 6px; font-weight: bold;")
        self.combo_kin_view.addItems([
            "📊 Vista Completa (8 Grafici: Sospensioni + Sterzo)",
            "🏎️ Sospensioni & Rollio (6 Grafici: Heave & Roll)",
            "🎯 Dettaglio Sterzo & Ackermann (2 Grafici)"
        ])
        self.combo_kin_view.currentIndexChanged.connect(self.plot_kinematic_maps)
        kin_top_row.addWidget(self.combo_kin_view)

        kin_top_row.addStretch()
        self.tab_kinematics_layout.addLayout(kin_top_row)

        # Row 2: Manual gain inputs and wheel travel
        kin_controls = QHBoxLayout()
        kin_controls.addWidget(QLabel("<b>Corsa z [mm]:</b>"))
        self.kin_zmax_input = QLineEdit("35.0")
        self.kin_zmax_input.setMaximumWidth(45)
        kin_controls.addWidget(self.kin_zmax_input)
        
        kin_controls.addWidget(QLabel("<b>Gain Camber Ant [deg/m]:</b>"))
        self.kin_cam_gain_f_input = QLineEdit("-15.0")
        self.kin_cam_gain_f_input.setMaximumWidth(50)
        kin_controls.addWidget(self.kin_cam_gain_f_input)
        
        kin_controls.addWidget(QLabel("<b>Post [deg/m]:</b>"))
        self.kin_cam_gain_r_input = QLineEdit("-12.0")
        self.kin_cam_gain_r_input.setMaximumWidth(50)
        kin_controls.addWidget(self.kin_cam_gain_r_input)
        
        kin_controls.addWidget(QLabel("<b>Gain Toe Ant [deg/m]:</b>"))
        self.kin_toe_gain_f_input = QLineEdit("0.5")
        self.kin_toe_gain_f_input.setMaximumWidth(45)
        kin_controls.addWidget(self.kin_toe_gain_f_input)
        
        kin_controls.addWidget(QLabel("<b>Post [deg/m]:</b>"))
        self.kin_toe_gain_r_input = QLineEdit("-0.8")
        self.kin_toe_gain_r_input.setMaximumWidth(45)
        kin_controls.addWidget(self.kin_toe_gain_r_input)
        
        self.btn_apply_kin_gains = QPushButton("[OK] Applica Gain")
        self.btn_apply_kin_gains.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_apply_kin_gains.clicked.connect(self.apply_kinematic_gains_from_ui)
        kin_controls.addWidget(self.btn_apply_kin_gains)

        self.fig_kinematics = Figure(figsize=(10, 7), dpi=100)
        self.canvas_kinematics = FigureCanvas(self.fig_kinematics)
        self.setup_popout(self.canvas_kinematics, "Mappe Cinematiche vs Wheel Travel (Digitizer)", kin_controls)
        kin_controls.addStretch()
        self.tab_kinematics_layout.addLayout(kin_controls)

        # Status Banner
        self.lbl_kin_status_banner = QLabel("🟢 <b>Caricamento stato cinematica...</b>")
        self.lbl_kin_status_banner.setStyleSheet("""
            QLabel {
                background-color: #2f3640;
                color: #81ecec;
                border: 1px solid #00cec9;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
            }
        """)
        self.tab_kinematics_layout.addWidget(self.lbl_kin_status_banner)
        
        self.toolbar_kinematics = NavigationToolbar(self.canvas_kinematics, self.tab_kinematics)
        self.tab_kinematics_layout.addWidget(self.toolbar_kinematics)
        self.tab_kinematics_layout.addWidget(self.canvas_kinematics)
        
        self.setup_subtabs.addTab(self.tab_kinematics, " Mappe Cinematiche vs Wheel Travel (Digitizer)")
        
        # =====================================================================
        # TAB: SENSITIVITY & FSAE 2026 EVENT SCORING
        # =====================================================================
        self.tab_sensitivity = QWidget()
        self.tab_sensitivity_layout = QVBoxLayout(self.tab_sensitivity)
        
        sens_header_bar = QHBoxLayout()
        lbl_sens_title = QLabel(" Analisi di Sensitività Parametrica & Simulatore Punteggi Evento FS 2026 (FS-Rules v1.1)")
        lbl_sens_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2d3436;")
        sens_header_bar.addWidget(lbl_sens_title)
        sens_header_bar.addStretch()
        self.tab_sensitivity_layout.addLayout(sens_header_bar)
        
        self.sens_subtabs = QTabWidget()
        self.tab_sensitivity_layout.addWidget(self.sens_subtabs)
        
        # -------------------------------------------------------------
        # Sub-Tab 1: 1D Sweep & Tornado Chart
        # -------------------------------------------------------------
        self.tab_sens_1d = QWidget()
        self.tab_sens_1d_layout = QVBoxLayout(self.tab_sens_1d)
        
        sens_ctrl_box = QGroupBox("Parametri di Controllo Sensitività")
        sens_ctrl_layout = QGridLayout(sens_ctrl_box)
        
        sens_ctrl_layout.addWidget(QLabel("Parametro di Progetto:"), 0, 0)
        self.sens_param_combo = QComboBox()
        sens_params_list = [
            ("Rapporto Finale (Final Drive)", "final_drive"),
            ("Massa Totale m [kg]", "m"),
            ("Ripartizione Masse Ant. wd_f [-]", "wd_f"),
            ("Bilanciamento Aero Ant. aero_balance [-]", "aero_balance"),
            ("Coefficiente Deportanza Cz [-]", "Cz"),
            ("Coefficiente Resistenza Cd [-]", "Cd"),
            ("Rigidezza Rollio Ant. K_phi_f [Nm/rad]", "K_phi_f"),
            ("Rigidezza Rollio Post. K_phi_r [Nm/rad]", "K_phi_r"),
            ("Ripartizione Rigidezza Rollio Ant. (LLTD) [-]", "lltd"),
            ("Rigidezza Rollio Totale K_phi_tot [Nm/rad]", "roll_stiffness_tot"),
            ("Centro Rollio Ant. q1 [m]", "q1"),
            ("Centro Rollio Post. q2 [m]", "q2"),
            ("Camber Statico Ant. [deg]", "camber_f_static"),
            ("Camber Statico Post. [deg]", "camber_r_static"),
            ("Toe Statico Ant. [deg]", "toe_f_static"),
            ("Toe Statico Post. [deg]", "toe_r_static"),
            ("Fattore Aderenza Laterale LMUY [-]", "LMUY"),
            ("Fattore Aderenza Longitudinale LMUX [-]", "LMUX"),
            ("Potenza Massima P_max [kW]", "P_max_kw"),
            ("Coppia Massima Motore T_max [Nm]", "T_max_motor"),
        ]
        for lbl, key in sens_params_list:
            self.sens_param_combo.addItem(lbl, key)
        sens_ctrl_layout.addWidget(self.sens_param_combo, 0, 1)
        
        sens_ctrl_layout.addWidget(QLabel("Metrica Obiettivo:"), 0, 2)
        self.sens_metric_combo = QComboBox()
        sens_metrics_list = [
            (" Punteggio Totale Dinamico FS 2026 [0-525 pt]", "total_fsae_points"),
            (" Grip Max ay Alta Velocità (20 m/s) [g]", "ay_peak_high_speed"),
            (" Grip Max ay Bassa Velocità (10 m/s) [g]", "ay_peak_low_speed"),
            (" Laptime Autocross [s]", "autox_time"),
            (" Tempo Skidpad (R=9.125m) [s]", "skidpad_time"),
            (" Tempo 0-75m Accelerazione [s]", "accel_time"),
            (" Velocità Uscita 75m [km/h]", "accel_vmax_kmh"),
            (" Grip Massimo Skidpad ay [g]", "skidpad_ay_peak"),
            (" Tempo Endurance 22km [s]", "endurance_time"),
            (" Consumo Energetico [kWh]", "energy_kwh"),
            (" Gradiente Sottosterzo Kus [deg/g]", "understeer_kus"),
        ]
        for lbl, key in sens_metrics_list:
            self.sens_metric_combo.addItem(lbl, key)
        sens_ctrl_layout.addWidget(self.sens_metric_combo, 0, 3)
        
        sens_ctrl_layout.addWidget(QLabel("Variazione ±%:"), 1, 0)
        self.sens_delta_pct_input = QLineEdit("20.0")
        self.sens_delta_pct_input.setMaximumWidth(60)
        sens_ctrl_layout.addWidget(self.sens_delta_pct_input, 1, 1)
        
        sens_ctrl_layout.addWidget(QLabel("Punti Sweep:"), 1, 2)
        self.sens_points_input = QLineEdit("11")
        self.sens_points_input.setMaximumWidth(60)
        sens_ctrl_layout.addWidget(self.sens_points_input, 1, 3)
        
        sens_btns = QHBoxLayout()
        self.btn_run_sens_1d = QPushButton(" Esegui Sweep 1D")
        self.btn_run_sens_1d.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 6px 12px; border-radius: 4px;")
        self.btn_run_sens_1d.clicked.connect(self.run_sensitivity_1d_gui)
        sens_btns.addWidget(self.btn_run_sens_1d)
        
        self.btn_run_sens_tornado = QPushButton(" Genera Diagramma Tornado (Top 8 Parametri ±10%)")
        self.btn_run_sens_tornado.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 6px 12px; border-radius: 4px;")
        self.btn_run_sens_tornado.clicked.connect(self.run_sensitivity_tornado_gui)
        sens_btns.addWidget(self.btn_run_sens_tornado)
        
        sens_ctrl_layout.addLayout(sens_btns, 0, 4, 2, 1)
        self.tab_sens_1d_layout.addWidget(sens_ctrl_box)
        
        self.fig_sens = Figure(figsize=(12, 6), dpi=100)
        self.canvas_sens = FigureCanvas(self.fig_sens)
        self.axes_sens_1d = self.fig_sens.add_subplot(121)
        self.axes_sens_tornado = self.fig_sens.add_subplot(122)
        self.toolbar_sens = NavigationToolbar(self.canvas_sens, self.tab_sens_1d)
        self.tab_sens_1d_layout.addWidget(self.toolbar_sens)
        self.tab_sens_1d_layout.addWidget(self.canvas_sens)
        self.sens_subtabs.addTab(self.tab_sens_1d, " Sensitività Parametrica (1D & Tornado)")
        
        # -------------------------------------------------------------
        # Sub-Tab 2: FSAE 2026 Event Scoring Simulator
        # -------------------------------------------------------------
        self.tab_sens_scoring = QWidget()
        self.tab_sens_scoring_layout = QVBoxLayout(self.tab_sens_scoring)
        
        score_ctrl_box = QGroupBox("Configurazione Evento & Benchmark Formula Student")
        score_ctrl_layout = QGridLayout(score_ctrl_box)
        
        score_ctrl_layout.addWidget(QLabel("Evento di Riferimento (FS-World Ranking):"), 0, 0)
        self.fsae_bench_combo = QComboBox()
        self.fsae_bench_combo.addItem("FS Germany (FSG Hockenheim)", "FS_Germany")
        self.fsae_bench_combo.addItem("FS Austria (FSA Red Bull Ring)", "FS_Austria")
        self.fsae_bench_combo.addItem("FS Italy (FSI Varano)", "FS_Italy")
        self.fsae_bench_combo.addItem("FS East (Hungaroring)", "FS_East")
        self.fsae_bench_combo.addItem("Personalizzato (Custom Benchmark)", "Custom")
        self.fsae_bench_combo.currentIndexChanged.connect(self.on_fsae_benchmark_changed)
        score_ctrl_layout.addWidget(self.fsae_bench_combo, 0, 1)
        
        score_ctrl_layout.addWidget(QLabel("T_min Accel [s]:"), 0, 2)
        self.bench_taccel_input = QLineEdit("3.35")
        self.bench_taccel_input.setMaximumWidth(55)
        score_ctrl_layout.addWidget(self.bench_taccel_input, 0, 3)
        
        score_ctrl_layout.addWidget(QLabel("T_min Skidpad [s]:"), 0, 4)
        self.bench_tskid_input = QLineEdit("4.75")
        self.bench_tskid_input.setMaximumWidth(55)
        score_ctrl_layout.addWidget(self.bench_tskid_input, 0, 5)
        
        score_ctrl_layout.addWidget(QLabel("T_min Autocross [s]:"), 1, 0)
        self.bench_tautox_input = QLineEdit("53.50")
        self.bench_tautox_input.setMaximumWidth(55)
        score_ctrl_layout.addWidget(self.bench_tautox_input, 1, 1)
        
        score_ctrl_layout.addWidget(QLabel("T_min Endurance [s]:"), 1, 2)
        self.bench_tendur_input = QLineEdit("1320.0")
        self.bench_tendur_input.setMaximumWidth(65)
        score_ctrl_layout.addWidget(self.bench_tendur_input, 1, 3)
        
        score_ctrl_layout.addWidget(QLabel("E_min Consumo [kWh]:"), 1, 4)
        self.bench_energy_input = QLineEdit("2.85")
        self.bench_energy_input.setMaximumWidth(55)
        score_ctrl_layout.addWidget(self.bench_energy_input, 1, 5)
        
        self.btn_calc_fsae_scores = QPushButton(" Calcola Punteggi Ufficiali FS 2026")
        self.btn_calc_fsae_scores.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 6px 14px; border-radius: 4px; font-size: 11px;")
        self.btn_calc_fsae_scores.clicked.connect(self.run_fsae_scoring_gui)
        score_ctrl_layout.addWidget(self.btn_calc_fsae_scores, 0, 6, 2, 1)
        self.tab_sens_scoring_layout.addWidget(score_ctrl_box)
        
        # KPI Badges
        kpi_scores_box = QHBoxLayout()
        self.lbl_score_total = QLabel("Punteggio Totale: -- / 525.0 pt")
        self.lbl_score_total.setStyleSheet("background-color: #2d3436; color: #00d2d3; font-size: 13px; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_total)
        
        self.lbl_score_accel = QLabel("Accel: --/50")
        self.lbl_score_accel.setStyleSheet("background-color: #dfe4ea; font-size: 11px; font-weight: bold; padding: 6px 8px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_accel)
        
        self.lbl_score_skid = QLabel("Skidpad: --/50")
        self.lbl_score_skid.setStyleSheet("background-color: #dfe4ea; font-size: 11px; font-weight: bold; padding: 6px 8px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_skid)
        
        self.lbl_score_autox = QLabel("AutoX: --/100")
        self.lbl_score_autox.setStyleSheet("background-color: #dfe4ea; font-size: 11px; font-weight: bold; padding: 6px 8px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_autox)
        
        self.lbl_score_endur = QLabel("Endurance: --/250")
        self.lbl_score_endur.setStyleSheet("background-color: #dfe4ea; font-size: 11px; font-weight: bold; padding: 6px 8px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_endur)
        
        self.lbl_score_eff = QLabel("Efficiency: --/75")
        self.lbl_score_eff.setStyleSheet("background-color: #dfe4ea; font-size: 11px; font-weight: bold; padding: 6px 8px; border-radius: 4px;")
        kpi_scores_box.addWidget(self.lbl_score_eff)
        self.tab_sens_scoring_layout.addLayout(kpi_scores_box)
        
        self.fig_scores = Figure(figsize=(12, 6), dpi=100)
        self.canvas_scores = FigureCanvas(self.fig_scores)
        self.axes_score_radar = self.fig_scores.add_subplot(121, polar=True)
        self.axes_score_bars = self.fig_scores.add_subplot(122)
        self.toolbar_scores = NavigationToolbar(self.canvas_scores, self.tab_sens_scoring)
        self.tab_sens_scoring_layout.addWidget(self.toolbar_scores)
        self.tab_sens_scoring_layout.addWidget(self.canvas_scores)
        self.sens_subtabs.addTab(self.tab_sens_scoring, " Simulatore Punteggi Evento FS 2026")
        
        self.tabs.addTab(self.tab_sensitivity, " Sensitività & Punteggi FSAE")
        
        self.last_sensitivity_res = None
        self.sens_active_worker = None
        self._on_sensitivity_scenario_changed()
        self._update_stability_card()
        self.plot_kinematic_maps()
        
        self.tabs.addTab(self.tab_setup_tools, " Setup Tools (101-Tips)")
        
        # Select and apply default vehicle profile (et18_2026_kinematics.json preferred)
        et18_idx = self.combo_setup_profile.findText("et18_2026_kinematics.json")
        if et18_idx >= 0:
            self.combo_setup_profile.setCurrentIndex(et18_idx)
            self.on_setup_profile_selected(et18_idx)
        else:
            def_idx = self.combo_setup_profile.findText("default_vehicle.json")
            if def_idx >= 0:
                self.combo_setup_profile.setCurrentIndex(def_idx)
                self.on_setup_profile_selected(def_idx)

        self.log("Ready. Press 'Run Global Optimization' to start.")

    def log(self, text):
        self.log_output.append(text)
        self.log_output.moveCursor(QtGui.QTextCursor.End) if hasattr(self, 'QtGui') else None

    def populate_tir_files(self):
        base_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        candidates = [
            os.path.join(repo_root, 'data', 'tire_tir'),
            os.path.join(repo_root, 'tire_tir'),
            os.path.join(os.getcwd(), 'data', 'tire_tir'),
            os.path.join(os.getcwd(), 'tire_tir'),
            os.path.join(base_dir, 'tire_tir')
        ]
        tir_dir = None
        for c in candidates:
            if os.path.isdir(c):
                tir_dir = c
                break
        default_tir = "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir"
        if tir_dir and os.path.exists(tir_dir):
            for file in sorted(os.listdir(tir_dir)):
                if file.endswith('.tir'):
                    self.tir_combo.addItem(file)
            for i in range(self.tir_combo.count()):
                if self.tir_combo.itemText(i).lower() == default_tir.lower():
                    self.tir_combo.setCurrentIndex(i)
                    break

    def on_model_changed(self):
        if hasattr(self, 'params_scroll'):
            self._rebuild_params_panel()
            self.load_equations()

    def load_equations(self):
        """Loads and displays the markdown formulations and guide for the selected model."""
        from src.models.params_vehicle import MODEL_GUIDE_TEXT
        idx = self.model_combo.currentIndex()
        if idx == 1:
            from src.models.vehicle_7dof import Vehicle7DOF
            md = Vehicle7DOF.get_equations_markdown()
        elif idx == 2:
            try:
                from src.models.vehicle_14dof import Vehicle14DOF
                md = Vehicle14DOF.get_equations_markdown()
            except ImportError:
                md = "### Modello 14-DOF Full Vehicle"
        else:
            from src.models.vehicle_3dof import Vehicle3DOF
            md = Vehicle3DOF.get_equations_markdown()
            
        if hasattr(self, 'guide_browser'):
            if hasattr(self.guide_browser, 'setMarkdown'):
                self.guide_browser.setMarkdown(md)
            else:
                self.guide_browser.setText(md)

    def _rebuild_params_panel(self):
        """Dynamically rebuild the optimizer params panel for the current model or all parameters."""
        from src.models.params_vehicle import MODEL_PARAM_CATALOGUE, MODEL_GUIDE_TEXT, VehicleParams
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        idx = self.model_combo.currentIndex()
        model_key = model_names[idx] if idx < len(model_names) else "3-DOF"
        
        view_all = False
        if hasattr(self, 'combo_param_view'):
            view_all = (self.combo_param_view.currentIndex() == 1)
            
        catalogue_key = "ALL" if view_all else model_key
        catalogue = MODEL_PARAM_CATALOGUE.get(catalogue_key, MODEL_PARAM_CATALOGUE.get("ALL", {}))
        
        # Preserve any currently entered values before clearing
        current_entered = {}
        if hasattr(self, 'bounds_inputs') and self.bounds_inputs:
            for k, itm in self.bounds_inputs.items():
                if itm and not itm.get('digitizer') and itm.get('val'):
                    try:
                        current_entered[k] = float(itm['val'].text())
                    except Exception:
                        pass
                        
        vp_defaults = self._get_current_vehicle_params()
        for k, v in current_entered.items():
            if hasattr(vp_defaults, k):
                setattr(vp_defaults, k, v)
                
        is_all_mode = view_all

        # Clear previous widget
        old_widget = self.params_scroll.widget()
        if old_widget:
            old_widget.deleteLater()

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        self.bounds_inputs = {}

        # Default bounds aligned with ET-18 prototype and physical chassis limits
        DEFAULT_BOUNDS = {
            'm':              (200.0, 400.0),
            'L':              (1.525, 1.550),
            't_f':            (1.220, 1.280),
            't_r':            (1.110, 1.160),
            'wd_f':           (0.35,  0.65),
            'lltd':           (0.35,  0.65),
            'aero_balance':   (0.35,  0.65),
            'final_drive':    (2.0,   6.5),
            'P_max_kw':       (20.0,  80.0),
            'T_max_motor':    (50.0,  250.0),
            'camber_f_static':(-3.5,  0.0),
            'camber_r_static':(-3.0,  0.0),
            'toe_f_static':   (-0.5,  0.5),
            'toe_r_static':   (-0.5,  0.5),
            'LMUX':           (0.80,  1.20),
            'LMUY':           (0.80,  1.20),
            'q1':             (0.005, 0.150),
            'q2':             (0.005, 0.180),
            'K_phi_f':        (15000., 60000.),
            'K_phi_r':        (15000., 60000.),
            'K_z_f':          (20000., 60000.),
            'K_z_r':          (20000., 60000.),
            'crr':            (0.005, 0.035),
        }

        for cat_name, params_list in catalogue.items():
            grp = QGroupBox(cat_name)
            gl   = QGridLayout(grp)
            is_digitizer_cat = 'Digitizer' in cat_name or 'Polinomiali' in cat_name

            if not is_digitizer_cat:
                gl.addWidget(QLabel("Opt"),        0, 0)
                gl.addWidget(QLabel("Parametro"),  0, 1)
                gl.addWidget(QLabel("Init / Val"), 0, 2)
                gl.addWidget(QLabel("Min"),        0, 3)
                gl.addWidget(QLabel("Max"),        0, 4)
                gl.addWidget(QLabel("Ottimizzato"), 0, 5)

            for i, (label, attr, unit, _default) in enumerate(params_list, start=1):
                if is_digitizer_cat:
                    # Read-only display row for kinematic map coefficients
                    gl.addWidget(QLabel(label), i, 0)
                    coeff_val = getattr(vp_defaults, attr, None)
                    coeff_str = str(coeff_val) if coeff_val is not None else "[not loaded]"
                    lbl_val = QLabel(coeff_str)
                    lbl_val.setStyleSheet("color: #888;")
                    gl.addWidget(lbl_val, i, 1, 1, 3)
                    self.bounds_inputs[attr] = {'opt': None, 'val': lbl_val,
                                                'min': None, 'max': None,
                                                'opt_lbl': None,
                                                'digitizer': True}
                    continue

                default_val = getattr(vp_defaults, attr, _default or 0.0)
                try:
                    num_val = float(default_val)
                    b_min, b_max = DEFAULT_BOUNDS.get(attr, (num_val*0.8, num_val*1.2 if num_val != 0 else 1.0))
                    val_str = f"{num_val:.4g}"
                    min_str = f"{b_min:.4g}"
                    max_str = f"{b_max:.4g}"
                    is_numeric = True
                except (ValueError, TypeError):
                    b_min, b_max = (0.0, 0.0)
                    val_str = str(default_val)
                    min_str = "-"
                    max_str = "-"
                    is_numeric = False

                # Chassis dimensions (L, t_f, t_r) are fixed by default to preserve ET-18 physical monocoque/track
                is_chassis_geom = (attr in ['L', 't_f', 't_r'])
                is_fixed = True if (is_chassis_geom or not is_numeric) else (False if is_all_mode else ('Fixed' in cat_name or 'Uncertainty' in cat_name))

                chk = QCheckBox()
                chk.setChecked(not is_fixed if is_numeric else False)
                chk.setEnabled(is_numeric)

                init_inp = QLineEdit(val_str)
                min_inp  = QLineEdit(min_str)
                max_inp  = QLineEdit(max_str)
                min_inp.setEnabled(is_numeric and (True if is_all_mode else not is_fixed))
                max_inp.setEnabled(is_numeric and (True if is_all_mode else not is_fixed))

                lbl_opt = QLabel("-")
                lbl_opt.setStyleSheet("font-weight: bold; color: #00b894; font-size: 11px;")

                if is_numeric:
                    def _toggle(checked, mn=min_inp, mx=max_inp):
                        mn.setEnabled(checked)
                        mx.setEnabled(checked)
                    chk.toggled.connect(_toggle)

                gl.addWidget(chk,             i, 0)
                gl.addWidget(QLabel(f"{label} [{unit}]"), i, 1)
                gl.addWidget(init_inp,        i, 2)
                gl.addWidget(min_inp,         i, 3)
                gl.addWidget(max_inp,         i, 4)
                gl.addWidget(lbl_opt,         i, 5)

                self.bounds_inputs[attr] = {
                    'opt': chk, 'val': init_inp,
                    'min': min_inp, 'max': max_inp,
                    'opt_lbl': lbl_opt,
                    'digitizer': False
                }

            scroll_layout.addWidget(grp)

        opt_btn_bar = QHBoxLayout()
        self.btn_apply_opt_to_init = QPushButton(" Carica Target Ottimizzati come Valori Iniziali")
        self.btn_apply_opt_to_init.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 5px 10px; border-radius: 4px;")
        self.btn_apply_opt_to_init.clicked.connect(self.apply_optimized_to_initial_inputs)
        opt_btn_bar.addWidget(self.btn_apply_opt_to_init)
        scroll_layout.addLayout(opt_btn_bar)

        scroll_layout.addStretch()
        self.params_scroll.setWidget(scroll_content)

        # Update guide box
        guide = MODEL_GUIDE_TEXT.get(model_key, "")
        if hasattr(self, 'guide_browser'):
            if hasattr(self.guide_browser, 'setMarkdown'):
                self.guide_browser.setMarkdown(guide)
            else:
                self.guide_browser.setText(guide)

    def get_setups_directory(self) -> str:
        """Returns the persistent directory used for saving/loading setup JSON files."""
        # Check potential setups directories (prioritizing repository root over dist)
        candidates = []
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(cur_dir, "..", ".."))
        candidates.append(os.path.join(repo_root, "data", "setups"))
        candidates.append(os.path.join(repo_root, "setups"))
        candidates.append(os.path.join(os.getcwd(), "data", "setups"))
        candidates.append(os.path.join(os.getcwd(), "setups"))
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, "data", "setups"))
            candidates.append(os.path.join(exe_dir, "setups"))
            candidates.append(os.path.join(exe_dir, "_internal", "data", "setups"))
            candidates.append(os.path.join(exe_dir, "_internal", "setups"))
            candidates.append(os.path.abspath(os.path.join(exe_dir, "..", "..", "data", "setups")))
            candidates.append(os.path.abspath(os.path.join(exe_dir, "..", "data", "setups")))
        else:
            candidates.append(os.path.join(cur_dir, "setups"))
            candidates.append(os.path.abspath(os.path.join(cur_dir, "..", "setups")))
        
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                return c
                
        d = candidates[0]
        os.makedirs(d, exist_ok=True)
        return d

    def get_design_suspension_directory(self) -> str:
        """Returns the default directory for digitized suspension curves (works both in dev and compiled standalone exe)."""
        candidates = []
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(cur_dir, "..", ".."))
        candidates.append(os.path.join(repo_root, "data", "design_suspension"))
        candidates.append(os.path.join(repo_root, "design_suspension"))
        candidates.append(os.path.join(os.getcwd(), "data", "design_suspension"))
        candidates.append(os.path.join(os.getcwd(), "design_suspension"))
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, "data", "design_suspension"))
            candidates.append(os.path.join(exe_dir, "design_suspension"))
            candidates.append(os.path.join(exe_dir, "_internal", "data", "design_suspension"))
            candidates.append(os.path.join(exe_dir, "_internal", "design_suspension"))
            candidates.append(os.path.abspath(os.path.join(exe_dir, "..", "data", "design_suspension")))
            candidates.append(os.path.abspath(os.path.join(exe_dir, "..", "..", "data", "design_suspension")))
        else:
            candidates.append(os.path.join(cur_dir, "design_suspension"))
            candidates.append(os.path.abspath(os.path.join(cur_dir, "..", "design_suspension")))
            
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                return c
        return candidates[0]

    def populate_setup_profiles(self):
        """Populates setup JSON profile files from the setups/ directory."""
        if not hasattr(self, 'combo_setup_profile'):
            return
        self.combo_setup_profile.blockSignals(True)
        self.combo_setup_profile.clear()
        
        setups_dir = self.get_setups_directory()
        def_file = os.path.join(setups_dir, "default_vehicle.json")
        if not os.path.exists(def_file):
            from src.models.params_vehicle import VehicleParams
            VehicleParams().save_json(def_file)
            
        json_files = [f for f in os.listdir(setups_dir) if f.endswith('.json')]
        if not json_files:
            from src.models.params_vehicle import VehicleParams
            VehicleParams().save_json(def_file)
            json_files = ["default_vehicle.json"]
            
        for f in sorted(json_files):
            self.combo_setup_profile.addItem(f, os.path.join(setups_dir, f))
            
        # Select et18_2026_kinematics.json by default if present, else default_vehicle.json
        idx = self.combo_setup_profile.findText("et18_2026_kinematics.json")
        if idx < 0:
            idx = self.combo_setup_profile.findText("default_vehicle.json")
        if idx >= 0:
            self.combo_setup_profile.setCurrentIndex(idx)

        self.combo_setup_profile.blockSignals(False)

    def on_setup_profile_selected(self, index):
        """Loads and applies parameters from the selected setup JSON file."""
        path = self.combo_setup_profile.currentData()
        if path and os.path.exists(path):
            try:
                from src.models.params_vehicle import VehicleParams
                vp = VehicleParams.load_json(path)
                self._apply_vp_to_gui(vp)
                self.plot_kinematic_maps()
                prof_name = self.combo_setup_profile.currentText()
                self._active_setup_filename = prof_name
                self._update_kinematics_status_banner(f"Profilo {prof_name}")
                if hasattr(self, 'setup_banner'):
                    self.setup_banner.setText(f" Setup Attivo: {prof_name} (L={vp.L:.3f}m, tf={vp.t_f:.3f}m, tr={vp.t_r:.3f}m, wd_f={vp.wd_f*100:.1f}%, LLTD={vp.lltd*100:.1f}%)")
                if hasattr(self, 'lbl_status_setup'):
                    self.lbl_status_setup.setText(f"Profilo: {prof_name}")
                self.log(f"[SETUP] Profilo '{prof_name}' caricato con successo!")
            except Exception as e:
                self.log(f"[SETUP] Errore caricamento profilo {path}: {e}")

    def save_current_setup_profile(self):
        """Saves current GUI vehicle parameters directly to active profile JSON."""
        path = self.combo_setup_profile.currentData()
        if not path:
            self.save_as_new_setup_profile()
            return
        try:
            vp = self._get_current_vehicle_params()
            vp.save_json(path)
            prof_name = self.combo_setup_profile.currentText()
            QMessageBox.information(self, "Salva Setup", f"Profilo '{prof_name}' salvato con successo in:\n{path}")
            self.log(f"[SETUP] Modifiche salvate con successo sul profilo '{prof_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il profilo: {e}")

    def save_as_new_setup_profile(self):
        """Prompts user to save current parameters into a new .json profile file."""
        setups_dir = self.get_setups_directory()
        path, _ = QFileDialog.getSaveFileName(self, "Salva Nuovo Profilo Parametri", setups_dir, "JSON Setup Files (*.json)")
        if path:
            if not path.endswith('.json'):
                path += '.json'
            try:
                vp = self._get_current_vehicle_params()
                vp.save_json(path)
                self.populate_setup_profiles()
                file_name = os.path.basename(path)
                idx = self.combo_setup_profile.findText(file_name)
                if idx >= 0:
                    self.combo_setup_profile.setCurrentIndex(idx)
                QMessageBox.information(self, "Profilo Salvato", f"Nuovo profilo setup salvato con successo:\n{path}")
                self.log(f"[SETUP] Creato nuovo profilo setup: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file: {e}")

    def browse_load_setup_profile(self):
        """Prompts user to browse and load an external setup JSON profile file."""
        setups_dir = self.get_setups_directory()
        path, _ = QFileDialog.getOpenFileName(self, "Carica Profilo Setup Personalizzato", setups_dir, "JSON Setup Files (*.json);;All Files (*)")
        if path and os.path.exists(path):
            try:
                from src.models.params_vehicle import VehicleParams
                vp = VehicleParams.load_json(path)
                self._apply_vp_to_gui(vp)
                self.plot_kinematic_maps()
                file_name = os.path.basename(path)
                if hasattr(self, 'setup_banner'):
                    self.setup_banner.setText(f" Setup Attivo: {file_name} (L={vp.L:.3f}m, tf={vp.t_f:.3f}m, tr={vp.t_r:.3f}m, wd_f={vp.wd_f*100:.1f}%, LLTD={vp.lltd*100:.1f}%)")
                self.log(f"[SETUP] Profilo personalizzato caricato da {path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Caricamento", f"Impossibile caricare il file: {e}")

    def _apply_vp_to_gui(self, vp):
        """Applies a VehicleParams instance to all GUI table inputs and caches it as active."""
        import copy
        self._active_vehicle_params = copy.deepcopy(vp)
        if not hasattr(self, 'bounds_inputs') or not self.bounds_inputs:
            return
        for k, inp in self.bounds_inputs.items():
            if hasattr(vp, k):
                if inp.get('digitizer') and inp.get('val'):
                    coeff_val = getattr(vp, k, None)
                    if coeff_val is not None:
                        inp['val'].setText(str(coeff_val))
                elif not inp.get('digitizer') and inp.get('val'):
                    v = getattr(vp, k)
                    if isinstance(v, (int, float)):
                        inp['val'].setText(f"{v:.4f}".rstrip('0').rstrip('.'))
        if hasattr(self, 'sync_dyn_params_from_optimizer'):
            self.sync_dyn_params_from_optimizer()
        # Synchronize kinematic inputs if available
        if hasattr(self, 'kin_cam_gain_f_input') and len(getattr(vp, 'camber_gain_f_coeffs', [])) >= 3:
            self.kin_cam_gain_f_input.setText(f"{vp.camber_gain_f_coeffs[2]:.2f}")
        if hasattr(self, 'kin_cam_gain_r_input') and len(getattr(vp, 'camber_gain_r_coeffs', [])) >= 3:
            self.kin_cam_gain_r_input.setText(f"{vp.camber_gain_r_coeffs[2]:.2f}")
        if hasattr(self, 'kin_toe_gain_f_input') and len(getattr(vp, 'toe_gain_f_coeffs', [])) >= 3:
            self.kin_toe_gain_f_input.setText(f"{vp.toe_gain_f_coeffs[2]:.2f}")
        if hasattr(self, 'kin_toe_gain_r_input') and len(getattr(vp, 'toe_gain_r_coeffs', [])) >= 3:
            self.kin_toe_gain_r_input.setText(f"{vp.toe_gain_r_coeffs[2]:.2f}")
        tire_file = getattr(vp, 'tire_file', None)
        if tire_file and hasattr(self, 'tir_combo'):
            for i in range(self.tir_combo.count()):
                if self.tir_combo.itemText(i).lower() == tire_file.lower():
                    self.tir_combo.setCurrentIndex(i)
                    break

    def _update_optimized_params_display(self, opt_dict):
        """Updates the 'Ottimizzato' column in the Optimizer Params table."""
        for attr, val in opt_dict.items():
            item = self.bounds_inputs.get(attr)
            if item and item.get('opt_lbl'):
                if attr in ['wd_f', 'lltd', 'aero_balance']:
                    item['opt_lbl'].setText(f"{val*100:.1f}%")
                else:
                    item['opt_lbl'].setText(f"{val:.3f}")

    def apply_optimized_to_initial_inputs(self):
        """Applies latest optimized parameters back into the table's initial guess inputs."""
        if not hasattr(self, 'last_optimized_params') or not self.last_optimized_params:
            self.log("[Optimizer Params] Nessuna ottimizzazione recente da applicare.")
            return
            
        opt_p = self.last_optimized_params
        for attr, val in opt_p.items():
            item = self.bounds_inputs.get(attr)
            if item and not item.get('digitizer') and item.get('val'):
                item['val'].setText(f"{val:.4g}")
        self.log("[Optimizer Params] Parametri ottimizzati applicati come nuovi valori iniziali!")

    def launch_digitizer(self):
        try:
            from src.utils.curve_digitizer import CurveDigitizerWindow
            self.digitizer_win = CurveDigitizerWindow()
            self.digitizer_win.show()
            self.log("Curve Digitizer launched.")
        except Exception as e:
            self.log(f"Failed to launch Curve Digitizer: {e}")

    def import_digitized_curves_folder(self):
        """Allows user to select a folder containing digitized curve JSONs and imports all curves into current setup."""
        default_dir = self.get_design_suspension_directory()
        folder = QFileDialog.getExistingDirectory(self, "Seleziona Cartella Curve Digitalizzate (es. digitized_2026_curves o digitized_2025_curves)", default_dir)
        if folder:
            try:
                vp = self._get_current_vehicle_params()
                loaded = vp.import_digitized_curves(folder)
                if not loaded:
                    QMessageBox.warning(self, "Nessuna Curva Trovata", f"Nessun file di curve cinematiche riconosciuto in:\n{folder}")
                    return
                self._apply_vp_to_gui(vp)
                self.plot_kinematic_maps()
                folder_name = os.path.basename(folder)
                summary_msg = f"Importate con successo {len(loaded)} curve cinematiche da '{folder_name}':\n\n"
                for cname in sorted(loaded.keys()):
                    summary_msg += f"• {cname}\n"
                QMessageBox.information(self, "Curve Importate", summary_msg)
                self.log(f"[KINEMATICS] Importate {len(loaded)} curve cinematiche da '{folder_name}'!")
            except Exception as e:
                QMessageBox.critical(self, "Errore Importazione Curve", f"Errore durante l'importazione:\n{e}")

    def _get_current_vehicle_params(self):
        """Builds a VehicleParams instance reflecting current GUI inputs, preserving kinematic maps."""
        import copy
        from src.models.params_vehicle import VehicleParams
        if hasattr(self, '_active_vehicle_params') and self._active_vehicle_params is not None:
            vp = copy.deepcopy(self._active_vehicle_params)
        else:
            vp = VehicleParams()
        if hasattr(self, 'bounds_inputs') and self.bounds_inputs:
            for k, itm in self.bounds_inputs.items():
                if itm and not itm.get('digitizer') and itm.get('val'):
                    try:
                        v = float(itm['val'].text())
                        if hasattr(vp, k):
                            setattr(vp, k, v)
                    except (ValueError, TypeError):
                        pass
        return vp

    def run_handling_standalone(self):
        """Plot handling diagram standalone, using current GUI bounds/settings."""
        self.btn_plot_handling.setEnabled(False)
        self.btn_plot_handling.setText("Generating...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText()
            
            # Read current parameter values from bounds inputs, falling back to VehicleParams
            def get_param(attr, default):
                item = self.bounds_inputs.get(attr)
                if item and not item.get('digitizer'):
                    try:
                        return float(item['val'].text())
                    except (ValueError, AttributeError):
                        pass
                return getattr(vp, attr, default)
            
            L        = get_param('L', 1.529)
            t_f      = get_param('t_f', 1.252)
            t_r      = get_param('t_r', 1.136)
            wd_f     = get_param('wd_f', 0.45)
            aero_bal = get_param('aero_balance', 0.485)
            lmuy_mult= get_param('LMUY', getattr(vp, 'LMUY', 1.0))
            v_x      = float(self.v_x_input.text())
            q1       = get_param('q1', getattr(vp, 'q1', 0.046))
            q2       = get_param('q2', getattr(vp, 'q2', 0.077))
            
            camber_f = get_param('camber_f_static', getattr(vp, 'camber_f_static', -1.5))
            camber_r = get_param('camber_r_static', getattr(vp, 'camber_r_static', -1.0))
            toe_f    = get_param('toe_f_static', getattr(vp, 'toe_f_static', 0.0))
            toe_r    = get_param('toe_r_static', getattr(vp, 'toe_r_static', 0.1))
            
            # Check if K_phi_f / K_phi_r are in bounds inputs or active model is 14-DOF
            has_kphi = ('K_phi_f' in self.bounds_inputs and 'K_phi_r' in self.bounds_inputs) or (self.model_combo.currentIndex() >= 2)
            K_phi_f = get_param('K_phi_f', getattr(vp, 'K_phi_f', 8000.0)) if has_kphi else None
            K_phi_r = get_param('K_phi_r', getattr(vp, 'K_phi_r', 6000.0)) if has_kphi else None
            
            if has_kphi and K_phi_f is not None and K_phi_r is not None and (K_phi_f + K_phi_r) > 0:
                vp.K_phi_f = K_phi_f
                vp.K_phi_r = K_phi_r
                vp.q1 = q1
                vp.q2 = q2
                vp.wd_f = wd_f
                vp.L = L
                vp.t_f = t_f
                vp.t_r = t_r
                lltd = vp.get_effective_lltd(ay_g=1.0)
                self.log(f"Handling 14-DOF: wd_f={wd_f*100:.1f}%, K_phi_f={K_phi_f:.0f}, K_phi_r={K_phi_r:.0f} Nm/rad | q1={q1*1000:.1f}mm, q2={q2*1000:.1f}mm | Camber=({camber_f:.1f}°, {camber_r:.1f}°) -> LLTD eff = {lltd*100:.1f}%")
            else:
                lltd = get_param('lltd', 0.50)
                self.log(f"Handling: wd_f={wd_f*100:.1f}%, LLTD={lltd*100:.1f}% | q1={q1*1000:.1f}mm, q2={q2*1000:.1f}mm | Camber=({camber_f:.1f}°, {camber_r:.1f}°)")
                
            self.tabs.setCurrentWidget(self.tab_handling)
            self.plot_handling(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, lmuy_mult,
                              K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                              camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
        except Exception as e:
            import traceback
            self.log(f"Handling diagram error: {traceback.format_exc()}")
        finally:
            self.btn_plot_handling.setEnabled(True)
            self.btn_plot_handling.setText("Plot Handling")

    def run_maps_standalone(self):
        """Plot Map of Achievable Performance (MAPs) standalone, using current GUI bounds/settings."""
        self.btn_plot_maps.setEnabled(False)
        self.btn_plot_maps.setText("Generating...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText()
            
            def get_param(attr, default):
                item = self.bounds_inputs.get(attr)
                if item and not item.get('digitizer'):
                    try:
                        return float(item['val'].text())
                    except (ValueError, AttributeError):
                        pass
                return getattr(vp, attr, default)
            
            L      = get_param('L', 1.529)
            t_f    = get_param('t_f', 1.252)
            t_r    = get_param('t_r', 1.136)
            wd_f   = get_param('wd_f', 0.45)
            lltd   = get_param('lltd', 0.50)
            aero_bal = get_param('aero_balance', 0.485)
            lmuy_mult = get_param('LMUY', getattr(vp, 'LMUY', 1.0))
            v_x    = float(self.v_x_input.text())
            
            self.tabs.setCurrentWidget(self.tab_maps)
            self.plot_maps(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, lmuy_mult)
        except Exception as e:
            import traceback
            self.log(f"MAPs generation error: {traceback.format_exc()}")
        finally:
            self.btn_plot_maps.setEnabled(True)
            self.btn_plot_maps.setText("Plot MAPs")

    def run_mmm_standalone(self):
        """Plot Yaw Moment Diagram (YMD) standalone using current GUI settings."""
        if hasattr(self, 'btn_recalc_ymd'):
            self.btn_recalc_ymd.setEnabled(False)
            self.btn_recalc_ymd.setText("Generating...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
            
            def get_param(attr, default):
                item = self.bounds_inputs.get(attr)
                if item and not item.get('digitizer'):
                    try:
                        return float(item['val'].text())
                    except (ValueError, AttributeError):
                        pass
                return getattr(vp, attr, default)
            
            L = get_param('L', 1.529)
            t_f = get_param('t_f', 1.252)
            t_r = get_param('t_r', 1.136)
            wd_f = get_param('wd_f', 0.45)
            lltd = get_param('lltd', 0.50)
            aero_bal = get_param('aero_balance', 0.485)
            
            try:
                v_x = float(self.ymd_vx_input.text())
            except Exception:
                v_x = float(self.v_x_input.text()) if hasattr(self, 'v_x_input') else 15.0
                
            self.tabs.setCurrentWidget(self.tab_ymd)
            if hasattr(self, 'ymd_subtabs') and hasattr(self, 'tab_ymd_single'):
                self.ymd_subtabs.setCurrentWidget(self.tab_ymd_single)
            self.plot_mmm(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file)
        except Exception as e:
            import traceback
            self.log(f"YMD / MMM error: {traceback.format_exc()}")
        finally:
            if hasattr(self, 'btn_recalc_ymd'):
                self.btn_recalc_ymd.setEnabled(True)
                self.btn_recalc_ymd.setText(" Ricalcola YMD")

    def run_ggv_standalone(self):
        """Generates 3D G-G-V diagram standalone using current parameters."""
        self.btn_run_ggv.setEnabled(False)
        self.btn_run_ggv.setText("Generating...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText()
            
            def get_param(attr, default):
                item = self.bounds_inputs.get(attr)
                if item and not item.get('digitizer'):
                    try:
                        return float(item['val'].text())
                    except (ValueError, AttributeError):
                        pass
                return getattr(vp, attr, default)
            
            L        = get_param('L', 1.529)
            t_f      = get_param('t_f', 1.252)
            t_r      = get_param('t_r', 1.136)
            wd_f     = get_param('wd_f', 0.45)
            lltd     = get_param('lltd', 0.50)
            aero_bal = get_param('aero_balance', 0.485)
            
            base_p = [L, t_f, t_r, wd_f, lltd, aero_bal]
            self.tabs.setCurrentWidget(self.tab_ggv)
            self.plot_ggv(None, wd_f, lltd, aero_bal, L, t_f, t_r, base_params=base_p, tir_file=tir_file)
        except Exception as e:
            import traceback
            self.log(f"GGV Error: {traceback.format_exc()}")
        finally:
            self.btn_run_ggv.setEnabled(True)
            self.btn_run_ggv.setText(" Rigenera G-G-V 3D")

    def show_tv_guide_dialog(self):
        """Displays informative guide about Torque Vectoring in FSAE Electric."""
        html = """
        <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.5; color: #2d3436;">
            <h2 style="color: #0984e3; margin-top: 0;"> Torque Vectoring (TV) in FSAE</h2>
            <p>
                Nelle vetture <b>Formula Student Electric</b> dotate di motori elettrici indipendenti (2WD posteriore o 4WD), 
                il <b>Torque Vectoring</b> consiste nel modulare attivamente la coppia motrice/frenante erogata a ciascuna ruota:
            </p>
            <div style="background-color: #f1f2f6; padding: 10px; border-left: 4px solid #0984e3; margin: 10px 0;">
                <b>Differenziale di Coppia:</b> &Delta;T = T<sub>RR</sub> - T<sub>RL</sub> [Nm]<br>
                <b>Momento d'Imbardata Diretto:</b> M<sub>z,TV</sub> = (&Delta;T &times; i<sub>g</sub> / (2 &times; R<sub>w</sub>)) &times; t<sub>r</sub> [Nm]
            </div>
            <h3>Perché e come si usa nelle prove dinamiche:</h3>
            <ul>
                <li><b>Inserimento in Curva (+TV a favore di sterzo):</b> Assegnando più coppia motrice alla ruota esterna e riducendo quella interna, si genera un momento imbardante che <i>fa ruotare la vettura verso l'interno della curva</i>. Questo elimina il sottosterzo transitorio e riduce drasticamente il tempo di risposta allo sterzo.</li>
                <li><b>Stabilità in Uscita e Scarto Ostacoli (DLC):</b> Una corretta erogazione di TV permette di stabilizzare il retrotreno durante i veloci cambi di direzione (evitando testacoda e pendolo).</li>
                <li><b>Valore impostato nell'interfaccia:</b> Indica la differenza di coppia motrice &Delta;T tra le ruote posteriori (0 Nm = coppia simmetrica / differenziale aperto).</li>
            </ul>
        </div>
        """
        dialog = ChartGuideDialog("Guida: Torque Vectoring (FSAE)", html, self)
        dialog.setMinimumSize(600, 450)
        dialog.exec_()

    def show_simulation_options_dialog(self):
        """Full-featured dialog to configure discretization steps, solver settings, aero, and powertrain."""
        opts = get_simulation_options()
        dialog = QDialog(self)
        dialog.setWindowTitle("Impostazioni e Opzioni di Simulazione (Discretizzazione, Solutori, Aero)")
        dialog.setMinimumSize(650, 580)
        main_layout = QVBoxLayout(dialog)
        
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # --- TAB 1: Discretizzazione & Integrazione ---
        t1 = QWidget()
        l1 = QGridLayout(t1)
        
        inputs = {}
        
        l1.addWidget(QLabel("<b>Passo Integrazione ODE dt [s]:</b>"), 0, 0)
        inputs['dt_ode'] = QLineEdit(str(opts.dt_ode))
        inputs['dt_ode'].setToolTip("Passo temporale Runge-Kutta 4 per ODE 7-DOF e 14-DOF (consigliato: 0.001 - 0.005 s).")
        l1.addWidget(inputs['dt_ode'], 0, 1)
        
        l1.addWidget(QLabel("<b>Durata Manovra Step Steer t_end [s]:</b>"), 1, 0)
        inputs['t_end_step'] = QLineEdit(str(opts.t_end_step))
        l1.addWidget(inputs['t_end_step'], 1, 1)
        
        l1.addWidget(QLabel("<b>Durata Manovra Double Lane Change t_end [s]:</b>"), 2, 0)
        inputs['t_end_dlc'] = QLineEdit(str(opts.t_end_dlc))
        l1.addWidget(inputs['t_end_dlc'], 2, 1)
        
        l1.addWidget(QLabel("<b>Tempo Rampa Sterzata t_ramp [s]:</b>"), 3, 0)
        inputs['t_ramp_steer'] = QLineEdit(str(opts.t_ramp_steer))
        inputs['t_ramp_steer'].setToolTip("Tempo di salita dell'angolo di sterzo (0.0 = gradino istantaneo, 0.05-0.10s = sterzata reale pilota).")
        l1.addWidget(inputs['t_ramp_steer'], 3, 1)
        
        l1.addWidget(QLabel("<b>Discretizzazione Spaziale Pista ds [m]:</b>"), 4, 0)
        inputs['ds_track'] = QLineEdit(str(opts.ds_track))
        inputs['ds_track'].setToolTip("Risoluzione spaziale della traiettoria per AutoX ed Endurance (consigliato: 0.25 - 1.0 m).")
        l1.addWidget(inputs['ds_track'], 4, 1)
        
        l1.addWidget(QLabel("<b>Punti Risoluzione Handling Diagram:</b>"), 5, 0)
        inputs['hd_num_points'] = QLineEdit(str(opts.hd_num_points))
        l1.addWidget(inputs['hd_num_points'], 5, 1)
        
        l1.addWidget(QLabel("<b>Numero Step Speed Sweep:</b>"), 6, 0)
        inputs['speed_sweep_steps'] = QLineEdit(str(opts.speed_sweep_steps))
        l1.addWidget(inputs['speed_sweep_steps'], 6, 1)
        
        cb_maint = QCheckBox("Mantieni Velocità Vx durante manovre (Cruise Control attivo)")
        cb_maint.setChecked(opts.maintain_speed)
        l1.addWidget(cb_maint, 7, 0, 1, 2)
        l1.setRowStretch(8, 1)
        tabs.addTab(t1, " Discretizzazione & Tempi")
        
        # --- TAB 2: Solutori & Tolleranze ---
        t2 = QWidget()
        l2 = QGridLayout(t2)
        
        l2.addWidget(QLabel("<b>Tolleranza Newton Variabili di Stato (xtol):</b>"), 0, 0)
        inputs['newton_xtol'] = QLineEdit(str(opts.newton_xtol))
        l2.addWidget(inputs['newton_xtol'], 0, 1)
        
        l2.addWidget(QLabel("<b>Tolleranza Residui Forze/Momenti (ftol):</b>"), 1, 0)
        inputs['newton_ftol'] = QLineEdit(str(opts.newton_ftol))
        l2.addWidget(inputs['newton_ftol'], 1, 1)
        
        l2.addWidget(QLabel("<b>Max Iterazioni Newton:</b>"), 2, 0)
        inputs['newton_maxiter'] = QLineEdit(str(opts.newton_maxiter))
        l2.addWidget(inputs['newton_maxiter'], 2, 1)
        
        l2.addWidget(QLabel("<b>Passi Bisezione Grip Limite Ay:</b>"), 3, 0)
        inputs['bisection_steps'] = QLineEdit(str(opts.bisection_steps))
        l2.addWidget(inputs['bisection_steps'], 3, 1)
        
        l2.addWidget(QLabel("<b>Limite Angolo di Deriva Corpo &beta; [°]:</b>"), 4, 0)
        inputs['beta_limit_deg'] = QLineEdit(str(opts.beta_limit_deg))
        inputs['beta_limit_deg'].setToolTip("Massimo angolo di assetto fisico ammesso prima di considerare il veicolo in testacoda.")
        l2.addWidget(inputs['beta_limit_deg'], 4, 1)
        
        l2.addWidget(QLabel("<b>Limite Angolo di Sterzo Ruote &delta; [°]:</b>"), 5, 0)
        inputs['delta_limit_deg'] = QLineEdit(str(opts.delta_limit_deg))
        l2.addWidget(inputs['delta_limit_deg'], 5, 1)
        l2.setRowStretch(6, 1)
        tabs.addTab(t2, " Solutori & Tolleranze")
        
        # --- TAB 3: Aerodinamica & Ambiente ---
        t3 = QWidget()
        l3 = QGridLayout(t3)
        
        l3.addWidget(QLabel("<b>Densità dell'Aria &rho; [kg/m³]:</b>"), 0, 0)
        inputs['rho'] = QLineEdit(str(opts.rho))
        l3.addWidget(inputs['rho'], 0, 1)
        
        l3.addWidget(QLabel("<b>Superficie Frontale di Riferimento S [m²]:</b>"), 1, 0)
        inputs['S'] = QLineEdit(str(opts.S))
        l3.addWidget(inputs['S'], 1, 1)
        
        l3.addWidget(QLabel("<b>Coefficiente di Carico Cz (Downforce) [-]:</b>"), 2, 0)
        inputs['Cz'] = QLineEdit(str(opts.Cz))
        l3.addWidget(inputs['Cz'], 2, 1)
        
        l3.addWidget(QLabel("<b>Coefficiente di Resistenza Cd (Drag) [-]:</b>"), 3, 0)
        inputs['Cd'] = QLineEdit(str(opts.Cd))
        l3.addWidget(inputs['Cd'], 3, 1)
        
        cb_aeromap = QCheckBox("Abilita Mappe Aerodinamiche CFD (Sensibilità a Beccheggio e Ride Heights)")
        cb_aeromap.setChecked(opts.use_aeromap)
        l3.addWidget(cb_aeromap, 4, 0, 1, 2)
        l3.setRowStretch(5, 1)
        tabs.addTab(t3, " Aerodinamica & Ambiente")
        
        # --- TAB 4: Powertrain & Torque Vectoring ---
        t4 = QWidget()
        l4 = QGridLayout(t4)
        
        l4.addWidget(QLabel("<b>Potenza Massima Motori P_max [kW]:</b>"), 0, 0)
        inputs['P_max_kw'] = QLineEdit(str(opts.P_max_kw))
        l4.addWidget(inputs['P_max_kw'], 0, 1)
        
        l4.addWidget(QLabel("<b>Coppia Massima Motore T_max [Nm]:</b>"), 1, 0)
        inputs['T_max_motor'] = QLineEdit(str(opts.T_max_motor))
        l4.addWidget(inputs['T_max_motor'], 1, 1)
        
        l4.addWidget(QLabel("<b>Rapporto di Trasmissione Finale i_g:</b>"), 2, 0)
        inputs['final_drive'] = QLineEdit(str(opts.final_drive))
        l4.addWidget(inputs['final_drive'], 2, 1)
        
        l4.addWidget(QLabel("<b>Efficienza Meccanica Trasmissione &eta;:</b>"), 3, 0)
        inputs['n_drivetrain'] = QLineEdit(str(opts.n_drivetrain))
        l4.addWidget(inputs['n_drivetrain'], 3, 1)
        
        l4.addWidget(QLabel("<b>Ripartizione Frenata Anteriore (Brake Bias):</b>"), 4, 0)
        inputs['brake_bias'] = QLineEdit(str(opts.brake_bias))
        l4.addWidget(inputs['brake_bias'], 4, 1)
        
        l4.addWidget(QLabel("<b>Capacità Batteria [kWh]:</b>"), 5, 0)
        inputs['battery_capacity_kwh'] = QLineEdit(str(opts.battery_capacity_kwh))
        l4.addWidget(inputs['battery_capacity_kwh'], 5, 1)
        
        l4.addWidget(QLabel("<b>Max Potenza Rigenerativa [kW]:</b>"), 6, 0)
        inputs['regen_max_kw'] = QLineEdit(str(opts.regen_max_kw))
        l4.addWidget(inputs['regen_max_kw'], 6, 1)
        
        l4.addWidget(QLabel("<b>Coppia Differenziale Max Torque Vectoring &Delta;T [Nm]:</b>"), 7, 0)
        inputs['tv_max_torque'] = QLineEdit(str(opts.tv_max_torque))
        l4.addWidget(inputs['tv_max_torque'], 7, 1)
        l4.setRowStretch(8, 1)
        tabs.addTab(t4, " Powertrain & TV")
        
        # Button bar
        btn_bar = QHBoxLayout()
        btn_save = QPushButton(" Salva e Applica alla Sessione")
        btn_save.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 6px 14px;")
        
        btn_reset = QPushButton("↺ Ripristina Default")
        btn_reset.setStyleSheet("padding: 6px 14px;")
        
        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(dialog.reject)
        
        btn_bar.addWidget(btn_reset)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_save)
        btn_bar.addWidget(btn_close)
        main_layout.addLayout(btn_bar)
        
        def save_action():
            try:
                new_dict = {}
                for k, widget in inputs.items():
                    val_str = widget.text().strip()
                    if '.' in val_str or 'e' in val_str.lower():
                        new_dict[k] = float(val_str)
                    else:
                        try:
                            new_dict[k] = int(val_str)
                        except ValueError:
                            new_dict[k] = float(val_str)
                new_dict['maintain_speed'] = cb_maint.isChecked()
                new_dict['use_aeromap'] = cb_aeromap.isChecked()
                
                new_options = SimulationOptions.from_dict(new_dict)
                update_simulation_options(new_options)
                
                # Sync with GUI inputs
                if hasattr(self, 'step_dt_input'):
                    self.step_dt_input.setText(str(new_options.dt_ode))
                if hasattr(self, 'step_tend_input'):
                    self.step_tend_input.setText(str(new_options.t_end_step))
                if hasattr(self, 'step_tramp_input'):
                    self.step_tramp_input.setText(str(new_options.t_ramp_steer))
                if hasattr(self, 'autox_ds_input'):
                    self.autox_ds_input.setText(str(new_options.ds_track))
                if hasattr(self, 'step_maintain_speed_cb'):
                    self.step_maintain_speed_cb.setChecked(new_options.maintain_speed)
                if hasattr(self, 'hd_points_input'):
                    self.hd_points_input.setText(str(new_options.hd_num_points))
                if hasattr(self, 'v_steps_input'):
                    self.v_steps_input.setText(str(new_options.speed_sweep_steps))
                    
                # Update VehicleParams base aero & powertrain
                vp = self._get_current_vehicle_params()
                vp.rho = new_options.rho
                vp.S = new_options.S
                vp.Cz = new_options.Cz
                vp.Cd = new_options.Cd
                vp.P_max_kw = new_options.P_max_kw
                vp.T_max_motor = new_options.T_max_motor
                vp.final_drive = new_options.final_drive
                vp.n_drivetrain = new_options.n_drivetrain
                vp.brake_bias = new_options.brake_bias
                vp.battery_capacity_kwh = new_options.battery_capacity_kwh
                vp.regen_max_kw = new_options.regen_max_kw
                
                self.log(f"Opzioni di simulazione salvate e applicate con successo! (dt={new_options.dt_ode}s, ds={new_options.ds_track}m, Cz={new_options.Cz})")
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(dialog, "Errore Parametri", f"Valore non valido: {e}")
                
        def reset_action():
            def_opts = SimulationOptions()
            for k, widget in inputs.items():
                widget.setText(str(getattr(def_opts, k, "")))
            cb_maint.setChecked(def_opts.maintain_speed)
            cb_aeromap.setChecked(def_opts.use_aeromap)
            
        btn_save.clicked.connect(save_action)
        btn_reset.clicked.connect(reset_action)
        dialog.exec_()

    def show_options_dialog(self):
        """Dialog to view and edit Vehicle Physical Parameters across model tiers."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Vehicle Physical Parameters Editor")
        dialog.setMinimumSize(600, 720)
        layout = QVBoxLayout(dialog)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        from src.models.params_vehicle import VehicleParams, MODEL_PARAM_CATALOGUE
        from PyQt5.QtCore import Qt
        vp = self._get_current_vehicle_params()
        
        table_cells = {}
        
        for model_tier, categories in MODEL_PARAM_CATALOGUE.items():
            tier_widget = QWidget()
            tier_layout = QVBoxLayout(tier_widget)
            
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(["Categoria", "Parametro Fisico", "Valore Attivo"])
            table.horizontalHeader().setStretchLastSection(True)
            
            row = 0
            for cat_name, param_list in categories.items():
                for display_name, attr, unit, def_val in param_list:
                    table.insertRow(row)
                    
                    item_cat = QTableWidgetItem(cat_name)
                    item_cat.setFlags(item_cat.flags() ^ Qt.ItemIsEditable)
                    table.setItem(row, 0, item_cat)
                    
                    item_name = QTableWidgetItem(f"{display_name} [{unit}]")
                    item_name.setFlags(item_name.flags() ^ Qt.ItemIsEditable)
                    table.setItem(row, 1, item_name)
                    
                    val = getattr(vp, attr, def_val)
                    item_val = QTableWidgetItem(str(val) if val is not None else "-")
                    table.setItem(row, 2, item_val)
                    
                    table_cells[(model_tier, row)] = (attr, item_val)
                    row += 1
                    
            tier_layout.addWidget(table)
            tabs.addTab(tier_widget, model_tier)
            
        btn_bar = QHBoxLayout()
        btn_save = QPushButton(" Salva Modifiche Parametri")
        btn_save.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 6px 14px;")
        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(dialog.reject)
        
        btn_bar.addStretch()
        btn_bar.addWidget(btn_save)
        btn_bar.addWidget(btn_close)
        layout.addLayout(btn_bar)
        
        def save_vehicle_params():
            saved_count = 0
            for (tier, row_idx), (attr, item_val) in table_cells.items():
                txt = item_val.text().strip()
                if txt and txt != "-":
                    try:
                        if txt.startswith('[') and txt.endswith(']'):
                            # Polynomial list
                            val_list = [float(x.strip()) for x in txt[1:-1].split(',')]
                            setattr(vp, attr, val_list)
                        else:
                            val = float(txt)
                            setattr(vp, attr, val)
                        saved_count += 1
                    except Exception:
                        pass
            self.log(f"Parametri fisici del veicolo aggiornati ({saved_count} parametri salvati).")
            dialog.accept()
            
        btn_save.clicked.connect(save_vehicle_params)
        dialog.exec_()

    def update_active_setup_banner(self, text):
        """Updates active setup text across GUI banners."""
        if hasattr(self, 'setup_banner'):
            self.setup_banner.setText(f" Setup Attivo: {text}")
        if hasattr(self, 'dyn_setup_banner'):
            self.dyn_setup_banner.setText(f" Setup Dinamico: {text}")

    def show_optimization_history_dialog(self):
        """Dialog to browse, compare, and reload previous optimization runs."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Storico Ottimizzazioni MDO & Caricamento Setup")
        dialog.setMinimumSize(900, 520)
        layout = QVBoxLayout(dialog)
        
        history = load_history()
        if not history:
            lbl = QLabel("<b>Nessuna ottimizzazione registrata nello storico.</b><br>Esegui un'ottimizzazione globale per salvare automaticamente il setup.")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            btn_close = QPushButton("Chiudi")
            btn_close.clicked.connect(dialog.reject)
            layout.addWidget(btn_close)
            dialog.exec_()
            return
            
        layout.addWidget(QLabel("<b>Seleziona un'ottimizzazione passata per visualizzare i dettagli o applicare il setup:</b>"))
        
        table = QTableWidget(len(history), 9)
        table.setHorizontalHeaderLabels(["ID Run", "Timestamp", "Modello", "Obiettivo", "Vx [m/s]", "Max Ay [g]", "Passo L [m]", "Carreg. tf/tr [m]", "wd_f / LLTD"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        
        for row_idx, item in enumerate(history):
            table.setItem(row_idx, 0, QTableWidgetItem(str(item.get('run_id', ''))))
            table.setItem(row_idx, 1, QTableWidgetItem(str(item.get('timestamp', ''))))
            table.setItem(row_idx, 2, QTableWidgetItem(str(item.get('model_type', ''))))
            table.setItem(row_idx, 3, QTableWidgetItem(str(item.get('objective_mode', ''))))
            table.setItem(row_idx, 4, QTableWidgetItem(f"{item.get('v_x', 0):.1f}"))
            kpis = item.get('kpis', {})
            table.setItem(row_idx, 5, QTableWidgetItem(f"{kpis.get('max_ay_g', 0):.3f}"))
            opt_p = item.get('optimized_params', {})
            table.setItem(row_idx, 6, QTableWidgetItem(f"{opt_p.get('L', 0):.3f}"))
            table.setItem(row_idx, 7, QTableWidgetItem(f"{opt_p.get('t_f', 0):.3f} / {opt_p.get('t_r', 0):.3f}"))
            table.setItem(row_idx, 8, QTableWidgetItem(f"{opt_p.get('wd_f', 0)*100:.1f}% / {opt_p.get('lltd', 0)*100:.1f}%"))
            
        table.selectRow(len(history) - 1)
        layout.addWidget(table)
        
        btn_bar = QHBoxLayout()
        btn_apply = QPushButton("[OK] Applica questo Setup alla Sessione")
        btn_apply.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 6px 14px;")
        
        btn_export = QPushButton(" Esporta Storico in CSV")
        btn_export.setStyleSheet("padding: 6px 14px;")
        
        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(dialog.reject)
        
        btn_bar.addWidget(btn_export)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_apply)
        btn_bar.addWidget(btn_close)
        layout.addLayout(btn_bar)
        
        def apply_selected():
            sel = table.selectedIndexes()
            if not sel:
                return
            row = sel[0].row()
            item = history[row]
            opt_p = item.get('optimized_params', {})
            self.last_optimized_params = opt_p
            
            # Apply to GUI bounds_inputs
            for k, val in opt_p.items():
                if k in self.bounds_inputs and self.bounds_inputs[k].get('val'):
                    self.bounds_inputs[k]['val'].setText(f"{val:.3f}")
                    
            self.sync_dyn_params_from_optimizer()
            max_ay_str = f"{item.get('kpis', {}).get('max_ay_g', 0):.3f}g"
            self.update_active_setup_banner(f"Ripristinato da Run #{item.get('run_id')} ({item.get('model_type')}) - Ay={max_ay_str}")
            self.log(f"[SETUP] Ripristinato Setup da Storico Run #{item.get('run_id')} ({item.get('model_type')}, L={opt_p.get('L'):.3f}m, tf={opt_p.get('t_f'):.3f}m).")
            dialog.accept()
            
        def export_csv():
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(dialog, "Salva Storico Ottimizzazioni CSV", "optimization_history.csv", "CSV Files (*.csv)")
            if path:
                if export_history_to_csv(path):
                    QMessageBox.information(dialog, "Esporta CSV", f"Storico esportato con successo in:\n{path}")
                    
        btn_apply.clicked.connect(apply_selected)
        btn_export.clicked.connect(export_csv)
        dialog.exec_()

    def run_optimization(self):
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        model_idx = self.model_combo.currentIndex()
        model_key = model_names[model_idx] if model_idx < len(model_names) else "3-DOF"

        try:
            penalty = float(self.penalty_input.text())

            # Collect bounds from dynamic panel
            if model_key == "14-DOF":
                PARAM_KEYS = ['L', 't_f', 't_r', 'wd_f', 'lltd', 'aero_balance',
                              'camber_f_static', 'camber_r_static', 'toe_f_static', 'toe_r_static',
                              'K_phi_f', 'K_phi_r', 'K_z_f', 'K_z_r']
            elif model_key == "7-DOF":
                PARAM_KEYS = ['L', 't_f', 't_r', 'wd_f', 'lltd', 'aero_balance',
                              'camber_f_static', 'camber_r_static', 'toe_f_static', 'toe_r_static']
            else:
                PARAM_KEYS = ['L', 't_f', 't_r', 'wd_f', 'lltd', 'aero_balance']

            custom_bounds = []
            initial_guess = []
            curr_vp = self._get_current_vehicle_params()
            for key in PARAM_KEYS:
                item = self.bounds_inputs.get(key)
                if item is None or item.get('digitizer'):
                    val = getattr(curr_vp, key, 0.0)
                    initial_guess.append(float(val))
                    custom_bounds.append((float(val), float(val)))
                    continue
                val = float(item['val'].text())
                initial_guess.append(val)
                if item['opt'] and _safe_is_checked(item['opt']):
                    custom_bounds.append((float(item['min'].text()),
                                          float(item['max'].text())))
                else:
                    custom_bounds.append((val, val))

        except ValueError:
            self.log("Invalid input parameters or bounds!")
            return

        tir_file = self.tir_combo.currentText()
        if not tir_file:
            self.log("Nessun file .tir selezionato!")
            return

        if _safe_is_checked(self.enable_sweep):
            self.run_sweep_optimization(initial_guess, custom_bounds, penalty, tir_file, model_type=model_key)
        else:
            self.run_single_optimization(initial_guess, custom_bounds, penalty, tir_file, model_type=model_key)

    def run_constant_parameters(self):
        """
        Evaluates and generates all target diagrams, MAPs, handling curve, YMD, G-G-V,
        and KPI dashboard at current vehicle parameters without running numerical optimization.
        """
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        model_idx = self.model_combo.currentIndex()
        model_type = model_names[model_idx] if model_idx < len(model_names) else "3-DOF"
        
        try:
            v_x = float(self.v_x_input.text())
        except ValueError:
            v_x = 15.0
            
        penalty = float(self.penalty_input.text()) if hasattr(self, 'penalty_input') and self.penalty_input.text() else 0.0
        tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
        if not tir_file:
            self.log("Nessun file .tir selezionato!")
            return
            
        vp = self._get_current_vehicle_params()
        
        def get_p(attr, default):
            item = self.bounds_inputs.get(attr) if hasattr(self, 'bounds_inputs') else None
            if item and not item.get('digitizer'):
                try:
                    return float(item['val'].text())
                except (ValueError, AttributeError):
                    pass
            return float(getattr(vp, attr, default))
            
        L        = get_p('L', 1.529)
        t_f      = get_p('t_f', 1.252)
        t_r      = get_p('t_r', 1.136)
        wd_f     = get_p('wd_f', 0.45)
        lltd     = get_p('lltd', 0.50)
        aero_bal = get_p('aero_balance', 0.485)
        
        cam_f = get_p('camber_f_static', -1.5)
        cam_r = get_p('camber_r_static', -1.0)
        toe_f = get_p('toe_f_static', 0.0)
        toe_r = get_p('toe_r_static', 0.1)
        
        kphi_f = get_p('K_phi_f', 8000.0)
        kphi_r = get_p('K_phi_r', 6000.0)
        kz_f   = get_p('K_z_f', 18000.0)
        kz_r   = get_p('K_z_r', 16000.0)
        
        current_params = [L, t_f, t_r, wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r, kphi_f, kphi_r, kz_f, kz_r]
        
        self.btn_run_constant.setEnabled(False)
        self.btn_run_constant.setText(" Valutazione in corso...")
        self.log(f"Valutazione globale su Parametri Costanti (Modello: {model_type}, Vx = {v_x:.1f} m/s)...")
        QApplication.processEvents()
        
        try:
            skid_radius = float(self.skid_radius_input.text()) if hasattr(self, 'skid_radius_input') else 9.125
            accel_dist = float(self.accel_dist_input.text()) if hasattr(self, 'accel_dist_input') else 75.0
            tire_unc = float(self.tire_unc_input.text()) if hasattr(self, 'tire_unc_input') and _safe_is_checked(getattr(self, 'robust_opt_cb', None)) else 0.0
            
            from src.mdo.mdo_global_optimizer import GlobalOptimizer
            opt = GlobalOptimizer(
                v_x=v_x, penalty_factor=penalty, tir_file=tir_file,
                objective_mode="max_ay", skidpad_radius=skid_radius,
                accel_distance=accel_dist, tire_uncertainty_pct=tire_unc,
                model_type=model_type
            )
            
            if model_type == "14-DOF":
                eval_p = current_params[:14]
            elif model_type == "7-DOF":
                eval_p = current_params[:10]
            else:
                eval_p = current_params[:6]
                
            max_ay = opt._evaluate_limit_ay(eval_p)
            
            self.last_optimized_params = {
                'L': float(L),
                't_f': float(t_f),
                't_r': float(t_r),
                'wd_f': float(wd_f),
                'lltd': float(lltd),
                'aero_balance': float(aero_bal),
                'camber_f_static': float(cam_f),
                'camber_r_static': float(cam_r),
                'toe_f_static': float(toe_f),
                'toe_r_static': float(toe_r),
                'K_phi_f': float(kphi_f),
                'K_phi_r': float(kphi_r),
                'K_z_f': float(kz_f),
                'K_z_r': float(kz_r),
                'max_ay': float(max_ay)
            }
            
            self._update_optimized_params_display(self.last_optimized_params)
            
            # Plot all target diagrams with active model physics
            self.plot_mmm(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, model_type=model_type)
            self.plot_ggv(opt, wd_f, lltd, aero_bal, L, t_f, t_r, base_params=eval_p, tir_file=tir_file)
            df_handling = self.plot_handling(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file,
                                             K_phi_f=float(kphi_f) if '14' in str(model_type) else None,
                                             K_phi_r=float(kphi_r) if '14' in str(model_type) else None)
            
            if hasattr(self, 'auto_maps_cb') and _safe_is_checked(self.auto_maps_cb):
                self.plot_maps(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file)
                
            self.update_kpi_dashboard(v_x, L, t_f, t_r, wd_f, lltd, aero_bal, max_ay, df_handling, tir_file, base_params=eval_p)
            
            self.plot_steering_torque()
            self.calculate_and_plot_suspension_sizing()
            self.plot_4_ratios_radar(base_params=eval_p)
            self.plot_kinematic_maps()
            
            self.update_active_setup_banner(f"Parametri Costanti ({model_type} - max_ay = {max_ay:.3f}g)")
            self.log(f"--- VALUTAZIONE PARAMETRI COSTANTI COMPLETATA ({model_type}: max_ay={max_ay:.3f}g) ---")
            self.tabs.setCurrentWidget(self.tab_kpi)
            
        except Exception as e:
            import traceback
            self.log(f"Errore durante la valutazione parametri costanti: {traceback.format_exc()}")
        finally:
            self.btn_run_constant.setEnabled(True)
            self.btn_run_constant.setText(" Valuta Parametri Costanti (No Ottimizzazione)")

    def show_chart_guide(self, chart_type):
        GUIDES = {
            "YMD": (
                "Yaw Moment Diagram (YMD / Moment Method) & Stabilità",
                r"""
                <h2>Yaw Moment Diagram (YMD / Moment Method)</h2>
                <p><b>Standard e Riferimenti:</b> Trattazione standard della dinamica del veicolo ad alte prestazioni (SAE International, ISO 4138); analisi d'equilibrio stazionario ad assetto imposto.</p>
                <hr/>
                
                <h3>1. Cosa Rappresenta lo Yaw Moment Diagram (YMD)</h3>
                <p>Lo <b>Yaw Moment Diagram</b> mappa le capacità limite complessive del veicolo nel piano <b>(Accelerazione Laterale $a_y$, Momento d'Imbardata Totale $M_z$)</b>, combinando una griglia incrociata di angoli di sterzo alle ruote $\delta$ e angoli di deriva/assetto del corpo $\beta$ a velocità costante $v_x$.</p>
                
                <h3>2. Come Interpretare la Griglia del Singolo YMD</h3>
                <ul>
                    <li><b>Linea di Trim Stazionario ($M_z = 0$):</b> Rappresenta la condizione di equilibrio puro in curva stazionaria (nessuna accelerazione angolare d'imbardata $\dot{r} = 0$). L'intersezione di ciascuna curva a $\delta = \text{cost}$ con l'asse $M_z = 0$ identifica la massima accelerazione laterale sostenibile stabilmente per quell'angolo di sterzo.</li>
                    <li><b>Autorità di Controllo dello Sterzo ($\partial M_z / \partial \delta$):</b> Corrisponde alla <b>spaziatura verticale</b> tra le curve a $\delta = \text{cost}$.
                        <ul>
                            <li>Ampia spaziatura $\rightarrow$ Lo sterzo è reattivo, incisivo e genera coppia d'imbardata elevata per l'inserimento.</li>
                            <li>Addensamento delle curve a $\delta$ elevato $\rightarrow$ L'avantreno sta saturando l'aderenza (perdita di autorità direzionale / sottosterzo al limite).</li>
                        </ul>
                    </li>
                    <li><b>Stabilità Direzionale Statica ($\partial M_z / \partial \beta$ - Effetto Banderuola):</b> Corrisponde alla <b>pendenza locale delle linee tratteggiate a $\beta = \text{cost}$</b>:
                        <ul>
                            <li><b>Pendenza negativa ($\partial M_z / \partial \beta < 0$):</b> Il veicolo è <i>direzionalmente stabile</i> (la forza laterale posteriore genera un momento raddrizzante che si oppone alla sbandata).</li>
                            <li><b>Pendenza nulla ($\partial M_z / \partial \beta = 0$):</b> Limite neutro di stabilità ad anello aperto.</li>
                            <li><b>Pendenza positiva ($\partial M_z / \partial \beta > 0$):</b> Il veicolo è <i>instabile ad anello aperto</i> e richiede un rapido controsterzo del pilota per evitare il testacoda.</li>
                        </ul>
                    </li>
                    <li><b>Bordi dell'Inviluppo (Ceiling &amp; Floor):</b>
                        <ul>
                            <li><b>Bordo Superiore ($M_z > 0$ max):</b> Massima tendenza sovrasterzante (posteriore saturo, anteriore con grip).</li>
                            <li><b>Bordo Inferiore ($M_z < 0$ min):</b> Massima tendenza sottosterzante (anteriore saturo, posteriore con grip).</li>
                            <li><b>Bordo Destro/Sinistro ($a_y$ max):</b> Limite assoluto di aderenza laterale a 4 ruote sature.</li>
                        </ul>
                    </li>
                </ul>
                
                <hr/>
                
                <h3>3. Interpretazione del Multi-Speed YMD Overlay</h3>
                <p>L'overlay multi-velocità confronta gli inviluppi limite a diverse velocità d'avanzamento ($v_x = 8 \div 25\text{ m/s}$):</p>
                <ul>
                    <li><b>Espansione Laterale ($a_y$):</b> Con l'aumento della velocità, la deportanza aerodinamica ($F_{\text{down}} \propto v_x^2$) schiaccia le 4 gomme a terra aumentando il carico $F_z$, espandendo l'inviluppo verso l'esterno fino a superare 2.0 - 2.5 g.</li>
                    <li><b>Spostamento del Bilanciamento (Aero Balance vs Weight Distribution):</b>
                        <ul>
                            <li>Se il centro di pressione aerodinamico è più arretrato rispetto al baricentro ($\text{AeroBal} < wd_f$), ad alta velocità il retrotreno riceve proporzionalmente più carico: l'inviluppo si restringe in alto ($M_z$) e la vettura diventa più stabile e sottosterzante nei curvoni veloci.</li>
                            <li>Se $\text{AeroBal} > wd_f$, l'auto guadagna autorità all'anteriore ad alta velocità ma diventa più critica al posteriore.</li>
                        </ul>
                    </li>
                    <li><b>Intersezioni con gli Iso-Raggi ($R = 9.1\text{m}, 15\text{m}, 30\text{m}$):</b> Identificano il punto di lavoro operativo della vettura nelle tipiche curve FSAE (Slalom/Hairpin $R \approx 9\text{m}$, Corner medi $R \approx 15\text{m}$, Curve veloci $R \approx 30\text{m}$).</li>
                </ul>
                
                <hr/>
                
                <h3>4. Interpretazione delle Derivate di Stabilità &amp; Controllo vs Velocità</h3>
                <p>I grafici inferiori mostrano l'evoluzione dei parametri linearizzati al variare della velocità $v_x$:</p>
                <ul>
                    <li><b>Autorità di Controllo $N_\delta = \frac{\partial M_z}{\partial \delta} \approx a \cdot C_{\alpha,f}$ [Nm/deg]:</b>
                        <br/>Misura la prontezza della vettura nel generare momento d'imbardata in risposta ad un input volante. È il guadagno primario di manovrabilità nello slalom e nell'inserimento curva.
                    </li>
                    <li><b>Stabilità Direzionale Statica $N_\beta = \frac{\partial M_z}{\partial \beta} \approx -(a C_{\alpha,f} - b C_{\alpha,r})$ [Nm/deg]:</b>
                        <br/>Esprime la reazione naturale della vettura quando si presenta un angolo di assetto/traversone $\beta$. Deve mantenersi negativa ($N_\beta < 0$) per garantire che il veicolo si raddrizzi da solo senza reazioni violente.
                    </li>
                    <li><b>Smorzamento Naturale d'Imbardata $N_r = \frac{\partial M_z}{\partial r} \approx -\frac{a^2 C_{\alpha,f} + b^2 C_{\alpha,r}}{v_x}$ [Nm$\cdot$s/rad]:</b>
                        <br/>Rappresenta la resistenza alla rotazione d'imbardata. Poiché scala inversamente con $v_x$, alle alte velocità lo smorzamento naturale diminuisce, rendendo fondamentale la corretta taratura di barre antirollio, altezze dei centri di rollio e bilanciamento aerodinamico.
                    </li>
                    <li><b>Control Margin $CM = \frac{N_\delta}{|N_\beta|}$:</b>
                        <br/>Rapporto adimensionale tra autorità di sterzo e stabilità di assetto. Un valore bilanciato ($1.0 \div 2.0$) garantisce che il pilota mantenga sempre sufficiente autorità di recupero con lo sterzo anche in caso di sbandata ad alta velocità.
                    </li>
                    <li><b>Frequenza Naturale $\omega_n$ &amp; Smorzamento $\zeta$:</b>
                        <br/>Definiscono il modo rigido d'imbardata/deriva: un target ottimale di $\zeta \in [0.70, 0.95]$ elimina le oscillazioni pendolari post-transitorio senza rendere l'avantreno pigro.
                    </li>
                </ul>
                """
            ),
            "GGV": (
                "Inviluppo Tridimensionale G-G-V",
                r"""
                <h2>Diagramma G-G-V (Friction &amp; Performance Envelope 3D)</h2>
                <p><b>Riferimento:</b> Standard di valutazione prestazionale tridimensionale dell'aderenza (SAE / FSAE Vehicle Dynamics).</p>
                <hr/>
                <h3>1. Cosa Rappresenta il Grafico</h3>
                <p>Il diagramma G-G-V rappresenta la superficie limite di prestazione del veicolo nello spazio tridimensionale <b>(Accelerazione Longitudinale $a_x$, Accelerazione Laterale $a_y$, Velocità d'Avanzamento $v_x$)</b>.</p>
                
                <h3>2. Come Leggere e Interpretare le Regioni</h3>
                <ul>
                    <li><b>Regione a Bassa Velocità ($v_x < 12\text{ m/s}$):</b> Dominata dall'ellisse di attrito pura degli pneumatici $\mu \cdot F_z$. La massima accelerazione laterale e la frenata longitudinale raggiungono valori quasi simmetrici.</li>
                    <li><b>Regione ad Alta Velocità ($v_x > 18-25\text{ m/s}$):</b>
                        <ul>
                            <li><b>Espansione Laterale ($a_y$):</b> La deportanza generata dal pacchetto aerodinamico ($F_{\text{down}} \propto v_x^2$) incrementa il carico verticale $F_z$ sulle 4 ruote, allargando considerevolmente la capacità in curva fino a superare 2.0 - 2.5 g.</li>
                            <li><b>Restringimento Trazione ($a_x > 0$):</b> In trazione, la potenza massima del powertrain ($P_{\max} = 80\text{ kW}$ da regolamento FSAE) limita la forza motrice disponibile secondo $F_x = P_{\max} / v_x$, restringendo l'ellisse nella metà superiore.</li>
                            <li><b>Resistenza Aerodinamica ($F_{\text{drag}}$):</b> Il drag aerodinamico aumenta proporzionalmente a $v_x^2$, riducendo la trazione netta accelerante.</li>
                        </ul>
                    </li>
                </ul>
                """
            ),
            "HANDLING": (
                "Handling Diagram (Formulazione SAE / ISO)",
                r"""
                <h2>Handling Diagram (Formulazione di Manovrabilità Stazionaria)</h2>
                <p><b>Riferimento Teorico:</b> Normative SAE J670e e ISO 4138 sulla dinamica stazionaria del veicolo.</p>
                <hr/>
                <h3>1. Cosa Rappresenta il Grafico</h3>
                <p>Rappresentazione cartesiana standard del comportamento direzionale del veicolo in curva stazionaria:
                <ul>
                    <li><b>Asse Verticale ($Y$):</b> Accelerazione laterale $a_y$ [g].</li>
                    <li><b>Asse Orizzontale ($X$):</b> Differenza di angolo di deriva $(\alpha_f - \alpha_r) = \delta - \frac{L}{R}$ [gradi].</li>
                </ul>
                </p>
                
                <h3>2. Interpretazione Direzionale e Fisica</h3>
                <ul>
                    <li><b>Sottosterzo ($X > 0$ a destra dell'asse $Y$):</b> Per percorrere una curva a velocità crescente occorre aumentare l'angolo di sterzo $\delta$ oltre il valore cinematico $L/R$. L'avantreno lavora con angoli di deriva superiori al retrotreno $(\alpha_f > \alpha_r)$.</li>
                    <li><b>Sovrasterzo ($X < 0$ a sinistra dell'asse $Y$):</b> All'aumentare della velocità occorre ridurre l'angolo volante o controsterzare per mantenere la traiettoria $(\alpha_r > \alpha_f)$.</li>
                    <li><b>Gradiente di Sottosterzo ($K_{us}$):</b> È definito dalla pendenza locale della curva rispetto all'asse verticale:
                        <br/>$$\frac{d(\alpha_f - \alpha_r)}{d a_y} = K_{us}$$
                        Se la curva piega verso destra, $K_{us} > 0$ (sottosterzo progressivo); se piega verso sinistra, $K_{us} < 0$ (sovrasterzo progressivo).
                    </li>
                    <li><b>Raggi Cinematici $L/R$:</b> Le rette tratteggiate uscenti dall'origine rappresentano la curvatura cinematica per ciascuna velocità: $\text{Ray} = \frac{L a_y}{v_x^2}$.</li>
                    <li><b>Limite di Aderenza e Continuazione Arc-Length:</b> Il punto di massimo $a_y$ indica il picco di aderenza. Il metodo numerico della continuazione Arc-Length permette di calcolare la curva oltre il limite senza divergere matematicamente.</li>
                    <li><b>Bande di Incertezza ($\pm \Delta\mu\%$):</b> Le curve tratteggiate visualizzano l'inviluppo di dispersione dovuto a variazioni di aderenza dello pneumatico.</li>
                </ul>
                """
            ),
            "MAPS": (
                "Map of Achievable Performance (MAPs)",
                r"""
                <h2>Suite Completa delle 5 MAPs</h2>
                <p><b>Riferimento:</b> Spazio di stato non-lineare delle manovre e mappe di performance stazionaria.</p>
                <hr/>
                <h3>1. Cosa Rappresentano le MAPs</h3>
                <p>Le <b>Map of Achievable Performance</b> costituiscono la descrizione globale più rigorosa e non-lineare della dinamica del veicolo, suddividendo lo spazio delle manovre in 5 piani coordinati.</p>
                
                <h3>2. Significato Fisico dei 5 Piani</h3>
                <ul>
                    <li><b>1. Piano ($\delta, u$) - Regione di Input:</b> Mostra l'intervallo ammissibile degli angoli di sterzo $\delta$ in funzione della velocità $u$, con curve a raggio costante (iso-$\rho$) e ad accelerazione costante (iso-$a_y$). Il contorno definisce il massimo corridoio di controllabilità.</li>
                    <li><b>2. Piano ($u, a_y$) - Ceiling di Accelerazione:</b> Rappresenta il tetto massimo di $a_y$ erogabile a ciascuna velocità d'avanzamento, evidenziando il contributo della deportanza aerodinamica.</li>
                    <li><b>3. Piano ($\delta, a_y$) - Risposta Angolo Sterzo / $a_y$:</b> Mostra quanto sterzo $\delta$ occorre applicare per produrre una data accelerazione $a_y$, con le curve di assetto del corpo vettura (iso-$\beta$).</li>
                    <li><b>4. Piano ($\rho, \beta$) - Spazio di Stato Output:</b> Mappa la relazione fondamentale tra curvatura di traiettoria $\rho = 1/R$ e angolo di assetto al baricentro $\beta$.</li>
                    <li><b>5. Piano ($\delta, \rho$) - Regione Mista:</b> Visualizza il guadagno di curvatura dello sterzo (steering gain $\partial \rho / \partial \delta$). Una pendenza elevata indica una vettura molto reattiva e agile nell'inserimento.</li>
                </ul>
                """
            ),
            "DNA_MAPS": (
                "Mappe del DNA del Veicolo & Stabilità Modale",
                r"""
                <h2>Mappe del DNA del Veicolo &amp; Stabilità Modale</h2>
                <p><b>Riferimento:</b> Legame analitico fondamentale tra mappe stazionarie e derivate di stabilità dinamiche transitorie (SAE J670e / ISO 4138).</p>
                <hr/>
                <h3>1. Il Collegamento tra Dati Stazionari e Dinamica Transitoria</h3>
                <p>Le derivate di stabilità dinamiche ($Y_\beta, Y_r, N_\beta, N_r$) e di controllo ($Y_\delta, N_\delta$) sono <b>univocamente legate ai gradienti delle mappe stazionarie</b>:</p>
                <ul>
                    <li>$\operatorname{grad} \rho_p = (\partial \rho/\partial a_y, \partial \rho/\partial \delta_v) = (\rho_y, \rho_\delta)$</li>
                    <li>$\operatorname{grad} \beta_p = (\partial \beta/\partial a_y, \partial \beta/\partial \delta_v) = (\beta_y, \beta_\delta)$</li>
                    <li>Determinante del gradiente: $D = \beta_y \rho_\delta - \beta_\delta \rho_y < 0$.</li>
                </ul>
                <hr/>
                <h3>2. Le 6 Mappe del "DNA" del Veicolo ($n_1 \dots n_6$)</h3>
                <p>Il comportamento dinamico nell'intorno di qualsiasi punto di equilibrio è completamente governato da 6 funzioni spaziali nel piano $(\delta_v, a_y)$:</p>
                <ul>
                    <li><b>$n_1(\delta_v, a_y) = 2\zeta\omega_n = -\operatorname{tr}(A)$ [Smorzamento Equivalente]:</b> Definisce la velocità di dissipazione dell'energia oscillatoria.</li>
                    <li><b>$n_2(\delta_v, a_y) = \omega_n^2 = \det(A)$ [Rigidezza Dinamica Equivalente]:</b> Condizione di stabilità ad anello aperto. Se $n_2 > 0$ il veicolo è stabile; se $n_2 < 0$ uno degli autovalori diventa reale positivo innescando un testacoda aperiodico.</li>
                    <li><b>$n_3, n_4$ [Forzamento Dinamico Deriva $F_\beta$]:</b> $n_3$ è il guadagno statico di assetto $\beta$, $n_4 = \hat{Y}_\delta / u$ è il termine impulsivo.</li>
                    <li><b>$n_5, n_6$ [Forzamento Dinamico Imbardata $F_\rho$]:</b> $n_5$ è il guadagno statico di curvatura $\rho$, $n_6 = \hat{N}_\delta / u$ è l'accelerazione angolare iniziale.</li>
                </ul>
                <hr/>
                <h3>3. La Definizione del Gradiente di Sottosterzo</h3>
                <p>Nel formalismo dei gradienti di curvatura, il gradiente di sottosterzo è la derivata parziale dell'angolo di sterzo rispetto all'accelerazione laterale a velocità costante:</p>
                <p>$$K_{us} = \left(\frac{\partial \delta_v}{\partial a_y}\right)_{u = \text{cost}} = -\frac{\rho_y - \frac{1}{u^2}}{\rho_\delta}$$</p>
                <ul>
                    <li><b>Sottosterzante:</b> $\rho_y < \frac{1}{u^2} \iff K_{us} > 0 \iff n_2 = \omega_n^2 > 0$ (La curvatura cresce meno di $1/u^2$, il pilota deve sterzare di più).</li>
                    <li><b>Neutro:</b> $\rho_y = \frac{1}{u^2} \iff K_{us} = 0$.</li>
                    <li><b>Sovrasterzante:</b> $\rho_y > \frac{1}{u^2} \iff K_{us} < 0$ (La curvatura cresce più di $1/u^2$, occorre riallineare o controsterzare).</li>
                    <li><b>Soglia di Instabilità Dinamica:</b> Quando $\rho_y > 0$ e la velocità supera la velocità critica $u > u_{\text{crit}} = \frac{1}{\sqrt{\rho_y}}$, si ha $n_2 < 0$ e la vettura diventa instabile.</li>
                </ul>
                """
            ),
            "TELEMETRY": (
                "Telemetria Prove Dinamiche Formula SAE (Skidpad &amp; Acceleration)",
                r"""
                <h2>Telemetria e Simulazione Eventi Dinamici FSAE</h2>
                <p><b>Riferimento Tecnico:</b> Formula SAE / Formula Student Rules (Events EV/IC).</p>
                <hr/>
                <h3>1. Prova Skidpad ($R=9.125\text{ m}$)</h3>
                <ul>
                    <li><b>Geometria del Circuito:</b> Percorrenza su anello circolare con raggio alla mezzeria $R_0 = 9.125\text{ m}$ (diametro $18.25\text{ m}$). I coni interni sono posizionati a $7.625\text{ m}$ e quelli esterni a $10.625\text{ m}$ (larghezza corsia $3.0\text{ m}$).</li>
                    <li><b>Vincolo di Corridoio:</b> La simulazione calcola e garantisce che l'ingombro di tutte e 4 le ruote rispetti il margine di pista $\pm (1.5 - t_{\max}/2)\text{ m}$.</li>
                    <li><b>Telemetria sul Giro:</b> Traccia per un giro completo di $360^\circ$ i trasferimenti di carico verticale $F_z$ tra ruote interne ed esterne, le forze laterali $F_y$, l'angolo di deriva $\beta$ e l'errore di puntamento $\psi_{\text{err}}$.</li>
                </ul>
                <hr/>
                <h3>2. Prova Acceleration ($75\text{ m}$)</h3>
                <ul>
                    <li><b>Profilo di Partenza da Fermo:</b> Integrazione Runge-Kutta a passo variabile dell'equazione differenziale del moto $m \dot{v} = F_{\text{drive}} - F_{\text{drag}}$.</li>
                    <li><b>Transizione Grip vs Potenza:</b>
                        <ul>
                            <li><b>Fase Iniziale (Grip Limited):</b> La forza motrice è limitata dall'aderenza delle ruote posteriori motrici $F_{\text{grip}} = \mu_x \cdot W_r$, massimizzata dal trasferimento di carico longitudinale $\Delta W = \frac{m a_x h_{cg}}{L}$.</li>
                            <li><b>Fase Finale (Power Limited):</b> Superata la velocità di transizione, la forza di trazione è governata dal limite di potenza massima $F_{\text{motor}} = P_{\max} / v$.</li>
                        </ul>
                    </li>
                </ul>
                """
            ),
            "SWEEP": (
                "Speed Sweep &amp; Trend di Ottimizzazione Multi-Velocità",
                r"""
                <h2>Speed Sweep &amp; Target Ottimali al Variare della Velocità</h2>
                <hr/>
                <h3>1. Cosa Rappresenta il Grafico</h3>
                <p>Mostra come l'ottimizzatore multidimensionale globale SLSQP adatta i target geometrici ed elastici al variare della velocità d'avanzamento $v_x$.</p>
                
                <h3>2. Interpretazione dei Trend Ottimali</h3>
                <ul>
                    <li><b>Ripartizione dei Pesi ($wd_f$):</b> Parametro strutturale fisso o zavorrabile, ottimizzato per garantire il massimo grip combinato.</li>
                    <li><b>Bilancio Aerodinamico ($aero\_bal$):</b> Ad alta velocità, un bilancio aerodinamico avanzato (46-48% ant.) contrasta il sottosterzo indotto dal trasferimento di carico e massimizza l'accelerazione laterale $a_y$.</li>
                    <li><b>Ripartizione Trasferimento di Carico (LLTD):</b> Aumentando la rigidezza antirollio anteriore (LLTD > 50%) si sposta il carico laterale sull'avantreno, aumentando il sottosterzo e la stabilità ad alta velocità.</li>
                </ul>
                """
            ),
            "KPI": (
                "Metriche di Stabilità Direzionale &amp; KPI di Calcolo",
                r"""
                <h2>Metriche di Stabilità Direzionale &amp; KPI di Sintesi</h2>
                <hr/>
                <h3>1. Metriche Fondamentali di Stabilità</h3>
                <ul>
                    <li><b>Gradiente di Sottosterzo ($K_{us}$):</b>
                        <br/>$$K_{us} = \frac{m \cdot g}{L} \left( \frac{b}{C_{\alpha f}} - \frac{a}{C_{\alpha r}} \right)$$
                        Valori tipici FSAE: $+1.0 \dots +3.5^\circ/\text{g}$ per una guida stabile e prevedibile.
                    </li>
                    <li><b>Velocità Critica ($v_{\text{crit}}$):</b>
                        <br/>$$v_{\text{crit}} = \sqrt{\frac{L \cdot g}{-K_{us}}}$$
                        Esiste solo per veicoli sovrasterzanti ($K_{us} < 0$). Oltre questa velocità, il moto rettilineo diventa asintoticamente instabile.
                    </li>
                    <li><b>Velocità Caratteristica ($v_{\text{char}}$):</b>
                        <br/>$$v_{\text{char}} = \sqrt{\frac{L \cdot g}{K_{us}}}$$
                        Per veicoli sottosterzanti ($K_{us} > 0$), è la velocità alla quale l'angolo di sterzo necessario per percorrere un dato raggio raddoppia rispetto al valore cinematico $L/R$.
                    </li>
                </ul>
                """
            ),
            "SETUP_TOOLS": (
                "Guida ai Tool di Setup & Principi di Assetto",
                r"""
                <h2>Guida ai Tool di Setup Avanzato &amp; Bilanciamento</h2>
                <p><b>Riferimento:</b> Principi di ingegneria di pista e dinamica del veicolo per Formula Student e prototipi da corsa.</p>
                <hr/>
                <h3>1. Coppia di Sterzo &amp; Sforzo Pilota</h3>
                <ul>
                    <li><b>Soglia dei 5 Nm:</b> In Formula Student, senza servosterzo, la coppia massima al volante $T_{sw}$ <b>non deve superare i 5.0 Nm</b> (ideale 3.0 - 4.5 Nm). Valori superiori affaticano rapidamente il pilota portando a cali prestazionali ed errori di guida.</li>
                    <li><b>Trail Pneumatico ($t_p$) vs Meccanico ($t_m$):</b> Il trail pneumatico si azzera al picco di aderenza dello pneumatico (perdita di feeling). Il caster fornisce il trail meccanico $t_m$ necessario a mantenere un feedback tattile progressivo.</li>
                </ul>
                <hr/>
                <h3>2. Sizing Sospensioni &amp; Barre Antirollio ARB</h3>
                <ul>
                    <li><b>Frequenze di Ride &amp; Flat Ride:</b> Per un corretto controllo del beccheggio sui dossi, l'asse posteriore deve avere una frequenza naturale leggermente superiore all'anteriore (es. $f_f = 3.0\text{ Hz}, f_r = 3.3\text{ Hz}$) per completare l'oscillazione in fase.</li>
                    <li><b>Raggiungimento del Target LLTD:</b> Le molle da sole generano un LLTD naturale ($\approx 45-48\%$). La barra antirollio anteriore e posteriore colma la differenza per ottenere l'LLTD target impostato dall'ottimizzatore.</li>
                </ul>
                <hr/>
                <h3>3. I 4 Rapporti di Bilanciamento</h3>
                <ul>
                    <li><b>I 4 Pilastri del Grip:</b> 1. Ripartizione Pesi ($wd_f$), 2. Rigidezza Antirollio ($LLTD$), 3. Bilancio Aerodinamico ($aero\_bal$), 4. Rapporto Carreggiate ($t_f/t_r$).</li>
                    <li><b>Regola di Stabilità ad Alta Velocità:</b> Per evitare pericolose transizioni sovrasterzanti nei curvoni veloci, la ripartizione aerodinamica deve essere arretrata rispetto al bilanciamento meccanico: $aero\_bal \le LLTD - 2.0\%$.</li>
                </ul>
                """,
            ),
            "SENSITIVITY_TRANSIENT": (
                "Guida Analisi di Sensibilità Transitoria & Centri di Rollio",
                r"""
                <h2>Analisi di Sensibilità Transitoria &amp; Impatto dei Centri di Rollio</h2>
                <p><b>Riferimenti:</b> Trattazione dei centri di rollio, migrazione cinematica e rigidezza elastica nella dinamica del veicolo.</p>
                <hr/>
                <h3>1. Centri di Rollio: Trasferimento di Carico Elastico vs Geometrico</h3>
                <p>Il trasferimento di carico laterale totale $\Delta F_z$ su ciascun asse si scompone in due contributi fondamentali:</p>
                <ul>
                    <li><b>Contributo Elastico (Roll Moment):</b> Prodotto dal momento ribaltante della massa sospesa $M_{roll} = m a_y (h_{cg} - h_{rc})$. Questo carico passa attraverso le molle e le barre antirollio ($K_{\phi,f}, K_{\phi,r}$) e genera rollio del telaio $\phi$.</li>
                    <li><b>Contributo Geometrico (Direct / Non-Roll Load Transfer):</b> Trasmesso istantaneamente attraverso i braccetti della sospensione direttamente al contatto ruota-suolo:
                    $$\Delta F_{z,geo,f} = \frac{m a_y (b/L) h_{rc,f}}{t_f}$$
                    Aumentare l'altezza del centro di rollio $q_1$ ($h_{rc,f}$) aumenta la reattività istantanea dell'avantreno, riducendo il ritardo di fase nel transitorio ma aumentando il carico diagonale non ammortizzato.</li>
                </ul>
                <hr/>
                <h3>2. Migrazione del Centro di Rollio (Roll Center Migration)</h3>
                <p>Durante la rotazione di rollio $\phi$ e il moto differenziale delle ruote ($z_{left} - z_{right}$), la geometria a quadrilateri articolati fa <b>migrare il centro di rollio sia verticalmente che lateralmente</b>:</p>
                <ul>
                    <li><b>Migrazione Laterale:</b> Il centro di rollio si sposta verso la ruota esterna alla curva. Questo sposta il polo di rotazione, creando un braccio di momento asimmetrico e alterando il LLTD effettivo tra curva a destra e a sinistra.</li>
                    <li><b>Migrazione Verticale:</b> In compressione simmetrica (heave aerodinamico), i centri di rollio si abbassano o si alzano a seconda dell'inclinazione dei bracci, modificando il bilancio a diverse velocità.</li>
                </ul>
                <hr/>
                <h3>3. Come Interpretare i KPI Estratti</h3>
                <ul>
                    <li><b>$LLTD_{peak}$ vs $LLTD_{ss}$:</b> Mostra quanto il bilanciamento di aderenza nel picco transitorio si discosta dall'equilibrio a regime. Se $LLTD_{peak} > LLTD_{ss}$, l'auto tende a inserire con sottosterzo dinamico per poi stabilizzarsi.</li>
                    <li><b>Tempo di Risposta $t_{90}$ &amp; Settling Time:</b> Misura la prontezza di imbardata del veicolo al comando di sterzo. Centri di rollio più alti e $K_\phi$ maggiore riducono $t_{90}$.</li>
                    <li><b>Overshoot di Imbardata ($\%$):</b> Indica l'eccesso di risposta transitoria. Uno smorzamento $C_\phi$ inadeguato provoca rimbalzi e instabilità in rilascio.</li>
                    <li><b>Asimmetria Carichi $\Delta F_z$ Asym ($\%$):</b> Evidenzia l'impatto della migrazione laterale del centro di rollio sulla distribuzione di carico.</li>
                </ul>
                """
            ),
            "TTX25": (
                "Guida alla Taratura Ammortizzatori Öhlins TTX25 MkII (FSAE 4-Way)",
                r"""
                <h2>Guida alla Taratura Ammortizzatori Öhlins TTX25 MkII FSAE</h2>
                <p><b>Riferimento:</b> Öhlins Racing AB, <i>TTX25 MkII FSAE Damper Manual &amp; Dynomometer Response Curves</i>; Linee guida di taratura ammortizzatori da corsa.</p>
                <hr/>
                <h3>1. Architettura Twin-Tube &amp; 4 Regolazioni Indipendenti</h3>
                <p>L'ammortizzatore <b>Öhlins TTX25 MkII</b> è un'unità a doppio tubo senza cavitazione con pistone solido non forato e 4 registri esterni:</p>
                <ul>
                    <li><b>LSC (Low Speed Compression - Allen 3mm):</b> Regola lo spillo di bleed a bassa velocità ($0 \dots 25\text{ mm/s}$). Controlla la velocità di rollio in inserimento e il beccheggio in frenata.
                        <br/><i>Scala:</i> $0\text{ click}$ = tutto chiuso (massimo smorzamento), $25\text{ click}$ = tutto aperto (minimo smorzamento).</li>
                    <li><b>HSC (High Speed Compression - Esagono 12mm):</b> Regola il precarico molla sul pacco lamellare primario. Controlla la risposta ad alta velocità ($> 50\text{ mm/s}$) su cordoli e dossi.
                        <br/><i>Scala:</i> $0\text{ giri}$ = tutto aperto (soft), $4.3\text{ giri}$ = tutto chiuso (stiff).</li>
                    <li><b>LSR (Low Speed Rebound - Allen 3mm):</b> Regola il bleed di estensione a bassa velocità. Cruciale per controllare la riapertura della sospensione interna in uscita curva e stabilizzare il posteriore in staccata.
                        <br/><i>Scala:</i> $0\text{ click}$ = tutto chiuso (stiff), $25\text{ click}$ = tutto aperto (soft).</li>
                    <li><b>HSR (High Speed Rebound - Esagono 12mm):</b> Regola il ritorno rapido dopo l'impatto con un cordolo, evitando che la ruota rimanga sospesa in aria o rimbalzi.
                        <br/><i>Scala:</i> $0\text{ giri}$ = tutto aperto (soft), $4.3\text{ giri}$ = tutto chiuso (stiff).</li>
                </ul>
                <hr/>
                <h3>2. Target di Damping Ratio ($\zeta = c / c_{\text{crit}}$)</h3>
                <ul>
                    <li><b>Low-Speed Chassis Control (Heave &amp; Roll):</b> Target ottimale $\zeta_{\text{LS}} \approx 0.65 \dots 0.85$. Elimina le oscillazioni parassite garantendo transitori rapidi e reattivi.</li>
                    <li><b>High-Speed Compliance (Kerbs &amp; Bumps):</b> Target ottimale $\zeta_{\text{HS}} \approx 0.25 \dots 0.40$ per evitare la perdita di contatto gomma-suolo sui cordoli.</li>
                    <li><b>Rapporto di Asimmetria Rebound/Bump:</b> Tipicamente $\zeta_{\text{reb}} / \zeta_{\text{bump}} \approx 1.5 \dots 2.0$ per controllare il rilascio elastico della molla.</li>
                </ul>
                """
            )
        }
        
        guide_info = GUIDES.get(chart_type)
        if guide_info:
            title, html = guide_info
            dialog = ChartGuideDialog(title, html, parent=self)
            dialog.exec_()
        else:
            QMessageBox.information(self, "Guida", "Nessuna guida disponibile per questo grafico.")

    def show_user_guide_dialog(self):
        """Displays the comprehensive User Guide and Vehicle Dynamics Study Handbook."""
        guide_path = os.path.join(os.path.dirname(__file__), "docs", "USER_GUIDE.md")
        content = ""
        if os.path.exists(guide_path):
            with open(guide_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = "# Manuale Utente\nFile docs/USER_GUIDE.md non trovato."

        dialog = QDialog(self)
        dialog.setWindowTitle(" Manuale Utente & Guida Dinamica del Veicolo — Dynamic Target Suite")
        dialog.resize(950, 750)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top bar
        top = QHBoxLayout()
        lbl = QLabel("<b> Manuale Utente & Studio Dinamica del Veicolo</b>")
        lbl.setStyleSheet("font-size: 13px; color: #2d3436;")
        top.addWidget(lbl)
        top.addStretch()

        btn_fs = QPushButton(" Schermo Intero")
        btn_fs.setStyleSheet("font-weight: bold; padding: 4px 10px; background-color: #0984e3; color: white; border-radius: 3px;")
        def toggle_fs():
            if dialog.isFullScreen():
                dialog.showNormal()
                btn_fs.setText(" Schermo Intero")
            else:
                dialog.showFullScreen()
                btn_fs.setText(" Esci Schermo Intero")
        btn_fs.clicked.connect(toggle_fs)
        top.addWidget(btn_fs)

        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet("padding: 4px 12px; font-weight: bold; border-radius: 3px;")
        btn_close.clicked.connect(dialog.accept)
        top.addWidget(btn_close)
        layout.addLayout(top)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("background-color: #ffffff; color: #2d3436; font-size: 13px; padding: 14px; line-height: 1.4;")
        browser.setMarkdown(content)
        layout.addWidget(browser)

        dialog.exec_()

    def is_english_mode(self) -> bool:
        """Returns True if charts should be rendered in English / Report Mode."""
        cb_val = False
        if hasattr(self, 'lang_en_cb'):
            try:
                cb_val = bool(self.lang_en_cb.isChecked())
            except (RuntimeError, Exception):
                cb_val = False
        act_val = False
        if hasattr(self, 'lang_en_action'):
            try:
                act_val = bool(self.lang_en_action.isChecked())
            except (RuntimeError, Exception):
                act_val = False
        return cb_val or act_val

    def on_language_toggled(self, state=None):
        """Toggles real-time English translation across all open and future charts."""
        is_en = self.is_english_mode()
        try:
            if hasattr(self, 'lang_en_cb') and hasattr(self, 'lang_en_action'):
                if isinstance(state, bool):
                    self.lang_en_cb.setChecked(state)
                elif isinstance(state, int):
                    from PyQt5 import QtCore
                    self.lang_en_action.setChecked(state == QtCore.Qt.Checked)
        except (RuntimeError, Exception):
            pass
        
        self.log(f"Lingua Grafici: {'Inglese (SAE / Report Mode)' if is_en else 'Italiano (Default)'}")

        # Translate all active figures in place
        for canvas_attr in [
            'canvas_handling', 'canvas_mmm', 'canvas_mmm_multi', 'canvas_ggv',
            'canvas_maps_input', 'canvas_maps_output', 'canvas_maps_dna',
            'canvas_steer_torque', 'canvas_susp_sizing', 'canvas_damper_dyno',
            'canvas_step_resp', 'canvas_freq_resp', 'canvas_eigen', 'canvas_sens'
        ]:
            if hasattr(self, canvas_attr):
                c = getattr(self, canvas_attr)
                if hasattr(c, 'figure') and c.figure:
                    translate_figure_language(c.figure, to_english=is_en)
                    c.draw()

    def switch_to_pro_gui(self):
        """Passa all'interfaccia moderna Dynamic Target Suite Pro in-process."""
        try:
            from src.ui import main_gui_pro
            from src.ui import ui_theme_engine
            settings = ui_theme_engine.load_ui_settings()
            settings['preferred_gui'] = 'pro'
            ui_theme_engine.save_ui_settings(settings)
            
            if not hasattr(self, '_pro_win') or self._pro_win is None:
                self._pro_win = main_gui_pro.MainWindowPro()
            self._pro_win.show()
            self._pro_win.raise_()
            self._pro_win.activateWindow()
            self.hide()
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile passare al layout Pro:\n{e}")

    def setup_popout(self, canvas, title, top_bar_layout=None):
        """Adds a popout button to the top bar and connects double-click on canvas to open PopoutPlotDialog."""
        def open_popout():
            dlg = PopoutPlotDialog(canvas, title=title, parent=self)
            dlg.exec_()

        def on_canvas_dblclick(event):
            if event.dblclick:
                open_popout()

        canvas.mpl_connect('button_press_event', on_canvas_dblclick)

        if top_bar_layout is not None:
            btn_pop = QPushButton(" Finestra / Fullscreen")
            btn_pop.setToolTip("Apri questo grafico in una finestra separata ad alta risoluzione o a schermo intero (F11 / doppio clic sul grafico).")
            btn_pop.setStyleSheet("font-size: 10.5px; font-weight: bold; background-color: #dfe6e9; color: #2d3436; padding: 4px 8px; border-radius: 3px;")
            btn_pop.clicked.connect(open_popout)
            top_bar_layout.addWidget(btn_pop)
            return btn_pop
            
    def run_single_optimization(self, initial_guess, custom_bounds, penalty, tir_file, model_type="3-DOF"):
        try:
            v_x = float(self.v_x_input.text())
        except ValueError:
            self.log("Invalid Target Speed!")
            return
            
        obj_idx = self.obj_combo.currentIndex()
        if obj_idx == 1:
            objective_mode = "target_neutral"
        elif obj_idx == 2:
            objective_mode = "min_skidpad"
        elif obj_idx == 3:
            objective_mode = "min_accel"
        else:
            objective_mode = "max_ay"
            
        skid_radius = float(self.skid_radius_input.text())
        accel_dist = float(self.accel_dist_input.text())
        
        self.log(f"Starting SLSQP Global Optimization (Model={model_type}, Objective={objective_mode}, Penalty={penalty}, Tire={tir_file})...")
        QApplication.processEvents()
        
        try:
            tire_unc = float(self.tire_unc_input.text()) if hasattr(self, 'tire_unc_input') and _safe_is_checked(getattr(self, 'robust_opt_cb', None)) else 0.0
            enf_understeer = _safe_is_checked(getattr(self, 'enforce_understeer_cb', None))
            kus_min_val = float(self.kus_min_input.text()) if hasattr(self, 'kus_min_input') else 0.10
            kus_wt_val = float(self.kus_weight_input.text()) if hasattr(self, 'kus_weight_input') else 1.0
            target_kus_val = float(self.kus_target_input.text()) if hasattr(self, 'kus_target_input') and self.kus_target_input.text().strip() else 0.0
            opt = GlobalOptimizer(v_x=v_x, penalty_factor=penalty, tir_file=tir_file, 
                                  objective_mode=objective_mode, skidpad_radius=skid_radius, 
                                  accel_distance=accel_dist, tire_uncertainty_pct=tire_unc,
                                  model_type=model_type, enforce_understeer=enf_understeer,
                                  understeer_min_deg_g=kus_min_val, understeer_weight=kus_wt_val,
                                  target_kus_deg_g=target_kus_val)
            res = opt.run_optimization(initial_guess=initial_guess, custom_bounds=tuple(custom_bounds))
            
            if res.success:
                x_opt = list(res.x)
                L, t_f, t_r, wd_f, lltd, aero_bal = x_opt[:6]
                max_ay = opt._evaluate_limit_ay(x_opt)
                self.log(f"--- OPTIMIZATION SUCCESS ({model_type}) ---")
                
                self.last_optimized_params = {
                    'L': float(L),
                    't_f': float(t_f),
                    't_r': float(t_r),
                    'wd_f': float(wd_f),
                    'lltd': float(lltd),
                    'aero_balance': float(aero_bal),
                    'max_ay': float(max_ay)
                }
                if len(x_opt) >= 10:
                    self.last_optimized_params['camber_f_static'] = float(x_opt[6])
                    self.last_optimized_params['camber_r_static'] = float(x_opt[7])
                    self.last_optimized_params['toe_f_static'] = float(x_opt[8])
                    self.last_optimized_params['toe_r_static'] = float(x_opt[9])
                if len(x_opt) >= 14:
                    self.last_optimized_params['K_phi_f'] = float(x_opt[10])
                    self.last_optimized_params['K_phi_r'] = float(x_opt[11])
                    self.last_optimized_params['K_z_f']   = float(x_opt[12])
                    self.last_optimized_params['K_z_r']   = float(x_opt[13])
                    
                # 1. Update GUI table values with optimized setup
                for k, val in self.last_optimized_params.items():
                    if k in self.bounds_inputs and self.bounds_inputs[k].get('val'):
                        self.bounds_inputs[k]['val'].setText(f"{val:.3f}")
                        
                # 2. Auto-sync with Dynamic Events setup
                self.sync_dyn_params_from_optimizer()
                
                # 3. Save to persistent JSON history
                init_keys = ['L', 't_f', 't_r', 'wd_f', 'lltd', 'aero_balance']
                if len(initial_guess) >= 10:
                    init_keys += ['camber_f_static', 'camber_r_static', 'toe_f_static', 'toe_r_static']
                if len(initial_guess) >= 14:
                    init_keys += ['K_phi_f', 'K_phi_r', 'K_z_f', 'K_z_r']
                init_dict = {k: float(v) for k, v in zip(init_keys, initial_guess)}
                
                run_rec = save_optimization_run(
                    model_type=model_type,
                    objective_mode=objective_mode,
                    v_x=v_x,
                    penalty=penalty,
                    tir_file=tir_file,
                    initial_params=init_dict,
                    optimized_params=self.last_optimized_params,
                    kpis={'max_ay_g': float(max_ay)}
                )
                
                # 4. Update banner status
                self.update_active_setup_banner(f"Ottimizzato (Run #{run_rec['run_id']} - {model_type} - max_ay = {max_ay:.3f}g)")
                self.log(f"[STORICO] Setup salvato nello storico (Run #{run_rec['run_id']}) ed applicato a sessione, grafici e prove dinamiche.")
                
                self._update_optimized_params_display(self.last_optimized_params)
                
                self.plot_mmm(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file)
                self.plot_ggv(opt, wd_f, lltd, aero_bal, L, t_f, t_r, base_params=initial_guess, tir_file=tir_file)
                df_handling = self.plot_handling(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file,
                                                 K_phi_f=self.last_optimized_params.get('K_phi_f') if '14' in str(model_type) else None,
                                                 K_phi_r=self.last_optimized_params.get('K_phi_r') if '14' in str(model_type) else None)
                
                if _safe_is_checked(getattr(self, 'auto_maps_cb', None)):
                    self.plot_maps(v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file)
                else:
                    self.log("[INFO] MAPs complete non generate automaticamente per massimizzare la reattivita'. Clicca 'Genera Suite Completa MAPs' nel tab MAPs.")
                
                self.update_kpi_dashboard(v_x, L, t_f, t_r, wd_f, lltd, aero_bal, max_ay, df_handling, tir_file, base_params=initial_guess)
                
                # Update 101-Tips Setup Tools
                self.plot_steering_torque()
                self.calculate_and_plot_suspension_sizing()
                self.plot_4_ratios_radar(base_params=initial_guess)
                self.plot_kinematic_maps()
                
                self.tabs.setCurrentWidget(self.tab_kpi)
            else:
                self.log("--- OPTIMIZATION FAILED ---")
                self.log(str(res.message))
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.log("--- ERROR DURING OPTIMIZATION ---")
            self.log(err)

    def run_sweep_optimization(self, initial_guess, custom_bounds, penalty, tir_file, model_type="3-DOF"):
        try:
            v_min = float(self.v_min_input.text())
            v_max = float(self.v_max_input.text())
            v_steps = int(self.v_steps_input.text())
        except ValueError:
            self.log("Invalid Sweep Parameters!")
            return
            
        obj_idx = self.obj_combo.currentIndex()
        if obj_idx == 1:
            objective_mode = "target_neutral"
        elif obj_idx == 2:
            objective_mode = "min_skidpad"
        elif obj_idx == 3:
            objective_mode = "min_accel"
        else:
            objective_mode = "max_ay"
            
        skid_radius = float(self.skid_radius_input.text())
        accel_dist = float(self.accel_dist_input.text())
        tire_unc = float(self.tire_unc_input.text()) if hasattr(self, 'tire_unc_input') and _safe_is_checked(getattr(self, 'robust_opt_cb', None)) else 0.0
        enforce_us = _safe_is_checked(getattr(self, 'enforce_understeer_cb', None))
        kus_min_val = float(self.kus_min_input.text()) if hasattr(self, 'kus_min_input') else 0.10
        kus_wt_val = float(self.kus_weight_input.text()) if hasattr(self, 'kus_weight_input') else 1.0
        target_kus_val = float(self.kus_target_input.text()) if hasattr(self, 'kus_target_input') and self.kus_target_input.text().strip() else 0.0
        
        speeds = list(np.linspace(v_min, v_max, v_steps))
        
        self.log(f"Starting Smooth Warm-Started Speed Sweep ({v_min:.1f} to {v_max:.1f} m/s, {v_steps} speeds) with Model {model_type}...")
        QApplication.processEvents()
        
        results = []
        curr_guess = list(initial_guess)
        
        for v_val in speeds:
            task_args = (v_val, curr_guess, custom_bounds, penalty, tir_file, tire_unc, enforce_us, objective_mode, skid_radius, accel_dist, model_type, kus_min_val, kus_wt_val, target_kus_val)
            try:
                res_dict = _worker_optimize_single_speed(task_args)
                if res_dict.get('success', False):
                    self.log(f"  [OK] Vx = {v_val:5.1f} m/s | Max Ay: {res_dict['max_ay']:.3f} g | wd_f: {res_dict['wd_f']*100:.1f}% | LLTD: {res_dict['lltd']*100:.1f}% | Aero: {res_dict['aero_bal']*100:.1f}%")
                    results.append(res_dict)
                    curr_guess = list(res_dict['x'])  # Warm-start next speed from this optimal point
                else:
                    err_info = res_dict.get('error', 'Optimization did not converge')
                    self.log(f"  [X] Vx = {v_val:5.1f} m/s | {err_info}")
            except Exception as exc:
                self.log(f"  [X] Vx = {v_val:5.1f} m/s | Worker error: {exc}")
            QApplication.processEvents()
            
        if not results:
            self.log("Sweep failed to find any successful optimizations.")
            return
            
        # Sort results by velocity ascending
        results = sorted(results, key=lambda d: d['v_x'])
        df = pd.DataFrame(results)
        self.last_sweep_df = df
        self.last_sweep_model_type = model_type
        
        # Populate interactive parameter selector list & render initial plot
        self._init_sweep_param_list(df)
        self.plot_sweep_results(df, model_type)
        
        self.tabs.setCurrentWidget(self.tab_sweep_res)
        self.log("--- SPEED SWEEP OPTIMIZATION COMPLETED SUCCESSFULLY IN PARALLEL ---")

    def _init_sweep_param_list(self, df):
        """Populates the sweep parameters QListWidget with available channels from results DataFrame."""
        if not hasattr(self, 'sweep_param_list') or self.sweep_param_list is None:
            return
        
        SWEEP_PARAM_DEFS = {
            'aero_bal':        ('Aero Bal Front (%)', 100.0, '%', '#0984e3', 'o', 'left'),
            'wd_f':            ('Weight Dist Front (%)', 100.0, '%', '#00cec9', 's', 'left'),
            'lltd':            ('LLTD Front (%)', 100.0, '%', '#e84393', '^', 'left'),
            'max_ay':          ('Max Ay [g]', 1.0, 'g', '#d63031', '^', 'right'),
            'K_us':            ('Gradiente Sottosterzo Kus [deg/g]', 1.0, 'deg/g', '#f39c12', 'd', 'right'),
            'L':               ('Passo L [m]', 1.0, 'm', '#27ae60', 'v', 'left'),
            't_f':             ('Carreggiata Ant tf [m]', 1.0, 'm', '#16a085', '<', 'left'),
            't_r':             ('Carreggiata Post tr [m]', 1.0, 'm', '#2980b9', '>', 'left'),
            'camber_f_static': ('Camber Ant Statico [deg]', 1.0, 'deg', '#8e44ad', 'p', 'left'),
            'camber_r_static': ('Camber Post Statico [deg]', 1.0, 'deg', '#9b59b6', 'h', 'left'),
            'toe_f_static':    ('Toe Ant Statico [deg]', 1.0, 'deg', '#d35400', '*', 'left'),
            'toe_r_static':    ('Toe Post Statico [deg]', 1.0, 'deg', '#e67e22', 'x', 'left'),
            'K_phi_f':         ('Rigidezza Rollio Ant K_phi_f [Nm/rad]', 1.0, 'Nm/rad', '#0984e3', 's', 'left'),
            'K_phi_r':         ('Rigidezza Rollio Post K_phi_r [Nm/rad]', 1.0, 'Nm/rad', '#6c5ce7', '^', 'left'),
            'K_z_f':           ('Rigidezza Molla Ant K_z_f [N/m]', 1.0, 'N/m', '#00cec9', 'o', 'left'),
            'K_z_r':           ('Rigidezza Molla Post K_z_r [N/m]', 1.0, 'N/m', '#00b894', 'D', 'left'),
        }
        
        self.sweep_param_list.blockSignals(True)
        self.sweep_param_list.clear()
        
        default_checked = {'aero_bal', 'wd_f', 'lltd', 'max_ay'}
        
        # Add recognized channels that exist in df
        for col, meta in SWEEP_PARAM_DEFS.items():
            if col in df.columns:
                item = QListWidgetItem(f"{meta[0]}")
                item.setData(Qt.UserRole, col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if col in default_checked else Qt.Unchecked)
                self.sweep_param_list.addItem(item)
                
        # Add any remaining numerical columns in df (excluding v_x and x)
        for col in df.columns:
            if col not in SWEEP_PARAM_DEFS and col not in ['v_x', 'x', 'success', 'error']:
                item = QListWidgetItem(str(col))
                item.setData(Qt.UserRole, col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.sweep_param_list.addItem(item)
                
        self.sweep_param_list.blockSignals(False)

    def _on_sweep_param_item_changed(self, item):
        if hasattr(self, 'last_sweep_df') and self.last_sweep_df is not None:
            self.plot_sweep_results(self.last_sweep_df, getattr(self, 'last_sweep_model_type', '3-DOF'))

    def _reset_sweep_params_default(self):
        if not hasattr(self, 'sweep_param_list') or self.sweep_param_list is None:
            return
        default_keys = {'aero_bal', 'wd_f', 'lltd', 'max_ay'}
        self.sweep_param_list.blockSignals(True)
        for i in range(self.sweep_param_list.count()):
            it = self.sweep_param_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in default_keys else Qt.Unchecked)
        self.sweep_param_list.blockSignals(False)
        self._on_sweep_param_item_changed(None)

    def _select_sweep_params_suspension(self):
        if not hasattr(self, 'sweep_param_list') or self.sweep_param_list is None:
            return
        susp_keys = {'K_phi_f', 'K_phi_r', 'K_z_f', 'K_z_r', 'lltd', 'max_ay'}
        self.sweep_param_list.blockSignals(True)
        for i in range(self.sweep_param_list.count()):
            it = self.sweep_param_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in susp_keys else Qt.Unchecked)
        self.sweep_param_list.blockSignals(False)
        self._on_sweep_param_item_changed(None)

    def _select_all_sweep_params(self):
        if not hasattr(self, 'sweep_param_list') or self.sweep_param_list is None:
            return
        self.sweep_param_list.blockSignals(True)
        for i in range(self.sweep_param_list.count()):
            self.sweep_param_list.item(i).setCheckState(Qt.Checked)
        self.sweep_param_list.blockSignals(False)
        self._on_sweep_param_item_changed(None)

    def _deselect_all_sweep_params(self):
        if not hasattr(self, 'sweep_param_list') or self.sweep_param_list is None:
            return
        self.sweep_param_list.blockSignals(True)
        for i in range(self.sweep_param_list.count()):
            self.sweep_param_list.item(i).setCheckState(Qt.Unchecked)
        self.sweep_param_list.blockSignals(False)
        self._on_sweep_param_item_changed(None)

    def plot_sweep_results(self, df=None, model_type="3-DOF"):
        if df is None:
            df = getattr(self, 'last_sweep_df', None)
        if df is None or df.empty:
            return
            
        SWEEP_PARAM_DEFS = {
            'aero_bal':        ('Aero Bal Front (%)', 100.0, '%', '#0984e3', 'o', 'left'),
            'wd_f':            ('Weight Dist Front (%)', 100.0, '%', '#00cec9', 's', 'left'),
            'lltd':            ('LLTD Front (%)', 100.0, '%', '#e84393', '^', 'left'),
            'max_ay':          ('Max Ay [g]', 1.0, 'g', '#d63031', '^', 'right'),
            'K_us':            ('Gradiente Sottosterzo Kus [deg/g]', 1.0, 'deg/g', '#f39c12', 'd', 'right'),
            'L':               ('Passo L [m]', 1.0, 'm', '#27ae60', 'v', 'left'),
            't_f':             ('Carreggiata Ant tf [m]', 1.0, 'm', '#16a085', '<', 'left'),
            't_r':             ('Carreggiata Post tr [m]', 1.0, 'm', '#2980b9', '>', 'left'),
            'camber_f_static': ('Camber Ant Statico [deg]', 1.0, 'deg', '#8e44ad', 'p', 'left'),
            'camber_r_static': ('Camber Post Statico [deg]', 1.0, 'deg', '#9b59b6', 'h', 'left'),
            'toe_f_static':    ('Toe Ant Statico [deg]', 1.0, 'deg', '#d35400', '*', 'left'),
            'toe_r_static':    ('Toe Post Statico [deg]', 1.0, 'deg', '#e67e22', 'x', 'left'),
            'K_phi_f':         ('Rigidezza Rollio Ant K_phi_f [Nm/rad]', 1.0, 'Nm/rad', '#0984e3', 's', 'left'),
            'K_phi_r':         ('Rigidezza Rollio Post K_phi_r [Nm/rad]', 1.0, 'Nm/rad', '#6c5ce7', '^', 'left'),
            'K_z_f':           ('Rigidezza Molla Ant K_z_f [N/m]', 1.0, 'N/m', '#00cec9', 'o', 'left'),
            'K_z_r':           ('Rigidezza Molla Post K_z_r [N/m]', 1.0, 'N/m', '#00b894', 'D', 'left'),
        }

        checked_cols = []
        if hasattr(self, 'sweep_param_list') and self.sweep_param_list is not None:
            for i in range(self.sweep_param_list.count()):
                it = self.sweep_param_list.item(i)
                if it.checkState() == Qt.Checked:
                    checked_cols.append(it.data(Qt.UserRole))
        else:
            checked_cols = ['aero_bal', 'wd_f', 'lltd', 'max_ay']

        self.canvas_sweep.fig.clear()
        ax1 = self.canvas_sweep.fig.add_subplot(111)
        
        # Determine if any right-axis variable is checked
        right_cols = [c for c in checked_cols if SWEEP_PARAM_DEFS.get(c, ('', 1.0, '', '', '', 'left'))[5] == 'right']
        left_cols = [c for c in checked_cols if c not in right_cols]
        
        ax2 = ax1.twinx() if (right_cols and left_cols) else None
        
        fallback_colors = ['#1abc9c', '#f1c40f', '#e67e22', '#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
        
        left_labels = []
        for idx, col in enumerate(left_cols):
            if col not in df.columns:
                continue
            meta = SWEEP_PARAM_DEFS.get(col, (col, 1.0, '', fallback_colors[idx % len(fallback_colors)], 'o', 'left'))
            lbl, scale, unit, color, marker, _ = meta
            y_data = df[col] * scale
            ax1.plot(df['v_x'], y_data, color=color, marker=marker, linewidth=2.0, label=f"{lbl}")
            if unit:
                left_labels.append(unit)
                
        right_labels = []
        target_ax2 = ax2 if ax2 is not None else ax1
        for idx, col in enumerate(right_cols):
            if col not in df.columns:
                continue
            meta = SWEEP_PARAM_DEFS.get(col, (col, 1.0, '', '#d63031', '^', 'right'))
            lbl, scale, unit, color, marker, _ = meta
            y_data = df[col] * scale
            ls = '--' if ax2 is not None else '-'
            target_ax2.plot(df['v_x'], y_data, color=color, marker=marker, linestyle=ls, linewidth=2.2, label=f"{lbl}")
            if unit:
                right_labels.append(unit)

        ax1.set_xlabel('Velocità Vx [m/s]', fontsize=9)
        if left_labels:
            u_str = "/".join(sorted(set(left_labels)))
            ax1.set_ylabel(f"Parametri Selezionati [{u_str}]", fontsize=9)
        else:
            ax1.set_ylabel("Parametri Selezionati", fontsize=9)
            
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        if ax2 is not None:
            if right_labels:
                u2_str = "/".join(sorted(set(right_labels)))
                ax2.set_ylabel(f"KPI Prestazionali [{u2_str}]", color='#d63031', fontsize=9)
            else:
                ax2.set_ylabel("KPI Prestazionali", color='#d63031', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='#d63031')
            ax2.legend(loc='upper right', fontsize=8)
            ax1.legend(loc='upper left', fontsize=8)
        else:
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend(loc='best', fontsize=8)

        self.canvas_sweep.fig.suptitle(f"Optimal Target Parameters vs Speed ({model_type})", fontsize=10, fontweight='bold')
        self.canvas_sweep.fig.tight_layout()
        self.canvas_sweep.draw()

    def update_kpi_dashboard(self, v_x, L, t_f, t_r, wd_f, lltd, aero_bal, max_ay, df_handling, tir_file, base_params=None):
        self.log("Evaluating Software & Dynamics KPIs...")
        QApplication.processEvents()
        cb = getattr(self, 'check_partials_cb', None)
        do_check = cb.isChecked() if cb is not None else False
        try:
            model_names = ["3-DOF", "7-DOF", "14-DOF"]
            idx = self.model_combo.currentIndex()
            model_key = model_names[idx] if idx < len(model_names) else "3-DOF"
            
            om_kpis = evaluate_kpis_with_openmdao(v_x, 3.0, wd_f, lltd, aero_bal, L, t_f, t_r, do_check_partials=do_check, tir_file=tir_file, model_type=model_key)
        except Exception as e:
            self.log(f"Error evaluating OpenMDAO KPIs: {e}")
            om_kpis = {'solver_iters': 'N/A', 'exec_time_ms': 'N/A', 'jac_max_error': 'N/A'}
            
        from src.mdo.mdo_global_optimizer import compute_stability_metrics
        from src.mdo.mdo_skidpad import simulate_skidpad
        from src.models.params_vehicle import VehicleParams

        stab = compute_stability_metrics(L, wd_f, lltd, aero_bal, v_x=v_x, tir_path=tir_file)
        
        vp_bench = VehicleParams()
        vp_bench.L = L; vp_bench.t_f = t_f; vp_bench.t_r = t_r
        vp_bench.wd_f = wd_f; vp_bench.lltd = lltd; vp_bench.aero_balance = aero_bal
        skid_res = simulate_skidpad(vp_bench, radius=9.125, model_type=model_key, tir_path=tir_file)
        
        us_grad = mdo.calculate_understeer_gradient(df_handling, L, v_x) if df_handling is not None else stab['K_us_degg']
        
        df_mmm = mdo.generate_mmm_diagram(v_x, wd_f, lltd, aero_bal, 1.0, L, t_f, t_r, delta_deg_list=[0], beta_range_deg=(-10, 10), num_beta=20, tir_file=tir_file)
        ctrl_margin = mdo.calculate_control_margin(df_mmm)
        
        roll_grad_val = om_kpis.get('roll_gradient_deg_g')
        if self.model_combo.currentIndex() == 2 and isinstance(roll_grad_val, (int, float)):
            roll_grad = f"{roll_grad_val:.3f} deg/g"
        else:
            roll_grad = "N/A (14-DOF Only)"
        
        kpis = []
        if base_params is not None and len(base_params) >= 6:
            kpis.extend([
                ("=== TARGET OTTIMIZZATI VS BASELINE ===", ""),
                ("Passo L [m]", f"{L:.3f} m  (Init: {base_params[0]:.3f} m | Δ: {((L-base_params[0])/base_params[0])*100:+.1f}%)"),
                ("Carreggiata Ant. t_f [m]", f"{t_f:.3f} m  (Init: {base_params[1]:.3f} m | Δ: {((t_f-base_params[1])/base_params[1])*100:+.1f}%)"),
                ("Carreggiata Post. t_r [m]", f"{t_r:.3f} m  (Init: {base_params[2]:.3f} m | Δ: {((t_r-base_params[2])/base_params[2])*100:+.1f}%)"),
                ("Ripartizione Massa wd_f [%]", f"{wd_f*100:.2f}%  (Init: {base_params[3]*100:.2f}% | Δ: {((wd_f-base_params[3])/base_params[3])*100:+.1f}%)"),
                ("Ripartizione Rollio LLTD [%]", f"{lltd*100:.2f}%  (Init: {base_params[4]*100:.2f}% | Δ: {((lltd-base_params[4])/base_params[4])*100:+.1f}%)"),
                ("Bilanciamento Aero Ant. [%]", f"{aero_bal*100:.2f}%  (Init: {base_params[5]*100:.2f}% | Δ: {((aero_bal-base_params[5])/base_params[5])*100:+.1f}%)"),
            ])
            if hasattr(self, 'last_optimized_params') and 'K_phi_f' in self.last_optimized_params:
                kphi_f_val = self.last_optimized_params['K_phi_f']
                kphi_r_val = self.last_optimized_params['K_phi_r']
                kpis.extend([
                    ("Rigidezza Rollio Ant. K_phi_f [Nm/rad]", f"{kphi_f_val:.1f} Nm/rad"),
                    ("Rigidezza Rollio Post. K_phi_r [Nm/rad]", f"{kphi_r_val:.1f} Nm/rad"),
                ])
            
        kpis.extend([
            ("--- SOFTWARE & COMPUTATIONAL ---", ""),
            ("NewtonSolver Iterations", f"{om_kpis['solver_iters']}"),
            ("Execution Time per Loop (ms)", f"{om_kpis['exec_time_ms']:.2f}" if isinstance(om_kpis['exec_time_ms'], (int, float)) else "N/A"),
            ("Jacobian Max Error", f"{om_kpis['jac_max_error']:.2e}" if (do_check and isinstance(om_kpis.get('jac_max_error'), (int, float))) else "Skipped (Check Box)"),
            ("--- VEHICLE DYNAMICS & STABILITY ---", ""),
            ("Max Lateral Acceleration (g)", f"{max_ay:.3f}"),
            ("Understeer Gradient K_us (deg/g)", f"{stab['K_us_degg']:+.3f}"),
            ("Roll Gradient (deg/g)", roll_grad),
            ("Dynamic Balance Behavior", f"{stab['behavior']}"),
            ("Critical Speed v_crit (km/h)", f"{stab['v_crit_kmh']:.1f}" if stab['behavior'] == "Oversteer" else "N/A (Stable at all speeds)"),
            ("Characteristic Speed v_char (km/h)", f"{stab['v_char_kmh']:.1f}" if stab['behavior'] == "Understeer" else "N/A"),
            ("Control Margin (Nm @ 0 deg steer)", f"{ctrl_margin:.1f}"),
            ("--- FSAE BENCHMARK (SKIDPAD R=9.125m) ---", ""),
            ("Skidpad Lap Time (s)", f"{skid_res['time']:.3f}" if skid_res['success'] else "Non Converge"),
            ("Skidpad Max Velocity (km/h)", f"{skid_res['v_max']*3.6:.1f}" if skid_res['success'] else "N/A"),
            ("Skidpad Lateral Ay (g)", f"{skid_res['ay']:.3f}" if skid_res['success'] else "N/A")
        ])
        
        if self.model_combo.currentIndex() == 2:
            kpis.append(("Roll Gradient (deg/g)", roll_grad))
        
        self.kpi_table.setRowCount(len(kpis))
        for row, (name, val) in enumerate(kpis):
            self.kpi_table.setItem(row, 0, QTableWidgetItem(name))
            self.kpi_table.setItem(row, 1, QTableWidgetItem(val))
            
        self.kpi_table.resizeColumnsToContents()

        # Plot optimization comparison & deltas on self.canvas_kpi_delta
        if hasattr(self, 'canvas_kpi_delta'):
            self.canvas_kpi_delta.fig.clear()
            if base_params is not None and len(base_params) >= 6:
                ax_top = self.canvas_kpi_delta.fig.add_subplot(211)
                ax_bot = self.canvas_kpi_delta.fig.add_subplot(212)
                
                param_labels = ['L [m]', 't_f [m]', 't_r [m]', 'wd_f [%]', 'lltd [%]', 'aero [%]']
                init_vals = [base_params[0], base_params[1], base_params[2], base_params[3]*100, base_params[4]*100, base_params[5]*100]
                opt_vals  = [L, t_f, t_r, wd_f*100, lltd*100, aero_bal*100]
                
                y_pos = np.arange(len(param_labels))
                bar_h = 0.35
                
                ax_top.barh(y_pos + bar_h/2, init_vals, height=bar_h, color='#e17055', alpha=0.85, label='Baseline')
                ax_top.barh(y_pos - bar_h/2, opt_vals,  height=bar_h, color='#00b894', alpha=0.85, label='Optimized')
                ax_top.set_yticks(y_pos)
                ax_top.set_yticklabels(param_labels, fontsize=8, fontweight='bold')
                ax_top.set_title("Vehicle Parameters: Baseline vs Optimized Target", fontsize=9, fontweight='bold')
                ax_top.grid(True, linestyle=':', alpha=0.5)
                ax_top.legend(loc='lower right', fontsize=8)
                
                # Numeric value labels on bars
                for y, iv, ov in zip(y_pos, init_vals, opt_vals):
                    ax_top.text(iv*1.01, y + bar_h/2, f" {iv:.2f}", va='center', ha='left', fontsize=7, color='#c0392b', fontweight='bold')
                    ax_top.text(ov*1.01, y - bar_h/2, f" {ov:.2f}", va='center', ha='left', fontsize=7, color='#27ae60', fontweight='bold')
                
                # Bottom subplot: Percentage Delta (%)
                deltas_pct = [((opt - init) / (init if abs(init) > 1e-4 else 1.0)) * 100.0 for init, opt in zip(init_vals, opt_vals)]
                colors = ['#00b894' if d >= 0 else '#d63031' for d in deltas_pct]
                bars = ax_bot.barh(y_pos, deltas_pct, height=0.45, color=colors, alpha=0.85)
                ax_bot.axvline(0, color='black', linewidth=1.0)
                ax_bot.set_yticks(y_pos)
                ax_bot.set_yticklabels(param_labels, fontsize=8, fontweight='bold')
                ax_bot.set_xlabel("Variazione Relativa [%]", fontsize=8)
                ax_bot.set_title("Variazione Parametri Ottimizzati (Delta %)", fontsize=9, fontweight='bold')
                ax_bot.grid(True, linestyle=':', alpha=0.5)
                
                for bar, d_val in zip(bars, deltas_pct):
                    x_text = d_val + (0.6 if d_val >= 0 else -0.6)
                    ha = 'left' if d_val >= 0 else 'right'
                    ax_bot.text(x_text, bar.get_y() + bar.get_height()/2, f"{d_val:+.1f}%", 
                                va='center', ha=ha, fontsize=8, fontweight='bold',
                                color='#2d3436')
                
                self.canvas_kpi_delta.fig.tight_layout()
                self.canvas_kpi_delta.draw()

    def plot_mmm(self, v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, model_type=None):
        if model_type is None:
            model_names = ["3-DOF", "7-DOF", "14-DOF"]
            idx = self.model_combo.currentIndex() if hasattr(self, 'model_combo') else 0
            model_type = model_names[idx] if idx < len(model_names) else "3-DOF"

        vp = self._get_current_vehicle_params()
        self.log(f"Generating Yaw Moment Diagram (YMD) [{model_type}] at Vx = {v_x:.1f} m/s...")
        QApplication.processEvents()
        
        try:
            raw_deltas = self.mmm_deltas_input.text().strip()
            if raw_deltas and ',' in raw_deltas and len(raw_deltas.split(',')) > 1:
                deltas = [float(s.strip()) for s in raw_deltas.split(',')]
            else:
                num_d = int(self.mmm_dpoints_input.text().strip()) if hasattr(self, 'mmm_dpoints_input') else 9
                deltas = list(np.linspace(-12, 12, num_d))
                
            b_range = [float(s.strip()) for s in self.mmm_beta_input.text().split(',')]
            if len(b_range) != 2:
                b_range = [-10.0, 10.0]
            num_beta = int(self.mmm_bpoints_input.text().strip())
        except ValueError:
            deltas = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
            b_range = [-10.0, 10.0]
            num_beta = 21
            
        def get_param(attr, default):
            item = self.bounds_inputs.get(attr) if hasattr(self, 'bounds_inputs') else None
            if item and not item.get('digitizer'):
                try:
                    return float(item['val'].text())
                except (ValueError, AttributeError):
                    pass
            return float(getattr(vp, attr, default))

        cam_f = get_param('camber_f_static', -1.5)
        cam_r = get_param('camber_r_static', -1.0)
        toe_f = get_param('toe_f_static', 0.0)
        toe_r = get_param('toe_r_static', 0.1)
        kphi_f = get_param('K_phi_f', 8000.0)
        kphi_r = get_param('K_phi_r', 6000.0)
        kz_f   = get_param('K_z_f', 18000.0)
        kz_r   = get_param('K_z_r', 16000.0)

        df_mmm = mdo.generate_mmm_diagram(
            v_x=v_x, wd_f=wd_f, lltd=lltd, aero_bal=aero_bal, lmuy_mult=1.0,
            L=L, t_f=t_f, t_r=t_r,
            delta_deg_list=deltas,
            beta_range_deg=(b_range[0], b_range[1]), num_beta=num_beta,
            tir_file=tir_file,
            model_type=model_type,
            vehicle_params=vp,
            camber_f_static=cam_f, camber_r_static=cam_r,
            toe_f_static=toe_f, toe_r_static=toe_r,
            K_phi_f=kphi_f, K_phi_r=kphi_r,
            K_z_f=kz_f, K_z_r=kz_r
        )
        
        self.canvas_ymd.axes.clear()
        ax = self.canvas_ymd.axes
        
        if not df_mmm.empty:
            from scipy.interpolate import PchipInterpolator
            is_en = self.is_english_mode()

            # 1. Constant delta lines — parameterised by ay_g on the YMD plane.
            # Sort by ay_g and keep only the pre-peak monotonic section to avoid
            # zigzag / hourglass artifacts when ay_g is not monotone in beta.
            for delta in sorted(df_mmm['delta_deg'].unique()):
                subset = df_mmm[df_mmm['delta_deg'] == delta].copy()
                if len(subset) < 2:
                    continue
                # Sort by lateral acceleration (natural abscissa of YMD)
                subset = subset.sort_values('ay_g').reset_index(drop=True)
                ay_vals = subset['ay_g'].values
                mz_vals = subset['yaw_moment'].values

                # Restrict to monotone pre-peak: find last index where ay still increases
                # (drop folded-back points that create the hourglass collapse)
                peak_idx = int(np.argmax(ay_vals))
                # Take from the most-negative ay up to the peak, and symmetrically the
                # negative-ay side (negative delta has its peak at negative ay).
                # Strategy: split into negative-ay and positive-ay arms and keep each arm
                # monotone separately, then recombine.
                neg_mask = ay_vals < 0
                pos_mask = ay_vals >= 0
                arms = []
                if np.sum(neg_mask) >= 2:
                    # Negative arm: sort descending (most negative → 0), keep monotone
                    neg_ay = ay_vals[neg_mask]
                    neg_mz = mz_vals[neg_mask]
                    order_n = np.argsort(neg_ay)          # ascending
                    neg_ay, neg_mz = neg_ay[order_n], neg_mz[order_n]
                    # Keep only strictly increasing abs(ay) run from the most-negative end
                    keep = [0]
                    for k in range(1, len(neg_ay)):
                        if neg_ay[k] > neg_ay[keep[-1]]:
                            keep.append(k)
                    arms.append((neg_ay[keep], neg_mz[keep]))
                if np.sum(pos_mask) >= 2:
                    pos_ay = ay_vals[pos_mask]
                    pos_mz = mz_vals[pos_mask]
                    order_p = np.argsort(pos_ay)
                    pos_ay, pos_mz = pos_ay[order_p], pos_mz[order_p]
                    keep = [0]
                    for k in range(1, len(pos_ay)):
                        if pos_ay[k] > pos_ay[keep[-1]]:
                            keep.append(k)
                    arms.append((pos_ay[keep], pos_mz[keep]))

                # Merge arms into a single ordered sequence
                if arms:
                    all_ay = np.concatenate([a[0] for a in arms])
                    all_mz = np.concatenate([a[1] for a in arms])
                    order = np.argsort(all_ay)
                    all_ay, all_mz = all_ay[order], all_mz[order]
                else:
                    all_ay, all_mz = ay_vals, mz_vals

                # PCHIP smooth only when enough unique points are available
                all_ay_u, idx_u2 = np.unique(all_ay, return_index=True)
                all_mz_u = all_mz[idx_u2]
                if len(all_ay_u) >= 4:
                    cs_mz2 = PchipInterpolator(all_ay_u, all_mz_u)
                    ay_p = np.linspace(all_ay_u[0], all_ay_u[-1], 240)
                    mz_p = cs_mz2(ay_p)
                elif len(all_ay_u) >= 2:
                    ay_p, mz_p = all_ay_u, all_mz_u
                else:
                    continue

                is_zero_delta = (abs(delta) < 1e-3)
                lw = 2.2 if is_zero_delta else 1.25
                c = '#2d3436' if is_zero_delta else '#34495e'
                alpha_val = 0.95 if is_zero_delta else 0.80
                ax.plot(ay_p, mz_p, color=c, alpha=alpha_val, linewidth=lw, zorder=3)

                # Label near ay ≈ 0 (closest point to origin on this delta curve)
                sub_lbl = subset.iloc[np.abs(subset['ay_g'].values).argsort()[:1]]
                if not sub_lbl.empty:
                    ax.text(
                        sub_lbl['ay_g'].values[0], sub_lbl['yaw_moment'].values[0] + (18.0 if delta >= 0 else -32.0),
                        f" $\\delta={delta:+.0f}^\\circ$", fontsize=7.5, color='#2c3e50', fontweight='bold', zorder=5
                    )


            # 3. Constant beta lines (clean, uniform step with PCHIP smoothing)
            all_betas = sorted(df_mmm['beta_deg'].unique())
            step_beta = 2.0 if (b_range[1] - b_range[0]) <= 24.0 else 4.0
            target_betas = np.arange(round(b_range[0]), round(b_range[1]) + 0.1, step_beta)
            
            for beta_tgt in target_betas:
                closest_b = min(all_betas, key=lambda b: abs(b - beta_tgt))
                if abs(closest_b - beta_tgt) < 0.4:
                    subset = df_mmm[np.isclose(df_mmm['beta_deg'], closest_b, atol=0.25)].sort_values('delta_deg')
                    if len(subset) > 1:
                        d_vals = subset['delta_deg'].values
                        ay_vals = subset['ay_g'].values
                        mz_vals = subset['yaw_moment'].values
                        d_u, idx_u = np.unique(d_vals, return_index=True)
                        
                        if len(d_u) >= 4:
                            cs_ay_b = PchipInterpolator(d_u, ay_vals[idx_u])
                            cs_mz_b = PchipInterpolator(d_u, mz_vals[idx_u])
                            d_dense = np.linspace(d_u.min(), d_u.max(), 180)
                            ay_p, mz_p = cs_ay_b(d_dense), cs_mz_b(d_dense)
                        else:
                            ay_p, mz_p = ay_vals, mz_vals
                            
                        is_zero_beta = (abs(closest_b) < 1e-3)
                        lw = 1.6 if is_zero_beta else 1.05
                        alpha_val = 0.85 if is_zero_beta else 0.50
                        line_color = '#0984e3' if is_zero_beta else '#74b9ff'
                        ax.plot(ay_p, mz_p, color=line_color, linestyle='--', alpha=alpha_val, linewidth=lw, zorder=2)
                        
                        # Add label on select lines (e.g. 0, +-4, +-8, +-12)
                        if round(closest_b) in [-8, -4, 0, 4, 8]:
                            sub_d0 = subset.iloc[(subset['delta_deg'] - 0).abs().argsort()[:1]]
                            if not sub_d0.empty:
                                ax.text(
                                    sub_d0['ay_g'].values[0] + 0.02, sub_d0['yaw_moment'].values[0],
                                    f" $\\beta={closest_b:+.0f}^\\circ$", fontsize=6.8, color='#0984e3', zorder=5
                                )
                    
            # 4. Extract stability & control derivatives at origin (delta=0, beta=0)
            sub_d0 = df_mmm[np.isclose(df_mmm['delta_deg'], 0.0, atol=1e-2)].sort_values('beta_deg')
            sub_b0 = df_mmm[np.isclose(df_mmm['beta_deg'], 0.0, atol=1e-2)].sort_values('delta_deg')
            
            N_beta = 0.0
            if len(sub_d0) >= 3:
                b_arr = sub_d0['beta_deg'].values
                mz_arr = sub_d0['yaw_moment'].values
                lin_mask = np.abs(b_arr) <= 3.5
                if np.sum(lin_mask) >= 2:
                    N_beta = float(np.polyfit(b_arr[lin_mask], mz_arr[lin_mask], 1)[0])
                else:
                    N_beta = float(np.polyfit(b_arr, mz_arr, 1)[0])
                    
            N_delta = 0.0
            if len(sub_b0) >= 3:
                d_arr = sub_b0['delta_deg'].values
                mz_d_arr = sub_b0['yaw_moment'].values
                # Velocity-adaptive linear window (mirrors generate_multispeed_mmm_envelopes)
                delta_lin_lim = max(1.0, 3.5 * (10.0 / max(v_x, 10.0)))
                lin_mask_d = np.abs(d_arr) <= delta_lin_lim
                if np.sum(lin_mask_d) >= 2:
                    N_delta = float(np.polyfit(d_arr[lin_mask_d], mz_d_arr[lin_mask_d], 1)[0])
                else:
                    N_delta = float(np.polyfit(d_arr, mz_d_arr, 1)[0])

                    
            cm_val = abs(N_delta) / abs(N_beta) if abs(N_beta) > 1e-3 else 1.0
            mz_max_val = float(df_mmm['yaw_moment'].abs().max())
            ay_max_val = float(df_mmm['ay_g'].abs().max())
            
            # 5. Iso-Radius Reference Rays (R = const)
            if hasattr(self, 'ymd_show_radii_cb') and _safe_is_checked(self.ymd_show_radii_cb, default=True):
                radii = [
                    (9.125, "R=9.1m (Skidpad)", "#e67e22"),
                    (15.0,  "R=15m (Hairpin)",  "#9b59b6"),
                    (30.0,  "R=30m (Esse)",     "#16a085"),
                    (50.0,  "R=50m (Media)",    "#2980b9"),
                    (100.0, "R=100m (Veloce)",  "#7f8c8d")
                ]
                y_max = float(df_mmm['yaw_moment'].max()) if not df_mmm.empty else 500.0
                
                for r_val, r_lbl, r_col in radii:
                    ay_req = (v_x ** 2) / (9.81 * r_val)
                    if ay_req <= ay_max_val * 1.15:
                        ax.axvline(ay_req, color=r_col, linestyle=':', linewidth=1.3, alpha=0.75)
                        ax.axvline(-ay_req, color=r_col, linestyle=':', linewidth=1.3, alpha=0.75)
                        ax.plot(ay_req, 0.0, marker='*', markersize=8, color=r_col, zorder=7)
                        ax.plot(-ay_req, 0.0, marker='*', markersize=8, color=r_col, zorder=7)
                        ax.text(
                            ay_req + 0.015, y_max * 0.70, f" {r_lbl}",
                            rotation=90, verticalalignment='center', fontsize=6.8, color=r_col, fontweight='bold'
                        )
            
            # Steady-State Trim line (Mz = 0) and Origin Marker
            trim_lbl = "Steady-State Trim ($M_z = 0$)" if is_en else "Trim Stazionario ($M_z = 0$)"
            ax.axhline(0, color='#e74c3c', linewidth=2.0, linestyle='-', label=trim_lbl, zorder=4)
            ax.axvline(0, color='#636e72', linewidth=1.0, linestyle=':', zorder=2)
            ax.plot(0, 0, marker='o', markersize=5, color='#2d3436', zorder=6)
            
            if is_en:
                ax.set_xlabel("Lateral Acceleration $a_y$ [g]", fontsize=10)
                ax.set_ylabel("Total Yaw Moment $M_z$ [Nm]", fontsize=10)
                ax.set_title(
                    f"Yaw Moment Diagram (YMD) [{model_type}] — $v_x={v_x:.1f}$ m/s\n"
                    f"Control Authority $N_\\delta = {N_delta:+.1f}$ Nm/deg | Directional Stability $N_\\beta = {N_beta:+.1f}$ Nm/deg | Control Margin $CM = {cm_val:.2f}$ | Max $M_z = \\pm {mz_max_val:.0f}$ Nm",
                    fontsize=9.5, fontweight='bold'
                )
            else:
                ax.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=10)
                ax.set_ylabel("Momento d'Imbardata Totale $M_z$ [Nm]", fontsize=10)
                ax.set_title(
                    f"Yaw Moment Diagram (YMD) [{model_type}] — $v_x={v_x:.1f}$ m/s\n"
                    f"Autorità Controllo $N_\\delta = {N_delta:+.1f}$ Nm/deg | Stabilità $N_\\beta = {N_beta:+.1f}$ Nm/deg | Margine di Controllo $CM = {cm_val:.2f}$ | Max $M_z = \\pm {mz_max_val:.0f}$ Nm",
                    fontsize=9.5, fontweight='bold'
                )
            
            import matplotlib.lines as mlines
            steer_lbl = "Steer Angle $\\delta = \\mathrm{const}$" if is_en else "Angolo Sterzo $\\delta = \\mathrm{cost}$"
            beta_lbl = "Sideslip Angle $\\beta = \\mathrm{const}$" if is_en else "Angolo Deriva $\\beta = \\mathrm{cost}$"
            line_steer = mlines.Line2D([], [], color='#2d3436', linewidth=1.8, label=steer_lbl)
            line_beta = mlines.Line2D([], [], color='#0984e3', linestyle='--', linewidth=1.4, alpha=0.7, label=beta_lbl)
            handles, labels = ax.get_legend_handles_labels()
            leg_ymd = ax.legend(handles=handles + [line_steer, line_beta], loc='upper right', fontsize=7.5)
            if leg_ymd:
                leg_ymd.set_draggable(True)
            
            ax.grid(True, linestyle=':', alpha=0.55)
            self.canvas_ymd.figure.tight_layout()
            self.canvas_ymd.draw()
            self.log(f"Yaw Moment Diagram (MMM / YMD) [{model_type}] generato con successo.")
        else:
            self.log("Attenzione: Impossibile generare i dati per il diagramma MMM.")

    def plot_mmm_multispeed(self):
        """Generates multi-speed overlay of YMD boundary envelopes and derivative trends."""
        self.btn_calc_multispeed_ymd.setEnabled(False)
        self.btn_calc_multispeed_ymd.setText("Generazione...")
        
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        idx = self.model_combo.currentIndex() if hasattr(self, 'model_combo') else 0
        model_type = model_names[idx] if idx < len(model_names) else "3-DOF"
        
        self.log(f"Generazione Multi-Speed YMD Overlay & Analisi Stabilità [{model_type}] vs Velocità...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
            
            try:
                speeds = [float(s.strip()) for s in self.ymd_multispeeds_input.text().split(',')]
            except Exception:
                speeds = [8.0, 12.0, 16.0, 20.0, 25.0]
                
            def get_param(attr, default):
                item = self.bounds_inputs.get(attr) if hasattr(self, 'bounds_inputs') else None
                if item and not item.get('digitizer'):
                    try:
                        return float(item['val'].text())
                    except (ValueError, AttributeError):
                        pass
                return getattr(vp, attr, default)
            
            L = get_param('L', 1.529)
            t_f = get_param('t_f', 1.252)
            t_r = get_param('t_r', 1.136)
            wd_f = get_param('wd_f', 0.45)
            lltd = get_param('lltd', 0.50)
            aero_bal = get_param('aero_balance', 0.485)
            
            cam_f = get_param('camber_f_static', -1.5)
            cam_r = get_param('camber_r_static', -1.0)
            toe_f = get_param('toe_f_static', 0.0)
            toe_r = get_param('toe_r_static', 0.1)
            kphi_f = get_param('K_phi_f', 8000.0)
            kphi_r = get_param('K_phi_r', 6000.0)
            kz_f   = get_param('K_z_f', 18000.0)
            kz_r   = get_param('K_z_r', 16000.0)
            
            res = mdo.generate_multispeed_mmm_envelopes(
                speeds_list=speeds,
                wd_f=wd_f,
                lltd=lltd,
                aero_bal=aero_bal,
                lmuy_mult=getattr(vp, 'LMUY', 1.0),
                L=L,
                t_f=t_f,
                t_r=t_r,
                delta_max_deg=12.0,
                num_beta=21,
                tir_file=tir_file,
                model_type=model_type,
                vehicle_params=vp,
                camber_f_static=cam_f, camber_r_static=cam_r,
                toe_f_static=toe_f, toe_r_static=toe_r,
                K_phi_f=kphi_f, K_phi_r=kphi_r,
                K_z_f=kz_f, K_z_r=kz_r
            )
            
            envs = res['speed_envelopes']
            df_derivs = res['derivatives_df']
            
            self.fig_ymd_multi.clear()
            ax1 = self.fig_ymd_multi.add_subplot(121)
            ax2 = self.fig_ymd_multi.add_subplot(222)
            ax3 = self.fig_ymd_multi.add_subplot(224)
            
            colors = ['#0984e3', '#00b894', '#e17055', '#6c5ce7', '#d63031', '#fdcb6e']
            while len(colors) < len(speeds):
                colors.extend(colors)
                
            # --- Subplot 1: Multi-Speed YMD Envelopes Overlay ---
            for idx, (spd, c) in enumerate(zip(speeds, colors)):
                df_spd = envs.get(spd, pd.DataFrame())
                if df_spd.empty:
                    continue
                
                # Constant delta curves — sort by ay_g, keep monotone pre-peak arm
                for delta in [-12.0, 0.0, 12.0]:
                    sub = df_spd[np.isclose(df_spd['delta_deg'], delta, atol=0.5)].copy()
                    if len(sub) < 2:
                        continue
                    sub = sub.sort_values('ay_g').reset_index(drop=True)
                    ay_s = sub['ay_g'].values
                    mz_s = sub['yaw_moment'].values
                    # Keep only the monotone pre-peak section (largest contiguous run)
                    keep = [0]
                    for k in range(1, len(ay_s)):
                        if ay_s[k] > ay_s[keep[-1]]:
                            keep.append(k)
                    ay_s, mz_s = ay_s[keep], mz_s[keep]
                    if len(ay_s) < 2:
                        continue
                    ls = '-' if delta == 0.0 else '--'
                    lw = 1.8 if delta == 0.0 else 1.2
                    lbl = f'v = {spd:.0f} m/s' if delta == 0.0 else None
                    ax1.plot(ay_s, mz_s, color=c, linestyle=ls, linewidth=lw, alpha=0.85, label=lbl)

                        
                # Extreme beta caps
                for b_val in [-10.0, 10.0]:
                    sub_b = df_spd[np.isclose(df_spd['beta_deg'], b_val, atol=0.5)].sort_values('delta_deg')
                    if len(sub_b) > 1:
                        ax1.plot(sub_b['ay_g'], sub_b['yaw_moment'], color=c, linestyle=':', linewidth=1.0, alpha=0.5)

            ax1.axhline(0, color='#2d3436', linewidth=1.6, linestyle='-', label='Steady-State Trim ($M_z=0$)')
            ax1.axvline(0, color='gray', linewidth=1.0, linestyle=':')
            
            # Iso-Radius references on multi-speed YMD
            if hasattr(self, 'ymd_show_radii_cb') and _safe_is_checked(self.ymd_show_radii_cb, default=True):
                radii = [(9.125, "R=9.1m", "#e67e22"), (15.0, "R=15m", "#9b59b6"), (30.0, "R=30m", "#16a085")]
                for r_val, r_lbl, r_col in radii:
                    # Representative reference at mid speed 15 m/s
                    ay_r = (15.0 ** 2) / (9.81 * r_val)
                    if ay_r <= 2.8:
                        ax1.axvline(ay_r, color=r_col, linestyle=':', alpha=0.6)
                        ax1.text(ay_r, 450.0, f" {r_lbl} @15m/s", rotation=90, fontsize=6.5, color=r_col, fontweight='bold')

            ax1.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
            ax1.set_ylabel("Momento Imbardata Totale $M_z$ [Nm]", fontsize=9)
            ax1.set_title("1. Inviluppi YMD Multi-Velocità ($v_x = 8 \\dots 25$ m/s)", fontsize=10, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            leg1 = ax1.legend(fontsize='x-small', loc='upper right')
            if leg1:
                leg1.set_draggable(True)
                
            # --- Subplot 2: Control Authority N_delta & Directional Stability N_beta vs Speed ---
            if not df_derivs.empty:
                v_arr = df_derivs['v_x'].values
                ax2.plot(v_arr, df_derivs['N_delta'], 'o-', color='#0984e3', linewidth=2.0, label='Autorità Controllo $N_\\delta$')
                ax2.plot(v_arr, df_derivs['N_beta'], 's--', color='#e17055', linewidth=2.0, label='Stabilità Direzionale $N_\\beta$')
                ax2.set_xlabel("Velocità di Avanzamento $v_x$ [m/s]", fontsize=8.5)
                ax2.set_ylabel("Derivate [Nm/deg]", fontsize=8.5)
                ax2.set_title("2. Derivate di Controllo & Stabilità vs $v_x$", fontsize=9.5, fontweight='bold')
                ax2.grid(True, linestyle=':', alpha=0.6)
                leg2 = ax2.legend(fontsize='x-small', loc='best')
                if leg2:
                    leg2.set_draggable(True)
                    
                # --- Subplot 3: Peak Ay & Control Margin vs Speed ---
                ax3.plot(v_arr, df_derivs['max_ay_g'], 'd-', color='#00b894', linewidth=2.0, label='Grip Limite $a_{y,max}$ [g]')
                ax3.set_xlabel("Velocità di Avanzamento $v_x$ [m/s]", fontsize=8.5)
                ax3.set_ylabel("Grip Max $a_y$ [g]", fontsize=8.5, color='#00b894')
                ax3.tick_params(axis='y', labelcolor='#00b894')
                
                ax3_cm = ax3.twinx()
                ax3_cm.plot(v_arr, df_derivs['control_margin'], '^-', color='#6c5ce7', linewidth=1.8, label='Control Margin $CM$')
                ax3_cm.set_ylabel("Control Margin $CM$", fontsize=8.5, color='#6c5ce7')
                ax3_cm.tick_params(axis='y', labelcolor='#6c5ce7')
                ax3.set_title("3. Grip Limite & Margine di Controllo vs $v_x$", fontsize=9.5, fontweight='bold')
                ax3.grid(True, linestyle=':', alpha=0.6)
                
            self.fig_ymd_multi.tight_layout()
            self.canvas_ymd_multi.draw()
            if hasattr(self, 'ymd_subtabs') and hasattr(self, 'tab_ymd_multi'):
                self.ymd_subtabs.setCurrentWidget(self.tab_ymd_multi)
            self.log("Multi-Speed YMD Overlay & Analisi Stabilità generati con successo.")
        except Exception as e:
            import traceback
            self.log(f"Errore generazione Multi-Speed YMD: {traceback.format_exc()}")
        finally:
            self.btn_calc_multispeed_ymd.setEnabled(True)
            self.btn_calc_multispeed_ymd.setText(" Genera Multi-Speed Overlay")

    def _compute_ggv_mesh(self, speeds, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file):
        from scipy.spatial import ConvexHull
        from scipy.interpolate import interp1d
        from src.models.params_vehicle import VehicleParams
        from src.models.params_tire import TireParams
        from src.models.tire_model_mf61 import (
            mf61_lateral_force, mf61_longitudinal_force,
            mf61_combined_lateral_force, mf61_combined_longitudinal_force
        )
        
        base_dir = os.path.dirname(__file__)
        tir_path = os.path.join(base_dir, "tire_tir", tir_file if tir_file else "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir")
        tp = TireParams(tir_path)
        vp = self._get_current_vehicle_params()
        vp.L = L
        vp.t_f = t_f
        vp.t_r = t_r
        
        X, Y, Z = [], [], []
        kappas = np.linspace(-0.25, 0.25, 17)
        alphas = np.linspace(0.0, np.deg2rad(18.0), 17)
        
        for v in speeds:
            F_dwnf = 0.5 * vp.rho * vp.S * vp.Cz * v**2
            Cd = getattr(vp, 'Cd', 1.2)
            F_drag = 0.5 * vp.rho * vp.S * Cd * v**2
            
            Fz_f = max((vp.m * 9.81 * (1.0 - wd_f) + F_dwnf * aero_bal) / 2.0, 100.0)
            Fz_r = max((vp.m * 9.81 * wd_f + F_dwnf * (1.0 - aero_bal)) / 2.0, 100.0)
            
            pts = []
            for a in alphas:
                for k in kappas:
                    Fx_f_p = float(mf61_longitudinal_force(k, Fz_f, 0.0, tp, LMUX=vp.LMUX))
                    Fy_f_p = float(mf61_lateral_force(a, Fz_f, 0.0, tp, LMUY=vp.LMUY))
                    Fx_f = float(mf61_combined_longitudinal_force(Fx_f_p, k, a, Fz_f, 0.0, tp))
                    Fy_f = float(mf61_combined_lateral_force(Fy_f_p, a, k, Fz_f, 0.0, tp))
                    
                    Fx_r_p = float(mf61_longitudinal_force(k, Fz_r, 0.0, tp, LMUX=vp.LMUX))
                    Fy_r_p = float(mf61_lateral_force(a, Fz_r, 0.0, tp, LMUY=vp.LMUY))
                    Fx_r = float(mf61_combined_longitudinal_force(Fx_r_p, k, a, Fz_r, 0.0, tp))
                    Fy_r = float(mf61_combined_lateral_force(Fy_r_p, a, k, Fz_r, 0.0, tp))
                    
                    Fx_tot = 2.0 * Fx_r if k > 0 else (2.0 * Fx_f + 2.0 * Fx_r)
                    Fy_tot = 2.0 * Fy_f + 2.0 * Fy_r
                    
                    ax_g = (Fx_tot - F_drag) / (vp.m * 9.81)
                    ay_g = Fy_tot / (vp.m * 9.81)
                    
                    pts.append([ay_g, ax_g])
                    pts.append([-ay_g, ax_g])
                    
            pts = np.array(pts)
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            
            center = np.mean(hull_pts, axis=0)
            angles = np.arctan2(hull_pts[:, 1] - center[1], hull_pts[:, 0] - center[0])
            idx = np.argsort(angles)
            hull_pts = hull_pts[idx]
            hull_pts = np.vstack((hull_pts, hull_pts[0]))
            
            t_old = np.linspace(0, 1, len(hull_pts))
            t_new = np.linspace(0, 1, 36)
            x_res = interp1d(t_old, hull_pts[:, 0])(t_new)
            y_res = interp1d(t_old, hull_pts[:, 1])(t_new)
            z_res = np.full_like(x_res, v)
            
            X.append(x_res)
            Y.append(y_res)
            Z.append(z_res)
            
        return np.array(X), np.array(Y), np.array(Z)

    def plot_ggv(self, opt, wd_f, lltd, aero_bal, L, t_f, t_r, base_params=None, tir_file=None):
        self.log("Generating 3D G-G-V Performance Comparison Diagram...")
        QApplication.processEvents()
        
        try:
            vmin = float(self.ggv_vmin_input.text())
            vmax = float(self.ggv_vmax_input.text())
            nslices = int(self.ggv_nslices_input.text())
        except Exception:
            vmin, vmax, nslices = 8.0, 28.0, 6
            
        speeds = np.linspace(vmin, vmax, max(nslices, 3))
        self.axes_ggv.clear()
        
        # 1. Compute and plot Baseline GGV surface if provided
        if base_params is not None and len(base_params) >= 6:
            L_b, tf_b, tr_b, wdf_b, lltd_b, aerobal_b = base_params[:6]
            X_b, Y_b, Z_b = self._compute_ggv_mesh(speeds, wdf_b, lltd_b, aerobal_b, L_b, tf_b, tr_b, tir_file)
            surf_b = self.axes_ggv.plot_surface(X_b, Y_b, Z_b, color='#ff7979', alpha=0.30, edgecolor='crimson', linewidth=0.4, label=f'Baseline (wd_f={wdf_b*100:.0f}%, lltd={lltd_b*100:.0f}%)')
            surf_b._facecolors2d = surf_b._facecolor3d
            surf_b._edgecolors2d = surf_b._edgecolor3d
            
        # 2. Compute and plot Optimized GGV surface
        X_opt, Y_opt, Z_opt = self._compute_ggv_mesh(speeds, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file)
        surf_opt = self.axes_ggv.plot_surface(X_opt, Y_opt, Z_opt, color='#00d2d3', alpha=0.48, edgecolor='navy', linewidth=0.5, label=f'Target (wd_f={wd_f*100:.0f}%, lltd={lltd*100:.0f}%)')
        surf_opt._facecolors2d = surf_opt._facecolor3d
        surf_opt._edgecolors2d = surf_opt._edgecolor3d
        
        self.axes_ggv.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
        self.axes_ggv.set_ylabel("Accelerazione Longitudinale $a_x$ [g]", fontsize=9)
        self.axes_ggv.set_zlabel("Velocità $u$ [m/s]", fontsize=9)
        self.axes_ggv.set_title("G-G-V Performance Bowl: Baseline vs Optimized Target", fontsize=10, fontweight='bold')
        leg_ggv = self.axes_ggv.legend(loc='upper right', fontsize=8)
        if leg_ggv:
            leg_ggv.set_draggable(True)
        self.canvas_ggv.draw()
        self.log("G-G-V Comparative Diagram generated successfully!")

    def plot_handling(self, v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, lmuy_mult=1.0,
                      K_phi_f=None, K_phi_r=None, q1=None, q2=None,
                      camber_f=None, camber_r=None, toe_f=None, toe_r=None):
        """Handling Diagram (SAE / ISO):
        Vertical axis: Lateral Acceleration a_y [g] (or v^2/(g*R))
        Horizontal axis (left of 0): Handling Curve (delta - L/R) = (alpha_f - alpha_r) [deg] (understeer curves left)
        Horizontal axis (right of 0): Kinematic Curvature rays L/R [deg] for each speed v
        Required steer angle delta = L/R + (alpha_f - alpha_r) is the horizontal distance between curve and ray.
        """
        self.log("Generating Handling Diagram...")
        QApplication.processEvents()
        
        try:
            speeds = [float(s.strip()) for s in self.hd_speeds_input.text().split(',')]
            hd_points = int(self.hd_points_input.text().strip())
        except ValueError:
            speeds = [8.0, 12.0, 16.0, 20.0, 25.0]
            hd_points = 120

        try:
            tire_unc_pct = float(self.tire_unc_input.text())
        except Exception:
            tire_unc_pct = 10.0

        # Determine the designated speed for tire uncertainty analysis (typical cornering speed: 12.0 m/s)
        unc_speed_target = 12.0 if any(abs(s - 12.0) < 1e-3 for s in speeds) else min(speeds, key=lambda s: abs(s - 12.0))

        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']
        while len(colors) < len(speeds):
            colors.extend(colors)

        ax = self.canvas_handling.axes
        ax.clear()
        import matplotlib.ticker as ticker

        df_target = None
        use_classic = self.rb_classic.isChecked()
        stop_at_peak = self.rb_peak.isChecked()
        max_ay_all = 0.0
        stability_frontier_points = []

        for idx, (v, c) in enumerate(zip(speeds, colors)):
            self.log(f"Handling: Computing speed {v:.0f} m/s ({idx+1}/{len(speeds)})...")
            QApplication.processEvents()
            
            if use_classic:
                df = mdo.generate_handling_sweep_classic(
                    v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
                    num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                    K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                    camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
            else:
                df = mdo.generate_handling_sweep(
                    v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
                    delta_max_deg=None, num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                    method='arclength', stop_at_peak=stop_at_peak,
                    K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                    camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)

            if abs(v - unc_speed_target) < 1e-3 or df_target is None:
                df_target = df

            if df.empty:
                continue

            ay = df['ay_g'].values
            if len(ay) == 0:
                continue
            max_ay_all = max(max_ay_all, float(np.max(ay)))

            # Understeer metric: (delta - L/R) = (alpha_f - alpha_r) [deg] (> 0 for understeer)
            if 'alpha_f_deg' in df.columns and 'alpha_r_deg' in df.columns:
                alpha_diff = (df['alpha_f_deg'] - df['alpha_r_deg']).values
            elif 'slip_diff_deg' in df.columns:
                alpha_diff = df['slip_diff_deg'].values
            else:
                continue

            # 1. Handling Curve: alpha_diff = (delta - L/R) = (alpha_f - alpha_r)
            # Standard Cartesian: Understeer (alpha_f > alpha_r) develops to the RIGHT (X > 0)
            ax.plot(alpha_diff, ay, color=c, linewidth=2.2, label=f'Handling ($v={v:.0f}$ m/s)')

            # Mark peak tire saturation point
            if len(ay) > 1:
                idx_max = np.argmax(ay)
                ax.plot(alpha_diff[idx_max], ay[idx_max], marker='o', color=c, markersize=5, zorder=5)
                stability_frontier_points.append((float(alpha_diff[idx_max]), float(ay[idx_max]), float(v)))

            # 2. Kinematic Curvature / Speed ray: L/R vs ay (X >= 0)
            ay_ray = np.linspace(0, np.max(ay), 40)
            ray_x = np.rad2deg((ay_ray * 9.81 * L) / (v**2))
            ax.plot(ray_x, ay_ray, color=c, linestyle=':', linewidth=1.5, alpha=0.7)

            # 3. Tire Uncertainty Envelope (evaluated at typical speed 12.0 m/s)
            if tire_unc_pct > 0.5 and (abs(v - unc_speed_target) < 1e-3 or len(speeds) <= 1):
                unc_frac = tire_unc_pct / 100.0
                lmuy_low = max(1.0 - unc_frac, 0.4)
                lmuy_high = 1.0 + unc_frac
                
                df_low = mdo.generate_handling_sweep(
                    v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_low * lmuy_mult,
                    delta_max_deg=None, num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                    method='arclength', stop_at_peak=stop_at_peak,
                    K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                    camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
                df_high = mdo.generate_handling_sweep(
                    v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_high * lmuy_mult,
                    delta_max_deg=None, num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                    method='arclength', stop_at_peak=stop_at_peak,
                    K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                    camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
                    
                if not df_low.empty and not df_high.empty:
                    d_low = (df_low['alpha_f_deg'] - df_low['alpha_r_deg']).values if 'alpha_f_deg' in df_low else df_low['slip_diff_deg'].values
                    d_high = (df_high['alpha_f_deg'] - df_high['alpha_r_deg']).values if 'alpha_f_deg' in df_high else df_high['slip_diff_deg'].values
                    ax.plot(d_low, df_low['ay_g'], color=c, linestyle='--', alpha=0.75, linewidth=1.4, label=f'Incertezza Gomme $-{tire_unc_pct:.0f}\\%$ ($v={v:.0f}$ m/s)')
                    ax.plot(d_high, df_high['ay_g'], color=c, linestyle='--', alpha=0.75, linewidth=1.4, label=f'Incertezza Gomme $+{tire_unc_pct:.0f}\\%$ ($v={v:.0f}$ m/s)')

        # 4. Frontiera di Stabilità (Stability Frontier connecting peak handling points across speeds)
        show_frontier = getattr(self, 'show_stability_frontier_cb', None) is None or _safe_is_checked(self.show_stability_frontier_cb)
        if show_frontier and len(stability_frontier_points) > 1:
            stability_frontier_points = sorted(stability_frontier_points, key=lambda pt: pt[2])
            sf_x = [pt[0] for pt in stability_frontier_points]
            sf_y = [pt[1] for pt in stability_frontier_points]
            frontier_label = r'Frontiera di Stabilità (Picchi $\partial a_y/\partial\delta=0$)' if not self.is_english_mode() else r'Stability Frontier (Peak $\partial a_y/\partial\delta=0$)'
            ax.plot(sf_x, sf_y, color='#d63031', linestyle='-.', linewidth=2.2, marker='D', markersize=6, label=frontier_label, zorder=6)

        # Vertical axis at X=0 and horizontal axis at Y=0
        ax.axvline(0, color='#2d3436', linewidth=1.8, linestyle='--', label='Linea Neutra ($K_{us}=0$)')
        ax.axhline(0, color='black', linewidth=1.2)

        # Set focused X-limits to highlight handling curve curvature & saturation
        x_min, x_max = ax.get_xlim()
        x_left = min(x_min * 1.25, -4.0)
        x_right = max(x_max * 1.08, 6.0)
        ax.set_xlim(x_left, x_right)
        y_top = max_ay_all * 1.15 if max_ay_all > 0 else 2.5
        ax.set_ylim(0, y_top)

        # Background tinting for Oversteer (left, X < 0) and Understeer (right, X > 0)
        ax.axvspan(x_left, 0, color='#ff7675', alpha=0.08, label='Zona Sovrasterzante ($K_{us} < 0$)')
        ax.axvspan(0, x_right, color='#74b9ff', alpha=0.08, label='Zona Sottosterzante ($K_{us} > 0$)')

        is_en = self.is_english_mode()
        q1_mm = (q1 * 1000.0) if q1 is not None else 46.0
        q2_mm = (q2 * 1000.0) if q2 is not None else 77.0
        
        camber_suffix = ""
        if camber_f is not None or camber_r is not None:
            cf = camber_f if camber_f is not None else -1.5
            cr = camber_r if camber_r is not None else -1.0
            camber_suffix = f" [$\\gamma_0=({cf:.1f}^\\circ, {cr:.1f}^\\circ)$]"

        if K_phi_f is not None and K_phi_r is not None:
            title_suffix = f"\n[$wd_f={wd_f*100:.1f}\\%$ | $K_{{\\phi,f}}={K_phi_f:.0f}$, $K_{{\\phi,r}}={K_phi_r:.0f}$ Nm/rad | $q_1={q1_mm:.0f}$, $q_2={q2_mm:.0f}$ mm $\\to$ LLTD = {lltd*100:.1f}%]{camber_suffix}"
        else:
            title_suffix = f"\n[$wd_f={wd_f*100:.1f}\\%$ | LLTD = {lltd*100:.1f}% | $q_1={q1_mm:.0f}$, $q_2={q2_mm:.0f}$ mm]{camber_suffix}"
            
        if is_en:
            ax.set_ylabel("Lateral Acceleration $a_y = \\frac{v^2}{g R}$ [g]", fontsize=11)
            ax.set_xlabel("Oversteer $\\leftarrow$ $(\\alpha_f - \\alpha_r) = (\\delta - L/R)$ [deg] $\\rightarrow$ Understeer & $L/R$ [deg]", fontsize=11)
            ax.set_title(f"Handling Diagram ($a_y/g$ vs $L\\cdot\\rho - \\delta$){title_suffix}", fontsize=10.0, fontweight='bold')
        else:
            ax.set_ylabel("Accelerazione Laterale $a_y = \\frac{v^2}{g R}$ [g]", fontsize=11)
            ax.set_xlabel("Sovrasterzo $\\leftarrow$ $(\\alpha_f - \\alpha_r) = (\\delta - L/R)$ [deg] $\\rightarrow$ Sottosterzo & $L/R$ [deg]", fontsize=11)
            ax.set_title(f"Handling Diagram ($a_y/g$ vs $L\\cdot\\rho - \\delta$){title_suffix}", fontsize=10.0, fontweight='bold')

        ax.grid(True, linestyle=':', alpha=0.6)
        leg_hd = ax.legend(fontsize='x-small', loc='upper left', ncol=2)
        if leg_hd:
            leg_hd.set_draggable(True)
        self.canvas_handling.figure.tight_layout()
        self.canvas_handling.draw()

        # ==========================================
        # 2. SUB-TAB 2: K_us VARIATION ANALYSIS PLOT
        # ==========================================
        if hasattr(self, 'axes_kus_ay') and hasattr(self, 'axes_kus_v'):
            self.axes_kus_ay.clear()
            self.axes_kus_v.clear()
            
            # --- Subplot 1: Instantaneous Kus(ay) vs ay ---
            for idx, (v, c) in enumerate(zip(speeds, colors)):
                # Retrieve computed sweep data if available
                df_sp = None
                if v == v_x and df_target is not None and not df_target.empty:
                    df_sp = df_target
                else:
                    if use_classic:
                        df_sp = mdo.generate_handling_sweep_classic(
                            v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
                            num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                            K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                            camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
                    else:
                        df_sp = mdo.generate_handling_sweep(
                            v, wd_f, lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
                            delta_max_deg=None, num_points=hd_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
                            method='arclength', stop_at_peak=stop_at_peak,
                            K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
                            camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r)
                            
                if df_sp is not None and not df_sp.empty and len(df_sp) > 5:
                    ay_vals = df_sp['ay_g'].values
                    if 'alpha_f_deg' in df_sp.columns and 'alpha_r_deg' in df_sp.columns:
                        ad_vals = (df_sp['alpha_f_deg'] - df_sp['alpha_r_deg']).values
                    elif 'slip_diff_deg' in df_sp.columns:
                        ad_vals = df_sp['slip_diff_deg'].values
                    else:
                        continue
                        
                    d_ay = np.diff(ay_vals)
                    d_ad = np.diff(ad_vals)
                    valid = np.abs(d_ay) > 1e-4
                    k_inst = np.zeros_like(d_ay)
                    k_inst[valid] = d_ad[valid] / d_ay[valid]
                    ay_m = 0.5 * (ay_vals[:-1] + ay_vals[1:])
                    
                    # Filter boundary singularities near peak
                    mask = (ay_m <= np.max(ay_vals) * 0.96) & (np.abs(k_inst) < 30.0)
                    if np.any(mask):
                        lw = 2.4 if v == v_x else 1.5
                        lbl = f'$v = {v:.0f}$ m/s' + (' (target)' if v == v_x else '')
                        self.axes_kus_ay.plot(ay_m[mask], k_inst[mask], color=c, linewidth=lw, label=lbl)
                        self.axes_kus_ay.scatter(ay_m[mask][0], k_inst[mask][0], color=c, s=25, zorder=5)

            self.axes_kus_ay.axhline(0, color='#2d3436', linestyle='--', linewidth=1.5, label='Neutro ($K_{us}=0$)')
            self.axes_kus_ay.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
            self.axes_kus_ay.set_ylabel("Gradiente Istantaneo $K_{us}(a_y)$ [$^\\circ$/g]", fontsize=9)
            self.axes_kus_ay.set_title("Variazione Locale $K_{us}(a_y) = \\frac{d(\\delta - L/R)}{d(a_y)}$ vs Accelerazione Laterale", fontsize=10, fontweight='bold')
            self.axes_kus_ay.grid(True, linestyle=':', alpha=0.6)
            leg_kus_ay = self.axes_kus_ay.legend(fontsize='x-small', loc='upper left', ncol=2)
            if leg_kus_ay:
                leg_kus_ay.set_draggable(True)
            
            y_lim_ay = self.axes_kus_ay.get_ylim()
            y_max_abs = max(abs(y_lim_ay[0]), abs(y_lim_ay[1]), 8.0)
            self.axes_kus_ay.set_ylim(-max(abs(y_lim_ay[0]), 3.0), y_max_abs)
            self.axes_kus_ay.axhspan(0, y_max_abs, color='#74b9ff', alpha=0.07)
            self.axes_kus_ay.axhspan(-y_max_abs, 0, color='#ff7675', alpha=0.07)
            self.axes_kus_ay.text(0.02, 0.90, "▲ SOTTOSTERZANTE ($K_{us}>0$)", transform=self.axes_kus_ay.transAxes, fontsize=8, fontweight='bold', color='#0984e3')
            self.axes_kus_ay.text(0.02, 0.08, "▼ SOVRASTERZANTE ($K_{us}<0$)", transform=self.axes_kus_ay.transAxes, fontsize=8, fontweight='bold', color='#d63031')

            # --- Subplot 2: Linear Kus vs Speed vx (Aero downforce effect) ---
            from src.mdo.mdo_global_optimizer import compute_stability_metrics
            v_sweep_arr = np.linspace(4.0, 32.0, 29)
            kus_v_vals = [compute_stability_metrics(L, wd_f, lltd, aero_bal, v_x=sp, tir_path=tir_file)['K_us_degg'] for sp in v_sweep_arr]
            
            self.axes_kus_v.plot(v_sweep_arr, kus_v_vals, color='#0984e3', linewidth=2.4, marker='o', markersize=4, label='Assetto Target Corrente')
            self.axes_kus_v.axvline(v_x, color='#e84393', linestyle=':', linewidth=1.8, label=f'Velocità Target $v_x = {v_x:.1f}$ m/s')
            self.axes_kus_v.axhline(0, color='#2d3436', linestyle='--', linewidth=1.5, label='Neutro ($K_{us}=0$)')
            
            # Recommended FSAE Target band (+1.0 to +2.5 deg/g)
            self.axes_kus_v.axhspan(1.0, 2.5, color='#00b894', alpha=0.12, label='Target FSAE Consigliato ($+1.0 \\dots +2.5^\\circ/g$)')
            
            self.axes_kus_v.set_xlabel("Velocità Veicolo $v_x$ [m/s]", fontsize=9)
            self.axes_kus_v.set_ylabel("Gradiente Lineare $K_{us}$ [$^\\circ$/g]", fontsize=9)
            self.axes_kus_v.set_title("Evoluzione Gradiente Lineare $K_{us}$ vs Velocità (Effetto Aerodinamico $C_z \\cdot v^2$ & Downforce)", fontsize=10, fontweight='bold')
            self.axes_kus_v.grid(True, linestyle=':', alpha=0.6)
            leg_kus_v = self.axes_kus_v.legend(fontsize='x-small', loc='upper right', ncol=2)
            if leg_kus_v:
                leg_kus_v.set_draggable(True)

            self.fig_kus.tight_layout()
            self.canvas_kus.draw()

        self.log("Handling Diagram and Kus Analysis generated successfully!")

        return df_target

    def plot_maps(self, v_x, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file, lmuy_mult=1.0):
        """Generates and plots the complete suite of Map of Achievable Performance (MAPs):
        - Sub-Tab 1 (Input Achievable Regions):
          * (delta, u): Y=u [m/s], X=delta [deg] with iso-ay, iso-rho and achievable boundary.
          * (u, ay): Y=ay [g], X=u [m/s] with const-delta curves, const-rho parabolas, and max-ay envelope.
          * (delta, ay): Y=ay [g], X=delta [deg] with const-u handling trim curves and iso-beta lines.
        - Sub-Tab 2 (Output & Mixed Achievable Regions):
          * (rho, beta): Y=beta [deg], X=rho [1/m] with const-u, const-delta, const-ay curves and state-space envelope.
          * (delta, rho): Y=rho [1/m], X=delta [deg] with const-u, const-ay and const-beta lines.
        """
        self.log("Generating MAPs (Map of Achievable Performance)...")
        QApplication.processEvents()
        
        # --- Read from MAPs Options sub-tab ---
        try:
            speeds = [float(s.strip()) for s in self.maps_highlight_v_input.text().split(',')]
        except Exception:
            speeds = [10.0, 15.0, 20.0, 25.0]

        def _parse_floats(text, fallback):
            try:
                return [float(x.strip()) for x in text.split(',')]
            except Exception:
                return fallback

        try:
            maps_v_min = float(self.maps_v_min_input.text())
        except Exception:
            maps_v_min = 4.0
        try:
            maps_v_max = float(self.maps_v_max_input.text())
        except Exception:
            maps_v_max = 30.0
        try:
            maps_n_speeds = int(self.maps_n_speeds_input.text())
        except Exception:
            maps_n_speeds = 50
        try:
            maps_n_points = int(self.maps_n_points_input.text())
        except Exception:
            maps_n_points = 70
        try:
            maps_ay_max = float(self.maps_ay_max_input.text())
        except Exception:
            maps_ay_max = 3.5

        iso_delta_levels = _parse_floats(self.maps_iso_delta_input.text(),
                                         [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0])
        iso_ay_levels    = _parse_floats(self.maps_iso_ay_input.text(),
                                         [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
        iso_rho_levels   = _parse_floats(self.maps_iso_rho_input.text(),
                                         [0.01, 0.02, 0.05, 0.10, 0.20])
        iso_beta_levels  = _parse_floats(self.maps_iso_beta_input.text(),
                                         [-10, -5, -3, -1, 0, 1, 3, 5, 10])

        show_iso_ay    = self.maps_show_iso_ay_cb.isChecked()
        show_iso_rho   = self.maps_show_iso_rho_cb.isChecked()
        show_iso_beta  = self.maps_show_iso_beta_cb.isChecked()
        show_envelope  = self.maps_show_envelope_cb.isChecked()
        show_iso_delta = self.maps_show_iso_delta_cb.isChecked()
        show_highlight = self.maps_show_highlight_cb.isChecked()

        colors_v = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']
        
        # Clear all 5 axes
        self.axes_map_delta_u.clear()
        self.axes_map_u_ay.clear()
        self.axes_map_delta_ay.clear()
        self.axes_map_rho_beta.clear()
        self.axes_map_delta_rho.clear()
        
        maps_data = mdo.generate_complete_maps(
            v_x=v_x, wd_f=wd_f, lltd=lltd, aero_bal=aero_bal, lmuy_mult=1.0,
            L=L, t_f=t_f, t_r=t_r, tir_file=tir_file, speeds_list=speeds,
            deltas_list=iso_delta_levels, ay_levels=iso_ay_levels, rho_levels=iso_rho_levels,
            v_min=maps_v_min, v_max=maps_v_max, n_speeds=maps_n_speeds,
            n_points=maps_n_points, ay_max=maps_ay_max
        )
        
        speed_curves = maps_data['speed_curves']
        delta_slices = maps_data['delta_slices']
        iso_ay_curves = maps_data['iso_ay_curves']
        iso_rho_curves = maps_data['iso_rho_curves']
        iso_beta_curves = maps_data['iso_beta_curves']
        envelope_df = maps_data['envelope_df']
        speeds_list = maps_data.get('speeds_list', speeds)
        
        # =========================================================================
        # SUB-TAB 1: INPUT ACHIEVABLE REGIONS
        # =========================================================================
        
        # --- 1.1 MAP (delta, u) ---
        ax1 = self.axes_map_delta_u
        if show_iso_ay:
            for ay_val, df_ay in iso_ay_curves.items():
                if not df_ay.empty:
                    d = df_ay['delta_deg'].values
                    u = df_ay['u'].values
                    lbl = f'{ay_val:.1f}g' if ay_val in iso_ay_levels[:3] + iso_ay_levels[-2:] else ""
                    ax1.plot(d, u, 'r--', alpha=0.55, linewidth=1.0, label=f'$a_y={lbl}$' if lbl else "")
                    ax1.plot(-d, u, 'r--', alpha=0.55, linewidth=1.0)
                
        if show_iso_rho:
            for rho_val, df_rho in iso_rho_curves.items():
                if not df_rho.empty:
                    d = df_rho['delta_deg'].values
                    u = df_rho['u'].values
                    lbl = f'{rho_val:.2f}'
                    ax1.plot(d, u, 'g:', alpha=0.55, linewidth=1.0, label=f'$\\rho={lbl}$')
                    ax1.plot(-d, u, 'g:', alpha=0.55, linewidth=1.0)
                
        if show_envelope and not envelope_df.empty:
            env_clean = envelope_df.sort_values('u').reset_index(drop=True)
            env_u = env_clean['u'].values
            env_d = env_clean['max_delta_deg'].values
            d_env_poly = np.concatenate([-env_d, env_d[::-1]])
            u_env_poly = np.concatenate([env_u, env_u[::-1]])
            ax1.plot(env_d, env_u, 'k-', linewidth=2.2, label='Limit Envelope')
            ax1.plot(-env_d, env_u, 'k-', linewidth=2.2)
            ax1.fill(d_env_poly, u_env_poly, color='#74b9ff', alpha=0.22)
            
        ax1.axvline(0, color='gray', linewidth=0.8)
        ax1.set_xlabel("Angolo Sterzo $\\delta$ [deg]", fontsize=9)
        ax1.set_ylabel("Velocità $u$ [m/s]", fontsize=9)
        ax1.set_title("MAP $(\\delta, u)$ - Input Region (Iso-$a_y$, Iso-$\\rho$)", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        leg1 = ax1.legend(fontsize='xx-small', loc='upper right', ncol=2)
        if leg1:
            leg1.set_draggable(True)
        
        # --- 1.2 MAP (u, ay) ---
        ax2 = self.axes_map_u_ay
        if show_iso_delta:
            for d_val, df_d in delta_slices.items():
                if not df_d.empty:
                    lw = 1.4 if d_val in [2.0, 4.0, 8.0, 15.0, 25.0] else 0.8
                    alpha = 0.85 if d_val in [2.0, 4.0, 8.0, 15.0, 25.0] else 0.4
                    lbl = f'$\\delta={d_val:.0f}^\\circ$' if d_val in [2.0, 4.0, 8.0, 15.0, 25.0] else ""
                    ax2.plot(df_d['u'], df_d['ay_g'], label=lbl, linewidth=lw, alpha=alpha)

        if show_iso_rho:
            u_parab = np.linspace(maps_v_min, maps_v_max, 80)
            for rho_val in iso_rho_levels:
                ay_parab = (rho_val * u_parab**2) / 9.81
                lbl = f'$\\rho={rho_val:.2f}$'
                ax2.plot(u_parab, ay_parab, 'k:', alpha=0.45, linewidth=0.9, label=lbl)
            
        if show_envelope and not envelope_df.empty:
            ax2.plot(envelope_df['u'], envelope_df['max_ay_g'], 'r-o', linewidth=2.2, markersize=3.0, label='Max $a_y(u)$ Limit')
            
        ax2.set_xlabel("Velocità $u$ [m/s]", fontsize=9)
        ax2.set_ylabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
        ax2.set_title("MAP $(u, a_y)$ - Speed vs Grip (Iso-$\\delta$, Iso-$\\rho$)", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        leg2 = ax2.legend(fontsize='xx-small', loc='upper left', ncol=2)
        if leg2:
            leg2.set_draggable(True)
        
        # --- 1.3 MAP (delta, ay) ---
        ax3 = self.axes_map_delta_ay
        if show_iso_beta:
            for df_b in iso_beta_curves:
                b_val = df_b['beta_deg'].iloc[0]
                ax3.plot(df_b['delta_deg'], df_b['ay_g'], color='gray', linestyle=':', alpha=0.40, linewidth=0.8)
                if len(df_b) > 0 and int(round(b_val)) in [-6, -4, -2, 0, 2, 4, 6] and abs(b_val - round(b_val)) < 0.1:
                    idx_lbl = len(df_b) // 2
                    ax3.text(df_b['delta_deg'].iloc[idx_lbl], df_b['ay_g'].iloc[idx_lbl], f'$\\beta$={b_val:.0f}°', fontsize=6, color='dimgray')
                
        if show_highlight:
            for idx, spd in enumerate(speeds_list):
                df_sp = speed_curves.get(spd)
                if df_sp is not None and not df_sp.empty:
                    d = df_sp['delta_deg'].values
                    ay = df_sp['ay_g'].values
                    d_sym = np.concatenate([-d[::-1], d])
                    ay_sym = np.concatenate([-ay[::-1], ay])
                    c = colors_v[idx % len(colors_v)]
                    lw = 2.4 if spd == v_x else 1.4
                    lbl = f'$u={spd:.0f}$ m/s' + (' (target)' if spd == v_x else '')
                    ax3.plot(d_sym, ay_sym, color=c, linewidth=lw, label=lbl)
                
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.axvline(0, color='black', linewidth=0.8)
        ax3.set_xlabel("Angolo Sterzo $\\delta$ [deg]", fontsize=9)
        ax3.set_ylabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
        ax3.set_title("MAP $(\\delta, a_y)$ - Trim Curves & Iso-$\\beta$", fontsize=10, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.6)
        leg3 = ax3.legend(fontsize='xx-small', loc='upper left', ncol=2)
        if leg3:
            leg3.set_draggable(True)
        
        self.fig_maps_input.tight_layout()
        self.canvas_maps_input.draw()
        
        # =========================================================================
        # SUB-TAB 2: OUTPUT & MIXED ACHIEVABLE REGIONS
        # =========================================================================
        
        # --- 2.1 MAP (rho, beta) - Output Region ---
        ax4 = self.axes_map_rho_beta
        for idx, spd in enumerate(speeds_list):
            df_sp = speed_curves.get(spd)
            if df_sp is not None and not df_sp.empty:
                rho = df_sp['rho'].values
                beta = df_sp['beta_deg'].values
                rho_sym = np.concatenate([-rho[::-1], rho])
                beta_sym = np.concatenate([-beta[::-1], beta])
                c = colors_v[idx % len(colors_v)]
                lw = 2.2 if spd == v_x else 1.3
                lbl = f'$u={spd:.0f}$ m/s' + (' (target)' if spd == v_x else '')
                ax4.plot(rho_sym, beta_sym, color=c, linewidth=lw, label=lbl)
                
        if show_iso_delta:
            for d_val, df_d in delta_slices.items():
                if not df_d.empty:
                    rho = df_d['rho'].values
                    beta = df_d['beta_deg'].values
                    ax4.plot(rho, beta, 'k--', alpha=0.35, linewidth=0.9)
                    ax4.plot(-rho, -beta, 'k--', alpha=0.35, linewidth=0.9)
                
        if show_iso_ay:
            for ay_val, df_ay in iso_ay_curves.items():
                if not df_ay.empty and (round(ay_val*10) % 4 == 0 or ay_val in [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]):
                    rho = df_ay['rho'].values
                    beta = df_ay['beta_deg'].values
                    ax4.plot(rho, beta, 'r:', alpha=0.65, linewidth=1.1, label=f'$a_y={ay_val:.1f}$g' if ay_val in [0.8, 1.6, 2.4] else "")
                    ax4.plot(-rho, -beta, 'r:', alpha=0.65, linewidth=1.1)
                
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.axvline(0, color='black', linewidth=0.8)
        ax4.set_xlabel("Curvatura $\\rho = r/u$ [1/m]", fontsize=9)
        ax4.set_ylabel("Angolo Assetto / Sideslip $\\beta$ [deg]", fontsize=9)
        ax4.set_title("MAP $(\\rho, \\beta)$ - Output Region (Iso-$u$, Iso-$\\delta$, Iso-$a_y$)", fontsize=10, fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.6)
        leg4 = ax4.legend(fontsize='xx-small', loc='upper left', ncol=2)
        if leg4:
            leg4.set_draggable(True)
        
        # --- 2.2 MAP (delta, rho) - Mixed Region ---
        ax5 = self.axes_map_delta_rho
        for idx, spd in enumerate(speeds_list):
            df_sp = speed_curves.get(spd)
            if df_sp is not None and not df_sp.empty:
                d = df_sp['delta_deg'].values
                rho = df_sp['rho'].values
                d_sym = np.concatenate([-d[::-1], d])
                rho_sym = np.concatenate([-rho[::-1], rho])
                c = colors_v[idx % len(colors_v)]
                lw = 2.2 if spd == v_x else 1.3
                lbl = f'$u={spd:.0f}$ m/s' + (' (target)' if spd == v_x else '')
                ax5.plot(d_sym, rho_sym, color=c, linewidth=lw, label=lbl)
                
        for ay_val, df_ay in iso_ay_curves.items():
            if not df_ay.empty and (round(ay_val*10) % 4 == 0 or ay_val in [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]):
                d = df_ay['delta_deg'].values
                rho = df_ay['rho'].values
                ax5.plot(d, rho, 'r:', alpha=0.65, linewidth=1.1, label=f'$a_y={ay_val:.1f}$g' if ay_val in [0.8, 1.6, 2.4] else "")
                ax5.plot(-d, -rho, 'r:', alpha=0.65, linewidth=1.1)
                
        for df_b in iso_beta_curves:
            b_val = df_b['beta_deg'].iloc[0]
            if int(round(b_val)) in [-6, -4, -2, 0, 2, 4, 6] and abs(b_val - round(b_val)) < 0.1:
                ax5.plot(df_b['delta_deg'], df_b['rho'], color='gray', linestyle='--', alpha=0.35, linewidth=0.8)
                
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.axvline(0, color='black', linewidth=0.8)
        ax5.set_xlabel("Angolo Sterzo $\\delta$ [deg]", fontsize=9)
        ax5.set_ylabel("Curvatura $\\rho = r/u$ [1/m]", fontsize=9)
        ax5.set_title("MAP $(\\delta, \\rho)$ - Mixed Region (Iso-$u$, Iso-$a_y$, Iso-$\\beta$)", fontsize=10, fontweight='bold')
        ax5.grid(True, linestyle=':', alpha=0.6)
        leg5 = ax5.legend(fontsize='xx-small', loc='upper left', ncol=2)
        if leg5:
            leg5.set_draggable(True)
        
        self.fig_maps_output.tight_layout()
        self.canvas_maps_output.draw()
        self.log("Complete Dense MAPs generated successfully in parallel!")

    def plot_dna_maps(self):
        """Calculates and renders Vehicle DNA Maps and Understeer / Stability analysis."""
        self.log("Calcolo Mappe del DNA del Veicolo e Stabilità Modale...")
        QApplication.processEvents()
        try:
            v_val = float(self.maps_dna_v_input.text())
        except Exception:
            v_val = 15.0
            
        vp = self._get_current_vehicle_params()
        tir_file = self.tir_combo.currentText()
        
        view_idx = self.maps_dna_view_combo.currentIndex()
        view_mode = 'dna_eigen' if view_idx == 0 else 'understeer_gradient'
        
        dna_data = compute_vehicle_dna_fields(
            v_x=v_val, wd_f=vp.wd_f, lltd=vp.lltd, aero_bal=vp.aero_balance,
            L=vp.L, t_f=vp.t_f, t_r=vp.t_r, tir_file=tir_file
        )
        
        plot_vehicle_dna_figures(self.fig_maps_dna, dna_data, view_mode=view_mode, language='en' if self.is_english_mode() else 'it')
        self.canvas_maps_dna.draw()
        self.log("Mappe di Stabilità Dinamica & Sensibilità generate con successo.")

    def toggle_guide_visibility(self):
        is_vis = self.guide_widget.isVisible()
        self.guide_widget.setVisible(not is_vis)
        if not is_vis:
            self.params_splitter.setSizes([600, 400])
        else:
            self.params_splitter.setSizes([1000, 0])

    def sync_dyn_params_from_optimizer(self):
        """Copies latest optimized target values into Dynamic Events manual inputs."""
        try:
            vp = self._get_current_vehicle_params_from_table()
            
            # Prioritize latest optimization results if available
            if hasattr(self, 'last_optimized_params') and self.last_optimized_params:
                opt_p = self.last_optimized_params
                L = opt_p.get('L', vp.L)
                tf = opt_p.get('t_f', vp.t_f)
                tr = opt_p.get('t_r', vp.t_r)
                wdf = opt_p.get('wd_f', vp.wd_f)
                lltd = opt_p.get('lltd', getattr(vp, 'lltd', 0.55))
                aerobal = opt_p.get('aero_balance', vp.aero_balance)
                self.log(f"[Dynamic Events] Sincronizzati valori OTTIMIZZATI: L={L:.3f}m, tf={tf:.3f}m, tr={tr:.3f}m, wd_f={wdf*100:.1f}%, lltd={lltd*100:.1f}%, aero={aerobal*100:.1f}%")
            else:
                L = vp.L
                tf = vp.t_f
                tr = vp.t_r
                wdf = vp.wd_f
                lltd = getattr(vp, 'lltd', 0.55)
                aerobal = vp.aero_balance
                self.log("[Dynamic Events] Nessuna ottimizzazione eseguita: copiati valori correnti della tabella.")

            self.dyn_m_input.setText(f"{vp.m:.1f}")
            self.dyn_pmax_input.setText(f"{getattr(vp, 'P_max_kw', 80.0):.1f}")
            self.dyn_tmax_input.setText(f"{getattr(vp, 'T_max_motor', 100.0):.1f}")
            self.dyn_L_input.setText(f"{L:.3f}")
            self.dyn_tf_input.setText(f"{tf:.3f}")
            self.dyn_tr_input.setText(f"{tr:.3f}")
            self.dyn_wdf_input.setText(f"{wdf:.3f}")
            self.dyn_lltd_input.setText(f"{lltd:.3f}")
            self.dyn_aero_input.setText(f"{aerobal:.3f}")
            self.dyn_lmux_input.setText(f"{getattr(vp, 'LMUX', 1.0):.2f}")
            self.dyn_lmuy_input.setText(f"{getattr(vp, 'LMUY', 1.0):.2f}")
        except Exception as e:
            self.log(f"[Dynamic Events] Errore sincronizzazione: {e}")

    def reset_dyn_params_default(self):
        """Resets manual inputs in Dynamic Events to baseline VehicleParams defaults."""
        from src.models.params_vehicle import VehicleParams
        vp = VehicleParams()
        self.dyn_m_input.setText(f"{vp.m:.1f}")
        self.dyn_pmax_input.setText(f"{getattr(vp, 'P_max_kw', 80.0):.1f}")
        self.dyn_tmax_input.setText(f"{getattr(vp, 'T_max_motor', 100.0):.1f}")
        self.dyn_L_input.setText(f"{vp.L:.3f}")
        self.dyn_tf_input.setText(f"{vp.t_f:.3f}")
        self.dyn_tr_input.setText(f"{vp.t_r:.3f}")
        self.dyn_wdf_input.setText(f"{vp.wd_f:.3f}")
        self.dyn_lltd_input.setText(f"{getattr(vp, 'lltd', 0.55):.3f}")
        self.dyn_aero_input.setText(f"{vp.aero_balance:.3f}")
        self.dyn_lmux_input.setText("1.00")
        self.dyn_lmuy_input.setText("1.00")
        self.log("[Dynamic Events] Parametri ripristinati ai valori di default.")

    def _get_current_vehicle_params_from_table(self):
        import copy
        from src.models.params_vehicle import VehicleParams
        if hasattr(self, '_active_vehicle_params') and self._active_vehicle_params is not None:
            vp = copy.deepcopy(self._active_vehicle_params)
        else:
            vp = VehicleParams()
        if hasattr(self, 'bounds_inputs') and self.bounds_inputs:
            for attr, item in self.bounds_inputs.items():
                if item and not item.get('digitizer') and item.get('val'):
                    try:
                        val = float(item['val'].text())
                        if hasattr(vp, attr):
                            setattr(vp, attr, val)
                    except (ValueError, TypeError):
                        pass
        return vp

    def _get_current_vehicle_params(self, for_dynamic=False):
        vp = self._get_current_vehicle_params_from_table()
        if for_dynamic:
            try:
                if hasattr(self, 'dyn_m_input') and self.dyn_m_input.text():
                    vp.m = float(self.dyn_m_input.text())
                if hasattr(self, 'dyn_pmax_input') and self.dyn_pmax_input.text():
                    vp.P_max_kw = float(self.dyn_pmax_input.text())
                if hasattr(self, 'dyn_tmax_input') and self.dyn_tmax_input.text():
                    vp.T_max_motor = float(self.dyn_tmax_input.text())
                if hasattr(self, 'dyn_fd_input') and self.dyn_fd_input.text():
                    vp.final_drive = float(self.dyn_fd_input.text())
                # --- FIX BUG 1: propagate grip / setup overrides from Dynamic Events panel ---
                if hasattr(self, 'dyn_lmux_input') and self.dyn_lmux_input.text():
                    vp.LMUX = float(self.dyn_lmux_input.text())
                if hasattr(self, 'dyn_lmuy_input') and self.dyn_lmuy_input.text():
                    vp.LMUY = float(self.dyn_lmuy_input.text())
                if hasattr(self, 'dyn_wdf_input') and self.dyn_wdf_input.text():
                    vp.wd_f = float(self.dyn_wdf_input.text())
                if hasattr(self, 'dyn_lltd_input') and self.dyn_lltd_input.text():
                    vp.lltd = float(self.dyn_lltd_input.text())
                if hasattr(self, 'dyn_aero_input') and self.dyn_aero_input.text():
                    vp.aero_balance = float(self.dyn_aero_input.text())
            except ValueError:
                pass
        return vp

    def plot_skidpad_telemetry(self, res, radius):
        self.fig_telem_skidpad.clear()
        
        if not res.get('success', False) or res.get('telemetry') is None or res['telemetry'].empty:
            ax = self.fig_telem_skidpad.add_subplot(111)
            ax.text(0.5, 0.5, "Simulazione Skidpad non convergente o nessun dato disponibile.", 
                    ha='center', va='center', fontsize=12, color='red')
            self.canvas_telem_skidpad.draw()
            return
            
        sweep_df = res.get('sweep', pd.DataFrame())
        telem_df = res.get('telemetry', pd.DataFrame())
        
        # 2x2 grid
        ax1 = self.fig_telem_skidpad.add_subplot(221)
        ax2 = self.fig_telem_skidpad.add_subplot(222)
        ax3 = self.fig_telem_skidpad.add_subplot(223)
        ax4 = self.fig_telem_skidpad.add_subplot(224)
        
        # 1. Sweep: Ay vs Speed and Steer
        if not sweep_df.empty:
            ax1.plot(sweep_df['ay_g'], sweep_df['v'], 'b-', linewidth=2.0, label='Velocità $v$ [m/s]')
            ax1.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
            ax1.set_ylabel("Velocità $v$ [m/s]", color='b', fontsize=9)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(sweep_df['ay_g'], sweep_df['delta_deg'], 'g--', linewidth=1.8, label='Sterzo $\\delta$ [deg]')
            ax1_twin.set_ylabel("Angolo Sterzo $\\delta$ [deg]", color='g', fontsize=9)
            ax1_twin.tick_params(axis='y', labelcolor='g')
            
            # Mark limit
            ax1.plot(res['ay'], res['v_max'], 'ro', markersize=6, label=f'Max $v={res["v_max"]:.2f}$ m/s')
            ax1.set_title(f"Skidpad: Limite di Grip e Sterzo ($R={radius:.2f}$ m)", fontsize=10, fontweight='bold')
            
        # 2. Wheel Vertical Loads (Fz) & Lateral Forces (Fy)
        if not telem_df.empty:
            ax2.plot(telem_df['t'], telem_df['Fz_fl'], 'b-', label='$F_{z,FL}$')
            ax2.plot(telem_df['t'], telem_df['Fz_fr'], 'b--', label='$F_{z,FR}$')
            ax2.plot(telem_df['t'], telem_df['Fz_rl'], 'r-', label='$F_{z,RL}$')
            ax2.plot(telem_df['t'], telem_df['Fz_rr'], 'r--', label='$F_{z,RR}$')
            ax2.set_xlabel("Tempo $t$ [s]", fontsize=9)
            ax2.set_ylabel("Carico Verticale $F_z$ [N]", color='black', fontsize=9)
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(telem_df['t'], telem_df['Fy_fl'] + telem_df['Fy_fr'], 'g-', linewidth=1.8, label='$F_{y,Front}$')
            ax2_twin.plot(telem_df['t'], telem_df['Fy_rl'] + telem_df['Fy_rr'], 'm-', linewidth=1.8, label='$F_{y,Rear}$')
            ax2_twin.set_ylabel("Forza Laterale Totale [N]", color='g', fontsize=9)
            ax2_twin.tick_params(axis='y', labelcolor='g')
            ax2.set_title("Carichi Verticali $F_z$ e Forze Laterali $F_y$", fontsize=10, fontweight='bold')
            ax2.legend(fontsize='xx-small', loc='upper left', ncol=2)
            
        # 3. Left vs Right Wheel Torques & Slip Angles
        if not telem_df.empty:
            ax3.plot(telem_df['t'], telem_df['Torque_rl_Nm'], 'b-', linewidth=2.0, label='Coppia Post. SX $T_{RL}$ [Nm]')
            ax3.plot(telem_df['t'], telem_df['Torque_rr_Nm'], 'r--', linewidth=2.0, label='Coppia Post. DX $T_{RR}$ [Nm]')
            ax3.set_xlabel("Tempo $t$ [s]", fontsize=9)
            ax3.set_ylabel("Coppia Ruota [Nm]", fontsize=9)
            ax3.grid(True, linestyle=':', alpha=0.6)
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(telem_df['t'], telem_df['alpha_fl_deg'], 'g:', label='Deriva Ant. $\\alpha_f$ [deg]')
            ax3_twin.plot(telem_df['t'], telem_df['alpha_rl_deg'], 'm:', label='Deriva Post. $\\alpha_r$ [deg]')
            ax3_twin.set_ylabel("Angolo di Deriva [deg]", color='g', fontsize=9)
            ax3_twin.tick_params(axis='y', labelcolor='g')
            ax3.set_title("Coppie Ruote SX/DX ($T_{RL}, T_{RR}$) e Angoli Deriva ($\\alpha$)", fontsize=10, fontweight='bold')
            ax3.legend(fontsize='xx-small', loc='upper left')
            ax3_twin.legend(fontsize='xx-small', loc='upper right')
            
        # 4. Top-Down 2D Track Map (Global Trajectory and Vehicle Footprints)
        if not telem_df.empty:
            theta_full = np.linspace(0, 2*np.pi, 200)
            # Track circles
            ax4.plot((radius + 1.5) * np.cos(theta_full), (radius + 1.5) * np.sin(theta_full), 'r--', linewidth=1.5, label='Coni Esterni')
            ax4.plot((radius - 1.5) * np.cos(theta_full), (radius - 1.5) * np.sin(theta_full), 'r--', linewidth=1.5, label='Coni Interni')
            ax4.plot(radius * np.cos(theta_full), radius * np.sin(theta_full), 'k:', linewidth=1.0, label='Centro Pista')
            
            # CG trajectory
            ax4.plot(telem_df['x_cg'], telem_df['y_cg'], 'b-', linewidth=2.2, label='Traiettoria Baricentro')
            
            # Plot vehicle footprint at 8 positions around the circle
            sample_indices = np.linspace(0, len(telem_df)-1, 8, dtype=int)
            for idx in sample_indices:
                row = telem_df.iloc[idx]
                car_x = [row['x_fl'], row['x_fr'], row['x_rr'], row['x_rl'], row['x_fl']]
                car_y = [row['y_fl'], row['y_fr'], row['y_rr'], row['y_rl'], row['y_fl']]
                ax4.plot(car_x, car_y, color='#2e7d32', linewidth=1.2)
                ax4.fill(car_x, car_y, color='#a5d6a7', alpha=0.4)
                
            ax4.set_xlabel("X Globale [m]", fontsize=9)
            ax4.set_ylabel("Y Globale [m]", fontsize=9)
            margin_str = f"{res.get('corridor_margin', 0.0):.2f}"
            ax4.set_title(f"Mappa Tracciato Skidpad - Lap: {res['time']:.3f} s (Margine: {margin_str} m)", fontsize=10, fontweight='bold')
            ax4.set_aspect('equal', 'box')
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.legend(fontsize='xx-small', loc='upper right')
            
        self.fig_telem_skidpad.tight_layout()
        self.canvas_telem_skidpad.draw()

    def plot_accel_telemetry(self, res, distance):
        self.fig_telem_accel.clear()
        
        if not res.get('success', False) or res.get('telemetry') is None or res['telemetry'].empty:
            ax = self.fig_telem_accel.add_subplot(111)
            ax.text(0.5, 0.5, "Simulazione Accelerazione fallita o nessun dato disponibile.", 
                    ha='center', va='center', fontsize=12, color='red')
            self.canvas_telem_accel.draw()
            return
            
        telem_df = res['telemetry']
        
        ax1 = self.fig_telem_accel.add_subplot(221)
        ax2 = self.fig_telem_accel.add_subplot(222)
        ax3 = self.fig_telem_accel.add_subplot(223)
        ax4 = self.fig_telem_accel.add_subplot(224)
        
        # 1. Speed vs Time
        ax1.plot(telem_df['t'], telem_df['v_kmh'], 'b-', linewidth=2.2, label='Velocità [km/h]')
        ax1.set_xlabel("Tempo $t$ [s]", fontsize=9)
        ax1.set_ylabel("Velocità [km/h]", color='b', fontsize=9)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(telem_df['t'], telem_df['s'], 'purple', linestyle='--', linewidth=1.5, label='Spazio [m]')
        ax1_twin.set_ylabel("Distanza [m]", color='purple', fontsize=9)
        ax1_twin.tick_params(axis='y', labelcolor='purple')
        ax1.set_title(f"Profilo di Velocità e Distanza ({distance:.0f} m)", fontsize=10, fontweight='bold')
        
        # 2. Longitudinal Acceleration vs Time
        ax2.plot(telem_df['t'], telem_df['ax_g'], 'r-', linewidth=2.2, label='Accelerazione $a_x$ [g]')
        ax2.plot(telem_df['t'].iloc[telem_df['ax_g'].argmax()], telem_df['ax_g'].max(), 'ro', markersize=6, 
                 label=f'Picco $a_x={telem_df["ax_g"].max():.2f}$ g')
        ax2.set_xlabel("Tempo $t$ [s]", fontsize=9)
        ax2.set_ylabel("Accelerazione Longitudinale $a_x$ [g]", fontsize=9)
        ax2.set_title(f"Accelerazione Longitudinale (Tempo: {res['time']:.3f} s)", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize='x-small', loc='best')
        
        # 3. Motor & Wheel Torques vs Speed
        if 'Torque_motor_Nm' in telem_df.columns:
            ax3.plot(telem_df['v_kmh'], telem_df['Torque_motor_Nm'], 'b-', linewidth=2.0, label='Coppia Motore $T_{mot}$ [Nm]')
            ax3.plot(telem_df['v_kmh'], telem_df['Torque_rl_Nm'], 'r--', linewidth=2.0, label='Coppia Ruota SX $T_{RL}$ [Nm]')
            ax3.plot(telem_df['v_kmh'], telem_df['Torque_rr_Nm'], 'g:', linewidth=2.0, label='Coppia Ruota DX $T_{RR}$ [Nm]')
            ax3.set_xlabel("Velocità [km/h]", fontsize=9)
            ax3.set_ylabel("Coppia [Nm]", fontsize=9)
            ax3.set_title("Erogazione Coppia Motore & Ruote SX/DX", fontsize=10, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)
            ax3.legend(fontsize='x-small', loc='best')
            
        # 4. Forces Breakdown & Power vs Speed
        ax4.plot(telem_df['v_kmh'], telem_df['F_drive'], 'g-', linewidth=2.2, label='Forza Motrice $F_{drive}$')
        ax4.plot(telem_df['v_kmh'], telem_df['F_motor'], 'b--', linewidth=1.5, label='Limite Motore $F_{motor}$')
        ax4.plot(telem_df['v_kmh'], telem_df['F_grip'], 'r-.', linewidth=1.5, label='Limite Grip $F_{grip}$')
        ax4.plot(telem_df['v_kmh'], telem_df['F_drag'], 'k:', linewidth=1.5, label='Resistenza Aero $F_{drag}$')
        ax4.set_xlabel("Velocità [km/h]", fontsize=9)
        ax4.set_ylabel("Forza [N]", fontsize=9)
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        ax4_twin = ax4.twinx()
        if 'p_elec_kw' in telem_df.columns:
            ax4_twin.plot(telem_df['v_kmh'], telem_df['p_elec_kw'], color='#e74c3c', linestyle='-', linewidth=2.0, label='P_elec Batteria [kW] (Lim 80kW)')
        ax4_twin.plot(telem_df['v_kmh'], telem_df['power_kw'], color='#f39c12', linestyle='--', linewidth=1.8, label='P_wheel Utile Ruote [kW]')
        ax4_twin.set_ylabel("Potenza [kW]", color='#e67e22', fontsize=9)
        ax4_twin.set_ylim(0, 95)
        ax4_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax4.set_title("Bilancio Forze Longitudinali & Potenza [kW]", fontsize=10, fontweight='bold')
        ax4.legend(fontsize='xx-small', loc='upper left')
        ax4_twin.legend(fontsize='xx-small', loc='upper right')
        
        self.fig_telem_accel.tight_layout()
        self.canvas_telem_accel.draw()

    def export_skidpad_csv(self):
        if not hasattr(self, 'last_skidpad_res') or not self.last_skidpad_res.get('success', False) or self.last_skidpad_res.get('telemetry') is None or self.last_skidpad_res['telemetry'].empty:
            QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria Skidpad disponibile da esportare. Esegui prima la simulazione.")
            return
        
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Skidpad CSV", "skidpad_telemetry.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.last_skidpad_res['telemetry'].to_csv(path, index=False)
                self.log(f"Telemetria Skidpad esportata con successo in: {path}")
                QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
            except Exception as e:
                self.log(f"Errore durante l'esportazione CSV Skidpad: {e}")
                QMessageBox.critical(self, "Errore Esportazione", str(e))

    def export_accel_csv(self):
        if not hasattr(self, 'last_accel_res') or not self.last_accel_res.get('success', False) or self.last_accel_res.get('telemetry') is None or self.last_accel_res['telemetry'].empty:
            QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria Acceleration disponibile da esportare. Esegui prima la simulazione.")
            return
        
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Acceleration CSV", "acceleration_telemetry.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.last_accel_res['telemetry'].to_csv(path, index=False)
                self.log(f"Telemetria Acceleration esportata con successo in: {path}")
                QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
            except Exception as e:
                self.log(f"Errore durante l'esportazione CSV Acceleration: {e}")
                QMessageBox.critical(self, "Errore Esportazione", str(e))

    def run_skidpad_event(self):
        try:
            radius = float(self.skid_radius_input.text())
        except ValueError:
            self.log("Invalid Skidpad Radius!")
            return
            
        self.log(f"Running Skidpad Simulation (Radius = {radius} m)...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params(for_dynamic=True)
        use_7dof = (self.model_combo.currentIndex() == 1)
        tir_file = self.tir_combo.currentText()
        
        res = simulate_skidpad(vp, radius=radius, use_7dof=use_7dof, tir_path=tir_file)
        self.last_skidpad_res = res
        
        if res['success']:
            self.log(f"--- SKIDPAD RESULT (m={vp.m:.1f}kg, L={vp.L:.3f}m, tf={vp.t_f:.3f}m, LMUY={vp.LMUY:.2f}) ---")
            self.log(f"Time: {res['time']:.3f} s")
            self.log(f"Max Velocity: {res['v_max']:.2f} m/s ({res['v_max']*3.6:.1f} km/h)")
            self.log(f"Sustained Ay: {res['ay']:.3f} g")
            self.log(f"Steer Angle δ: {res['delta_deg']:.2f}°")
            self.log(f"Sideslip β: {res['beta_deg']:.2f}°")
            self.log(f"Heading Error ψ_err: {res['psi_err_deg']:.2f}°")
            self.log(f"Corridor Margin: {res['corridor_margin']:.2f} m")
            
            self.plot_skidpad_telemetry(res, radius)
            self.telemetry_subtabs.setCurrentWidget(self.tab_telem_skidpad)
            self.tabs.setCurrentWidget(self.tab_telemetry)
        else:
            self.log("Skidpad Simulation Failed (did not converge).")
            self.plot_skidpad_telemetry(res, radius)

    def run_accel_event(self):
        try:
            distance = float(self.accel_dist_input.text())
        except ValueError:
            self.log("Invalid Acceleration Distance!")
            return
            
        self.log(f"Running Acceleration Simulation (Distance = {distance} m)...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params(for_dynamic=True)
        if hasattr(self, 'dyn_fd_input'):
            try:
                vp.final_drive = float(self.dyn_fd_input.text())
            except ValueError:
                pass
                
        res = simulate_acceleration(vp, distance=distance, is_awd=False)
        self.last_accel_res = res
        
        if res['success']:
            t_mot = getattr(vp, 'T_max_motor', 100.0)
            fd = getattr(vp, 'final_drive', 4.0)
            t_wheel = t_mot * fd
            self.log(f"--- ACCELERATION RESULT (P={getattr(vp, 'P_max_kw', 80.0):.1f}kW, T_mot={t_mot:.1f}Nm, T_wheel={t_wheel:.1f}Nm, LMUX={vp.LMUX:.2f}) ---")
            self.log(f"Time: {res['time']:.3f} s")
            self.log(f"Terminal Velocity: {res['v_max']:.2f} m/s ({res['v_max']*3.6:.1f} km/h)")
            self.log(f"Max Acceleration ax: {res['a_max_g']:.2f} g")
            
            self.plot_accel_telemetry(res, distance)
            self.telemetry_subtabs.setCurrentWidget(self.tab_telem_accel)
            self.tabs.setCurrentWidget(self.tab_telemetry)
        else:
            self.log("Acceleration Simulation Failed.")
            self.plot_accel_telemetry(res, distance)

    def run_step_steer_event(self):
        try:
            vx = float(self.step_vx_input.text())
            steer_deg = float(self.step_steer_input.text())
            t_step = float(self.step_tstep_input.text())
            t_ramp = float(self.step_tramp_input.text()) if hasattr(self, 'step_tramp_input') else 0.08
            t_end = float(self.step_tend_input.text()) if hasattr(self, 'step_tend_input') else 4.0
            dt = float(self.step_dt_input.text()) if hasattr(self, 'step_dt_input') else 0.002
            tv_torque = float(self.step_tv_input.text())
            maintain_speed = self.step_maintain_speed_cb.isChecked() if hasattr(self, 'step_maintain_speed_cb') else True
        except ValueError:
            self.log("Valori non validi per Step Steer!")
            return
            
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        idx = self.model_combo.currentIndex()
        model_key = model_names[idx] if idx < len(model_names) else "14-DOF"
        # Step steer ODE requires 7-DOF or 14-DOF
        active_model = "14-DOF" if model_key == "14-DOF" else "7-DOF"
        
        speed_mode_str = "Cruise Maintain Speed" if maintain_speed else "Coastdown Free Decel"
        self.log(f"Esecuzione Step Steer ({active_model}, Vx={vx} m/s, Steer={steer_deg}°, dt={dt}s, t_end={t_end}s, t_ramp={t_ramp}s, TV={tv_torque} Nm, Mode={speed_mode_str})...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params(for_dynamic=True)
        tir_file = self.tir_combo.currentText()
        
        res = simulate_step_steer(vp, v_x=vx, steer_deg=steer_deg, t_end=t_end, dt=dt,
                                  t_step=t_step, t_ramp=t_ramp, torque_vectoring_nm=tv_torque,
                                  maintain_speed=maintain_speed, model_type=active_model, tir_path=tir_file)
        self.last_step_res = res
        
        if res['success']:
            self.log(f"--- STEP STEER RISULTATI ({active_model}) ---")
            self.log(f"Peak Lateral Accel: {res['ay_peak_g']:.3f} g")
            self.log(f"Steady-State Yaw Rate: {res['r_ss_deg_s']:.2f} °/s (Peak: {res['r_peak_deg_s']:.2f} °/s)")
            self.log(f"Yaw Rate Overshoot: {res['overshoot_pct']:.1f} %")
            self.log(f"Passo dt utilizzato: {res['dt_used']:.4f} s ({int(res['t_end_used']/res['dt_used'])} step RK4 calcolati)")
            
            self.plot_step_steer_telemetry(res, active_model)
            self.telemetry_subtabs.setCurrentWidget(self.tab_telem_step)
            self.tabs.setCurrentWidget(self.tab_telemetry)
        else:
            self.log("Step Steer Simulation Failed.")

    def run_dlc_event(self):
        try:
            vx = float(self.step_vx_input.text())
            t_end = float(self.step_tend_input.text()) if hasattr(self, 'step_tend_input') else 5.0
            dt = float(self.step_dt_input.text()) if hasattr(self, 'step_dt_input') else 0.002
            maintain_speed = self.step_maintain_speed_cb.isChecked() if hasattr(self, 'step_maintain_speed_cb') else True
        except ValueError:
            vx = 20.0
            t_end = 5.0
            dt = 0.002
            maintain_speed = True
            
        model_names = ["3-DOF", "7-DOF", "14-DOF"]
        idx = self.model_combo.currentIndex()
        model_key = model_names[idx] if idx < len(model_names) else "14-DOF"
        active_model = "14-DOF" if model_key == "14-DOF" else "7-DOF"
        
        self.log(f"Esecuzione Double Lane Change ({active_model}, Entry Speed={vx} m/s / {vx*3.6:.1f} km/h, dt={dt}s, t_end={t_end}s)...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params(for_dynamic=True)
        tir_file = self.tir_combo.currentText()
        
        res = simulate_double_lane_change(vp, v_entry=vx, t_end=t_end, dt=dt,
                                          maintain_speed=maintain_speed, model_type=active_model, tir_path=tir_file)
        self.last_step_res = res
        
        if res['success']:
            self.log(f"--- DOUBLE LANE CHANGE RISULTATI ({active_model}) ---")
            self.log(f"Max Lateral Accel: {res['max_ay_g']:.3f} g")
            self.log(f"Max Roll Angle: {res['max_phi_deg']:.2f}°")
            self.log(f"Max Lateral Displacement: {res['max_lateral_disp_m']:.2f} m")
            self.log(f"ISO 3888-2 Corridor Clearance: {'PASSED [OK]' if res['cone_clearance_ok'] else 'MARGINAL [!]'}")
            
            self.plot_dlc_telemetry(res, active_model)
            self.telemetry_subtabs.setCurrentWidget(self.tab_telem_step)
            self.tabs.setCurrentWidget(self.tab_telemetry)
        else:
            self.log("Double Lane Change Simulation Failed.")

    def on_autox_track_changed(self, idx):
        from src.mdo.mdo_direct_collocation import build_track_geometry, load_custom_track_file
        track_keys = ["fsae_autocross", "fsae_skidpad", "fsae_endurance", "fsae_acceleration"]
        if idx < len(track_keys):
            t_type = track_keys[idx]
            self.current_autox_track = build_track_geometry(track_type=t_type)
            if hasattr(self, 'autox_len_input'):
                self.autox_len_input.setText(f"{self.current_autox_track['s'][-1]:.1f}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Carica File Tracciato", "tracks", "Track Files (*.csv *.json);;All Files (*)")
            if not path:
                self.combo_autox_track.blockSignals(True)
                self.combo_autox_track.setCurrentIndex(0)
                self.combo_autox_track.blockSignals(False)
                return
            self.current_autox_track = load_custom_track_file(path)
            if hasattr(self, 'autox_len_input'):
                self.autox_len_input.setText(f"{self.current_autox_track['s'][-1]:.1f}")
        self.log(f"[TRACCIATO] Layout impostato su: {self.combo_autox_track.currentText()} (Lunghezza={self.current_autox_track['s'][-1]:.1f}m)")

    def run_autox_endurance_event(self):
        try:
            track_len = float(self.autox_len_input.text())
            ds = float(self.autox_ds_input.text()) if hasattr(self, 'autox_ds_input') else 0.50
        except ValueError:
            track_len = 850.0
            ds = 0.50
            
        evt_idx = self.autox_event_combo.currentIndex()
        event_type = "Endurance" if evt_idx == 1 else "AutoX"
        num_laps = 22 if evt_idx == 1 else 1
        
        model_names = ["3-DOF", "7-DOF", "14-DOF", "ALL"]
        idx = self.model_combo.currentIndex()
        model_key = model_names[idx] if idx < len(model_names) else "14-DOF"
        
        self.log(f"Esecuzione Simulazione {event_type} ({num_laps} Giri, Tracciato={track_len} m, ds={ds} m, Modello={model_key})...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params(for_dynamic=True)
        tir_file = self.tir_combo.currentText()
        track_dict = getattr(self, 'current_autox_track', None)
        
        res = simulate_autox_endurance(vp, event_type=event_type, num_laps=num_laps,
                                       track_length_m=track_len, ds=ds, model_type=model_key, tir_path=tir_file,
                                       track_dict=track_dict)
        self.last_autox_res = res
        
        if res['success']:
            self.log(f"--- {event_type.upper()} SIMULATION RESULTS ---")
            self.log(f"Lap Time: {res['lap_time_s']:.2f} s | Total Time: {res['total_time_s']/60.0:.2f} min")
            self.log(f"Max Speed: {res['v_max_kmh']:.1f} km/h | Avg Speed: {res['v_avg_kmh']:.1f} km/h")
            self.log(f"Energy Consumed per Lap: {res['energy_lap_kwh']*1000.0:.1f} Wh ({res['energy_lap_kwh']:.3f} kWh)")
            self.log(f"Total Energy Consumed: {res['total_energy_kwh']:.2f} kWh (Efficiency: {res['efficiency_kwh_km']:.3f} kWh/km)")
            self.log(f"Final Battery SoC: {res['battery_soc_end_pct']:.1f} % (Capacity: {getattr(vp, 'battery_capacity_kwh', 6.8):.1f} kWh)")
            
            self.plot_autox_endurance_telemetry(res)
            self.telemetry_subtabs.setCurrentWidget(self.tab_telem_autox)
            self.tabs.setCurrentWidget(self.tab_telemetry)
        else:
            self.log(f"{event_type} Simulation Failed.")

    def plot_step_steer_telemetry(self, res, model_type):
        df = res['telemetry']
        self.fig_telem_step.clear()
        
        ax1 = self.fig_telem_step.add_subplot(321)
        ax2 = self.fig_telem_step.add_subplot(322)
        ax3 = self.fig_telem_step.add_subplot(323)
        ax4 = self.fig_telem_step.add_subplot(324)
        ax5 = self.fig_telem_step.add_subplot(325)
        ax6 = self.fig_telem_step.add_subplot(326)
        
        # Subplot 1: Velocities & Steer
        ax1.plot(df['t'], df['v_kmh'], 'b-', linewidth=1.8, label='$v_x$ [km/h]')
        ax1.plot(df['t'], df['v_y']*3.6, 'c--', linewidth=1.4, label='$v_y$ [km/h]')
        ax1_t = ax1.twinx()
        ax1_t.plot(df['t'], df['steer_deg'], 'k:', linewidth=1.6, label='Sterzo $\\delta$ [°]')
        ax1.set_ylabel("Velocità [km/h]", fontsize=8)
        ax1_t.set_ylabel("Sterzo [°]", color='black', fontsize=8)
        ax1.set_title("Velocità & Input Angolo Sterzo", fontsize=9, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=7, loc='upper left')
        
        # Subplot 2: Yaw Rate & Lateral Acceleration
        ax2.plot(df['t'], df['yaw_rate_deg_s'], 'r-', linewidth=1.8, label='Yaw Rate $r$ [°/s]')
        if 'r_dot_deg_s2' in df.columns:
            ax2.plot(df['t'], df['r_dot_deg_s2']/10.0, 'orange', linestyle=':', linewidth=1.2, label='$\\dot{r}/10$ [°/s²]')
        ax2_t = ax2.twinx()
        ax2_t.plot(df['t'], df['ay_g'], 'm--', linewidth=1.8, label='$a_y$ [g]')
        ax2.set_ylabel("Yaw Rate [°/s]", color='r', fontsize=8)
        ax2_t.set_ylabel("Acc. Lat. [g]", color='m', fontsize=8)
        ax2.set_title("Risposta Dinamica: Imbardata & Accelerazione Laterale", fontsize=9, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=7, loc='upper left')
        
        # Subplot 3: 4-Wheel Dynamic Normal Loads Fz [N]
        if 'Fz_fl_N' in df.columns:
            ax3.plot(df['t'], df['Fz_fl_N'], color='#0984e3', linewidth=1.5, label='Fz FL')
            ax3.plot(df['t'], df['Fz_fr_N'], color='#74b9ff', linestyle='--', linewidth=1.5, label='Fz FR')
            ax3.plot(df['t'], df['Fz_rl_N'], color='#d63031', linewidth=1.5, label='Fz RL')
            ax3.plot(df['t'], df['Fz_rr_N'], color='#ff7675', linestyle='--', linewidth=1.5, label='Fz RR')
            ax3.set_ylabel("Carico Normale $F_z$ [N]", fontsize=8)
            ax3.set_title("Trasferimento di Carico su 4 Ruote (LLTD & Roll/Pitch)", fontsize=9, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)
            ax3.legend(fontsize=7, loc='best', ncol=2)
            
        # Subplot 4: 4-Wheel Slip Angles Alpha [deg]
        if 'alpha_fl_deg' in df.columns:
            ax4.plot(df['t'], df['alpha_fl_deg'], color='#0984e3', linewidth=1.5, label='$\\alpha_{FL}$')
            ax4.plot(df['t'], df['alpha_fr_deg'], color='#74b9ff', linestyle='--', linewidth=1.5, label='$\\alpha_{FR}$')
            ax4.plot(df['t'], df['alpha_rl_deg'], color='#d63031', linewidth=1.5, label='$\\alpha_{RL}$')
            ax4.plot(df['t'], df['alpha_rr_deg'], color='#ff7675', linestyle='--', linewidth=1.5, label='$\\alpha_{RR}$')
            ax4.set_ylabel("Angolo di Deriva $\\alpha$ [°]", fontsize=8)
            ax4.set_title("Angoli di Deriva Pneumatici (Saturazione Assale Ant/Post)", fontsize=9, fontweight='bold')
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.legend(fontsize=7, loc='best', ncol=2)
            
        # Subplot 5: 4-Wheel Lateral Forces & Grip Utilization
        if 'Fy_fl_N' in df.columns:
            ax5.plot(df['t'], df['Fy_fl_N'], color='#0984e3', linewidth=1.4, label='Fy FL')
            ax5.plot(df['t'], df['Fy_fr_N'], color='#74b9ff', linestyle='--', linewidth=1.4, label='Fy FR')
            ax5.plot(df['t'], df['Fy_rl_N'], color='#d63031', linewidth=1.4, label='Fy RL')
            ax5.plot(df['t'], df['Fy_rr_N'], color='#ff7675', linestyle='--', linewidth=1.4, label='Fy RR')
            ax5_t = ax5.twinx()
            if 'util_fl_pct' in df.columns:
                ax5_t.plot(df['t'], df['util_fl_pct'], color='purple', linestyle=':', alpha=0.6, label='Grip Util FL %')
            ax5.set_xlabel("Tempo $t$ [s]", fontsize=8)
            ax5.set_ylabel("Forza Laterale $F_y$ [N]", fontsize=8)
            ax5_t.set_ylabel("Utilizzo Grip [%]", color='purple', fontsize=8)
            ax5.set_title("Forze Laterali Pneumatici & Aderenza Disponibile", fontsize=9, fontweight='bold')
            ax5.grid(True, linestyle=':', alpha=0.6)
            ax5.legend(fontsize=7, loc='upper left', ncol=2)
            
        # Subplot 6: Chassis Attitude & Wheel Speeds
        if 'phi_deg' in df.columns:
            ax6.plot(df['t'], df['phi_deg'], 'g-', linewidth=1.8, label='Rollio $\\phi$ [°]')
        if 'theta_deg' in df.columns:
            ax6.plot(df['t'], df['theta_deg'], 'k--', linewidth=1.4, label='Beccheggio $\\theta$ [°]')
        if 'beta_deg' in df.columns:
            ax6.plot(df['t'], df['beta_deg'], 'c-.', linewidth=1.4, label='Sideslip $\\beta$ [°]')
        ax6.set_xlabel("Tempo $t$ [s]", fontsize=8)
        ax6.set_ylabel("Angoli Chassis [°]", fontsize=8)
        ax6.set_title("Angoli d'Assetto Scocca (Rollio, Beccheggio, Sideslip)", fontsize=9, fontweight='bold')
        ax6.grid(True, linestyle=':', alpha=0.6)
        ax6.legend(fontsize=7, loc='best')
        
        self.fig_telem_step.suptitle(f"Telemetria Completa Dinamica: Step Steer ({model_type})", fontsize=10, fontweight='bold')
        self.fig_telem_step.tight_layout()
        self.canvas_telem_step.draw()

    def plot_dlc_telemetry(self, res, model_type):
        df = res['telemetry']
        self.fig_telem_step.clear()
        
        ax1 = self.fig_telem_step.add_subplot(321)
        ax2 = self.fig_telem_step.add_subplot(322)
        ax3 = self.fig_telem_step.add_subplot(323)
        ax4 = self.fig_telem_step.add_subplot(324)
        ax5 = self.fig_telem_step.add_subplot(325)
        ax6 = self.fig_telem_step.add_subplot(326)
        
        # Subplot 1: Trajectory (X vs Y Global)
        ax1.plot(df['x_pos_m'], df['y_pos_m'], 'b-', linewidth=2.2, label='Traiettoria Veicolo')
        ax1.axhline(0.0, color='gray', linestyle=':')
        ax1.axhline(3.5, color='gray', linestyle=':', label='Corsia di Scarto (+3.5m)')
        ax1.set_xlabel("Spostamento Longitudinale X [m]", fontsize=8)
        ax1.set_ylabel("Spostamento Laterale Y [m]", fontsize=8)
        ax1.set_title("Traiettoria nel Piano (ISO 3888-2 Double Lane Change)", fontsize=9, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=7, loc='best')
        
        # Subplot 2: Steer & Yaw Rate
        ax2.plot(df['t'], df['steer_deg'], 'k-', linewidth=1.8, label='Sterzo $\\delta$ [°]')
        ax2_t = ax2.twinx()
        ax2_t.plot(df['t'], df['yaw_rate_deg_s'], 'r--', linewidth=2.0, label='Yaw Rate $r$ [°/s]')
        ax2.set_xlabel("Tempo $t$ [s]", fontsize=8)
        ax2.set_ylabel("Sterzo [°]", color='k', fontsize=8)
        ax2_t.set_ylabel("Yaw Rate [°/s]", color='r', fontsize=8)
        ax2.set_title("Input Sterzo & Risposta Imbardata", fontsize=9, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Subplot 3: 4-Wheel Dynamic Loads Fz [N]
        if 'Fz_fl_N' in df.columns:
            ax3.plot(df['t'], df['Fz_fl_N'], color='#0984e3', linewidth=1.5, label='Fz FL')
            ax3.plot(df['t'], df['Fz_fr_N'], color='#74b9ff', linestyle='--', linewidth=1.5, label='Fz FR')
            ax3.plot(df['t'], df['Fz_rl_N'], color='#d63031', linewidth=1.5, label='Fz RL')
            ax3.plot(df['t'], df['Fz_rr_N'], color='#ff7675', linestyle='--', linewidth=1.5, label='Fz RR')
            ax3.set_ylabel("Carico Normale $F_z$ [N]", fontsize=8)
            ax3.set_title("Trasferimento di Carico 4 Ruote in Transitorio", fontsize=9, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)
            ax3.legend(fontsize=7, loc='best', ncol=2)
            
        # Subplot 4: 4-Wheel Slip Angles Alpha [deg]
        if 'alpha_fl_deg' in df.columns:
            ax4.plot(df['t'], df['alpha_fl_deg'], color='#0984e3', linewidth=1.5, label='$\\alpha_{FL}$')
            ax4.plot(df['t'], df['alpha_fr_deg'], color='#74b9ff', linestyle='--', linewidth=1.5, label='$\\alpha_{FR}$')
            ax4.plot(df['t'], df['alpha_rl_deg'], color='#d63031', linewidth=1.5, label='$\\alpha_{RL}$')
            ax4.plot(df['t'], df['alpha_rr_deg'], color='#ff7675', linestyle='--', linewidth=1.5, label='$\\alpha_{RR}$')
            ax4.set_ylabel("Angolo di Deriva $\\alpha$ [°]", fontsize=8)
            ax4.set_title("Angoli di Deriva 4 Ruote", fontsize=9, fontweight='bold')
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.legend(fontsize=7, loc='best', ncol=2)
            
        # Subplot 5: Lateral Acceleration
        ax5.plot(df['t'], df['ay_g'], 'm-', linewidth=2.0, label='Acc. Laterale $a_y$ [g]')
        ax5.set_xlabel("Tempo $t$ [s]", fontsize=8)
        ax5.set_ylabel("Acc. Laterale $a_y$ [g]", fontsize=8)
        ax5.set_title(f"Accelerazione Laterale Dinamica (Max: {res['max_ay_g']:.2f} g)", fontsize=9, fontweight='bold')
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.legend(fontsize=8, loc='best')
        
        # Subplot 6: Roll Angle & Attitude
        if 'phi_deg' in df.columns:
            ax6.plot(df['t'], df['phi_deg'], 'g-', linewidth=2.0, label='Roll Angle $\\phi$ [°]')
        if 'theta_deg' in df.columns:
            ax6.plot(df['t'], df['theta_deg'], 'k--', linewidth=1.4, label='Pitch $\\theta$ [°]')
        ax6.set_xlabel("Tempo $t$ [s]", fontsize=8)
        ax6.set_ylabel("Angolo d'Assetto [°]", fontsize=8)
        ax6.set_title(f"Rollio & Beccheggio (Max $\\phi$: {res['max_phi_deg']:.2f}°)", fontsize=9, fontweight='bold')
        ax6.grid(True, linestyle=':', alpha=0.6)
        ax6.legend(fontsize=8, loc='best')
        
        self.fig_telem_step.suptitle(f"Telemetria Completa Scarto Ostacolo: Double Lane Change ({model_type})", fontsize=10, fontweight='bold')
        self.fig_telem_step.tight_layout()
        self.canvas_telem_step.draw()

    def plot_autox_endurance_telemetry(self, res):
        df = res['telemetry']
        self.fig_telem_autox.clear()
        
        ax1 = self.fig_telem_autox.add_subplot(221)
        ax2 = self.fig_telem_autox.add_subplot(222)
        ax3 = self.fig_telem_autox.add_subplot(223)
        ax4 = self.fig_telem_autox.add_subplot(224)
        
        # Subplot 1: 2D Track Trajectory Map with Speed Colormap
        if 'x_m' in df.columns and 'y_m' in df.columns:
            sc = ax1.scatter(df['x_m'], df['y_m'], c=df['v_kmh'], cmap='turbo', s=16, edgecolors='none', zorder=2)
            cbar = self.fig_telem_autox.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label("Velocità [km/h]", fontsize=7)
            cbar.ax.tick_params(labelsize=7)
            
            # Start/Finish line marker
            ax1.plot(df['x_m'].iloc[0], df['y_m'].iloc[0], 'go', markersize=8, label='Start / Finish', zorder=3)
            ax1.set_xlabel("Posizione X [m]", fontsize=8)
            ax1.set_ylabel("Posizione Y [m]", fontsize=8)
            ax1.set_title("Mappa Traiettoria 2D del Tracciato FSAE", fontsize=9, fontweight='bold')
            ax1.set_aspect('equal', 'datalim')
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.legend(fontsize=7, loc='best')
        else:
            ax1.text(0.5, 0.5, "Traiettoria 2D non disponibile", ha='center', va='center')
            
        # Subplot 2: Speed Profile
        ax2.plot(df['s_m'], df['v_kmh'], 'b-', linewidth=2.0, label='Velocità $v$ [km/h]')
        ax2.set_xlabel("Distanza sul Tracciato $s$ [m]", fontsize=8)
        ax2.set_ylabel("Velocità [km/h]", fontsize=8)
        ax2.set_title(f"Profilo Velocità ({res['event_type']} - Lap Time: {res['lap_time_s']:.2f} s | V_avg: {res['v_avg_kmh']:.1f} km/h)", fontsize=9, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=8, loc='upper right')
        
        # Subplot 3: Accelerations
        ax3.plot(df['s_m'], df['ax_g'], 'g-', linewidth=1.5, label='Acc. Longitudinale $a_x$ [g]')
        ax3.plot(df['s_m'], df['ay_g'], 'r-', linewidth=1.5, label='Acc. Laterale $a_y$ [g]')
        ax3.set_xlabel("Distanza sul Tracciato $s$ [m]", fontsize=8)
        ax3.set_ylabel("Accelerazione [g]", fontsize=8)
        ax3.set_title("Accelerazioni Longitudinale & Laterale Combinate", fontsize=9, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(fontsize=8, loc='upper right')
        
        # Subplot 4: Power & Energy
        ax4.plot(df['s_m'], df['power_elec_kw'], 'm-', linewidth=1.5, label='Potenza Elettrica $P_{elec}$ [kW] (Negativo = Regen)')
        ax4_t = ax4.twinx()
        ax4_t.plot(df['s_m'], df['energy_consumed_wh'], 'k--', linewidth=2.0, label='Energia Cumulata [Wh]')
        ax4.set_xlabel("Distanza sul Tracciato $s$ [m]", fontsize=8)
        ax4.set_ylabel("Potenza [kW]", color='m', fontsize=8)
        ax4_t.set_ylabel("Energia [Wh]", color='k', fontsize=8)
        ax4.set_title(f"Potenza & Consumi ({res['energy_lap_kwh']*1000.0:.1f} Wh/lap | {res['efficiency_kwh_km']:.3f} kWh/km | SoC: {res['battery_soc_end_pct']:.1f}%)", fontsize=9, fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        self.fig_telem_autox.suptitle(f"Simulazione FSAE {res['event_type']}: Traiettoria 2D, Telemetria & Analisi Consumi Batteria", fontsize=10, fontweight='bold')
        self.fig_telem_autox.tight_layout()
        self.canvas_telem_autox.draw()

    def export_step_csv(self):
        if not hasattr(self, 'last_step_res') or self.last_step_res is None or 'telemetry' not in self.last_step_res:
            QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria Step/DLC disponibile da esportare. Esegui prima la simulazione.")
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria Step/DLC CSV", "transient_telemetry.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.last_step_res['telemetry'].to_csv(path, index=False)
                self.log(f"Telemetria Step/DLC esportata con successo in: {path}")
                QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
            except Exception as e:
                self.log(f"Errore durante l'esportazione CSV Step/DLC: {e}")
                QMessageBox.critical(self, "Errore Esportazione", str(e))

    def export_autox_csv(self):
        if not hasattr(self, 'last_autox_res') or self.last_autox_res is None or 'telemetry' not in self.last_autox_res:
            QMessageBox.warning(self, "Esporta CSV", "Nessun dato di telemetria AutoX/Endurance disponibile da esportare. Esegui prima la simulazione.")
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Salva Telemetria AutoX/Endurance CSV", "autox_endurance_telemetry.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.last_autox_res['telemetry'].to_csv(path, index=False)
                self.log(f"Telemetria AutoX/Endurance esportata con successo in: {path}")
                QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
            except Exception as e:
                self.log(f"Errore durante l'esportazione CSV AutoX/Endurance: {e}")
                QMessageBox.critical(self, "Errore Esportazione", str(e))

    def plot_steering_torque(self):
        self.log("Computing Steering Torque & Driver Effort Curves (Tip #8, #61)...")
        QApplication.processEvents()
        
        try:
            caster = float(self.st_caster_input.text())
            kpi = float(self.st_kpi_input.text())
            mech_trail = float(self.st_mtrail_input.text())
            scrub = float(self.st_scrub_input.text())
            gr = float(self.st_ratio_input.text())
        except ValueError:
            caster, kpi, mech_trail, scrub, gr = 4.5, 8.0, 18.0, 12.0, 4.5
            
        vp = self._get_current_vehicle_params()
        tir_file = self.tir_combo.currentText()
        base_dir = os.path.dirname(__file__)
        from src.models.params_tire import TireParams
        tp = TireParams(os.path.join(base_dir, "tire_tir", tir_file if tir_file else "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir"))
        
        speeds = [8.0, 12.0, 16.0, 20.0]
        colors = ['#0984e3', '#00b894', '#e17055', '#6c5ce7']
        
        self.fig_steer_torque.clear()
        ax1 = self.fig_steer_torque.add_subplot(121)
        ax2 = self.fig_steer_torque.add_subplot(122)
        
        target_res = None
        max_torque_all = 0.0
        
        for v, c in zip(speeds, colors):
            df_h = mdo.generate_handling_sweep(
                v, vp.wd_f, vp.lltd, aero_bal=vp.aero_balance,
                num_points=60, L=vp.L, t_f=vp.t_f, t_r=vp.t_r,
                tir_file=tir_file, method='arclength', stop_at_peak=True
            )
            df_st = generate_steering_torque_curve(
                v, df_h, vp, tp, caster_deg=caster, kpi_deg=kpi,
                mech_trail_mm=mech_trail, scrub_radius_mm=scrub,
                steering_ratio=gr, rack_efficiency=0.90
            )
            if not df_st.empty:
                max_t = df_st['T_sw_Nm'].max()
                max_torque_all = max(max_torque_all, max_t)
                lw = 2.4 if v == getattr(vp, 'v_x', 15.0) else 1.5
                ax1.plot(df_st['ay_g'], df_st['T_sw_Nm'], color=c, linewidth=lw, label=f'$v = {v:.0f}$ m/s (Max: {max_t:.1f} Nm)')
                if v == speeds[1]:  # nominal 15 m/s
                    target_res = df_st
                    
        # Plot 1: Steering Wheel Torque vs Lateral Accel
        ax1.axhline(5.0, color='red', linestyle='--', linewidth=2.0, label='Limite 5.0 Nm (Schwarzenegger Limit)')
        y_top = max(max_torque_all * 1.15, 6.5)
        ax1.set_ylim(0, y_top)
        ax1.axhspan(5.0, y_top, color='#ff7675', alpha=0.15)
        ax1.axhspan(0, 5.0, color='#55efc4', alpha=0.08)
        ax1.text(0.03, 0.92, "[!] ZONA SFORZO ECCESSIVO (> 5 Nm)", transform=ax1.transAxes, fontsize=8, fontweight='bold', color='#d63031')
        ax1.text(0.03, 0.08, "[OK] ZONA ERGONOMICA OTTIMALE (< 5 Nm)", transform=ax1.transAxes, fontsize=8, fontweight='bold', color='#00b894')
        ax1.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
        ax1.set_ylabel("Coppia al Volante $T_{sw}$ [Nm]", fontsize=9)
        ax1.set_title("Coppia al Volante vs $a_y$ (Effort Pilota)", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        leg1 = ax1.legend(fontsize='x-small', loc='upper left')
        if leg1:
            leg1.set_draggable(True)
            
        # Plot 2: Trail Breakdown (Pneumatic vs Mechanical)
        if target_res is not None and not target_res.empty:
            ax2.plot(target_res['ay_g'], target_res['t_p_mm'], color='#0984e3', linewidth=2.0, label='Trail Pneumatico $t_p$ [mm]')
            ax2.plot(target_res['ay_g'], target_res['t_m_mm'], color='#e17055', linestyle='--', linewidth=2.0, label='Trail Meccanico $t_m$ [mm]')
            ax2.plot(target_res['ay_g'], target_res['t_tot_mm'], color='#2d3436', linewidth=2.2, label='Trail Totale $t_{tot} = t_p + t_m$ [mm]')
            ax2.set_xlabel("Accelerazione Laterale $a_y$ [g]", fontsize=9)
            ax2.set_ylabel("Braccio a Terra (Trail) [mm]", fontsize=9)
            ax2.set_title("Decomposizione Trail Pneumatico & Meccanico ($v=15$ m/s)", fontsize=10, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            leg2 = ax2.legend(fontsize='x-small', loc='upper right')
            if leg2:
                leg2.set_draggable(True)
                
        self.fig_steer_torque.tight_layout()
        self.canvas_steer_torque.draw()
        self.log(f"Steering Torque curves generated (Peak torque: {max_torque_all:.2f} Nm).")

    def calculate_and_plot_suspension_sizing(self):
        self.log("Calculating Suspension Rates & ARB Sizing (Tip #29, #87)...")
        QApplication.processEvents()
        
        try:
            fn_f = float(self.susp_fn_f_input.text())
            fn_r = float(self.susp_fn_r_input.text())
            mr_s = [float(s.strip()) for s in self.susp_mr_spring_input.text().split(',')]
            mr_arb = [float(s.strip()) for s in self.susp_mr_arb_input.text().split(',')]
            mat = "Titanium" if "Titanium" in self.susp_arb_mat_combo.currentText() else "Steel"
            k_inst_f = float(self.susp_k_inst_f_input.text())
            k_inst_r = float(self.susp_k_inst_r_input.text())
            preload_f = float(self.susp_preload_f_input.text()) if hasattr(self, 'susp_preload_f_input') else 5.0
            preload_r = float(self.susp_preload_r_input.text()) if hasattr(self, 'susp_preload_r_input') else 5.0
            stroke_tot = float(self.susp_stroke_tot_input.text()) if hasattr(self, 'susp_stroke_tot_input') else 57.0
            z_gap_mm = float(self.susp_bumpstop_gap_input.text())
            h_ride_f_mm = float(self.susp_ride_h_f_input.text())
            h_ride_r_mm = float(self.susp_ride_h_r_input.text())
        except Exception:
            fn_f, fn_r = 3.0, 3.3
            mr_s = [0.80, 0.80]
            mr_arb = [0.40, 0.40]
            mat = "Steel"
            k_inst_f, k_inst_r = 40.0, 45.0
            preload_f, preload_r = 5.0, 5.0
            stroke_tot = 57.0
            z_gap_mm = 20.0
            h_ride_f_mm, h_ride_r_mm = 30.0, 35.0
            
        vp = self._get_current_vehicle_params()
        res = calculate_suspension_and_arb_sizing(
            m_total_kg=vp.m,
            wd_f=vp.wd_f,
            lltd_target=vp.lltd,
            t_f_m=vp.t_f,
            t_r_m=vp.t_r,
            f_ride_f_hz=fn_f,
            f_ride_r_hz=fn_r,
            mr_spring_f=mr_s[0],
            mr_spring_r=mr_s[1],
            mr_arb_f=mr_arb[0],
            mr_arb_r=mr_arb[1],
            arb_material=mat,
            k_spring_installed_f_Nmm=k_inst_f,
            k_spring_installed_r_Nmm=k_inst_r,
            spring_preload_f_mm=preload_f,
            spring_preload_r_mm=preload_r,
            damper_stroke_total_mm=stroke_tot,
            bumpstop_length_mm=z_gap_mm,
            static_ride_height_f_mm=h_ride_f_mm,
            static_ride_height_r_mm=h_ride_r_mm
        )
        
        # Populate results table
        table_rows = [
            ("Massa Sospesa Totale", f"{res['m_sprung_kg']:.1f} kg"),
            ("Corsa Totale Ammortizzatore (TTX25)", f"{stroke_tot:.1f} mm (Corsa Ruota: {stroke_tot / mr_s[0]:.1f} mm @ MR={mr_s[0]:.2f})"),
            ("Motion Ratio (Damper / ARB)", f"Damper MR: {mr_s[0]:.2f} | ARB MR: {mr_arb[0]:.2f}"),
            ("Precarico Molla Installato (F/R)", f"Ant: {preload_f:.1f} mm | Post: {preload_r:.1f} mm"),
            ("Wheel Rate Target Ant. / Post.", f"{res['Kw_f_Nmm']:.2f} / {res['Kw_r_Nmm']:.2f} N/mm"),
            ("Molla Target Ant. / Post.", f"{res['Kspring_f_Nmm']:.1f} / {res['Kspring_r_Nmm']:.1f} N/mm"),
            ("Molla Installata Reale Ant. / Post.", f"{k_inst_f:.1f} / {k_inst_r:.1f} N/mm (Kw: {res.get('Kw_inst_f_Nmm', 0):.1f} / {res.get('Kw_inst_r_Nmm', 0):.1f} N/mm)"),
            ("Freq. Naturale Target vs Installata (Ant)", f"{fn_f:.2f} Hz  ->  Reale: {res.get('fn_inst_f_hz', fn_f):.2f} Hz"),
            ("Freq. Naturale Target vs Installata (Post)", f"{fn_r:.2f} Hz  ->  Reale: {res.get('fn_inst_r_hz', fn_r):.2f} Hz"),
            ("Compressione Statica a 1g (Static Sag)", f"Damper: {res.get('damper_sag_f_mm', 0):.1f} / {res.get('damper_sag_r_mm', 0):.1f} mm | Ruota: {res.get('z_static_sag_f_mm', 0):.1f} / {res.get('z_static_sag_r_mm', 0):.1f} mm"),
            ("Corsa Disponibile in Compressione (Bump)", f"Damper: {res.get('damper_bump_f_mm', 0):.1f} / {res.get('damper_bump_r_mm', 0):.1f} mm | Ruota: {res.get('wheel_bump_f_mm', 0):.1f} / {res.get('wheel_bump_r_mm', 0):.1f} mm"),
            ("Corsa Disponibile in Estensione (Droop/Reb)", f"Damper: {res.get('damper_reb_f_mm', 0):.1f} / {res.get('damper_reb_r_mm', 0):.1f} mm | Ruota: {res.get('wheel_reb_f_mm', 0):.1f} / {res.get('wheel_reb_r_mm', 0):.1f} mm"),
            ("Corsa Libera prima Tampone (Bump Headroom)", f"Damper: {max(res.get('damper_bump_f_mm', 0) - z_gap_mm, 0):.1f} / {max(res.get('damper_bump_r_mm', 0) - z_gap_mm, 0):.1f} mm | Ruota: {res.get('bump_headroom_f_mm', 0):.1f} / {res.get('bump_headroom_r_mm', 0):.1f} mm"),
            ("Altezza da Terra Statica (Ride Height)", f"Ant: {h_ride_f_mm:.1f} mm | Post: {h_ride_r_mm:.1f} mm"),
            ("LLTD Naturale Solo Molle", f"{res['lltd_springs_only_pct']:.1f} %"),
            ("LLTD Target Ottimizzato", f"{vp.lltd*100:.1f} %"),
            ("Rigidezza Rollio Molle Ant. / Post.", f"{res['Kphi_spring_f_Nmdeg']:.1f} / {res['Kphi_spring_r_Nmdeg']:.1f} Nm/deg"),
            ("Rigidezza ARB Necessaria Ant. / Post.", f"{res['Kphi_arb_f_Nmdeg']:.1f} / {res['Kphi_arb_r_Nmdeg']:.1f} Nm/deg"),
            ("Diametro Esterno ARB Ant. / Post. (Do)", f"Ant: {res['Do_arb_f_mm']:.1f} mm | Post: {res['Do_arb_r_mm']:.1f} mm ({mat})"),
            ("Gradiente di Rollio Target", f"{res['roll_grad_target_degg']:.2f} deg/g")
        ]
        
        self.susp_results_table.setRowCount(len(table_rows))
        for row_idx, (name, val) in enumerate(table_rows):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(QtCore.Qt.ItemIsEnabled)
            item_val = QTableWidgetItem(val)
            item_val.setFlags(QtCore.Qt.ItemIsEnabled)
            item_val.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
            self.susp_results_table.setItem(row_idx, 0, item_name)
            self.susp_results_table.setItem(row_idx, 1, item_val)
            
        # Plot 2 Subplots: Roll Stiffness breakdown & Damper Stroke / Bumpstop Usage
        self.fig_susp_sizing.clear()
        ax1 = self.fig_susp_sizing.add_subplot(121)
        ax2 = self.fig_susp_sizing.add_subplot(122)

        # 1. Roll Stiffness Breakdown
        labels = ['Avantreno (Front)', 'Retrotreno (Rear)']
        k_springs = [res['Kphi_spring_f_Nmdeg'], res['Kphi_spring_r_Nmdeg']]
        k_arbs = [res['Kphi_arb_f_Nmdeg'], res['Kphi_arb_r_Nmdeg']]
        
        width = 0.45
        ax1.bar(labels, k_springs, width, label='Molle Sospensione', color='#0984e3')
        ax1.bar(labels, k_arbs, width, bottom=k_springs, label='Barra Antirollio ARB', color='#00b894')
        ax1.set_ylabel("Rigidezza di Rollio $K_\\phi$ [Nm/deg]", fontsize=8.5)
        ax1.set_title(f"1. Ripartizione Rigidezza Rollio\n(LLTD Target = {vp.lltd*100:.1f}%)", fontsize=9.5, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
        leg1 = ax1.legend(fontsize='x-small', loc='upper right')
        if leg1:
            leg1.set_draggable(True)

        # 2. Damper Travel & Bump Stop Headroom (Physical 57mm Stroke Partitioning)
        categories = ['Damper Ant.', 'Damper Post.']
        sag_vals = [res.get('damper_sag_f_mm', 14.0), res.get('damper_sag_r_mm', 16.0)]
        bump_vals = [res.get('damper_bump_f_mm', 43.0), res.get('damper_bump_r_mm', 41.0)]
        gap_vals = [max(res.get('damper_bump_f_mm', 43.0) - z_gap_mm, 0.0), max(res.get('damper_bump_r_mm', 41.0) - z_gap_mm, 0.0)]

        x_pos = np.arange(len(categories))
        ax2.bar(x_pos - 0.15, sag_vals, width=0.28, label='Static Sag / Droop Disp. (1g)', color='#e17055')
        ax2.bar(x_pos + 0.15, gap_vals, width=0.28, label='Corsa Libera prima Tampone', color='#fdcb6e')
        ax2.axhline(stroke_tot, color='#d63031', linestyle='--', linewidth=1.5, label=f'Corsa Max TTX25 ({stroke_tot:.0f} mm)')
        ax2.axhline(0, color='black', linewidth=1.0)
        ax2.axhline(-stroke_tot * (1.0 - mr_s[0]), color='gray', linestyle=':', linewidth=0.8)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, fontsize=9)
        ax2.set_ylabel("Corsa Ammortizzatore TTX25 [mm]", fontsize=8.5)
        ax2.set_title(f"2. Corsa TTX25 ({stroke_tot:.0f}mm) & Precarico\n(MR Damper={mr_s[0]:.2f}, ARB={mr_arb[0]:.2f})", fontsize=9.5, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
        leg2 = ax2.legend(fontsize='x-small', loc='upper right')
        if leg2:
            leg2.set_draggable(True)
            
        self.fig_susp_sizing.tight_layout()
        self.canvas_susp_sizing.draw()
        self.log(f"Suspension sizing calculation completed. (Installed fn: Front {res.get('f_ride_installed_f_hz', 0):.2f} Hz, Rear {res.get('f_ride_installed_r_hz', 0):.2f} Hz).")

    def plot_damper_dyno(self):
        """Generates F-V dyno response curves and modal damping analysis for Öhlins TTX25 MkII."""
        self.log("Generazione Curve Dyno Ammortizzatori Öhlins TTX25 MkII...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            mode = self.damper_mode_combo.currentData() if hasattr(self, 'damper_mode_combo') else 'linear'
            valv_f = self.damper_valv_f_combo.currentText() if hasattr(self, 'damper_valv_f_combo') else 'C12/R12'
            valv_r = self.damper_valv_r_combo.currentText() if hasattr(self, 'damper_valv_r_combo') else 'C12/R12'

            lsc_f = self.damper_lsc_f_spin.value() if hasattr(self, 'damper_lsc_f_spin') else 10
            hsc_f = self.damper_hsc_f_spin.value() if hasattr(self, 'damper_hsc_f_spin') else 2.0
            lsr_f = self.damper_lsr_f_spin.value() if hasattr(self, 'damper_lsr_f_spin') else 10
            hsr_f = self.damper_hsr_f_spin.value() if hasattr(self, 'damper_hsr_f_spin') else 2.0
            mr_f  = self.damper_mr_f_spin.value() if hasattr(self, 'damper_mr_f_spin') else 0.85

            lsc_r = self.damper_lsc_r_spin.value() if hasattr(self, 'damper_lsc_r_spin') else 10
            hsc_r = self.damper_hsc_r_spin.value() if hasattr(self, 'damper_hsc_r_spin') else 2.0
            lsr_r = self.damper_lsr_r_spin.value() if hasattr(self, 'damper_lsr_r_spin') else 10
            hsr_r = self.damper_hsr_r_spin.value() if hasattr(self, 'damper_hsr_r_spin') else 2.0
            mr_r  = self.damper_mr_r_spin.value() if hasattr(self, 'damper_mr_r_spin') else 0.85

            # Apply to vp
            vp.damper_mode = mode
            vp.damper_valving_f = valv_f
            vp.damper_valving_r = valv_r
            vp.damper_lsc_clicks_f = float(lsc_f)
            vp.damper_hsc_turns_f = float(hsc_f)
            vp.damper_lsr_clicks_f = float(lsr_f)
            vp.damper_hsr_turns_f = float(hsr_f)
            vp.damper_lsc_clicks_r = float(lsc_r)
            vp.damper_hsc_turns_r = float(hsc_r)
            vp.damper_lsr_clicks_r = float(lsr_r)
            vp.damper_hsr_turns_r = float(hsr_r)
            vp.motion_ratio_f = float(mr_f)
            vp.motion_ratio_r = float(mr_r)

            # Generate Dyno data
            curve_f = dm_ttx25.get_dyno_curve(valv_f, lsc_f, hsc_f, lsr_f, hsr_f)
            curve_r = dm_ttx25.get_dyno_curve(valv_r, lsc_r, hsc_r, lsr_r, hsr_r)

            # Corner masses & wheel stiffnesses
            m_sprung = vp.m * 0.88
            m_corner_f = (m_sprung * vp.wd_f) / 2.0
            m_corner_r = (m_sprung * (1.0 - vp.wd_f)) / 2.0
            Kw_f = vp.K_z_f
            Kw_r = vp.K_z_r

            modal_f = dm_ttx25.compute_modal_damping_ratios(m_corner_f, Kw_f, valv_f, lsc_f, hsc_f, lsr_f, hsr_f, mr_f)
            modal_r = dm_ttx25.compute_modal_damping_ratios(m_corner_r, Kw_r, valv_r, lsc_r, hsc_r, lsr_r, hsr_r, mr_r)

            # Equivalent roll damping
            C_phi_f_equiv = 0.5 * (vp.t_f**2) * modal_f['c_wheel_avg']
            C_phi_r_equiv = 0.5 * (vp.t_r**2) * modal_r['c_wheel_avg']
            C_phi_tot_equiv = C_phi_f_equiv + C_phi_r_equiv

            # Populate table
            asym_f = modal_f['zeta_reb_ls'] / max(modal_f['zeta_bump_ls'], 1e-4)
            asym_r = modal_r['zeta_reb_ls'] / max(modal_r['zeta_bump_ls'], 1e-4)

            mode_str = "Öhlins TTX25 Non-Lineare (Attivo)" if mode == 'ttx25_nonlinear' else "Costante Lineare (Fallback)"
            table_rows = [
                ("Modalità Attiva", mode_str),
                ("Specifiche Valvola Ant. / Post.", f"{valv_f} / {valv_r}"),
                ("Regolazioni Anteriore (LSC / HSC / LSR / HSR)", f"{lsc_f} clk / {hsc_f:.1f} giri / {lsr_f} clk / {hsr_f:.1f} giri"),
                ("Regolazioni Posteriore (LSC / HSC / LSR / HSR)", f"{lsc_r} clk / {hsc_r:.1f} giri / {lsr_r} clk / {hsr_r:.1f} giri"),
                ("Forza Stelo @ 25 mm/s Bump (Ant / Post)", f"{float(dm_ttx25.calculate_damper_force(0.025, valv_f, lsc_f, hsc_f, lsr_f, hsr_f)):.1f} N / {float(dm_ttx25.calculate_damper_force(0.025, valv_r, lsc_r, hsc_r, lsr_r, hsr_r)):.1f} N"),
                ("Forza Stelo @ 25 mm/s Reb (Ant / Post)", f"{float(dm_ttx25.calculate_damper_force(-0.025, valv_f, lsc_f, hsc_f, lsr_f, hsr_f)):.1f} N / {float(dm_ttx25.calculate_damper_force(-0.025, valv_r, lsc_r, hsc_r, lsr_r, hsr_r)):.1f} N"),
                ("Forza Stelo @ 250 mm/s Bump (Ant / Post)", f"{float(dm_ttx25.calculate_damper_force(0.250, valv_f, lsc_f, hsc_f, lsr_f, hsr_f)):.1f} N / {float(dm_ttx25.calculate_damper_force(0.250, valv_r, lsc_r, hsc_r, lsr_r, hsr_r)):.1f} N"),
                ("Damping Ratio Low-Speed Bump (zeta,f / zeta,r)", f"{modal_f['zeta_bump_ls']:.2f} / {modal_r['zeta_bump_ls']:.2f}"),
                ("Damping Ratio Low-Speed Reb (zeta,f / zeta,r)", f"{modal_f['zeta_reb_ls']:.2f} / {modal_r['zeta_reb_ls']:.2f}"),
                ("Asimmetria Rebound / Bump (Target 1.5 - 2.0)", f"Ant: {asym_f:.2f}x | Post: {asym_r:.2f}x"),
                ("Smorzamento Rollio Equivalente (C_phi,tot)", f"{C_phi_tot_equiv:.1f} Nm·s/rad (Ant: {C_phi_f_equiv:.1f}, Post: {C_phi_r_equiv:.1f})")
            ]

            self.damper_results_table.setRowCount(len(table_rows))
            for r_i, (k, val) in enumerate(table_rows):
                it_k = QTableWidgetItem(k)
                it_k.setFlags(QtCore.Qt.ItemIsEnabled)
                it_v = QTableWidgetItem(val)
                it_v.setFlags(QtCore.Qt.ItemIsEnabled)
                it_v.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                self.damper_results_table.setItem(r_i, 0, it_k)
                self.damper_results_table.setItem(r_i, 1, it_v)

            # Draw Figures
            self.fig_damper_dyno.clear()
            ax1 = self.fig_damper_dyno.add_subplot(121)
            ax2 = self.fig_damper_dyno.add_subplot(122)

            # Subplot 1: Dyno F-V Response Curve
            ax1.plot(curve_f['v_mmps'], curve_f['f_N'], color='#0984e3', linewidth=2.2, label=f'Ant. {valv_f} ({lsc_f}/{hsc_f:.1f}/{lsr_f}/{hsr_f:.1f})')
            ax1.plot(curve_r['v_mmps'], curve_r['f_N'], color='#e17055', linewidth=2.2, linestyle='--', label=f'Post. {valv_r} ({lsc_r}/{hsc_r:.1f}/{lsr_r}/{hsr_r:.1f})')
            ax1.axhline(0, color='black', linewidth=1.2)
            ax1.axvline(0, color='black', linewidth=1.0)
            ax1.axvline(25.0, color='gray', linestyle=':', alpha=0.6, label='Knee (25 mm/s)')
            ax1.axvline(-25.0, color='gray', linestyle=':', alpha=0.6)

            ax1.set_xlabel("Velocità Stelo Ammortizzatore [mm/s]", fontsize=8.5)
            ax1.set_ylabel("Forza Ammortizzatore $F_{\\text{damper}}$ [N]", fontsize=8.5)
            ax1.set_title("1. Risposta Dyno Öhlins TTX25 (Forza vs Velocità)", fontsize=9.5, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            leg1 = ax1.legend(fontsize='x-small', loc='upper left')
            if leg1:
                leg1.set_draggable(True)

            # Subplot 2: Damping Ratios vs OptimumG Target Zones
            cats = ['Ant. Bump (LS)', 'Ant. Reb (LS)', 'Post. Bump (LS)', 'Post. Reb (LS)']
            zetas = [modal_f['zeta_bump_ls'], modal_f['zeta_reb_ls'], modal_r['zeta_bump_ls'], modal_r['zeta_reb_ls']]
            colors_bar = ['#0984e3', '#74b9ff', '#e17055', '#fab1a0']

            x_pos = np.arange(len(cats))
            ax2.bar(x_pos, zetas, width=0.45, color=colors_bar, edgecolor='#2d3436')
            ax2.axhspan(0.65, 0.85, color='#55efc4', alpha=0.25, label='Target Ottimale LS (0.65 - 0.85)')
            ax2.axhline(1.0, color='#d63031', linestyle='--', linewidth=1.2, label='Smorzamento Critico (zeta=1.0)')

            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(cats, fontsize=8.0, rotation=15)
            ax2.set_ylabel("Damping Ratio $\\zeta = c / c_{\\text{crit}}$ [-]", fontsize=8.5)
            ax2.set_title("2. Rapporti di Smorzamento $\\zeta$ (Low Speed)", fontsize=9.5, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
            leg2 = ax2.legend(fontsize='x-small', loc='upper right')
            if leg2:
                leg2.set_draggable(True)

            self.fig_damper_dyno.tight_layout()
            self.canvas_damper_dyno.draw()
            self.log("Curve Dyno TTX25 e Damping Ratios aggiornati con successo.")
        except Exception as e:
            self.log(f"Errore generazione Dyno Damper: {e}")

    def _on_bal_param_changed(self):
        param_key = self.combo_bal_param.currentData()
        cfg = SWEEP_PARAMETER_CONFIGS.get(param_key)
        if not cfg:
            return
        if cfg['is_fraction']:
            self.bal_min_input.setText(f"{cfg['default_min'] * 100.0:.1f}")
            self.bal_max_input.setText(f"{cfg['default_max'] * 100.0:.1f}")
        else:
            self.bal_min_input.setText(f"{cfg['default_min'] * cfg['scale']:.1f}")
            self.bal_max_input.setText(f"{cfg['default_max'] * cfg['scale']:.1f}")

    def plot_optimal_balance_sweep(self):
        self.btn_calc_balance_range.setEnabled(False)
        self.btn_calc_balance_range.setText("Calcolo in corso...")
        self.log("Calcolo Finestra di Bilanciamento Ottimale & Sweep Incertezza...")
        QApplication.processEvents()
        try:
            vp = self._get_current_vehicle_params()
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
            
            param_key = self.combo_bal_param.currentData()
            cfg = SWEEP_PARAMETER_CONFIGS.get(param_key, SWEEP_PARAMETER_CONFIGS['lltd'])
            
            p_min_disp = float(self.bal_min_input.text())
            p_max_disp = float(self.bal_max_input.text())
            p_min = p_min_disp / cfg['scale']
            p_max = p_max_disp / cfg['scale']
            
            n_pts = int(self.bal_pts_input.text())
            v_eval = float(self.bal_vx_input.text())
            
            tire_unc = float(self.bal_unc_mu_input.text())
            wd_unc = float(self.bal_unc_wd_input.text())
            aero_unc = float(self.bal_unc_aero_input.text())
            lltd_unc = float(self.bal_unc_lltd_input.text())
            
            kus_min_tgt = float(self.bal_kus_min_input.text())
            kus_max_tgt = float(self.bal_kus_max_input.text())
            
            res = evaluate_optimal_balance_sweep(
                vp=vp,
                param_key=param_key,
                param_min=p_min,
                param_max=p_max,
                n_points=n_pts,
                v_eval=v_eval,
                tire_unc_pct=tire_unc,
                wd_unc_pct=wd_unc,
                aero_unc_pct=aero_unc,
                lltd_unc_pct=lltd_unc,
                target_kus_min_degg=kus_min_tgt,
                target_kus_max_degg=kus_max_tgt,
                tir_file=tir_file
            )
            
            df = res['sweep_df']
            opt_min_d, opt_max_d = res['optimal_range_display']
            cur_d = res['current_val_display']
            rob_score = res['robust_score_pct']
            in_opt = res['is_current_in_optimal']
            
            # Status Banner Update
            status_color = "#00b894" if in_opt else "#d63031"
            status_badge = "[OK] IN TARGET" if in_opt else "[!] FUORI RANGE OTTIMO"
            banner_text = (
                f"{status_badge} | Finestra Ottimale {cfg['name']}: [{opt_min_d:.1f} - {opt_max_d:.1f} {cfg['unit']}] "
                f"(Attuale: {cur_d:.1f} {cfg['unit']}) | Score Robustezza: {rob_score:.1f}% @ v={v_eval:.0f} m/s"
            )
            self.lbl_stability_margin.setText(banner_text)
            self.lbl_stability_margin.setStyleSheet(f"font-weight: bold; font-size: 11px; color: {status_color}; padding: 4px;")
            
            self.fig_radar.clear()
            ax1 = self.fig_radar.add_subplot(121)
            ax2 = self.fig_radar.add_subplot(122)
            
            x_vals = df['param_display'].values
            
            # Subplot 1: Kus vs Swept Parameter
            ax1.fill_between(x_vals, df['kus_min'], df['kus_max'], color='#74b9ff', alpha=0.25, label=f'Incertezza Dispersione (±{tire_unc:.0f}% μ, ±{wd_unc:.1f}% WD)')
            ax1.plot(x_vals, df['kus_nom'], 'b-o', linewidth=2.2, markersize=5, label='K_us Nominale')
            ax1.plot(x_vals, df['kus_min'], 'r--', linewidth=1.2, alpha=0.7, label='Worst-Case Min (Oversteer Hazard)')
            ax1.plot(x_vals, df['kus_max'], 'g--', linewidth=1.2, alpha=0.7, label='Worst-Case Max (Understeer Push)')
            
            # Target Kus Band
            ax1.axhspan(kus_min_tgt, kus_max_tgt, color='#00b894', alpha=0.12, label=f'Target K_us [{kus_min_tgt:.2f}, {kus_max_tgt:.2f}] deg/g')
            ax1.axhline(0, color='#2d3436', linestyle='--', linewidth=1.5, label='Linea Neutra (K_us = 0)')
            
            # Highlight Optimal Parameter Range
            if opt_min_d <= opt_max_d:
                ax1.axvspan(opt_min_d, opt_max_d, color='#55efc4', alpha=0.22, label=f'Finestra Ottimale [{opt_min_d:.1f}, {opt_max_d:.1f}]')
                
            # Current Operating Point
            kus_cur = float(np.interp(cur_d, x_vals, df['kus_nom']))
            ax1.plot(cur_d, kus_cur, marker='D', markersize=9, color='#d63031', markeredgecolor='black', zorder=10, label=f'Setup Attuale ({cur_d:.1f} {cfg["unit"]})')
            
            ax1.set_xlabel(f"{cfg['name']} [{cfg['unit']}]", fontsize=9.5)
            ax1.set_ylabel("Gradiente di Sottosterzo $K_{us}$ [deg/g]", fontsize=9.5)
            ax1.set_title(f"1. Risposta Sottosterzo vs {cfg['name']}", fontsize=10, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            leg1 = ax1.legend(fontsize='x-small', loc='best')
            if leg1:
                leg1.set_draggable(True)
                
            # Subplot 2: Multi-Metric Corridor (Stability Margin & Max Ay)
            color_stab = '#6c5ce7'
            ax2.plot(x_vals, df['stab_nom'], color=color_stab, linewidth=2.2, marker='s', markersize=4, label='Margine Stabilità (LLTD - Aero) %')
            ax2.fill_between(x_vals, df['stab_min'], df['stab_max'], color=color_stab, alpha=0.18, label='Incertezza Margine Stabilità')
            ax2.axhline(2.0, color='#00b894', linestyle='--', linewidth=1.4, label='Soglia Sicurezza Bilanciamento (+2.0%)')
            ax2.axhspan(-20, 2.0, color='#d63031', alpha=0.08, label='Zona Rischio Sovrasterzo Alta Velocità')
            ax2.set_xlabel(f"{cfg['name']} [{cfg['unit']}]", fontsize=9.5)
            ax2.set_ylabel("Margine Stabilità Alta Velocità [%]", fontsize=9.5, color=color_stab)
            ax2.tick_params(axis='y', labelcolor=color_stab)
            
            # Secondary Y axis for Max Lateral Accel
            ax2_ay = ax2.twinx()
            color_ay = '#e17055'
            ax2_ay.plot(x_vals, df['ay_nom'], color=color_ay, linestyle='-.', linewidth=2.0, label='Grip Max $a_{y,max}$ [g]')
            ax2_ay.set_ylabel("Accelerazione Laterale Max $a_{y,max}$ [g]", fontsize=9.5, color=color_ay)
            ax2_ay.tick_params(axis='y', labelcolor=color_ay)
            
            if opt_min_d <= opt_max_d:
                ax2.axvspan(opt_min_d, opt_max_d, color='#55efc4', alpha=0.15)
                
            ax2.set_title("2. Margine Stabilità Alta Velocità & Grip Limite", fontsize=10, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            # Combined legend for twinx
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_ay.get_legend_handles_labels()
            leg2 = ax2.legend(lines1 + lines2, labels1 + labels2, fontsize='x-small', loc='upper left')
            if leg2:
                leg2.set_draggable(True)
                
            self.fig_radar.tight_layout()
            self.canvas_radar.draw()
            self.log(f"Analisi Range di Bilanciamento completata con successo: Finestra Ottimale [{opt_min_d:.1f} - {opt_max_d:.1f} {cfg['unit']}].")
        except Exception as e:
            import traceback
            self.log(f"Errore calcolo range bilanciamento: {traceback.format_exc()}")
        finally:
            self.btn_calc_balance_range.setEnabled(True)
            self.btn_calc_balance_range.setText(" Calcola Finestra di Bilanciamento & Incertezza")

    def plot_4_ratios_radar(self, base_params=None):
        self.log("Plotting 4-Ratio Balance Radar & High-Speed Stability Margin (Tip #82, #84, #89)...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params()
        if base_params is not None and len(base_params) >= 6:
            base_L, base_tf, base_tr, base_wdf, base_lltd, base_aero = base_params[:6]
        else:
            base_L, base_tf, base_tr, base_wdf, base_lltd, base_aero = 1.53, 1.25, 1.15, 0.50, 0.55, 0.48
            
        bal_res = evaluate_4_ratios_balance(
            wd_f=vp.wd_f,
            lltd=vp.lltd,
            aero_bal=vp.aero_balance,
            t_f=vp.t_f,
            t_r=vp.t_r
        )
        
        margin = bal_res['stability_margin_pct']
        if margin >= 2.0:
            self.lbl_stability_margin.setText(f"[OK] Stabilità Alta Velocità: OTTIMALE (LLTD - Aero = +{margin:.1f}% >= +2.0%)")
            self.lbl_stability_margin.setStyleSheet("font-weight: bold; font-size: 11px; color: #00b894; padding: 4px;")
        else:
            self.lbl_stability_margin.setText(f"[!] Stabilità Alta Velocità: ATTENZIONE (LLTD - Aero = {margin:+.1f}% < +2.0%, rischio sovrasterzo nei curvoni veloci)")
            self.lbl_stability_margin.setStyleSheet("font-weight: bold; font-size: 11px; color: #d63031; padding: 4px;")
            
        self.fig_radar.clear()
        
        # Subplot 1: Spider / Radar Chart
        ax1 = self.fig_radar.add_subplot(121, polar=True)
        categories = ['Distribuzione Masse\n(wd_f %)', 'Rigidezza Rollio\n(LLTD %)', 'Bilancio Aero\n(aero_bal %)', 'Rapporto Carreggiate\n(tf/tr %)']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Values in percentage (30-70% design window)
        v_base = [base_wdf * 100.0, base_lltd * 100.0, base_aero * 100.0, (base_tf / (base_tf + base_tr)) * 100.0]
        v_base += v_base[:1]
        
        v_opt = [vp.wd_f * 100.0, vp.lltd * 100.0, vp.aero_balance * 100.0, (vp.t_f / (vp.t_f + vp.t_r)) * 100.0]
        v_opt += v_opt[:1]
        
        ax1.plot(angles, v_base, 'o--', linewidth=2.0, color='#e17055', label=f'Baseline (LLTD-Aero = {base_lltd*100-base_aero*100:+.1f}%)')
        ax1.fill(angles, v_base, color='#e17055', alpha=0.15)
        
        ax1.plot(angles, v_opt, 's-', linewidth=2.6, color='#0984e3', label=f'Target Ottimizzato (LLTD-Aero = {margin:+.1f}%)')
        ax1.fill(angles, v_opt, color='#0984e3', alpha=0.25)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, fontsize=8, fontweight='bold')
        ax1.set_ylim(30.0, 70.0)
        ax1.set_yticks([35, 45, 55, 65])
        ax1.set_yticklabels(["35%", "45%", "55%", "65%"], fontsize=7, color='#636e72')
        ax1.set_title("Radar dei 4 Rapporti Fondamentali di Assetto", fontsize=10, fontweight='bold', pad=15)
        leg1 = ax1.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.22), fontsize='x-small')
        if leg1:
            leg1.set_draggable(True)
            
        # Subplot 2: Dynamic Front Bias Transition vs Velocity
        ax2 = self.fig_radar.add_subplot(122)
        df_bal = bal_res['balance_curve']
        ax2.plot(df_bal['v_x'], df_bal['dyn_front_bias_pct'], 'b-o', linewidth=2.2, label='Bias Aderenza Anteriore Totale (%)')
        ax2.plot(df_bal['v_x'], df_bal['aero_frac_pct'], 'g:', linewidth=1.8, label='Quota Carico Aerodinamico / Totale (%)')
        ax2.axhline(vp.wd_f * 100, color='#e17055', linestyle='--', linewidth=1.2, label=f'Massa Anteriore $wd_f$ ({vp.wd_f*100:.1f}%)')
        ax2.axhline(vp.aero_balance * 100, color='#6c5ce7', linestyle='-.', linewidth=1.2, label=f'Aero Front $aero\\_bal$ ({vp.aero_balance*100:.1f}%)')
        
        ax2.set_xlabel("Velocità di Avanzamento $v_x$ [m/s]", fontsize=9)
        ax2.set_ylabel("Ripartizione / Quota Carico [%]", fontsize=9)
        ax2.set_title("Transizione Grip Meccanico (Low Speed) $\\rightarrow$ Aerodinamico (High Speed)", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        leg2 = ax2.legend(fontsize='x-small', loc='best')
        if leg2:
            leg2.set_draggable(True)
            
        self.fig_radar.tight_layout()
        self.canvas_radar.draw()
        self.log("4-Ratio Balance Radar generated successfully.")

    def _init_tab_arb_diamond(self):
        """Costruisce la scheda interattiva per il Rombo di Setting ARB & Bilanciamento Meccanico 14-DOF."""
        self.tab_arb_diamond = QWidget()
        self.tab_arb_diamond_layout = QVBoxLayout(self.tab_arb_diamond)

        # 1. Top Configuration Group
        diamond_cfg_group = QGroupBox(" Rombo di Setting ARB & Bilanciamento Meccanico (Modello 14-DOF) — Asse X: K_phi,tot [Nm/deg], Asse Y: RigRollioFront/RigRollioTot [%]")
        diamond_cfg_layout = QVBoxLayout(diamond_cfg_group)

        row1_layout = QHBoxLayout()

        # Tabella editabile dei 4 vertici del rombo
        self.table_diamond_vertices = QTableWidget(4, 4)
        self.table_diamond_vertices.setHorizontalHeaderLabels(["Vertice", "Configurazione", "K_tot [Nm/deg]", "Ratio Front [%]"])
        self.table_diamond_vertices.horizontalHeader().setStretchLastSection(True)
        self.table_diamond_vertices.setMaximumHeight(155)
        self.table_diamond_vertices.setMinimumWidth(430)

        default_v_data = [
            ("V1", "Soft-Soft (Min Totale)", "436.3", "48.0"),
            ("V2", "Stiff F / Soft R (Max Sottosterzo)", "959.9", "64.0"),
            ("V3", "Stiff-Stiff (Max Totale)", "1483.5", "49.0"),
            ("V4", "Soft F / Stiff R (Max Sovrasterzo)", "959.9", "36.0")
        ]
        for r_idx, (v_id, v_desc, ktot_s, rf_s) in enumerate(default_v_data):
            item_id = QTableWidgetItem(v_id)
            item_id.setFlags(QtCore.Qt.ItemIsEnabled)
            item_desc = QTableWidgetItem(v_desc)
            item_desc.setFlags(QtCore.Qt.ItemIsEnabled)
            item_k = QTableWidgetItem(ktot_s)
            item_r = QTableWidgetItem(rf_s)
            self.table_diamond_vertices.setItem(r_idx, 0, item_id)
            self.table_diamond_vertices.setItem(r_idx, 1, item_desc)
            self.table_diamond_vertices.setItem(r_idx, 2, item_k)
            self.table_diamond_vertices.setItem(r_idx, 3, item_r)

        row1_layout.addWidget(self.table_diamond_vertices, stretch=2)

        # Controlli a destra: Preset, Parametri, Azioni
        cfg_side = QVBoxLayout()

        # Profilo e Salvataggio Configurazioni con Nome
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("<b>Profilo / Preset:</b>"))
        self.combo_diamond_presets = QComboBox()
        self.combo_diamond_presets.setMinimumWidth(180)
        preset_row.addWidget(self.combo_diamond_presets)

        self.btn_save_diamond_cfg = QPushButton(" Salva")
        self.btn_save_diamond_cfg.setToolTip("Salva le modifiche sul file di configurazione corrente")
        self.btn_save_diamond_cfg.clicked.connect(self.save_current_diamond_config)
        preset_row.addWidget(self.btn_save_diamond_cfg)

        self.btn_save_as_diamond_cfg = QPushButton(" Salva con Nome...")
        self.btn_save_as_diamond_cfg.setToolTip("Salva la configurazione dei vertici con un nuovo nome (.json)")
        self.btn_save_as_diamond_cfg.setStyleSheet("font-weight: bold; background-color: #2980b9; color: white; padding: 3px 8px; border-radius: 3px;")
        self.btn_save_as_diamond_cfg.clicked.connect(self.save_as_diamond_config_dialog)
        preset_row.addWidget(self.btn_save_as_diamond_cfg)

        self.btn_load_diamond_cfg = QPushButton(" Sfoglia...")
        self.btn_load_diamond_cfg.setToolTip("Carica un file di configurazione rombo (.json) da qualsiasi cartella")
        self.btn_load_diamond_cfg.clicked.connect(self.browse_load_diamond_config)
        preset_row.addWidget(self.btn_load_diamond_cfg)

        self.btn_delete_diamond_cfg = QPushButton(" Elimina")
        self.btn_delete_diamond_cfg.setToolTip("Elimina la configurazione personalizzata selezionata")
        self.btn_delete_diamond_cfg.clicked.connect(self.delete_diamond_config_dialog)
        preset_row.addWidget(self.btn_delete_diamond_cfg)

        self.btn_calc_from_arb_ranges = QPushButton(" Calcola da ARB...")
        self.btn_calc_from_arb_ranges.setToolTip("Calcola i vertici del rombo dai limiti torsionali fisici delle barre antirollio")
        self.btn_calc_from_arb_ranges.clicked.connect(self._open_calc_from_arb_ranges_dialog)
        preset_row.addWidget(self.btn_calc_from_arb_ranges)

        cfg_side.addLayout(preset_row)

        curve_btn_row = QHBoxLayout()
        curve_btn_row.addWidget(QLabel("<b>Curve Digitalizzate:</b>"))
        self.btn_load_curve05_ideal = QPushButton("Curva 05: Ideale (Telaio Rigido)")
        self.btn_load_curve05_ideal.setToolTip("Carica i 4 vertici del rombo dalla curva digitalizzata 05 (1000, 1370, 1650, 1420 Nm/deg)")
        self.btn_load_curve05_ideal.setStyleSheet("font-size: 10.5px; padding: 2px 6px;")
        self.btn_load_curve05_ideal.clicked.connect(self.load_digitized_arb_curve_05)
        curve_btn_row.addWidget(self.btn_load_curve05_ideal)

        self.btn_load_curve06_real = QPushButton("Curva 06: Reale (Torsione Inclusa)")
        self.btn_load_curve06_real.setToolTip("Carica i 4 vertici del rombo dalla curva digitalizzata 06 con flessibilità telaio (1000, 1300, 1600, 1350 Nm/deg)")
        self.btn_load_curve06_real.setStyleSheet("font-size: 10.5px; padding: 2px 6px; font-weight: bold; color: #0984e3;")
        self.btn_load_curve06_real.clicked.connect(self.load_digitized_arb_curve_06)
        curve_btn_row.addWidget(self.btn_load_curve06_real)
        cfg_side.addLayout(curve_btn_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Velocità Test $v_x$ [m/s]:"))
        self.diamond_vx_input = QLineEdit("15.0")
        self.diamond_vx_input.setMaximumWidth(50)
        param_row.addWidget(self.diamond_vx_input)

        param_row.addWidget(QLabel("Risoluzione Griglia 2D:"))
        self.diamond_grid_n_input = QLineEdit("9")
        self.diamond_grid_n_input.setMaximumWidth(40)
        param_row.addWidget(self.diamond_grid_n_input)
        param_row.addStretch()
        cfg_side.addLayout(param_row)

        btn_action_row = QHBoxLayout()
        self.btn_calc_diamond_fast = QPushButton(" Analizza Vertici (14-DOF)")
        self.btn_calc_diamond_fast.setStyleSheet("font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_diamond_fast.clicked.connect(self.plot_arb_diamond_vertices)
        btn_action_row.addWidget(self.btn_calc_diamond_fast)

        self.btn_calc_diamond_2d = QPushButton(" Mappa 2D Rombo (Contour)")
        self.btn_calc_diamond_2d.setStyleSheet("font-weight: bold; background-color: #00b894; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_diamond_2d.clicked.connect(self.plot_arb_diamond_2d_map)
        btn_action_row.addWidget(self.btn_calc_diamond_2d)

        self.btn_calc_diamond_1d = QPushButton(" Sweep 1D Bilanciamento")
        self.btn_calc_diamond_1d.setStyleSheet("font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_calc_diamond_1d.clicked.connect(self.plot_arb_diamond_1d_sweep)
        btn_action_row.addWidget(self.btn_calc_diamond_1d)

        cfg_side.addLayout(btn_action_row)

        # Barra di Progresso Asincrono e Pulsante Annulla (Zero Blocco UI)
        progress_row = QHBoxLayout()
        self.diamond_progress_bar = QProgressBar()
        self.diamond_progress_bar.setRange(0, 100)
        self.diamond_progress_bar.setValue(0)
        self.diamond_progress_bar.setTextVisible(True)
        self.diamond_progress_bar.setStyleSheet("QProgressBar { border: 1px solid #b2bec3; border-radius: 3px; text-align: center; font-size: 10px; } QProgressBar::chunk { background-color: #0984e3; }")
        self.diamond_progress_bar.setVisible(False)
        progress_row.addWidget(self.diamond_progress_bar, stretch=1)

        self.diamond_btn_cancel = QPushButton("[X] Annulla")
        self.diamond_btn_cancel.setStyleSheet("background-color: #d63031; color: white; font-weight: bold; padding: 2px 8px; border-radius: 3px;")
        self.diamond_btn_cancel.clicked.connect(self.cancel_diamond_calculation)
        self.diamond_btn_cancel.setVisible(False)
        progress_row.addWidget(self.diamond_btn_cancel)

        cfg_side.addLayout(progress_row)

        row1_layout.addLayout(cfg_side, stretch=3)
        diamond_cfg_layout.addLayout(row1_layout)

        # Banner informativo stato
        self.lbl_diamond_status = QLabel("Stato Rombo ARB: Pronto per l'analisi con modello 14-DOF.")
        self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #2d3436;")
        diamond_cfg_layout.addWidget(self.lbl_diamond_status)

        self.tab_arb_diamond_layout.addWidget(diamond_cfg_group)

        # 2. Area Analitica Splitter (Canvas + Tabella Risultati)
        diamond_splitter = QSplitter(QtCore.Qt.Horizontal)

        fig_container = QWidget()
        fig_layout = QVBoxLayout(fig_container)
        fig_layout.setContentsMargins(0, 0, 0, 0)
        
        diamond_toolbar_row = QHBoxLayout()
        self.fig_arb_diamond = Figure(figsize=(10, 6), dpi=100)
        self.canvas_arb_diamond = FigureCanvas(self.fig_arb_diamond)
        self.toolbar_arb_diamond = NavigationToolbar(self.canvas_arb_diamond, self.tab_arb_diamond)
        diamond_toolbar_row.addWidget(self.toolbar_arb_diamond)
        
        self.btn_apply_diamond_sweetspot = QPushButton("[OK] Applica Ottimo Consigliato al Setup Attivo")
        self.btn_apply_diamond_sweetspot.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_apply_diamond_sweetspot.clicked.connect(self.apply_diamond_sweetspot_to_vehicle)
        diamond_toolbar_row.addWidget(self.btn_apply_diamond_sweetspot)
        
        self.setup_popout(self.canvas_arb_diamond, "Rombo di Setting ARB & Bilanciamento Meccanico 14-DOF", diamond_toolbar_row)
        fig_layout.addLayout(diamond_toolbar_row)
        fig_layout.addWidget(self.canvas_arb_diamond)
        diamond_splitter.addWidget(fig_container)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        lbl_tbl = QLabel("<b>Tabella KPI Ingegneristici dei Vertici & Setup Attuale:</b>")
        table_layout.addWidget(lbl_tbl)
        self.table_diamond_results = QTableWidget()
        self.table_diamond_results.setColumnCount(12)
        self.table_diamond_results.setHorizontalHeaderLabels([
            "Punto", "K_tot [Nm/deg]", "Ratio F [%]", "ay_max [g]", "T_skid [s]",
            "K_us [deg/g]", "Roll Grad [deg/g]", "t90 [s]", "Overshoot [%]",
            "Delta Util [%]", "Score Perf", "Comportamento"
        ])
        self.table_diamond_results.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table_diamond_results)
        diamond_splitter.addWidget(table_container)

        diamond_splitter.setSizes([650, 350])
        self.tab_arb_diamond_layout.addWidget(diamond_splitter)
        self.setup_subtabs.addTab(self.tab_arb_diamond, " Rombo ARB & Bilanciamento (14-DOF)")

        # Inizializza i preset dal disco
        self.populate_diamond_presets()
        self.combo_diamond_presets.currentIndexChanged.connect(self._on_diamond_preset_selected)

    def _get_diamond_setups_dir(self) -> str:
        """Restituisce il percorso della cartella delle configurazioni rombo ARB (setups/arb_diamond)."""
        setups_dir = self.get_setups_directory()
        arb_dir = os.path.join(setups_dir, "arb_diamond")
        os.makedirs(arb_dir, exist_ok=True)
        return arb_dir

    def populate_diamond_presets(self):
        """Popola il combo box con tutte le configurazioni salvate nella cartella setups/arb_diamond."""
        if not hasattr(self, 'combo_diamond_presets'):
            return
        arb_dir = self._get_diamond_setups_dir()
        self.combo_diamond_presets.blockSignals(True)
        self.combo_diamond_presets.clear()

        files = sorted([f for f in os.listdir(arb_dir) if f.endswith('.json')])
        if not files:
            default_presets = {
                'FSAE_Default.json': {'name': 'FSAE Predefinito (Medio)', 'v_soft_soft': [25000.0, 0.48], 'v_stiff_soft': [55000.0, 0.64], 'v_stiff_stiff': [85000.0, 0.49], 'v_soft_stiff': [55000.0, 0.36]},
                'Asciutto_High_Grip_Stiff.json': {'name': 'Asciutto High Grip (Stiff)', 'v_soft_soft': [40107.0, 0.50], 'v_stiff_soft': [68755.0, 0.66], 'v_stiff_stiff': [103132.0, 0.52], 'v_soft_stiff': [68755.0, 0.38]},
                'Bagnato_Rain_Soft.json': {'name': 'Bagnato Rain (Soft)', 'v_soft_soft': [14324.0, 0.46], 'v_stiff_soft': [31513.0, 0.62], 'v_stiff_stiff': [51566.0, 0.48], 'v_soft_stiff': [31513.0, 0.34]},
                'Ampio_Range_Sweep.json': {'name': 'Ampio Range (Sweep Esteso)', 'v_soft_soft': [17189.0, 0.45], 'v_stiff_soft': [57296.0, 0.70], 'v_stiff_stiff': [97403.0, 0.50], 'v_soft_stiff': [57296.0, 0.30]}
            }
            for fname, d in default_presets.items():
                p = os.path.join(arb_dir, fname)
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(d, f, indent=4)
            files = sorted([f for f in os.listdir(arb_dir) if f.endswith('.json')])

        for f in files:
            self.combo_diamond_presets.addItem(f)

        self.combo_diamond_presets.blockSignals(False)
        if self.combo_diamond_presets.count() > 0:
            self.combo_diamond_presets.setCurrentIndex(0)
            self._load_diamond_preset_by_filename(self.combo_diamond_presets.currentText())

    def _on_diamond_preset_selected(self, idx: int):
        """Carica la configurazione selezionata dal dropdown."""
        fname = self.combo_diamond_presets.currentText()
        if fname:
            self._load_diamond_preset_by_filename(fname)

    def _load_diamond_preset_by_filename(self, filename: str):
        """Carica la configurazione del rombo dal file JSON specificato."""
        arb_dir = self._get_diamond_setups_dir()
        path = os.path.join(arb_dir, filename)
        if not os.path.exists(path):
            return
        try:
            d_loaded = ARBDiamondConfig.load_json(path)
            verts_deg = convert_diamond_units(d_loaded, 'Nm/deg')
            for r_idx, (ktot_deg, rf) in enumerate(verts_deg):
                self.table_diamond_vertices.item(r_idx, 2).setText(f"{ktot_deg:.1f}")
                self.table_diamond_vertices.item(r_idx, 3).setText(f"{rf*100.0:.1f}")
            self.log(f"[Rombo ARB] Caricata configurazione: {filename} ({d_loaded.name})")
        except Exception as e:
            self.log(f"[Rombo ARB] Errore caricamento preset {filename}: {e}")

    def save_as_diamond_config_dialog(self):
        """Salva la configurazione corrente del rombo con un nuovo nome (Salva con Nome)."""
        arb_dir = self._get_diamond_setups_dir()
        curr_name = self.combo_diamond_presets.currentText()
        default_path = os.path.join(arb_dir, curr_name if curr_name else "nuovo_rombo_arb.json")

        path, _ = QFileDialog.getSaveFileName(
            self, "Salva Configurazione Rombo ARB con Nome",
            default_path, "Configurazioni Rombo ARB (*.json)"
        )
        if path:
            if not path.endswith('.json'):
                path += '.json'
            try:
                diamond = self._get_current_diamond_config()
                base_name = os.path.splitext(os.path.basename(path))[0]
                diamond.name = base_name.replace('_', ' ')
                diamond.save_json(path)

                self.populate_diamond_presets()
                idx = self.combo_diamond_presets.findText(os.path.basename(path))
                if idx >= 0:
                    self.combo_diamond_presets.setCurrentIndex(idx)

                QMessageBox.information(self, "Configurazione Salvata", f"Configurazione '{base_name}' salvata con successo in:\n{path}")
                self.log(f"[Rombo ARB] Salvata nuova configurazione con nome: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare la configurazione: {e}")

    def save_current_diamond_config(self):
        """Salva le modifiche dei vertici sul file di configurazione attualmente selezionato."""
        fname = self.combo_diamond_presets.currentText()
        if not fname:
            self.save_as_diamond_config_dialog()
            return
        arb_dir = self._get_diamond_setups_dir()
        path = os.path.join(arb_dir, fname)
        try:
            diamond = self._get_current_diamond_config()
            diamond.name = os.path.splitext(fname)[0].replace('_', ' ')
            diamond.save_json(path)
            QMessageBox.information(self, "Salva Configurazione", f"Modifiche salvate con successo su:\n{fname}")
            self.log(f"[Rombo ARB] Configurazione aggiornata su file: {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare: {e}")

    def browse_load_diamond_config(self):
        """Sfoglia e carica un file di configurazione rombo (.json) da qualsiasi posizione."""
        arb_dir = self._get_diamond_setups_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "Carica Configurazione Rombo ARB",
            arb_dir, "JSON Files (*.json);;All Files (*)"
        )
        if path and os.path.exists(path):
            try:
                d_loaded = ARBDiamondConfig.load_json(path)
                verts_deg = convert_diamond_units(d_loaded, 'Nm/deg')
                for r_idx, (ktot_deg, rf) in enumerate(verts_deg):
                    self.table_diamond_vertices.item(r_idx, 2).setText(f"{ktot_deg:.1f}")
                    self.table_diamond_vertices.item(r_idx, 3).setText(f"{rf*100.0:.1f}")

                fname = os.path.basename(path)
                idx = self.combo_diamond_presets.findText(fname)
                if idx >= 0:
                    self.combo_diamond_presets.setCurrentIndex(idx)
                self.log(f"[Rombo ARB] Configurazione caricata da: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Caricamento", f"Impossibile caricare il file: {e}")

    def load_digitized_arb_curve_05(self):
        """Carica direttamente i 4 vertici dalla curva digitalizzata 05 (Ideale - Telaio Rigido)."""
        d05 = ARBDiamondConfig.from_digitized_curve_05_ideal()
        verts_deg = convert_diamond_units(d05, 'Nm/deg')
        for r_idx, (ktot_deg, rf) in enumerate(verts_deg):
            self.table_diamond_vertices.item(r_idx, 2).setText(f"{ktot_deg:.1f}")
            self.table_diamond_vertices.item(r_idx, 3).setText(f"{rf*100.0:.1f}")
        idx = self.combo_diamond_presets.findText("05_Ideale_Telaio_Rigido.json")
        if idx >= 0:
            self.combo_diamond_presets.setCurrentIndex(idx)
        self.log("[Rombo ARB] Vertici impostati dalla Curva Digitalizzata 05 (Ideale - Telaio Rigido: 1000, 1370, 1650, 1420 Nm/deg).")

    def load_digitized_arb_curve_06(self):
        """Carica direttamente i 4 vertici dalla curva digitalizzata 06 (Reale - Torsione Telaio Inclusa)."""
        d06 = ARBDiamondConfig.from_digitized_curve_06_real()
        verts_deg = convert_diamond_units(d06, 'Nm/deg')
        for r_idx, (ktot_deg, rf) in enumerate(verts_deg):
            self.table_diamond_vertices.item(r_idx, 2).setText(f"{ktot_deg:.1f}")
            self.table_diamond_vertices.item(r_idx, 3).setText(f"{rf*100.0:.1f}")
        idx = self.combo_diamond_presets.findText("06_Reale_Torsione_Inclusa.json")
        if idx >= 0:
            self.combo_diamond_presets.setCurrentIndex(idx)
        self.log("[Rombo ARB] Vertici impostati dalla Curva Digitalizzata 06 (Reale - Torsione Telaio Inclusa: 1000, 1300, 1600, 1350 Nm/deg).")

    def delete_diamond_config_dialog(self):
        """Elimina la configurazione personalizzata corrente dopo conferma."""
        fname = self.combo_diamond_presets.currentText()
        if not fname:
            return
        if fname in ['ET18_FSAE_Default.json']:
            QMessageBox.warning(self, "Eliminazione Non Consentita", "La configurazione predefinita di base non può essere eliminata.")
            return

        ret = QMessageBox.question(
            self, "Conferma Eliminazione",
            f"Sei sicuro di voler eliminare la configurazione '{fname}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret == QMessageBox.Yes:
            arb_dir = self._get_diamond_setups_dir()
            path = os.path.join(arb_dir, fname)
            try:
                if os.path.exists(path):
                    os.remove(path)
                self.populate_diamond_presets()
                self.log(f"[Rombo ARB] Eliminata configurazione: {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Errore Eliminazione", f"Impossibile eliminare: {e}")

    def _get_current_diamond_config(self) -> ARBDiamondConfig:
        """Legge i 4 vertici dalla tabella table_diamond_vertices e restituisce ARBDiamondConfig in Nm/rad."""
        deg2rad = np.pi / 180.0
        try:
            k1_deg = float(self.table_diamond_vertices.item(0, 2).text())
            r1 = float(self.table_diamond_vertices.item(0, 3).text()) / 100.0

            k2_deg = float(self.table_diamond_vertices.item(1, 2).text())
            r2 = float(self.table_diamond_vertices.item(1, 3).text()) / 100.0

            k3_deg = float(self.table_diamond_vertices.item(2, 2).text())
            r3 = float(self.table_diamond_vertices.item(2, 3).text()) / 100.0

            k4_deg = float(self.table_diamond_vertices.item(3, 2).text())
            r4 = float(self.table_diamond_vertices.item(3, 3).text()) / 100.0

            v1 = (k1_deg / deg2rad, r1)
            v2 = (k2_deg / deg2rad, r2)
            v3 = (k3_deg / deg2rad, r3)
            v4 = (k4_deg / deg2rad, r4)
            return ARBDiamondConfig(v1, v2, v3, v4, name="Rombo Utente")
        except Exception:
            return ARBDiamondConfig()

    def _open_calc_from_arb_ranges_dialog(self):
        """Dialog per calcolare i vertici del rombo dai limiti fisici min/max delle barre ARB."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Calcolo Vertici del Rombo da Range Fisici ARB")
        layout = QGridLayout(dlg)
        
        vp = self._get_current_vehicle_params()
        deg2rad = np.pi / 180.0
        
        k_sp_f_deg = (vp.K_z_f * (vp.t_f**2) / 2.0) * deg2rad if hasattr(vp, 'K_z_f') else 140.0
        k_sp_r_deg = (vp.K_z_r * (vp.t_r**2) / 2.0) * deg2rad if hasattr(vp, 'K_z_r') else 105.0

        layout.addWidget(QLabel("Rigidezza Rollio Molle Ant. [Nm/deg]:"), 0, 0)
        in_sp_f = QLineEdit(f"{k_sp_f_deg:.1f}")
        layout.addWidget(in_sp_f, 0, 1)

        layout.addWidget(QLabel("Rigidezza Rollio Molle Post. [Nm/deg]:"), 1, 0)
        in_sp_r = QLineEdit(f"{k_sp_r_deg:.1f}")
        layout.addWidget(in_sp_r, 1, 1)

        layout.addWidget(QLabel("Barra ARB Ant. Minima [Nm/deg]:"), 2, 0)
        in_arb_f_min = QLineEdit("100.0")
        layout.addWidget(in_arb_f_min, 2, 1)

        layout.addWidget(QLabel("Barra ARB Ant. Massima [Nm/deg]:"), 3, 0)
        in_arb_f_max = QLineEdit("700.0")
        layout.addWidget(in_arb_f_max, 3, 1)

        layout.addWidget(QLabel("Barra ARB Post. Minima [Nm/deg]:"), 4, 0)
        in_arb_r_min = QLineEdit("80.0")
        layout.addWidget(in_arb_r_min, 4, 1)

        layout.addWidget(QLabel("Barra ARB Post. Massima [Nm/deg]:"), 5, 0)
        in_arb_r_max = QLineEdit("650.0")
        layout.addWidget(in_arb_r_max, 5, 1)

        btn_ok = QPushButton("Calcola e Applica al Rombo")
        btn_ok.setStyleSheet("font-weight: bold; background-color: #00b894; color: white;")
        layout.addWidget(btn_ok, 6, 0, 1, 2)

        def on_calc():
            try:
                rad_scale = 180.0 / np.pi
                d_calc = ARBDiamondConfig.from_arb_stiffness_ranges(
                    k_spring_phi_f=float(in_sp_f.text()) * rad_scale,
                    k_spring_phi_r=float(in_sp_r.text()) * rad_scale,
                    k_arb_f_min=float(in_arb_f_min.text()) * rad_scale,
                    k_arb_f_max=float(in_arb_f_max.text()) * rad_scale,
                    k_arb_r_min=float(in_arb_r_min.text()) * rad_scale,
                    k_arb_r_max=float(in_arb_r_max.text()) * rad_scale,
                    name="Calcolato da Range ARB"
                )
                verts_deg = convert_diamond_units(d_calc, 'Nm/deg')
                for r_idx, (ktot_deg, rf) in enumerate(verts_deg):
                    self.table_diamond_vertices.item(r_idx, 2).setText(f"{ktot_deg:.1f}")
                    self.table_diamond_vertices.item(r_idx, 3).setText(f"{rf*100.0:.1f}")
                self.log("[Rombo ARB] Vertici ricalcolati con successo dai limiti fisici delle barre.")
                dlg.accept()
            except Exception as e:
                QMessageBox.warning(dlg, "Errore Calcolo", f"Dati non validi: {e}")

        btn_ok.clicked.connect(on_calc)
        dlg.exec_()

    # -----------------------------------------------------------------------
    # Esecuzione Asincrona con Worker Thread (Nessun Congelamento GUI)
    # -----------------------------------------------------------------------

    def _set_diamond_ui_computing(self, computing: bool):
        """Abilita o disabilita i controlli durante il calcolo in background."""
        self.btn_calc_diamond_fast.setEnabled(not computing)
        self.btn_calc_diamond_2d.setEnabled(not computing)
        self.btn_calc_diamond_1d.setEnabled(not computing)
        self.diamond_progress_bar.setVisible(computing)
        self.diamond_btn_cancel.setVisible(computing)
        if computing:
            self.diamond_progress_bar.setValue(0)

    def cancel_diamond_calculation(self):
        """Interrompe il calcolo in corso."""
        if hasattr(self, 'diamond_worker') and self.diamond_worker.isRunning():
            self.diamond_worker.cancel()
            self.lbl_diamond_status.setText("Calcolo annullato dall'utente.")
            self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #e17055;")
            self._set_diamond_ui_computing(False)
            self.log("[Rombo ARB] Calcolo interrotto dall'utente.")

    def _on_diamond_worker_progress(self, done: int, total: int, msg: str):
        """Aggiorna la barra di progresso e lo stato live."""
        pct = int(done / max(total, 1) * 100)
        self.diamond_progress_bar.setValue(pct)
        self.lbl_diamond_status.setText(f" {msg} ({pct}%)")
        self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #0984e3;")

    def _on_diamond_worker_failed(self, err_msg: str):
        """Gestisce errori o eccezioni dal worker thread."""
        self._set_diamond_ui_computing(False)
        self.lbl_diamond_status.setText("Errore durante il calcolo 14-DOF.")
        self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #d63031;")
        self.log(f"[Rombo ARB Error] {err_msg}")
        QMessageBox.warning(self, "Errore Calcolo 14-DOF", f"Si è verificato un errore durante l'esecuzione:\n\n{err_msg}")

    def plot_arb_diamond_vertices(self):
        """Avvia la valutazione 14-DOF dei 4 vertici, centro e setup attuale in background."""
        self._set_diamond_ui_computing(True)
        self.log("Avvio valutazione Vertici del Rombo ARB (Modello 14-DOF)...")

        vp = self._get_current_vehicle_params()
        tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
        diamond = self._get_current_diamond_config()
        v_eval = float(self.diamond_vx_input.text()) if hasattr(self, 'diamond_vx_input') else 15.0

        self.diamond_worker = ARBDiamondWorker('vertices', vp, diamond, v_eval=v_eval, tir_path=tir_file)
        self.diamond_worker.progress.connect(self._on_diamond_worker_progress)
        self.diamond_worker.finished.connect(self._render_diamond_vertices_results)
        self.diamond_worker.failed.connect(self._on_diamond_worker_failed)
        self.diamond_worker.start()

    def _render_diamond_vertices_results(self, res_all: dict):
        """Esegue il rendering grafico dei vertici e popola la tabella risultati sul thread GUI."""
        self._set_diamond_ui_computing(False)
        try:
            vp = self._get_current_vehicle_params()
            diamond = self._get_current_diamond_config()
            results = res_all['results_dict']
            df_sum = res_all['summary_df']
            is_inside = res_all['is_current_inside']

            # Popola tabella risultati
            self.table_diamond_results.setRowCount(len(df_sum))
            for r_idx, row in df_sum.iterrows():
                self.table_diamond_results.setItem(r_idx, 0, QTableWidgetItem(str(row['Punto'])))
                self.table_diamond_results.setItem(r_idx, 1, QTableWidgetItem(f"{row['K_tot [Nm/deg]']:.1f}"))
                self.table_diamond_results.setItem(r_idx, 2, QTableWidgetItem(f"{row['Ratio Front [%]']:.1f}"))
                self.table_diamond_results.setItem(r_idx, 3, QTableWidgetItem(f"{row['ay_max [g]']:.3f}"))
                self.table_diamond_results.setItem(r_idx, 4, QTableWidgetItem(f"{row['T_skidpad [s]']:.2f}"))
                self.table_diamond_results.setItem(r_idx, 5, QTableWidgetItem(f"{row['K_us [deg/g]']:+.3f}"))
                self.table_diamond_results.setItem(r_idx, 6, QTableWidgetItem(f"{row['Roll Grad [deg/g]']:.2f}"))
                self.table_diamond_results.setItem(r_idx, 7, QTableWidgetItem(f"{row['t90 [s]']:.3f}"))
                self.table_diamond_results.setItem(r_idx, 8, QTableWidgetItem(f"{row['Overshoot [%]']:.1f}"))
                self.table_diamond_results.setItem(r_idx, 9, QTableWidgetItem(f"{row['Delta Util [%]']:+.1f}"))
                self.table_diamond_results.setItem(r_idx, 10, QTableWidgetItem(f"{row['Score Perf']:.1f}"))
                self.table_diamond_results.setItem(r_idx, 11, QTableWidgetItem(str(row['Comportamento'])))

            # Aggiorna Banner di Stato
            res_curr = results.get("Setup Attuale", {})
            status_txt = "[OK] NEL ROMBO" if is_inside else "[!] FUORI ROMBO"
            status_color = "#00b894" if is_inside else "#d63031"
            self.lbl_diamond_status.setText(
                f"{status_txt} | Setup Attuale: K_tot={res_curr.get('k_tot_nmdeg', 0):.1f} Nm/deg, "
                f"Ratio={res_curr.get('ratio_f_pct', 0):.1f}% | K_us={res_curr.get('kus_degg', 0):+.2f} deg/g "
                f"({res_curr.get('balance_class', '')}) | Score Performance: {res_curr.get('score_perf', 0):.1f}/100"
            )
            self.lbl_diamond_status.setStyleSheet(f"font-weight: bold; font-size: 11px; padding: 4px; color: {status_color};")

            self._last_diamond_best = results.get("Centro Geometria Rombo", res_curr)

            # Rendering Matplotlib (3 Subplots)
            self.fig_arb_diamond.clear()
            ax1 = self.fig_arb_diamond.add_subplot(121)
            ax2 = self.fig_arb_diamond.add_subplot(222, polar=True)
            ax3 = self.fig_arb_diamond.add_subplot(224)

            # Subplot 1: Rombo ARB nel piano (K_tot, Ratio_f)
            verts_deg = convert_diamond_units(diamond, 'Nm/deg')
            x_poly = [v[0] for v in verts_deg] + [verts_deg[0][0]]
            y_poly = [v[1] * 100.0 for v in verts_deg] + [verts_deg[0][1] * 100.0]

            from matplotlib.patches import Polygon
            poly_patch = Polygon(np.column_stack([x_poly, y_poly]), closed=True, facecolor='#74b9ff', edgecolor='#0984e3', alpha=0.25, linewidth=2.0, label='Range di Setting ARB')
            ax1.add_patch(poly_patch)

            ax1.axhspan(45.0, 55.0, color='#00b894', alpha=0.15, label='Corridoio Ottimale Sottosterzo (Target 45-55%)')
            ax1.axhline(vp.wd_f * 100.0, color='gray', linestyle=':', linewidth=1.2, label=f'Massa Anteriore $wd_f$ ({vp.wd_f*100:.1f}%)')

            labels = ["V1: Soft-Soft", "V2: Stiff F / Soft R", "V3: Stiff-Stiff", "V4: Soft F / Stiff R"]
            colors = ['#0984e3', '#e17055', '#2d3436', '#6c5ce7']
            for (vx, vy), lbl, col in zip(verts_deg, labels, colors):
                ax1.plot(vx, vy * 100.0, marker='o', markersize=8, color=col, markeredgecolor='black', label=lbl)
                ax1.annotate(lbl.split(':')[0], (vx, vy * 100.0), textcoords="offset points", xytext=(6, 4), fontweight='bold', fontsize=8)

            c_k, c_r = diamond.get_center_point()
            ax1.plot(c_k * (np.pi/180), c_r * 100.0, marker='P', markersize=10, color='#fdcb6e', markeredgecolor='black', label='Centro Rombo (Medio)')

            cur_k_deg = (vp.K_phi_f + vp.K_phi_r) * (np.pi/180)
            cur_r_pct = (vp.K_phi_f / max(vp.K_phi_f + vp.K_phi_r, 1.0)) * 100.0
            ax1.plot(cur_k_deg, cur_r_pct, marker='D', markersize=11, color='#d63031', markeredgecolor='black', zorder=10, label=f'Setup Attuale ({cur_k_deg:.1f} Nm/deg, {cur_r_pct:.1f}%)')

            ax1.set_xlabel("Rigidezza a Rollio Totale $K_{\\phi,tot}$ [Nm/deg]", fontsize=9.5)
            ax1.set_ylabel("Ripartizione Rigidezza Rollio Ant. $\\frac{K_{\\phi,f}}{K_{\\phi,tot}}$ [%]", fontsize=9.5)
            ax1.set_title("1. Rombo di Setting ARB & Dominio di Bilanciamento", fontsize=10, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            leg1 = ax1.legend(fontsize='x-small', loc='best')
            if leg1: leg1.set_draggable(True)

            # Subplot 2: Radar Chart
            categories = ['Aderenza\nay_max', 'Velocità\nSkidpad', 'Controllo\nRollio', 'Sottosterzo\nTarget', 'Reattività\nt90', 'Stabilità\nTransitoria']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            def _get_radar_norm_vals(res_dict):
                ay_n = np.clip((res_dict.get('ay_max_g', 1.5) - 1.2) / (1.85 - 1.2) * 100, 10, 100)
                v_n = np.clip((res_dict.get('v_skid_max_ms', 11) - 9) / (13.5 - 9) * 100, 10, 100)
                roll_n = np.clip((1.5 - res_dict.get('roll_grad_degg', 1.0)) / (1.5 - 0.5) * 100, 10, 100)
                kus_n = np.clip(100 - abs(res_dict.get('kus_degg', 0.3) - 0.35) * 150, 10, 100)
                t90_n = np.clip((0.25 - res_dict.get('t90_s', 0.15)) / (0.25 - 0.08) * 100, 10, 100)
                stab_n = np.clip(100 - res_dict.get('overshoot_pct', 5) * 4, 10, 100)
                vals = [ay_n, v_n, roll_n, kus_n, t90_n, stab_n]
                vals += vals[:1]
                return vals

            vals_v2 = _get_radar_norm_vals(results.get("V2 (Ant. Rigido / Post. Morbido - Max Sottosterzo)", {}))
            vals_v4 = _get_radar_norm_vals(results.get("V4 (Ant. Morbido / Post. Rigido - Max Sovrasterzo)", {}))
            vals_cur = _get_radar_norm_vals(res_curr)

            ax2.plot(angles, vals_v2, 'o--', linewidth=1.8, color='#e17055', label='V2: Max Sottosterzo')
            ax2.fill(angles, vals_v2, color='#e17055', alpha=0.10)

            ax2.plot(angles, vals_v4, 's--', linewidth=1.8, color='#6c5ce7', label='V4: Max Sovrasterzo')
            ax2.fill(angles, vals_v4, color='#6c5ce7', alpha=0.10)

            ax2.plot(angles, vals_cur, 'D-', linewidth=2.4, color='#d63031', label='Setup Attuale')
            ax2.fill(angles, vals_cur, color='#d63031', alpha=0.20)

            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories, fontsize=7.5, fontweight='bold')
            ax2.set_ylim(0, 100)
            ax2.set_title("2. Radar Performance vs Guidabilità", fontsize=9.5, fontweight='bold', pad=12)
            leg2 = ax2.legend(loc='lower left', bbox_to_anchor=(-0.25, -0.22), fontsize='x-small')
            if leg2: leg2.set_draggable(True)

            # Subplot 3: Risposta Transitoria Step Steer
            telem_curr = res_curr.get('telemetry_df', None)
            res_v2 = results.get("V2 (Ant. Rigido / Post. Morbido - Max Sottosterzo)", {})
            telem_v2 = res_v2.get('telemetry_df', None)

            if telem_curr is not None and not telem_curr.empty:
                t_arr = telem_curr['t'].values
                r_curr = telem_curr['yaw_rate_deg_s'].values
                ax3.plot(t_arr, r_curr, 'r-', linewidth=2.0, label='Yaw Rate r(t) [Attuale]')
                if telem_v2 is not None and not telem_v2.empty:
                    ax3.plot(telem_v2['t'].values, telem_v2['yaw_rate_deg_s'].values, 'k--', linewidth=1.5, label='Yaw Rate [V2 Stiff F]')

                ax3_phi = ax3.twinx()
                ax3_phi.plot(t_arr, telem_curr['phi_deg'].values, 'b:', linewidth=1.6, label='Rollio φ(t) [Attuale]')
                ax3_phi.set_ylabel("Angolo di Rollio $\\phi$ [deg]", fontsize=8, color='blue')
                ax3_phi.tick_params(axis='y', labelcolor='blue')
                
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_phi.get_legend_handles_labels()
                leg3 = ax3.legend(lines1 + lines2, labels1 + labels2, fontsize='x-small', loc='best')
                if leg3: leg3.set_draggable(True)
            else:
                ax3.text(0.5, 0.5, "Telemetria Step Steer 14-DOF\nCalcolata con successo.", ha='center', va='center')

            ax3.set_xlabel("Tempo $t$ [s]", fontsize=8.5)
            ax3.set_ylabel("Velocità di Imbardata $r$ [deg/s]", fontsize=8.5)
            ax3.set_title("3. Risposta Transitoria Step Steer 14-DOF", fontsize=9.5, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)

            self.fig_arb_diamond.tight_layout()
            self.canvas_arb_diamond.draw()
            self.log("Analisi Vertici del Rombo ARB 14-DOF completata con successo.")
        except Exception as e:
            self.log(f"Errore rendering vertici: {traceback.format_exc()}")

    def plot_arb_diamond_2d_map(self):
        """Avvia la mappatura 2D del rombo in un worker thread asincrono."""
        self._set_diamond_ui_computing(True)
        self.log("Avvio Mappatura 2D Rombo ARB (Modello 14-DOF)...")

        vp = self._get_current_vehicle_params()
        tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
        diamond = self._get_current_diamond_config()
        v_eval = float(self.diamond_vx_input.text()) if hasattr(self, 'diamond_vx_input') else 15.0
        n_grid = int(self.diamond_grid_n_input.text()) if hasattr(self, 'diamond_grid_n_input') else 9
        n_grid = max(min(n_grid, 15), 5)

        self.diamond_worker = ARBDiamondWorker('2d_mesh', vp, diamond, n_k=n_grid, n_ratio=n_grid, v_eval=v_eval, tir_path=tir_file)
        self.diamond_worker.progress.connect(self._on_diamond_worker_progress)
        self.diamond_worker.finished.connect(self._render_diamond_2d_mesh_results)
        self.diamond_worker.failed.connect(self._on_diamond_worker_failed)
        self.diamond_worker.start()

    def _render_diamond_2d_mesh_results(self, mesh_res: dict):
        """Rendering delle mappe 2D di contour sul thread GUI."""
        self._set_diamond_ui_computing(False)
        try:
            vp = self._get_current_vehicle_params()
            diamond = self._get_current_diamond_config()
            
            K_deg = mesh_res['K_mesh_deg']
            R_pct = mesh_res['R_mesh'] * 100.0
            AY = mesh_res['AY_mesh']
            KUS = mesh_res['KUS_mesh']
            mask = mesh_res['INSIDE_mask']
            best_pt = mesh_res['best_point']
            self._last_diamond_best = best_pt

            verts_deg = convert_diamond_units(diamond, 'Nm/deg')
            x_poly = [v[0] for v in verts_deg] + [verts_deg[0][0]]
            y_poly = [v[1] * 100.0 for v in verts_deg] + [verts_deg[0][1] * 100.0]

            self.fig_arb_diamond.clear()
            ax1 = self.fig_arb_diamond.add_subplot(121)
            ax2 = self.fig_arb_diamond.add_subplot(122)

            # Subplot 1: Contour Performance
            AY_masked = np.ma.masked_where(~mask, AY)
            cf1 = ax1.contourf(K_deg, R_pct, AY_masked, levels=12, cmap='viridis', alpha=0.85)
            cbar1 = self.fig_arb_diamond.colorbar(cf1, ax=ax1)
            cbar1.set_label("Accelerazione Laterale Max $a_{y,max}$ [g]", fontsize=8.5)
            
            ax1.plot(x_poly, y_poly, 'k--', linewidth=2.0, label='Confine Rombo ARB')
            
            cur_k_deg = (vp.K_phi_f + vp.K_phi_r) * (np.pi/180)
            cur_r_pct = (vp.K_phi_f / max(vp.K_phi_f + vp.K_phi_r, 1.0)) * 100.0
            ax1.plot(cur_k_deg, cur_r_pct, marker='D', markersize=10, color='#d63031', markeredgecolor='black', label='Setup Attuale')
            ax1.plot(best_pt['k_tot_nmdeg'], best_pt['ratio_f_pct'], marker='*', markersize=14, color='#f1c40f', markeredgecolor='black', label='Sweet Spot (Ottimo)')

            ax1.set_xlabel("Rigidezza a Rollio Totale $K_{\\phi,tot}$ [Nm/deg]", fontsize=9)
            ax1.set_ylabel("Ripartizione Rollio Ant. $\\frac{K_{\\phi,f}}{K_{\\phi,tot}}$ [%]", fontsize=9)
            ax1.set_title("1. Mappa Performance Aderenza ($a_{y,max}$ [g])", fontsize=9.5, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.5)
            leg1 = ax1.legend(fontsize='x-small', loc='best')
            if leg1: leg1.set_draggable(True)

            # Subplot 2: Contour Guidabilita
            KUS_masked = np.ma.masked_where(~mask, KUS)
            cf2 = ax2.contourf(K_deg, R_pct, KUS_masked, levels=12, cmap='coolwarm', alpha=0.85)
            cbar2 = self.fig_arb_diamond.colorbar(cf2, ax=ax2)
            cbar2.set_label("Gradiente Sottosterzo $K_{us}$ [deg/g]", fontsize=8.5)

            try:
                cs_zero = ax2.contour(K_deg, R_pct, KUS_masked, levels=[0.0], colors=['black'], linewidths=[2.2], linestyles=['--'])
                ax2.clabel(cs_zero, inline=True, fontsize=8, fmt='K_us=0 (Neutro)')
            except Exception:
                pass

            ax2.plot(x_poly, y_poly, 'k--', linewidth=2.0, label='Confine Rombo ARB')
            ax2.plot(cur_k_deg, cur_r_pct, marker='D', markersize=10, color='#d63031', markeredgecolor='black', label='Setup Attuale')
            ax2.plot(best_pt['k_tot_nmdeg'], best_pt['ratio_f_pct'], marker='*', markersize=14, color='#f1c40f', markeredgecolor='black', label='Sweet Spot')

            ax2.set_xlabel("Rigidezza a Rollio Totale $K_{\\phi,tot}$ [Nm/deg]", fontsize=9)
            ax2.set_ylabel("Ripartizione Rollio Ant. $\\frac{K_{\\phi,f}}{K_{\\phi,tot}}$ [%]", fontsize=9)
            ax2.set_title("2. Mappa Guidabilità & Sottosterzo ($K_{us}$)", fontsize=9.5, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.5)
            leg2 = ax2.legend(fontsize='x-small', loc='best')
            if leg2: leg2.set_draggable(True)

            self.lbl_diamond_status.setText(
                f"[OK] Mappatura 2D Completata | Sweet Spot Ottimale: K_tot={best_pt['k_tot_nmdeg']:.1f} Nm/deg, "
                f"Ratio={best_pt['ratio_f_pct']:.1f}% | ay_max={best_pt.get('ay_max_g', 0):.3f} g, Kus={best_pt.get('kus_degg', 0):+.2f} deg/g"
            )
            self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #00b894;")

            self.fig_arb_diamond.tight_layout()
            self.canvas_arb_diamond.draw()
            self.log(f"Mappatura 2D completata con successo. Sweet Spot identificato: K_tot={best_pt['k_tot_nmdeg']:.1f} Nm/deg, Ratio={best_pt['ratio_f_pct']:.1f}%.")
        except Exception as e:
            self.log(f"Errore rendering mappa 2D: {traceback.format_exc()}")

    def plot_arb_diamond_1d_sweep(self):
        """Avvia lo sweep 1D di bilanciamento in un worker thread asincrono."""
        self._set_diamond_ui_computing(True)
        self.log("Avvio Sweep 1D Bilanciamento Meccanico (Modello 14-DOF)...")

        vp = self._get_current_vehicle_params()
        tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else ""
        k_tot = vp.K_phi_f + vp.K_phi_r
        v_eval = float(self.diamond_vx_input.text()) if hasattr(self, 'diamond_vx_input') else 15.0

        self.diamond_worker = ARBDiamondWorker('1d_sweep', vp, self._get_current_diamond_config(), k_tot=k_tot, v_eval=v_eval, tir_path=tir_file)
        self.diamond_worker.progress.connect(self._on_diamond_worker_progress)
        self.diamond_worker.finished.connect(self._render_diamond_1d_sweep_results)
        self.diamond_worker.failed.connect(self._on_diamond_worker_failed)
        self.diamond_worker.start()

    def _render_diamond_1d_sweep_results(self, res_data: dict):
        """Rendering dei grafici dello sweep 1D sul thread GUI."""
        self._set_diamond_ui_computing(False)
        try:
            df_sweep = res_data['sweep_df']
            k_tot = res_data['k_tot']

            self.fig_arb_diamond.clear()
            ax1 = self.fig_arb_diamond.add_subplot(121)
            ax2 = self.fig_arb_diamond.add_subplot(122)

            r_vals = df_sweep['ratio_f_pct'].values

            # Subplot 1: ay_max & Skidpad Time
            color_ay = '#0984e3'
            ax1.plot(r_vals, df_sweep['ay_max_g'], color=color_ay, linewidth=2.2, marker='o', label='Aderenza Max ay [g]')
            ax1.set_xlabel("Ripartizione Rigidezza Rollio Ant. [%]", fontsize=9)
            ax1.set_ylabel("Accelerazione Laterale Max $a_{y,max}$ [g]", fontsize=9, color=color_ay)
            ax1.tick_params(axis='y', labelcolor=color_ay)

            ax1_t = ax1.twinx()
            color_t = '#e17055'
            ax1_t.plot(r_vals, df_sweep['t_skidpad_s'], color=color_t, linestyle='--', linewidth=2.0, marker='s', label='Tempo Skidpad [s]')
            ax1_t.set_ylabel("Tempo Skidpad [s]", fontsize=9, color=color_t)
            ax1_t.tick_params(axis='y', labelcolor=color_t)

            ax1.set_title(f"1. Performance vs Bilanciamento ($K_{{tot}}={k_tot*(np.pi/180):.0f}$ Nm/deg)", fontsize=9.5, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)

            # Subplot 2: Kus & Delta Util
            color_k = '#6c5ce7'
            ax2.plot(r_vals, df_sweep['kus_degg'], color=color_k, linewidth=2.2, marker='^', label='Gradiente Sottosterzo K_us [deg/g]')
            ax2.axhline(0.0, color='black', linestyle='--', linewidth=1.2, label='Neutro (K_us = 0)')
            ax2.axhspan(0.15, 0.60, color='#00b894', alpha=0.15, label='Corridoio Ottimale Target')
            ax2.set_xlabel("Ripartizione Rigidezza Rollio Ant. [%]", fontsize=9)
            ax2.set_ylabel("Gradiente di Sottosterzo $K_{us}$ [deg/g]", fontsize=9, color=color_k)
            ax2.tick_params(axis='y', labelcolor=color_k)

            ax2_u = ax2.twinx()
            color_u = '#2d3436'
            ax2_u.plot(r_vals, df_sweep['delta_util_pct'], color=color_u, linestyle='-.', linewidth=1.8, label='Delta Util (F - R) [%]')
            ax2_u.set_ylabel("Differenziale Utilizzazione $\\Delta Util$ [%]", fontsize=8.5, color=color_u)
            ax2_u.tick_params(axis='y', labelcolor=color_u)

            ax2.set_title("2. Guidabilità & Sottosterzo vs Bilanciamento", fontsize=9.5, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            leg2 = ax2.legend(fontsize='x-small', loc='upper left')
            if leg2: leg2.set_draggable(True)

            self.lbl_diamond_status.setText("[OK] Sweep 1D Bilanciamento Meccanico completato.")
            self.lbl_diamond_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 4px; color: #00b894;")

            self.fig_arb_diamond.tight_layout()
            self.canvas_arb_diamond.draw()
            self.log("Sweep 1D Bilanciamento completato con successo.")
        except Exception as e:
            self.log(f"Errore rendering sweep 1D: {traceback.format_exc()}")

    def apply_diamond_sweetspot_to_vehicle(self):
        """Applica la configurazione ottima (Sweet Spot) ai parametri attivi del veicolo."""
        best_pt = getattr(self, '_last_diamond_best', None)
        if not best_pt:
            QMessageBox.information(self, "Applica Setup", "Esegui prima un'analisi dei vertici o la mappa 2D del rombo.")
            return

        try:
            ktot_nmrad = best_pt.get('k_tot', best_pt.get('k_tot_nmrad', 0.0))
            ratio_f = best_pt.get('ratio_f', 0.50)
            if ktot_nmrad <= 0:
                ktot_nmrad = best_pt.get('k_tot_nmdeg', 1000.0) * (180.0 / np.pi)

            kf, kr = stiffness_from_coords(ktot_nmrad, ratio_f)

            if hasattr(self, 'bounds_inputs') and self.bounds_inputs:
                if 'K_phi_f' in self.bounds_inputs:
                    self.bounds_inputs['K_phi_f']['val'].setText(f"{kf:.1f}")
                if 'K_phi_r' in self.bounds_inputs:
                    self.bounds_inputs['K_phi_r']['val'].setText(f"{kr:.1f}")
                if 'lltd' in self.bounds_inputs:
                    self.bounds_inputs['lltd']['val'].setText(f"{ratio_f:.3f}")

            self.vp.K_phi_f = kf
            self.vp.K_phi_r = kr
            self.vp.lltd = ratio_f

            QMessageBox.information(
                self, "Setup Applicato",
                f"Configurazione ottimale applicata al veicolo:\n\n"
                f"- Rigidezza Rollio Ant. (K_phi_f): {kf:.1f} Nm/rad ({kf*np.pi/180:.1f} Nm/deg)\n"
                f"- Rigidezza Rollio Post. (K_phi_r): {kr:.1f} Nm/rad ({kr*np.pi/180:.1f} Nm/deg)\n"
                f"- LLTD Elastico: {ratio_f*100.0:.1f}%\n\n"
                f"I campi della tabella parametri e le configurazioni attive sono state aggiornate."
            )
            self.log(f"[Rombo ARB] Applicato Sweet Spot: K_phi_f={kf:.1f} Nm/rad, K_phi_r={kr:.1f} Nm/rad, LLTD={ratio_f*100:.1f}%.")
        except Exception as e:
            QMessageBox.warning(self, "Errore Applicazione", f"Impossibile applicare setup: {e}")

    @property
    def vp(self):
        """Restituisce l'istanza VehicleParams coerente con gli input attivi dell'interfaccia."""
        if hasattr(self, '_active_vehicle_params') and self._active_vehicle_params is not None:
            return self._active_vehicle_params
        return self._get_current_vehicle_params()

    @vp.setter
    def vp(self, val):
        self._active_vehicle_params = val

    def _init_tab_seven_post(self):
        """Costruisce e integra la scheda interattiva per il Banco Dinamico 7-Post Shaker Rig."""
        try:
            from src.ui.ui_seven_post_rig import SevenPostRigTab
            self.tab_seven_post = SevenPostRigTab(main_window=self)
            self.setup_subtabs.addTab(self.tab_seven_post, " Banco 7-Post Shaker Rig (4 Stradali + 3 Aero)")
        except Exception as e:
            if hasattr(self, 'log'):
                self.log(f"[WARN] Impossibile inizializzare SevenPostRigTab in MainWindow: {e}")
            self.tab_seven_post = None

    def plot_kinematic_maps(self):
        """Plots the 6 comprehensive kinematic maps (Camber, Toe, Roll Center Heights, Roll Center Lateral Migration, Anti-Pitch, Motion Ratios) from active VehicleParams."""
        self.log("Plotting Kinematic Maps (Camber, Toe, Roll Centers, Anti-Features, Motion Ratios)...")
        QApplication.processEvents()
        
        vp = self._get_current_vehicle_params()
        try:
            z_max_mm = float(self.kin_zmax_input.text())
        except ValueError:
            z_max_mm = 35.0
            
        z_mm, cam_f, cam_r = vp.get_camber_curve(z_range_mm=(-z_max_mm, z_max_mm))
        _, toe_f, toe_r = vp.get_toe_curve(z_range_mm=(-z_max_mm, z_max_mm))
        phi_deg, rc_f, rc_r = vp.get_rollcenter_curve(phi_range_deg=(-2.0, 2.0))
        phi_lat_deg, rc_lat_f, rc_lat_r = vp.get_rollcenter_lat_curve(phi_range_deg=(-2.0, 2.0))
        _, anti_dive, anti_squat, anti_lift = vp.get_anti_features_curve(z_range_mm=(-z_max_mm, z_max_mm))
        _, mr_df, mr_dr, mr_af, mr_ar = vp.get_motion_ratio_curve(z_range_mm=(-z_max_mm, z_max_mm))
        delta_sw_deg, ack_pct = vp.get_ackermann_curve(delta_range_deg=(-125.0, 125.0))
        _, d_fl_deg, d_fr_deg, d_diff_deg = vp.get_steer_wheel_angles(delta_range_deg=(-125.0, 125.0))
        
        view_mode = self.combo_kin_view.currentIndex() if hasattr(self, 'combo_kin_view') else 0

        self.fig_kinematics.clear()
        
        if view_mode == 0:
            # Mode 0: 8 Subplots (4x2 Full View: Suspension Heave + Roll + Steering Ackermann)
            self.fig_kinematics.set_size_inches(10.0, 8.8)
            ax1 = self.fig_kinematics.add_subplot(421)
            ax2 = self.fig_kinematics.add_subplot(422)
            ax3 = self.fig_kinematics.add_subplot(423)
            ax4 = self.fig_kinematics.add_subplot(424)
            ax5 = self.fig_kinematics.add_subplot(425)
            ax6 = self.fig_kinematics.add_subplot(426)
            ax7 = self.fig_kinematics.add_subplot(427)
            ax8 = self.fig_kinematics.add_subplot(428)
        elif view_mode == 1:
            # Mode 1: 6 Subplots (3x2 Suspension & Roll Centers)
            self.fig_kinematics.set_size_inches(10.0, 7.0)
            ax1 = self.fig_kinematics.add_subplot(321)
            ax2 = self.fig_kinematics.add_subplot(322)
            ax3 = self.fig_kinematics.add_subplot(323)
            ax4 = self.fig_kinematics.add_subplot(324)
            ax5 = self.fig_kinematics.add_subplot(325)
            ax6 = self.fig_kinematics.add_subplot(326)
            ax7, ax8 = None, None
        else:
            # Mode 2: 2 Subplots (1x2 Detailed Steering & Ackermann Geometry)
            self.fig_kinematics.set_size_inches(10.0, 4.5)
            ax1 = ax2 = ax3 = ax4 = ax5 = ax6 = None
            ax7 = self.fig_kinematics.add_subplot(121)
            ax8 = self.fig_kinematics.add_subplot(122)
        
        if ax1 is not None:
            # 1. Camber Angle vs Wheel Travel
            ax1.plot(z_mm, cam_f, 'b-', linewidth=2.0, label=f'Front (Stat: {vp.camber_f_static:+.1f}°, a1: {vp.camber_gain_f_coeffs[2]:.1f}°/m)')
            ax1.plot(z_mm, cam_r, 'r--', linewidth=2.0, label=f'Rear (Stat: {vp.camber_r_static:+.1f}°, a1: {vp.camber_gain_r_coeffs[2]:.1f}°/m)')
            ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax1.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax1.set_xlabel("Escursione Ruota $z$ [mm] (+ = Bump)", fontsize=8)
            ax1.set_ylabel(r"Campanatura $\gamma$ [°]", fontsize=8)
            ax1.set_title("1. Campanatura Dinamica $\\gamma(z)$ vs Corsa Ruota", fontsize=9, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.5)
            leg1 = ax1.legend(fontsize=7, loc='best')
            if leg1: leg1.set_draggable(True)
            
            # 2. Toe / Bump-Steer vs Wheel Travel
            ax2.plot(z_mm, toe_f, 'b-', linewidth=2.0, label=f'Front (Stat: {vp.toe_f_static:+.1f}°, a1: {vp.toe_gain_f_coeffs[2]:.2f}°/m)')
            ax2.plot(z_mm, toe_r, 'r--', linewidth=2.0, label=f'Rear (Stat: {vp.toe_r_static:+.1f}°, a1: {vp.toe_gain_r_coeffs[2]:.2f}°/m)')
            ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax2.set_xlabel("Escursione Ruota $z$ [mm] (+ = Bump)", fontsize=8)
            ax2.set_ylabel(r"Convergenza Dinamica $\delta_{toe}$ [°]", fontsize=8)
            ax2.set_title("2. Bump-Steer $\\delta_{toe}(z)$ vs Corsa Ruota", fontsize=9, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.5)
            leg2 = ax2.legend(fontsize=7, loc='best')
            if leg2: leg2.set_draggable(True)
            
            # 3. Roll Center Height vs Roll Angle phi
            ax3.plot(phi_deg, rc_f, 'b-', linewidth=2.0, label=f'RC Front ($h_{{rc0}}={vp.rollcenter_f_coeffs[3]*1000:.1f}$ mm)')
            ax3.plot(phi_deg, rc_r, 'r--', linewidth=2.0, label=f'RC Rear ($h_{{rc0}}={vp.rollcenter_r_coeffs[3]*1000:.1f}$ mm)')
            ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax3.set_xlabel(r"Angolo di Rollio Telaio $\phi$ [°]", fontsize=8)
            ax3.set_ylabel("Altezza Centro di Rollio $h_{rc}$ [mm]", fontsize=8)
            ax3.set_title(r"3. Altezza Centri di Rollio $h_{rc}(\phi)$ vs Rollio $\phi$", fontsize=9, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.5)
            leg3 = ax3.legend(fontsize=7, loc='best')
            if leg3: leg3.set_draggable(True)

            # 4. Roll Center Lateral Migration vs Roll Angle phi
            rc_lat_f_grad = vp.rollcenter_lat_f_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_f_coeffs') and len(vp.rollcenter_lat_f_coeffs) > 2 else 0.0
            rc_lat_r_grad = vp.rollcenter_lat_r_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_r_coeffs') and len(vp.rollcenter_lat_r_coeffs) > 2 else 0.0
            ax4.plot(phi_lat_deg, rc_lat_f, 'b-', linewidth=2.0, label=f'RC Lat Front ({rc_lat_f_grad:+.1f} mm/°)')
            ax4.plot(phi_lat_deg, rc_lat_r, 'r--', linewidth=2.0, label=f'RC Lat Rear ({rc_lat_r_grad:+.1f} mm/°)')
            ax4.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax4.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax4.set_xlabel(r"Angolo di Rollio Telaio $\phi$ [°]", fontsize=8)
            ax4.set_ylabel("Migrazione Laterale $y_{rc}$ [mm]", fontsize=8)
            ax4.set_title(r"4. Migrazione Laterale Centri di Rollio $y_{rc}(\phi)$ vs Rollio $\phi$", fontsize=9, fontweight='bold')
            ax4.grid(True, linestyle=':', alpha=0.5)
            leg4 = ax4.legend(fontsize=7, loc='best')
            if leg4: leg4.set_draggable(True)

            # 5. Anti-Pitch Features vs Wheel Travel
            ax5.plot(z_mm, anti_dive, 'b-', linewidth=2.0, label=f'Anti-Dive Ant. ({anti_dive[len(anti_dive)//2]:.1f}%)')
            ax5.plot(z_mm, anti_squat, 'tab:orange', linestyle='--', linewidth=2.0, label=f'Anti-Squat Post. ({anti_squat[len(anti_squat)//2]:.1f}%)')
            ax5.plot(z_mm, anti_lift, 'g-.', linewidth=2.0, label=f'Anti-Lift Post. ({anti_lift[len(anti_lift)//2]:.1f}%)')
            ax5.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax5.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax5.set_xlabel("Escursione Ruota $z$ [mm] (+ = Bump)", fontsize=8)
            ax5.set_ylabel("Anti-Pitch Fraction [%]", fontsize=8)
            ax5.set_title("5. Geometrie Anti-Pitch vs Corsa Ruota", fontsize=9, fontweight='bold')
            ax5.grid(True, linestyle=':', alpha=0.5)
            leg5 = ax5.legend(fontsize=7, loc='best')
            if leg5: leg5.set_draggable(True)
            
            # 6. Motion Ratios
            ax6.plot(z_mm, mr_df, 'b-', linewidth=1.8, label=f'MR Damper F ({vp.motion_ratio_f:.2f})')
            ax6.plot(z_mm, mr_dr, 'r--', linewidth=1.8, label=f'MR Damper R ({vp.motion_ratio_r:.2f})')
            ax6.plot(z_mm, mr_af, 'c:', linewidth=1.8, label=f'MR ARB F ({vp.motion_ratio_arb_f:.2f})')
            ax6.plot(z_mm, mr_ar, 'm-.', linewidth=1.8, label=f'MR ARB R ({vp.motion_ratio_arb_r:.2f})')
            ax6.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax6.set_xlabel("Escursione Ruota $z$ [mm]", fontsize=8)
            ax6.set_ylabel("Motion Ratio [-]", fontsize=8)
            ax6.set_title(f"6. Motion Ratios Ammortizzatori & Barre ARB", fontsize=9, fontweight='bold')
            ax6.grid(True, linestyle=':', alpha=0.5)
            leg6 = ax6.legend(fontsize=7, loc='best')
            if leg6: leg6.set_draggable(True)
        
        if ax7 is not None and ax8 is not None:
            # 7. Ackermann Steering Percentage vs Steering Wheel Angle
            ax7.plot(delta_sw_deg, ack_pct, color='#00cec9', linewidth=2.3, label=f'Ackermann Attivo (Stat: {vp.ackermann_pct*100:.1f}%)')
            ax7.scatter([0.0], [vp.ackermann_pct * 100.0], color='#d63031', s=45, zorder=5, label=f'Limite a 0° (L\'Hôpital: {vp.ackermann_pct*100:.1f}%)')
            ax7.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax7.axvspan(-8, 8, color='yellow', alpha=0.15, label='Zona Asintotica Adams (|δ| < 8°)')
            ax7.set_xlim(-125, 125)
            ax7.set_ylim(45, max(75.0, float(np.nanmax(ack_pct)) + 5.0))
            ax7.set_xlabel(r"Angolo Volante / Pignone $\delta_{sw}$ [°]", fontsize=8)
            ax7.set_ylabel(r"Ackermann [\%]", fontsize=8)
            ax7.set_title(r"7. Geometria di Ackermann \%($\delta_{sw}$) vs Sterzo", fontsize=9, fontweight='bold')
            ax7.grid(True, linestyle=':', alpha=0.5)
            leg7 = ax7.legend(fontsize=7, loc='best')
            if leg7: leg7.set_draggable(True)

            # 8. Steered Wheel Angles & Steering Difference
            ax8.plot(delta_sw_deg, d_fl_deg, color='#0984e3', linewidth=1.8, label=r'Ruota SX $\delta_{fl}$ (Interna a SX)')
            ax8.plot(delta_sw_deg, d_fr_deg, color='#e84393', linewidth=1.8, linestyle='--', label=r'Ruota DX $\delta_{fr}$ (Esterna a SX)')
            ax8.plot(delta_sw_deg, d_diff_deg, color='#00b894', linewidth=2.0, label=r'Diff. $\Delta\delta = \delta_{fl} - \delta_{fr}$')
            ax8.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax8.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax8.set_xlim(-125, 125)
            ax8.set_xlabel(r"Angolo Volante / Pignone $\delta_{sw}$ [°]", fontsize=8)
            ax8.set_ylabel(r"Angoli Ruota $\delta$ [°]", fontsize=8)
            ax8.set_title(f"8. Angoli Ruote Sterzanti & Ackermann Diff. (Ratio: {vp.steering_ratio:.1f})", fontsize=9, fontweight='bold')
            ax8.grid(True, linestyle=':', alpha=0.5)
            leg8 = ax8.legend(fontsize=7, loc='best')
            if leg8: leg8.set_draggable(True)
        
        self.fig_kinematics.tight_layout()
        self.canvas_kinematics.draw()
        self._update_kinematics_status_banner()
        self.log("Kinematic maps plotted successfully.")

    def apply_kinematic_gains_from_ui(self):
        """Applies linear kinematic gains entered in the UI to active VehicleParams."""
        try:
            vp = self._get_current_vehicle_params()
            cam_f = float(self.kin_cam_gain_f_input.text())
            cam_r = float(self.kin_cam_gain_r_input.text())
            toe_f = float(self.kin_toe_gain_f_input.text())
            toe_r = float(self.kin_toe_gain_r_input.text())
            
            vp.camber_gain_f_coeffs[2] = cam_f
            vp.camber_gain_r_coeffs[2] = cam_r
            vp.toe_gain_f_coeffs[2] = toe_f
            vp.toe_gain_r_coeffs[2] = toe_r
            
            self._apply_vp_to_gui(vp)
            self.plot_kinematic_maps()
            self.log(f"Kinematic gains updated: Camber F={cam_f:.1f}, R={cam_r:.1f} deg/m | Toe F={toe_f:.2f}, R={toe_r:.2f} deg/m.")
            QMessageBox.information(self, "Gain Cinematici", "Gain cinematici lineari applicati con successo al veicolo attivo!")
        except Exception as e:
            self.log(f"Errore applicazione gain: {e}")
            QMessageBox.warning(self, "Errore", f"Impossibile applicare i gain: {e}")

    def load_kinematic_json(self):
        """Loads a 3rd-degree polynomial JSON exported by the Curve Digitizer and applies it to active VehicleParams."""
        import json
        options = QFileDialog.Options()
        default_dir = self.get_design_suspension_directory()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Carica Mappa Cinematica JSON", default_dir, "JSON Files (*.json)", options=options
        )
        if not file_name:
            return
            
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            curve_name = data.get("curve_name")
            if not curve_name:
                base = os.path.splitext(os.path.basename(file_name))[0]
                if base.endswith("_poly3"):
                    base = base[:-6]
                curve_name = base

            coeffs = data.get("coeffs")
            static_val = data.get("static_val")
            
            if not coeffs or len(coeffs) != 4:
                QMessageBox.warning(self, "Errore Formato", "Il file JSON selezionato non contiene 'coeffs' validi (4 coefficienti).")
                return
                
            vp = self._get_current_vehicle_params()
            vp.set_kinematic_map(curve_name, coeffs, static_val=static_val)
            
            # Persist to active parameters & synchronize GUI
            self._apply_vp_to_gui(vp)
            
            # Immediately replot
            self.plot_kinematic_maps()
            
            self.log(f"Mappa cinematica '{curve_name}' caricata con successo da: {file_name}")
            self._update_kinematics_status_banner(f"Caricata curva {curve_name}")
            QMessageBox.information(self, "Mappa Caricata", f"Mappa cinematica '{curve_name}' applicata con successo al veicolo attivo e grafici aggiornati!\nPolinomio: {data.get('description', '')}")
        except Exception as e:
            self.log(f"Errore caricamento mappa cinematica JSON: {e}")
            QMessageBox.critical(self, "Errore", f"Impossibile caricare il file JSON:\n{e}")

    def show_active_parameters_dialog(self):
        """Opens the comprehensive ActiveVehicleParametersInspectorDialog showing all active vehicle parameters and 15 kinematic curves."""
        try:
            from src.ui.vehicle_parameters_inspector_dialog import ActiveVehicleParametersInspectorDialog
            vp = self._get_current_vehicle_params()
            tp = getattr(self, 'tp', None)
            dlg = ActiveVehicleParametersInspectorDialog(vp, tp=tp, title="📋 Ispezione Parametri Attivi in Memoria (MDO Target Suite)", parent=self)
            dlg.exec_()
        except Exception as e:
            self.log(f"Errore apertura ispezione parametri: {e}")
            QMessageBox.critical(self, "Errore Ispezione", f"Impossibile aprire l'ispezione parametri:\n{e}")

    def _update_kinematics_status_banner(self, source_info=None):
        """Updates the status banner in tab_kinematics with active setup information."""
        if not hasattr(self, 'lbl_kin_status_banner'):
            return
        vp = self._get_current_vehicle_params()
        rc_f_mm = vp.rollcenter_f_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_f_coeffs') and len(vp.rollcenter_f_coeffs) > 3 else 0.0
        rc_r_mm = vp.rollcenter_r_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_r_coeffs') and len(vp.rollcenter_r_coeffs) > 3 else 0.0
        rc_lat_f_grad = vp.rollcenter_lat_f_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_f_coeffs') and len(vp.rollcenter_lat_f_coeffs) > 2 else 0.0
        rc_lat_r_grad = vp.rollcenter_lat_r_coeffs[2] * 1000.0 if hasattr(vp, 'rollcenter_lat_r_coeffs') and len(vp.rollcenter_lat_r_coeffs) > 2 else 0.0
        cam_f_a1 = vp.camber_gain_f_coeffs[2] if hasattr(vp, 'camber_gain_f_coeffs') and len(vp.camber_gain_f_coeffs) > 2 else 0.0
        cam_r_a1 = vp.camber_gain_r_coeffs[2] if hasattr(vp, 'camber_gain_r_coeffs') and len(vp.camber_gain_r_coeffs) > 2 else 0.0
        
        setup_name = getattr(self, '_active_setup_filename', None)
        if not setup_name:
            if hasattr(self, 'combo_setup_profile') and self.combo_setup_profile.count() > 0:
                setup_name = self.combo_setup_profile.currentText()
            else:
                setup_name = "Setup Corrente in Memoria"
        if source_info:
            setup_name = f"{setup_name} — {source_info}"

        banner_text = (
            f"🟢 <b>Setup Cinematica:</b> {setup_name} | <b>19/19 Curve Adams Car Caricate</b><br>"
            f"📐 <b>Centri di Rollio (f(φ [deg])):</b> Front={rc_f_mm:.1f} mm ({rc_lat_f_grad:+.1f} mm/°), "
            f"Rear={rc_r_mm:.1f} mm ({rc_lat_r_grad:+.1f} mm/°) | "
            f"🏎️ <b>Camber Gain (a1):</b> Ant={cam_f_a1:.1f}°/m, Post={cam_r_a1:.1f}°/m | "
            f"🎯 <b>Ackermann:</b> {vp.ackermann_pct*100.0:.1f}% (Ratio {vp.steering_ratio:.1f})"
        )
        self.lbl_kin_status_banner.setText(banner_text)

    def _on_kin_preset_combo_changed(self, idx):
        """Loads full digitized Adams Car kinematics pack for 2026 or 2025."""
        if idx <= 0:
            return
        setups_dir = self.get_setups_directory()
        if idx == 1:
            target_json = os.path.join(setups_dir, "et18_2026_kinematics.json")
            label = "ET-18 (2026)"
        elif idx == 2:
            target_json = os.path.join(setups_dir, "et17_2025_kinematics.json")
            label = "ET-17 (2025)"
        else:
            return

        if os.path.exists(target_json):
            try:
                from src.models.params_vehicle import VehicleParams
                vp = VehicleParams.load_json(target_json)
                self._active_setup_filename = os.path.basename(target_json)
                self._apply_vp_to_gui(vp)
                self.plot_kinematic_maps()
                self._update_kinematics_status_banner(f"Preset {label} Caricato")
                self.log(f"[KINEMATICS] Caricato preset '{label}' da: {target_json}")
                QMessageBox.information(
                    self, "Preset Cinematica Applicato",
                    f"Caricato con successo il setup '{label}'!\n\n"
                    f"File: {os.path.basename(target_json)}\n"
                    f"Tutte le 16 curve Adams Car (Camber, Toe, Roll Center vs φ, Anti-Pitch, MR, Ackermann) sono ora attive nel modello."
                )
            except Exception as e:
                self.log(f"[KINEMATICS] Errore caricamento preset {label}: {e}")
                QMessageBox.critical(self, "Errore Caricamento", f"Impossibile caricare il preset {label}:\n{e}")
        else:
            QMessageBox.warning(self, "File Non Trovato", f"Impossibile trovare il file preset:\n{target_json}")

    def _apply_sens_preset(self, preset_name):
        """Applies one-click quick configurations for common vehicle dynamics transient studies."""
        if preset_name == "rc_f":
            self.sens_mode_combo.setCurrentIndex(0) # 1D
            idx = self.sens_scen_x_combo.findData("rc_height_f")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("0.02")
            self.sens_max_x_input.setText("0.14")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("lltd_peak")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Sweep 1D Altezza Centro di Rollio Anteriore q₁ [0.02 - 0.14 m]")

        elif preset_name == "rc_r":
            self.sens_mode_combo.setCurrentIndex(0) # 1D
            idx = self.sens_scen_x_combo.findData("rc_height_r")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("0.02")
            self.sens_max_x_input.setText("0.16")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("lltd_peak")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Sweep 1D Altezza Centro di Rollio Posteriore q₂ [0.02 - 0.16 m]")

        elif preset_name == "rc_2d":
            self.sens_mode_combo.setCurrentIndex(1) # 2D
            idx_x = self.sens_scen_x_combo.findData("rc_height_f")
            if idx_x >= 0: self.sens_scen_x_combo.setCurrentIndex(idx_x)
            idx_y = self.sens_scen_y_combo.findData("rc_height_r")
            if idx_y >= 0: self.sens_scen_y_combo.setCurrentIndex(idx_y)
            self.sens_min_x_input.setText("0.02")
            self.sens_max_x_input.setText("0.12")
            self.sens_pts_x_spin.setValue(7)
            self.sens_min_y_input.setText("0.03")
            self.sens_max_y_input.setText("0.14")
            self.sens_pts_y_spin.setValue(7)
            kpi_idx = self.sens_kpi_combo.findData("lltd_peak")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Griglia 2D Heatmap Interazione Centri di Rollio (q₁ × q₂)")

        elif preset_name == "arb":
            self.sens_mode_combo.setCurrentIndex(0)
            idx = self.sens_scen_x_combo.findData("arb_distribution")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("0.35")
            self.sens_max_x_input.setText("0.75")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("overshoot_pct")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Sweep Ripartizione Rigidezza Antirollio ARB K_φ,f / K_tot")

        elif preset_name == "damp":
            self.sens_mode_combo.setCurrentIndex(0)
            idx = self.sens_scen_x_combo.findData("damping_phi")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("300.0")
            self.sens_max_x_input.setText("2500.0")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("overshoot_pct")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Tuning Smorzamento di Rollio C_φ (Overshoot & Settling Time)")

        elif preset_name == "mig":
            self.sens_mode_combo.setCurrentIndex(0)
            idx = self.sens_scen_x_combo.findData("rc_lat_gain_f")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("0.0")
            self.sens_max_x_input.setText("2.0")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("dFz_asym_pct")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Sweep Migrazione Laterale Centro di Rollio Anteriore a₁")

        elif preset_name == "hcg":
            self.sens_mode_combo.setCurrentIndex(0)
            idx = self.sens_scen_x_combo.findData("h_cg")
            if idx >= 0: self.sens_scen_x_combo.setCurrentIndex(idx)
            self.sens_min_x_input.setText("0.20")
            self.sens_max_x_input.setText("0.36")
            self.sens_pts_x_spin.setValue(11)
            kpi_idx = self.sens_kpi_combo.findData("phi_max_deg")
            if kpi_idx >= 0: self.sens_kpi_combo.setCurrentIndex(kpi_idx)
            self.sens_status_lbl.setText("Preset Caricato: Sweep Altezza Baricentro h_cg (Rollio & Trasferimento di Carico)")

    def _update_stability_card(self):
        """Calculates analytical stability derivatives and dynamic modes for the active vehicle setup."""
        if not hasattr(self, 'sens_stab_browser'):
            return
        try:
            vp = self._get_current_vehicle_params()
            try:
                v_x_val = float(self.sens_vx_input.text())
            except Exception:
                v_x_val = 15.0
            
            from src.mdo.mdo_global_optimizer import compute_stability_metrics
            tir_file = self.tir_combo.currentText()
            metrics = compute_stability_metrics(
                L=vp.L, wd_f=vp.wd_f, lltd=vp.lltd, aero_bal=vp.aero_balance,
                v_x=v_x_val, tir_path=tir_file
            )
            
            is_st = metrics.get('is_stable', True)
            badge_color = "#00b894" if is_st else "#d63031"
            badge_text = "[OK] DINAMICAMENTE STABILE" if is_st else "[!] ATTENZIONE INSTABILITÀ"
            
            N_b = metrics.get('N_beta', 0.0)
            N_b_deg = N_b * (np.pi / 180.0)
            N_r = metrics.get('N_r', 0.0)
            N_r_deg = N_r * (np.pi / 180.0)
            N_d = metrics.get('N_delta', 0.0)
            N_d_deg = N_d * (np.pi / 180.0)
            
            fn_val = metrics.get('f_n_hz')
            fn_str = f"{fn_val:.2f} Hz" if fn_val is not None else "Aperiodico (Instabile)"
            zeta_val = metrics.get('zeta')
            zeta_str = f"{zeta_val:.2f}" if zeta_val is not None else "N/A"
            
            html = f"""
            <div style="line-height: 1.25;">
                <span style="background-color: {badge_color}; color: white; font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 10px;">
                    {badge_text}
                </span>
                &nbsp;<b>v = {v_x_val:.1f} m/s ({v_x_val*3.6:.0f} km/h)</b> | <i>{metrics.get('behavior_it', '')}</i>
                <hr style="margin: 3px 0; border: 0.5px solid #636e72;"/>
                <table width="100%" style="font-size: 10px; color: #dfe6e9;">
                    <tr>
                        <td><b>N<sub>β</sub> (Stab. Direzionale):</b> <span style="color: #55efc4;">{N_b_deg:+.1f} Nm/°</span></td>
                        <td><b>N<sub>r</sub> (Smorz. Imbardata):</b> <span style="color: #74b9ff;">{N_r_deg:+.1f} Nm·s/°</span></td>
                        <td><b>N<sub>δ</sub> (Autorità Sterzo):</b> <span style="color: #ffeaa7;">{N_d_deg:+.1f} Nm/°</span></td>
                    </tr>
                    <tr>
                        <td><b>Frequenza Nat. f<sub>n</sub>:</b> <span style="color: #fab1a0;">{fn_str}</span></td>
                        <td><b>Smorzamento ζ:</b> <span style="color: #fab1a0;">{zeta_str}</span></td>
                        <td><b>Dynamic Index DI:</b> <span style="color: #a29bfe;">{metrics.get('DI', 1.0):.2f}</span></td>
                    </tr>
                    <tr>
                        <td><b>Gradiente K<sub>us</sub>:</b> <b>{metrics.get('K_us_degg', 0.0):+.2f} °/g</b></td>
                        <td colspan="2"><b>Velocità Limite:</b> {f"v_char = {metrics.get('v_char_kmh'):.1f} km/h" if metrics.get('v_char_kmh') != float('inf') else f"v_crit = {metrics.get('v_crit_kmh'):.1f} km/h"}</td>
                    </tr>
                </table>
            </div>
            """
            self.sens_stab_browser.setHtml(html)
        except Exception as e:
            self.sens_stab_browser.setText(f"Calcolo derivate di stabilità: {e}")

    def _on_sensitivity_mode_changed(self):
        is_2d = (self.sens_mode_combo.currentIndex() == 1)
        self.lbl_sens_scen_y.setVisible(is_2d)
        self.sens_scen_y_combo.setVisible(is_2d)
        self.lbl_sens_range_y.setVisible(is_2d)
        self.sens_min_y_input.setVisible(is_2d)
        self.lbl_sens_max_y.setVisible(is_2d)
        self.sens_max_y_input.setVisible(is_2d)
        self.lbl_sens_pts_y.setVisible(is_2d)
        self.sens_pts_y_spin.setVisible(is_2d)
        if is_2d:
            self.sens_pts_x_spin.setValue(7)
            self._on_sensitivity_scenario_changed()
        else:
            self.sens_pts_x_spin.setValue(11)

    def _on_sensitivity_scenario_changed(self):
        scen_x = self.sens_scen_x_combo.currentData()
        if scen_x in SCENARIO_CATALOG:
            info_x = SCENARIO_CATALOG[scen_x]
            self.sens_desc_browser.setText(f"<b>{info_x['label']}:</b>\n{info_x['description']}")
            self.sens_min_x_input.setText(str(info_x['default_range'][0]))
            self.sens_max_x_input.setText(str(info_x['default_range'][1]))

        if hasattr(self, 'sens_scen_y_combo') and self.sens_mode_combo.currentIndex() == 1:
            scen_y = self.sens_scen_y_combo.currentData()
            if scen_y in SCENARIO_CATALOG:
                info_y = SCENARIO_CATALOG[scen_y]
                self.sens_min_y_input.setText(str(info_y['default_range'][0]))
                self.sens_max_y_input.setText(str(info_y['default_range'][1]))

    def run_sensitivity_sweep(self):
        mode = '1D' if self.sens_mode_combo.currentIndex() == 0 else '2D'
        scen_x = self.sens_scen_x_combo.currentData()
        try:
            min_x = float(self.sens_min_x_input.text())
            max_x = float(self.sens_max_x_input.text())
            pts_x = self.sens_pts_x_spin.value()
            sweep_x = np.linspace(min_x, max_x, pts_x)
            v_x = float(self.sens_vx_input.text())
            steer_deg = float(self.sens_steer_input.text())
        except ValueError:
            QMessageBox.warning(self, "Parametri Non Validi", "Controlla i valori numerici inseriti per il range X o velocità/sterzo.")
            return

        params_dict = {
            'scenario_x': scen_x,
            'sweep_x': sweep_x,
            'v_x': v_x,
            'steer_deg': steer_deg,
            'maneuver': 'step_steer' if self.sens_maneuver_combo.currentIndex() == 0 else 'dlc',
            'model_type': "14-DOF" if self.model_combo.currentIndex() == 2 else ("7-DOF" if self.model_combo.currentIndex() == 1 else "14-DOF"),
            'tir_path': self.tir_combo.currentText()
        }

        if mode == '2D':
            scen_y = self.sens_scen_y_combo.currentData()
            try:
                min_y = float(self.sens_min_y_input.text())
                max_y = float(self.sens_max_y_input.text())
                pts_y = self.sens_pts_y_spin.value()
                sweep_y = np.linspace(min_y, max_y, pts_y)
            except ValueError:
                QMessageBox.warning(self, "Parametri Non Validi", "Controlla i valori numerici inseriti per il range Y.")
                return
            params_dict['scenario_y'] = scen_y
            params_dict['sweep_y'] = sweep_y
            params_dict['kpi_key'] = self.sens_kpi_combo.currentData()

        vp = self._get_current_vehicle_params(for_dynamic=True)

        self.btn_run_sensitivity.setEnabled(False)
        self.btn_export_sensitivity.setEnabled(False)
        self.sens_prog_bar.setValue(0)
        self.sens_status_lbl.setText("Avvio simulazioni in background...")
        self.log(f"[SENSITIVITY] Avvio sweep {mode} su {scen_x} ({params_dict['maneuver']}, {len(sweep_x)} punti)...")

        worker = SensitivityWorker(mode, vp, params_dict)
        worker.progress.connect(self._on_sensitivity_progress)
        worker.finished.connect(self._on_sensitivity_finished)
        worker.failed.connect(self._on_sensitivity_failed)
        self.sens_active_worker = worker
        worker.start()

    def _on_sensitivity_progress(self, done, total, msg):
        pct = int(done / max(total, 1) * 100)
        self.sens_prog_bar.setValue(pct)
        self.sens_status_lbl.setText(f" [{pct}%] {msg}")

    def _on_sensitivity_finished(self, res):
        self.btn_run_sensitivity.setEnabled(True)
        self.btn_export_sensitivity.setEnabled(True)
        self.sens_prog_bar.setValue(100)
        self.sens_status_lbl.setText("[OK] Simulazioni completate con successo!")
        self.last_sensitivity_res = res

        # Populate KPI Table
        df_kpis = res.get('kpi_table', res.get('all_kpis', pd.DataFrame()))
        if not df_kpis.empty:
            self.sens_table.clear()
            self.sens_table.setRowCount(len(df_kpis))
            self.sens_table.setColumnCount(len(df_kpis.columns))
            self.sens_table.setHorizontalHeaderLabels([str(c) for c in df_kpis.columns])
            for r_idx, (_, row) in enumerate(df_kpis.iterrows()):
                for c_idx, val in enumerate(row):
                    if isinstance(val, (int, float, np.floating)):
                        item = QTableWidgetItem(f"{val:.3f}")
                    else:
                        item = QTableWidgetItem(str(val))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.sens_table.setItem(r_idx, c_idx, item)
            self.sens_table.resizeColumnsToContents()

        # Render Plots
        if 'Z' in res:
            self.plot_sensitivity_2d(res)
        else:
            self.plot_sensitivity_1d(res)

        self.log(f"[SENSITIVITY] Sweep completato: {len(df_kpis)} simulazioni analizzate.")

    def _on_sensitivity_failed(self, err_msg):
        self.btn_run_sensitivity.setEnabled(True)
        self.btn_export_sensitivity.setEnabled(True)
        self.sens_prog_bar.setValue(0)
        self.sens_status_lbl.setText(f"[ERR] Errore durante la simulazione.")
        self.log(f"[SENSITIVITY] Errore: {err_msg}")
        QMessageBox.critical(self, "Errore Sensitivity", f"Errore durante l'esecuzione del sweep:\n{err_msg}")

    def plot_sensitivity_1d(self, res):
        self.fig_sensitivity.clear()
        sweep_vals = res.get('sweep_values', [])
        telemetries = res.get('telemetries', [])
        kpi_table = res.get('kpi_table', pd.DataFrame())
        param_label = res.get('param_label', 'Parametro')
        maneuver = res.get('maneuver', 'step_steer')
        target_kpi_key = self.sens_kpi_combo.currentData()
        target_kpi_name = self.sens_kpi_combo.currentText()
        unit_str = f" [{res.get('unit', '')}]" if res.get('unit') else ""

        N = len(telemetries)
        if N == 0 or all(df.empty for df in telemetries):
            ax = self.fig_sensitivity.add_subplot(111)
            ax.text(0.5, 0.5, "Nessun dato di telemetria disponibile.", ha='center', va='center')
            self.canvas_sensitivity.draw()
            return

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        v_min = float(min(sweep_vals))
        v_max = float(max(sweep_vals))
        if v_min == v_max:
            v_max = v_min + 1.0
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        cmap = plt.get_cmap('viridis')

        # 2x2 Subplots + 1 dedicated Colorbar column
        gs = self.fig_sensitivity.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], hspace=0.35, wspace=0.28)
        ax1 = self.fig_sensitivity.add_subplot(gs[0, 0])
        ax2 = self.fig_sensitivity.add_subplot(gs[0, 1])
        ax3 = self.fig_sensitivity.add_subplot(gs[1, 0])
        ax4 = self.fig_sensitivity.add_subplot(gs[1, 1])
        cax = self.fig_sensitivity.add_subplot(gs[:, 2])

        for i, df in enumerate(telemetries):
            if df.empty:
                continue
            val = sweep_vals[i]
            color = cmap(norm(val))
            lbl = f"{val:.3g}"

            # 1. Yaw Rate r(t)
            if 'yaw_rate_deg_s' in df.columns:
                ax1.plot(df['t'], df['yaw_rate_deg_s'], color=color, linewidth=1.6, label=lbl)

            # 2. Dynamic LLTD(t)
            if all(c in df.columns for c in ['Fz_fl_N', 'Fz_fr_N', 'Fz_rl_N', 'Fz_rr_N']):
                dFz_f = np.abs(df['Fz_fr_N'] - df['Fz_fl_N'])
                dFz_r = np.abs(df['Fz_rr_N'] - df['Fz_rl_N'])
                tot_dFz = dFz_f + dFz_r
                lltd_dyn = np.where(tot_dFz > 1.0, (dFz_f / tot_dFz) * 100.0, np.nan)
                ax2.plot(df['t'], lltd_dyn, color=color, linewidth=1.6, label=lbl)

            # 3. Roll Angle phi(t)
            if 'phi_deg' in df.columns:
                ax3.plot(df['t'], df['phi_deg'], color=color, linewidth=1.6, label=lbl)
            elif 'beta_deg' in df.columns:
                ax3.plot(df['t'], df['beta_deg'], color=color, linewidth=1.6, label=lbl)

        # Ax1 formatting
        ax1.set_xlabel("Tempo $t$ [s]", fontsize=8.5)
        ax1.set_ylabel("Yaw Rate $r$ [deg/s]", fontsize=8.5)
        ax1.set_title(f"1. Risposta Imbardata $r(t)$ vs {param_label}", fontsize=9.5, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        if maneuver == 'step_steer':
            ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Inizio Sterzo')

        # Ax2 formatting
        ax2.set_xlabel("Tempo $t$ [s]", fontsize=8.5)
        ax2.set_ylabel("LLTD Dinamico Frontale [%]", fontsize=8.5)
        ax2.set_title(f"2. Evoluzione LLTD Dinamico [%]", fontsize=9.5, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # Ax3 formatting
        ax3.set_xlabel("Tempo $t$ [s]", fontsize=8.5)
        ax3.set_ylabel("Angolo di Rollio $\\phi$ [deg]" if 'phi_deg' in telemetries[0].columns else "Side-slip $\\beta$ [deg]", fontsize=8.5)
        ax3.set_title("3. Dinamica di Rollio $\\phi(t)$ / Assetto", fontsize=9.5, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.6)

        # Ax4: Target KPI vs Parameter Curve
        if not kpi_table.empty and target_kpi_key in kpi_table.columns and 'param_value' in kpi_table.columns:
            pv = kpi_table['param_value'].values
            kv = kpi_table[target_kpi_key].values
            ax4.plot(pv, kv, 'o-', color='#0984e3', linewidth=2.2, markersize=6, label=target_kpi_name)
            
            # Linear gradient / trend
            if len(pv) >= 2:
                dp = pv[-1] - pv[0]
                dk = kv[-1] - kv[0]
                grad = dk / max(abs(dp), 1e-6)
                ax4.set_title(f"4. Sensibilità: {target_kpi_name} ($\\Delta={grad:+.2f}$/u)", fontsize=9.5, fontweight='bold')
            else:
                ax4.set_title(f"4. Trend {target_kpi_name}", fontsize=9.5, fontweight='bold')
                
            ax4.set_xlabel(f"{param_label}{unit_str}", fontsize=8.5)
            ax4.set_ylabel(target_kpi_name, fontsize=8.5)
            ax4.grid(True, linestyle=':', alpha=0.6)
            leg4 = ax4.legend(fontsize='x-small')
            if leg4:
                leg4.set_draggable(True)

        # Colorbar configuration
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.fig_sensitivity.colorbar(sm, cax=cax)
        cbar.set_label(f"Valore Parametro: {param_label}{unit_str}", fontsize=9, fontweight='bold')

        self.canvas_sensitivity.draw()

    def plot_sensitivity_2d(self, res):
        self.fig_sensitivity.clear()
        Z = res.get('Z')
        x_vals = np.array(res.get('x_values', []))
        y_vals = np.array(res.get('y_values', []))
        x_lbl = res.get('x_label', 'Parametro X')
        y_lbl = res.get('y_label', 'Parametro Y')
        kpi_lbl = res.get('kpi_label', 'KPI')

        if Z is None or np.all(np.isnan(Z)):
            ax = self.fig_sensitivity.add_subplot(111)
            ax.text(0.5, 0.5, "Matrice 2D non valida o simulazioni fallite.", ha='center', va='center')
            self.canvas_sensitivity.draw()
            return

        X, Y = np.meshgrid(x_vals, y_vals)

        # 2 Subplots: 121 Contour Heatmap, 122 Slices Cross-Section
        ax1 = self.fig_sensitivity.add_subplot(121)
        ax2 = self.fig_sensitivity.add_subplot(122)

        # Heatmap Contourf
        cf = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
        cs = ax1.contour(X, Y, Z, levels=10, colors='white', linewidths=0.9, alpha=0.7)
        ax1.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        cb = self.fig_sensitivity.colorbar(cf, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cb.set_label(f"KPI: {kpi_lbl}", fontsize=9, fontweight='bold')

        ax1.set_xlabel(x_lbl, fontsize=9, fontweight='bold')
        ax1.set_ylabel(y_lbl, fontsize=9, fontweight='bold')
        ax1.set_title(f"Mappa Bivariata 2D: {kpi_lbl}", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.4, color='white')

        # Slices in ax2
        import matplotlib.pyplot as plt
        cmap_slices = plt.get_cmap('coolwarm', max(len(y_vals), 1))
        for iy, y_v in enumerate(y_vals):
            row_z = Z[iy, :]
            c = cmap_slices(iy / max(len(y_vals) - 1, 1))
            ax2.plot(x_vals, row_z, 'o-', color=c, linewidth=1.8, label=f"{y_lbl.split()[0]}={y_v:.3g}")

        ax2.set_xlabel(x_lbl, fontsize=9, fontweight='bold')
        ax2.set_ylabel(kpi_lbl, fontsize=9, fontweight='bold')
        ax2.set_title(f"Curve di Sensibilità a {y_lbl.split()[0]} costante", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        leg2 = ax2.legend(fontsize='x-small', loc='best')
        if leg2:
            leg2.set_draggable(True)

        self.fig_sensitivity.tight_layout()
        self.canvas_sensitivity.draw()

    def export_sensitivity_csv(self):
        if not self.last_sensitivity_res:
            QMessageBox.information(self, "Esporta CSV", "Nessun risultato di simulazione da esportare. Esegui prima uno sweep.")
            return

        df = self.last_sensitivity_res.get('kpi_table', self.last_sensitivity_res.get('all_kpis', pd.DataFrame()))
        if df.empty:
            QMessageBox.warning(self, "Esporta CSV", "Nessun dato valido presente nella tabella KPI.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Salva Risultati Sensitivity Sweep CSV", "sensitivity_sweep_results.csv", "CSV Files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Esporta CSV", f"File salvato con successo:\n{path}")
            self.log(f"[EXPORT] Risultati sensitivity esportati in: {path}")

    def launch_driving_simulator(self):
        """Launches the real-time interactive 14-DOF Gamepad driving simulator with bidirectional MDO link."""
        try:
            from src.ui.interactive_driver_14dof import Interactive14DOFDrivingWindow
            vp = self._get_current_vehicle_params(for_dynamic=True)
            tir_file = self.tir_combo.currentText()
            self.driver_window = Interactive14DOFDrivingWindow(vehicle_params=vp, tir_file=tir_file)
            self.driver_window.setup_changed.connect(self._on_simulator_setup_changed)
            self.driver_window.show()
            self.log("[SIM] Simulatore di Guida 14-DOF Real-Time avviato con successo! (Link MDO bidirezionale attivo)")
        except Exception as e:
            import traceback
            self.log(f"Errore avvio Simulatore di Guida 14-DOF: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Errore Simulatore", f"Impossibile avviare il simulatore di guida:\n{e}")

    def launch_3d_preview_studio(self):
        """Avvia lo Studio Anteprima 3D per il veicolo selezionato (Formula Student o Piccolo SUV)."""
        try:
            vp = self._get_current_vehicle_params()
            v_type = getattr(vp, 'vehicle_type', 'formula')
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
            self.log(f"[STUDIO 3D] Studio Anteprima 3D aperto in-process ({v_type})!")
        except Exception as e:
            import traceback
            self.log(f"Errore avvio Studio 3D: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Errore Studio 3D", f"Impossibile avviare lo Studio 3D:\n{e}")

    def _on_simulator_setup_changed(self, new_vp):
        """Callback triggered when vehicle setup is modified inside the driving simulator."""
        try:
            self._apply_vp_to_gui(new_vp)
            if hasattr(self, 'plot_kinematic_maps'):
                self.plot_kinematic_maps()
            self.log(f"[MDO SYNC] Assetto sincronizzato dal simulatore: Cz={new_vp.Cz:.3f}, Bal={new_vp.aero_balance*100:.1f}%, Camber F/R={new_vp.camber_f_static:.2f}°/{new_vp.camber_r_static:.2f}°")
        except Exception as e:
            self.log(f"[MDO SYNC] Errore sincronizzazione assetto: {e}")

    def run_direct_collocation_dialog(self):
        """Opens the Direct Collocation Trajectory & Racing Line Optimization dialog."""
        from src.ui.ui_direct_collocation import DirectCollocationDialog
        
        theme_name = getattr(self, 'current_theme', 'dark')
        density = getattr(self, 'current_density', 'standard')
        vp = self._get_current_vehicle_params(for_dynamic=True)
        tir_file = getattr(self, 'current_tir_file', None)

        dialog = DirectCollocationDialog(
            parent=self,
            vehicle_params=vp,
            tir_file=tir_file,
            theme_name=theme_name,
            density=density
        )
        dialog.exec_()

    def on_fsae_benchmark_changed(self):
        from src.mdo import mdo_sensitivity as ms
        key = self.fsae_bench_combo.currentData()
        bench = ms.FS_WORLD_RANKING_BENCHMARKS.get(key, ms.FS_WORLD_RANKING_BENCHMARKS['FS_Germany'])
        self.bench_taccel_input.setText(f"{bench['accel_tmin']:.2f}")
        self.bench_tskid_input.setText(f"{bench['skid_tmin']:.2f}")
        self.bench_tautox_input.setText(f"{bench['autox_tmin']:.2f}")
        self.bench_tendur_input.setText(f"{bench['endur_tmin']:.1f}")
        self.bench_energy_input.setText(f"{bench['energy_min_kwh']:.2f}")
        
        is_custom = (key == "Custom")
        self.bench_taccel_input.setEnabled(is_custom)
        self.bench_tskid_input.setEnabled(is_custom)
        self.bench_tautox_input.setEnabled(is_custom)
        self.bench_tendur_input.setEnabled(is_custom)
        self.bench_energy_input.setEnabled(is_custom)

    def run_sensitivity_1d_gui(self):
        from src.mdo import mdo_sensitivity as ms
        self.btn_run_sens_1d.setEnabled(False)
        self.btn_run_sens_1d.setText("Calcolo...")
        QApplication.processEvents()
        
        try:
            param_key = self.sens_param_combo.currentData()
            param_name = self.sens_param_combo.currentText()
            metric_key = self.sens_metric_combo.currentData()
            metric_name = self.sens_metric_combo.currentText()
            delta_pct = float(self.sens_delta_pct_input.text())
            n_pts = max(int(self.sens_points_input.text()), 5)
            
            vp = self._get_current_vehicle_params(for_dynamic=True)
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else None
            nom_val = getattr(vp, param_key, 1.0)
            
            if abs(nom_val) < 1e-7:
                val_min = -1.0
                val_max = 1.0
            else:
                val_min = nom_val * (1.0 - delta_pct / 100.0) if nom_val >= 0 else nom_val * (1.0 + delta_pct / 100.0)
                val_max = nom_val * (1.0 + delta_pct / 100.0) if nom_val >= 0 else nom_val * (1.0 - delta_pct / 100.0)
                
            vals = np.linspace(min(val_min, val_max), max(val_min, val_max), n_pts)
            self.log(f"[Sensitività] Calcolo sweep 1D su {param_name} ({len(vals)} punti) per metrica: {metric_name}...")
            QApplication.processEvents()
            
            df_res, base_m = ms.run_sensitivity_1d(param_key, vals, metric_key, base_params=vp, tir_file=tir_file)
            
            clean_param_lbl = re.sub(r'[^\w\s\(\)\[\]\/\-\.\,\%\:\+\=]', '', param_name).strip()
            clean_metric_lbl = re.sub(r'[^\w\s\(\)\[\]\/\-\.\,\%\:\+\=]', '', metric_name).strip()
            
            self.axes_sens_1d.clear()
            self.axes_sens_1d.plot(df_res['param_value'], df_res['metric_value'], 'o-', color='#0984e3', linewidth=2.2, markersize=5, label='Risposta Metrica')
            self.axes_sens_1d.axvline(nom_val, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Nominale ({nom_val:.3g})')
            self.axes_sens_1d.axhline(base_m, color='gray', linestyle=':', linewidth=1.0)
            self.axes_sens_1d.plot(nom_val, base_m, 'r*', markersize=12, zorder=6)
            
            self.axes_sens_1d.set_xlabel(f"Valore Parametro: {clean_param_lbl}", fontsize=9, fontweight='bold')
            self.axes_sens_1d.set_ylabel(clean_metric_lbl, fontsize=9, fontweight='bold')
            self.axes_sens_1d.set_title(f"1. Sweep Sensitività 1D — {clean_param_lbl}\n(Valore Nominale = {base_m:.2f})", fontsize=10, fontweight='bold')
            self.axes_sens_1d.grid(True, linestyle=':', alpha=0.6)
            self.axes_sens_1d.legend(fontsize=8, loc='best')
            
            self.fig_sens.tight_layout()
            self.canvas_sens.draw()
            self.log(f"[Sensitività] Sweep 1D completato: range [{df_res['metric_value'].min():.2f} - {df_res['metric_value'].max():.2f}].")
        except Exception as e:
            import traceback
            self.log(f"Errore sweep sensitività: {traceback.format_exc()}")
        finally:
            self.btn_run_sens_1d.setEnabled(True)
            self.btn_run_sens_1d.setText(" Esegui Sweep 1D")

    def run_sensitivity_tornado_gui(self):
        from src.mdo import mdo_sensitivity as ms
        self.btn_run_sens_tornado.setEnabled(False)
        self.btn_run_sens_tornado.setText("Generazione Tornado...")
        QApplication.processEvents()
        
        try:
            metric_key = self.sens_metric_combo.currentData()
            metric_name = self.sens_metric_combo.currentText()
            delta_pct = 10.0
            
            vp = self._get_current_vehicle_params(for_dynamic=True)
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else None
            
            candidate_params = [
                'm', 'final_drive', 'wd_f', 'aero_balance', 'Cz', 'Cd',
                'K_phi_f', 'K_phi_r', 'q1', 'q2', 'camber_f_static', 'camber_r_static',
                'LMUY', 'P_max_kw', 'T_max_motor'
            ]
            self.log(f"[Sensitività] Generazione Diagramma Tornado Pareto (±{delta_pct:.0f}%) per {metric_name}...")
            QApplication.processEvents()
            
            df_tor, base_m = ms.run_tornado_sensitivity(candidate_params, delta_pct=delta_pct, metric_key=metric_key, base_params=vp, tir_file=tir_file)
            
            if not df_tor.empty:
                # Keep top 8 parameters
                df_top = df_tor.tail(8).reset_index(drop=True)
                
                self.axes_sens_tornado.clear()
                y_pos = np.arange(len(df_top))
                
                for idx, r in df_top.iterrows():
                    d_low = r['delta_low']
                    d_high = r['delta_high']
                    c_low = '#e17055' if d_low < 0 else '#00b894'
                    c_high = '#00b894' if d_high >= 0 else '#e17055'
                    
                    self.axes_sens_tornado.barh(idx, d_low, color=c_low, alpha=0.7, height=0.55, edgecolor='black', linewidth=0.5)
                    self.axes_sens_tornado.barh(idx, d_high, color=c_high, alpha=0.7, height=0.55, edgecolor='black', linewidth=0.5)
                    
                self.axes_sens_tornado.axvline(0, color='black', linewidth=1.2)
                self.axes_sens_tornado.set_yticks(y_pos)
                self.axes_sens_tornado.set_yticklabels(df_top['param_name'], fontsize=8.5, fontweight='bold')
                self.axes_sens_tornado.set_xlabel(f"Variazione Metrica Δ rispetto a Nominale ({base_m:.2f})", fontsize=8.5)
                self.axes_sens_tornado.set_title(f"2. Tornado / Pareto Chart (Variazione ±{delta_pct:.0f}%)\n[Verde: Guadagno | Rosso: Perdita]", fontsize=10, fontweight='bold')
                self.axes_sens_tornado.grid(True, linestyle=':', alpha=0.6)
                
                self.fig_sens.tight_layout()
                self.canvas_sens.draw()
                self.log(f"[Sensitività] Tornado Chart generato con successo. Parametro più influente: {df_top.iloc[-1]['param_name']} (Swing = {df_top.iloc[-1]['total_swing']:.2f}).")
        except Exception as e:
            import traceback
            self.log(f"Errore generazione Tornado: {traceback.format_exc()}")
        finally:
            self.btn_run_sens_tornado.setEnabled(True)
            self.btn_run_sens_tornado.setText(" Genera Diagramma Tornado (Top 8 Parametri ±10%)")

    def run_fsae_scoring_gui(self):
        from src.mdo import mdo_sensitivity as ms
        self.btn_calc_fsae_scores.setEnabled(False)
        self.btn_calc_fsae_scores.setText("Simulazione...")
        QApplication.processEvents()
        
        try:
            bench_key = self.fsae_bench_combo.currentData()
            custom_bench = {
                'accel_tmin': float(self.bench_taccel_input.text()),
                'skid_tmin': float(self.bench_tskid_input.text()),
                'autox_tmin': float(self.bench_tautox_input.text()),
                'endur_tmin': float(self.bench_tendur_input.text()),
                'energy_min_kwh': float(self.bench_energy_input.text()),
            }
            
            vp = self._get_current_vehicle_params(for_dynamic=True)
            tir_file = self.tir_combo.currentText() if hasattr(self, 'tir_combo') else None
            
            self.log(f"[FSAE Scoring] Simulazione completa discipline dinamiche FS 2026 ({bench_key})...")
            QApplication.processEvents()
            
            perf = ms.simulate_vehicle_performance_metrics(vp, tir_file=tir_file)
            scores = ms.evaluate_fsae_2026_full_event(
                perf['accel_time'], perf['skidpad_time'], perf['autox_time'],
                perf['endurance_time'], perf['energy_kwh'],
                benchmark_event=bench_key, custom_benchmark=custom_bench
            )
            
            tot_pts = scores['total_dynamic_score']
            self.lbl_score_total.setText(f"Punteggio Totale: {tot_pts:.1f} / 525.0 pt")
            self.lbl_score_accel.setText(f"Accel: {scores['acceleration']['score']:.1f}/50")
            self.lbl_score_skid.setText(f"Skidpad: {scores['skidpad']['score']:.1f}/50")
            self.lbl_score_autox.setText(f"AutoX: {scores['autox']['score']:.1f}/100")
            self.lbl_score_endur.setText(f"Endurance: {scores['endurance']['score']:.1f}/250")
            self.lbl_score_eff.setText(f"Efficiency: {scores['efficiency']['score']:.1f}/75")
            
            # --- 1. RADAR / SPIDER CHART ---
            categories = ['Accel\n(50pt)', 'Skidpad\n(50pt)', 'AutoX\n(100pt)', 'Endur\n(250pt)', 'Eff\n(75pt)']
            pct_scores = [
                scores['acceleration']['score'] / 50.0 * 100.0,
                scores['skidpad']['score'] / 50.0 * 100.0,
                scores['autox']['score'] / 100.0 * 100.0,
                scores['endurance']['score'] / 250.0 * 100.0,
                scores['efficiency']['score'] / 75.0 * 100.0,
            ]
            N_cat = len(categories)
            angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
            angles += angles[:1]
            pct_scores += pct_scores[:1]
            
            self.axes_score_radar.clear()
            self.axes_score_radar.plot(angles, pct_scores, 'o-', color='#0984e3', linewidth=2.0)
            self.axes_score_radar.fill(angles, pct_scores, color='#0984e3', alpha=0.25)
            self.axes_score_radar.set_xticks(angles[:-1])
            self.axes_score_radar.set_xticklabels(categories, fontsize=8.5, fontweight='bold')
            self.axes_score_radar.set_ylim(0, 100)
            self.axes_score_radar.set_title(f"1. Efficacia % Discipline Dinamiche\n(Totale: {tot_pts:.1f} / 525 pt)", fontsize=10, fontweight='bold')
            
            # --- 2. BAR CHART: TIME COMPARISONS ---
            self.axes_score_bars.clear()
            disc_names = ['Accel [s]', 'Skidpad [s]', 'AutoX [s]', 'Endur/20 [s]']
            t_veh = [scores['acceleration']['time'], scores['skidpad']['time'], scores['autox']['time'], scores['endurance']['time'] / 20.0]
            t_min = [scores['acceleration']['t_min'], scores['skidpad']['t_min'], scores['autox']['t_min'], scores['endurance']['t_min'] / 20.0]
            t_max = [scores['acceleration']['t_max'], scores['skidpad']['t_max'], scores['autox']['t_max'], scores['endurance']['t_max'] / 20.0]
            
            x_b = np.arange(len(disc_names))
            w = 0.28
            self.axes_score_bars.bar(x_b - w, t_min, width=w, color='#00b894', alpha=0.85, label='Benchmark T_min (100% Punti)')
            self.axes_score_bars.bar(x_b, t_veh, width=w, color='#0984e3', alpha=0.85, label='Tempo Simulato Vettura')
            self.axes_score_bars.bar(x_b + w, t_max, width=w, color='#e17055', alpha=0.85, label='Soglia Max T_max (0 Punti)')
            
            for i in range(len(disc_names)):
                self.axes_score_bars.text(x_b[i], t_veh[i] + 0.3, f"{t_veh[i]:.2f}s", ha='center', fontsize=7.5, fontweight='bold')
                
            self.axes_score_bars.set_xticks(x_b)
            self.axes_score_bars.set_xticklabels(disc_names, fontsize=8.5, fontweight='bold')
            self.axes_score_bars.set_ylabel("Tempo [s]", fontsize=8.5)
            self.axes_score_bars.set_title("2. Confronto Tempi: Vettura vs T_min & T_max", fontsize=10, fontweight='bold')
            self.axes_score_bars.grid(True, linestyle=':', alpha=0.6)
            self.axes_score_bars.legend(fontsize=7.5, loc='upper left')
            
            self.fig_scores.tight_layout()
            self.canvas_scores.draw()
            self.log(f"[FSAE Scoring] Punteggi calcolati con successo: Totale = {tot_pts:.1f} / 525.0 pt.")
        except Exception as e:
            import traceback
            self.log(f"Errore calcolo punteggi FSAE: {traceback.format_exc()}")
        finally:
            self.btn_calc_fsae_scores.setEnabled(True)
            self.btn_calc_fsae_scores.setText(" Calcola Punteggi Ufficiali FS 2026")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    setup_global_exception_handler()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

