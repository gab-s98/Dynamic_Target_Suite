import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from PyQt5.QtWidgets import QApplication
from src.ui import main_gui

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_gui_full_lifecycle(qapp):
    """Systematically tests all MainWindow tabs, callbacks, models, and plotting routines."""
    w = main_gui.MainWindow()
    assert w is not None

    # 1. Test model switching and dynamic parameter panel rebuild
    for model_idx in [0, 1, 2]: # 3-DOF, 7-DOF, 14-DOF
        w.model_combo.setCurrentIndex(model_idx)
        w._rebuild_params_panel()
        assert len(w.bounds_inputs) > 0

    # 2. Test Dynamic Events (Skidpad, Accel, Step, DLC, AutoX, Endurance)
    w.run_skidpad_event()
    assert hasattr(w, 'last_skidpad_res') and w.last_skidpad_res['success'] is True

    w.run_accel_event()
    assert hasattr(w, 'last_accel_res') and w.last_accel_res['success'] is True

    w.run_step_steer_event()
    assert hasattr(w, 'last_step_res') and w.last_step_res['success'] is True

    w.run_dlc_event()
    assert hasattr(w, 'last_step_res') and w.last_step_res['success'] is True

    # Test AutoX across 3-DOF, 7-DOF, 14-DOF
    for model_idx in [0, 1, 2]:
        w.model_combo.setCurrentIndex(model_idx)
        w.autox_event_combo.setCurrentIndex(0) # AutoX
        w.run_autox_endurance_event()
        assert hasattr(w, 'last_autox_res') and w.last_autox_res['success'] is True

    # Test Endurance
    w.autox_event_combo.setCurrentIndex(1) # Endurance
    w.run_autox_endurance_event()
    assert hasattr(w, 'last_autox_res') and w.last_autox_res['success'] is True

    # 3. Test Setup Tools (Tips #8, #29, #82, #87, #89)
    w.plot_steering_torque()
    w.calculate_and_plot_suspension_sizing()
    w.plot_4_ratios_radar()

    # 4. Test Constant Parameters evaluation across models
    for model_idx in [0, 1, 2]:
        w.model_combo.setCurrentIndex(model_idx)
        w.run_constant_parameters()
        assert hasattr(w, 'last_optimized_params')

    # 5. Test Parameter view filter mode (Active Model vs ALL)
    w.combo_param_view.setCurrentIndex(1) # ALL
    assert len(w.bounds_inputs) > 10
    w.combo_param_view.setCurrentIndex(0) # Active Model

    # 6. Test Single Optimization Run on active models
    w.model_combo.setCurrentIndex(0) # 3-DOF
    w.run_single_optimization([1.529, 1.252, 1.136, 0.45, 0.50, 0.485],
                              [(1.525, 1.8), (1.0, 1.5), (1.0, 1.5), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65)],
                              penalty=0.0, tir_file=w.tir_combo.currentText(), model_type="3-DOF")
    assert hasattr(w, 'last_optimized_params')
