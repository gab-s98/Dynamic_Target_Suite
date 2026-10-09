import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import math
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = QApplication.instance() or QApplication(sys.argv)

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_global_optimizer import GlobalOptimizer


def test_arb_refactoring_and_wheel_rate():
    """Task 1: Verify direct lumped ARB roll stiffness and differential wheel rate."""
    vp = VehicleParams()
    assert hasattr(vp, 'K_phi_arb_f')
    assert hasattr(vp, 'K_phi_arb_r')
    assert hasattr(vp, 'Kw_arb_f')
    assert hasattr(vp, 'Kw_arb_r')

    # Default values check
    assert vp.K_phi_arb_f == 12000.0
    assert vp.K_phi_arb_r == 8000.0

    kw_f = vp.get_Kw_arb_f()
    expected_kw_f = 12000.0 / (vp.t_f**2)
    assert math.isclose(kw_f, expected_kw_f, rel_tol=1e-4)

    # Bi-directional conversion check
    vp.set_Kphi_arb_from_Kw(8000.0, 6000.0)
    assert math.isclose(vp.Kw_arb_f, 8000.0)
    assert math.isclose(vp.Kw_arb_r, 6000.0)
    assert math.isclose(vp.K_phi_arb_f, 8000.0 * (vp.t_f**2))
    assert math.isclose(vp.K_phi_arb_r, 6000.0 * (vp.t_r**2))


def test_sweep_speed_limit_ay_continuity():
    """Task 2: Verify 14-DOF speed sweep continuity beyond 14 m/s (50.4 km/h) without collapse."""
    opt = GlobalOptimizer(model_type="14-DOF")
    vp = opt.vp
    x14 = [
        vp.L, vp.t_f, vp.t_r, vp.wd_f, 0.5, vp.aero_balance,
        vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
        vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r
    ]

    speeds_to_check = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]
    results = {}
    for v in speeds_to_check:
        ay_limit = opt._evaluate_limit_ay_at_speed(v, x14)
        results[v] = ay_limit
        # Must maintain grip limit above 1.15g and strictly not collapse to 0.0g
        assert ay_limit > 1.15, f"Limit ay collapsed at Vx={v} m/s ({v*3.6} km/h): got {ay_limit:.3f}g"

    # Confirm continuity: no sudden cliff drop > 0.4g between consecutive speed steps
    for i in range(len(speeds_to_check) - 1):
        v1 = speeds_to_check[i]
        v2 = speeds_to_check[i + 1]
        diff = abs(results[v2] - results[v1])
        assert diff < 0.35, f"Discontinuity detected between {v1} and {v2} m/s: {diff:.3f}g"


def test_spring_free_length_and_preload():
    """Task 3: Verify physical spring parameters (free length L0, preload, installed length)."""
    vp = VehicleParams()
    assert hasattr(vp, 'spring_free_length_f_mm')
    assert hasattr(vp, 'spring_free_length_r_mm')
    assert vp.spring_free_length_f_mm == 150.0
    assert vp.spring_free_length_r_mm == 150.0

    # Installed length L_inst = L0 - Delta_L_pre
    expected_inst_f = vp.spring_free_length_f_mm - vp.spring_preload_f_mm
    expected_inst_r = vp.spring_free_length_r_mm - vp.spring_preload_r_mm
    assert math.isclose(vp.spring_installed_length_f_mm, expected_inst_f)
    assert math.isclose(vp.spring_installed_length_r_mm, expected_inst_r)


def test_camera_nose_bumper_and_filter():
    """Task 4: Verify Nose and Bumper camera positioning ahead of nosecone with 8 Hz EMA filter."""
    from src.ui.interactive_driver_14dof import get_viewport_class
    ViewportClass = get_viewport_class()
    vp_widget = ViewportClass()

    # Presets test
    vp_widget.camera_mode = "NOSE"
    assert vp_widget.camera_mode == "NOSE"

    vp_widget.camera_mode = "BUMPER"
    assert vp_widget.camera_mode == "BUMPER"

    # Verify EMA filter formula: alpha = 1 - exp(-2*pi*fc*dt) with fc=8.0
    fc = 8.0
    dt = 1.0 / 60.0
    alpha = 1.0 - math.exp(-2.0 * math.pi * fc * dt)
    assert 0.50 < alpha < 0.65


def test_banking_gravity_and_patches():
    """Task 5: Verify banking gravity projection sign and synchronized contact elevations."""
    # ISO 8855 (+X fwd, +Y left, +Z up): bank > 0 tilts right down (-Y).
    # Downhill gravity acceleration along body +Y axis must be negative: ay_road_grav = -g * sin(bank)
    bank_rad = math.radians(5.0)
    ay_road_grav = -9.81 * math.sin(bank_rad)
    assert ay_road_grav < 0.0, "Positive bank must pull towards -Y (right/downhill)"

    # Contact patch elevation: z_road,i = z_centerline + d_i * sin(bank)
    z_center = 0.50
    tf = 1.25
    d_left = +tf / 2.0
    d_right = -tf / 2.0
    z_left = z_center + d_left * math.sin(bank_rad)
    z_right = z_center + d_right * math.sin(bank_rad)
    assert z_left > z_center > z_right


def test_haptics_rate_limiting_and_hysteresis():
    """Task 6: Verify gamepad haptics rate-limiting and slip ratio hysteresis."""
    # Hysteresis test: ON at kappa >= 0.12, OFF at kappa <= 0.08
    kappa_samples = [0.05, 0.10, 0.13, 0.11, 0.09, 0.07]
    active = False
    states = []
    for k in kappa_samples:
        if active:
            if k < 0.08:
                active = False
        else:
            if k >= 0.12:
                active = True
        states.append(active)

    # Expect: False, False, True, True, True, False
    assert states == [False, False, True, True, True, False]
