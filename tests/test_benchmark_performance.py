import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.utils.vehicle_dna_maps import compute_vehicle_dna_fields
from src.mdo.mdo_dynamic_events import simulate_step_steer


def get_fixture_params():
    vp = VehicleParams()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    tp = TireParams(tir_path)
    return vp, tp, tir_path


@pytest.mark.benchmark
def test_benchmark_14dof_trim_residual_speed():
    """Benchmark: CasADi 14-DOF steady-state trim residual evaluation must be < 5.0 ms per call.
    Ensures symbolic graph evaluation is compiled/optimized and doesn't regress.
    """
    vp, tp, _ = get_fixture_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    funcs = veh14.get_trim_casadi_functions()
    res_func = funcs['res_func']

    inputs = [15.0, np.deg2rad(3.0)]
    w0 = 15.0 / vp.R_w
    states = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, w0, w0, w0, w0]
    params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r, vp.aero_balance,
              vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
              vp.L, vp.t_f, vp.t_r, vp.LMUY]

    # Warmup
    res_func(inputs, states, params)

    N_runs = 50
    t0 = time.perf_counter()
    for _ in range(N_runs):
        res_func(inputs, states, params)
    dt_ms = (time.perf_counter() - t0) / N_runs * 1000.0

    assert dt_ms < 5.0, f"14-DOF trim residual evaluation too slow: {dt_ms:.3f} ms (expected < 5.0 ms)"


@pytest.mark.benchmark
def test_benchmark_vehicle_dna_maps_speed():
    """Benchmark: Vehicle DNA Maps computation on 2D grid must complete in < 250 ms.
    Guarantees responsive UI updates when dragging sliders in the DNA tab.
    """
    _, _, tir_path = get_fixture_params()

    t0 = time.perf_counter()
    res = compute_vehicle_dna_fields(v_x=15.0, tir_file=tir_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert res is not None
    assert dt_ms < 250.0, f"Vehicle DNA maps calculation too slow: {dt_ms:.1f} ms (expected < 250.0 ms)"


@pytest.mark.benchmark
def test_benchmark_14dof_step_steer_speed():
    """Benchmark: 2.0s transient step-steer integration of 14-DOF full vehicle must be < 2000 ms.
    Ensures numerical integrator step size and CasADi ODE call efficiency remain viable for interactive use.
    """
    vp, _, tir_path = get_fixture_params()

    t0 = time.perf_counter()
    res = simulate_step_steer(vp, v_x=15.0, steer_deg=3.0, t_end=2.0, model_type='14-DOF', tir_path=tir_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert res['success'] is True
    assert dt_ms < 2000.0, f"14-DOF Step Steer transient integration too slow: {dt_ms:.1f} ms (expected < 2000.0 ms)"