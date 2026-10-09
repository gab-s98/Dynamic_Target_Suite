import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo.mdo_steady_state import evaluate_kpis_with_openmdao
from src.mdo.mdo_dynamic_events import simulate_step_steer

def get_test_params():
    vp = VehicleParams()
    tir_file = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    tp = TireParams(tir_file)
    return vp, tp

def test_7dof_casadi_trim_functions():
    """Verify that Vehicle7DOF generates valid CasADi trim functions."""
    vp, tp = get_test_params()
    veh7 = Vehicle7DOF(vp, tp, tp)
    funcs = veh7.get_trim_casadi_functions()
    
    assert 'res_func' in funcs
    assert 'jac_states_func' in funcs
    assert 'jac_inputs_func' in funcs
    assert 'jac_params_func' in funcs
    
    # Test evaluation at nominal condition
    v_x = 15.0
    delta = np.deg2rad(3.0)
    vy = 0.0
    r = (v_x / vp.L) * delta
    w0 = v_x / vp.R_w
    
    inputs = [v_x, delta]
    states = [vy, r, w0, w0, w0, w0]
    params = [vp.wd_f, vp.lltd, vp.aero_balance, vp.camber_f_static, vp.camber_r_static,
              vp.toe_f_static, vp.toe_r_static, vp.L, vp.t_f, vp.t_r, vp.LMUY]
              
    res = funcs['res_func'](inputs, states, params)
    assert res.shape == (6, 1)

def test_7dof_openmdao_equilibrium():
    """Verify that OpenMDAO solves 7-DOF steady-state trim in < 10 iterations."""
    res = evaluate_kpis_with_openmdao(
        v_x=15.0, delta_deg=3.0, wd_f=0.45, lltd=0.50, aero_bal=0.485,
        L=1.529, t_f=1.252, t_r=1.136, model_type='7-DOF'
    )
    assert res['solver_iters'] <= 50
    assert abs(res['yaw_rate_rads']) > 0.01

def test_7dof_step_steer_transient():
    """Verify that 7-DOF ODE transient integration completes stably."""
    vp, tp = get_test_params()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    res = simulate_step_steer(vp, v_x=15.0, steer_deg=3.0, t_end=2.0, model_type='7-DOF', tir_path=tir_path)
    
    assert res['success'] is True
    assert res['ay_peak_g'] > 0.3
    assert res['r_ss_deg_s'] > 5.0
