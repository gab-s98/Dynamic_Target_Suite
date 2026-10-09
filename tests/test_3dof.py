import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import casadi as ca
import numpy as np
import scipy.optimize as opt

from src.models.vehicle_3dof import Vehicle3DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import mf61_lateral_force

def get_test_params():
    vp = VehicleParams()
    tir_file = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    tp = TireParams(tir_file)
    return vp, tp

def test_tire_model_zero_slip():
    """Verify that the tire model returns near 0 lateral force at 0 slip angle."""
    alpha = ca.SX.sym('alpha')
    fz = ca.SX.sym('fz')
    _, tp = get_test_params()
    fy = mf61_lateral_force(alpha, fz, 0.0, tp)
    f = ca.Function('f', [alpha, fz], [fy])
    assert float(f(0.0, 1000.0)) == pytest.approx(0.0, abs=50.0)

def test_vehicle_steady_state_equilibrium():
    """Verify that the 3-DOF vehicle equilibrium is well-posed and solvable."""
    vp, tp = get_test_params()
    vehicle = Vehicle3DOF(vehicle_params=vp, tire_params_f=tp, tire_params_r=tp)
    funcs = vehicle.get_global_casadi_functions()
    res_func = funcs['res_func']
    
    v_x = 15.0
    delta_steer = np.deg2rad(3.0)
    
    def obj(x, wd):
        b, r = x
        res = res_func([v_x, delta_steer], [b, r], [wd, 0.5, 1.0, vp.L, vp.t_f, vp.t_r, vp.aero_balance])
        return np.array(res).flatten()
        
    x0 = [0.0, (v_x / vp.L) * delta_steer]
    
    sol1 = opt.root(obj, x0, args=(0.5,), method='hybr')
    assert sol1.success is True
    beta1, r1 = sol1.x
    assert abs(r1) > 0.0
    
    sol2 = opt.root(obj, x0, args=(0.6,), method='hybr')
    assert sol2.success is True
    beta2, r2 = sol2.x
    assert not pytest.approx(r2, abs=1e-5) == r1
