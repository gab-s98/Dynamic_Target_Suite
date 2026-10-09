import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.mdo.mdo_steady_state import evaluate_kpis_with_openmdao

def test_openmdao_analytic_derivatives():
    """
    Validates that the analytical Jacobians match OpenMDAO finite differences.
    """
    res = evaluate_kpis_with_openmdao(
        v_x=15.0, delta_deg=2.0, wd_f=0.45, lltd=0.50, aero_bal=0.485,
        L=1.529, t_f=1.252, t_r=1.136, do_check_partials=True, model_type='3-DOF'
    )
    assert res['solver_iters'] <= 15
    assert res['jac_max_error'] < 1e-3

def test_openmdao_equilibrium_solver():
    """
    Verifies that the OpenMDAO NewtonSolver successfully converges across models.
    """
    for m in ['3-DOF', '7-DOF', '14-DOF']:
        res = evaluate_kpis_with_openmdao(
            v_x=15.0, delta_deg=3.0, wd_f=0.45, lltd=0.50, aero_bal=0.485,
            L=1.529, t_f=1.252, t_r=1.136, model_type=m
        )
        assert res['solver_iters'] <= 50
        assert abs(res['yaw_rate_rads']) > 0.05
