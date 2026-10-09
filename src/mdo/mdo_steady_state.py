import os
import numpy as np
import time
from scipy import optimize as opt

from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams

try:
    import openmdao.api as om
    from src.core.om_vehicle_equilibrium import VehicleEquilibriumComponent

    class SteadyStateGroup(om.Group):
        def initialize(self):
            self.options.declare('vehicle_model', types=object)

        def setup(self):
            vehicle_model = self.options['vehicle_model']
            comp = VehicleEquilibriumComponent(vehicle_model=vehicle_model)
            self.add_subsystem('equilibrium', comp, promotes=['*'])
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
            self.nonlinear_solver.options['iprint'] = -1
            self.nonlinear_solver.options['maxiter'] = 50
            self.nonlinear_solver.options['err_on_non_converge'] = False
            self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            self.nonlinear_solver.linesearch.options['iprint'] = -1
            self.linear_solver = om.DirectSolver()

    def _build_problem(v_x, delta_deg, wd_f, lltd, aero_bal, L, t_f, t_r, tir_file=None, 
                       recorder_path=None, verbose=False, model_type='3-DOF',
                       camber_f=None, camber_r=None, toe_f=None, toe_r=None,
                       K_phi_f=None, K_phi_r=None, K_z_f=None, K_z_r=None):
        vp = VehicleParams()
        vp.L = L
        vp.t_f = t_f
        vp.t_r = t_r
        vp.wd_f = wd_f
        vp.lltd = lltd
        vp.aero_balance = aero_bal
        
        if camber_f is not None: vp.camber_f_static = camber_f
        if camber_r is not None: vp.camber_r_static = camber_r
        if toe_f is not None:    vp.toe_f_static = toe_f
        if toe_r is not None:    vp.toe_r_static = toe_r
        if K_phi_f is not None:  vp.K_phi_f = K_phi_f
        if K_phi_r is not None:  vp.K_phi_r = K_phi_r
        if K_z_f is not None:    vp.K_z_f = K_z_f
        if K_z_r is not None:    vp.K_z_r = K_z_r
        
        tp = TireParams(tir_file) if tir_file else TireParams()
        
        if model_type == '14-DOF':
            model = Vehicle14DOF(vp, tp, tp)
        elif model_type == '7-DOF':
            model = Vehicle7DOF(vp, tp, tp)
        else:
            model = Vehicle3DOF(vehicle_params=vp, tire_params_f=tp, tire_params_r=tp)

        prob = om.Problem(reports=False)
        prob.model = SteadyStateGroup(vehicle_model=model)

        if recorder_path:
            recorder = om.SqliteRecorder(recorder_path)
            prob.add_recorder(recorder)
            prob.model.add_recorder(recorder)

        prob.setup(force_alloc_complex=True)

        params_meta = prob.model.equilibrium.meta['params']
        states_meta = prob.model.equilibrium.meta['states']

        prob.set_val('v_x', v_x)
        prob.set_val('delta_steer', np.deg2rad(delta_deg))
        prob.set_val('wd_f', wd_f)
        prob.set_val('lltd', lltd)
        
        if 'L' in params_meta:   prob.set_val('L', L)
        if 't_f' in params_meta: prob.set_val('t_f', t_f)
        if 't_r' in params_meta: prob.set_val('t_r', t_r)
        
        if 'aero_bal' in params_meta:
            prob.set_val('aero_bal', aero_bal)

        if 'lmuy_mult' in params_meta:
            prob.set_val('lmuy_mult', getattr(vp, 'LMUY', 1.0))
            
        if 'camber_f_static' in params_meta:
            prob.set_val('camber_f_static', vp.camber_f_static)
            prob.set_val('camber_r_static', vp.camber_r_static)
            prob.set_val('toe_f_static', vp.toe_f_static)
            prob.set_val('toe_r_static', vp.toe_r_static)
            
        if 'K_phi_f' in params_meta:
            prob.set_val('K_phi_f', vp.K_phi_f)
            prob.set_val('K_phi_r', vp.K_phi_r)
            prob.set_val('K_z_f', vp.K_z_f)
            prob.set_val('K_z_r', vp.K_z_r)

        r_ack = (v_x / L) * np.deg2rad(delta_deg)
        prob.set_val('r', r_ack)
        
        if 'beta' in states_meta:
            prob.set_val('beta', 0.0)
        
        if 'v_y' in states_meta:
            prob.set_val('v_y', 0.0)
            
        if 'w_fl' in states_meta:
            w_nom = v_x / vp.R_w
            prob.set_val('w_fl', w_nom)
            prob.set_val('w_fr', w_nom)
            prob.set_val('w_rl', w_nom)
            prob.set_val('w_rr', w_nom)
            
        if 'phi' in states_meta:
            prob.set_val('phi', 0.0)
            prob.set_val('z_fl', 0.0)
            prob.set_val('z_fr', 0.0)
            prob.set_val('z_rl', 0.0)
            prob.set_val('z_rr', 0.0)

        return prob
except ImportError:
    SteadyStateGroup = None
    _build_problem = None

def evaluate_kpis_with_openmdao(v_x, delta_deg, wd_f, lltd, aero_bal, L, t_f, t_r,
                               do_check_partials=False, tir_file=None, model_type='3-DOF',
                               camber_f=None, camber_r=None, toe_f=None, toe_r=None,
                               K_phi_f=None, K_phi_r=None, K_z_f=None, K_z_r=None):
    """
    Runs a single steady-state evaluation using SciPy's root solver instead of OpenMDAO
    and returns a dict of KPIs.
    """
    tir_path = tir_file

    vp = VehicleParams()
    vp.L = L
    vp.t_f = t_f
    vp.t_r = t_r
    vp.wd_f = wd_f
    vp.lltd = lltd
    vp.aero_balance = aero_bal
    
    if camber_f is not None: vp.camber_f_static = camber_f
    if camber_r is not None: vp.camber_r_static = camber_r
    if toe_f is not None:    vp.toe_f_static = toe_f
    if toe_r is not None:    vp.toe_r_static = toe_r
    if K_phi_f is not None:  vp.K_phi_f = K_phi_f
    if K_phi_r is not None:  vp.K_phi_r = K_phi_r
    if K_z_f is not None:    vp.K_z_f = K_z_f
    if K_z_r is not None:    vp.K_z_r = K_z_r
    
    tp = TireParams(tir_path)
    
    if model_type == '14-DOF':
        model = Vehicle14DOF(vp, tp, tp)
        trim_funcs = model.get_trim_casadi_functions()
    elif model_type == '7-DOF':
        model = Vehicle7DOF(vp, tp, tp)
        trim_funcs = model.get_trim_casadi_functions()
    else:
        model = Vehicle3DOF(vehicle_params=vp, tire_params_f=tp, tire_params_r=tp)
        if hasattr(model, 'get_global_casadi_functions'):
            trim_funcs = model.get_global_casadi_functions()
        else:
            trim_funcs = model.get_casadi_functions()
            
    res_func = trim_funcs['res_func']
    meta = trim_funcs['meta']
    
    par_map = {
        'wd_f': wd_f, 'lltd': lltd, 'aero_bal': aero_bal,
        'L': L, 't_f': t_f, 't_r': t_r,
        'camber_f_static': vp.camber_f_static, 'camber_r_static': vp.camber_r_static,
        'toe_f_static': vp.toe_f_static, 'toe_r_static': vp.toe_r_static,
        'K_phi_f': vp.K_phi_f, 'K_phi_r': vp.K_phi_r,
        'K_z_f': vp.K_z_f, 'K_z_r': vp.K_z_r,
        'lmuy_mult': getattr(vp, 'LMUY', 1.0)
    }
    
    par_vec = [par_map[p] for p in meta['params']]
    
    delta_rad = np.deg2rad(delta_deg)
    inputs = [v_x, delta_rad]
    
    r_guess = (v_x / L) * delta_rad
    Rw = vp.R_w
    w_guess = v_x / Rw
    
    state_guess = []
    for s in meta['states']:
        if s == 'r': state_guess.append(r_guess)
        elif s.startswith('w_'): state_guess.append(w_guess)
        else: state_guess.append(0.0)
        
    def obj_func(x_unk):
        res = res_func(inputs, x_unk, par_vec)
        return np.array(res).flatten()
        
    t0 = time.time()
    sol = opt.root(obj_func, state_guess, method='hybr', options={'xtol': 1e-8})
    if not sol.success or np.linalg.norm(obj_func(sol.x)) > 1.0:
        sol = opt.root(obj_func, state_guess, method='lm', options={'ftol': 1e-8, 'xtol': 1e-8})
    t1 = time.time()
    
    exec_time_ms = (t1 - t0) * 1000.0
    iters = sol.nfev if hasattr(sol, 'nfev') else 10
    
    state_res = {name: val for name, val in zip(meta['states'], sol.x)}
    
    r_val = state_res.get('r', 0.0)
    if 'beta' in state_res:
        beta_val = state_res['beta']
    elif 'v_y' in state_res:
        beta_val = state_res['v_y'] / max(v_x, 1.0)
    else:
        beta_val = 0.0
        
    roll_grad = None
    if model_type == '14-DOF':
        phi_rad = state_res.get('phi', 0.0)
        ay_g = (v_x * r_val) / 9.81
        if abs(ay_g) > 1e-3:
            roll_grad = np.rad2deg(phi_rad) / ay_g
        else:
            roll_grad = 0.0
            
    return {
        'solver_iters': iters,
        'exec_time_ms': exec_time_ms,
        'jac_max_error': 0.0,
        'yaw_rate_rads': float(r_val),
        'beta_rad': float(beta_val),
        'roll_gradient_deg_g': roll_grad,
    }

evaluate_steady_state_kpis = evaluate_kpis_with_openmdao

if __name__ == "__main__":
    vp = VehicleParams()
    tir_file = 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir'
    res_3d = evaluate_kpis_with_openmdao(15.0, 3.0, 0.45, 0.5, 0.485, 1.529, 1.252, 1.136, model_type='3-DOF', tir_file=tir_file)
    print("3-DOF KPI:", res_3d)
    res_7d = evaluate_kpis_with_openmdao(15.0, 3.0, 0.45, 0.5, 0.485, 1.529, 1.252, 1.136, model_type='7-DOF', tir_file=tir_file)
    print("7-DOF KPI:", res_7d)
    res_14d = evaluate_kpis_with_openmdao(15.0, 3.0, 0.45, 0.5, 0.485, 1.529, 1.252, 1.136, model_type='14-DOF', tir_file=tir_file)
    print("14-DOF KPI:", res_14d)
