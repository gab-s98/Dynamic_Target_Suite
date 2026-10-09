import openmdao.api as om
import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
import casadi as ca

def _resolve_tir_path(tir_file=None):
    return TireParams._resolve_path(tir_file)



def load_vehicle(tir_file=None):
    vp = VehicleParams()
    tir_path = _resolve_tir_path(tir_file)
    tp = TireParams(tir_path)
    return Vehicle3DOF(vehicle_params=vp, tire_params_f=tp, tire_params_r=tp)


def generate_handling_sweep_classic(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0,
                                     num_points=60, ay_max=None, L=None, t_f=None, t_r=None,
                                     tir_file=None, K_phi_f=None, K_phi_r=None, q1=None, q2=None,
                                     camber_f=None, camber_r=None, toe_f=None, toe_r=None):
    """
    Classic handling diagram generation: steady-state lateral force and moment balance
    with physical Lateral Load Transfer (LLT) from roll stiffnesses (K_phi_f, K_phi_r) & roll centers (q1, q2),
    camber thrust, toe kinematics, and non-linear tire load sensitivity.
    """
    import os
    from src.models.tire_model_mf61 import mf61_lateral_force

    tir_path = _resolve_tir_path(tir_file)
    tp = TireParams(tir_path)
    vp = VehicleParams()

    m  = vp.m
    g  = vp.g
    L_val = L if L is not None else vp.L
    tf_val = t_f if t_f is not None else vp.t_f
    tr_val = t_r if t_r is not None else vp.t_r

    lr = wd_f * L_val          # distance CG -> rear axle
    lf = (1.0 - wd_f) * L_val  # distance CG -> front axle

    # Roll kinematics if physical roll stiffnesses are provided
    has_roll_stiffness = (K_phi_f is not None and K_phi_r is not None and (K_phi_f + K_phi_r) > 0)
    if has_roll_stiffness:
        k_tot = float(K_phi_f + K_phi_r)
        q1_val = q1 if q1 is not None else getattr(vp, 'q1', 0.046)
        q2_val = q2 if q2 is not None else getattr(vp, 'q2', 0.077)
        h_ra = q1_val + (q2_val - q1_val) * (1.0 - wd_f)
        h_arm = max(vp.h_cg - h_ra, 0.02)
        m_s = m * (1.0 - getattr(vp, 'unsprung_mass_pct', 0.12))
        m_f = m * wd_f
        m_r = m * (1.0 - wd_f)

    # Aerodynamic downforce
    Cz  = getattr(vp, 'Cz', 3.0)
    rho = getattr(vp, 'rho', 1.225)
    S   = getattr(vp, 'S', 1.0)
    F_down = 0.5 * rho * S * Cz * v_x**2

    # Vertical loads per axle
    Fz_axle_f = m * g * (lr / L_val) + F_down * aero_bal
    Fz_axle_r = m * g * (lf / L_val) + F_down * (1.0 - aero_bal)

    # Check calibrated TYRESIDE from .tir property files
    is_right = getattr(tp, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if tp else True

    # Static camber and toe angles [rad]
    cam_f_val = camber_f if camber_f is not None else getattr(vp, 'camber_f_static', -1.5)
    cam_r_val = camber_r if camber_r is not None else getattr(vp, 'camber_r_static', -1.0)
    toe_f_val = toe_f if toe_f is not None else getattr(vp, 'toe_f_static', 0.0)
    toe_r_val = toe_r if toe_r is not None else getattr(vp, 'toe_r_static', 0.1)

    if is_right:
        gamma_fl = -np.deg2rad(cam_f_val)
        gamma_fr =  np.deg2rad(cam_f_val)
        gamma_rl = -np.deg2rad(cam_r_val)
        gamma_rr =  np.deg2rad(cam_r_val)
    else:
        gamma_fl =  np.deg2rad(cam_f_val)
        gamma_fr = -np.deg2rad(cam_f_val)
        gamma_rl =  np.deg2rad(cam_r_val)
        gamma_rr = -np.deg2rad(cam_r_val)

    toe_f_rad = np.deg2rad(toe_f_val)
    toe_r_rad = np.deg2rad(toe_r_val)

    lvy_fl = -1.0 if is_right else 1.0
    lvy_fr =  1.0 if is_right else -1.0
    lvy_rl = -1.0 if is_right else 1.0
    lvy_rr =  1.0 if is_right else -1.0

    lhy_fl = -1.0 if is_right else 1.0
    lhy_fr =  1.0 if is_right else -1.0
    lhy_rl = -1.0 if is_right else 1.0
    lhy_rr =  1.0 if is_right else -1.0

    results = []
    ay_limit = ay_max if ay_max is not None else 3.0
    ay_values = np.linspace(0.02, ay_limit, num_points)

    for ay_g in ay_values:
        ay = ay_g * g
        if has_roll_stiffness:
            phi = (m_s * ay * h_arm) / max(k_tot - m_s * g * h_arm, 10.0)
            dFz_f = (K_phi_f * phi + m_f * ay * q1_val) / tf_val
            dFz_r = (K_phi_r * phi + m_r * ay * q2_val) / tr_val
        else:
            total_lat_moment = m * ay * vp.h_cg
            dFz_f = (total_lat_moment * lltd) / tf_val
            dFz_r = (total_lat_moment * (1.0 - lltd)) / tr_val

        fz_fl = max(Fz_axle_f / 2.0 - dFz_f, 10.0)
        fz_fr = max(Fz_axle_f / 2.0 + dFz_f, 10.0)
        fz_rl = max(Fz_axle_r / 2.0 - dFz_r, 10.0)
        fz_rr = max(Fz_axle_r / 2.0 + dFz_r, 10.0)

        Fy_f_req = m * ay * (lr / L_val)
        Fy_r_req = m * ay * (lf / L_val)

        def axle_f_obj(a_deg):
            a_rad = np.deg2rad(a_deg)
            a_fl = a_rad + toe_f_rad
            a_fr = a_rad - toe_f_rad
            fy1 = -(float(mf61_lateral_force(a_fl, fz_fl, gamma_fl, tp, LMUY=lmuy_mult, LVY=lvy_fl, LHY=lhy_fl)) - float(mf61_lateral_force(toe_f_rad, fz_fl, gamma_fl, tp, LMUY=lmuy_mult, LVY=lvy_fl, LHY=lhy_fl)))
            fy2 = -(float(mf61_lateral_force(a_fr, fz_fr, gamma_fr, tp, LMUY=lmuy_mult, LVY=lvy_fr, LHY=lhy_fr)) - float(mf61_lateral_force(-toe_f_rad, fz_fr, gamma_fr, tp, LMUY=lmuy_mult, LVY=lvy_fr, LHY=lhy_fr)))
            return (fy1 + fy2) - Fy_f_req

        def axle_r_obj(a_deg):
            a_rad = np.deg2rad(a_deg)
            a_rl = a_rad + toe_r_rad
            a_rr = a_rad - toe_r_rad
            fy1 = -(float(mf61_lateral_force(a_rl, fz_rl, gamma_rl, tp, LMUY=lmuy_mult, LVY=lvy_rl, LHY=lhy_rl)) - float(mf61_lateral_force(toe_r_rad, fz_rl, gamma_rl, tp, LMUY=lmuy_mult, LVY=lvy_rl, LHY=lhy_rl)))
            fy2 = -(float(mf61_lateral_force(a_rr, fz_rr, gamma_rr, tp, LMUY=lmuy_mult, LVY=lvy_rr, LHY=lhy_rr)) - float(mf61_lateral_force(-toe_r_rad, fz_rr, gamma_rr, tp, LMUY=lmuy_mult, LVY=lvy_rr, LHY=lhy_rr)))
            return (fy1 + fy2) - Fy_r_req

        a_peak = 18.0
        fy_max_f = axle_f_obj(a_peak) + Fy_f_req
        fy_max_r = axle_r_obj(a_peak) + Fy_r_req

        if Fy_f_req > fy_max_f or Fy_r_req > fy_max_r:
            break

        try:
            alpha_f_deg = opt.brentq(axle_f_obj, 0.0, a_peak, xtol=1e-4)
            alpha_r_deg = opt.brentq(axle_r_obj, 0.0, a_peak, xtol=1e-4)
        except ValueError:
            break

        L_R_rad = (ay * L_val) / (v_x**2)
        L_R_deg = np.rad2deg(L_R_rad)
        delta_deg = L_R_deg + alpha_f_deg - alpha_r_deg

        if delta_deg > 25.0:
            break

        results.append({
            'ay_g': ay_g,
            'alpha_f_deg': alpha_f_deg,
            'alpha_r_deg': alpha_r_deg,
            'slip_diff_deg': alpha_f_deg - alpha_r_deg,
            'L_R': L_R_rad,
            'L_R_deg': L_R_deg,
            'delta_deg': delta_deg,
            'is_stable': True
        })

    return pd.DataFrame(results)

def generate_handling_sweep_arclength(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0,
                                      stop_at_peak=False, num_points=70, L=None, t_f=None, t_r=None, tir_file=None,
                                      K_phi_f=None, K_phi_r=None, q1=None, q2=None,
                                      camber_f=None, camber_r=None, toe_f=None, toe_r=None):
    """
    Traces the non-linear steady-state handling curve using Keller's Pseudo-Arc-Length continuation method.
    Solves the augmented 3x3 equilibrium system:
      F_lat(beta, r, delta) = 0
      M_yaw(beta, r, delta) = 0
      T_k^T (X - X_k) - ds = 0
    This eliminates Jacobian singularities at limit points in ay and turning points in delta (countersteering),
    allowing smooth continuation past peak grip into post-saturation, or cleanly stopping at the exact peak.
    """
    tir_path = _resolve_tir_path(tir_file)
    tp = TireParams(tir_path)
    vehicle = Vehicle3DOF(VehicleParams(), tp, tp)
    
    if camber_f is not None: vehicle.vp.camber_f_static = float(camber_f)
    if camber_r is not None: vehicle.vp.camber_r_static = float(camber_r)
    if toe_f is not None: vehicle.vp.toe_f_static = float(toe_f)
    if toe_r is not None: vehicle.vp.toe_r_static = float(toe_r)

    # Calculate derived physical LLTD if roll stiffnesses are provided
    lltd_use = lltd
    if K_phi_f is not None and K_phi_r is not None and (K_phi_f + K_phi_r) > 0:
        vehicle.vp.K_phi_f = float(K_phi_f)
        vehicle.vp.K_phi_r = float(K_phi_r)
        if q1 is not None: vehicle.vp.q1 = float(q1)
        if q2 is not None: vehicle.vp.q2 = float(q2)
        vehicle.vp.wd_f = float(wd_f)
        if L is not None: vehicle.vp.L = float(L)
        if t_f is not None: vehicle.vp.t_f = float(t_f)
        if t_r is not None: vehicle.vp.t_r = float(t_r)
        lltd_use = vehicle.vp.get_effective_lltd(ay_g=1.0)
    
    funcs = vehicle.get_global_casadi_functions()
    res_func = funcs['res_func']
    jac_states_func = funcs['jac_states_func']
    
    L_val = L if L is not None else vehicle.vp.L
    tf_val = t_f if t_f is not None else vehicle.vp.t_f
    tr_val = t_r if t_r is not None else vehicle.vp.t_r
    a = L_val * (1.0 - wd_f)
    b = L_val * wd_f
    
    # Step 0: Initial state at origin
    X0 = np.array([0.0, 0.0, 0.0]) # beta [rad], r [rad/s], delta [rad]
    
    # Step 1: Small perturbation in delta to establish initial tangent vector
    d1 = np.deg2rad(0.25)
    r1_guess = (v_x / L_val) * d1
    def obj_step1(x):
        return np.array(res_func([v_x, d1], x, [wd_f, lltd_use, lmuy_mult, L_val, tf_val, tr_val, aero_bal])).flatten()
    sol1 = opt.root(obj_step1, [0.0, r1_guess], method='hybr')
    if not sol1.success:
        return pd.DataFrame()
    X1 = np.array([sol1.x[0], sol1.x[1], d1])
    
    # Arc-length step size calibrated to num_points for smooth, defined curves
    ds = 2.8 / max(num_points, 40)
    max_steps = int(num_points * 2.5)
    
    T_k = (X1 - X0) / np.linalg.norm(X1 - X0)
    X_k = X1.copy()
    
    results = []
    
    # Add initial points
    for pt in [X0, X1]:
        b_val, r_val, d_val = pt
        ay_val = v_x * r_val
        v_y = v_x * np.tan(b_val)
        alpha_f = d_val - np.arctan2(v_y + a * r_val, v_x)
        alpha_r = -np.arctan2(v_y - b * r_val, v_x)
        L_R = (L_val * r_val) / v_x
        results.append({
            'ay_g': float(ay_val / 9.81),
            'delta_deg': float(np.rad2deg(d_val)),
            'sideslip_deg': float(np.rad2deg(b_val)),
            'alpha_f_deg': float(np.rad2deg(alpha_f)),
            'alpha_r_deg': float(np.rad2deg(alpha_r)),
            'slip_diff_deg': float(np.rad2deg(alpha_f - alpha_r)),
            'L_R': float(L_R),
            'L_R_deg': float(np.rad2deg(L_R)),
            'yaw_rate': float(r_val),
            'is_stable': True
        })
        
    peak_detected = False
    max_ay_g = results[-1]['ay_g']
    
    for step in range(max_steps):
        # Predictor step: Euler step along tangent
        X_pred = X_k + ds * T_k
        
        # Corrector step: solve augmented system F(X) = 0, T_k^T (X - X_k) - ds = 0
        def augmented_system(X):
            b_val, r_val, d_val = X
            res = res_func([v_x, d_val], [b_val, r_val], [wd_f, lltd_use, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
            eq1_2 = np.array(res).flatten()
            eq3 = np.dot(T_k, X - X_k) - ds
            return np.array([eq1_2[0], eq1_2[1], eq3])
            
        sol = opt.root(augmented_system, X_pred, method='hybr')
        if not sol.success:
            # Fallback with smaller step if solver diverges
            ds *= 0.5
            if ds < 1e-4:
                break
            continue
            
        # Extract corrected state
        b_val, r_val, d_val = sol.x
        ay_val = v_x * r_val
        ay_g = ay_val / 9.81
        
        # Check termination conditions (max steering angle 35 deg, excessive sideslip, or extreme post-peak)
        if abs(np.rad2deg(d_val)) > 35.0 or abs(np.rad2deg(b_val)) > 25.0:
            break
            
        # Detect peak lateral acceleration
        if ay_g > max_ay_g:
            max_ay_g = ay_g
        elif ay_g < max_ay_g - 0.005:
            peak_detected = True
            if stop_at_peak:
                break
                
        # If continuing past peak, terminate if ay drops significantly below peak (>20% drop)
        if peak_detected and ay_g < max_ay_g * 0.80:
            break
            
        v_y = v_x * np.tan(b_val)
        alpha_f = d_val - np.arctan2(v_y + a * r_val, v_x)
        alpha_r = -np.arctan2(v_y - b * r_val, v_x)
        L_R = (L_val * r_val) / v_x
        
        results.append({
            'ay_g': float(ay_g),
            'delta_deg': float(np.rad2deg(d_val)),
            'sideslip_deg': float(np.rad2deg(b_val)),
            'alpha_f_deg': float(np.rad2deg(alpha_f)),
            'alpha_r_deg': float(np.rad2deg(alpha_r)),
            'slip_diff_deg': float(np.rad2deg(alpha_f - alpha_r)),
            'L_R': float(L_R),
            'L_R_deg': float(np.rad2deg(L_R)),
            'yaw_rate': float(r_val),
            'is_stable': not peak_detected
        })
        
        # Update tangent vector for next continuation step
        new_T = (sol.x - X_k) / np.linalg.norm(sol.x - X_k)
        # Ensure direction consistency (dot product > 0)
        if np.dot(new_T, T_k) > 0.0:
            T_k = new_T
            
        X_k = sol.x.copy()
        
    return pd.DataFrame(results)


def generate_handling_sweep(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0, 
                            delta_max_deg=None, num_points=60, L=None, t_f=None, t_r=None, tir_file=None,
                            method='arclength', stop_at_peak=False,
                            K_phi_f=None, K_phi_r=None, q1=None, q2=None,
                            camber_f=None, camber_r=None, toe_f=None, toe_r=None):
    """
    Traces the non-linear steady-state handling curve.
    Methods:
      - 'arclength': Pseudo-Arc-Length continuation (Keller's method) - robust through peak grip
      - 'ay_sweep': Lateral acceleration continuation
    
    If delta_max_deg is None, it is derived from VehicleParams.road_wheel_max_deg
    (steering_wheel_max_deg / steering_ratio).
    """
    # Resolve delta_max_deg from steering ratio if not provided
    if delta_max_deg is None:
        delta_max_deg = VehicleParams().road_wheel_max_deg

    if method == 'arclength':
        return generate_handling_sweep_arclength(
            v_x=v_x, wd_f=wd_f, lltd=lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
            stop_at_peak=stop_at_peak, num_points=num_points, L=L, t_f=t_f, t_r=t_r, tir_file=tir_file,
            K_phi_f=K_phi_f, K_phi_r=K_phi_r, q1=q1, q2=q2,
            camber_f=camber_f, camber_r=camber_r, toe_f=toe_f, toe_r=toe_r
        )
        
    tir_path = _resolve_tir_path(tir_file)
    tp = TireParams(tir_path)
    vehicle = Vehicle3DOF(VehicleParams(), tp, tp)
    
    funcs = vehicle.get_global_casadi_functions()
    res_func = funcs['res_func']
    jac_states_func = funcs['jac_states_func']
    
    L_val = L if L is not None else vehicle.vp.L
    tf_val = t_f if t_f is not None else vehicle.vp.t_f
    tr_val = t_r if t_r is not None else vehicle.vp.t_r
    
    results = []
    beta_guess = 0.0
    delta_guess = 0.0
    max_ay_found = 0.0
    
    for ay_g in np.linspace(0.02, 3.2, num_points):
        ay = ay_g * 9.81
        r_val = ay / v_x
        
        def obj(x):
            b, d = x
            res = res_func([v_x, d], [b, r_val], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
            return np.array(res).flatten()
            
        sol = opt.root(obj, [beta_guess, delta_guess], method='hybr', options={'xtol': 1e-7})
        if not sol.success:
            sol = opt.root(obj, [beta_guess, delta_guess], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
            
        res_norm = np.linalg.norm(obj(sol.x))
        b_sol, d_sol = sol.x
        
        if not sol.success or res_norm > 1.0 or abs(np.rad2deg(b_sol)) > 25.0:
            break
            
        if stop_at_peak and max_ay_found > 0.3 and ay_g < max_ay_found - 0.01:
            break
        max_ay_found = max(max_ay_found, ay_g)
            
        beta_guess = b_sol
        delta_guess = d_sol
        
        v_y = v_x * np.tan(b_sol)
        a = L_val * (1.0 - wd_f)
        b = L_val * wd_f
        alpha_f = d_sol - np.arctan2(v_y + a * r_val, v_x)
        alpha_r = - np.arctan2(v_y - b * r_val, v_x)
        L_R = (L_val * r_val) / v_x
        
        # Stability Analysis (Lyapunov linearisation)
        jac_states = jac_states_func([v_x, d_sol], [b_sol, r_val], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
        m_val = vehicle.vp.m
        Iz_val = vehicle.vp.Iz
        A_mat = np.array(jac_states)
        A_mat[0, :] = A_mat[0, :] / (m_val * v_x)
        A_mat[1, :] = A_mat[1, :] / Iz_val
        eigvals = np.linalg.eigvals(A_mat)
        max_real_part = np.max(np.real(eigvals))
        is_stable = bool(max_real_part < 0)
        
        results.append({
            'delta_deg': float(np.rad2deg(d_sol)),
            'sideslip_deg': float(np.rad2deg(b_sol)),
            'yaw_rate': float(r_val),
            'ay_g': float(ay_g),
            'alpha_f_deg': float(np.rad2deg(alpha_f)),
            'alpha_r_deg': float(np.rad2deg(alpha_r)),
            'slip_diff_deg': float(np.rad2deg(alpha_f - alpha_r)),
            'L_R': float(L_R),
            'L_R_deg': float(np.rad2deg(L_R)),
            'max_eig_real': float(max_real_part),
            'is_stable': is_stable
        })
        
    return pd.DataFrame(results)


_cached_vehicle_funcs = {}

def _get_worker_casadi_funcs(tir_path):
    global _cached_vehicle_funcs
    if tir_path not in _cached_vehicle_funcs:
        from src.models.vehicle_3dof import Vehicle3DOF
        from src.models.params_vehicle import VehicleParams
        from src.models.params_tire import TireParams
        tp = TireParams(tir_path)
        veh = Vehicle3DOF(VehicleParams(), tp, tp)
        _cached_vehicle_funcs[tir_path] = veh.get_global_casadi_functions()
    return _cached_vehicle_funcs[tir_path]


def _worker_solve_speed_curve(args):
    """
    Worker function executed in parallel across CPU cores.
    Solves full vehicle 3DOF lateral force and yaw moment balance across the entire achievable
    handling manifold using robust pseudo arc-length continuation (Keller's method).
    Smoothly traces from delta = 0 deg through the peak maximum lateral acceleration.
    args: (u, wd_f, lltd, aero_bal, lmuy_mult, L_val, tf_val, tr_val, tir_path, delta_max_deg, n_points, ay_max)
    """
    u, wd_f, lltd, aero_bal, lmuy_mult, L_val, tf_val, tr_val, tir_path, delta_max_deg, n_points, ay_max = args
    funcs = _get_worker_casadi_funcs(tir_path)
    res_func = funcs['res_func']
    jac_states_func = funcs['jac_states_func']
    
    # Step 0: Initial equilibrium state at origin
    X0 = np.array([0.0, 0.0, 0.0]) # beta [rad], r [rad/s], delta [rad]
    
    # Step 1: Small perturbation in delta to establish initial tangent vector
    d1 = np.deg2rad(0.20)
    r1_guess = (u / L_val) * d1
    def obj_step1(x):
        return np.array(res_func([u, d1], x, [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])).flatten()
    sol1 = opt.root(obj_step1, [0.0, r1_guess], method='hybr', options={'xtol': 1e-7})
    if not sol1.success:
        return u, []
    X1 = np.array([sol1.x[0], sol1.x[1], d1])
    
    num_pts = max(int(n_points), 40)
    ds = 2.4 / num_pts
    T_k = (X1 - X0) / np.linalg.norm(X1 - X0)
    X_k = X1.copy()
    
    curve_u = []
    
    # Add initial points
    for pt in [X0, X1]:
        b_val, r_val, d_val = pt
        ay_g = float((u * r_val) / 9.81)
        rho = float(r_val / u)
        curve_u.append({
            'u': float(u),
            'delta_deg': float(np.rad2deg(d_val)),
            'beta_deg': float(np.rad2deg(b_val)),
            'r': float(r_val),
            'rho': float(rho),
            'ay_g': float(ay_g),
            'is_stable': True
        })
        
    max_ay = curve_u[-1]['ay_g']
    peak_detected = False
    
    for _ in range(num_pts * 3):
        # Predictor step: Euler step along tangent in (beta, r, delta) space
        X_pred = X_k + ds * T_k
        
        # Corrector step: solve augmented system [res(X) == 0; T_k^T (X - X_k) - ds == 0]
        def augmented_sys(X):
            b_v, r_v, d_v = X
            res = res_func([u, d_v], [b_v, r_v], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
            eq1_2 = np.array(res).flatten()
            eq3 = np.dot(T_k, X - X_k) - ds
            return np.array([eq1_2[0], eq1_2[1], eq3])
            
        sol = opt.root(augmented_sys, X_pred, method='hybr', options={'xtol': 1e-7})
        if not sol.success:
            ds *= 0.5
            if ds < 1e-4:
                break
            continue
            
        b_sol, r_sol, d_sol = sol.x
        ay_g = float((u * r_sol) / 9.81)
        rho = float(r_sol / u)
        
        # Check geometric and physical limits (steering lock 35 deg, excessive drift)
        if abs(np.rad2deg(d_sol)) > float(delta_max_deg) or abs(np.rad2deg(b_sol)) > 25.0:
            break
            
        # Detect maximum achievable lateral acceleration
        if ay_g > max_ay:
            max_ay = ay_g
        elif ay_g < max_ay - 0.015:
            peak_detected = True
            # Stop just past peak to form clean handling envelope
            curve_u.append({
                'u': float(u),
                'delta_deg': float(np.rad2deg(d_sol)),
                'beta_deg': float(np.rad2deg(b_sol)),
                'r': float(r_sol),
                'rho': float(rho),
                'ay_g': float(ay_g),
                'is_stable': False
            })
            break
            
        # Stability Analysis (Lyapunov linearisation)
        jac_states = jac_states_func([u, d_sol], [b_sol, r_sol], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
        A_mat = np.array(jac_states)
        A_mat[0, :] = A_mat[0, :] / (292.0 * u)
        A_mat[1, :] = A_mat[1, :] / 120.0
        eigvals = np.linalg.eigvals(A_mat)
        is_stable = bool(np.max(np.real(eigvals)) < 0)
        
        curve_u.append({
            'u': float(u),
            'delta_deg': float(np.rad2deg(d_sol)),
            'beta_deg': float(np.rad2deg(b_sol)),
            'r': float(r_sol),
            'rho': float(rho),
            'ay_g': float(ay_g),
            'is_stable': is_stable
        })
        
        # Update tangent vector
        new_T = (sol.x - X_k) / np.linalg.norm(sol.x - X_k)
        if np.dot(new_T, T_k) > 0.0:
            T_k = new_T
        X_k = sol.x.copy()
        
    return u, curve_u


def generate_complete_maps(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0, 
                           L=None, t_f=None, t_r=None, tir_file=None, 
                           speeds_list=None, deltas_list=None, ay_levels=None, rho_levels=None,
                           v_min=4.0, v_max=30.0, n_speeds=50, n_points=70, ay_max=3.5):
    """
    Generates the complete, densely-sampled suite of Map of Achievable Performance (MAPs)
    leveraging full multi-core CPU parallelism.
    1. Input Achievable Regions:
       - Plane (delta, u): Y=u [m/s], X=delta [deg] with 12 dense iso-ay, 10 iso-rho and closed limit envelope.
       - Plane (u, ay): Y=ay [g], X=u [m/s] with 14 const-delta curves, const-rho parabolas, and max-ay envelope.
       - Plane (delta, ay): Y=ay [g], X=delta [deg] with 12 const-u handling curves and dense iso-beta grid.
    2. Output Achievable Regions:
       - Plane (rho, beta): Y=beta [deg], X=rho [1/m] with const-u, const-delta and const-ay curves.
    3. Mixed Input/Output Regions:
       - Plane (delta, rho): Y=rho [1/m], X=delta [deg] with const-u, const-ay and const-beta curves.
    """
    tir_path = _resolve_tir_path(tir_file)
    vehicle = load_vehicle(tir_file=tir_path)
    funcs = vehicle.get_global_casadi_functions()
    res_func = funcs['res_func']
    
    L_val = L if L is not None else vehicle.vp.L
    tf_val = t_f if t_f is not None else vehicle.vp.t_f
    tr_val = t_r if t_r is not None else vehicle.vp.t_r
    
    # Dense parameter grids
    if speeds_list is None:
        speeds_list = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    if deltas_list is None:
        deltas_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0]
    if ay_levels is None:
        ay_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    if rho_levels is None:
        rho_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]
        
    dense_speeds = np.linspace(v_min, v_max, n_speeds)
    all_speeds = np.unique(np.sort(np.concatenate([dense_speeds, speeds_list, [v_x]])))
    
    # Parallel execution across CPU cores
    worker_args = [
        (u, wd_f, lltd, aero_bal, lmuy_mult, L_val, tf_val, tr_val, tir_path, 25.0, n_points, ay_max)
        for u in all_speeds
    ]
    
    speed_curves = {}
    envelope_pts = []
    
    num_cpus = os.cpu_count() or 4
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            raw_results = list(executor.map(_worker_solve_speed_curve, worker_args))
    except Exception:
        raw_results = [_worker_solve_speed_curve(arg) for arg in worker_args]
        
    for u, curve_u in raw_results:
        if curve_u:
            df_u = pd.DataFrame(curve_u)
            speed_curves[u] = df_u
            idx_max = df_u['ay_g'].idxmax()
            envelope_pts.append({
                'u': float(u),
                'max_delta_deg': float(df_u.loc[idx_max, 'delta_deg']),
                'max_ay_g': float(df_u.loc[idx_max, 'ay_g']),
                'max_rho': float(df_u.loc[idx_max, 'rho']),
                'max_beta_deg': float(df_u.loc[idx_max, 'beta_deg'])
            })
            
    envelope_df = pd.DataFrame(envelope_pts).sort_values('u').reset_index(drop=True)
            
    # 2. Delta Slices (delta = const across u) solved directly
    delta_slices = {}
    for d_t in deltas_list:
        pts = []
        d_rad = np.deg2rad(d_t)
        b_guess, r_guess = 0.0, 0.0
        for u in sorted(speed_curves.keys()):
            def obj_delta(x):
                b, r = x
                res = res_func([u, d_rad], [b, r], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
                return np.array(res).flatten()
            sol = opt.root(obj_delta, [b_guess, r_guess], method='hybr', options={'xtol': 1e-6})
            if not sol.success:
                sol = opt.root(obj_delta, [b_guess, r_guess], method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
            if sol.success and np.linalg.norm(obj_delta(sol.x)) < 1.0:
                b_sol, r_sol = sol.x
                ay_g = (u * r_sol) / 9.81
                rho = r_sol / u
                if abs(np.rad2deg(b_sol)) <= 25.0 and ay_g <= 3.5:
                    pts.append({'u': u, 'delta_deg': d_t, 'beta_deg': float(np.rad2deg(b_sol)), 'rho': float(rho), 'ay_g': float(ay_g)})
                    b_guess, r_guess = b_sol, r_sol
        if pts:
            delta_slices[d_t] = pd.DataFrame(pts)
            
    # 3. Iso-Ay Curves (ay = const across u, strictly pre-peak monotonic)
    iso_ay_curves = {}
    for ay_t in ay_levels:
        pts = []
        for u in sorted(speed_curves.keys()):
            df_u = speed_curves[u]
            df_pre = df_u.iloc[:df_u['ay_g'].idxmax() + 1]
            if df_pre['ay_g'].max() >= ay_t and df_pre['ay_g'].min() <= ay_t:
                d_val = float(np.interp(ay_t, df_pre['ay_g'], df_pre['delta_deg']))
                b_val = float(np.interp(ay_t, df_pre['ay_g'], df_pre['beta_deg']))
                rho_val = float(np.interp(ay_t, df_pre['ay_g'], df_pre['rho']))
                pts.append({'u': u, 'delta_deg': d_val, 'beta_deg': b_val, 'rho': rho_val, 'ay_g': ay_t})
        if pts:
            iso_ay_curves[ay_t] = pd.DataFrame(pts)
            
    # 4. Iso-Rho Curves (rho = const across u, strictly pre-peak monotonic)
    iso_rho_curves = {}
    for rho_t in rho_levels:
        pts = []
        for u in sorted(speed_curves.keys()):
            df_u = speed_curves[u]
            df_pre = df_u.iloc[:df_u['rho'].idxmax() + 1]
            if df_pre['rho'].max() >= rho_t and df_pre['rho'].min() <= rho_t:
                d_val = float(np.interp(rho_t, df_pre['rho'], df_pre['delta_deg']))
                b_val = float(np.interp(rho_t, df_pre['rho'], df_pre['beta_deg']))
                ay_val = float(np.interp(rho_t, df_pre['rho'], df_pre['ay_g']))
                pts.append({'u': u, 'delta_deg': d_val, 'beta_deg': b_val, 'rho': rho_t, 'ay_g': ay_val})
        if pts:
            iso_rho_curves[rho_t] = pd.DataFrame(pts)
            
    # 5. Dense Iso-Beta Curves at Target Speed v_x (yaw moment equilibrium)
    betas_deg = np.linspace(-8.0, 8.0, 17)
    deltas_deg = np.linspace(-20.0, 20.0, 41)
    iso_beta_curves = []
    for b_deg in betas_deg:
        b_rad = np.deg2rad(b_deg)
        curve_pts = []
        r_guess = 0.0
        for d_deg in deltas_deg:
            d_rad = np.deg2rad(d_deg)
            def obj_yaw(r_val):
                r_scalar = float(np.atleast_1d(r_val)[0])
                res = res_func([v_x, d_rad], [b_rad, r_scalar], [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal])
                return float(res[1])
            sol_r = opt.root(obj_yaw, r_guess, method='hybr')
            if sol_r.success:
                r_sol = float(sol_r.x[0])
                ay_g = (v_x * r_sol) / 9.81
                rho = r_sol / v_x
                if abs(ay_g) < 3.2:
                    curve_pts.append({'u': v_x, 'delta_deg': d_deg, 'beta_deg': b_deg, 'r': r_sol, 'rho': rho, 'ay_g': ay_g})
                    r_guess = r_sol
        if curve_pts:
            iso_beta_curves.append(pd.DataFrame(curve_pts))
            
    envelope_df = pd.DataFrame(envelope_pts)
    
    return {
        'speed_curves': speed_curves,
        'delta_slices': delta_slices,
        'iso_ay_curves': iso_ay_curves,
        'iso_rho_curves': iso_rho_curves,
        'iso_beta_curves': iso_beta_curves,
        'envelope_df': envelope_df,
        'speeds_list': speeds_list,
        'deltas_list': deltas_list,
        'ay_levels': ay_levels,
        'rho_levels': rho_levels,
        'v_x': v_x
    }


def generate_achievable_performance_map(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0, 
                                        L=None, t_f=None, t_r=None, tir_file=None, speeds_list=None):
    """
    Backwards-compatible wrapper returning (iso_beta_curves, multi_speed_envelopes).
    """
    if speeds_list is None:
        speeds_list = [10.0, 15.0, 20.0, 25.0]
    maps_data = generate_complete_maps(
        v_x=v_x, wd_f=wd_f, lltd=lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
        L=L, t_f=t_f, t_r=t_r, tir_file=tir_file, speeds_list=speeds_list
    )
    multi_speed_envelopes = {spd: maps_data['speed_curves'].get(spd, pd.DataFrame()) for spd in speeds_list}
    return maps_data['iso_beta_curves'], multi_speed_envelopes


# Backwards compatibility aliases
generate_complete_guiggiani_maps = generate_complete_maps
generate_guiggiani_map = generate_achievable_performance_map


def generate_mmm_diagram(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0, 
                         L=None, t_f=None, t_r=None, delta_deg_list=None, 
                         beta_range_deg=(-10.0, 10.0), num_beta=21, tir_file=None,
                         model_type="3-DOF", vehicle_params=None,
                         camber_f_static=None, camber_r_static=None,
                         toe_f_static=None, toe_r_static=None,
                         K_phi_f=None, K_phi_r=None, K_z_f=None, K_z_r=None):
    """
    Generates high-definition Moment Method / Yaw Moment Diagram (YMD) data
    across all available vehicle dynamics models (3-DOF, 7-DOF, 14-DOF).
    
    For each (delta, beta) at speed v_x:
      - Enforces lateral kinematic sideslip vy = vx * tan(beta).
      - Solves equilibrium state (yaw rate r, rolling wheel speeds, roll phi, suspensions)
        such that the lateral force balance (res_lat = 0) and model kinematic/elastic
        constraints are met.
      - Extracts resultant lateral acceleration ay = (vx * r) / g and unbalanced yaw moment Mz.
    """
    if delta_deg_list is None:
        delta_deg_list = [-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0]
        
    tir_path = _resolve_tir_path(tir_file)
    tp = TireParams(tir_path)
    vp = vehicle_params or VehicleParams()
    
    L_val = float(L if L is not None else vp.L)
    tf_val = float(t_f if t_f is not None else vp.t_f)
    tr_val = float(t_r if t_r is not None else vp.t_r)
    
    cam_f_val = float(camber_f_static if camber_f_static is not None else getattr(vp, 'camber_f_static', -1.5))
    cam_r_val = float(camber_r_static if camber_r_static is not None else getattr(vp, 'camber_r_static', -1.0))
    toe_f_val = float(toe_f_static if toe_f_static is not None else getattr(vp, 'toe_f_static', 0.0))
    toe_r_val = float(toe_r_static if toe_r_static is not None else getattr(vp, 'toe_r_static', 0.1))
    
    kphi_f_val = float(K_phi_f if K_phi_f is not None else getattr(vp, 'K_phi_f', 8000.0))
    kphi_r_val = float(K_phi_r if K_phi_r is not None else getattr(vp, 'K_phi_r', 6000.0))
    kz_f_val = float(K_z_f if K_z_f is not None else getattr(vp, 'K_z_f', 18000.0))
    kz_r_val = float(K_z_r if K_z_r is not None else getattr(vp, 'K_z_r', 16000.0))
    
    model_str = str(model_type).upper()
    
    if "14" in model_str:
        veh_model = Vehicle14DOF(vp, tp, tp)
        trim_funcs = veh_model.get_trim_casadi_functions()
        res_func = trim_funcs['res_func']
        par_vec = [wd_f, lltd, kphi_f_val, kphi_r_val, kz_f_val, kz_r_val, aero_bal,
                   cam_f_val, cam_r_val, toe_f_val, toe_r_val, L_val, tf_val, tr_val, lmuy_mult]
        active_tier = 14
    elif "7" in model_str:
        veh_model = Vehicle7DOF(vp, tp, tp)
        trim_funcs = veh_model.get_trim_casadi_functions()
        res_func = trim_funcs['res_func']
        par_vec = [wd_f, lltd, aero_bal, cam_f_val, cam_r_val, toe_f_val, toe_r_val, L_val, tf_val, tr_val, lmuy_mult]
        active_tier = 7
    else:
        veh_model = Vehicle3DOF(vp, tp, tp)
        trim_funcs = veh_model.get_global_casadi_functions()
        res_func = trim_funcs['res_func']
        par_vec = [wd_f, lltd, lmuy_mult, L_val, tf_val, tr_val, aero_bal]
        active_tier = 3

    Rw = vp.R_w
    results = []
    betas_rad = np.deg2rad(np.linspace(beta_range_deg[0], beta_range_deg[1], num_beta))
    
    for delta_deg in delta_deg_list:
        delta_rad = np.deg2rad(delta_deg)
        r_prev = (v_x / L_val) * delta_rad
        
        # State caches for warm starting across beta sweep
        w_guess = v_x / Rw
        phi_guess = 0.0
        z_guess = [0.0, 0.0, 0.0, 0.0]
        
        for beta in betas_rad:
            v_y = v_x * np.tan(beta)
            r_guess = (v_x / L_val) * (delta_rad - 0.3 * beta)
            if abs(r_prev) > 1e-4:
                r_guess = 0.5 * r_guess + 0.5 * r_prev
                
            if active_tier == 3:
                def obj_3dof(r_val):
                    r_scalar = float(np.atleast_1d(r_val)[0])
                    res = res_func([v_x, delta_rad], [beta, r_scalar], par_vec)
                    return float(res[0])
                    
                sol = opt.root(obj_3dof, [r_guess], method='hybr', options={'xtol': 1e-7})
                if not sol.success or abs(obj_3dof(sol.x)) > 2.0:
                    sol = opt.root(obj_3dof, [r_guess], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                if not sol.success or abs(obj_3dof(sol.x)) > 2.0:
                    sol = opt.root(obj_3dof, [0.0], method='hybr', options={'xtol': 1e-7})
                if not sol.success or abs(obj_3dof(sol.x)) > 2.0:
                    sol = opt.root(obj_3dof, [0.0], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                    
                if sol.success and abs(obj_3dof(sol.x)) < 10.0:
                    r_sol = float(sol.x[0])
                    res = res_func([v_x, delta_rad], [beta, r_sol], par_vec)
                    unbalanced_yaw_moment = -float(res[1])
                    ay_g = (v_x * r_sol) / 9.81
                    
                    if abs(ay_g) <= 4.0:
                        r_prev = r_sol
                        results.append({
                            'delta_deg': float(delta_deg),
                            'beta_deg': float(np.rad2deg(beta)),
                            'yaw_rate': float(r_sol),
                            'ay_g': float(ay_g),
                            'yaw_moment': float(unbalanced_yaw_moment)
                        })

            elif active_tier == 7:
                guess7 = [r_guess, w_guess, w_guess, w_guess, w_guess]
                def obj_7dof(x_unk):
                    r_s, w1, w2, w3, w4 = x_unk
                    res = res_func([v_x, delta_rad], [v_y, r_s, w1, w2, w3, w4], par_vec)
                    arr = np.array(res).flatten()
                    # Balance lat force and 4-wheel pure rolling spin residuals
                    return np.array([arr[0], arr[2], arr[3], arr[4], arr[5]])
                    
                sol = opt.root(obj_7dof, guess7, method='hybr', options={'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_7dof(sol.x)) > 2.0:
                    sol = opt.root(obj_7dof, guess7, method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_7dof(sol.x)) > 2.0:
                    sol = opt.root(obj_7dof, [0.0, v_x/Rw, v_x/Rw, v_x/Rw, v_x/Rw], method='hybr', options={'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_7dof(sol.x)) > 2.0:
                    sol = opt.root(obj_7dof, [0.0, v_x/Rw, v_x/Rw, v_x/Rw, v_x/Rw], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                    
                if sol.success and np.linalg.norm(obj_7dof(sol.x)) < 15.0:
                    r_sol = float(sol.x[0])
                    res_full = res_func([v_x, delta_rad], [v_y, r_sol, sol.x[1], sol.x[2], sol.x[3], sol.x[4]], par_vec)
                    unbalanced_yaw_moment = -float(np.array(res_full).flatten()[1])
                    ay_g = (v_x * r_sol) / 9.81
                    
                    if abs(ay_g) <= 4.0:
                        r_prev = r_sol
                        w_guess = float(sol.x[1])
                        results.append({
                            'delta_deg': float(delta_deg),
                            'beta_deg': float(np.rad2deg(beta)),
                            'yaw_rate': float(r_sol),
                            'ay_g': float(ay_g),
                            'yaw_moment': float(unbalanced_yaw_moment)
                        })

            elif active_tier == 14:
                guess14 = [r_guess, phi_guess, z_guess[0], z_guess[1], z_guess[2], z_guess[3],
                           w_guess, w_guess, w_guess, w_guess]
                def obj_14dof(x_unk):
                    r_s, phi_s, z1, z2, z3, z4, w1, w2, w3, w4 = x_unk
                    res = res_func([v_x, delta_rad], [v_y, r_s, phi_s, z1, z2, z3, z4, w1, w2, w3, w4], par_vec)
                    arr = np.array(res).flatten()
                    # Lateral force balance (0), roll equilibrium (2), 4 suspension travels (3,4,5,6), 4 wheel spins (7,8,9,10)
                    return np.array([arr[0], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10]])
                    
                sol = opt.root(obj_14dof, guess14, method='hybr', options={'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_14dof(sol.x)) > 3.0:
                    sol = opt.root(obj_14dof, guess14, method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_14dof(sol.x)) > 3.0:
                    neutral_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v_x/Rw, v_x/Rw, v_x/Rw, v_x/Rw]
                    sol = opt.root(obj_14dof, neutral_guess, method='hybr', options={'xtol': 1e-7})
                if not sol.success or np.linalg.norm(obj_14dof(sol.x)) > 3.0:
                    neutral_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v_x/Rw, v_x/Rw, v_x/Rw, v_x/Rw]
                    sol = opt.root(obj_14dof, neutral_guess, method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                    
                if sol.success and np.linalg.norm(obj_14dof(sol.x)) < 25.0:
                    r_sol = float(sol.x[0])
                    st_full = [v_y, r_sol] + list(sol.x[1:])
                    res_full = res_func([v_x, delta_rad], st_full, par_vec)
                    unbalanced_yaw_moment = -float(np.array(res_full).flatten()[1])
                    ay_g = (v_x * r_sol) / 9.81
                    
                    if abs(ay_g) <= 4.0:
                        r_prev = r_sol
                        phi_guess = float(sol.x[1])
                        z_guess = [float(sol.x[2]), float(sol.x[3]), float(sol.x[4]), float(sol.x[5])]
                        w_guess = float(sol.x[6])
                        results.append({
                            'delta_deg': float(delta_deg),
                            'beta_deg': float(np.rad2deg(beta)),
                            'yaw_rate': float(r_sol),
                            'ay_g': float(ay_g),
                            'yaw_moment': float(unbalanced_yaw_moment)
                        })
                    
    return pd.DataFrame(results)

def generate_multispeed_mmm_envelopes(
    speeds_list=(8.0, 12.0, 16.0, 20.0, 25.0),
    wd_f=0.5, lltd=0.5, aero_bal=0.48, lmuy_mult=1.0,
    L=None, t_f=None, t_r=None,
    delta_max_deg=12.0, num_beta=21, tir_file=None,
    model_type="3-DOF", vehicle_params=None,
    camber_f_static=None, camber_r_static=None,
    toe_f_static=None, toe_r_static=None,
    K_phi_f=None, K_phi_r=None, K_z_f=None, K_z_r=None
):
    """
    Generates multi-speed boundary envelopes and stability/control derivatives across velocities
    for the specified vehicle model type.
    """
    results_by_speed = {}
    derivatives_list = []
    
    # High-density steer angle grid with fine resolution in the linear regime (|delta| <= 2 deg)
    d_max = float(delta_max_deg)
    deltas = sorted(list(set([
        -d_max, -d_max * 0.65, -d_max * 0.35, -2.0, -1.0, 0.0, 1.0, 2.0, d_max * 0.35, d_max * 0.65, d_max
    ])))

    for vx in speeds_list:
        df_spd = generate_mmm_diagram(
            v_x=vx, wd_f=wd_f, lltd=lltd, aero_bal=aero_bal, lmuy_mult=lmuy_mult,
            L=L, t_f=t_f, t_r=t_r, delta_deg_list=deltas,
            beta_range_deg=(-10.0, 10.0), num_beta=max(num_beta, 31), tir_file=tir_file,
            model_type=model_type, vehicle_params=vehicle_params,
            camber_f_static=camber_f_static, camber_r_static=camber_r_static,
            toe_f_static=toe_f_static, toe_r_static=toe_r_static,
            K_phi_f=K_phi_f, K_phi_r=K_phi_r, K_z_f=K_z_f, K_z_r=K_z_r
        )
        results_by_speed[vx] = df_spd
        
        if not df_spd.empty:
            sub_d0 = df_spd[np.isclose(df_spd['delta_deg'], 0.0, atol=1e-2)].sort_values('beta_deg')
            sub_b0 = df_spd[np.isclose(df_spd['beta_deg'], 0.0, atol=1e-2)].sort_values('delta_deg')
            
            n_beta = 0.0
            if len(sub_d0) >= 3:
                b_arr = sub_d0['beta_deg'].values
                mz_arr = sub_d0['yaw_moment'].values
                mask = np.abs(b_arr) <= 3.5
                if np.sum(mask) >= 2:
                    n_beta = float(np.polyfit(b_arr[mask], mz_arr[mask], 1)[0])
                else:
                    n_beta = float(np.polyfit(b_arr, mz_arr, 1)[0])
                    
            n_delta = 0.0
            if len(sub_b0) >= 3:
                d_arr = sub_b0['delta_deg'].values
                mz_d = sub_b0['yaw_moment'].values
                # Velocity-adaptive linear window: at high speed the front tire
                # saturates earlier, so the linear regime in delta is narrower.
                # 3.5° at 10 m/s -> ~1.4° at 25 m/s; minimum 1.0° always kept.
                delta_lin_lim = max(1.0, 3.5 * (10.0 / max(vx, 10.0)))
                mask_d = np.abs(d_arr) <= delta_lin_lim
                if np.sum(mask_d) >= 2:
                    n_delta = float(np.polyfit(d_arr[mask_d], mz_d[mask_d], 1)[0])
                else:
                    n_delta = float(np.polyfit(d_arr, mz_d, 1)[0])

                    
            derivatives_list.append({
                'v_x': float(vx),
                'N_delta': float(n_delta),
                'N_beta': float(n_beta),
                'control_margin': float(abs(n_delta) / max(abs(n_beta), 1e-3)),
                'max_ay_g': float(df_spd['ay_g'].max()),
                'max_yaw_moment': float(df_spd['yaw_moment'].max())
            })
            
    return {
        'speed_envelopes': results_by_speed,
        'derivatives_df': pd.DataFrame(derivatives_list)
    }

def generate_limit_handling_surface(v_x_range=(10, 30), num_vx=10, wd_f_range=(0.4, 0.6), num_wdf=10, lltd=0.5, lmuy_mult=1.0):
    """
    Sweeps over Speeds and Weight Distributions to find the maximum Ay.
    Returns a DataFrame representing a 3D surface.
    """
    v_xs = np.linspace(v_x_range[0], v_x_range[1], num_vx)
    wd_fs = np.linspace(wd_f_range[0], wd_f_range[1], num_wdf)
    
    results = []
    
    for vx in v_xs:
        for wdf in wd_fs:
            df_sweep = generate_handling_sweep(vx, wdf, lltd, lmuy_mult, delta_max_deg=10.0, num_points=30)
            if not df_sweep.empty:
                max_ay = df_sweep['ay_g'].max()
                # Find the delta that caused max Ay
                best_row = df_sweep.loc[df_sweep['ay_g'].idxmax()]
                
                results.append({
                    'v_x': vx,
                    'wd_f': wdf,
                    'max_ay_g': max_ay,
                    'delta_at_max': best_row['delta_deg']
                })
            else:
                results.append({
                    'v_x': vx,
                    'wd_f': wdf,
                    'max_ay_g': 0.0,
                    'delta_at_max': 0.0
                })
                
    return pd.DataFrame(results)

def calculate_understeer_gradient(df_handling, L, v_x):
    """
    Calculates Understeer Gradient K [deg/g] from the handling sweep DataFrame.
    K = d(delta) / d(ay)  in the linear region (ay < 0.3g).
    
    The Ackermann reference is computed from L_R if available, or from ay directly.
    Works with both classic and dynamic sweep DataFrames.
    """
    if df_handling.empty or len(df_handling) < 3:
        return 0.0

    # Linear region (low ay)
    df_lin = df_handling[df_handling['ay_g'] < 0.3]
    if len(df_lin) < 2:
        df_lin = df_handling.head(3)

    ay = df_lin['ay_g'].values
    delta_deg = df_lin['delta_deg'].values

    # Ackermann angle: delta_ack = L/R [rad] → deg
    # L/R = ay * g * L / v_x^2 (quasi-static), or use L_R column if present
    if 'L_R' in df_lin.columns:
        delta_ack_deg = np.rad2deg(df_lin['L_R'].values)
    elif 'yaw_rate' in df_lin.columns:
        r_rad = df_lin['yaw_rate'].values
        delta_ack_deg = np.rad2deg((L * r_rad) / v_x)
    else:
        delta_ack_deg = np.rad2deg(ay * 9.81 * L / v_x**2)

    # Handling parameter = deviation from Ackermann
    handling_param = delta_deg - delta_ack_deg

    if len(ay) > 1:
        slope, _ = np.polyfit(ay, handling_param, 1)
        return slope
    return 0.0


def calculate_control_margin(df_mmm):
    """
    Extracts the control margin from the MMM diagram.
    This is defined as the max unbalanced yaw moment at delta=0 before instability.
    """
    if df_mmm.empty:
        return 0.0
    
    # Look at delta=0 curve
    df_zero_steer = df_mmm[df_mmm['delta_deg'] == 0]
    if not df_zero_steer.empty:
        return df_zero_steer['yaw_moment'].abs().max()
    return 0.0
