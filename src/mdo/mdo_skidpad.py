import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import mf61_lateral_force

def simulate_skidpad(params: VehicleParams, radius=9.125, use_7dof=False, model_type=None, tir_path=None, ay_steps=None):
    """
    Simulates the official Formula Student Skidpad event (radius = 9.125 m, track width = 3.0 m).
    Finds the maximum sustained cornering velocity v_x that satisfies steady-state dynamic equilibrium
    and strictly respects the track boundaries (corridor constraint: all 4 wheels inside [R-1.5m, R+1.5m]).
    
    Supports '3-DOF', '7-DOF', and '14-DOF' vehicle models.
    """
    tp = TireParams(tir_path)
    
    # Resolve active model
    if model_type is None:
        model_type = '7-DOF' if use_7dof else '3-DOF'
        
    L = params.L
    wd_f = params.wd_f
    t_f = params.t_f
    t_r = params.t_r
    a = L * (1.0 - wd_f)
    b = L * wd_f
    lltd = getattr(params, 'lltd', 0.50)
    aero_bal = params.aero_balance
    lmuy_val = getattr(params, 'LMUY', 1.0)
    
    cam_f = getattr(params, 'camber_f_static', -1.5)
    cam_r = getattr(params, 'camber_r_static', -1.0)
    toe_f = getattr(params, 'toe_f_static', 0.0)
    toe_r = getattr(params, 'toe_r_static', 0.1)
    
    K_phi_f = getattr(params, 'K_phi_f', 8000.0)
    K_phi_r = getattr(params, 'K_phi_r', 6000.0)
    K_z_f   = getattr(params, 'K_z_f', 18000.0)
    K_z_r   = getattr(params, 'K_z_r', 16000.0)
    
    if ay_steps is None:
        ay_grid = np.linspace(0.1, 3.2, 120)
    else:
        ay_grid = np.linspace(0.2, 2.5, int(ay_steps))
    sweep_results = []
    best_sol = None
    v_max = 0.0
    last_failed_ay = 3.2
    
    # Initial kinematic guesses
    v0 = np.sqrt(0.1 * 9.81 * radius)
    vy_guess = (b / radius) * v0
    delta_guess = (L / radius)
    w_guess = v0 / params.R_w
    
    if model_type == '14-DOF':
        veh14 = Vehicle14DOF(params, tp, tp)
        funcs14 = veh14.get_trim_casadi_functions()
        res_func14 = funcs14['res_func']
        jac_states_func14 = funcs14.get('jac_states_func', None)
        jac_inputs_func14 = funcs14.get('jac_inputs_func', None)
        
        # 11 unknowns: vy, delta, phi, z_fl, z_fr, z_rl, z_rr, w_fl, w_fr, w_rl, w_rr
        x_guess = [vy_guess, delta_guess, 0.0, 0.0, 0.0, 0.0, 0.0, w_guess, w_guess, w_guess, w_guess]
        par14 = [wd_f, lltd, K_phi_f, K_phi_r, K_z_f, K_z_r, aero_bal,
                 cam_f, cam_r, toe_f, toe_r, L, t_f, t_r, lmuy_val]
                 
        for ay_g in ay_grid:
            ay = ay_g * 9.81
            v = np.sqrt(ay * radius)
            r = v / radius
            
            def obj_14dof(x):
                vy_val, d_val, phi_val, z1, z2, z3, z4, w1, w2, w3, w4 = x
                states = [vy_val, r, phi_val, z1, z2, z3, z4, w1, w2, w3, w4]
                inputs = [v, d_val]
                res = res_func14(inputs, states, par14)
                return np.array(res).flatten()

            def jac_14dof(x):
                if jac_states_func14 is None or jac_inputs_func14 is None:
                    return None
                vy_val, d_val, phi_val, z1, z2, z3, z4, w1, w2, w3, w4 = x
                states = [vy_val, r, phi_val, z1, z2, z3, z4, w1, w2, w3, w4]
                inputs = [v, d_val]
                J_st = np.array(jac_states_func14(inputs, states, par14))
                J_in = np.array(jac_inputs_func14(inputs, states, par14))
                J = np.empty((11, 11))
                J[:, 0] = J_st[:, 0]
                J[:, 1] = J_in[:, 1]
                J[:, 2:] = J_st[:, 2:]
                return J
                
            jac_call14 = jac_14dof if (jac_states_func14 is not None and jac_inputs_func14 is not None) else None
            sol = opt.root(obj_14dof, x_guess, jac=jac_call14, method='hybr', options={'xtol': 1e-6})
            if not sol.success:
                sol = opt.root(obj_14dof, x_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                
            if not sol.success:
                last_failed_ay = ay_g
                break
                
            res_norm = np.linalg.norm(obj_14dof(sol.x))
            res_tol = max(15.0, 0.05 * params.m * ay)
            if res_norm > res_tol:
                last_failed_ay = ay_g
                break
                
            vy_sol, delta_sol, phi_sol = sol.x[0], sol.x[1], sol.x[2]
            beta_sol = vy_sol / v
            
            if abs(np.rad2deg(beta_sol)) > 25.0 or abs(np.rad2deg(delta_sol)) > 35.0:
                last_failed_ay = ay_g
                break
                
            c_b, s_b = np.cos(beta_sol), np.sin(beta_sol)
            wheels = {
                'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
            }
            radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
            r_min, r_max = min(radii), max(radii)
            corridor_ok = (r_min >= radius - 1.5) and (r_max <= radius + 1.5)
            
            sweep_results.append({
                'v': float(v), 'ay_g': float(ay_g),
                'delta_deg': float(np.rad2deg(delta_sol)),
                'beta_deg': float(np.rad2deg(beta_sol)),
                'psi_err_deg': float(np.rad2deg(beta_sol)),
                'phi_deg': float(np.rad2deg(phi_sol)),
                'r': float(r), 'r_min': float(r_min), 'r_max': float(r_max),
                'corridor_ok': corridor_ok
            })
            
            if corridor_ok:
                v_max = v
                best_sol = {
                    'beta_rad': float(beta_sol), 'delta_rad': float(delta_sol),
                    'phi_rad': float(phi_sol),
                    'ay_g': float(ay_g), 'v': float(v), 'r': float(r),
                    'r_min': float(r_min), 'r_max': float(r_max),
                    'x_raw': list(sol.x)
                }
            else:
                last_failed_ay = ay_g
                break
            x_guess = sol.x
            
        # Continuous bisection refinement for 14-DOF
        if best_sol is not None and 'x_raw' in best_sol:
            ay_low = best_sol['ay_g']
            ay_high = min(last_failed_ay, ay_low + 0.15)
            x_b_guess = best_sol['x_raw']
            for _ in range(16):
                ay_mid = 0.5 * (ay_low + ay_high)
                v_m = np.sqrt(ay_mid * 9.81 * radius)
                r_m = v_m / radius
                
                def obj_14_b(x):
                    vy_val, d_val, phi_val, z1, z2, z3, z4, w1, w2, w3, w4 = x
                    states = [vy_val, r_m, phi_val, z1, z2, z3, z4, w1, w2, w3, w4]
                    inputs = [v_m, d_val]
                    res = res_func14(inputs, states, par14)
                    return np.array(res).flatten()

                def jac_14_b(x):
                    if jac_states_func14 is None or jac_inputs_func14 is None:
                        return None
                    vy_val, d_val, phi_val, z1, z2, z3, z4, w1, w2, w3, w4 = x
                    states = [vy_val, r_m, phi_val, z1, z2, z3, z4, w1, w2, w3, w4]
                    inputs = [v_m, d_val]
                    J_st = np.array(jac_states_func14(inputs, states, par14))
                    J_in = np.array(jac_inputs_func14(inputs, states, par14))
                    J = np.empty((11, 11))
                    J[:, 0] = J_st[:, 0]
                    J[:, 1] = J_in[:, 1]
                    J[:, 2:] = J_st[:, 2:]
                    return J

                jac_14_b_call = jac_14_b if (jac_states_func14 is not None and jac_inputs_func14 is not None) else None
                sol_b = opt.root(obj_14_b, x_b_guess, jac=jac_14_b_call, method='hybr', options={'xtol': 1e-6})
                if not sol_b.success:
                    sol_b = opt.root(obj_14_b, x_b_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    
                res_tol_b = max(15.0, 0.05 * params.m * ay_mid)
                if sol_b.success and np.linalg.norm(obj_14_b(sol_b.x)) < res_tol_b:
                    vy_s, d_s, phi_s = sol_b.x[0], sol_b.x[1], sol_b.x[2]
                    b_s = vy_s / v_m
                    if abs(np.rad2deg(b_s)) <= 25.0 and abs(np.rad2deg(d_s)) <= 35.0:
                        c_b, s_b = np.cos(b_s), np.sin(b_s)
                        wheels = {
                            'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                            'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                            'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                            'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
                        }
                        radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
                        if min(radii) >= radius - 1.5 and max(radii) <= radius + 1.5:
                            ay_low = ay_mid
                            v_max = v_m
                            x_b_guess = sol_b.x
                            best_sol = {
                                'beta_rad': float(b_s), 'delta_rad': float(d_s),
                                'phi_rad': float(phi_s),
                                'ay_g': float(ay_mid), 'v': float(v_m), 'r': float(r_m),
                                'r_min': float(min(radii)), 'r_max': float(max(radii))
                            }
                            continue
                ay_high = ay_mid

    elif model_type == '7-DOF':
        veh7 = Vehicle7DOF(params, tp, tp)
        funcs7 = veh7.get_trim_casadi_functions()
        res_func7 = funcs7['res_func']
        
        x_guess = [vy_guess, delta_guess, w_guess, w_guess, w_guess, w_guess]
        par7 = [wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r, L, t_f, t_r, lmuy_val]
        
        for ay_g in ay_grid:
            ay = ay_g * 9.81
            v = np.sqrt(ay * radius)
            r = v / radius
            
            def obj_7dof(x):
                vy, d, w1, w2, w3, w4 = x
                states = [vy, r, w1, w2, w3, w4]
                inputs = [v, d]
                res = res_func7(inputs, states, par7)
                return np.array(res).flatten()
                
            sol = opt.root(obj_7dof, x_guess, method='hybr', options={'xtol': 1e-6})
            if not sol.success:
                sol = opt.root(obj_7dof, x_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                
            if not sol.success:
                last_failed_ay = ay_g
                break
                
            res_norm = np.linalg.norm(obj_7dof(sol.x))
            if res_norm > 8.0:
                last_failed_ay = ay_g
                break
                
            vy_sol, delta_sol = sol.x[0], sol.x[1]
            beta_sol = vy_sol / v
            
            if abs(np.rad2deg(beta_sol)) > 25.0 or abs(np.rad2deg(delta_sol)) > 35.0:
                last_failed_ay = ay_g
                break
                
            c_b, s_b = np.cos(beta_sol), np.sin(beta_sol)
            wheels = {
                'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
            }
            radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
            r_min, r_max = min(radii), max(radii)
            corridor_ok = (r_min >= radius - 1.5) and (r_max <= radius + 1.5)
            
            sweep_results.append({
                'v': float(v), 'ay_g': float(ay_g),
                'delta_deg': float(np.rad2deg(delta_sol)),
                'beta_deg': float(np.rad2deg(beta_sol)),
                'psi_err_deg': float(np.rad2deg(beta_sol)),
                'r': float(r), 'r_min': float(r_min), 'r_max': float(r_max),
                'corridor_ok': corridor_ok
            })
            
            if corridor_ok:
                v_max = v
                best_sol = {
                    'beta_rad': float(beta_sol), 'delta_rad': float(delta_sol),
                    'ay_g': float(ay_g), 'v': float(v), 'r': float(r),
                    'r_min': float(r_min), 'r_max': float(r_max),
                    'x_raw': list(sol.x)
                }
            else:
                last_failed_ay = ay_g
                break
            x_guess = sol.x
            
        # Continuous bisection refinement for 7DOF
        if best_sol is not None and 'x_raw' in best_sol:
            ay_low = best_sol['ay_g']
            ay_high = min(last_failed_ay, ay_low + 0.15)
            x_b_guess = best_sol['x_raw']
            for _ in range(16):
                ay_mid = 0.5 * (ay_low + ay_high)
                v_m = np.sqrt(ay_mid * 9.81 * radius)
                r_m = v_m / radius
                
                def obj_7_b(x):
                    vy, d, w1, w2, w3, w4 = x
                    states = [vy, r_m, w1, w2, w3, w4]
                    inputs = [v_m, d]
                    res = res_func7(inputs, states, par7)
                    return np.array(res).flatten()
                    
                sol_b = opt.root(obj_7_b, x_b_guess, method='hybr', options={'xtol': 1e-6})
                if not sol_b.success:
                    sol_b = opt.root(obj_7_b, x_b_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    
                if sol_b.success and np.linalg.norm(obj_7_b(sol_b.x)) < 5.0:
                    vy_s, d_s = sol_b.x[0], sol_b.x[1]
                    b_s = vy_s / v_m
                    if abs(np.rad2deg(b_s)) <= 25.0 and abs(np.rad2deg(d_s)) <= 35.0:
                        c_b, s_b = np.cos(b_s), np.sin(b_s)
                        wheels = {
                            'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                            'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                            'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                            'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
                        }
                        radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
                        if min(radii) >= radius - 1.5 and max(radii) <= radius + 1.5:
                            ay_low = ay_mid
                            v_max = v_m
                            x_b_guess = sol_b.x
                            best_sol = {
                                'beta_rad': float(b_s), 'delta_rad': float(d_s),
                                'ay_g': float(ay_mid), 'v': float(v_m), 'r': float(r_m),
                                'r_min': float(min(radii)), 'r_max': float(max(radii))
                            }
                            continue
                ay_high = ay_mid

    else:
        # 3-DOF Model
        veh3 = Vehicle3DOF(params, tp, tp)
        funcs3 = veh3.get_global_casadi_functions()
        res_func3 = funcs3['res_func']
        
        beta_guess = (b / radius)
        delta_guess = (L / radius)
        
        for ay_g in ay_grid:
            ay = ay_g * 9.81
            v = np.sqrt(ay * radius)
            r = v / radius
            
            def obj_3dof(x):
                beta, delta = x
                res = res_func3([v, delta], [beta, r], [wd_f, lltd, lmuy_val, L, t_f, t_r, aero_bal])
                return np.array(res).flatten()
                
            sol = opt.root(obj_3dof, [beta_guess, delta_guess], method='hybr', options={'xtol': 1e-7})
            if not sol.success:
                sol = opt.root(obj_3dof, [beta_guess, delta_guess], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                
            if not sol.success:
                last_failed_ay = ay_g
                break
                
            res_norm = np.linalg.norm(obj_3dof(sol.x))
            if res_norm > 8.0:
                last_failed_ay = ay_g
                break
                
            beta_sol, delta_sol = sol.x
            if abs(np.rad2deg(beta_sol)) > 25.0 or abs(np.rad2deg(delta_sol)) > 35.0:
                last_failed_ay = ay_g
                break
                
            c_b, s_b = np.cos(beta_sol), np.sin(beta_sol)
            wheels = {
                'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
            }
            radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
            r_min, r_max = min(radii), max(radii)
            corridor_ok = (r_min >= radius - 1.5) and (r_max <= radius + 1.5)
            
            sweep_results.append({
                'v': float(v), 'ay_g': float(ay_g),
                'delta_deg': float(np.rad2deg(delta_sol)),
                'beta_deg': float(np.rad2deg(beta_sol)),
                'psi_err_deg': float(np.rad2deg(beta_sol)),
                'r': float(r), 'r_min': float(r_min), 'r_max': float(r_max),
                'corridor_ok': corridor_ok
            })
            
            if corridor_ok:
                v_max = v
                best_sol = {
                    'beta_rad': float(beta_sol), 'delta_rad': float(delta_sol),
                    'ay_g': float(ay_g), 'v': float(v), 'r': float(r),
                    'r_min': float(r_min), 'r_max': float(r_max)
                }
            else:
                last_failed_ay = ay_g
                break
            beta_guess, delta_guess = beta_sol, delta_sol

        # Continuous bisection refinement for 3DOF
        if best_sol is not None:
            ay_low = best_sol['ay_g']
            ay_high = min(last_failed_ay, ay_low + 0.15)
            b_guess_b = best_sol['beta_rad']
            d_guess_b = best_sol['delta_rad']
            
            for _ in range(16):
                ay_mid = 0.5 * (ay_low + ay_high)
                v_m = np.sqrt(ay_mid * 9.81 * radius)
                r_m = v_m / radius
                
                def obj_3_b(x):
                    b_val, d_val = x
                    res = res_func3([v_m, d_val], [b_val, r_m], [wd_f, lltd, lmuy_val, L, t_f, t_r, aero_bal])
                    return np.array(res).flatten()
                    
                sol_b = opt.root(obj_3_b, [b_guess_b, d_guess_b], method='hybr', options={'xtol': 1e-7})
                if not sol_b.success:
                    sol_b = opt.root(obj_3_b, [b_guess_b, d_guess_b], method='lm', options={'ftol': 1e-7, 'xtol': 1e-7})
                    
                if sol_b.success and np.linalg.norm(obj_3_b(sol_b.x)) < 1.0:
                    b_s, d_s = sol_b.x
                    if abs(np.rad2deg(b_s)) <= 25.0 and abs(np.rad2deg(d_s)) <= 35.0:
                        c_b, s_b = np.cos(b_s), np.sin(b_s)
                        wheels = {
                            'FL': (a * c_b - (t_f/2) * s_b, radius + a * s_b + (t_f/2) * c_b),
                            'FR': (a * c_b - (-t_f/2) * s_b, radius + a * s_b + (-t_f/2) * c_b),
                            'RL': (-b * c_b - (t_r/2) * s_b, radius - b * s_b + (t_r/2) * c_b),
                            'RR': (-b * c_b - (-t_r/2) * s_b, radius - b * s_b + (-t_r/2) * c_b),
                        }
                        radii = [np.sqrt(wx**2 + wy**2) for wx, wy in wheels.values()]
                        if min(radii) >= radius - 1.5 and max(radii) <= radius + 1.5:
                            ay_low = ay_mid
                            v_max = v_m
                            b_guess_b, d_guess_b = b_s, d_s
                            best_sol = {
                                'beta_rad': float(b_s), 'delta_rad': float(d_s),
                                'ay_g': float(ay_mid), 'v': float(v_m), 'r': float(r_m),
                                'r_min': float(min(radii)), 'r_max': float(max(radii))
                            }
                            continue
                ay_high = ay_mid

    sweep_df = pd.DataFrame(sweep_results)
    
    if v_max > 3.0 and best_sol is not None:
        lap_time = 2.0 * np.pi * radius / v_max
        best_ay = (v_max**2 / radius) / 9.81
        beta_opt = best_sol['beta_rad']
        delta_opt = best_sol['delta_rad']
        phi_opt = best_sol.get('phi_rad', 0.0)
        
        # Calculate dynamic tire loads and forces at optimal condition
        F_down = 0.5 * params.rho * params.S * params.Cz * v_max**2
        Cd = getattr(params, 'Cd', 1.0)
        F_drag = 0.5 * params.rho * params.S * Cd * v_max**2
        F_roll = 0.015 * params.m * 9.81
        Fx_rear_tot = F_drag + F_roll
        
        Fz_stat_f = params.m * 9.81 * (b / L) + F_down * aero_bal
        Fz_stat_r = params.m * 9.81 * (a / L) + F_down * (1.0 - aero_bal)
        
        lat_wt_moment = params.m * (best_ay * 9.81) * params.h_cg
        dFz_f = (lat_wt_moment * lltd) / t_f
        dFz_r = (lat_wt_moment * (1.0 - lltd)) / t_r
        
        fz_fl = max(Fz_stat_f / 2.0 - dFz_f, 10.0)
        fz_fr = max(Fz_stat_f / 2.0 + dFz_f, 10.0)
        fz_rl = max(Fz_stat_r / 2.0 - dFz_r, 10.0)
        fz_rr = max(Fz_stat_r / 2.0 + dFz_r, 10.0)
        
        v_y = v_max * np.tan(beta_opt)
        r_opt = v_max / radius
        alpha_f = delta_opt - np.arctan2(v_y + a * r_opt, v_max)
        alpha_r = -np.arctan2(v_y - b * r_opt, v_max)
        
        fy_fl = abs(float(mf61_lateral_force(alpha_f, fz_fl, 0.0, tp)))
        fy_fr = abs(float(mf61_lateral_force(alpha_f, fz_fr, 0.0, tp)))
        fy_rl = abs(float(mf61_lateral_force(alpha_r, fz_rl, 0.0, tp)))
        fy_rr = abs(float(mf61_lateral_force(alpha_r, fz_rr, 0.0, tp)))
        
        fx_fl, fx_fr = 0.0, 0.0
        fx_rl = float(Fx_rear_tot / 2.0)
        fx_rr = float(Fx_rear_tot / 2.0)
        R_w = getattr(params, 'R_w', 0.228)
        torque_rl = float(fx_rl * R_w)
        torque_rr = float(fx_rr * R_w)
        torque_total = float((fx_rl + fx_rr) * R_w)
        
        # Power / Energy Consumption
        P_aero_w = F_drag * v_max
        P_roll_w = F_roll * v_max
        P_mech_w = P_aero_w + P_roll_w
        motor_eff = getattr(params, 'motor_base_eff', 0.92) * getattr(params, 'inverter_eff', 0.96)
        P_elec_kw = (P_mech_w / max(motor_eff, 0.5)) / 1000.0
        energy_per_lap_wh = P_elec_kw * (lap_time / 3600.0) * 1000.0
        
        n_pts = 160
        t_arr = np.linspace(0.0, lap_time, n_pts)
        theta_arr = (v_max * t_arr) / radius
        
        records = []
        for t_val, th in zip(t_arr, theta_arr):
            s_val = v_max * t_val
            x_cg = radius * np.cos(th)
            y_cg = radius * np.sin(th)
            
            psi = th + np.pi/2.0 + beta_opt
            c_psi, s_psi = np.cos(psi), np.sin(psi)
            
            x_fl = x_cg + a * c_psi - (t_f/2.0) * s_psi
            y_fl = y_cg + a * s_psi + (t_f/2.0) * c_psi
            x_fr = x_cg + a * c_psi - (-t_f/2.0) * s_psi
            y_fr = y_cg + a * s_psi + (-t_f/2.0) * c_psi
            x_rl = x_cg + (-b) * c_psi - (t_r/2.0) * s_psi
            y_rl = y_cg + (-b) * s_psi + (t_r/2.0) * c_psi
            x_rr = x_cg + (-b) * c_psi - (-t_r/2.0) * s_psi
            y_rr = y_cg + (-b) * s_psi + (-t_r/2.0) * c_psi
            
            r_wheels = [np.sqrt(x_fl**2 + y_fl**2), np.sqrt(x_fr**2 + y_fr**2),
                        np.sqrt(x_rl**2 + y_rl**2), np.sqrt(x_rr**2 + y_rr**2)]
            
            records.append({
                't': float(t_val),
                's': float(s_val),
                'theta_deg': float(np.rad2deg(th)),
                'v_ms': float(v_max),
                'v_kmh': float(v_max * 3.6),
                'ay_g': float(best_ay),
                'ay_ms2': float(best_ay * 9.81),
                'ax_g': 0.0,
                'delta_deg': float(np.rad2deg(delta_opt)),
                'beta_deg': float(np.rad2deg(beta_opt)),
                'phi_deg': float(np.rad2deg(phi_opt)),
                'psi_err_deg': float(np.rad2deg(beta_opt)),
                'yaw_rate_rads': float(r_opt),
                'x_cg': float(x_cg),
                'y_cg': float(y_cg),
                'psi_deg': float(np.rad2deg(psi)),
                'x_fl': float(x_fl), 'y_fl': float(y_fl),
                'x_fr': float(x_fr), 'y_fr': float(y_fr),
                'x_rl': float(x_rl), 'y_rl': float(y_rl),
                'x_rr': float(x_rr), 'y_rr': float(y_rr),
                'r_min': float(min(r_wheels)),
                'r_max': float(max(r_wheels)),
                'Fz_fl': float(fz_fl), 'Fz_fr': float(fz_fr),
                'Fz_rl': float(fz_rl), 'Fz_rr': float(fz_rr),
                'Fy_fl': float(fy_fl), 'Fy_fr': float(fy_fr),
                'Fy_rl': float(fy_rl), 'Fy_rr': float(fy_rr),
                'Fx_fl': float(fx_fl), 'Fx_fr': float(fx_fr),
                'Fx_rl': float(fx_rl), 'Fx_rr': float(fx_rr),
                'alpha_fl_deg': float(np.rad2deg(alpha_f)),
                'alpha_fr_deg': float(np.rad2deg(alpha_f)),
                'alpha_rl_deg': float(np.rad2deg(alpha_r)),
                'alpha_rr_deg': float(np.rad2deg(alpha_r)),
                'Torque_fl_Nm': 0.0,
                'Torque_fr_Nm': 0.0,
                'Torque_rl_Nm': float(torque_rl),
                'Torque_rr_Nm': float(torque_rr),
                'Torque_total_Nm': float(torque_total),
                'Power_elec_kw': float(P_elec_kw),
                'Energy_lap_wh': float(energy_per_lap_wh)
            })
            
        telemetry_df = pd.DataFrame(records)
        corridor_margin = float(1.5 - max(best_sol['r_max'] - radius, radius - best_sol['r_min']))
        
        return {
            'success': True,
            'time': float(lap_time),
            'v_max': float(v_max),
            'ay': float(best_ay),
            'delta_deg': float(np.rad2deg(delta_opt)),
            'beta_deg': float(np.rad2deg(beta_opt)),
            'phi_deg': float(np.rad2deg(phi_opt)),
            'psi_err_deg': float(np.rad2deg(beta_opt)),
            'r_min': float(best_sol['r_min']),
            'r_max': float(best_sol['r_max']),
            'corridor_margin': corridor_margin,
            'power_elec_kw': float(P_elec_kw),
            'energy_lap_wh': float(energy_per_lap_wh),
            'sweep': sweep_df,
            'telemetry': telemetry_df
        }
    else:
        return {
            'success': False,
            'time': 999.0,
            'v_max': 0.0,
            'ay': 0.0,
            'delta_deg': 0.0,
            'beta_deg': 0.0,
            'phi_deg': 0.0,
            'psi_err_deg': 0.0,
            'r_min': 0.0,
            'r_max': 0.0,
            'corridor_margin': 0.0,
            'power_elec_kw': 0.0,
            'energy_lap_wh': 0.0,
            'sweep': sweep_df,
            'telemetry': pd.DataFrame()
        }
