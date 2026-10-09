import os
import numpy as np
from scipy.optimize import minimize, root
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo.mdo_skidpad import simulate_skidpad
from src.mdo.mdo_acceleration import simulate_acceleration
from src.models.tire_model_mf61 import mf61_lateral_force

def compute_stability_metrics(L, wd_f, lltd, aero_bal, v_x=15.0, tir_path=None):
    """
    Computes linear Understeer Gradient (K_us), Critical Speed (v_crit), and Characteristic Speed (v_char).
    - K_us > 0: Understeering vehicle (Stable at all speeds, v_crit = inf, v_char = sqrt(L / K_us))
    - K_us < 0: Oversteering vehicle (Yaw unstable above v_crit = sqrt(L / -K_us))
    - K_us = 0: Neutral steering vehicle (v_crit = inf, v_char = inf)
    """
    tp = TireParams(tir_path)
    vp = VehicleParams()

    m = vp.m
    g = vp.g
    a = L * (1.0 - wd_f)
    b = L * wd_f
    
    F_down = 0.5 * vp.rho * vp.S * vp.Cz * (v_x**2)
    Fz_f = max((m * g * (b / L) + F_down * aero_bal) / 2.0, 100.0)
    Fz_r = max((m * g * (a / L) + F_down * (1.0 - aero_bal)) / 2.0, 100.0)
    
    # Cornering stiffness per axle (sum of 2 tires) at small slip alpha = 1 deg
    d_alpha = np.deg2rad(1.0)
    Fy_f_1deg = abs(float(mf61_lateral_force(d_alpha, Fz_f, 0.0, tp))) * 2.0
    Fy_r_1deg = abs(float(mf61_lateral_force(d_alpha, Fz_r, 0.0, tp))) * 2.0
    
    C_alpha_f = max(Fy_f_1deg / d_alpha, 1000.0) # [N/rad]
    C_alpha_r = max(Fy_r_1deg / d_alpha, 1000.0) # [N/rad]
    
    K_us = (m / L) * ((b / C_alpha_f) - (a / C_alpha_r))
    K_us_degg = K_us * (180.0 / np.pi) * g
    
    # Stability and Control Derivatives (Linearized 2-DOF Bicycle Model)
    Y_beta = -(C_alpha_f + C_alpha_r)                         # [N/rad]
    Y_r = -(a * C_alpha_f - b * C_alpha_r) / max(v_x, 1.0)     # [N·s/rad]
    N_beta = -(a * C_alpha_f - b * C_alpha_r)                 # [N·m/rad]  (Directional Stability)
    N_r = -(a**2 * C_alpha_f + b**2 * C_alpha_r) / max(v_x, 1.0) # [N·m·s/rad] (Yaw Damping)
    N_delta = a * C_alpha_f                                  # [N·m/rad]  (Control Authority)
    Y_delta = C_alpha_f                                      # [N/rad]
    
    # Dynamic Index (Yaw inertia centralization)
    Iz = vp.Iz
    DI = Iz / max(m * a * b, 1e-3)
    
    # State-Space Matrix in (beta, r) coordinates
    a11 = Y_beta / (m * max(v_x, 1.0))
    a12 = (Y_r / (m * max(v_x, 1.0))) - 1.0
    a21 = N_beta / Iz
    a22 = N_r / Iz
    A_mat = np.array([[a11, a12], [a21, a22]])
    
    tr_A = float(np.trace(A_mat))
    det_A = float(np.linalg.det(A_mat))
    eigenvalues = np.linalg.eigvals(A_mat)
    
    is_stable = bool(tr_A < 0 and det_A > 0 and np.all(np.real(eigenvalues) < 0))
    disc = tr_A**2 - 4.0 * det_A

    if det_A <= 0:
        # Aperiodic unstable mode: one positive real eigenvalue (divergent yaw)
        omega_n = None
        f_n = None
        zeta = None
        mode_desc = f"Aperiodico Instabile (v > v_crit, det(A)={det_A:.2e})"
    elif disc >= 0:
        # Overdamped stable/aperiodic mode: two negative real eigenvalues
        omega_n = float(np.sqrt(det_A))
        f_n = float(omega_n / (2.0 * np.pi))
        zeta = float(-tr_A / (2.0 * max(omega_n, 1e-3)))
        mode_desc = f"Aperiodico Sovrasmorzato (fn={f_n:.2f} Hz, ζ={zeta:.2f})"
    else:
        # Underdamped oscillatory mode: complex conjugate eigenvalues
        omega_n = float(np.sqrt(det_A))
        f_n = float(omega_n / (2.0 * np.pi))
        zeta = float(-tr_A / (2.0 * max(omega_n, 1e-3)))
        mode_desc = f"Oscillatorio Smorzato (fn={f_n:.2f} Hz, ζ={zeta:.2f})"
    
    if K_us > 1e-6:
        v_char = np.sqrt(L / K_us)
        v_crit = float('inf')
        behavior = "Understeer"
        behavior_it = "Sottosterzante (Stabile a tutte le velocità)"
    elif K_us < -1e-6:
        v_char = float('inf')
        v_crit = np.sqrt(L / (-K_us))
        behavior = "Oversteer"
        behavior_it = f"Sovrasterzante (Instabile sopra v_crit = {v_crit*3.6:.1f} km/h)"
    else:
        v_char = float('inf')
        v_crit = float('inf')
        behavior = "Neutral"
        behavior_it = "Neutro"
        
    return {
        'K_us': float(K_us),
        'K_us_degg': float(K_us_degg),
        'behavior': behavior,
        'behavior_it': behavior_it,
        'is_stable': is_stable,
        'mode_desc': mode_desc,
        'v_char_ms': float(v_char),
        'v_char_kmh': float(v_char * 3.6) if v_char != float('inf') else float('inf'),
        'v_crit_ms': float(v_crit),
        'v_crit_kmh': float(v_crit * 3.6) if v_crit != float('inf') else float('inf'),
        'C_alpha_f': float(C_alpha_f),
        'C_alpha_r': float(C_alpha_r),
        'Y_beta': float(Y_beta),
        'Y_r': float(Y_r),
        'N_beta': float(N_beta),
        'N_r': float(N_r),
        'N_delta': float(N_delta),
        'Y_delta': float(Y_delta),
        'DI': float(DI),
        'tr_A': tr_A,
        'det_A': det_A,
        'omega_n_rads': omega_n,
        'f_n_hz': f_n,
        'zeta': zeta,
        'eigenvalues': [complex(e) for e in eigenvalues]
    }


class GlobalOptimizer:
    def __init__(self, v_x=15.0, penalty_factor=0.0, tir_file=None, objective_mode="max_ay", 
                 skidpad_radius=9.125, accel_distance=75.0, tire_uncertainty_pct=10.0,
                 model_type="3-DOF", enforce_understeer=False, understeer_min_deg_g=0.10,
                 understeer_weight=1.0, target_kus_deg_g=0.0):
        self.v_x = v_x
        self.penalty_factor = penalty_factor
        self.objective_mode = objective_mode
        self.skidpad_radius = skidpad_radius
        self.accel_distance = accel_distance
        self.tire_uncertainty_pct = float(tire_uncertainty_pct)
        self.model_type = model_type
        self.enforce_understeer = bool(enforce_understeer)
        self.understeer_min_deg_g = float(understeer_min_deg_g)
        self.understeer_weight = float(understeer_weight)
        self.target_kus_deg_g = float(target_kus_deg_g)
        
        self.tp = TireParams(tir_file)
        self.tir_path = self.tp.filepath
        self.vp = VehicleParams()
        
        # Build CasADi trim functions for the selected model
        if self.model_type == "14-DOF":
            self.model_inst = Vehicle14DOF(self.vp, self.tp, self.tp)
            funcs = self.model_inst.get_trim_casadi_functions()
        elif self.model_type == "7-DOF":
            self.model_inst = Vehicle7DOF(self.vp, self.tp, self.tp)
            funcs = self.model_inst.get_trim_casadi_functions()
        else:
            self.model_inst = Vehicle3DOF(self.vp, self.tp, self.tp)
            funcs = self.model_inst.get_global_casadi_functions()
            
        self.res_func = funcs['res_func']
        self.jac_states_func = funcs.get('jac_states_func', None)
        self.jac_inputs_func = funcs.get('jac_inputs_func', None)

    def calc_understeer_gradient(self, X):
        """
        Calculates physical Understeer Gradient Kus [deg/g] based on MF6.1 tire load sensitivity,
        weight distribution, LLTD, and suspension roll stiffness.
        Evaluates pure cornering stiffness with respect to slip angle at zero inclination (gamma=0)
        to avoid camber thrust contamination.
        """
        L = float(X[0])
        t_f = float(X[1])
        t_r = float(X[2])
        wd_f = float(X[3])
        lltd = float(X[4])
        aero_bal = float(X[5])
        
        # In 14-DOF, couple roll stiffness distribution (K_phi_f, K_phi_r) into effective LLTD
        if len(X) >= 14 and (float(X[10]) + float(X[11])) > 0:
            K_phi_f = float(X[10])
            K_phi_r = float(X[11])
            # Physical LLTD fully determined by roll-stiffness ratio; override the
            # algebraic lltd parameter (the old 50/50 average distorted Kus in 14-DOF).
            lltd = K_phi_f / (K_phi_f + K_phi_r)
        

        m = self.vp.m
        g = self.vp.g
        a = L * (1.0 - wd_f)
        b = L * wd_f
        
        F_down = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (self.v_x**2)
        Fz_f_static = (m * g * (b / L) + F_down * aero_bal) / 2.0
        Fz_r_static = (m * g * (a / L) + F_down * (1.0 - aero_bal)) / 2.0
        
        # 1.0 g lateral load transfer effect on axle cornering stiffness
        ay_nom = 1.0 * g
        dFz_f = (m * ay_nom * self.vp.h_cg * lltd) / (2.0 * max(t_f, 0.5))
        dFz_r = (m * ay_nom * self.vp.h_cg * (1.0 - lltd)) / (2.0 * max(t_r, 0.5))
        
        Fz_f_in = max(Fz_f_static - dFz_f, 50.0)
        Fz_f_out = Fz_f_static + dFz_f
        Fz_r_in = max(Fz_r_static - dFz_r, 50.0)
        Fz_r_out = Fz_r_static + dFz_r
        
        d_alpha = np.deg2rad(1.0)
        Fy_f = (abs(float(mf61_lateral_force(d_alpha, Fz_f_in, 0.0, self.tp))) +
                abs(float(mf61_lateral_force(d_alpha, Fz_f_out, 0.0, self.tp))))
        Fy_r = (abs(float(mf61_lateral_force(d_alpha, Fz_r_in, 0.0, self.tp))) +
                abs(float(mf61_lateral_force(d_alpha, Fz_r_out, 0.0, self.tp))))
                
        C_alpha_f = max(Fy_f / d_alpha, 1000.0)
        C_alpha_r = max(Fy_r / d_alpha, 1000.0)
        
        K_us = (m / L) * ((b / C_alpha_f) - (a / C_alpha_r))
        K_us_degg = K_us * (180.0 / np.pi) * g
        return K_us_degg

    def _evaluate_limit_ay_at_speed(self, v_eval, X_params, delta_max_deg=30.0, lmuy_mult=1.0):
        """
        Finds the true physical maximum Ay (Limit of Grip) for a given speed v_eval
        and parameter vector X_params.
        """
        from src.core.simulation_options import get_simulation_options
        opts = get_simulation_options()
        beta_limit = getattr(opts, 'beta_limit_deg', 35.0)

        ay_start  = 0.05 * 9.81   # [m/s^2]
        ay_end    = 4.2  * 9.81 * lmuy_mult
        n_steps   = 40
        Rw = self.vp.R_w

        if self.model_type == "14-DOF":
            # X_params: [L, t_f, t_r, wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r, K_phi_f, K_phi_r, K_z_f, K_z_r]
            L, t_f, t_r, wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r, K_phi_f, K_phi_r, K_z_f, K_z_r = X_params
            par14 = [wd_f, lltd, K_phi_f, K_phi_r, K_z_f, K_z_r, aero_bal,
                     cam_f, cam_r, toe_f, toe_r, L, t_f, t_r, lmuy_mult]
            
            # Physically grounded initial guess from kinematics and aero downforce
            F_down_init = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (v_eval**2)
            zf_guess = -(self.vp.m * self.vp.g * (1.0 - wd_f) + F_down_init * aero_bal) / (2.0 * max(K_z_f, 5000.0))
            zr_guess = -(self.vp.m * self.vp.g * wd_f + F_down_init * (1.0 - aero_bal)) / (2.0 * max(K_z_r, 5000.0))
            delta_kin_0 = (L * ay_start) / (v_eval**2)
            vy_kin_0 = -(L * wd_f) * (ay_start / v_eval)
            x_guess = [vy_kin_0, delta_kin_0, 0.0, zf_guess, zf_guess, zr_guess, zr_guess,
                       v_eval/Rw, v_eval/Rw, v_eval/Rw, v_eval/Rw]
            max_ay_g = 0.0
            last_failed_ay_g = ay_end / 9.81
            delta_prev = -1e9

            for ay in np.linspace(ay_start, ay_end, n_steps):
                r_val = ay / v_eval
                ay_g  = ay / 9.81

                def obj14(x_st):
                    vy, d, phi, z1, z2, z3, z4, w1, w2, w3, w4 = x_st
                    states = [vy, r_val, phi, z1, z2, z3, z4, w1, w2, w3, w4]
                    inputs = [v_eval, d]
                    res = self.res_func(inputs, states, par14)
                    return np.array(res).flatten()

                def jac14(x_st):
                    if self.jac_states_func is None or self.jac_inputs_func is None:
                        return None
                    vy, d, phi, z1, z2, z3, z4, w1, w2, w3, w4 = x_st
                    st = [vy, r_val, phi, z1, z2, z3, z4, w1, w2, w3, w4]
                    inp = [v_eval, d]
                    J_st = np.array(self.jac_states_func(inp, st, par14))
                    J_in = np.array(self.jac_inputs_func(inp, st, par14))
                    J = np.empty((11, 11))
                    J[:, 0] = J_st[:, 0]
                    J[:, 1] = J_in[:, 1]
                    J[:, 2:] = J_st[:, 2:]
                    return J

                jac_call = jac14 if (self.jac_states_func is not None and self.jac_inputs_func is not None) else None
                sol = root(obj14, x_guess, jac=jac_call, method='hybr', options={'xtol': 1e-6})
                res_norm = np.linalg.norm(obj14(sol.x))
                res_limit = max(20.0, 0.06 * self.vp.m * ay)

                if not sol.success or res_norm > res_limit:
                    sol = root(obj14, x_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    res_norm = np.linalg.norm(obj14(sol.x))

                if not sol.success or res_norm > res_limit:
                    last_failed_ay_g = ay_g
                    break

                vy_sol, delta_sol = sol.x[0], sol.x[1]
                b_sol = vy_sol / v_eval

                # Physical stop: excessive sideslip/steer or true counter-steering / spin drop (> 1.5 deg)
                if (abs(np.rad2deg(b_sol)) > beta_limit or abs(np.rad2deg(delta_sol)) > delta_max_deg + 5.0
                        or (max_ay_g > 0.6 and delta_sol < delta_prev - np.deg2rad(1.5))):
                    last_failed_ay_g = ay_g
                    break

                delta_prev = delta_sol
                max_ay_g = ay_g
                x_guess = sol.x

            # Bisection refinement
            if max_ay_g > 0.1:
                ay_low = max_ay_g * 9.81
                ay_high = min(last_failed_ay_g * 9.81, ay_low + (ay_end - ay_start) / (n_steps - 1) * 1.5)
                x_b = x_guess
                for _ in range(12):
                    ay_mid = 0.5 * (ay_low + ay_high)
                    r_m = ay_mid / v_eval
                    def obj_b(x_st):
                        vy, d, phi, z1, z2, z3, z4, w1, w2, w3, w4 = x_st
                        return np.array(self.res_func([v_eval, d], [vy, r_m, phi, z1, z2, z3, z4, w1, w2, w3, w4], par14)).flatten()

                    def jac_b(x_st):
                        if self.jac_states_func is None or self.jac_inputs_func is None:
                            return None
                        vy, d, phi, z1, z2, z3, z4, w1, w2, w3, w4 = x_st
                        st = [vy, r_m, phi, z1, z2, z3, z4, w1, w2, w3, w4]
                        inp = [v_eval, d]
                        J_st = np.array(self.jac_states_func(inp, st, par14))
                        J_in = np.array(self.jac_inputs_func(inp, st, par14))
                        J = np.empty((11, 11))
                        J[:, 0] = J_st[:, 0]
                        J[:, 1] = J_in[:, 1]
                        J[:, 2:] = J_st[:, 2:]
                        return J

                    jac_b_call = jac_b if (self.jac_states_func is not None and self.jac_inputs_func is not None) else None
                    sol_b = root(obj_b, x_b, jac=jac_b_call, method='hybr', options={'xtol': 1e-6})
                    res_limit_b = max(20.0, 0.06 * self.vp.m * ay_mid)
                    if not sol_b.success or np.linalg.norm(obj_b(sol_b.x)) > res_limit_b:
                        sol_b = root(obj_b, x_b, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    if sol_b.success and np.linalg.norm(obj_b(sol_b.x)) <= res_limit_b:
                        b_s = sol_b.x[0] / v_eval
                        d_s = sol_b.x[1]
                        if abs(np.rad2deg(b_s)) <= beta_limit and abs(np.rad2deg(d_s)) <= delta_max_deg + 5.0 and d_s >= delta_prev - np.deg2rad(1.5):
                            ay_low = ay_mid
                            x_b = sol_b.x
                            continue
                    ay_high = ay_mid
                return ay_low / 9.81
            else:
                # Robust fallback to analytical aero-enhanced tire grip to prevent 0g collapse
                F_down = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (v_eval**2)
                ay_analytic = (self.vp.mu * lmuy_mult) * (1.0 + F_down / (self.vp.m * self.vp.g))
                return max(0.5, float(ay_analytic * 0.90))

        elif self.model_type == "7-DOF":
            # X_params: [L, t_f, t_r, wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r]
            L, t_f, t_r, wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r = X_params
            par7 = [wd_f, lltd, aero_bal, cam_f, cam_r, toe_f, toe_r, L, t_f, t_r, lmuy_mult]
            
            delta_kin_0 = (L * ay_start) / (v_eval**2)
            vy_kin_0 = -(L * wd_f) * (ay_start / v_eval)
            x_guess = [vy_kin_0, delta_kin_0, v_eval/Rw, v_eval/Rw, v_eval/Rw, v_eval/Rw]
            max_ay_g = 0.0
            last_failed_ay_g = ay_end / 9.81
            delta_prev = -1e9

            for ay in np.linspace(ay_start, ay_end, n_steps):
                r_val = ay / v_eval
                ay_g  = ay / 9.81

                def obj7(x_st):
                    vy, d, w1, w2, w3, w4 = x_st
                    states = [vy, r_val, w1, w2, w3, w4]
                    inputs = [v_eval, d]
                    res = self.res_func(inputs, states, par7)
                    return np.array(res).flatten()

                sol = root(obj7, x_guess, method='hybr', options={'xtol': 1e-6})
                res_norm = np.linalg.norm(obj7(sol.x))
                res_limit7 = max(15.0, 0.05 * self.vp.m * ay)

                if not sol.success or res_norm > res_limit7:
                    sol = root(obj7, x_guess, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    res_norm = np.linalg.norm(obj7(sol.x))

                if not sol.success or res_norm > res_limit7:
                    last_failed_ay_g = ay_g
                    break

                vy_sol, delta_sol = sol.x[0], sol.x[1]
                b_sol = vy_sol / v_eval
                if (abs(np.rad2deg(b_sol)) > beta_limit or abs(np.rad2deg(delta_sol)) > delta_max_deg + 5.0
                        or (max_ay_g > 0.6 and delta_sol < delta_prev - np.deg2rad(1.5))):
                    last_failed_ay_g = ay_g
                    break

                delta_prev = delta_sol
                max_ay_g = ay_g
                x_guess = sol.x

            # Bisection refinement
            if max_ay_g > 0.1:
                ay_low = max_ay_g * 9.81
                ay_high = min(last_failed_ay_g * 9.81, ay_low + (ay_end - ay_start) / (n_steps - 1) * 1.5)
                x_b = x_guess
                for _ in range(12):
                    ay_mid = 0.5 * (ay_low + ay_high)
                    r_m = ay_mid / v_eval
                    def obj_b(x_st):
                        vy, d, w1, w2, w3, w4 = x_st
                        return np.array(self.res_func([v_eval, d], [vy, r_m, w1, w2, w3, w4], par7)).flatten()
                    sol_b = root(obj_b, x_b, method='hybr', options={'xtol': 1e-6})
                    res_limit7_b = max(15.0, 0.05 * self.vp.m * ay_mid)
                    if not sol_b.success or np.linalg.norm(obj_b(sol_b.x)) > res_limit7_b:
                        sol_b = root(obj_b, x_b, method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})
                    if sol_b.success and np.linalg.norm(obj_b(sol_b.x)) <= res_limit7_b:
                        b_s = sol_b.x[0] / v_eval
                        d_s = sol_b.x[1]
                        if abs(np.rad2deg(b_s)) <= beta_limit and abs(np.rad2deg(d_s)) <= delta_max_deg + 5.0 and d_s >= delta_prev - np.deg2rad(1.5):
                            ay_low = ay_mid
                            x_b = sol_b.x
                            continue
                    ay_high = ay_mid
                return ay_low / 9.81
            else:
                F_down = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (v_eval**2)
                ay_analytic = (self.vp.mu * lmuy_mult) * (1.0 + F_down / (self.vp.m * self.vp.g))
                return max(0.5, float(ay_analytic * 0.90))

        else:
            # 3-DOF Model
            L, t_f, t_r, wd_f, lltd, aero_bal = X_params[:6]
            beta_guess  = 0.0
            delta_guess = 0.0
            max_ay_g    = 0.0
            last_failed_ay_g = ay_end / 9.81
            delta_prev = -1e9

            for ay in np.linspace(ay_start, ay_end, n_steps):
                r_val = ay / v_eval
                ay_g  = ay / 9.81

                def obj3(x):
                    b, d = x
                    res = self.res_func([v_eval, d], [b, r_val], [wd_f, lltd, lmuy_mult, L, t_f, t_r, aero_bal])
                    return np.array(res).flatten()

                sol = root(obj3, [beta_guess, delta_guess], method='hybr', options={'xtol': 1e-6})
                if not sol.success:
                    sol = root(obj3, [beta_guess, delta_guess], method='lm', options={'ftol': 1e-6, 'xtol': 1e-6})

                if not sol.success:
                    last_failed_ay_g = ay_g
                    break

                b_sol, d_sol = sol.x
                res_norm = np.linalg.norm(obj3(sol.x))

                if (res_norm > 10.0 or abs(np.rad2deg(b_sol)) > beta_limit or abs(np.rad2deg(d_sol)) > delta_max_deg + 5.0
                        or (max_ay_g > 0.5 and d_sol < delta_prev - 1e-4)):
                    last_failed_ay_g = ay_g
                    break

                delta_prev = d_sol
                max_ay_g = ay_g
                beta_guess = b_sol
                delta_guess = d_sol

            # Bisection refinement
            if max_ay_g > 0.1:
                ay_low = max_ay_g * 9.81
                ay_high = min(last_failed_ay_g * 9.81, ay_low + (ay_end - ay_start) / (n_steps - 1) * 1.5)
                b_g, d_g = beta_guess, delta_guess
                for _ in range(12):
                    ay_mid = 0.5 * (ay_low + ay_high)
                    r_m = ay_mid / v_eval
                    def obj_b(x):
                        b, d = x
                        return np.array(self.res_func([v_eval, d], [b, r_m], [wd_f, lltd, lmuy_mult, L, t_f, t_r, aero_bal])).flatten()
                    sol_b = root(obj_b, [b_g, d_g], method='hybr', options={'xtol': 1e-6})
                    if sol_b.success and np.linalg.norm(obj_b(sol_b.x)) < 10.0:
                        b_s, d_s = sol_b.x
                        if abs(np.rad2deg(b_s)) <= beta_limit and abs(np.rad2deg(d_s)) <= delta_max_deg + 5.0 and d_s >= delta_prev - 1e-3:
                            ay_low = ay_mid
                            b_g, d_g = b_s, d_s
                            continue
                    ay_high = ay_mid
                return ay_low / 9.81
            return max_ay_g

    def _evaluate_limit_ay(self, X_params, delta_max_deg=30.0, lmuy_mult=1.0):
        return self._evaluate_limit_ay_at_speed(self.v_x, X_params, delta_max_deg, lmuy_mult)

    def _pack_vehicle_params(self, X):
        vp = VehicleParams()
        vp.L = float(X[0])
        vp.t_f = float(X[1])
        vp.t_r = float(X[2])
        vp.wd_f = float(X[3])
        vp.lltd = float(X[4])
        vp.aero_balance = float(X[5])
        
        if len(X) >= 10:
            vp.camber_f_static = float(X[6])
            vp.camber_r_static = float(X[7])
            vp.toe_f_static = float(X[8])
            vp.toe_r_static = float(X[9])
            
        if len(X) >= 14:
            vp.K_phi_f = float(X[10])
            vp.K_phi_r = float(X[11])
            vp.K_z_f   = float(X[12])
            vp.K_z_r   = float(X[13])
            
        return vp

    def objective(self, X):
        L, t_f, t_r = X[0], X[1], X[2]
        
        try:
            unc_fraction = max(self.tire_uncertainty_pct / 100.0, 0.0)
            lmuy_low = max(1.0 - unc_fraction, 0.5)
            
            ay_target_nom = self._evaluate_limit_ay_at_speed(self.v_x, X, lmuy_mult=1.0)
            
            if unc_fraction > 0.01:
                ay_target_low = self._evaluate_limit_ay_at_speed(self.v_x, X, lmuy_mult=lmuy_low)
                ay_target = 0.50 * ay_target_nom + 0.50 * ay_target_low
            else:
                ay_target = ay_target_nom
                
        except Exception:
            return 20.0
            
        # Chassis geometry penalty: penalize deviations from nominal monocoque dimensions (L_nom, tf_nom, tr_nom)
        L_nom = getattr(self.vp, 'L', 1.529)
        tf_nom = getattr(self.vp, 't_f', 1.252)
        tr_nom = getattr(self.vp, 't_r', 1.136)
        geom_penalty = self.penalty_factor * 5.0 * (
            ((L - L_nom) / L_nom)**2 +
            ((t_f - tf_nom) / tf_nom)**2 +
            ((t_r - tr_nom) / tr_nom)**2
        )
        
        if self.objective_mode == "target_neutral":
            kus = self.calc_understeer_gradient(X)
            # Quadratic penalty towards target neutrality (K_us ≈ target_kus_deg_g, default 0.0 deg/g)
            neutral_error = kus - self.target_kus_deg_g
            neutral_penalty = self.understeer_weight * 1.0 * (neutral_error**2)
            score = ay_target - geom_penalty - neutral_penalty
            return -score
        elif self.objective_mode == "max_ay":
            score = ay_target - geom_penalty
            if self.enforce_understeer:
                kus = self.calc_understeer_gradient(X)
                if kus < self.understeer_min_deg_g:
                    deficit = self.understeer_min_deg_g - kus
                    # Calibrated, smooth progressive understeer penalty
                    score -= self.understeer_weight * 0.10 * (deficit + deficit**2)
            return -score
        elif self.objective_mode == "min_skidpad":
            vp = self._pack_vehicle_params(X)
            res = simulate_skidpad(vp, radius=self.skidpad_radius, model_type=self.model_type, tir_path=self.tir_path)
            time_score = res['time'] + geom_penalty
            if self.enforce_understeer:
                kus = self.calc_understeer_gradient(X)
                if kus < self.understeer_min_deg_g:
                    deficit = self.understeer_min_deg_g - kus
                    time_score += self.understeer_weight * 0.10 * (deficit + deficit**2)
            return time_score
        elif self.objective_mode == "min_accel":
            vp = self._pack_vehicle_params(X)
            res = simulate_acceleration(vp, distance=self.accel_distance, is_awd=False)
            return res['time']
        else:
            return -ay_target

    def run_optimization(self, initial_guess=None, custom_bounds=None, n_starts=3):
        """
        Runs Multi-Start SLSQP optimization across the parameter bounds.
        """
        # Determine parameter dimension based on model
        if self.model_type == "14-DOF":
            default_x0 = [1.529, 1.252, 1.136, 0.485, 0.525, 0.509, -1.5, -1.0, 0.0, -0.135, 34377.0, 34377.0, 37600.0, 38500.0]
            default_bounds = [
                (1.525, 1.550), (1.220, 1.280), (1.110, 1.160),
                (0.35, 0.65),   (0.35, 0.65),   (0.35, 0.65),
                (-3.5, 0.0),    (-3.0, 0.0),    (-1.0, 1.0),    (-1.0, 1.0),
                (15000.0, 60000.0), (15000.0, 60000.0), (20000.0, 60000.0), (20000.0, 60000.0)
            ]
        elif self.model_type == "7-DOF":
            default_x0 = [1.529, 1.252, 1.136, 0.485, 0.525, 0.509, -1.5, -1.0, 0.0, -0.135]
            default_bounds = [
                (1.525, 1.550), (1.220, 1.280), (1.110, 1.160),
                (0.35, 0.65),   (0.35, 0.65),   (0.35, 0.65),
                (-3.5, 0.0),    (-3.0, 0.0),    (-1.0, 1.0),    (-1.0, 1.0)
            ]
        else:
            default_x0 = [1.529, 1.252, 1.136, 0.485, 0.525, 0.509]
            default_bounds = [
                (1.525, 1.550), (1.220, 1.280), (1.110, 1.160),
                (0.35, 0.65),   (0.35, 0.65),   (0.35, 0.65)
            ]

        x0 = list(initial_guess) if initial_guess is not None else default_x0
        bounds = list(custom_bounds) if custom_bounds is not None else default_bounds
        
        # Ensure sizes match
        if len(x0) < len(default_x0):
            x0.extend(default_x0[len(x0):])
        if len(bounds) < len(default_bounds):
            bounds.extend(default_bounds[len(bounds):])

        start_points = [x0]
        rng = np.random.default_rng(42)
        for _ in range(n_starts - 1):
            pt = []
            for b_low, b_high in bounds:
                if b_low == b_high:
                    pt.append(b_low)
                else:
                    pt.append(float(rng.uniform(b_low, b_high)))
            start_points.append(pt)

        cons = []
        if self.enforce_understeer and self.objective_mode != "target_neutral":
            cons.append({
                'type': 'ineq',
                'fun': lambda x_vec: self.calc_understeer_gradient(x_vec) - self.understeer_min_deg_g
            })

        best_res = None
        best_fun = float('inf')

        for pt in start_points:
            res = minimize(
                self.objective,
                pt,
                method='SLSQP',
                bounds=bounds,
                constraints=cons if len(cons) > 0 else None,
                options={'ftol': 1e-5, 'maxiter': 80, 'disp': False, 'eps': 5e-4}
            )
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res

        return best_res

    def run_sweep_optimization(self, initial_guess, custom_bounds, v_min=8.0, v_max=25.0, num_v=8):
        """
        Runs speed-sweep MDO, locking geometry (L, tf, tr) to nominal values.
        """
        v_points = np.linspace(v_min, v_max, num_v)
        sweep_results = []

        # Lock geometry bounds
        fixed_bounds = list(custom_bounds)
        fixed_bounds[0] = (initial_guess[0], initial_guess[0])
        fixed_bounds[1] = (initial_guess[1], initial_guess[1])
        fixed_bounds[2] = (initial_guess[2], initial_guess[2])

        x_current = list(initial_guess)

        for v in v_points:
            self.v_x = float(v)
            # n_starts=2: warm start from previous node + one random recovery start
            res = self.run_optimization(initial_guess=x_current, custom_bounds=fixed_bounds, n_starts=2)
            
            if res.success or res.fun < 10.0:
                x_opt = list(res.x)
                x_current = x_opt
                max_ay = float(-res.fun if self.objective_mode == "max_ay" else self._evaluate_limit_ay(x_opt))
                
                row = {
                    'v_x': float(v),
                    'v_kmh': float(v * 3.6),
                    'max_ay_g': float(max_ay),
                    'L': float(x_opt[0]),
                    't_f': float(x_opt[1]),
                    't_r': float(x_opt[2]),
                    'wd_f': float(x_opt[3]),
                    'lltd': float(x_opt[4]),
                    'aero_bal': float(x_opt[5]),
                }
                if len(x_opt) >= 10:
                    row['camber_f_static'] = float(x_opt[6])
                    row['camber_r_static'] = float(x_opt[7])
                    row['toe_f_static'] = float(x_opt[8])
                    row['toe_r_static'] = float(x_opt[9])
                if len(x_opt) >= 14:
                    row['K_phi_f'] = float(x_opt[10])
                    row['K_phi_r'] = float(x_opt[11])
                    row['K_z_f']   = float(x_opt[12])
                    row['K_z_r']   = float(x_opt[13])
                    
                sweep_results.append(row)

        import pandas as pd
        return pd.DataFrame(sweep_results)
