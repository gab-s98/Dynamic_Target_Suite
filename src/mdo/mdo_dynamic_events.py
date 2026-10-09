"""
mdo_dynamic_events.py
----------------------
Dynamic Events & Maneuver Simulation Engine for FSAE Dynamic Target Suite.

Events & Maneuvers:
1. Step Steer / Transient Maneuver (7-DOF & 14-DOF time-domain ODE with closed-loop speed control & Torque Vectoring).
2. Double Lane Change (DLC / ISO 3888-2 Obstacle Avoidance with configurable rate & speed control).
3. FSAE AutoX Simulation (Single lap sprint with speed-dependent aerodynamic cornering envelope & energy balance).
4. FSAE Endurance Simulation (22km multi-lap performance + battery consumption + regen).
"""

import os
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import mf61_lateral_force
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_skidpad import simulate_skidpad
from src.mdo.mdo_acceleration import simulate_acceleration
from src.core.simulation_options import get_simulation_options


def rk4_step(f, x, u, p, dt):
    """Classic 4th-order Runge-Kutta numerical integration step."""
    k1 = np.array(f(x, u, p)).flatten()
    k2 = np.array(f(x + k1 * dt / 2.0, u, p)).flatten()
    k3 = np.array(f(x + k2 * dt / 2.0, u, p)).flatten()
    k4 = np.array(f(x + k3 * dt, u, p)).flatten()
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_step_steer(params: VehicleParams, v_x=15.0, steer_deg=3.0, t_end=None, dt=None,
                         t_step=0.5, t_ramp=None, torque_vectoring_nm=0.0,
                         maintain_speed=None, model_type="7-DOF", tir_path=None):
    """
    Simulates a Step Steer transient response using 7-DOF or 14-DOF ODE.
    
    Parameters:
    - params: VehicleParams instance
    - v_x: Target entry velocity [m/s]
    - steer_deg: Amplitude of steering angle at wheels [deg]
    - t_end: Maneuver total duration [s] (default from SimulationOptions)
    - dt: Discretization time step [s] (default from SimulationOptions)
    - t_step: Time when steering starts [s]
    - t_ramp: Steering rise time / ramp duration [s] (default from SimulationOptions)
    - torque_vectoring_nm: Differential torque applied to rear wheels [Nm]
    - maintain_speed: If True, applies closed-loop drive torque to hold v_x against aero/tire drag
    - model_type: "7-DOF" or "14-DOF"
    - tir_path: Path to .tir tire file
    """
    opts = get_simulation_options()
    dt = dt if dt is not None else getattr(opts, 'dt_ode', 0.002)
    t_end = t_end if t_end is not None else getattr(opts, 't_end_step', 4.0)
    t_ramp = t_ramp if t_ramp is not None else getattr(opts, 't_ramp_steer', 0.08)
    maintain_speed = maintain_speed if maintain_speed is not None else getattr(opts, 'maintain_speed', True)

    tp = TireParams(tir_path)

    N = max(int(t_end / dt), 10)
    t_arr = np.linspace(0.0, t_end, N)
    w0 = v_x / params.R_w

    if model_type == "14-DOF":
        veh = Vehicle14DOF(params, tp, tp)
        ode = veh.get_casadi_ode()
        # States (28): [vx, vy, r, phi, phi_dot, theta, theta_dot, w_heave, z_fl..rr, vz_fl..rr, w_fl..rr, ux_fl..uy_rr]
        x0 = np.array([
            v_x, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            w0, w0, w0, w0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        p = np.array([
            params.wd_f, params.lltd, params.K_phi_f, params.K_phi_r,
            params.K_z_f, params.K_z_r, params.aero_balance,
            params.camber_f_static, params.camber_r_static,
            params.toe_f_static, params.toe_r_static,
            params.L, params.t_f, params.t_r, params.LMUY, params.LMUX,
            params.Cd
        ])
    else:
        veh = Vehicle7DOF(params, tp, tp)
        ode = veh.get_casadi_ode()
        # States (7): [vx, vy, r, w_fl, w_fr, w_rl, w_rr]
        x0 = np.array([v_x, 0.0, 0.0, w0, w0, w0, w0])
        p = np.array([
            params.wd_f, params.lltd, params.LMUX, params.LMUY,
            params.L, params.t_f, params.t_r,
            params.camber_f_static, params.camber_r_static,
            params.toe_f_static, params.toe_r_static
        ])

    x_arr = np.zeros((N, len(x0)))
    x_arr[0] = x0
    u_hist = np.zeros((N, 4))
    
    # Speed controller (PI) parameters
    integral_err = 0.0
    kp_speed = 350.0  # N / (m/s)
    ki_speed = 80.0   # N / (m/s * s)
    pt = getattr(params, 'get_powertrain', lambda: None)()
    if pt is None:
        from src.models.powertrain_electric import get_default_powertrain
        pt = get_default_powertrain()

    for i in range(1, N):
        t = t_arr[i]
        
        # Steering input with realistic ramp
        if t < t_step:
            d_val = 0.0
        elif t_ramp > 1e-4:
            ramp_frac = min((t - t_step) / t_ramp, 1.0)
            d_val = np.deg2rad(steer_deg) * ramp_frac
        else:
            d_val = np.deg2rad(steer_deg)
        
        # Speed Maintenance Controller (closed-loop drive torque on rear wheels)
        vx_curr = x_arr[i-1, 0]
        if maintain_speed:
            f_drag = 0.5 * params.rho * params.S * params.Cd * (vx_curr**2)
            f_roll = 0.015 * params.m * 9.81
            e_v = v_x - vx_curr
            integral_err += e_v * dt
            f_drive_req = max(f_drag + f_roll + kp_speed * e_v + ki_speed * integral_err, 0.0)
            t_base_wheel = (f_drive_req * params.R_w) / 2.0
        else:
            t_base_wheel = 0.0

        # Torque Vectoring: outer wheel +TV, inner wheel -TV
        if t >= t_step and abs(torque_vectoring_nm) > 1e-3:
            # Positive steer (left turn): outer wheel is Right (RR), inner wheel is Left (RL)
            sign_tv = 1.0 if steer_deg >= 0 else -1.0
            t_rl_req = t_base_wheel - sign_tv * (torque_vectoring_nm / 2.0)
            t_rr_req = t_base_wheel + sign_tv * (torque_vectoring_nm / 2.0)
        else:
            t_rl_req = t_base_wheel
            t_rr_req = t_base_wheel

        w_rl = x_arr[i-1, 5] if len(x0) == 7 else x_arr[i-1, 18]
        w_rr = x_arr[i-1, 6] if len(x0) == 7 else x_arr[i-1, 19]
        t_rl, t_rr, _ = pt.allocate_dual_motor_torques(w_rl, w_rr, t_rl_req, t_rr_req)
            
        u = np.array([d_val, t_rl, t_rr, 0.0])
        u_hist[i] = u
        x_arr[i] = rk4_step(ode, x_arr[i-1], u, p, dt)

    records = []
    a_dist = params.L * (1.0 - params.wd_f)
    b_dist = params.L * params.wd_f
    tf = max(params.t_f, 0.5)
    tr = max(params.t_r, 0.5)
    mu_nom = getattr(params, 'LMUY', 1.6)

    cam_f_val = getattr(params, 'camber_f_static', -1.5)
    cam_r_val = getattr(params, 'camber_r_static', -1.0)
    is_right = getattr(tp, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if tp else True
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

    lvy_fl = -1.0 if is_right else 1.0
    lvy_fr =  1.0 if is_right else -1.0
    lvy_rl = -1.0 if is_right else 1.0
    lvy_rr =  1.0 if is_right else -1.0

    lhy_fl = -1.0 if is_right else 1.0
    lhy_fr =  1.0 if is_right else -1.0
    lhy_rl = -1.0 if is_right else 1.0
    lhy_rr =  1.0 if is_right else -1.0

    for i in range(N):
        t = t_arr[i]
        vx = x_arr[i, 0]
        vy = x_arr[i, 1]
        r = x_arr[i, 2]
        r_prev = x_arr[i-1, 2] if i > 0 else r
        r_dot_deg_s2 = float(np.rad2deg((r - r_prev) / dt))
        
        ay_ms2 = vx * r + (vy - x_arr[i-1, 1])/dt if i > 0 else (vx * r)
        beta_rad = vy / max(abs(vx), 0.5)
        d_rad = u_hist[i, 0]
        
        # Aerodynamic downforce & drag
        f_down_curr = 0.5 * params.rho * params.S * params.Cz * (vx**2)
        f_drag_curr = 0.5 * params.rho * params.S * params.Cd * (vx**2)
        
        # 4-Wheel Slip Angles [deg]
        alpha_fl = float(np.rad2deg(d_rad - np.arctan2(vy + a_dist * r, max(vx - 0.5 * tf * r, 0.5))))
        alpha_fr = float(np.rad2deg(d_rad - np.arctan2(vy + a_dist * r, max(vx + 0.5 * tf * r, 0.5))))
        alpha_rl = float(np.rad2deg(-np.arctan2(vy - b_dist * r, max(vx - 0.5 * tr * r, 0.5))))
        alpha_rr = float(np.rad2deg(-np.arctan2(vy - b_dist * r, max(vx + 0.5 * tr * r, 0.5))))
        
        # 4-Wheel Dynamic Loads [N]
        if model_type == "14-DOF":
            phi_curr = x_arr[i, 3]
            phi_curr_deg = float(np.rad2deg(phi_curr))
            z_fl_c = x_arr[i, 8]
            z_fr_c = x_arr[i, 9]
            z_rl_c = x_arr[i, 10]
            z_rr_c = x_arr[i, 11]
            h_rc_f_curr = params.eval_poly3(params.rollcenter_f_coeffs, phi_curr_deg)
            h_rc_r_curr = params.eval_poly3(params.rollcenter_r_coeffs, phi_curr_deg)
            dFz_f_llt = (params.K_phi_f * phi_curr) / tf + (params.m * ay_ms2 * (b_dist / params.L) * h_rc_f_curr) / tf
            dFz_r_llt = (params.K_phi_r * phi_curr) / tr + (params.m * ay_ms2 * (a_dist / params.L) * h_rc_r_curr) / tr
            Fz_fl = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 - dFz_f_llt, 20.0)
            Fz_fr = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 + dFz_f_llt, 20.0)
            Fz_rl = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 - dFz_r_llt, 20.0)
            Fz_rr = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 + dFz_r_llt, 20.0)
        else:
            dFz_lat_f = (params.m * ay_ms2 * params.h_cg * params.lltd) / tf
            dFz_lat_r = (params.m * ay_ms2 * params.h_cg * (1.0 - params.lltd)) / tr
            Fz_fl = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 - dFz_lat_f / 2.0, 20.0)
            Fz_fr = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 + dFz_lat_f / 2.0, 20.0)
            Fz_rl = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 - dFz_lat_r / 2.0, 20.0)
            Fz_rr = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 + dFz_lat_r / 2.0, 20.0)
        
        # 4-Wheel Lateral Tire Forces [N]
        Fy_fl = -float(mf61_lateral_force(np.deg2rad(alpha_fl), Fz_fl, gamma_fl, tp, LMUY=params.LMUY, LVY=lvy_fl, LHY=lhy_fl))
        Fy_fr = -float(mf61_lateral_force(np.deg2rad(alpha_fr), Fz_fr, gamma_fr, tp, LMUY=params.LMUY, LVY=lvy_fr, LHY=lhy_fr))
        Fy_rl = -float(mf61_lateral_force(np.deg2rad(alpha_rl), Fz_rl, gamma_rl, tp, LMUY=params.LMUY, LVY=lvy_rl, LHY=lhy_rl))
        Fy_rr = -float(mf61_lateral_force(np.deg2rad(alpha_rr), Fz_rr, gamma_rr, tp, LMUY=params.LMUY, LVY=lvy_rr, LHY=lhy_rr))
        
        # Friction utilization [%]
        util_fl = float(min((abs(Fy_fl) / max(mu_nom * Fz_fl, 1.0)) * 100.0, 100.0))
        util_fr = float(min((abs(Fy_fr) / max(mu_nom * Fz_fr, 1.0)) * 100.0, 100.0))
        util_rl = float(min((abs(Fy_rl) / max(mu_nom * Fz_rl, 1.0)) * 100.0, 100.0))
        util_rr = float(min((abs(Fy_rr) / max(mu_nom * Fz_rr, 1.0)) * 100.0, 100.0))
        
        row = {
            't': float(t),
            'v_x': float(vx),
            'v_y': float(vy),
            'v_kmh': float(vx * 3.6),
            'yaw_rate_deg_s': float(np.rad2deg(r)),
            'r_dot_deg_s2': r_dot_deg_s2,
            'ay_g': float(ay_ms2 / 9.81),
            'beta_deg': float(np.rad2deg(beta_rad)),
            'steer_deg': float(np.rad2deg(u_hist[i, 0])),
            'T_drive_rl_Nm': float(u_hist[i, 1]),
            'T_drive_rr_Nm': float(u_hist[i, 2]),
            'TV_torque_Nm': float(u_hist[i, 2] - u_hist[i, 1]),
            'F_downforce_N': float(f_down_curr),
            'F_drag_N': float(f_drag_curr),
            # 4-wheel slip angles
            'alpha_fl_deg': alpha_fl,
            'alpha_fr_deg': alpha_fr,
            'alpha_rl_deg': alpha_rl,
            'alpha_rr_deg': alpha_rr,
            # 4-wheel normal loads
            'Fz_fl_N': float(Fz_fl),
            'Fz_fr_N': float(Fz_fr),
            'Fz_rl_N': float(Fz_rl),
            'Fz_rr_N': float(Fz_rr),
            # 4-wheel lateral forces
            'Fy_fl_N': Fy_fl,
            'Fy_fr_N': Fy_fr,
            'Fy_rl_N': Fy_rl,
            'Fy_rr_N': Fy_rr,
            # Friction utilization
            'util_fl_pct': util_fl,
            'util_fr_pct': util_fr,
            'util_rl_pct': util_rl,
            'util_rr_pct': util_rr
        }
        
        if model_type == "14-DOF":
            row['phi_deg'] = float(np.rad2deg(x_arr[i, 3]))
            row['phi_dot_deg_s'] = float(np.rad2deg(x_arr[i, 4]))
            row['theta_deg'] = float(np.rad2deg(x_arr[i, 5]))
            row['z_fl_mm'] = float(x_arr[i, 7] * 1000.0)
            row['z_fr_mm'] = float(x_arr[i, 8] * 1000.0)
            row['z_rl_mm'] = float(x_arr[i, 9] * 1000.0)
            row['z_rr_mm'] = float(x_arr[i, 10] * 1000.0)
            row['w_fl'] = float(x_arr[i, 11])
            row['w_fr'] = float(x_arr[i, 12])
            row['w_rl'] = float(x_arr[i, 13])
            row['w_rr'] = float(x_arr[i, 14])
        else:
            row['phi_deg'] = 0.0
            row['w_fl'] = float(x_arr[i, 3])
            row['w_fr'] = float(x_arr[i, 4])
            row['w_rl'] = float(x_arr[i, 5])
            row['w_rr'] = float(x_arr[i, 6])
            
        records.append(row)

    df_res = pd.DataFrame(records)
    
    # Calculate response metrics
    post_step = df_res[df_res['t'] >= t_step + t_ramp]
    if not post_step.empty:
        r_ss = float(post_step['yaw_rate_deg_s'].iloc[-max(int(len(post_step)*0.2), 5):].mean())
        r_peak = float(post_step['yaw_rate_deg_s'].max())
        overshoot_pct = max(((r_peak - r_ss) / max(abs(r_ss), 1e-3)) * 100.0, 0.0)
        ay_peak_g = float(post_step['ay_g'].max())
    else:
        r_ss, r_peak, overshoot_pct, ay_peak_g = 0.0, 0.0, 0.0, 0.0

    return {
        'success': True,
        'telemetry': df_res,
        'r_ss_deg_s': r_ss,
        'r_peak_deg_s': r_peak,
        'overshoot_pct': overshoot_pct,
        'ay_peak_g': ay_peak_g,
        'dt_used': dt,
        't_end_used': t_end,
        'maintain_speed': maintain_speed
    }


def simulate_double_lane_change(params: VehicleParams, v_entry=20.0, t_end=None, dt=None,
                                maintain_speed=None, model_type="14-DOF", tir_path=None):
    """
    Simulates standard ISO 3888-2 Double Lane Change obstacle avoidance maneuver.
    """
    opts = get_simulation_options()
    dt = dt if dt is not None else getattr(opts, 'dt_ode', 0.002)
    t_end = t_end if t_end is not None else getattr(opts, 't_end_dlc', 5.0)
    maintain_speed = maintain_speed if maintain_speed is not None else getattr(opts, 'maintain_speed', True)

    tp = TireParams(tir_path)

    N = max(int(t_end / dt), 10)
    t_arr = np.linspace(0.0, t_end, N)
    w0 = v_entry / params.R_w

    if model_type == "14-DOF":
        veh = Vehicle14DOF(params, tp, tp)
        ode = veh.get_casadi_ode()
        x0 = np.array([
            v_entry, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            w0, w0, w0, w0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        p = np.array([
            params.wd_f, params.lltd, params.K_phi_f, params.K_phi_r,
            params.K_z_f, params.K_z_r, params.aero_balance,
            params.camber_f_static, params.camber_r_static,
            params.toe_f_static, params.toe_r_static,
            params.L, params.t_f, params.t_r, params.LMUY, params.LMUX,
            params.Cd
        ])
    else:
        veh = Vehicle7DOF(params, tp, tp)
        ode = veh.get_casadi_ode()
        x0 = np.array([v_entry, 0.0, 0.0, w0, w0, w0, w0])
        p = np.array([
            params.wd_f, params.lltd, params.LMUX, params.LMUY,
            params.L, params.t_f, params.t_r,
            params.camber_f_static, params.camber_r_static,
            params.toe_f_static, params.toe_r_static
        ])

    # Standard ISO DLC Steer Pulse (Full cycle sinusoid into counter-steer)
    steer_amp = min(2.5, 32.0 / max(v_entry, 5.0))
    steer_profile = np.zeros(N)
    for i, t in enumerate(t_arr):
        if 0.5 <= t < 1.7:
            tau = (t - 0.5) / 1.2
            steer_profile[i] = steer_amp * np.sin(2.0 * np.pi * tau)
        elif 1.8 <= t < 3.0:
            tau = (t - 1.8) / 1.2
            steer_profile[i] = -steer_amp * np.sin(2.0 * np.pi * tau)
        else:
            steer_profile[i] = 0.0

    x_arr = np.zeros((N, len(x0)))
    x_arr[0] = x0
    
    # Trajectory integration
    x_pos = 0.0
    y_pos = 0.0
    psi_pos = 0.0
    pos_records = []
    
    kp_speed = 300.0
    pt = getattr(params, 'get_powertrain', lambda: None)()
    if pt is None:
        from src.models.powertrain_electric import get_default_powertrain
        pt = get_default_powertrain()

    for i in range(1, N):
        d_val = np.deg2rad(steer_profile[i])
        vx_curr = x_arr[i-1, 0]
        
        if maintain_speed:
            f_drag = 0.5 * params.rho * params.S * params.Cd * (vx_curr**2)
            f_roll = 0.015 * params.m * 9.81
            e_v = v_entry - vx_curr
            f_drive_req = max(f_drag + f_roll + kp_speed * e_v, 0.0)
            t_base_wheel = (f_drive_req * params.R_w) / 2.0
        else:
            t_base_wheel = 0.0
            
        w_rl = x_arr[i-1, 5] if len(x0) == 7 else x_arr[i-1, 18]
        w_rr = x_arr[i-1, 6] if len(x0) == 7 else x_arr[i-1, 19]
        t_rl, t_rr, _ = pt.allocate_dual_motor_torques(w_rl, w_rr, t_base_wheel, t_base_wheel)

        u = np.array([d_val, t_rl, t_rr, 0.0])
        x_arr[i] = rk4_step(ode, x_arr[i-1], u, p, dt)
        
        vx = x_arr[i, 0]
        vy = x_arr[i, 1]
        r = x_arr[i, 2]
        
        psi_pos += r * dt
        x_pos += (vx * np.cos(psi_pos) - vy * np.sin(psi_pos)) * dt
        y_pos += (vx * np.sin(psi_pos) + vy * np.cos(psi_pos)) * dt
        
        pos_records.append((x_pos, y_pos, psi_pos))

    records = []
    a_dist = params.L * (1.0 - params.wd_f)
    b_dist = params.L * params.wd_f
    tf = max(params.t_f, 0.5)
    tr = max(params.t_r, 0.5)
    mu_nom = getattr(params, 'LMUY', 1.6)

    cam_f_val = getattr(params, 'camber_f_static', -1.5)
    cam_r_val = getattr(params, 'camber_r_static', -1.0)
    is_right = getattr(tp, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if tp else True
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

    lvy_fl = -1.0 if is_right else 1.0
    lvy_fr =  1.0 if is_right else -1.0
    lvy_rl = -1.0 if is_right else 1.0
    lvy_rr =  1.0 if is_right else -1.0

    lhy_fl = -1.0 if is_right else 1.0
    lhy_fr =  1.0 if is_right else -1.0
    lhy_rl = -1.0 if is_right else 1.0
    lhy_rr =  1.0 if is_right else -1.0

    for i in range(N):
        t = t_arr[i]
        vx = x_arr[i, 0]
        vy = x_arr[i, 1]
        r = x_arr[i, 2]
        r_prev = x_arr[i-1, 2] if i > 0 else r
        r_dot_deg_s2 = float(np.rad2deg((r - r_prev) / dt))
        ay_ms2 = vx * r + (vy - x_arr[i-1, 1])/dt if i > 0 else (vx * r)
        
        xp = pos_records[i-1][0] if i > 0 else 0.0
        yp = pos_records[i-1][1] if i > 0 else 0.0
        psip = pos_records[i-1][2] if i > 0 else 0.0
        d_rad = np.deg2rad(steer_profile[i])
        
        f_down_curr = 0.5 * params.rho * params.S * params.Cz * (vx**2)
        f_drag_curr = 0.5 * params.rho * params.S * params.Cd * (vx**2)
        
        # 4-Wheel Slip Angles [deg]
        alpha_fl = float(np.rad2deg(d_rad - np.arctan2(vy + a_dist * r, max(vx - 0.5 * tf * r, 0.5))))
        alpha_fr = float(np.rad2deg(d_rad - np.arctan2(vy + a_dist * r, max(vx + 0.5 * tf * r, 0.5))))
        alpha_rl = float(np.rad2deg(-np.arctan2(vy - b_dist * r, max(vx - 0.5 * tr * r, 0.5))))
        alpha_rr = float(np.rad2deg(-np.arctan2(vy - b_dist * r, max(vx + 0.5 * tr * r, 0.5))))
        
        # 4-Wheel Dynamic Loads [N]
        if model_type == "14-DOF":
            phi_curr = x_arr[i, 3]
            phi_curr_deg = float(np.rad2deg(phi_curr))
            z_fl_c = x_arr[i, 8]
            z_fr_c = x_arr[i, 9]
            z_rl_c = x_arr[i, 10]
            z_rr_c = x_arr[i, 11]
            h_rc_f_curr = params.eval_poly3(params.rollcenter_f_coeffs, phi_curr_deg)
            h_rc_r_curr = params.eval_poly3(params.rollcenter_r_coeffs, phi_curr_deg)
            dFz_f_llt = (params.K_phi_f * phi_curr) / tf + (params.m * ay_ms2 * (b_dist / params.L) * h_rc_f_curr) / tf
            dFz_r_llt = (params.K_phi_r * phi_curr) / tr + (params.m * ay_ms2 * (a_dist / params.L) * h_rc_r_curr) / tr
            Fz_fl = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 - dFz_f_llt, 20.0)
            Fz_fr = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 + dFz_f_llt, 20.0)
            Fz_rl = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 - dFz_r_llt, 20.0)
            Fz_rr = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 + dFz_r_llt, 20.0)
        else:
            dFz_lat_f = (params.m * ay_ms2 * params.h_cg * params.lltd) / tf
            dFz_lat_r = (params.m * ay_ms2 * params.h_cg * (1.0 - params.lltd)) / tr
            Fz_fl = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 - dFz_lat_f / 2.0, 20.0)
            Fz_fr = max((params.m * 9.81 * params.wd_f + f_down_curr * params.aero_balance) / 2.0 + dFz_lat_f / 2.0, 20.0)
            Fz_rl = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 - dFz_lat_r / 2.0, 20.0)
            Fz_rr = max((params.m * 9.81 * (1.0 - params.wd_f) + f_down_curr * (1.0 - params.aero_balance)) / 2.0 + dFz_lat_r / 2.0, 20.0)
        
        # 4-Wheel Lateral Tire Forces [N]
        Fy_fl = -float(mf61_lateral_force(np.deg2rad(alpha_fl), Fz_fl, gamma_fl, tp, LMUY=params.LMUY, LVY=lvy_fl, LHY=lhy_fl))
        Fy_fr = -float(mf61_lateral_force(np.deg2rad(alpha_fr), Fz_fr, gamma_fr, tp, LMUY=params.LMUY, LVY=lvy_fr, LHY=lhy_fr))
        Fy_rl = -float(mf61_lateral_force(np.deg2rad(alpha_rl), Fz_rl, gamma_rl, tp, LMUY=params.LMUY, LVY=lvy_rl, LHY=lhy_rl))
        Fy_rr = -float(mf61_lateral_force(np.deg2rad(alpha_rr), Fz_rr, gamma_rr, tp, LMUY=params.LMUY, LVY=lvy_rr, LHY=lhy_rr))
        
        # Friction utilization [%]
        util_fl = float(min((abs(Fy_fl) / max(mu_nom * Fz_fl, 1.0)) * 100.0, 100.0))
        util_fr = float(min((abs(Fy_fr) / max(mu_nom * Fz_fr, 1.0)) * 100.0, 100.0))
        util_rl = float(min((abs(Fy_rl) / max(mu_nom * Fz_rl, 1.0)) * 100.0, 100.0))
        util_rr = float(min((abs(Fy_rr) / max(mu_nom * Fz_rr, 1.0)) * 100.0, 100.0))
        
        row = {
            't': float(t),
            'v_x': float(vx),
            'v_kmh': float(vx * 3.6),
            'yaw_rate_deg_s': float(np.rad2deg(r)),
            'r_dot_deg_s2': r_dot_deg_s2,
            'ay_g': float(ay_ms2 / 9.81),
            'beta_deg': float(np.rad2deg(vy / max(abs(vx), 0.5))),
            'steer_deg': float(steer_profile[i]),
            'x_pos_m': float(xp),
            'y_pos_m': float(yp),
            'psi_deg': float(np.rad2deg(psip)),
            'F_downforce_N': float(f_down_curr),
            'F_drag_N': float(f_drag_curr),
            # 4-wheel slip angles
            'alpha_fl_deg': alpha_fl,
            'alpha_fr_deg': alpha_fr,
            'alpha_rl_deg': alpha_rl,
            'alpha_rr_deg': alpha_rr,
            # 4-wheel normal loads
            'Fz_fl_N': float(Fz_fl),
            'Fz_fr_N': float(Fz_fr),
            'Fz_rl_N': float(Fz_rl),
            'Fz_rr_N': float(Fz_rr),
            # 4-wheel lateral forces
            'Fy_fl_N': Fy_fl,
            'Fy_fr_N': Fy_fr,
            'Fy_rl_N': Fy_rl,
            'Fy_rr_N': Fy_rr,
            # Friction utilization
            'util_fl_pct': util_fl,
            'util_fr_pct': util_fr,
            'util_rl_pct': util_rl,
            'util_rr_pct': util_rr
        }
        if model_type == "14-DOF":
            row['phi_deg'] = float(np.rad2deg(x_arr[i, 3]))
            row['phi_dot_deg_s'] = float(np.rad2deg(x_arr[i, 4]))
            row['theta_deg'] = float(np.rad2deg(x_arr[i, 5]))
            row['z_fl_mm'] = float(x_arr[i, 7] * 1000.0)
            row['z_fr_mm'] = float(x_arr[i, 8] * 1000.0)
            row['z_rl_mm'] = float(x_arr[i, 9] * 1000.0)
            row['z_rr_mm'] = float(x_arr[i, 10] * 1000.0)
            row['w_fl'] = float(x_arr[i, 11])
            row['w_fr'] = float(x_arr[i, 12])
            row['w_rl'] = float(x_arr[i, 13])
            row['w_rr'] = float(x_arr[i, 14])
        else:
            row['phi_deg'] = 0.0
            row['w_fl'] = float(x_arr[i, 3])
            row['w_fr'] = float(x_arr[i, 4])
            row['w_rl'] = float(x_arr[i, 5])
            row['w_rr'] = float(x_arr[i, 6])
        records.append(row)

    df_res = pd.DataFrame(records)
    max_ay_g = float(df_res['ay_g'].abs().max())
    max_phi_deg = float(df_res['phi_deg'].abs().max())
    max_lat_dev = float(df_res['y_pos_m'].abs().max())

    return {
        'success': True,
        'telemetry': df_res,
        'max_ay_g': max_ay_g,
        'max_phi_deg': max_phi_deg,
        'max_lateral_disp_m': max_lat_dev,
        'cone_clearance_ok': (max_lat_dev >= 2.5 and max_lat_dev <= 4.2),
        'dt_used': dt,
        't_end_used': t_end
    }


def simulate_autox_endurance(params: VehicleParams, event_type="AutoX", num_laps=1,
                             track_length_m=850.0, ds=None, model_type="14-DOF", tir_path=None,
                             track_dict=None):
    """
    Simulates complete FSAE AutoX or Endurance event with speed-dependent
    aerodynamic cornering limits, computing lap time, speed profile,
    electrical energy consumption (kWh), regen recovery, and efficiency (kWh/km).
    Supports custom or preset track geometries via track_dict.
    """
    opts = get_simulation_options()
    ds = ds if ds is not None else getattr(opts, 'ds_track', 0.5)

    tp = TireParams(tir_path)

    # 1. Base mechanical parameters
    acc = simulate_acceleration(params, distance=75.0)
    ax_acc_max_g = acc['a_max_g'] if acc['success'] else 1.25
    ax_brk_max_g = 1.45 * getattr(params, 'LMUX', 1.0)
    base_ay_mech = 1.32 * getattr(params, 'LMUY', 1.0)

    # 2. Track Layout (either from track_dict or default FSAE AutoX layout)
    if track_dict is not None:
        s_arr = track_dict['s']
        ds = float(track_dict.get('ds', np.mean(np.diff(s_arr)) if len(s_arr) > 1 else 0.5))
        kappa_arr = track_dict['kappa']
        n_track = len(s_arr)
        r_grid = [1.0 / abs(k) if abs(k) > 1e-4 else float('inf') for k in kappa_arr]
        dir_grid = [int(np.sign(k)) for k in kappa_arr]
        x_pos = track_dict.get('x_center', np.zeros(n_track))
        y_pos = track_dict.get('y_center', np.zeros(n_track))
        heading = track_dict.get('theta', np.zeros(n_track))
    else:
        # Realistic FSAE AutoX Track layout (Slalom, Chicanes, Decreasing Hairpins, Fast Sweepers)
        segments = [
            (60.0, float('inf'),  0, "Rettilineo Partenza"),
            (10.5,  8.0,          1, "Slalom Cono 1 (Sx)"),
            (10.5,  8.0,         -1, "Slalom Cono 2 (Dx)"),
            (10.5,  8.0,          1, "Slalom Cono 3 (Sx)"),
            (10.5,  8.0,         -1, "Slalom Cono 4 (Dx)"),
            (10.5,  8.0,          1, "Slalom Cono 5 (Sx)"),
            (10.5,  8.0,         -1, "Slalom Cono 6 (Dx)"),
            (35.0, float('inf'),  0, "Allungo Uscita Slalom"),
            (25.0, 14.0,         -1, "Tornante Ingresso (Dx)"),
            (35.0,  7.5,         -1, "Tornante Corda Stretta (Dx)"),
            (55.0, float('inf'),  0, "Rettilineo Intermedio 1"),
            (22.0,  9.5,          1, "Chicane Ingresso (Sx)"),
            (22.0,  9.5,         -1, "Chicane Uscita (Dx)"),
            (45.0, float('inf'),  0, "Allungo Pre-Curvone"),
            (65.0, 26.0,          1, "Curvone Veloce Aero (Sx)"),
            (30.0, 12.0,         -1, "Esse Tecnica 1 (Dx)"),
            (30.0, 12.0,          1, "Esse Tecnica 2 (Sx)"),
            (45.0,  9.125,       -1, "Curva Skidpad Stretta (Dx)"),
            (60.0, float('inf'),  0, "Rettilineo Finale / Arrivo")
        ]
        
        tot_seg_len = sum(s[0] for s in segments)
        scale_fac = track_length_m / tot_seg_len
        
        s_grid = []
        r_grid = []
        dir_grid = []
        for length, r_curv, turn_dir, _ in segments:
            l_scaled = length * scale_fac
            n_pts = max(int(l_scaled / ds), 1)
            s_grid.extend([ds] * n_pts)
            r_grid.extend([r_curv] * n_pts)
            dir_grid.extend([turn_dir] * n_pts)
            
        s_arr = np.cumsum(s_grid) - ds
        n_track = len(s_arr)

        x_pos = np.zeros(n_track)
        y_pos = np.zeros(n_track)
        heading = np.zeros(n_track)
        
        for i in range(1, n_track):
            rad = r_grid[i]
            d_sign = dir_grid[i]
            kappa = (1.0 / rad) * d_sign if rad != float('inf') else 0.0
            heading[i] = heading[i-1] + kappa * ds
            x_pos[i] = x_pos[i-1] + np.cos(heading[i]) * ds
            y_pos[i] = y_pos[i-1] + np.sin(heading[i]) * ds

    # 3. Dynamic Speed-Dependent Cornering Limits with Aerodynamic Downforce
    # For each corner radius R, solve: v = sqrt(ay(v) * R)
    # where ay(v) = (m*g + 0.5*rho*S*Cz*v^2) * mu_eff / m
    v_corner_limit = np.zeros(n_track)
    v_top_straight = getattr(params, 'v_max_straight', 32.5) # 117 km/h
    rho = params.rho
    S_ref = params.S
    Cz_val = params.Cz
    m_veh = params.m

    for i in range(n_track):
        rad = r_grid[i]
        if rad == float('inf') or rad > 250.0:
            v_corner_limit[i] = v_top_straight
        else:
            # Iterative solution for speed with aero downforce
            v_iter = np.sqrt(base_ay_mech * 9.81 * rad)
            for _ in range(6):
                f_down = 0.5 * rho * S_ref * Cz_val * (v_iter**2)
                f_z_tot = m_veh * 9.81 + f_down
                # Non-linear tire load sensitivity factor
                load_factor = (f_z_tot / (m_veh * 9.81))
                mu_eff = (base_ay_mech / 1.0) * (1.0 - 0.08 * (f_down / (m_veh * 9.81)))
                ay_speed = (f_z_tot * mu_eff) / (m_veh * 9.81)
                v_iter = min(np.sqrt(ay_speed * 9.81 * rad), v_top_straight)
            v_corner_limit[i] = v_iter

    # 4. Forward Acceleration Pass
    v_fwd = np.zeros(n_track)
    v_fwd[0] = 8.0 # Entry speed
    pt = getattr(params, 'get_powertrain', lambda: None)()
    if pt is None:
        from src.models.powertrain_electric import get_default_powertrain
        pt = get_default_powertrain()

    for i in range(1, n_track):
        v_prev = v_fwd[i-1]
        F_drag = 0.5 * params.rho * params.S * params.Cd * (v_prev**2)
        P_max_w = getattr(params, 'P_max_kw', 80.0) * 1000.0
        w_w = max(v_prev / params.R_w, 0.05)
        t_rl_max, t_rr_max, _ = pt.allocate_dual_motor_torques(w_w, w_w, 800.0, 800.0, p_elec_limit_total_w=P_max_w)
        F_trac_motor = (t_rl_max + t_rr_max) / params.R_w
        F_grip = ax_acc_max_g * params.m * 9.81 * (1.0 - params.wd_f)
        F_net = min(F_trac_motor, F_grip) - F_drag
        ax_allow = max(F_net / params.m, 0.1)
        v_next_sq = v_prev**2 + 2.0 * ax_allow * ds
        v_fwd[i] = min(np.sqrt(max(v_next_sq, 0.1)), v_corner_limit[i])

    # 5. Backward Braking Pass
    v_bwd = np.zeros(n_track)
    v_bwd[-1] = v_fwd[-1]
    for i in range(n_track - 2, -1, -1):
        v_next = v_bwd[i+1]
        ax_brk = ax_brk_max_g * 9.81
        v_prev_sq = v_next**2 + 2.0 * ax_brk * ds
        v_bwd[i] = min(np.sqrt(v_prev_sq), v_fwd[i])

    v_profile = v_bwd

    # 6. Time & Power & Energy Integration
    dt_arr = ds / np.maximum(v_profile, 1.0)
    lap_time_s = float(np.sum(dt_arr))
    t_cum = np.cumsum(dt_arr)

    # Accelerations
    ax_profile = np.zeros(n_track)
    for i in range(n_track - 1):
        ax_profile[i] = (v_profile[i+1]**2 - v_profile[i]**2) / (2.0 * ds)
    ax_profile[-1] = ax_profile[-2]

    ay_profile = np.zeros(n_track)
    for i in range(n_track):
        rad = r_grid[i]
        ay_profile[i] = (v_profile[i]**2 / rad) / 9.81 if rad != float('inf') else 0.0

    # Electrical Power & Energy
    P_elec_kw_arr = np.zeros(n_track)
    motor_eff = getattr(params, 'motor_base_eff', 0.92) * getattr(params, 'inverter_eff', 0.96)
    regen_eff = getattr(params, 'regen_efficiency', 0.65)
    regen_max_kw = getattr(params, 'regen_max_kw', 40.0)

    for i in range(n_track):
        v = v_profile[i]
        ax = ax_profile[i]
        F_drag = 0.5 * params.rho * params.S * params.Cd * (v**2)
        F_roll = 0.015 * params.m * 9.81
        F_wheel = params.m * ax + F_drag + F_roll
        P_mech_kw = (F_wheel * v) / 1000.0

        if P_mech_kw >= 0.0:
            P_elec_kw_arr[i] = P_mech_kw / max(motor_eff, 0.5)
        else:
            P_regen_kw = abs(P_mech_kw) * regen_eff
            P_elec_kw_arr[i] = -min(P_regen_kw, regen_max_kw)

    # Energy per lap
    energy_lap_kj = float(np.sum(P_elec_kw_arr * dt_arr))
    energy_lap_kwh = energy_lap_kj / 3600.0
    
    # Total Endurance Calculation
    total_laps = num_laps if event_type == "AutoX" else 22
    total_time_s = lap_time_s * total_laps
    total_energy_kwh = energy_lap_kwh * total_laps
    total_distance_km = (track_length_m * total_laps) / 1000.0
    efficiency_kwh_per_km = total_energy_kwh / max(total_distance_km, 0.1)

    # Telemetry DataFrame (1 Lap)
    records = []
    for i in range(n_track):
        records.append({
            's_m': float(s_arr[i]),
            't_s': float(t_cum[i]),
            'x_m': float(x_pos[i]),
            'y_m': float(y_pos[i]),
            'heading_deg': float(np.rad2deg(heading[i])),
            'v_ms': float(v_profile[i]),
            'v_kmh': float(v_profile[i] * 3.6),
            'ax_g': float(ax_profile[i] / 9.81),
            'ay_g': float(ay_profile[i]),
            'downforce_N': float(0.5 * params.rho * params.S * params.Cz * (v_profile[i]**2)),
            'power_elec_kw': float(P_elec_kw_arr[i]),
            'energy_consumed_wh': float(np.sum(P_elec_kw_arr[:i+1] * dt_arr[:i+1]) / 3.6)
        })

    df_telemetry = pd.DataFrame(records)

    return {
        'success': True,
        'event_type': event_type,
        'total_laps': total_laps,
        'lap_time_s': float(lap_time_s),
        'total_time_s': float(total_time_s),
        'v_max_kmh': float(v_profile.max() * 3.6),
        'v_avg_kmh': float((track_length_m / lap_time_s) * 3.6),
        'max_ay_g': float(ay_profile.max()),
        'energy_lap_kwh': float(energy_lap_kwh),
        'total_energy_kwh': float(total_energy_kwh),
        'efficiency_kwh_km': float(efficiency_kwh_per_km),
        'battery_soc_end_pct': max(100.0 - (total_energy_kwh / getattr(params, 'battery_capacity_kwh', 6.8)) * 100.0, 0.0),
        'ds_used': ds,
        'telemetry': df_telemetry
    }
