import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from src.models.params_vehicle import VehicleParams

def simulate_acceleration(params: VehicleParams, distance=75.0, is_awd=False, ay=0.0):
    """
    Simulates a straight-line acceleration event from standing start.
    Calculates dynamic quasi-static longitudinal load transfer:
      dFz_long = (m * ax * hcg) / L
    and dynamic lateral load transfer:
      dFz_lat = (m * ay * hcg) / track
    applying wd_f and LMUX directly to vertical load equilibrium [Fz_f, Fz_r] at each step ax.
    Returns:
    {
        'success': bool,
        'time': float,
        'v_max': float,
        'a_max_g': float,
        'telemetry': pd.DataFrame (t, s, v_ms, v_kmh, ax_g, ax_ms2, F_drive, F_motor, F_grip, F_drag, W_f, W_r, power_kw)
    }
    """
    m = float(params.m)
    g = float(params.g)
    L = float(params.L)
    h = float(params.h_cg)
    wd_f = float(getattr(params, 'wd_f', 0.50))
    rho = float(params.rho)
    S = float(params.S)
    Cz = float(params.Cz)
    Cd = float(getattr(params, 'Cd', 1.0))
    aero_bal = float(params.aero_balance)

    tf = float(getattr(params, 't_f', getattr(params, 'track_f', 1.25)))
    tr = float(getattr(params, 't_r', getattr(params, 'track_r', 1.20)))
    lltd = float(getattr(params, 'lltd', 0.50))
    ay_val = float(ay if ay is not None else getattr(params, 'ay', 0.0))
    
    P_max = getattr(params, 'P_max_kw', 80.0) * 1000.0
    T_max_single = getattr(params, 'T_max_motor', 100.0)
    final_drive = getattr(params, 'final_drive', 4.0)
    n_drivetrain = getattr(params, 'n_drivetrain', 0.95)
    R_w = float(params.R_w)

    pt = getattr(params, 'get_powertrain', lambda: None)()
    if pt is None:
        from src.models.powertrain_electric import get_default_powertrain
        pt = get_default_powertrain()
    
    # Dual-Motor powertrain architecture: on ET-18 RWD dual motor, each rear wheel
    # has an independent motor (n_motors = 2). AWD has 4 motors.
    diff_mode_str = str(getattr(params, 'diff_mode', getattr(params, 'diff_type', 'DUAL_MOTOR_OPEN'))).upper()
    is_dual = ('DUAL' in diff_mode_str) or (getattr(params, 'drivetrain_type', 'rwd') == 'rwd')
    n_motors = 4 if is_awd else (2 if is_dual else 1)
    
    # Peak longitudinal tire grip for Hoosier FSAE slick on competition asphalt scaled by LMUX
    lmux = float(getattr(params, 'LMUX', 1.0))
    mu_x = 1.62 * lmux
    
    # Rotating inertia equivalent mass (wheels + rotors)
    Iw = getattr(params, 'Iw', 1.0)
    m_rot_equiv = (4.0 * Iw) / (R_w**2)
    m_effective = m + m_rot_equiv * 0.40  # effective inertia factor
    
    def compute_forces(v):
        v = max(v, 0.0)
        F_aero_z = 0.5 * rho * S * Cz * v**2
        F_drag = 0.5 * rho * S * Cd * v**2
        
        # Max torque from EMRAX 188 motor curve & 80 kW accumulator electrical limit
        w_wheel = max(v / R_w, 0.05)
        t_req_single = 800.0  # full throttle torque demand at wheels
        t_w_rl, t_w_rr, telem = pt.allocate_dual_motor_torques(
            w_wheel, w_wheel, t_req_single, t_req_single, p_elec_limit_total_w=P_max
        )
        if is_awd:
            F_motor = 2.0 * (t_w_rl + t_w_rr) / R_w
        else:
            F_motor = (t_w_rl + t_w_rr) / R_w
        
        ax = F_motor / m_effective
        W_f, W_r, F_grip, F_drive = 0.0, 0.0, 0.0, 0.0
        dFz_long, dFz_lat_f, dFz_lat_r = 0.0, 0.0, 0.0
        Fz_fl, Fz_fr, Fz_rl, Fz_rr = 0.0, 0.0, 0.0, 0.0
        
        for _ in range(4):
            # Dynamic quasi-static longitudinal load transfer: dFz_long = (m * ax * hcg) / L
            dFz_long = (m * ax * h) / L
            W_f = max(m * g * wd_f - dFz_long + F_aero_z * aero_bal, 0.0)
            W_r = max(m * g * (1.0 - wd_f) + dFz_long + F_aero_z * (1.0 - aero_bal), 0.0)
            
            # Dynamic quasi-static lateral load transfer
            dFz_lat_f = (m * ay_val * h * lltd) / max(tf, 1e-3)
            dFz_lat_r = (m * ay_val * h * (1.0 - lltd)) / max(tr, 1e-3)
            
            Fz_fl = max(0.5 * W_f - 0.5 * dFz_lat_f, 0.0)
            Fz_fr = max(0.5 * W_f + 0.5 * dFz_lat_f, 0.0)
            Fz_rl = max(0.5 * W_r - 0.5 * dFz_lat_r, 0.0)
            Fz_rr = max(0.5 * W_r + 0.5 * dFz_lat_r, 0.0)
            
            if is_awd:
                F_grip = mu_x * (Fz_fl + Fz_fr + Fz_rl + Fz_rr)
            else:
                F_grip = mu_x * (Fz_rl + Fz_rr)
                
            F_drive = min(F_motor, F_grip)
            ax = (F_drive - F_drag) / m_effective
            
        return ax, F_drive, F_motor, F_grip, F_drag, W_f, W_r, telem, dFz_long, dFz_lat_f, dFz_lat_r, Fz_fl, Fz_fr, Fz_rl, Fz_rr
        
    def dynamics(t, y):
        s, v = y
        forces = compute_forces(v)
        ax = forces[0]
        return [v, ax]
        
    def hit_distance(t, y):
        return y[0] - distance
    hit_distance.terminal = True
    hit_distance.direction = 1

    y0 = [0.0, 0.1]
    t_span = (0.0, 12.0)
    
    sol = solve_ivp(dynamics, t_span, y0, events=hit_distance, max_step=0.02, method='RK45')
    
    if sol.status == 1:
        time_res = float(sol.t_events[0][0])
        v_max_res = float(sol.y_events[0][0][1])
        
        t_arr = sol.t
        s_arr = sol.y[0]
        v_arr = sol.y[1]
        
        # Ensure exact end point at distance
        if s_arr[-1] < distance - 1e-4 or s_arr[-1] > distance + 1e-4:
            t_arr = np.append(t_arr, time_res)
            s_arr = np.append(s_arr, distance)
            v_arr = np.append(v_arr, v_max_res)
            
        records = []
        for t_val, s_val, v_val in zip(t_arr, s_arr, v_arr):
            ax_val, f_drive, f_motor, f_grip, f_drag, w_f, w_r, telem, dfz_l, dfz_lat_f, dfz_lat_r, fz_fl, fz_fr, fz_rl, fz_rr = compute_forces(v_val)
            power_kw = (f_drive * v_val) / 1000.0
            
            t_wheel_tot = (f_drive * R_w)
            t_motor = (telem['t_motor_rl_nm'] + telem['t_motor_rr_nm']) / 2.0
            
            records.append({
                't': float(t_val),
                's': float(s_val),
                'v_ms': float(v_val),
                'v_kmh': float(v_val * 3.6),
                'ax_g': float(ax_val / g),
                'ax_ms2': float(ax_val),
                'F_drive': float(f_drive),
                'F_motor': float(f_motor),
                'F_grip': float(f_grip),
                'F_drag': float(f_drag),
                'W_f': float(w_f),
                'W_r': float(w_r),
                'dFz_long': float(dfz_l),
                'dFz_lat_f': float(dfz_lat_f),
                'dFz_lat_r': float(dfz_lat_r),
                'Fz_fl': float(fz_fl),
                'Fz_fr': float(fz_fr),
                'Fz_rl': float(fz_rl),
                'Fz_rr': float(fz_rr),
                'Fx_rl': float(f_drive / 2.0),
                'Fx_rr': float(f_drive / 2.0),
                'Torque_motor_Nm': float(t_motor),
                'Torque_wheel_total_Nm': float(t_wheel_tot),
                'Torque_rl_Nm': float(t_wheel_tot / 2.0),
                'Torque_rr_Nm': float(t_wheel_tot / 2.0),
                'power_kw': float(power_kw),
                'p_mech_kw': float(telem['p_mech_total_kw']),
                'p_elec_kw': float(telem['p_elec_total_kw']),
                'eta_motor': float((telem['eta_rl'] + telem['eta_rr']) / 2.0),
                'rpm_motor': float(telem['rpm_rl'])
            })
            
        telemetry_df = pd.DataFrame(records)
        a_max_g = float(telemetry_df['ax_g'].max())
        
        # Energy consumption in Wh (Mechanical output and Electrical accumulator draw)
        energy_kj = float(np.trapezoid(telemetry_df['power_kw'], telemetry_df['t'])) if hasattr(np, 'trapezoid') else float(np.trapz(telemetry_df['power_kw'], telemetry_df['t']))
        energy_wh = float(energy_kj / 3.6)

        elec_energy_kj = float(np.trapezoid(telemetry_df['p_elec_kw'], telemetry_df['t'])) if hasattr(np, 'trapezoid') else float(np.trapz(telemetry_df['p_elec_kw'], telemetry_df['t']))
        elec_energy_wh = float(elec_energy_kj / 3.6)
        avg_efficiency = float(energy_kj / max(elec_energy_kj, 1e-3))
        
        return {
            'success': True,
            'time': time_res,
            'v_max': v_max_res,
            'a_max_g': a_max_g,
            'energy_wh': energy_wh,
            'energy_kj': energy_kj,
            'elec_energy_wh': elec_energy_wh,
            'elec_energy_kj': elec_energy_kj,
            'avg_efficiency': avg_efficiency,
            'telemetry': telemetry_df
        }
    else:
        return {
            'success': False,
            'time': 999.0,
            'v_max': 0.0,
            'a_max_g': 0.0,
            'energy_wh': 0.0,
            'energy_kj': 0.0,
            'telemetry': pd.DataFrame()
        }
