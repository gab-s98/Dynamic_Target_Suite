"""
MDO Direct Collocation Trajectory & Racing Line Optimization
===========================================================
Clean, non-intersecting FSAE Circuit geometry, Direct Collocation OCP solver,
and Quasi-Static Backward-Forward Velocity Profiler.
"""

import math
import os
import sys
import time

try:
    import casadi
    casadi_dir = os.path.dirname(casadi.__file__)
    if hasattr(os, 'add_dll_directory') and os.path.exists(casadi_dir):
        os.add_dll_directory(casadi_dir)
    os.environ['PATH'] = casadi_dir + ';' + os.environ.get('PATH', '')
except Exception:
    pass

import numpy as np
import pandas as pd
import casadi as ca
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams


def build_track_geometry(track_type="fsae_autocross", track_length_m=850.0, ds=2.0, track_width_m=3.8, custom_df=None):
    """
    Builds a clean, non-intersecting, continuous-curvature FSAE circuit layout.
    Supports:
    - 'fsae_autocross' / 'fsae_circuit' (850m technical track)
    - 'fsae_skidpad' (Figure-8 dual-circle track, R=9.125m)
    - 'fsae_endurance' (1250m fast flowing track)
    - 'fsae_acceleration' / 'fsae_sprint' (150m drag & braking track)
    - 'custom' (Interpolates from custom_df containing 'x','y' or 's','kappa')
    """
    if track_type == "custom" and custom_df is not None:
        if 'x' in custom_df.columns and 'y' in custom_df.columns:
            # Reconstruct from Cartesian (x, y)
            x_raw = custom_df['x'].values
            y_raw = custom_df['y'].values
            dx = np.diff(x_raw)
            dy = np.diff(y_raw)
            ds_raw = np.sqrt(dx**2 + dy**2)
            s_cum = np.concatenate(([0], np.cumsum(ds_raw)))
            tot_len = s_cum[-1]
            s_arr = np.arange(0, tot_len + ds/2.0, ds)
            x_center = np.interp(s_arr, s_cum, x_raw)
            y_center = np.interp(s_arr, s_cum, y_raw)
            
            dx_c = np.gradient(x_center, ds)
            dy_c = np.gradient(y_center, ds)
            theta = np.arctan2(dy_c, dx_c)
            dtheta = np.gradient(np.unwrap(theta), ds)
            kappa = dtheta
            N = len(s_arr)
            
            w_half = track_width_m / 2.0
            x_left = x_center - w_half * np.sin(theta)
            y_left = y_center + w_half * np.cos(theta)
            x_right = x_center + w_half * np.sin(theta)
            y_right = y_center - w_half * np.cos(theta)
            
            is_closed = bool(np.hypot(x_center[-1] - x_center[0], y_center[-1] - y_center[0]) < 10.0)
            return {
                'track_type': 'custom',
                's': s_arr, 'N': N, 'ds': ds, 'kappa': kappa, 'theta': theta,
                'x_center': x_center, 'y_center': y_center,
                'x_left': x_left, 'y_left': y_left,
                'x_right': x_right, 'y_right': y_right,
                'track_width': float(track_width_m),
                'is_closed': is_closed
            }

    if track_type == "fsae_skidpad":
        # Official FSAE Skidpad (FS-Rules D 4.1):
        # Entry (15m) -> Right Circle (2 laps: Lap 1 entry + Lap 2 TIMED) -> Left Circle (2 laps: Lap 1 entry + Lap 2 TIMED) -> Exit (25m)
        R_skid = 9.125
        c_len = 2.0 * np.pi * R_skid  # ~57.334 m per lap
        s_entry = 15.0
        s_r_lap1 = s_entry + c_len
        s_r_lap2 = s_r_lap1 + c_len
        s_l_lap1 = s_r_lap2 + c_len
        s_l_lap2 = s_l_lap1 + c_len
        s_exit = s_l_lap2 + 25.0
        tot_len = s_exit
        
        s_arr = np.arange(0, tot_len + ds/2.0, ds)
        N = len(s_arr)
        kappa = np.zeros(N)
        for i, s in enumerate(s_arr):
            if s < s_entry:
                kappa[i] = 0.0                      # Entry straight
            elif s < s_r_lap2:
                kappa[i] = -1.0 / R_skid            # Right circle (2 laps: entry + timed)
            elif s < s_l_lap2:
                kappa[i] = 1.0 / R_skid             # Left circle (2 laps: entry + timed)
            else:
                kappa[i] = 0.0                      # Exit straight
    elif track_type == "fsae_endurance":
        tot_len = 1250.0
        s_arr = np.arange(0, tot_len + ds/2.0, ds)
        N = len(s_arr)
        kappa = np.zeros(N)
        for i, s in enumerate(s_arr):
            if s < 150.0: kappa[i] = 0.0
            elif s < 280.0: kappa[i] = -1.0 / 35.0  # Fast sweeper
            elif s < 380.0: kappa[i] = 0.0
            elif s < 460.0: kappa[i] = 1.0 / 20.0
            elif s < 580.0: kappa[i] = -1.0 / 22.0
            elif s < 700.0: kappa[i] = 0.0          # Long back straight
            elif s < 820.0: kappa[i] = 1.0 / 12.0   # Hairpin
            elif s < 950.0: kappa[i] = -1.0 / 25.0
            elif s < 1080.0: kappa[i] = 1.0 / 18.0
            elif s < 1180.0: kappa[i] = -1.0 / 30.0
            else: kappa[i] = 0.0
    elif track_type in ["fsae_acceleration", "fsae_sprint"]:
        tot_len = 200.0
        s_arr = np.arange(0, tot_len + ds/2.0, ds)
        N = len(s_arr)
        kappa = np.zeros(N)
    elif track_type == "fsae_autocross":
        tot_len = track_length_m
        s_arr = np.arange(0, tot_len + ds/2.0, ds)
        N = len(s_arr)
        kappa = np.zeros(N)
        for i, s in enumerate(s_arr):
            # 1. Main Straight (0 - 90m)
            if s < 90.0:
                kappa[i] = 0.0
            # 2. Turn 1 Fast Right Sweeper (90 - 180m, R=25m)
            elif 90.0 <= s < 180.0:
                kappa[i] = -1.0 / 26.0
            # 3. Short Chute (180 - 240m)
            elif 180.0 <= s < 240.0:
                kappa[i] = 0.0
            # 4. Technical Chicane (240 - 330m)
            elif 240.0 <= s < 285.0:
                kappa[i] = 1.0 / 14.0
            elif 285.0 <= s < 330.0:
                kappa[i] = -1.0 / 14.0
            # 5. Back Straight (330 - 430m)
            elif 330.0 <= s < 430.0:
                kappa[i] = 0.0
            # 6. Decreasing Radius Hairpin (430 - 530m, R=16m -> 9m)
            elif 430.0 <= s < 530.0:
                prog = (s - 430.0) / 100.0
                r_cur = 16.0 - prog * 7.0
                kappa[i] = 1.0 / r_cur
            # 7. Infield Connecting Straight (530 - 610m)
            elif 530.0 <= s < 610.0:
                kappa[i] = 0.0
            # 8. Fast Esses (610 - 720m)
            elif 610.0 <= s < 665.0:
                kappa[i] = -1.0 / 18.0
            elif 665.0 <= s < 720.0:
                kappa[i] = 1.0 / 18.0
            # 9. Final Turn & Pit Straight (720 - 850m)
            elif 720.0 <= s < 780.0:
                kappa[i] = -1.0 / 22.0
            else:
                kappa[i] = 0.0
    else:
        # Default 'fsae_circuit' (Standard 850m Closed-Loop Circuit)
        tot_len = track_length_m
        s_arr = np.arange(0, tot_len + ds/2.0, ds)
        N = len(s_arr)

        from scipy.interpolate import splprep, splev
        pts = np.array([
            [0.0, 0.0],          # Start / Finish Line
            [60.0, 0.0],         # Main Straight
            [100.0, -10.0],      # Entry Turn 1 Fast Sweeper
            [130.0, -40.0],      # Turn 1 Apex
            [120.0, -80.0],      # Turn 1 Exit
            [80.0, -110.0],      # Turn 2 Technical Left
            [50.0, -90.0],       # Short Connecting Chute
            [20.0, -110.0],      # Turn 3 Hairpin Entry
            [-20.0, -120.0],     # Hairpin Apex
            [-40.0, -90.0],      # Hairpin Exit
            [-30.0, -50.0],      # Infield Esses Left
            [-60.0, -20.0],      # Infield Esses Right
            [-80.0, 20.0],       # Fast Chicane Entry
            [-50.0, 50.0],       # Fast Chicane Exit
            [-20.0, 40.0],       # Final Sweeper Entry
            [-5.0, 15.0],        # Final Corner
            [0.0, 0.0]           # Seamless Closed Loop
        ], dtype=float)

        tck_raw, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_fine = np.linspace(0, 1, 10000)
        xf_raw, yf_raw = splev(u_fine, tck_raw)
        L_raw = float(np.sum(np.hypot(np.diff(xf_raw), np.diff(yf_raw))))
        scale_f = tot_len / L_raw
        pts_scaled = pts * scale_f
        tck, _ = splprep([pts_scaled[:, 0], pts_scaled[:, 1]], s=0, per=True, k=3)

        xf, yf = splev(u_fine, tck)
        seg_lens = np.hypot(np.diff(xf), np.diff(yf))
        s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        s_cum *= (tot_len / s_cum[-1])

        u_s = np.interp(s_arr, s_cum, u_fine)
        xc, yc = splev(u_s, tck)
        dxc, dyc = splev(u_s, tck, der=1)
        d2xc, d2yc = splev(u_s, tck, der=2)

        th0 = float(np.arctan2(dyc[0], dxc[0]))
        cos_th = math.cos(-th0)
        sin_th = math.sin(-th0)

        x_center = xc * cos_th - yc * sin_th
        y_center = xc * sin_th + yc * cos_th
        x_center -= x_center[0]
        y_center -= y_center[0]

        dxc_rot = dxc * cos_th - dyc * sin_th
        dyc_rot = dxc * sin_th + dyc * cos_th
        theta = np.arctan2(dyc_rot, dxc_rot)

        denom = (dxc**2 + dyc**2)**1.5
        denom = np.maximum(denom, 1e-12)
        kappa = (dxc * d2yc - dyc * d2xc) / denom

        # Numerical boundary welding for exact periodic closure
        x_center[-1] = x_center[0]
        y_center[-1] = y_center[0]
        theta[-1] = theta[0]
        kappa[-1] = kappa[0]

    if 'x_center' not in locals():
        # Integrate heading and centerline Cartesian coordinates for open tracks
        theta = np.zeros(N)
        x_center = np.zeros(N)
        y_center = np.zeros(N)
        
        for i in range(1, N):
            theta[i] = theta[i-1] + kappa[i-1] * ds
            x_center[i] = x_center[i-1] + np.cos(theta[i-1]) * ds
            y_center[i] = y_center[i-1] + np.sin(theta[i-1]) * ds

    w_half = track_width_m / 2.0
    x_left = np.zeros(N)
    y_left = np.zeros(N)
    x_right = np.zeros(N)
    y_right = np.zeros(N)
    
    for i in range(N):
        nx = -np.sin(theta[i])
        ny = np.cos(theta[i])
        x_left[i] = x_center[i] + w_half * nx
        y_left[i] = y_center[i] + w_half * ny
        x_right[i] = x_center[i] - w_half * nx
        y_right[i] = y_center[i] - w_half * ny

    is_closed = bool(np.hypot(x_center[-1] - x_center[0], y_center[-1] - y_center[0]) < 5.0)

    return {
        'track_type': track_type,
        's': s_arr,
        'N': N,
        'ds': ds,
        'kappa': kappa,
        'theta': theta,
        'x_center': x_center,
        'y_center': y_center,
        'x_left': x_left,
        'y_left': y_left,
        'x_right': x_right,
        'y_right': y_right,
        'track_width': float(track_width_m),
        'is_closed': is_closed
    }


def load_custom_track_file(filepath: str, ds: float = 2.0, track_width_m: float = 3.8) -> dict:
    """Loads a custom track from CSV/JSON file and converts to track dictionary."""
    if filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        df = pd.read_csv(filepath)
    return build_track_geometry(track_type="custom", ds=ds, track_width_m=track_width_m, custom_df=df)


def save_track_file(track: dict, filepath: str):
    """Saves track geometry coordinates (s, kappa, x_center, y_center, x_left, y_left, x_right, y_right) to CSV."""
    df = pd.DataFrame({
        's': track['s'],
        'kappa': track['kappa'],
        'x': track['x_center'],
        'y': track['y_center'],
        'x_left': track['x_left'],
        'y_left': track['y_left'],
        'x_right': track['x_right'],
        'y_right': track['y_right']
    })
    df.to_csv(filepath, index=False)


def compute_backward_forward_velocity_profile(track, vp):
    """
    Quasi-Static Velocity Profile (Corner Apex Limit + Backward Braking + Forward Accel)
    along the fixed track centerline.
    """
    s_arr = track['s']
    N = track['N']
    ds = track['ds']
    kappa = track['kappa']
    
    m = float(getattr(vp, 'm', 300.0))
    g = vp.g
    mu_y = float(getattr(vp, 'LMUY', 1.60))
    mu_x = float(getattr(vp, 'LMUX', 1.65))
    P_max = float(getattr(vp, 'P_max_kw', 80.0)) * 1000.0
    Cz = float(getattr(vp, 'Cz', 1.8))
    Cd = float(getattr(vp, 'Cd', 1.1))
    rho = float(getattr(vp, 'rho', 1.225))
    S = float(getattr(vp, 'S', 1.0))
    
    # 1. Apex Limits
    v_corner = np.zeros(N)
    for i in range(N):
        k_val = abs(kappa[i])
        if k_val < 1e-4:
            v_corner[i] = 45.0
        else:
            v_est = np.sqrt(mu_y * g / k_val)
            for _ in range(5):
                fz_tot = m * g + 0.5 * rho * S * Cz * (v_est**2)
                v_est = np.sqrt((mu_y * fz_tot) / (m * k_val))
            v_corner[i] = min(v_est, 45.0)

    # 2. Backward Pass
    v_back = np.copy(v_corner)
    for i in range(N - 2, -1, -1):
        v_next = v_back[i+1]
        fz_tot = m * g + 0.5 * rho * S * Cz * (v_next**2)
        f_drag = 0.5 * rho * S * Cd * (v_next**2)
        f_brake_max = mu_x * fz_tot + f_drag
        a_brake = f_brake_max / m
        v_allow = np.sqrt(v_next**2 + 2.0 * a_brake * ds)
        v_back[i] = min(v_back[i], v_allow)

    # 3. Forward Pass
    v_fwd = np.zeros(N)
    v_fwd[0] = max(v_back[0], 8.0)
    for i in range(1, N):
        v_prev = v_fwd[i-1]
        fz_tot = m * g + 0.5 * rho * S * Cz * (v_prev**2)
        f_drag = 0.5 * rho * S * Cd * (v_prev**2)
        f_trac = min(mu_x * fz_tot, P_max / max(v_prev, 2.0))
        a_acc = (f_trac - f_drag) / m
        v_acc = np.sqrt(max(v_prev**2 + 2.0 * a_acc * ds, 1.0))
        v_fwd[i] = min(v_back[i], v_acc)

    v_prof = np.maximum(v_fwd, 3.0)
    dt_arr = ds / v_prof
    t_prof_lap = float(np.sum(dt_arr))
    
    ax_prof = np.zeros(N)
    ax_prof[:-1] = (v_prof[1:]**2 - v_prof[:-1]**2) / (2.0 * ds)
    ax_prof[-1] = ax_prof[-2]
    ay_prof = (v_prof**2) * kappa

    return {
        'v_kmh': v_prof * 3.6,
        'v_mps': v_prof,
        'ax_g': ax_prof / g,
        'ay_g': ay_prof / g,
        'lap_time_s': t_prof_lap,
        'v_avg_kmh': float(np.mean(v_prof) * 3.6),
        'v_max_kmh': float(np.max(v_prof) * 3.6),
        'max_ay_g': float(np.max(np.abs(ay_prof)) / g)
    }


@dataclass
class DirectCollocationCostWeights:
    """
    Cost Function Weights and Normalization Factors for Direct Collocation OCP.
    Faithfully reproduces the multi-term objective structure of controlloModello_OCP_FSAE_1.m:
    
      J = sum_k( dt_k / J_s )                                   [Lap Time Quadrature]
        + sumsqr( rdu .* duk )                                  [1st Derivative: Slew Rate]
        + sumsqr( rdu2 .* duk2 )                                [2nd Derivative: Control Jerk/Curvature]
        + sumsqr( ru .* Uk )                                    [Control Effort / Magnitude]
        + sumsqr( rdy .* dzausk ) + sumsqr( rdy2 .* dzausk2 )   [State/Auxiliary Variations]
        + optional centerline tracking & Mayer terminal cost
    
    Attributes:
        w_time (float): Weight on lap time running cost (default: 1.0).
        T_scale (float): Normalization time scale [s] (corresponds to J_s in MATLAB, default: 45.0).
        rdu_ax (float): Weight on 1st derivative of normalized ax (d(ax)/ds or d(ax)/dt) (default: 1e-4).
        rdu_delta (float): Weight on 1st derivative of normalized delta (d(delta)/ds or d(delta)/dt) (default: 1e-3).
        use_time_rates (bool): If True, computes temporal slew rate du/dt; if False, spatial du/ds (default: False).
        rdu2_ax (float): Weight on 2nd derivative/difference of ax (longitudinal jerk) (default: 1e-4).
        rdu2_delta (float): Weight on 2nd derivative/difference of delta (steering curvature) (default: 1e-3).
        ru_ax (float): Weight on control magnitude effort for ax (default: 1e-6).
        ru_delta (float): Weight on control magnitude effort for delta (default: 1e-6).
        rdy_n (float): Weight on 1st derivative of lateral offset d(n)/ds (anti-weaving) (default: 1e-4).
        rdy2_n (float): Weight on 2nd derivative of lateral offset d^2(n)/ds^2 (default: 1e-4).
        rdy_xi (float): Weight on 1st derivative of heading angle d(xi)/ds (default: 1e-5).
        rdy2_xi (float): Weight on 2nd derivative of heading angle d^2(xi)/ds^2 (default: 1e-5).
        r_n (float): Weight on lateral offset magnitude n^2 (centerline tracking) (default: 0.0).
        r_xi (float): Weight on heading deviation xi^2 (default: 0.0).
        w_mayer_n (float): Terminal Mayer penalty on final lateral deviation n_end^2 (default: 0.0).
        w_mayer_xi (float): Terminal Mayer penalty on final heading deviation xi_end^2 (default: 0.0).
    """
    w_time: float = 1.0
    T_scale: float = 45.0
    rdu_ax: float = 0.005
    rdu_delta: float = 0.05
    use_time_rates: bool = False
    rdu2_ax: float = 0.01
    rdu2_delta: float = 0.10
    ru_ax: float = 1e-5
    ru_delta: float = 1e-4
    rdy_n: float = 0.05
    rdy2_n: float = 0.10
    rdy_xi: float = 0.01
    rdy2_xi: float = 0.05
    r_n: float = 0.005
    r_xi: float = 0.005
    w_mayer_n: float = 0.0
    w_mayer_xi: float = 0.0


def solve_direct_collocation_racing_line(vehicle_params=None, track_data=None, tir_file=None,
                                         track_length_m=850.0, ds=2.5, track_width_m=3.8,
                                         model_type="14-DOF", max_iter=200, progress_callback=None,
                                         cost_weights: Optional[Union[DirectCollocationCostWeights, Dict[str, Any]]] = None,
                                         collocation_scheme: str = "radau_3"):
    """
    Solves the Optimal Racing Line and Minimum Lap Time OCP using CasADi + IPOPT.
    Formulates a preconditioned Direct Collocation OCP aligned with MATLAB reference
    controlloModello_OCP_FSAE_1.m, supporting multi-term Lagrange running cost,
    control rate (slew rate) penalties, second-order control curvature (jerk) penalties,
    control effort regularizations, auxiliary/state smoothing, and Mayer terminal penalties.

    Parameters
    ----------
    collocation_scheme : str
        Collocation discretization scheme:
        - "radau_3" (recommended default): 3rd-degree Radau IIA orthogonal collocation (O(ds^5), L-stable).
        - "legendre_3": 3rd-degree Gauss-Legendre orthogonal collocation (O(ds^6)).
        - "radau_2": 2nd-degree Radau IIA orthogonal collocation (O(ds^3)).
        - "legendre_2": 2nd-degree Gauss-Legendre orthogonal collocation (O(ds^4)).
        - "midpoint": 1st-degree implicit midpoint / Crank-Nicolson (O(ds^2)).
    """
    if cost_weights is None:
        cw = DirectCollocationCostWeights()
    elif isinstance(cost_weights, dict):
        cw = DirectCollocationCostWeights(**cost_weights)
    elif isinstance(cost_weights, DirectCollocationCostWeights):
        cw = cost_weights
    else:
        cw = DirectCollocationCostWeights()

    if vehicle_params is None:
        vp = VehicleParams()
    else:
        vp = vehicle_params

    if track_data is None:
        track = build_track_geometry(track_length_m=track_length_m, ds=ds, track_width_m=track_width_m)
    else:
        track = track_data

    if progress_callback:
        progress_callback(10, "Costruzione geometria tracciato e formulazione NLP CasADi...")

    N = track['N']
    ds_val = track['ds']
    kappa_arr = track['kappa']
    w_track = track['track_width']
    
    w_veh = max(getattr(vp, 't_f', 1.25), getattr(vp, 't_r', 1.15)) + 0.20
    n_max_limit = max((w_track - w_veh) / 2.0, 0.40)
    
    m = float(getattr(vp, 'm', 300.0))
    g = vp.g
    L = float(getattr(vp, 'L', 1.53))
    rho = float(getattr(vp, 'rho', 1.225))
    S = float(getattr(vp, 'S', 1.0))
    Cz = float(getattr(vp, 'Cz', 1.8))
    Cd = float(getattr(vp, 'Cd', 1.1))
    mu_y = float(getattr(vp, 'LMUY', 1.60))
    mu_x = float(getattr(vp, 'LMUX', 1.65))
    P_max_watts = float(getattr(vp, 'P_max_kw', 80.0)) * 1000.0

    if model_type == "3-DOF":
        eff_lat_grip = mu_y * 0.98
    elif model_type == "7-DOF":
        eff_lat_grip = mu_y * 0.985
    else:
        eff_lat_grip = mu_y * 0.99

    # --- NUMERICAL PRECONDITIONING & VARIABLE SCALING ---
    # Non-dimensionalize all state and control variables to O(1) in [-1, 1] to avoid KKT ill-conditioning
    n_scale = max(n_max_limit, 0.50)
    xi_scale = np.deg2rad(42.0)  # ~42 deg (~0.733 rad)
    E_scale = 0.5 * (45.0**2)  # 1012.5 m^2/s^2 (corresponds to 45 m/s)
    ax_scale = max(mu_x * g, 5.0)  # ~16.2 m/s^2
    delta_scale = np.deg2rad(28.0)  # ~28 deg (~0.489 rad)
    T_scale = cw.T_scale if cw.T_scale > 0.0 else 45.0  # Nominal lap time scaling

    opti = ca.Opti()

    # Non-dimensional scaled decision variables in [-1, 1]
    n_bar = opti.variable(N)
    xi_bar = opti.variable(N)
    E_bar = opti.variable(N)
    ax_bar = opti.variable(N)
    delta_bar = opti.variable(N)

    # Dimensional physical states reconstructed for dynamics
    n = n_bar * n_scale
    xi = xi_bar * xi_scale
    E = E_bar * E_scale
    ax = ax_bar * ax_scale
    delta = delta_bar * delta_scale
    v = ca.sqrt(2.0 * E + 1.0)

    # --- PARSE COLLOCATION SCHEME ---
    # Supports high-order orthogonal collocation (Radau IIA or Gauss-Legendre d=2,3)
    # as well as 1st-order implicit midpoint scheme.
    scheme_str = str(collocation_scheme).lower().strip() if collocation_scheme is not None else "radau_3"
    if scheme_str == "midpoint":
        poly_deg = 1
        poly_method = "midpoint"
    elif "legendre" in scheme_str:
        poly_method = "legendre"
        poly_deg = 2 if "2" in scheme_str else 3
    elif "radau" in scheme_str:
        poly_method = "radau"
        poly_deg = 2 if "2" in scheme_str else 3
    else:
        poly_method = "radau"
        poly_deg = 3

    # 1. Collocation formulation (Orthogonal Radau/Legendre vs Implicit Midpoint)
    if poly_deg >= 2:
        coll_pts = ca.collocation_points(poly_deg, poly_method)
        C_coll, D_coll, B_coll = ca.collocation_coeff(coll_pts)

        # Internal collocation variables for all N-1 elements: (poly_deg x N-1)
        n_coll = opti.variable(poly_deg, N - 1)
        xi_coll = opti.variable(poly_deg, N - 1)
        E_coll = opti.variable(poly_deg, N - 1)

        opti.subject_to(opti.bounded(-1.0, n_coll, 1.0))
        opti.subject_to(opti.bounded(-1.0, xi_coll, 1.0))
        opti.subject_to(opti.bounded((0.5 * 3.0**2) / E_scale, E_coll, 1.0))

        opti.set_initial(n_coll, 0.0)
        opti.set_initial(xi_coll, 0.0)
        opti.set_initial(E_coll, (0.5 * 16.0**2) / E_scale)

        # Vectorized polynomial state derivatives across all elements
        dn_poly = ca.mtimes(C_coll.T, ca.vertcat(n_bar[:-1].T, n_coll))
        dxi_poly = ca.mtimes(C_coll.T, ca.vertcat(xi_bar[:-1].T, xi_coll))
        dE_poly = ca.mtimes(C_coll.T, ca.vertcat(E_bar[:-1].T, E_coll))

        # Continuity condition at interval boundary: Z * D_coll - x_{k+1} = 0
        n_end = ca.mtimes(D_coll.T, ca.vertcat(n_bar[:-1].T, n_coll))
        xi_end = ca.mtimes(D_coll.T, ca.vertcat(xi_bar[:-1].T, xi_coll))
        E_end = ca.mtimes(D_coll.T, ca.vertcat(E_bar[:-1].T, E_coll))
        opti.subject_to((n_end.T - n_bar[1:]) == 0.0)
        opti.subject_to((xi_end.T - xi_bar[1:]) == 0.0)
        opti.subject_to((E_end.T - E_bar[1:]) == 0.0)

        # Quadrature and dynamic collocation constraints
        lap_time = 0.0
        dt_k = 0.0
        s_base = track['s'][:-1]
        for j in range(poly_deg):
            tau = float(coll_pts[j])
            ax_j = ax[:-1] + tau * (ax[1:] - ax[:-1])
            del_j = delta[:-1] + tau * (delta[1:] - delta[:-1])
            n_j = n_coll[j, :].T * n_scale
            xi_j = xi_coll[j, :].T * xi_scale
            E_j = E_coll[j, :].T * E_scale
            s_j = s_base + tau * ds_val
            kap_j = np.interp(s_j, track['s'], kappa_arr)
            scale_j = (1.0 - n_j * kap_j) / ca.cos(xi_j)
            v_j = ca.sqrt(2.0 * E_j + 1.0)
            dt_j = (ds_val * scale_j) / ca.fmax(v_j, 1.0)
            bj = float(B_coll[j])
            lap_time = lap_time + bj * ca.sum1(dt_j)
            dt_k = dt_k + bj * dt_j

            dn_dyn = ca.tan(xi_j) * (1.0 - n_j * kap_j)
            dxi_dyn = ((del_j / L) * scale_j) - kap_j
            dE_dyn = ax_j * scale_j

            opti.subject_to((ds_val * dn_dyn / n_scale) - dn_poly[j, :].T == 0.0)
            opti.subject_to((ds_val * dxi_dyn / xi_scale) - dxi_poly[j, :].T == 0.0)
            opti.subject_to((ds_val * dE_dyn / E_scale) - dE_poly[j, :].T == 0.0)
    else:
        # Midpoint implicit scheme (degree 1)
        v = ca.sqrt(2.0 * E + 1.0)
        v_mid = 0.5 * (v[:-1] + v[1:])
        n_mid = 0.5 * (n[:-1] + n[1:])
        xi_mid = 0.5 * (xi[:-1] + xi[1:])
        kap_mid = 0.5 * (kappa_arr[:-1] + kappa_arr[1:])
        scale_mid = (1.0 - n_mid * kap_mid) / ca.cos(xi_mid)
        dt_k = (ds_val * scale_mid) / ca.fmax(v_mid, 1.0)
        lap_time = ca.sum1(dt_k)

        dn_ds = ca.tan(xi_mid) * (1.0 - n_mid * kap_mid)
        opti.subject_to((n[1:] - n[:-1] - ds_val * dn_ds) / n_scale == 0.0)

        delta_mid = 0.5 * (delta[:-1] + delta[1:])
        ax_mid = 0.5 * (ax[:-1] + ax[1:])
        dxi_ds = ((delta_mid / L) * scale_mid) - kap_mid
        opti.subject_to((xi[1:] - xi[:-1] - ds_val * dxi_ds) / xi_scale == 0.0)
        opti.subject_to((E[1:] - E[:-1] - ds_val * ax_mid * scale_mid) / E_scale == 0.0)

    # Term 1: Running Cost - Lap Time Quadrature (Qk * B_coll * dsk / J_s in MATLAB)
    cost_time = cw.w_time * (lap_time / T_scale)

    # Term 2: Control First Derivatives / Slew Rate (sumsqr(rdu .* duk) in MATLAB)
    # duk = diff(Uk) / ds (spatial) or duk_t = duk / sf = diff(Uk) / dt (temporal)
    if cw.use_time_rates:
        dt_eff = ca.fmax(dt_k, 1e-3)
        duk_ax = (ax_bar[1:] - ax_bar[:-1]) / dt_eff
        duk_delta = (delta_bar[1:] - delta_bar[:-1]) / dt_eff
        dt_mean = ca.sum1(dt_k) / max(N - 1, 1)
        cost_du = (cw.rdu_ax * ca.sum1(duk_ax**2) + cw.rdu_delta * ca.sum1(duk_delta**2)) * dt_mean
    else:
        duk_ax = (ax_bar[1:] - ax_bar[:-1]) / ds_val
        duk_delta = (delta_bar[1:] - delta_bar[:-1]) / ds_val
        cost_du = (cw.rdu_ax * ca.sum1(duk_ax**2) + cw.rdu_delta * ca.sum1(duk_delta**2)) * ds_val

    # Term 3: Control Second Difference / Curvature & Jerk (sumsqr(rdu2 .* duk2) in MATLAB)
    # Physical second derivative: (duk[k+1] - duk[k]) / ds
    if N > 2:
        d2uk_ax_core = (duk_ax[1:] - duk_ax[:-1]) / ds_val
        d2uk_ax = ca.vertcat(d2uk_ax_core, d2uk_ax_core[-1])
        d2uk_delta_core = (duk_delta[1:] - duk_delta[:-1]) / ds_val
        d2uk_delta = ca.vertcat(d2uk_delta_core, d2uk_delta_core[-1])
        cost_du2 = (cw.rdu2_ax * ca.sum1(d2uk_ax**2) + cw.rdu2_delta * ca.sum1(d2uk_delta**2)) * ds_val
    else:
        cost_du2 = 0.0

    # Term 4: Control Magnitude Effort (normalized by total track length for mesh invariance)
    L_tot = float(track_length_m) if track_length_m > 0 else 850.0
    cost_u = (cw.ru_ax * ca.sum1(ax_bar**2) + cw.ru_delta * ca.sum1(delta_bar**2)) * (ds_val / L_tot)

    # Term 5: Auxiliary / Lateral State Rates
    d_n = (n_bar[1:] - n_bar[:-1]) / ds_val
    d_xi = (xi_bar[1:] - xi_bar[:-1]) / ds_val
    cost_dy = (cw.rdy_n * ca.sum1(d_n**2) + cw.rdy_xi * ca.sum1(d_xi**2)) * ds_val

    if N > 2:
        d2_n_core = (d_n[1:] - d_n[:-1]) / ds_val
        d2_n = ca.vertcat(d2_n_core, d2_n_core[-1])
        d2_xi_core = (d_xi[1:] - d_xi[:-1]) / ds_val
        d2_xi = ca.vertcat(d2_xi_core, d2_xi_core[-1])
        cost_dy2 = (cw.rdy2_n * ca.sum1(d2_n**2) + cw.rdy2_xi * ca.sum1(d2_xi**2)) * ds_val
    else:
        cost_dy2 = 0.0

    # Term 6: Centerline & Heading Deviation Regularization (essential to keep car centered in straights!)
    cost_state = (cw.r_n * ca.sum1(n_bar**2) + cw.r_xi * ca.sum1(xi_bar**2)) * (ds_val / L_tot)

    # Term 7: Terminal Mayer Penalty on Arrival Residuals
    cost_mayer = cw.w_mayer_n * (n_bar[-1]**2) + cw.w_mayer_xi * (xi_bar[-1]**2)

    # Total Objective Function
    cost_obj = cost_time + cost_du + cost_du2 + cost_u + cost_dy + cost_dy2 + cost_state + cost_mayer
    opti.minimize(cost_obj)

    # 3. Vectorized Normalized Box Bounds (all [-1, 1])
    opti.subject_to(opti.bounded(-1.0, n_bar, 1.0))
    opti.subject_to(opti.bounded(-1.0, xi_bar, 1.0))
    opti.subject_to(opti.bounded(-1.0, delta_bar, 1.0))
    opti.subject_to(opti.bounded((0.5 * 3.0**2) / E_scale, E_bar, 1.0))

    # 4. Scaled Aero & Combined Friction Envelope Constraints
    f_down = 0.5 * rho * S * Cz * (2.0 * E)
    f_drag = 0.5 * rho * S * Cd * (2.0 * E)
    total_fz = m * g + f_down
    mu_eff_y = eff_lat_grip * (total_fz / (m * g))
    mu_eff_x = mu_x * (total_fz / (m * g))

    ay_req = (2.0 * E) * (delta / L)
    fric_comb = (ax / (mu_eff_x * g))**2 + (ay_req / (mu_eff_y * g))**2
    opti.subject_to(fric_comb <= 1.0)

    # Scaled Power, Torque & Braking Limits
    T_max_motor = float(getattr(vp, 'T_max_motor', 100.0))
    final_drive = float(getattr(vp, 'final_drive', 4.0))
    n_drivetrain = float(getattr(vp, 'n_drivetrain', 0.95))
    R_w = float(getattr(vp, 'R_w', 0.203))
    drivetrain_type = str(getattr(vp, 'drivetrain_type', 'rwd')).lower()
    n_motors = 4 if drivetrain_type == 'awd' else 2
    F_x_max_torque = (n_motors * T_max_motor * final_drive * n_drivetrain) / R_w

    f_trac = m * ax + f_drag
    opti.subject_to(f_trac <= F_x_max_torque)
    opti.subject_to((f_trac * v) / P_max_watts <= 1.0)
    opti.subject_to(ax / (mu_x * g) >= -1.80)

    # Boundary conditions
    opti.subject_to(n_bar[0] == 0.0)
    opti.subject_to(xi_bar[0] == 0.0)

    # Well-conditioned initial guesses (Ackermann steering alignment on curvature)
    opti.set_initial(n_bar, 0.0)
    opti.set_initial(xi_bar, 0.0)
    opti.set_initial(E_bar, (0.5 * 16.0**2) / E_scale)
    opti.set_initial(ax_bar, 0.0)
    opti.set_initial(delta_bar, (L * kappa_arr) / delta_scale)

    num_cpus = min(os.cpu_count() or 4, 4)
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)
    os.environ["MKL_NUM_THREADS"] = str(num_cpus)

    if progress_callback:
        progress_callback(15, f"Risoluzione NLP Precondizionato ({num_cpus} Thread CPU attivi)...")

    # Real-Time Iteration Callback for live UI monitoring
    def opti_iteration_callback(it_idx):
        if progress_callback:
            try:
                t_curr = float(opti.debug.value(lap_time))
                n_curr = opti.debug.value(n)
                dev_m = float(np.max(np.abs(n_curr)))
                e_curr = opti.debug.value(E)
                v_curr = np.sqrt(np.maximum(2.0 * e_curr, 1.0))
                d_curr = opti.debug.value(delta)
                ay_m = float(np.max(np.abs((v_curr**2) * (d_curr / L)))) / g
                pct = min(15 + int(((it_idx + 1) / max_iter) * 75.0), 92)
                msg = f"Iterazione {it_idx+1}/{max_iter} | Lap Time: {t_curr:.2f} s | Deviazione n: ±{dev_m:.2f} m | Max Ay: {ay_m:.2f} g"
                progress_callback(pct, msg)
            except Exception:
                pass

    opti.callback(opti_iteration_callback)

    p_opts = {'expand': True, 'print_time': False}
    s_opts = {
        'max_iter': max_iter,
        'tol': 1e-3,
        'acceptable_tol': 2e-2,
        'acceptable_iter': 5,
        'print_level': 0,
        'linear_solver': 'mumps',
        'nlp_scaling_method': 'gradient-based',
        'sb': 'yes'
    }
    opti.solver('ipopt', p_opts, s_opts)

    cost_breakdown = {}
    try:
        sol = opti.solve()
        success = True
        n_opt = sol.value(n)
        xi_opt = sol.value(xi)
        E_opt = sol.value(E)
        ax_opt = sol.value(ax)
        delta_opt = sol.value(delta)
        t_lap = float(sol.value(lap_time))
        cost_breakdown = {
            'cost_total': float(sol.value(cost_obj)),
            'cost_time': float(sol.value(cost_time)),
            'cost_du': float(sol.value(cost_du)),
            'cost_du2': float(sol.value(cost_du2)) if N > 2 else 0.0,
            'cost_u': float(sol.value(cost_u)),
            'cost_dy': float(sol.value(cost_dy)),
            'cost_dy2': float(sol.value(cost_dy2)) if N > 2 else 0.0,
            'cost_state': float(sol.value(cost_state)),
            'cost_mayer': float(sol.value(cost_mayer)),
        }
    except Exception:
        status = ""
        try:
            status = opti.return_status()
        except Exception:
            pass
        success = ("Solved" in status or "Acceptable" in status or "Optimal" in status)
        try:
            n_opt = opti.debug.value(n)
            xi_opt = opti.debug.value(xi)
            E_opt = opti.debug.value(E)
            ax_opt = opti.debug.value(ax)
            delta_opt = opti.debug.value(delta)
            t_lap = float(opti.debug.value(lap_time))
            cost_breakdown = {
                'cost_total': float(opti.debug.value(cost_obj)),
                'cost_time': float(opti.debug.value(cost_time)),
                'cost_du': float(opti.debug.value(cost_du)),
                'cost_du2': float(opti.debug.value(cost_du2)) if N > 2 else 0.0,
                'cost_u': float(opti.debug.value(cost_u)),
                'cost_dy': float(opti.debug.value(cost_dy)),
                'cost_dy2': float(opti.debug.value(cost_dy2)) if N > 2 else 0.0,
                'cost_state': float(opti.debug.value(cost_state)),
                'cost_mayer': float(opti.debug.value(cost_mayer)),
            }
        except Exception:
            n_opt = np.zeros(N)
            xi_opt = np.zeros(N)
            v_safe = np.minimum(30.0, np.sqrt(np.maximum(mu_y * 9.81 / (np.abs(kappa_arr) + 1e-4), 16.0)))
            E_opt = 0.5 * (v_safe**2)
            ax_opt = np.zeros(N)
            delta_opt = L * kappa_arr
            t_lap = float(np.sum(ds_val / v_safe))

    if progress_callback:
        progress_callback(80, "Calcolo del confronto con Profiler di Velocità...")

    profiler_res = compute_backward_forward_velocity_profile(track, vp)
    v_opt = np.sqrt(np.maximum(2.0 * E_opt, 1.0))
    
    x_opt = np.zeros(N)
    y_opt = np.zeros(N)
    psi_opt = np.zeros(N)
    theta_arr = track['theta']
    x_cnt = track['x_center']
    y_cnt = track['y_center']
    
    for i in range(N):
        nx = -np.sin(theta_arr[i])
        ny = np.cos(theta_arr[i])
        x_opt[i] = x_cnt[i] + n_opt[i] * nx
        y_opt[i] = y_cnt[i] + n_opt[i] * ny
        psi_opt[i] = theta_arr[i] + xi_opt[i]

    t_arr = np.zeros(N)
    for i in range(1, N):
        v_mid = 0.5 * (v_opt[i-1] + v_opt[i])
        n_m = 0.5 * (n_opt[i-1] + n_opt[i])
        xi_m = 0.5 * (xi_opt[i-1] + xi_opt[i])
        kap_m = 0.5 * (kappa_arr[i-1] + kappa_arr[i])
        scale_s = (1.0 - n_m * kap_m) / max(np.cos(xi_m), 0.1)
        dt = (ds_val * scale_s) / max(v_mid, 1.0)
        t_arr[i] = t_arr[i-1] + dt

    ay_opt_g = (v_opt**2 * (delta_opt / L)) / g

    records = {
        's_m': track['s'],
        'x_center': x_cnt,
        'y_center': y_cnt,
        'x_left': track['x_left'],
        'y_left': track['y_left'],
        'x_right': track['x_right'],
        'y_right': track['y_right'],
        'x_opt': x_opt,
        'y_opt': y_opt,
        'n_m': n_opt,
        'xi_deg': np.rad2deg(xi_opt),
        'theta_deg': np.rad2deg(theta_arr),
        'psi_deg': np.rad2deg(psi_opt),
        'v_mps': v_opt,
        'v_kmh': v_opt * 3.6,
        'ax_g': ax_opt / g,
        'ay_g': ay_opt_g,
        'steer_deg': np.rad2deg(delta_opt),
        't_s': t_arr,
        'v_profiler_kmh': profiler_res['v_kmh'],
        'ax_profiler_g': profiler_res['ax_g'],
        'ay_profiler_g': profiler_res['ay_g']
    }
    df_res = pd.DataFrame(records)

    # Compute official Skidpad breakdown if applicable (FS-Rules D 4.1)
    skidpad_info = {}
    if track.get('track_type') == 'fsae_skidpad':
        try:
            R_skid = 9.125
            c_len = 2.0 * np.pi * R_skid
            s_entry = 15.0
            s_r1 = s_entry + c_len
            s_r2 = s_r1 + c_len
            s_l1 = s_r2 + c_len
            s_l2 = s_l1 + c_len
            
            s_vals = df_res['s_m'].values
            t_vals = df_res['t_s'].values
            
            t_r1 = float(np.interp(s_r1, s_vals, t_vals))
            t_r2 = float(np.interp(s_r2, s_vals, t_vals))
            t_l1 = float(np.interp(s_l1, s_vals, t_vals))
            t_l2 = float(np.interp(s_l2, s_vals, t_vals))
            
            t_timed_r = t_r2 - t_r1
            t_timed_l = t_l2 - t_l1
            t_skid_official = 0.5 * (t_timed_r + t_timed_l)
            
            skidpad_info = {
                'is_skidpad': True,
                't_timed_right_s': t_timed_r,
                't_timed_left_s': t_timed_l,
                't_official_s': t_skid_official,
                't_total_run_s': float(t_vals[-1]),
                'v_avg_timed_kmh': float((c_len / max(t_skid_official, 0.1)) * 3.6)
            }
        except Exception:
            pass

    return {
        'success': success,
        'model_type': model_type,
        'lap_time_s': t_lap,
        'v_max_kmh': float(df_res['v_kmh'].max()),
        'v_avg_kmh': float(df_res['v_kmh'].mean()),
        'max_ay_g': float(df_res['ay_g'].abs().max()),
        'max_ax_g': float(df_res['ax_g'].max()),
        'min_ax_g': float(df_res['ax_g'].min()),
        'max_lateral_dev_m': float(df_res['n_m'].abs().max()),
        'telemetry': df_res,
        'track': track,
        'profiler': profiler_res,
        'skidpad_info': skidpad_info,
        'cost_breakdown': cost_breakdown,
        'cost_weights': cw,
        'delta_time_s': t_lap - profiler_res['lap_time_s'],
        'delta_time_pct': ((t_lap - profiler_res['lap_time_s']) / profiler_res['lap_time_s']) * 100.0,
        'collocation_scheme': collocation_scheme
    }
