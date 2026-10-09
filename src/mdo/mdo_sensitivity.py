"""
mdo_sensitivity.py — Sensitivity Analysis & Formula Student 2026 Event Scoring Engine

Implements:
1. Official Formula Student Rules 2026 (FS-Rules 2026 v1.1 D 9 and Table 3, Table 11) Dynamic Event Scoring:
   - Skidpad (50 pts, quadratic scaling, Tmax = 1.35 * Tmin, Pmin = 0.05 * Pmax)
   - Acceleration 75m (50 pts, quadratic scaling, Tmax = 1.70 * Tmin, Pmin = 0.05 * Pmax)
   - Autocross (100 pts, quadratic scaling, Tmax = 1.40 * Tmin, Pmin = 0.10 * Pmax)
   - Endurance 22km (250 pts, quadratic scaling, Tmax = 1.50 * Tmin, Pmin = 0.10 * Pmax)
   - Efficiency (75 pts, quadratic scaling with EF = T^2 * E, EFmax = 2.0 * EFmin)
   - Overall Dynamic Score: 525 pts max
2. Competition Strength / Event Benchmarks from FS-World Ranking:
   - FS Germany (Hockenheimring)
   - FS Austria (Red Bull Ring)
   - FS Italy (Varano de' Melegari)
   - FS East (Hungaroring)
   - Custom / Baseline-anchored benchmarks
3. Full Quasi-Static Event Simulations for Sensitivity Analysis (Acceleration, Skidpad, AutoX, Endurance, High-Speed Grip, Kus).
4. 1D Parameter Sweeps and Multi-Parameter Tornado / Pareto Sensitivity Analysis.
"""

import os
import copy
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo import mdo_target_optimization as mdo
from src.mdo.mdo_acceleration import simulate_acceleration
from src.mdo.mdo_skidpad import simulate_skidpad
from src.mdo.mdo_dynamic_events import simulate_autox_endurance

# ==============================================================================
# FS-RULES 2026 BENCHMARKS & SCORING CONSTANTS
# ==============================================================================
FS2026_DISCIPLINES = {
    'skidpad': {
        'name': 'Skidpad (Manual)',
        'P_max': 50.0,
        'P_min_factor': 0.05,
        'T_max_factor': 1.35,
    },
    'acceleration': {
        'name': 'Acceleration 75m (Manual)',
        'P_max': 50.0,
        'P_min_factor': 0.05,
        'T_max_factor': 1.70,
    },
    'autocross': {
        'name': 'Autocross (Manual)',
        'P_max': 100.0,
        'P_min_factor': 0.10,
        'T_max_factor': 1.40,
    },
    'endurance': {
        'name': 'Endurance 22km (Manual)',
        'P_max': 250.0,
        'P_min_factor': 0.10,
        'T_max_factor': 1.50,
    },
    'efficiency': {
        'name': 'Efficiency (Manual)',
        'P_max': 75.0,
        'P_min_factor': 0.0,
        'EF_max_factor': 2.0,
    }
}

FS_WORLD_RANKING_BENCHMARKS = {
    'FS_Germany': {
        'label': 'FS Germany (FSG Hockenheim)',
        'accel_tmin': 3.35,      # [s] 75m
        'skid_tmin': 4.75,       # [s] Skidpad
        'autox_tmin': 53.50,     # [s] Autocross lap
        'endur_tmin': 1320.0,    # [s] Endurance 22km (22.0 min)
        'energy_min_kwh': 2.85,  # [kWh] Endurance energy consumption
    },
    'FS_Austria': {
        'label': 'FS Austria (FSA Red Bull Ring)',
        'accel_tmin': 3.45,
        'skid_tmin': 4.85,
        'autox_tmin': 55.00,
        'endur_tmin': 1360.0,
        'energy_min_kwh': 3.00,
    },
    'FS_Italy': {
        'label': 'FS Italy (FSI Varano)',
        'accel_tmin': 3.55,
        'skid_tmin': 4.95,
        'autox_tmin': 56.50,
        'endur_tmin': 1400.0,
        'energy_min_kwh': 3.20,
    },
    'FS_East': {
        'label': 'FS East (Hungaroring)',
        'accel_tmin': 3.40,
        'skid_tmin': 4.80,
        'autox_tmin': 54.50,
        'endur_tmin': 1340.0,
        'energy_min_kwh': 2.90,
    },
    'Custom': {
        'label': 'Personalizzato (Custom Benchmark)',
        'accel_tmin': 3.40,
        'skid_tmin': 4.80,
        'autox_tmin': 54.00,
        'endur_tmin': 1350.0,
        'energy_min_kwh': 3.00,
    }
}

def compute_fs2026_score_discipline(t_team, t_min, disc_key):
    """
    Computes points for a standard dynamic discipline according to FS-Rules 2026 Section D 9.1.1:
      SCORE = (Pmax - Pmin) * ((Tmax - Tteam) / (Tmax - Tmin))^2 + Pmin
    """
    disc = FS2026_DISCIPLINES[disc_key]
    P_max = disc['P_max']
    P_min = disc['P_min_factor'] * P_max
    T_max = disc['T_max_factor'] * t_min
    
    if t_team is None or t_team <= 0 or np.isnan(t_team):
        return 0.0
    
    t_capped = min(max(t_team, t_min), T_max)
    if t_team > T_max:
        return 0.0
    
    ratio = (T_max - t_capped) / (T_max - t_min)
    score = (P_max - P_min) * (ratio ** 2) + P_min
    return float(np.clip(score, 0.0, P_max))

def compute_fs2026_efficiency_score(t_endur_team, energy_kwh_team, t_endur_min, energy_kwh_min):
    """
    Computes Efficiency score according to FS-Rules 2026 Section D 9.4.1:
      EF = T^2 * E
      SCORE = Pmax * ((EFmax - EFteam) / (EFmax - EFmin))^2
    where EFmax = 2.0 * EFmin.
    """
    P_max = FS2026_DISCIPLINES['efficiency']['P_max']
    if t_endur_team is None or energy_kwh_team is None or t_endur_team <= 0 or energy_kwh_team <= 0:
        return 0.0
    
    EF_team = (t_endur_team ** 2) * energy_kwh_team
    EF_min = (t_endur_min ** 2) * energy_kwh_min
    EF_max = 2.0 * EF_min
    
    if EF_team > EF_max:
        return 0.0
    
    EF_capped = max(EF_team, EF_min)
    ratio = (EF_max - EF_capped) / (EF_max - EF_min)
    score = P_max * (ratio ** 2)
    return float(np.clip(score, 0.0, P_max))

def evaluate_fsae_2026_full_event(
    t_accel, t_skid, t_autox, t_endur, energy_kwh,
    benchmark_event='FS_Germany', custom_benchmark=None
):
    """
    Evaluates complete Formula Student 2026 dynamic scoring breakdown.
    Returns: dict with scores for each event, total dynamic points, and benchmark info.
    """
    bench = FS_WORLD_RANKING_BENCHMARKS.get(benchmark_event, FS_WORLD_RANKING_BENCHMARKS['FS_Germany']).copy()
    if custom_benchmark is not None:
        bench.update(custom_benchmark)
        
    s_accel = compute_fs2026_score_discipline(t_accel, bench['accel_tmin'], 'acceleration')
    s_skid = compute_fs2026_score_discipline(t_skid, bench['skid_tmin'], 'skidpad')
    s_autox = compute_fs2026_score_discipline(t_autox, bench['autox_tmin'], 'autocross')
    s_endur = compute_fs2026_score_discipline(t_endur, bench['endur_tmin'], 'endurance')
    s_eff = compute_fs2026_efficiency_score(t_endur, energy_kwh, bench['endur_tmin'], bench['energy_min_kwh'])
    
    total_score = s_accel + s_skid + s_autox + s_endur + s_eff
    
    accel_data = {
        'time': t_accel,
        't_min': bench['accel_tmin'],
        't_max': bench['accel_tmin'] * 1.70,
        'score': s_accel,
        'max_score': 50.0
    }
    skid_data = {
        'time': t_skid,
        't_min': bench['skid_tmin'],
        't_max': bench['skid_tmin'] * 1.35,
        'score': s_skid,
        'max_score': 50.0
    }
    autox_data = {
        'time': t_autox,
        't_min': bench['autox_tmin'],
        't_max': bench['autox_tmin'] * 1.40,
        'score': s_autox,
        'max_score': 100.0
    }
    endur_data = {
        'time': t_endur,
        't_min': bench['endur_tmin'],
        't_max': bench['endur_tmin'] * 1.50,
        'score': s_endur,
        'max_score': 250.0
    }
    eff_data = {
        'energy_kwh': energy_kwh,
        'energy_min_kwh': bench['energy_min_kwh'],
        'score': s_eff,
        'max_score': 75.0
    }
    
    return {
        'benchmark_label': bench.get('label', benchmark_event),
        'acceleration': accel_data,
        'skidpad': skid_data,
        'autocross': autox_data,
        'autox': autox_data,
        'endurance': endur_data,
        'efficiency': eff_data,
        'total_dynamic_score': total_score,
        'max_total_score': 525.0
    }

# ==============================================================================
# SENSITIVITY ANALYSIS EVALUATION WITH FULL DYNAMIC EVENTS
# ==============================================================================

def simulate_vehicle_performance_metrics(vp: VehicleParams, tir_file=None, model_type="14-DOF"):
    """
    Computes a comprehensive dictionary of vehicle dynamic metrics for sensitivity analysis
    using the full quasi-static dynamic event simulators:
    - accel_time (75m) [s]
    - accel_vmax_kmh [km/h]
    - skidpad_time (R=9.125m) [s] & skidpad_ay_peak [g]
    - autox_time [s]
    - endurance_time [s]
    - energy_kwh [kWh]
    - ay_peak_high_speed [g] (at 20 m/s - strongly dependent on Cz aero downforce)
    - ay_peak_low_speed [g] (at 10 m/s - mechanical grip)
    - total_fsae_points [pts / 525]
    - understeer_kus [deg/g]
    """
    # 1. Acceleration 75m
    acc_res = simulate_acceleration(vp, distance=75.0, is_awd=False)
    t_accel = acc_res['time'] if acc_res.get('success', False) else 5.20
    v_accel_max = acc_res['v_max'] if acc_res.get('success', False) else 22.0

    # 2. Skidpad Steady-State Event (R=9.125m)
    skid_res = simulate_skidpad(vp, radius=9.125, tir_path=tir_file)
    t_skid = skid_res['time'] if skid_res.get('success', False) else 5.10
    ay_skid = skid_res.get('ay', 1.40)

    # 3. Autocross & Endurance Event (Quasi-Static Track Profiler on 850m FSAE Track)
    autox_res = simulate_autox_endurance(vp, event_type="AutoX", track_length_m=850.0, tir_path=tir_file)
    t_autox = autox_res.get('lap_time_s', 60.0) if autox_res.get('success', False) else 60.0
    t_endur = autox_res.get('total_time_s', t_autox * 22.0) if autox_res.get('success', False) else 1350.0
    energy_kwh = autox_res.get('total_energy_kwh', 3.10) if autox_res.get('success', False) else 3.10

    # 4. High-Speed & Low-Speed Grip (with downforce Fz_aero = 0.5 * rho * Cz * S * v^2)
    # At 20 m/s (72 km/h):
    f_down_20 = 0.5 * vp.rho * vp.Cz * vp.S * (20.0**2)
    fz_tot_20 = vp.m * 9.81 + f_down_20
    mu_eff = getattr(vp, 'LMUY', 1.0) * 1.42
    ay_peak_20 = (mu_eff * fz_tot_20) / (vp.m * 9.81)

    # At 10 m/s (36 km/h):
    f_down_10 = 0.5 * vp.rho * vp.Cz * vp.S * (10.0**2)
    fz_tot_10 = vp.m * 9.81 + f_down_10
    ay_peak_10 = (mu_eff * fz_tot_10) / (vp.m * 9.81)

    # 5. Understeer Kus at 12 m/s
    try:
        df_hd = mdo.generate_handling_sweep(
            v_x=12.0, wd_f=vp.wd_f, lltd=vp.get_effective_lltd(1.0),
            aero_bal=vp.aero_balance, lmuy_mult=vp.LMUY,
            L=vp.L, t_f=vp.t_f, t_r=vp.t_r, tir_file=tir_file,
            K_phi_f=vp.K_phi_f, K_phi_r=vp.K_phi_r, q1=vp.q1, q2=vp.q2,
            camber_f=vp.camber_f_static, camber_r=vp.camber_r_static,
            toe_f=vp.toe_f_static, toe_r=vp.toe_r_static,
            method='arclength', stop_at_peak=True, num_points=35
        )
        if not df_hd.empty:
            sub_kus = df_hd[(df_hd['ay_g'] >= 0.2) & (df_hd['ay_g'] <= 0.8)]
            if len(sub_kus) >= 2:
                diff = sub_kus['alpha_f_deg'] - sub_kus['alpha_r_deg'] if 'alpha_f_deg' in sub_kus else sub_kus['slip_diff_deg']
                kus = float(np.polyfit(sub_kus['ay_g'], diff, 1)[0])
            else:
                kus = 0.5
        else:
            kus = 0.5
    except Exception:
        kus = 0.5

    # 6. Total FSAE 2026 Points
    scores = evaluate_fsae_2026_full_event(t_accel, t_skid, t_autox, t_endur, energy_kwh, 'FS_Germany')
    total_pts = scores['total_dynamic_score']

    return {
        'accel_time': float(t_accel),
        'accel_vmax_kmh': float(v_accel_max * 3.6),
        'skidpad_time': float(t_skid),
        'skidpad_ay_peak': float(ay_skid),
        'autox_time': float(t_autox),
        'endurance_time': float(t_endur),
        'energy_kwh': float(energy_kwh),
        'ay_peak_high_speed': float(ay_peak_20),
        'ay_peak_low_speed': float(ay_peak_10),
        'total_fsae_points': float(total_pts),
        'understeer_kus': float(kus)
    }

def run_sensitivity_1d(param_name: str, values_list, metric_key='total_fsae_points', base_params: VehicleParams=None, tir_file=None):
    """
    Runs a 1D parameter sweep across values_list, recording the chosen metric.
    """
    vp_base = copy.deepcopy(base_params) if base_params is not None else VehicleParams()
    base_metrics = simulate_vehicle_performance_metrics(vp_base, tir_file=tir_file)
    base_metric_val = base_metrics.get(metric_key, 0.0)

    rows = []
    for val in values_list:
        vp_test = copy.deepcopy(vp_base)
        if param_name == 'lltd':
            val_clamped = float(np.clip(val, 0.05, 0.95))
            K_tot = max(vp_test.K_phi_f + vp_test.K_phi_r, 1.0)
            vp_test.K_phi_f = val_clamped * K_tot
            vp_test.K_phi_r = (1.0 - val_clamped) * K_tot
            vp_test.lltd = val_clamped
        elif param_name == 'roll_stiffness_tot':
            K_tot_old = max(vp_test.K_phi_f + vp_test.K_phi_r, 1.0)
            ratio = vp_test.K_phi_f / K_tot_old
            vp_test.K_phi_f = float(val) * ratio
            vp_test.K_phi_r = float(val) * (1.0 - ratio)
            vp_test.lltd = ratio
        else:
            setattr(vp_test, param_name, float(val))
            if param_name in ['K_z_f', 'K_z_r']:
                vp_test.sync_wheel_rates_and_springs()
            elif param_name in ['K_phi_f', 'K_phi_r']:
                vp_test.lltd = vp_test.lltd_elastic()
            
        res_m = simulate_vehicle_performance_metrics(vp_test, tir_file=tir_file)
        m_val = res_m.get(metric_key, 0.0)
        
        rows.append({
            'param_value': float(val),
            'metric_value': float(m_val),
            'delta_from_base': float(m_val - base_metric_val),
            'delta_pct': float((m_val - base_metric_val) / max(abs(base_metric_val), 1e-6) * 100.0)
        })

    return pd.DataFrame(rows), base_metric_val

def run_tornado_sensitivity(
    param_names_list,
    delta_pct=10.0,
    metric_key='total_fsae_points',
    base_params: VehicleParams=None,
    tir_file=None
):
    """
    Computes positive and negative variations (+/- delta_pct %) for each parameter,
    ranking them by overall absolute metric change (Tornado / Pareto diagram).
    """
    vp_base = copy.deepcopy(base_params) if base_params is not None else VehicleParams()
    base_metrics = simulate_vehicle_performance_metrics(vp_base, tir_file=tir_file)
    base_metric_val = base_metrics.get(metric_key, 0.0)

    results = []
    for p_name in param_names_list:
        nominal_val = getattr(vp_base, p_name, None)
        if nominal_val is None or abs(nominal_val) < 1e-7:
            continue
        
        val_high = nominal_val * (1.0 + delta_pct / 100.0) if nominal_val >= 0 else nominal_val * (1.0 - delta_pct / 100.0)
        vp_high = copy.deepcopy(vp_base)
        setattr(vp_high, p_name, float(val_high))
        if p_name in ['K_z_f', 'K_z_r']:
            vp_high.sync_wheel_rates_and_springs()
        elif p_name in ['K_phi_f', 'K_phi_r']:
            vp_high.lltd = vp_high.lltd_elastic()
        m_high = simulate_vehicle_performance_metrics(vp_high, tir_file=tir_file).get(metric_key, 0.0)
        
        val_low = nominal_val * (1.0 - delta_pct / 100.0) if nominal_val >= 0 else nominal_val * (1.0 + delta_pct / 100.0)
        vp_low = copy.deepcopy(vp_base)
        setattr(vp_low, p_name, float(val_low))
        if p_name in ['K_z_f', 'K_z_r']:
            vp_low.sync_wheel_rates_and_springs()
        elif p_name in ['K_phi_f', 'K_phi_r']:
            vp_low.lltd = vp_low.lltd_elastic()
        m_low = simulate_vehicle_performance_metrics(vp_low, tir_file=tir_file).get(metric_key, 0.0)
        
        d_high = m_high - base_metric_val
        d_low = m_low - base_metric_val
        total_swing = abs(m_high - m_low)
        
        results.append({
            'param_name': p_name,
            'nominal_val': float(nominal_val),
            'val_low': float(val_low),
            'val_high': float(val_high),
            'metric_low': float(m_low),
            'metric_high': float(m_high),
            'delta_low': float(d_low),
            'delta_high': float(d_high),
            'total_swing': float(total_swing),
            'sensitivity_coeff': float((m_high - m_low) / (2.0 * delta_pct))
        })

    df_tornado = pd.DataFrame(results)
    if not df_tornado.empty:
        df_tornado = df_tornado.sort_values('total_swing', ascending=True).reset_index(drop=True)
    return df_tornado, base_metric_val
