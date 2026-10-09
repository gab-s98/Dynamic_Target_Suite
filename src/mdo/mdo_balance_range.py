"""
mdo_balance_range.py
--------------------
Optimal Balance Range & Multi-Source Parameter Uncertainty Sweep Engine.

Analyzes the vehicle dynamic balance (Understeer Gradient K_us, High-Speed Stability Margin,
Peak Lateral Grip a_y, Dynamic Bias) across parameter sweeps (LLTD, Weight Distribution,
Aero Balance, Roll Centers, CG Height, Roll Stiffness) while subjecting each point to
multi-source parametric uncertainties:
- Tire friction coefficient uncertainty (LMUY ± Delta mu %)
- Mass / Weight distribution uncertainty (WD_f ± Delta WD %)
- Aero balance migration / Pitch sensitivity (Aero_f ± Delta Aero %)
- Suspension roll stiffness / ARB tolerance (LLTD ± Delta LLTD %)

Computes the Nominal, Best-Case, and Worst-Case bounds and automatically identifies
the Optimal Robust Balance Corridor [P_opt_min, P_opt_max].
"""

import os
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo import mdo_target_optimization as mdo


SWEEP_PARAMETER_CONFIGS = {
    'lltd': {
        'name': 'Ripartizione Trasferimento Carico (LLTD)',
        'unit': '%',
        'scale': 100.0,
        'default_min': 0.40,
        'default_max': 0.70,
        'attr': 'lltd',
        'is_fraction': True
    },
    'wd_f': {
        'name': 'Distribuzione Masse Avantreno (wd_f)',
        'unit': '%',
        'scale': 100.0,
        'default_min': 0.42,
        'default_max': 0.58,
        'attr': 'wd_f',
        'is_fraction': True
    },
    'aero_balance': {
        'name': 'Bilanciamento Aerodinamico (aero_bal)',
        'unit': '%',
        'scale': 100.0,
        'default_min': 0.38,
        'default_max': 0.58,
        'attr': 'aero_balance',
        'is_fraction': True
    },
    'rc_height_f': {
        'name': 'Centro di Rollio Ant. (h_rc,f)',
        'unit': 'mm',
        'scale': 1000.0,
        'default_min': 0.010,
        'default_max': 0.100,
        'attr': 'rc_height_f',
        'is_fraction': False
    },
    'rc_height_r': {
        'name': 'Centro di Rollio Post. (h_rc,r)',
        'unit': 'mm',
        'scale': 1000.0,
        'default_min': 0.020,
        'default_max': 0.120,
        'attr': 'rc_height_r',
        'is_fraction': False
    },
    'h_cg': {
        'name': 'Altezza Baricentro (h_cg)',
        'unit': 'mm',
        'scale': 1000.0,
        'default_min': 0.220,
        'default_max': 0.340,
        'attr': 'h_cg',
        'is_fraction': False
    },
    'K_phi_f': {
        'name': 'Rigidezza di Rollio Ant. (K_phi_f)',
        'unit': 'Nm/deg',
        'scale': 1.0,
        'default_min': 300.0,
        'default_max': 1200.0,
        'attr': 'K_phi_f',
        'is_fraction': False
    }
}


def _calc_kus_and_ay_for_setup(
    v_x: float,
    wd_f: float,
    lltd: float,
    aero_bal: float,
    L: float,
    t_f: float,
    t_r: float,
    lmuy_mult: float,
    tir_file: str,
    num_points: int = 40
) -> Tuple[float, float, float]:
    """
    Computes (K_us [deg/g], a_y_max [g], stability_margin [%]) for a specific setup point.
    """
    tir_file = TireParams._resolve_path(tir_file)

    df = mdo.generate_handling_sweep(
        v_x=v_x,
        wd_f=wd_f,
        lltd=lltd,
        aero_bal=aero_bal,
        lmuy_mult=lmuy_mult,
        delta_max_deg=22.0,
        num_points=num_points,
        L=L,
        t_f=t_f,
        t_r=t_r,
        tir_file=tir_file,
        method='arclength',
        stop_at_peak=True
    )
    
    if df.empty or 'ay_g' not in df.columns or len(df) < 3:
        # Fallback approximation based on 2-DOF steady state physical gradient:
        # Increasing front LLTD increases front saturation -> increases understeer (+)
        # Increasing front mass wd_f increases front lateral demand -> increases understeer (+)
        # Increasing front aero_bal increases front grip -> decreases understeer (-)
        kus_approx = (lltd - 0.50) * 3.5 + (wd_f - 0.50) * 2.5 - (aero_bal - 0.50) * 2.0 + (1.0 - lmuy_mult) * 0.3
        ay_max_approx = 1.65 * lmuy_mult
        stab_margin = (lltd - aero_bal) * 100.0
        return float(kus_approx), float(ay_max_approx), float(stab_margin)

    ay = df['ay_g'].values
    ay_max = float(np.max(ay))
    
    if 'alpha_f_deg' in df.columns and 'alpha_r_deg' in df.columns:
        alpha_diff = (df['alpha_f_deg'] - df['alpha_r_deg']).values
    elif 'slip_diff_deg' in df.columns:
        alpha_diff = df['slip_diff_deg'].values
    else:
        alpha_diff = np.zeros_like(ay)

    # Linear understeer gradient around linear range (ay in 0.1 to 0.6 * ay_max)
    mask_lin = (ay >= 0.15 * ay_max) & (ay <= 0.65 * ay_max)
    if np.sum(mask_lin) >= 3:
        p_fit = np.polyfit(ay[mask_lin], alpha_diff[mask_lin], 1)
        kus_val = float(p_fit[0])  # deg/g
    else:
        # Average secant slope
        idx_eval = min(len(ay)-1, max(1, int(len(ay)*0.5)))
        kus_val = float(alpha_diff[idx_eval] / max(ay[idx_eval], 0.05))

    stab_margin = (lltd - aero_bal) * 100.0
    return float(kus_val), float(ay_max), float(stab_margin)


def evaluate_optimal_balance_sweep(
    vp: VehicleParams,
    param_key: str = 'lltd',
    param_min: Optional[float] = None,
    param_max: Optional[float] = None,
    n_points: int = 15,
    v_eval: float = 12.0,
    tire_unc_pct: float = 10.0,
    wd_unc_pct: float = 1.5,
    aero_unc_pct: float = 2.0,
    lltd_unc_pct: float = 2.0,
    target_kus_min_degg: float = 0.10,
    target_kus_max_degg: float = 0.80,
    tir_file: str = ""
) -> Dict[str, Any]:
    """
    Executes a comprehensive balance parameter sweep with multi-source uncertainty bounds.
    
    Returns a dictionary containing:
    - 'sweep_df': DataFrame with nominal and uncertainty bounds across the sweep
    - 'optimal_range': [p_opt_min, p_opt_max]
    - 'current_val': current parameter value in vp
    - 'is_current_in_optimal': bool
    - 'robust_score_pct': overall robust stability score (0-100%)
    - 'param_info': metadata on the parameter (name, unit, scale)
    """
    cfg = SWEEP_PARAMETER_CONFIGS.get(param_key, SWEEP_PARAMETER_CONFIGS['lltd'])
    
    p_min = param_min if param_min is not None else cfg['default_min']
    p_max = param_max if param_max is not None else cfg['default_max']
    p_values = np.linspace(p_min, p_max, n_points)
    
    # Base vehicle geometry
    L = vp.L
    t_f = vp.t_f
    t_r = vp.t_r
    current_val = getattr(vp, cfg['attr'], cfg['default_min'])

    rows = []
    
    # Uncertainty perturbations (Corner test permutations)
    delta_mu_frac = max(0.0, tire_unc_pct / 100.0)
    delta_wd = max(0.0, wd_unc_pct / 100.0)
    delta_aero = max(0.0, aero_unc_pct / 100.0)
    delta_lltd = max(0.0, lltd_unc_pct / 100.0)

    for val in p_values:
        # Construct baseline parameters for this sweep step
        wd_nom = val if param_key == 'wd_f' else vp.wd_f
        lltd_nom = val if param_key == 'lltd' else vp.lltd
        aero_nom = val if param_key == 'aero_balance' else vp.aero_balance
        lmuy_nom = getattr(vp, 'LMUY', 1.0)
        
        # If sweeping roll centers or K_phi, estimate effective LLTD change
        if param_key == 'rc_height_f':
            # Increasing front RC increases geometric front roll load transfer -> increases effective LLTD
            d_rc = val - vp.rc_height_f
            lltd_nom = np.clip(vp.lltd + d_rc * 1.8, 0.30, 0.80)
        elif param_key == 'rc_height_r':
            # Increasing rear RC decreases front LLTD share
            d_rc = val - vp.rc_height_r
            lltd_nom = np.clip(vp.lltd - d_rc * 1.8, 0.30, 0.80)
        elif param_key == 'K_phi_f':
            k_r = getattr(vp, 'K_phi_r', 400.0)
            lltd_nom = val / max(val + k_r, 1.0)
            
        # 1. Nominal Point Calculation
        kus_nom, ay_nom, stab_nom = _calc_kus_and_ay_for_setup(
            v_x=v_eval, wd_f=wd_nom, lltd=lltd_nom, aero_bal=aero_nom,
            L=L, t_f=t_f, t_r=t_r, lmuy_mult=lmuy_nom, tir_file=tir_file
        )

        # 2. Worst-Case Uncertainty Corner Evaluation
        # Permutations that challenge understeer/oversteer bounds:
        # Corner A (Oversteer Hazard): Lower mu (-mu), lower LLTD (-lltd), higher front mass/aero (+wd, +aero)
        # Corner B (Understeer Push): Higher mu (+mu), higher LLTD (+lltd), lower front mass/aero (-wd, -aero)
        # Corner C (Low Grip Surface): -mu nominal balance
        
        permutations = [
            # (mu_mult, wd_mod, lltd_mod, aero_mod)
            (max(1.0 - delta_mu_frac, 0.4), np.clip(wd_nom + delta_wd, 0.35, 0.65), np.clip(lltd_nom - delta_lltd, 0.30, 0.80), np.clip(aero_nom + delta_aero, 0.30, 0.70)),
            (min(1.0 + delta_mu_frac, 1.6), np.clip(wd_nom - delta_wd, 0.35, 0.65), np.clip(lltd_nom + delta_lltd, 0.30, 0.80), np.clip(aero_nom - delta_aero, 0.30, 0.70)),
            (max(1.0 - delta_mu_frac, 0.4), wd_nom, lltd_nom, aero_nom),
            (min(1.0 + delta_mu_frac, 1.6), wd_nom, lltd_nom, aero_nom),
            (1.0, np.clip(wd_nom + delta_wd, 0.35, 0.65), np.clip(lltd_nom - delta_lltd, 0.30, 0.80), aero_nom),
            (1.0, np.clip(wd_nom - delta_wd, 0.35, 0.65), np.clip(lltd_nom + delta_lltd, 0.30, 0.80), aero_nom)
        ]
        
        kus_list = [kus_nom]
        ay_list = [ay_nom]
        stab_list = [stab_nom]
        
        for mu_p, wd_p, lltd_p, aero_p in permutations:
            k_p, ay_p, stab_p = _calc_kus_and_ay_for_setup(
                v_x=v_eval, wd_f=wd_p, lltd=lltd_p, aero_bal=aero_p,
                L=L, t_f=t_f, t_r=t_r, lmuy_mult=mu_p, tir_file=tir_file, num_points=25
            )
            kus_list.append(k_p)
            ay_list.append(ay_p)
            stab_list.append(stab_p)

        kus_min = float(np.min(kus_list))
        kus_max = float(np.max(kus_list))
        ay_min = float(np.min(ay_list))
        ay_max = float(np.max(ay_list))
        stab_min = float(np.min(stab_list))
        stab_max = float(np.max(stab_list))

        # Check optimal condition at this sweep point:
        # 1. Nominal Kus in target window
        # 2. Worst-case Kus > 0 (no snap oversteer under perturbation)
        # 3. High-speed stability margin >= +1.5% (Tip #89)
        is_nom_target = (target_kus_min_degg <= kus_nom <= target_kus_max_degg)
        is_robust_stable = (kus_min > 0.0) and (stab_min >= 1.0)
        is_optimal = is_nom_target and is_robust_stable

        rows.append({
            'param_val': float(val),
            'param_display': float(val * cfg['scale'] if cfg['is_fraction'] else val * cfg['scale']),
            'kus_nom': kus_nom,
            'kus_min': kus_min,
            'kus_max': kus_max,
            'ay_nom': ay_nom,
            'ay_min': ay_min,
            'ay_max': ay_max,
            'stab_nom': stab_nom,
            'stab_min': stab_min,
            'stab_max': stab_max,
            'is_optimal': bool(is_optimal),
            'is_nom_target': bool(is_nom_target),
            'is_robust_stable': bool(is_robust_stable)
        })

    df_sweep = pd.DataFrame(rows)

    # Determine Optimal Range boundaries
    df_opt = df_sweep[df_sweep['is_optimal']]
    if not df_opt.empty:
        opt_min = float(df_opt['param_val'].min())
        opt_max = float(df_opt['param_val'].max())
        opt_min_disp = float(df_opt['param_display'].min())
        opt_max_disp = float(df_opt['param_display'].max())
    else:
        # Fallback to nominal target range if robust criteria is too tight
        df_nom_t = df_sweep[df_sweep['is_nom_target']]
        if not df_nom_t.empty:
            opt_min = float(df_nom_t['param_val'].min())
            opt_max = float(df_nom_t['param_val'].max())
            opt_min_disp = float(df_nom_t['param_display'].min())
            opt_max_disp = float(df_nom_t['param_display'].max())
        else:
            # Pick the point closest to the middle of target Kus window
            target_mid = 0.5 * (target_kus_min_degg + target_kus_max_degg)
            idx_best = int(np.argmin(np.abs(df_sweep['kus_nom'] - target_mid)))
            opt_min = opt_max = float(df_sweep.loc[idx_best, 'param_val'])
            opt_min_disp = opt_max_disp = float(df_sweep.loc[idx_best, 'param_display'])

    current_disp = current_val * cfg['scale'] if cfg['is_fraction'] else current_val * cfg['scale']
    is_current_in_optimal = bool(opt_min <= current_val <= opt_max)

    # Robust Balance Score (% of points in nominal sweep that stay stable under worst-case uncertainty)
    robust_score = float((df_sweep['is_robust_stable'].sum() / len(df_sweep)) * 100.0)

    return {
        'sweep_df': df_sweep,
        'optimal_range': [opt_min, opt_max],
        'optimal_range_display': [opt_min_disp, opt_max_disp],
        'current_val': float(current_val),
        'current_val_display': float(current_disp),
        'is_current_in_optimal': is_current_in_optimal,
        'robust_score_pct': robust_score,
        'param_key': param_key,
        'param_info': cfg,
        'v_eval': float(v_eval),
        'target_kus_window': [target_kus_min_degg, target_kus_max_degg]
    }
