"""
transient_sensitivity.py
------------------------
Parametric Transient Sensitivity Simulator for Dynamic Target Suite.

Performs 1D and 2D parameter sweeps using the 14-DOF (or 7-DOF) transient ODE,
extracting scalar engineering KPIs from each maneuver telemetry.

Supported Scenarios
-------------------
  rc_height_f         : Front Roll Center Height q1 [m]  (migra verticalmente)
  rc_height_r         : Rear Roll Center Height q2 [m]
  rc_lat_gain_f       : Front RC Lateral Migration Gain (a1 of rollcenter_lat_f_coeffs) [m/m]
  rc_lat_gain_r       : Rear RC Lateral Migration Gain
  arb_distribution    : K_phi_f / (K_phi_f+K_phi_r) at constant total [-]  (LLTD elastico)
  roll_stiffness_tot  : Total roll stiffness K_phi_tot [Nm/rad]             (ratio fisso)
  h_cg                : Centre-of-Gravity height above ground [m]
  damping_phi         : Total roll damping C_phi_tot [Nm·s/rad]             (ratio fisso)

Supported Maneuvers
-------------------
  step_steer  : Ramp Step Steer (ISO 7401 / SAE J266)
  dlc         : Double Lane Change (ISO 3888-2)

KPIs computed per run
---------------------
  r_ss_deg_s      : Yaw rate steady-state                [deg/s]
  r_peak_deg_s    : Yaw rate peak during transient       [deg/s]
  overshoot_pct   : Yaw rate overshoot                   [%]
  t90_s           : Time to 90% of r_ss                  [s]
  t_settle_s      : Settling time (±2% of r_ss band)     [s]
  phi_max_deg     : Max roll angle                        [deg]
  ay_peak_g       : Peak lateral acceleration             [g]
  beta_peak_deg   : Peak side-slip angle                  [deg]
  lltd_peak       : Dynamic LLTD at peak Ay              [-]
  lltd_ss         : Dynamic LLTD in steady-state         [-]
  dFz_asym_pct    : L/R Fz asymmetry at peak Ay          [%]
  util_front_max  : Max front tire friction utilization   [%]
  util_rear_max   : Max rear tire friction utilization    [%]
"""

import copy
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_dynamic_events import simulate_step_steer, simulate_double_lane_change


# ---------------------------------------------------------------------------
# Scenario Catalogue
# ---------------------------------------------------------------------------

SCENARIO_CATALOG = {
    'rc_height_f': {
        'label': 'Front RC Height q₁ [m]',
        'unit': 'm',
        'default_range': (0.02, 0.14),
        'default_n': 11,
        'description': (
            'Altezza del centro di rollio anteriore sopra il suolo.\n'
            'Aumentando q₁ cresce il carico trasferito per via geometrica (direct LT),\n'
            'riducendo il momento ribaltante attorno all\'asse di rollio e quindi φ_max.\n'
            'Tuttavia una migrazione laterale più pronunciata altera l\'asimmetria Fz L/R.'
        ),
    },
    'rc_height_r': {
        'label': 'Rear RC Height q₂ [m]',
        'unit': 'm',
        'default_range': (0.02, 0.16),
        'default_n': 11,
        'description': (
            'Altezza del centro di rollio posteriore sopra il suolo.\n'
            'q₂ elevato aumenta il trasferimento di carico geometrico al retrotreno,\n'
            'spostando l\'LLTD effettivo verso il retrotreno e tendendo a sottosterzare.'
        ),
    },
    'rc_lat_gain_f': {
        'label': 'Front RC Lateral Migration Gain a₁ [m/m]',
        'unit': 'm/m',
        'default_range': (0.0, 2.0),
        'default_n': 11,
        'description': (
            'Gain lineare della migrazione laterale del centro di rollio anteriore\n'
            'rispetto al differenziale di corsa (z_fl − z_fr). Un valore più alto\n'
            'significa che il RC migra più rapidamente verso la ruota esterna in curva,\n'
            'aumentando l\'asimmetria del braccio ribaltante e modificando il LLTD dinamico.'
        ),
    },
    'rc_lat_gain_r': {
        'label': 'Rear RC Lateral Migration Gain a₁ [m/m]',
        'unit': 'm/m',
        'default_range': (0.0, 1.8),
        'default_n': 11,
        'description': (
            'Gain lineare della migrazione laterale del centro di rollio posteriore.\n'
            'Impatta principalmente il trasferimento di carico asimmetrico al retrotreno.'
        ),
    },
    'arb_distribution': {
        'label': 'ARB LLTD Distribution K_φ,f / K_φ,tot [-]',
        'unit': '[-]',
        'default_range': (0.30, 0.75),
        'default_n': 11,
        'description': (
            'Frazione di rigidezza di rollio (ARB+molla) attribuita al fronte.\n'
            'Determina l\'LLTD elastico: valori > 0.5 → sovrasterzo,\n'
            'valori < 0.5 → sottosterzo. Somma K_φ,tot mantenuta costante.'
        ),
    },
    'K_phi_f': {
        'label': 'Front Roll Stiffness K_φ,f [Nm/rad]',
        'unit': 'Nm/rad',
        'default_range': (5000.0, 45000.0),
        'default_n': 11,
        'description': (
            'Rigidezza di rollio dell\'avantreno (molle + barra ARB).\n'
            'Incrementandola cresce la saturazione dell\'avantreno,\n'
            'aumentando il sottosterzo. LLTD viene sincronizzato automaticamente.'
        ),
    },
    'K_phi_r': {
        'label': 'Rear Roll Stiffness K_φ,r [Nm/rad]',
        'unit': 'Nm/rad',
        'default_range': (5000.0, 45000.0),
        'default_n': 11,
        'description': (
            'Rigidezza di rollio del retrotreno (molle + barra ARB).\n'
            'Incrementandola cresce il trasferimento di carico al retrotreno,\n'
            'riducendo il sottosterzo o favorendo il sovrasterzo. LLTD sincronizzato.'
        ),
    },
    'roll_stiffness_tot': {
        'label': 'Total Roll Stiffness K_φ,tot [Nm/rad]',
        'unit': 'Nm/rad',
        'default_range': (4000.0, 22000.0),
        'default_n': 11,
        'description': (
            'Rigidezza di rollio totale del veicolo (fronte + retro).\n'
            'Valori elevati → φ ridotto, risposta più nervosa.\n'
            'Valori bassi → più rollio, risposta più smorzata.\n'
            'Il ratio K_φ,f/K_φ,r è mantenuto costante.'
        ),
    },
    'h_cg': {
        'label': 'CG Height h_cg [m]',
        'unit': 'm',
        'default_range': (0.18, 0.40),
        'default_n': 11,
        'description': (
            'Altezza del baricentro sopra il suolo.\n'
            'Più elevato → maggiore momento ribaltante → più rollio,\n'
            'più trasferimento di carico laterale → LLTD critico.\n'
            'È il parametro di layout con maggiore impatto globale.'
        ),
    },
    'damping_phi': {
        'label': 'Total Roll Damping C_φ,tot [Nm·s/rad]',
        'unit': 'Nm·s/rad',
        'default_range': (200.0, 3000.0),
        'default_n': 11,
        'description': (
            'Smorzamento di rollio totale.\n'
            'Valori bassi → risposta oscillatoria con overshoot di rollio.\n'
            'Valori alti → risposta monotona ma più lenta.\n'
            'Il ratio C_φ,f/C_φ,r è mantenuto costante.'
        ),
    },
}

# KPI display metadata: (column_name, display_label, format_string)
KPI_METADATA = [
    ('r_ss_deg_s',     'r_ss [deg/s]',             '{:.2f}'),
    ('r_peak_deg_s',   'r_peak [deg/s]',           '{:.2f}'),
    ('overshoot_pct',  'Overshoot [%]',            '{:.1f}'),
    ('t90_s',          't₉₀ [s]',                  '{:.3f}'),
    ('yaw_delay_s',    'Δt_yaw Delay [s]',         '{:.3f}'),
    ('ay_delay_s',     'Δt_ay Delay [s]',          '{:.3f}'),
    ('tb_factor',      'TB-Factor [deg·s]',        '{:.3f}'),
    ('r_dot_peak',     'r_dot_peak [deg/s²]',      '{:.1f}'),
    ('t_settle_s',     't_settle [s]',             '{:.3f}'),
    ('steer_sens_yaw', 'Steer Sens r [(°/s)/°]',   '{:.2f}'),
    ('steer_sens_ay',  'Steer Sens ay [g/°]',      '{:.3f}'),
    ('roll_gradient',  'Roll Grad [deg/g]',        '{:.2f}'),
    ('phi_max_deg',    'φ_max [deg]',              '{:.2f}'),
    ('ay_peak_g',      'ay_peak [g]',              '{:.3f}'),
    ('beta_peak_deg',  'β_peak [deg]',             '{:.2f}'),
    ('lltd_peak',      'LLTD_peak [-]',            '{:.3f}'),
    ('lltd_ss',        'LLTD_ss [-]',              '{:.3f}'),
    ('dFz_asym_pct',   'ΔFz asym [%]',            '{:.1f}'),
    ('util_front_max', 'Util_F_max [%]',          '{:.1f}'),
    ('util_rear_max',  'Util_R_max [%]',          '{:.1f}'),
    ('eff_grip_pct',   'Grip Efficiency η_μ [%]', '{:.1f}'),
]


# ---------------------------------------------------------------------------
# Parameter Modifier
# ---------------------------------------------------------------------------

def _apply_param(base_vp: VehicleParams, scenario_key: str, value: float) -> VehicleParams:
    """
    Return a deep copy of base_vp with a single parameter modified
    according to the scenario. Physical consistency is maintained where
    applicable (e.g., poly3 constant term kept in sync with q1/q2).
    """
    vp = copy.deepcopy(base_vp)

    if scenario_key == 'rc_height_f':
        vp.q1 = value
        # Update a0 term of the vertical RC poly3 map so nominal height matches
        c = list(vp.rollcenter_f_coeffs)
        c[3] = value
        vp.rollcenter_f_coeffs = c

    elif scenario_key == 'rc_height_r':
        vp.q2 = value
        c = list(vp.rollcenter_r_coeffs)
        c[3] = value
        vp.rollcenter_r_coeffs = c

    elif scenario_key == 'rc_lat_gain_f':
        c = list(getattr(vp, 'rollcenter_lat_f_coeffs', [0.0, 0.0, 0.85, 0.0]))
        c[2] = value   # a1 = linear gain
        vp.rollcenter_lat_f_coeffs = c

    elif scenario_key == 'rc_lat_gain_r':
        c = list(getattr(vp, 'rollcenter_lat_r_coeffs', [0.0, 0.0, 0.65, 0.0]))
        c[2] = value
        vp.rollcenter_lat_r_coeffs = c

    elif scenario_key == 'arb_distribution':
        K_tot = vp.K_phi_f + vp.K_phi_r
        K_tot = max(K_tot, 1.0)
        vp.K_phi_f = float(np.clip(value, 0.01, 0.99)) * K_tot
        vp.K_phi_r = K_tot - vp.K_phi_f
        # Update nominal lltd to reflect the new elastic LLTD
        vp.lltd = float(np.clip(value, 0.01, 0.99))

    elif scenario_key == 'K_phi_f':
        vp.K_phi_f = float(max(value, 100.0))
        vp.lltd = vp.lltd_elastic()

    elif scenario_key == 'K_phi_r':
        vp.K_phi_r = float(max(value, 100.0))
        vp.lltd = vp.lltd_elastic()

    elif scenario_key == 'roll_stiffness_tot':
        K_tot_old = vp.K_phi_f + vp.K_phi_r
        if K_tot_old > 0.0:
            ratio = vp.K_phi_f / K_tot_old
        else:
            ratio = 0.5
        vp.K_phi_f = value * ratio
        vp.K_phi_r = value * (1.0 - ratio)

    elif scenario_key == 'h_cg':
        vp.h_cg = value

    elif scenario_key == 'damping_phi':
        C_tot_old = vp.C_phi_f + vp.C_phi_r
        if C_tot_old > 0.0:
            ratio = vp.C_phi_f / C_tot_old
        else:
            ratio = 0.5
        vp.C_phi_f = value * ratio
        vp.C_phi_r = value * (1.0 - ratio)

    else:
        raise ValueError(f"Unknown scenario_key: '{scenario_key}'")

    return vp


# ---------------------------------------------------------------------------
# KPI Extraction from Telemetry DataFrame
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, maneuver: str = 'step_steer',
                 t_step: float = 0.5, t_ramp: float = 0.08,
                 steer_amp_deg: float = 3.0) -> dict:
    """
    Extract scalar KPIs from a telemetry DataFrame returned by
    simulate_step_steer() or simulate_double_lane_change().

    Parameters
    ----------
    df            : pd.DataFrame  — telemetry from a simulation run
    maneuver      : 'step_steer' or 'dlc'
    t_step        : time when steering starts (step steer only)
    t_ramp        : ramp duration (step steer only)
    steer_amp_deg : amplitude of steering input [deg]

    Returns
    -------
    dict of KPI name → float
    """
    kpis = {k: 0.0 for k, _, _ in KPI_METADATA}

    if df is None or df.empty:
        return kpis

    # -- Lateral acceleration peak ----------------------------------------
    ay_arr = df['ay_g'].values
    kpis['ay_peak_g'] = float(np.max(np.abs(ay_arr)))

    # -- Side-slip peak ---------------------------------------------------
    if 'beta_deg' in df.columns:
        kpis['beta_peak_deg'] = float(np.max(np.abs(df['beta_deg'].values)))

    # -- Roll angle peak --------------------------------------------------
    if 'phi_deg' in df.columns:
        kpis['phi_max_deg'] = float(np.max(np.abs(df['phi_deg'].values)))

    # -- Yaw acceleration peak --------------------------------------------
    if 'r_dot_deg_s2' in df.columns:
        kpis['r_dot_peak'] = float(np.max(np.abs(df['r_dot_deg_s2'].values)))

    # -- Dynamic LLTD from 4-wheel Fz ------------------------------------
    fz_cols = ['Fz_fl_N', 'Fz_fr_N', 'Fz_rl_N', 'Fz_rr_N']
    if all(c in df.columns for c in fz_cols):
        dFz_f = df['Fz_fr_N'].values - df['Fz_fl_N'].values   # [N]
        dFz_r = df['Fz_rr_N'].values - df['Fz_rl_N'].values
        total_dFz = np.abs(dFz_f) + np.abs(dFz_r)
        with np.errstate(invalid='ignore', divide='ignore'):
            lltd_arr = np.where(total_dFz > 1.0,
                                np.abs(dFz_f) / total_dFz,
                                np.nan)
        # LLTD peak at max |ay|
        idx_peak = int(np.argmax(np.abs(ay_arr)))
        kpis['lltd_peak'] = float(lltd_arr[idx_peak]) if not np.isnan(lltd_arr[idx_peak]) else 0.0

        # LLTD steady-state (last 20% of simulation)
        n_ss = max(int(len(lltd_arr) * 0.20), 1)
        ss_vals = lltd_arr[-n_ss:]
        ss_valid = ss_vals[~np.isnan(ss_vals)]
        kpis['lltd_ss'] = float(np.mean(ss_valid)) if len(ss_valid) > 0 else 0.0

        # L/R Fz asymmetry at peak Ay [%]
        Fz_tot_at_peak = (df['Fz_fl_N'].values[idx_peak] + df['Fz_fr_N'].values[idx_peak] +
                          df['Fz_rl_N'].values[idx_peak] + df['Fz_rr_N'].values[idx_peak])
        Fz_left_at_peak  = df['Fz_fl_N'].values[idx_peak] + df['Fz_rl_N'].values[idx_peak]
        Fz_right_at_peak = df['Fz_fr_N'].values[idx_peak] + df['Fz_rr_N'].values[idx_peak]
        if Fz_tot_at_peak > 1.0:
            kpis['dFz_asym_pct'] = float(abs(Fz_right_at_peak - Fz_left_at_peak) / Fz_tot_at_peak * 100.0)

    # -- Friction utilization max & Grip Efficiency ----------------------
    util_cols_f = ['util_fl_pct', 'util_fr_pct']
    util_cols_r = ['util_rl_pct', 'util_rr_pct']
    all_util_cols = util_cols_f + util_cols_r
    if all(c in df.columns for c in all_util_cols):
        kpis['util_front_max'] = float(df[util_cols_f].values.max())
        kpis['util_rear_max']  = float(df[util_cols_r].values.max())
        # Overall peak friction efficiency across the 4 corners
        idx_peak_util = int(np.argmax(np.abs(ay_arr)))
        kpis['eff_grip_pct'] = float(df[all_util_cols].iloc[idx_peak_util].mean())

    # -- Yaw rate & Lateral dynamics KPIs (step steer) --------------------
    r_arr = df['yaw_rate_deg_s'].values
    t_arr = df['t'].values

    if maneuver == 'step_steer':
        t_post = t_step + t_ramp
        mask_post = t_arr >= t_post
        if mask_post.any():
            r_post = r_arr[mask_post]
            t_post_arr = t_arr[mask_post]
            ay_post = ay_arr[mask_post]

            # Steady state = last 20% of post-step window
            n_ss = max(int(len(r_post) * 0.20), 1)
            r_ss = float(np.mean(r_post[-n_ss:]))
            ay_ss = float(np.mean(ay_post[-n_ss:]))
            beta_ss = float(np.mean(df['beta_deg'].values[-n_ss:])) if 'beta_deg' in df.columns else 0.0
            phi_ss = float(np.mean(df['phi_deg'].values[-n_ss:])) if 'phi_deg' in df.columns else 0.0

            kpis['r_ss_deg_s'] = r_ss

            # Peak
            idx_pk = int(np.argmax(np.abs(r_post)))
            r_peak = float(r_post[idx_pk])
            t_peak_val = float(t_post_arr[idx_pk] - t_step)
            kpis['r_peak_deg_s'] = abs(r_peak)

            # Overshoot
            if abs(r_ss) > 1e-3:
                kpis['overshoot_pct'] = max(0.0, (abs(r_peak) - abs(r_ss)) / abs(r_ss) * 100.0)

            # t90: time to reach 90% of r_ss (from t_step)
            thresh_90 = 0.90 * abs(r_ss)
            idx_90 = np.where(np.abs(r_post) >= thresh_90)[0]
            if len(idx_90) > 0:
                kpis['t90_s'] = float(t_post_arr[idx_90[0]] - t_step)

            # TB-Factor (ISO 7401 / Crolla): |beta_ss| * t_peak [deg*s]
            kpis['tb_factor'] = float(abs(beta_ss) * max(t_peak_val, 0.01))

            # Yaw & Ay Delay Times (from 50% of steer ramp to 50% response)
            t_50_steer = t_step + t_ramp / 2.0
            idx_50_r = np.where(np.abs(r_post) >= 0.50 * abs(r_ss))[0]
            if len(idx_50_r) > 0:
                kpis['yaw_delay_s'] = float(max(0.0, t_post_arr[idx_50_r[0]] - t_50_steer))

            idx_50_ay = np.where(np.abs(ay_post) >= 0.50 * abs(ay_ss))[0]
            if len(idx_50_ay) > 0:
                kpis['ay_delay_s'] = float(max(0.0, t_post_arr[idx_50_ay[0]] - t_50_steer))

            # Steering Sensitivity Gains
            effective_steer = max(abs(steer_amp_deg), 0.1)
            kpis['steer_sens_yaw'] = float(abs(r_ss) / effective_steer)
            kpis['steer_sens_ay']  = float(abs(ay_ss) / effective_steer)

            # Roll Gradient [deg/g]
            if abs(ay_ss) > 0.05:
                kpis['roll_gradient'] = float(abs(phi_ss) / abs(ay_ss))

            # t_settle: first time |r - r_ss| / |r_ss| < 0.02 and stays
            if abs(r_ss) > 1e-3:
                norm_err = np.abs(r_post - r_ss) / abs(r_ss)
                for ii in range(len(norm_err) - 1, -1, -1):
                    if norm_err[ii] > 0.02:
                        kpis['t_settle_s'] = float(t_post_arr[min(ii + 1, len(t_post_arr) - 1)] - t_step)
                        break

    elif maneuver == 'dlc':
        kpis['r_peak_deg_s'] = float(np.max(np.abs(r_arr)))
        kpis['r_ss_deg_s'] = 0.0

    return kpis


# ---------------------------------------------------------------------------
# 1D Parametric Sweep
# ---------------------------------------------------------------------------

def run_sensitivity_sweep_1d(
        base_params: VehicleParams,
        scenario_key: str,
        sweep_values,
        maneuver: str = 'step_steer',
        v_x: float = 15.0,
        steer_deg: float = 3.0,
        model_type: str = '14-DOF',
        tir_path: str = None,
        progress_cb=None
) -> dict:
    """
    Run a 1D parametric sweep: for each value in sweep_values, modify
    the VehicleParams according to scenario_key, run the maneuver simulation,
    and extract KPIs.

    Parameters
    ----------
    base_params   : VehicleParams — baseline vehicle configuration
    scenario_key  : str           — key from SCENARIO_CATALOG
    sweep_values  : array-like    — parameter values to sweep
    maneuver      : 'step_steer' or 'dlc'
    v_x           : float         — entry speed [m/s]
    steer_deg     : float         — steering amplitude [deg] (step steer)
    model_type    : '14-DOF' or '7-DOF'
    tir_path      : str or None   — path to .tir tire file
    progress_cb   : callable(n_done, n_total) — optional progress callback

    Returns
    -------
    dict with keys:
        'sweep_values'  : list of float
        'kpi_table'     : pd.DataFrame  (rows = sweep points, cols = KPIs)
        'telemetries'   : list of pd.DataFrame
        'param_label'   : str
        'scenario_key'  : str
        'maneuver'      : str
        'n_failed'      : int
    """
    sweep_values = list(sweep_values)
    N = len(sweep_values)

    meta = SCENARIO_CATALOG.get(scenario_key, {
        'label': scenario_key, 'unit': '', 'default_range': (0, 1), 'default_n': N
    })

    kpi_rows = []
    telemetries = []
    n_failed = 0

    for i, val in enumerate(sweep_values):
        if progress_cb is not None:
            progress_cb(i, N)

        vp = _apply_param(base_params, scenario_key, val)

        try:
            if maneuver == 'step_steer':
                res = simulate_step_steer(
                    vp, v_x=v_x, steer_deg=steer_deg,
                    model_type=model_type, tir_path=tir_path
                )
            elif maneuver == 'dlc':
                res = simulate_double_lane_change(
                    vp, v_entry=v_x,
                    model_type=model_type, tir_path=tir_path
                )
            else:
                raise ValueError(f"Unknown maneuver: '{maneuver}'")

            if res.get('success', False) and res.get('telemetry') is not None:
                df_telem = res['telemetry']
                kpis = compute_kpis(df_telem, maneuver=maneuver)
                kpis['param_value'] = val
                kpi_rows.append(kpis)
                telemetries.append(df_telem)
            else:
                n_failed += 1
                kpi_rows.append({'param_value': val})
                telemetries.append(pd.DataFrame())

        except Exception as exc:
            n_failed += 1
            kpi_rows.append({'param_value': val, '_error': str(exc)})
            telemetries.append(pd.DataFrame())

    if progress_cb is not None:
        progress_cb(N, N)

    kpi_table = pd.DataFrame(kpi_rows)
    # Reorder columns
    ordered_cols = ['param_value'] + [k for k, _, _ in KPI_METADATA]
    ordered_cols = [c for c in ordered_cols if c in kpi_table.columns]
    kpi_table = kpi_table[ordered_cols]

    return {
        'sweep_values': sweep_values,
        'kpi_table': kpi_table,
        'telemetries': telemetries,
        'param_label': meta['label'],
        'unit': meta.get('unit', ''),
        'scenario_key': scenario_key,
        'maneuver': maneuver,
        'n_failed': n_failed,
    }


# ---------------------------------------------------------------------------
# 2D Parametric Sweep (Heatmap)
# ---------------------------------------------------------------------------

def run_sensitivity_sweep_2d(
        base_params: VehicleParams,
        scenario_key_x: str,
        sweep_values_x,
        scenario_key_y: str,
        sweep_values_y,
        kpi_key: str = 'lltd_peak',
        maneuver: str = 'step_steer',
        v_x: float = 15.0,
        steer_deg: float = 3.0,
        model_type: str = '14-DOF',
        tir_path: str = None,
        progress_cb=None
) -> dict:
    """
    Run a 2D parametric sweep and return a heatmap matrix of a single KPI.

    The two parameters are varied independently (full factorial grid).
    Total runs = len(sweep_values_x) × len(sweep_values_y).

    Parameters
    ----------
    scenario_key_x  : first parameter (x-axis of heatmap)
    sweep_values_x  : values for x-axis parameter
    scenario_key_y  : second parameter (y-axis of heatmap)
    sweep_values_y  : values for y-axis parameter
    kpi_key         : which KPI to map (must be in KPI_METADATA keys)

    Returns
    -------
    dict with keys:
        'Z'             : np.ndarray shape (Ny, Nx)  — KPI heatmap matrix
        'x_values'      : list — x-axis parameter values
        'y_values'      : list — y-axis parameter values
        'x_label'       : str
        'y_label'       : str
        'kpi_key'       : str
        'kpi_label'     : str
        'n_total'       : int
        'n_failed'      : int
        'all_kpis'      : pd.DataFrame — all KPI values (multi-index)
    """
    sweep_values_x = list(sweep_values_x)
    sweep_values_y = list(sweep_values_y)
    Nx = len(sweep_values_x)
    Ny = len(sweep_values_y)
    n_total = Nx * Ny

    meta_x = SCENARIO_CATALOG.get(scenario_key_x, {'label': scenario_key_x})
    meta_y = SCENARIO_CATALOG.get(scenario_key_y, {'label': scenario_key_y})

    # KPI label
    kpi_label_map = {k: lbl for k, lbl, _ in KPI_METADATA}
    kpi_label = kpi_label_map.get(kpi_key, kpi_key)

    Z = np.full((Ny, Nx), np.nan)
    all_rows = []
    n_failed = 0
    n_done = 0

    for iy, vy in enumerate(sweep_values_y):
        for ix, vx_param in enumerate(sweep_values_x):
            if progress_cb is not None:
                progress_cb(n_done, n_total)

            # Apply both parameters sequentially (order matters if they interact)
            vp = _apply_param(base_params, scenario_key_x, vx_param)
            vp = _apply_param(vp, scenario_key_y, vy)

            try:
                if maneuver == 'step_steer':
                    res = simulate_step_steer(
                        vp, v_x=v_x, steer_deg=steer_deg,
                        model_type=model_type, tir_path=tir_path
                    )
                else:
                    res = simulate_double_lane_change(
                        vp, v_entry=v_x,
                        model_type=model_type, tir_path=tir_path
                    )

                if res.get('success', False) and res.get('telemetry') is not None:
                    kpis = compute_kpis(res['telemetry'], maneuver=maneuver)
                    Z[iy, ix] = kpis.get(kpi_key, np.nan)
                    row = {'x_val': vx_param, 'y_val': vy, **kpis}
                    all_rows.append(row)
                else:
                    n_failed += 1

            except Exception:
                n_failed += 1

            n_done += 1

    if progress_cb is not None:
        progress_cb(n_total, n_total)

    all_kpis_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    return {
        'Z': Z,
        'x_values': sweep_values_x,
        'y_values': sweep_values_y,
        'x_label': meta_x['label'],
        'y_label': meta_y['label'],
        'kpi_key': kpi_key,
        'kpi_label': kpi_label,
        'n_total': n_total,
        'n_failed': n_failed,
        'all_kpis': all_kpis_df,
        'scenario_key_x': scenario_key_x,
        'scenario_key_y': scenario_key_y,
        'maneuver': maneuver,
    }
