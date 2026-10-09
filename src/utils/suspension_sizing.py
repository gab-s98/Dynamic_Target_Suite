"""
Suspension Rates, ARB Sizing & 4-Ratio Balance Analyzer
Based on fundamental suspension design and balance principles.
"""

import numpy as np
import pandas as pd

def calculate_suspension_and_arb_sizing(
    m_total_kg,
    wd_f,
    lltd_target,
    t_f_m,
    t_r_m,
    f_ride_f_hz=3.0,
    f_ride_r_hz=3.3,
    mr_spring_f=0.80,
    mr_spring_r=0.80,
    mr_arb_f=0.40,
    mr_arb_r=0.40,
    unsprung_mass_pct=0.12,
    arb_arm_length_f_mm=120.0,
    arb_arm_length_r_mm=120.0,
    arb_bar_length_f_mm=350.0,
    arb_bar_length_r_mm=350.0,
    arb_material="Steel",  # "Steel" (G=79.3 GPa) or "Titanium" (G=45.0 GPa)
    k_spring_installed_f_Nmm=40.0,
    k_spring_installed_r_Nmm=45.0,
    spring_preload_f_mm=5.0,
    spring_preload_r_mm=5.0,
    damper_stroke_total_mm=57.0,
    bumpstop_length_mm=20.0,
    damper_stroke_bump_mm=35.0,
    damper_stroke_reb_mm=30.0,
    bumpstop_gap_mm=20.0,
    static_ride_height_f_mm=30.0,
    static_ride_height_r_mm=35.0,
    k_tire_Nmm=110.0,
    k_phi_target_total_Nmdeg=None
):
    """
    Calculates required spring rates, roll stiffnesses, and hollow ARB tube dimensions
    to achieve target LLTD and Flat Ride frequencies, comparing with installed real damper & spring setup,
    including exact spring preload and 57mm TTX25 stroke kinematic partitioning.
    """
    # Sprung Mass
    m_sprung = m_total_kg * (1.0 - unsprung_mass_pct)
    m_sprung_f = m_sprung * wd_f          # front axle sprung mass
    m_sprung_r = m_sprung * (1.0 - wd_f)  # rear axle sprung mass
    
    # 1. Ride Frequencies & Wheel Rates: K_w = (2 * pi * f)^2 * (m_sprung / 2) [N/m]
    m_corner_f = m_sprung_f / 2.0
    m_corner_r = m_sprung_r / 2.0
    
    Kw_f_Npm = (2.0 * np.pi * f_ride_f_hz)**2 * m_corner_f
    Kw_r_Npm = (2.0 * np.pi * f_ride_r_hz)**2 * m_corner_r
    
    Kw_f_Nmm = Kw_f_Npm / 1000.0
    Kw_r_Nmm = Kw_r_Npm / 1000.0
    
    # Target Spring Stiffness: K_spring = K_w / MR^2
    Kspring_f_Nmm = Kw_f_Nmm / (mr_spring_f**2)
    Kspring_r_Nmm = Kw_r_Nmm / (mr_spring_r**2)
    
    # Installed Spring actual Wheel Rates and Ride Frequencies
    Kw_inst_f_Nmm = k_spring_installed_f_Nmm * (mr_spring_f**2)
    Kw_inst_r_Nmm = k_spring_installed_r_Nmm * (mr_spring_r**2)
    fn_inst_f_hz = (1.0 / (2.0 * np.pi)) * np.sqrt((Kw_inst_f_Nmm * 1000.0) / max(m_corner_f, 1.0))
    fn_inst_r_hz = (1.0 / (2.0 * np.pi)) * np.sqrt((Kw_inst_r_Nmm * 1000.0) / max(m_corner_r, 1.0))
    
    # 1g Static Forces on Damper & Spring
    f_z0_f_N = m_corner_f * 9.81
    f_z0_r_N = m_corner_r * 9.81
    
    f_damper_static_f_N = f_z0_f_N / max(mr_spring_f, 0.01)
    f_damper_static_r_N = f_z0_r_N / max(mr_spring_r, 0.01)
    
    f_preload_f_N = k_spring_installed_f_Nmm * spring_preload_f_mm
    f_preload_r_N = k_spring_installed_r_Nmm * spring_preload_r_mm
    
    # Static Sag / Damper Compression under 1g (governed by spring rate and installed preload)
    if f_damper_static_f_N > f_preload_f_N:
        damper_sag_f_mm = (f_damper_static_f_N - f_preload_f_N) / max(k_spring_installed_f_Nmm, 0.1)
    else:
        damper_sag_f_mm = 0.0
        
    if f_damper_static_r_N > f_preload_r_N:
        damper_sag_r_mm = (f_damper_static_r_N - f_preload_r_N) / max(k_spring_installed_r_Nmm, 0.1)
    else:
        damper_sag_r_mm = 0.0
        
    z_static_sag_f_mm = damper_sag_f_mm / max(mr_spring_f, 0.01)
    z_static_sag_r_mm = damper_sag_r_mm / max(mr_spring_r, 0.01)
    
    # Available bump & rebound strokes from 1g static position
    damper_bump_f_mm = max(damper_stroke_total_mm - damper_sag_f_mm, 0.0)
    damper_bump_r_mm = max(damper_stroke_total_mm - damper_sag_r_mm, 0.0)
    
    damper_reb_f_mm = damper_sag_f_mm
    damper_reb_r_mm = damper_sag_r_mm
    
    wheel_bump_f_mm = damper_bump_f_mm / max(mr_spring_f, 0.01)
    wheel_bump_r_mm = damper_bump_r_mm / max(mr_spring_r, 0.01)
    
    wheel_reb_f_mm = damper_reb_f_mm / max(mr_spring_f, 0.01)
    wheel_reb_r_mm = damper_reb_r_mm / max(mr_spring_r, 0.01)
    
    # Damper travel headroom before bumpstop engagement
    bump_headroom_f_mm = max((damper_bump_f_mm - bumpstop_length_mm) / max(mr_spring_f, 0.01), 0.0)
    bump_headroom_r_mm = max((damper_bump_r_mm - bumpstop_length_mm) / max(mr_spring_r, 0.01), 0.0)
    damper_usage_bump_pct = min(((damper_stroke_total_mm - damper_bump_f_mm) / max(damper_stroke_total_mm, 1.0)) * 100.0, 100.0)

    # 2. Spring Contribution to Roll Stiffness: K_phi_spring = 0.5 * K_w * t^2 [Nm/rad]
    Kphi_spring_f_Nmrad = 0.5 * Kw_f_Npm * (t_f_m**2)
    Kphi_spring_r_Nmrad = 0.5 * Kw_r_Npm * (t_r_m**2)
    
    Kphi_spring_f_Nmdeg = Kphi_spring_f_Nmrad * (np.pi / 180.0)
    Kphi_spring_r_Nmdeg = Kphi_spring_r_Nmrad * (np.pi / 180.0)
    
    # Natural Spring LLTD (without ARBs)
    lltd_springs_only = Kphi_spring_f_Nmrad / (Kphi_spring_f_Nmrad + Kphi_spring_r_Nmrad)
    
    # 3. Required Total Roll Stiffness to satisfy Target LLTD with minimal chassis roll
    h_cg_m = 0.28
    roll_moment_per_g = m_total_kg * 9.81 * h_cg_m  # approx Nm/g
    if k_phi_target_total_Nmdeg is not None:
        Kphi_total_req_Nmdeg = float(k_phi_target_total_Nmdeg)
        target_roll_grad_degg = roll_moment_per_g / max(Kphi_total_req_Nmdeg, 1.0)
    else:
        target_roll_grad_degg = 0.85
        Kphi_total_req_Nmdeg = roll_moment_per_g / target_roll_grad_degg
    Kphi_total_req_Nmrad = Kphi_total_req_Nmdeg * (180.0 / np.pi)
    
    # Required Front and Rear Roll Stiffness
    Kphi_f_target_Nmdeg = lltd_target * Kphi_total_req_Nmdeg
    Kphi_r_target_Nmdeg = (1.0 - lltd_target) * Kphi_total_req_Nmdeg
    
    # ARB contributions required: K_phi_arb = K_phi_target - K_phi_spring
    Kphi_arb_f_Nmdeg = max(Kphi_f_target_Nmdeg - Kphi_spring_f_Nmdeg, 0.0)
    Kphi_arb_r_Nmdeg = max(Kphi_r_target_Nmdeg - Kphi_spring_r_Nmdeg, 0.0)
    
    Kphi_arb_f_Nmrad = Kphi_arb_f_Nmdeg * (180.0 / np.pi)
    Kphi_arb_r_Nmrad = Kphi_arb_r_Nmdeg * (180.0 / np.pi)

    # 4. Tire Vertical Compliance Roll Stiffness & Series Connection (Dual-Rate Series Spring Model)
    Kt_tire_Npm = max(k_tire_Nmm, 1.0) * 1000.0
    Kphi_tire_f_Nmrad = 0.5 * Kt_tire_Npm * (t_f_m**2)
    Kphi_tire_r_Nmrad = 0.5 * Kt_tire_Npm * (t_r_m**2)
    Kphi_tire_f_Nmdeg = Kphi_tire_f_Nmrad * (np.pi / 180.0)
    Kphi_tire_r_Nmdeg = Kphi_tire_r_Nmrad * (np.pi / 180.0)

    # Suspension roll stiffness (springs + ARB)
    Kphi_susp_f_Nmrad = Kphi_spring_f_Nmrad + Kphi_arb_f_Nmrad
    Kphi_susp_r_Nmrad = Kphi_spring_r_Nmrad + Kphi_arb_r_Nmrad

    # Effective roll stiffness in series with tire vertical compliance (1/K_tot = 1/K_susp + 1/K_tire):
    Kphi_series_f_Nmrad = (Kphi_susp_f_Nmrad * Kphi_tire_f_Nmrad) / max(Kphi_susp_f_Nmrad + Kphi_tire_f_Nmrad, 1.0)
    Kphi_series_r_Nmrad = (Kphi_susp_r_Nmrad * Kphi_tire_r_Nmrad) / max(Kphi_susp_r_Nmrad + Kphi_tire_r_Nmrad, 1.0)
    Kphi_series_f_Nmdeg = Kphi_series_f_Nmrad * (np.pi / 180.0)
    Kphi_series_r_Nmdeg = Kphi_series_r_Nmrad * (np.pi / 180.0)
    
    # 4. ARB Tube Sizing (Torsion of cylindrical bar)
    G_Pa = 79.3e9 if arb_material == "Steel" else 45.0e9
    wall_thk_m = 0.002
    
    def _size_arb_outer_diameter(Kphi_arb_Nmrad, L_bar_mm, b_arm_mm, mr_arb, t_axle_m):
        if Kphi_arb_Nmrad <= 10.0:
            return 0.0
        L_b_m = L_bar_mm / 1000.0
        b_a_m = b_arm_mm / 1000.0
        K_theta = Kphi_arb_Nmrad / ((t_axle_m / b_a_m)**2 * mr_arb**2)
        J_req = (K_theta * L_b_m) / G_Pa
        for Do_try in np.linspace(0.008, 0.035, 200):
            Di_try = max(Do_try - 2.0 * wall_thk_m, 0.001)
            J_try = (np.pi / 32.0) * (Do_try**4 - Di_try**4)
            if J_try >= J_req:
                return Do_try * 1000.0
        return 25.0
        
    Do_f_mm = _size_arb_outer_diameter(Kphi_arb_f_Nmrad, arb_bar_length_f_mm, arb_arm_length_f_mm, mr_arb_f, t_f_m)
    Do_r_mm = _size_arb_outer_diameter(Kphi_arb_r_Nmrad, arb_bar_length_r_mm, arb_arm_length_r_mm, mr_arb_r, t_r_m)
    
    Kw_arb_f_Npm = Kphi_arb_f_Nmrad / (t_f_m**2)
    Kw_arb_r_Npm = Kphi_arb_r_Nmrad / (t_r_m**2)

    return {
        'm_sprung_kg': float(m_sprung),
        'Kw_f_Nmm': float(Kw_f_Nmm),
        'Kw_r_Nmm': float(Kw_r_Nmm),
        'Kspring_f_Nmm': float(Kspring_f_Nmm),
        'Kspring_r_Nmm': float(Kspring_r_Nmm),
        'Kw_inst_f_Nmm': float(Kw_inst_f_Nmm),
        'Kw_inst_r_Nmm': float(Kw_inst_r_Nmm),
        'fn_inst_f_hz': float(fn_inst_f_hz),
        'fn_inst_r_hz': float(fn_inst_r_hz),
        'z_static_sag_f_mm': float(z_static_sag_f_mm),
        'z_static_sag_r_mm': float(z_static_sag_r_mm),
        'bump_headroom_f_mm': float(bump_headroom_f_mm),
        'bump_headroom_r_mm': float(bump_headroom_r_mm),
        'damper_usage_bump_pct': float(damper_usage_bump_pct),
        'Kphi_spring_f_Nmdeg': float(Kphi_spring_f_Nmdeg),
        'Kphi_spring_r_Nmdeg': float(Kphi_spring_r_Nmdeg),
        'lltd_springs_only_pct': float(lltd_springs_only * 100.0),
        'Kphi_total_req_Nmdeg': float(Kphi_total_req_Nmdeg),
        'Kphi_arb_f_Nmdeg': float(Kphi_arb_f_Nmdeg),
        'Kphi_arb_r_Nmdeg': float(Kphi_arb_r_Nmdeg),
        'Kphi_arb_f_Nmrad': float(Kphi_arb_f_Nmrad),
        'Kphi_arb_r_Nmrad': float(Kphi_arb_r_Nmrad),
        'Kw_arb_f_Npm': float(Kw_arb_f_Npm),
        'Kw_arb_r_Npm': float(Kw_arb_r_Npm),
        'damper_sag_f_mm': float(damper_sag_f_mm),
        'damper_sag_r_mm': float(damper_sag_r_mm),
        'damper_bump_f_mm': float(damper_bump_f_mm),
        'damper_bump_r_mm': float(damper_bump_r_mm),
        'damper_reb_f_mm': float(damper_reb_f_mm),
        'damper_reb_r_mm': float(damper_reb_r_mm),
        'wheel_bump_f_mm': float(wheel_bump_f_mm),
        'wheel_bump_r_mm': float(wheel_bump_r_mm),
        'wheel_reb_f_mm': float(wheel_reb_f_mm),
        'wheel_reb_r_mm': float(wheel_reb_r_mm),
        'Do_arb_f_mm': float(Do_f_mm),
        'Do_arb_r_mm': float(Do_r_mm),
        'roll_grad_target_degg': float(target_roll_grad_degg),
        'Kphi_tire_f_Nmdeg': float(Kphi_tire_f_Nmdeg),
        'Kphi_tire_r_Nmdeg': float(Kphi_tire_r_Nmdeg),
        'Kphi_susp_f_Nmdeg': float(Kphi_susp_f_Nmrad * np.pi / 180.0),
        'Kphi_susp_r_Nmdeg': float(Kphi_susp_r_Nmrad * np.pi / 180.0),
        'Kphi_series_f_Nmdeg': float(Kphi_series_f_Nmdeg),
        'Kphi_series_r_Nmdeg': float(Kphi_series_r_Nmdeg)
    }


def evaluate_4_ratios_balance(wd_f, lltd, aero_bal, t_f, t_r, v_range=(5.0, 30.0), n_pts=26):
    """
    Analyzes the 4 Fundamental Ratios and their dynamic evolution across speeds.
    """
    speeds = np.linspace(v_range[0], v_range[1], n_pts)
    track_ratio = t_f / t_r
    
    # High-speed stability margin: recommended aero_bal <= LLTD - 0.02 (understeering aerodynamic bias)
    stability_margin_pct = (lltd - aero_bal) * 100.0
    
    # Combined effective balance across velocity (from mechanical dominance at low speed to aero dominance at high speed)
    # Aero downforce fraction: F_aero / (F_mass + F_aero)
    # Approx: at 10 m/s, aero is ~15% of total normal load; at 28 m/s, aero is ~55% of total load.
    balance_v = []
    for v in speeds:
        # Aero weight factor: w_aero = (0.5 * rho * S * Cz * v^2) / (m*g + 0.5 * rho * S * Cz * v^2)
        q_aero = 0.5 * 1.225 * 1.1 * 3.5 * (v**2)
        w_tot = 230.0 * 9.81 + q_aero
        w_aero_frac = q_aero / w_tot
        
        # Effective dynamic front grip bias:
        # Mechanical grip bias approx: wd_f * (1 - LLTD) + (1 - wd_f) * LLTD
        # Aero grip bias: aero_bal
        dyn_front_bias = (1.0 - w_aero_frac) * (wd_f * 0.5 + (1.0 - lltd) * 0.5) + w_aero_frac * aero_bal
        balance_v.append({
            'v_x': float(v),
            'aero_frac_pct': float(w_aero_frac * 100.0),
            'dyn_front_bias_pct': float(dyn_front_bias * 100.0),
            'stability_margin_pct': float((lltd - aero_bal) * 100.0)
        })
        
    return {
        'wd_f_pct': float(wd_f * 100.0),
        'lltd_pct': float(lltd * 100.0),
        'aero_bal_pct': float(aero_bal * 100.0),
        'track_ratio': float(track_ratio),
        'stability_margin_pct': float(stability_margin_pct),
        'is_high_speed_stable': bool(stability_margin_pct >= 2.0),
        'balance_curve': pd.DataFrame(balance_v)
    }
