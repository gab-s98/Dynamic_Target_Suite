"""
Steering Dynamics & Driver Effort Module
Calculates steering wheel torque, driver effort, pneumatic & mechanical trail,
and kingpin moment contributions based on standard steering axis geometry.
"""

import numpy as np
import pandas as pd
from src.models.tire_model_mf61 import mf61_lateral_force, mf61_aligning_torque

def calculate_steering_torque(
    delta_deg,
    ay_g,
    v_x,
    Fz_front_single,
    tp,
    caster_deg=4.5,
    kpi_deg=8.0,
    mech_trail_mm=18.0,
    scrub_radius_mm=12.0,
    steering_ratio=4.5,  # deg steering wheel per deg road wheel (approx 120-140 deg lock to lock)
    rack_efficiency=0.90,
    tire_radius_m=0.203,  # 16x7.5-10
    caster_trail_coeffs=None,
    scrub_radius_coeffs=None
):
    """
    Calculates driver steering wheel torque for a single operating point.
    
    Parameters:
    -----------
    delta_deg : float
        Road wheel steer angle [deg].
    ay_g : float
        Lateral acceleration [g].
    v_x : float
        Vehicle forward speed [m/s].
    Fz_front_single : float
        Vertical load on one front tire [N].
    tp : TireParams
        Tire parameters object (MF6.1).
    caster_deg : float
        Caster angle [deg] (positive tilted rearward).
    kpi_deg : float
        Kingpin inclination angle [deg] (positive tilted inward).
    mech_trail_mm : float
        Mechanical trail at ground [mm] (t_m = r_tire * tan(caster) / cos(kpi)).
    scrub_radius_mm : float
        Ground scrub radius [mm] (positive = kingpin axis inboard of tire center).
    steering_ratio : float
        Steering ratio GR = delta_sw / delta_wheel (typically 3.5 to 5.5 for FSAE).
    rack_efficiency : float
        Mechanical transmission efficiency of rack, pinions, and U-joints (~0.85-0.92).
    tire_radius_m : float
        Loaded tire rolling radius [m].
    caster_trail_coeffs : list, optional
        Polynomial coeffs [a3, a2, a1, a0] for Caster Trail [m] vs delta_sw [deg].
    scrub_radius_coeffs : list, optional
        Polynomial coeffs [a3, a2, a1, a0] for Scrub Radius [m] vs delta_sw [deg].
        
    Returns:
    --------
    dict containing:
        - T_sw_Nm: Steering wheel torque [Nm]
        - Mz_kp_total: Total kingpin moment per wheel [Nm]
        - Fy_front: Lateral force on one front tire [N]
        - t_p_mm: Pneumatic trail [mm]
        - t_m_mm: Mechanical trail [mm]
        - Mz_tire: Tire self-aligning moment [Nm]
        - M_kpi_weight: Weight-jacking jacking moment [Nm]
        - is_over_limit: True if T_sw > 5.0 Nm (Schwarzenegger warning)
    """
    delta_rad = np.deg2rad(delta_deg)
    caster_rad = np.deg2rad(caster_deg)
    kpi_rad = np.deg2rad(kpi_deg)
    delta_sw_deg = delta_deg * steering_ratio
    
    # Calculate effective slip angle on front axle
    # Kinematic approx: alpha_f approx delta - (L*r)/v
    # For standalone estimation from delta & ay:
    alpha_f_rad = max(abs(delta_rad) * 0.70, np.deg2rad(0.5))
    
    # Tire forces from Pacejka MF6.1
    Fy_f = float(mf61_lateral_force(alpha_f_rad, Fz_front_single, 0.0, tp))
    Mz_tire = float(mf61_aligning_torque(alpha_f_rad, Fz_front_single, 0.0, tp))
    
    # Pneumatic trail: t_p = -Mz / Fy (in meters)
    if abs(Fy_f) > 10.0:
        t_p_m = abs(Mz_tire / Fy_f)
    else:
        t_p_m = 0.020  # default 20mm at zero slip
    t_p_mm = t_p_m * 1000.0
    
    # Mechanical trail: evaluate polynomial if provided, else use scalar or derive
    if caster_trail_coeffs is not None and len(caster_trail_coeffs) == 4:
        a3, a2, a1, a0 = caster_trail_coeffs
        t_m_m = float(a3 * (delta_sw_deg**3) + a2 * (delta_sw_deg**2) + a1 * delta_sw_deg + a0)
        t_m_mm = t_m_m * 1000.0
    elif mech_trail_mm is None or mech_trail_mm <= 0:
        t_m_m = (tire_radius_m * np.tan(caster_rad)) / np.cos(kpi_rad)
        t_m_mm = t_m_m * 1000.0
    else:
        t_m_m = mech_trail_mm / 1000.0
        t_m_mm = float(mech_trail_mm)
        
    # Scrub radius: evaluate polynomial if provided, else use scalar
    if scrub_radius_coeffs is not None and len(scrub_radius_coeffs) == 4:
        a3, a2, a1, a0 = scrub_radius_coeffs
        scrub_m = float(a3 * (delta_sw_deg**3) + a2 * (delta_sw_deg**2) + a1 * delta_sw_deg + a0)
        scrub_radius_mm = scrub_m * 1000.0
    else:
        scrub_m = scrub_radius_mm / 1000.0
    
    # Kingpin Moment components:
    # 1. Lateral force moment around kingpin axis: Fy * (t_p + t_m) * cos(caster) * cos(kpi)
    M_lat = Fy_f * (t_p_m + t_m_m) * np.cos(caster_rad) * np.cos(kpi_rad)
    
    # 2. Kingpin Weight-jacking moment (lifting wheel as steered): Fz * sin(kpi) * sin(delta) * scrub
    M_kpi_weight = Fz_front_single * np.sin(kpi_rad) * np.sin(delta_rad) * scrub_m
    
    # 3. Aligning torque projection onto Kingpin axis
    M_align_proj = Mz_tire * np.cos(caster_rad) * np.cos(kpi_rad)
    
    # Total kingpin torque per steered wheel
    # Note: M_align_proj = Mz_tire * cos(caster) * cos(kpi) accounts for the pneumatic
    # self-aligning torque projected onto the kingpin axis.
    Mz_kp_total = M_lat + M_kpi_weight + M_align_proj
    
    # Steering Wheel Torque (2 front wheels combined, scaled by steering ratio and efficiency)
    # T_sw = (2 * Mz_kp) / (GR * eta)
    T_sw_Nm = (2.0 * abs(Mz_kp_total)) / (steering_ratio * rack_efficiency)
    
    return {
        'T_sw_Nm': float(T_sw_Nm),
        'Mz_kp_total': float(Mz_kp_total),
        'Fy_front': float(Fy_f),
        't_p_mm': float(t_p_mm),
        't_m_mm': float(t_m_mm),
        't_tot_mm': float(t_p_mm + t_m_mm),
        'Mz_tire': float(Mz_tire),
        'M_kpi_weight': float(M_kpi_weight),
        'is_over_limit': bool(T_sw_Nm > 5.0)
    }


def generate_steering_torque_curve(
    v_x,
    df_handling,
    vp,
    tp,
    caster_deg=None,
    kpi_deg=None,
    mech_trail_mm=None,
    scrub_radius_mm=None,
    steering_ratio=None,
    rack_efficiency=0.90
):
    """
    Generates steering wheel torque curve across lateral acceleration for a given speed.
    """
    if df_handling is None or df_handling.empty:
        return pd.DataFrame()
        
    c_deg = caster_deg if caster_deg is not None else float(getattr(vp, 'caster_f', 4.5))
    k_deg = kpi_deg if kpi_deg is not None else float(getattr(vp, 'kpi_f', 8.0))
    mt_mm = mech_trail_mm if mech_trail_mm is not None else float(getattr(vp, 'mechanical_trail_f', 0.018) * 1000.0)
    sr_mm = scrub_radius_mm if scrub_radius_mm is not None else float(getattr(vp, 'scrub_radius_f', 0.012) * 1000.0)
    sr_ratio = steering_ratio if steering_ratio is not None else float(getattr(vp, 'steering_ratio', 4.5))
    ct_coeffs = getattr(vp, 'caster_trail_f_coeffs', None)
    sr_coeffs = getattr(vp, 'scrub_radius_f_coeffs', None)

    pts = []
    # Dynamic front vertical load with aero downforce and lateral load transfer
    F_dwnf = 0.5 * vp.rho * vp.S * vp.Cz * (v_x**2)
    Fz_front_axle = vp.m * 9.81 * vp.wd_f + F_dwnf * vp.aero_balance
    Fz_front_single = max(Fz_front_axle / 2.0, 100.0)
    
    for _, row in df_handling.iterrows():
        ay_g = float(row.get('ay_g', 0.0))
        delta_deg = float(row.get('delta_deg', row.get('slip_diff_deg', 0.0) + (ay_g * 9.81 * vp.L)/(v_x**2)))
        
        st_res = calculate_steering_torque(
            delta_deg=delta_deg,
            ay_g=ay_g,
            v_x=v_x,
            Fz_front_single=Fz_front_single,
            tp=tp,
            caster_deg=c_deg,
            kpi_deg=k_deg,
            mech_trail_mm=mt_mm,
            scrub_radius_mm=sr_mm,
            steering_ratio=sr_ratio,
            rack_efficiency=rack_efficiency,
            caster_trail_coeffs=ct_coeffs,
            scrub_radius_coeffs=sr_coeffs
        )
        st_res['ay_g'] = ay_g
        st_res['delta_deg'] = delta_deg
        st_res['v_x'] = v_x
        pts.append(st_res)
        
    return pd.DataFrame(pts)
