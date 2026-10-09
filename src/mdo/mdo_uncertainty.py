import os
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_steady_state import evaluate_steady_state_kpis

def run_handling_sweep(v_x=12.0, wd_f=0.5, lmuy_mult=1.0, delta_range_deg=(0, 6), num_points=30):
    """
    Runs a steady-state handling sweep over a range of steering angles.
    Returns a pandas DataFrame with the results.
    """
    vp = VehicleParams()
    vp.LMUY = lmuy_mult # Temporarily set LMUY
    
    deltas = np.linspace(delta_range_deg[0], delta_range_deg[1], num_points)
    results = []
    
    for delta_deg in deltas:
        try:
            res = evaluate_steady_state_kpis(
                v_x=v_x, 
                delta_deg=delta_deg, 
                wd_f=wd_f, 
                lltd=0.5, # default
                aero_bal=0.485, # default
                L=vp.L, 
                t_f=vp.t_f, 
                t_r=vp.t_r,
                model_type='3-DOF'
            )
            
            beta_res = res['beta_rad']
            r_res = res['yaw_rate_rads']
            
            # If the solver diverged, NewtonSolver might return huge unrealistic values
            if abs(beta_res) > 0.5 or abs(r_res) > 5.0:
                break # Vehicle spun out / solver diverged
                
            ay = v_x * r_res # Approximate lateral accel
            
            results.append({
                'delta_deg': delta_deg,
                'yaw_rate': r_res,
                'sideslip': np.rad2deg(beta_res),
                'ay': ay,
                'ay_g': ay / 9.81
            })
        except Exception:
            # Solver failed to converge (vehicle lost control)
            break
            
    return pd.DataFrame(results)

def get_uncertainty_envelope(v_x=12.0, wd_f=0.5, uncertainty_pct=15.0, delta_range_deg=(0, 6)):
    df_nom = run_handling_sweep(v_x, wd_f, 1.0, delta_range_deg)
    df_low = run_handling_sweep(v_x, wd_f, 1.0 - uncertainty_pct/100.0, delta_range_deg)
    df_high = run_handling_sweep(v_x, wd_f, 1.0 + uncertainty_pct/100.0, delta_range_deg)
    return df_nom, df_low, df_high
