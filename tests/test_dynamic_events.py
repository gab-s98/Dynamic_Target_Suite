import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_skidpad import simulate_skidpad
from src.mdo.mdo_acceleration import simulate_acceleration
from src.mdo.mdo_dynamic_events import simulate_autox_endurance, simulate_double_lane_change

def test_skidpad_multi_model():
    """Verify Skidpad simulation executes across 3-DOF, 7-DOF, and 14-DOF models."""
    vp = VehicleParams()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    
    r3 = simulate_skidpad(vp, radius=9.125, model_type='3-DOF', tir_path=tir_path)
    assert r3['success'] is True
    assert 3.5 <= r3['time'] <= 7.0
    assert 1.0 <= r3['ay'] <= 2.5
    
    r7 = simulate_skidpad(vp, radius=9.125, model_type='7-DOF', tir_path=tir_path)
    assert r7['success'] is True
    assert 3.5 <= r7['time'] <= 7.0
    
    r14 = simulate_skidpad(vp, radius=9.125, model_type='14-DOF', tir_path=tir_path)
    assert r14['success'] is True
    assert 3.5 <= r14['time'] <= 7.0
    assert r14['phi_deg'] > 0.5 # Has roll angle in 14-DOF

def test_acceleration_and_energy():
    """Verify Acceleration simulation computes time, terminal speed, and energy."""
    vp = VehicleParams()
    res = simulate_acceleration(vp, distance=75.0, is_awd=False)
    assert res['success'] is True
    assert 3.0 <= res['time'] <= 5.5
    assert res['v_max'] >= 25.0 # >= 90 km/h
    assert res['energy_wh'] > 10.0 # Consumes electrical energy

def test_autox_and_endurance_energy():
    """Verify AutoX sprint and Endurance 22km multi-lap battery consumption."""
    vp = VehicleParams()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    
    ax = simulate_autox_endurance(vp, event_type='AutoX', num_laps=1, track_length_m=850.0, model_type='14-DOF', tir_path=tir_path)
    assert ax['success'] is True
    assert 45.0 <= ax['lap_time_s'] <= 75.0
    assert ax['energy_lap_kwh'] > 0.05
    
    endur = simulate_autox_endurance(vp, event_type='Endurance', num_laps=22, track_length_m=850.0, model_type='14-DOF', tir_path=tir_path)
    assert endur['success'] is True
    assert endur['total_time_s'] > 900.0
    assert endur['total_energy_kwh'] > 1.0
    assert 0.0 <= endur['battery_soc_end_pct'] <= 100.0
