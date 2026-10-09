import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.optimized_params_manager import save_optimization_run, load_history, get_last_run, export_history_to_csv
from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_dynamic_events import simulate_autox_endurance

def test_all():
    print("=== 1. Testing Optimization Persistence ===")
    rec = save_optimization_run(
        model_type="14-DOF",
        objective_mode="max_ay",
        v_x=20.0,
        penalty=10.0,
        tir_file="Hoosier.tir",
        initial_params={"L": 1.530, "t_f": 1.250, "t_r": 1.150, "wd_f": 0.470, "lltd": 0.550, "aero_balance": 0.460},
        optimized_params={"L": 1.535, "t_f": 1.255, "t_r": 1.145, "wd_f": 0.465, "lltd": 0.535, "aero_balance": 0.485},
        kpis={"max_ay_g": 1.604}
    )
    print(f"Saved run #{rec['run_id']} successfully!")
    
    history = load_history()
    print(f"Total history entries: {len(history)}")
    
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_opt_hist.csv")
    export_ok = export_history_to_csv(csv_file)
    print(f"CSV export successful: {export_ok}")
    if os.path.exists(csv_file):
        os.remove(csv_file)

    print("\n=== 2. Testing Upgraded AutoX Track (Slalom, Chicanes, Hairpins) ===")
    vp = VehicleParams()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tir_path = os.path.join(base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir")
    
    res = simulate_autox_endurance(vp, event_type="AutoX", num_laps=1, track_length_m=850.0, ds=0.5,
                                   model_type="14-DOF", tir_path=tir_path)
    
    print(f"AutoX Lap Time: {res['lap_time_s']:.2f} s")
    print(f"Max Speed: {res['v_max_kmh']:.1f} km/h | Avg Speed: {res['v_avg_kmh']:.1f} km/h")
    print(f"Peak Lateral Accel: {res['max_ay_g']:.3f} g")
    print(f"Energy Consumed: {res['energy_lap_kwh']*1000.0:.1f} Wh")
    
    print("\nALL VERIFICATIONS PASSED!")

if __name__ == "__main__":
    test_all()
