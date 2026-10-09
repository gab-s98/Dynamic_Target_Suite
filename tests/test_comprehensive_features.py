import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo.mdo_global_optimizer import GlobalOptimizer
from src.mdo.mdo_target_optimization import generate_mmm_diagram, generate_handling_sweep
from src.mdo.mdo_dynamic_events import simulate_step_steer, simulate_double_lane_change, simulate_autox_endurance
from src.mdo.mdo_direct_collocation import solve_direct_collocation_racing_line
from src.ui.interactive_driver_14dof import PhysicsThread14DOF, GamepadManager
import time

def run_tests():
    print("==================================================")
    print("  DYNAMIC TARGET SUITE - COMPREHENSIVE TEST SUITE ")
    print("==================================================")

    # 1. Optimizer test
    print("\n--- 1. Testing Global Optimizer (High Ay & Understeer Penalty Fix) ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tir_path = os.path.join(base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir")
    
    opt14 = GlobalOptimizer(v_x=25.0, penalty_factor=0.0, tir_file=tir_path, objective_mode="max_ay", model_type="14-DOF", enforce_understeer=True)
    x0_14 = [1.529, 1.252, 1.136, 0.45, 0.50, 0.485, -1.5, -1.0, 0.0, 0.1, 8000.0, 6000.0, 18000.0, 16000.0]
    res14 = opt14.run_optimization(initial_guess=x0_14)
    ay14 = opt14._evaluate_limit_ay(list(res14.x)) if res14.success else 0.0
    print(f"14-DOF Optimizer at 25 m/s: Ay = {ay14:.3f} g (Success = {res14.success})")
    assert ay14 > 1.70, f"Expected Ay > 1.70g, got {ay14:.3f}g"

    # 2. MMM Diagram test
    print("\n--- 2. Testing MMM Diagram Resolution (17 deltas & 33 betas) ---")
    df_mmm = generate_mmm_diagram(v_x=20.0, wd_f=0.48, lltd=0.52, aero_bal=0.48,
                                  delta_deg_list=[-20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
                                  beta_range_deg=(-16, 16), num_beta=33, tir_file=tir_path)
    print(f"MMM Points generated: {len(df_mmm)} | Deltas unique: {len(df_mmm['delta_deg'].unique())} | Betas unique: {len(df_mmm['beta_deg'].unique())}")
    assert len(df_mmm['delta_deg'].unique()) == 17, "Expected 17 delta curves"

    # 3. Dynamic Events 4-Wheel Telemetry
    print("\n--- 3. Testing 4-Wheel Dynamic Telemetry Channels ---")
    vp = VehicleParams()
    res_step = simulate_step_steer(vp, v_x=20.0, steer_deg=10.0, model_type="14-DOF", tir_path=tir_path)
    df_step = res_step['telemetry']
    cols = ['alpha_fl_deg', 'Fz_fl_N', 'Fy_fl_N', 'util_fl_pct']
    for c in cols:
        assert c in df_step.columns, f"Missing column {c} in Step Steer telemetry!"
    print(f"Step Steer Telemetry Verified: {len(df_step)} rows, Peak Ay: {res_step['ay_peak_g']:.2f}g, Fz_FL max: {df_step['Fz_fl_N'].max():.0f} N")

    # 4. Direct Collocation Trajectory Optimization
    print("\n--- 4. Testing Direct Collocation Trajectory Solver ---")
    res_colloc = solve_direct_collocation_racing_line(vehicle_params=vp, ds=4.0, max_iter=100)
    print(f"Direct Collocation: Success = {res_colloc['success']} | Lap Time = {res_colloc['lap_time_s']:.2f} s | V_avg = {res_colloc['v_avg_kmh']:.1f} km/h")
    assert res_colloc['lap_time_s'] > 20.0 and res_colloc['lap_time_s'] < 100.0, "Unrealistic lap time"

    # 5. Interactive 14-DOF Gamepad Driver Physics Thread
    print("\n--- 5. Testing 14-DOF Real-Time Gamepad Driver (500 Hz) ---")
    driver_th = PhysicsThread14DOF(vehicle_params=vp, tir_file=tir_path)
    driver_th.start()
    time.sleep(0.4)
    driver_th.stop()
    print(f"Driver Thread Verified: {len(driver_th.telemetry_history)} samples generated in 400ms at 500Hz!")
    assert len(driver_th.telemetry_history) > 100, "Physics thread did not run fast enough"

    print("\n==================================================")
    print("  >>> ALL 5 VERIFICATION STAGES PASSED 100%! <<<  ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
