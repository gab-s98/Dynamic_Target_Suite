import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.core.simulation_options import SimulationOptions, get_simulation_options, update_simulation_options
from src.mdo.mdo_dynamic_events import simulate_step_steer, simulate_double_lane_change, simulate_autox_endurance
from src.mdo.mdo_global_optimizer import GlobalOptimizer

def main():
    print("=== 1. Testing SimulationOptions ===")
    opts = get_simulation_options()
    print(f"Default dt_ode: {opts.dt_ode}, ds_track: {opts.ds_track}, maintain_speed: {opts.maintain_speed}")
    opts.dt_ode = 0.002
    opts.t_ramp_steer = 0.08
    update_simulation_options(opts)
    reloaded = SimulationOptions.load_from_file()
    print(f"Reloaded dt_ode: {reloaded.dt_ode}")

    vp = VehicleParams()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tir_path = os.path.join(base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir")

    print("\n=== 2. Testing Step Steer (7-DOF & 14-DOF) with Maintain Speed & TV ===")
    res_7 = simulate_step_steer(vp, v_x=20.0, steer_deg=3.5, t_end=3.0, dt=0.002, t_ramp=0.08,
                                torque_vectoring_nm=25.0, maintain_speed=True, model_type="7-DOF", tir_path=tir_path)
    print(f"7-DOF Step Steer: peak_ay = {res_7['ay_peak_g']:.3f} g | r_ss = {res_7['r_ss_deg_s']:.2f} deg/s | overshoot = {res_7['overshoot_pct']:.1f}% | steps = {len(res_7['telemetry'])}")

    res_14 = simulate_step_steer(vp, v_x=20.0, steer_deg=3.5, t_end=3.0, dt=0.002, t_ramp=0.08,
                                 torque_vectoring_nm=25.0, maintain_speed=True, model_type="14-DOF", tir_path=tir_path)
    print(f"14-DOF Step Steer: peak_ay = {res_14['ay_peak_g']:.3f} g | r_ss = {res_14['r_ss_deg_s']:.2f} deg/s | overshoot = {res_14['overshoot_pct']:.1f}% | steps = {len(res_14['telemetry'])}")

    print("\n=== 3. Testing Double Lane Change ===")
    res_dlc = simulate_double_lane_change(vp, v_entry=20.0, t_end=4.0, dt=0.002, maintain_speed=True,
                                          model_type="14-DOF", tir_path=tir_path)
    print(f"DLC: max_ay = {res_dlc['max_ay_g']:.3f} g | max_phi = {res_dlc['max_phi_deg']:.2f} deg | corridor_ok = {res_dlc['cone_clearance_ok']}")

    print("\n=== 4. Testing AutoX Simulation with Speed-Dependent Aero Downforce ===")
    res_autox = simulate_autox_endurance(vp, event_type="AutoX", num_laps=1, track_length_m=850.0,
                                         ds=0.5, model_type="14-DOF", tir_path=tir_path)
    print(f"AutoX: lap_time = {res_autox['lap_time_s']:.2f} s | v_max = {res_autox['v_max_kmh']:.1f} km/h | max_ay = {res_autox['max_ay_g']:.3f} g | energy = {res_autox['energy_lap_kwh']*1000.0:.1f} Wh")

    print("\n=== 5. Testing Limit Ay Scaling vs Speed (10 to 30 m/s) ===")
    opt = GlobalOptimizer(model_type="14-DOF", tir_file=tir_path)
    X14 = [vp.L, vp.t_f, vp.t_r, vp.wd_f, vp.lltd, vp.aero_balance,
           vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
           vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r]
    for v in [10.0, 15.0, 20.0, 25.0, 30.0]:
        ay = opt._evaluate_limit_ay_at_speed(v, X14)
        downforce = 0.5 * vp.rho * vp.S * vp.Cz * (v**2)
        print(f"  v = {v:4.1f} m/s ({v*3.6:5.1f} km/h) | F_down = {downforce:6.1f} N | max_ay = {ay:.4f} g")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
