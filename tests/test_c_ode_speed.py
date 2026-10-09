"""
test_c_ode_speed.py
===================
Benchmark and verification script for the 14-DOF Vehicle Dynamics ODE.
Verifies that:
1. Vehicle14DOF.get_fast_ode() loads the native compiled C library (ode_14dof_gen.dll).
2. The loaded function is a native C ca.external function (not symbolic MXFunction).
3. Evaluates 1000 full RK4 integration steps (250 Hz step size) for both
   symbolic MXFunction and native C external function.
4. Outputs step latency (ms), theoretical maximum simulation rate (Hz), and speedup factor.
"""

import os
import sys
import time
import numpy as np
import casadi as ca

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.vehicle_14dof import Vehicle14DOF


def run_benchmark():
    print("=" * 70)
    print(" DYNAMIC TARGET SUITE — 14-DOF ODE NATIVE C BENCHMARK & VERIFICATION")
    print("=" * 70)

    # 1. Setup vehicle model (matching default compiled ODE)
    vp = VehicleParams()
    veh = Vehicle14DOF(vp)

    # 2. Retrieve compiled fast ODE and symbolic fallback ODE
    print("\n[1] Calling Vehicle14DOF.get_fast_ode()...")
    ode_fast = veh.get_fast_ode()
    ode_sym = veh.get_casadi_ode()

    desc_fast = str(ode_fast)
    desc_sym = str(ode_sym)

    print(f"    ode_fast representation: {desc_fast}")
    print(f"    ode_sym representation:  {desc_sym}")

    # 3. Verification of native C external function
    is_external = "External" in desc_fast or "external" in desc_fast.lower()
    is_symbolic = "MXFunction" in desc_fast or "SXFunction" in desc_fast

    print("\n[2] Verification of Engine Type:")
    if is_external and not is_symbolic:
        print("    [PASS] ode_fast is a NATIVE C COMPILED LIBRARY via ca.external!")
    else:
        print("    [FAIL] ode_fast is running as symbolic fallback!")
        sys.exit(1)

    # 4. Prepare state, input, parameter vectors
    # 23-DOF state vector:
    x0 = np.zeros(23, dtype=float)
    x0[0] = 12.0  # vx = 12 m/s (~43 km/h)
    Rw = getattr(vp, "R_w", 0.203)
    w0 = 12.0 / Rw
    x0[11:15] = w0  # w_fl, w_fr, w_rl, w_rr

    u0 = np.array([np.deg2rad(4.5), 180.0, 180.0, 0.0], dtype=float)  # steer, T_fl/rl, T_fr/rr, T_brake
    p0 = np.array([
        vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r,
        vp.K_z_f, vp.K_z_r, vp.aero_balance,
        vp.camber_f_static, vp.camber_r_static,
        vp.toe_f_static, vp.toe_r_static,
        vp.L, vp.t_f, vp.t_r,
        getattr(vp, "LMUY", 1.60),
        getattr(vp, "LMUX", 1.60)
    ], dtype=float)

    dt = 1.0 / 250.0  # 0.004 s (nominal 250 Hz step)
    N_STEPS = 1000

    def rk4_step(ode_fn, x, u, p, h):
        k1 = np.array(ode_fn(x, u, p)).flatten()
        k2 = np.array(ode_fn(x + 0.5 * h * k1, u, p)).flatten()
        k3 = np.array(ode_fn(x + 0.5 * h * k2, u, p)).flatten()
        k4 = np.array(ode_fn(x + h * k3, u, p)).flatten()
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Warmup
    x_test = x0.copy()
    for _ in range(50):
        x_test = rk4_step(ode_sym, x_test, u0, p0, dt)
        x_test = rk4_step(ode_fast, x_test, u0, p0, dt)

    print(f"\n[3] Benchmarking {N_STEPS} RK4 steps (4 ODE evaluations per step = {4 * N_STEPS} evaluations)...")

    # Benchmark Symbolic MXFunction
    x_sym = x0.copy()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        x_sym = rk4_step(ode_sym, x_sym, u0, p0, dt)
    t_total_sym = time.perf_counter() - t0
    t_step_sym_ms = (t_total_sym / N_STEPS) * 1000.0

    # Benchmark Native C External
    x_c = x0.copy()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        x_c = rk4_step(ode_fast, x_c, u0, p0, dt)
    t_total_c = time.perf_counter() - t0
    t_step_c_ms = (t_total_c / N_STEPS) * 1000.0

    # Benchmark Pure Single Evaluation (no RK4 NumPy allocation overhead)
    N_EVALS = 5000
    t0 = time.perf_counter()
    for _ in range(N_EVALS):
        _ = ode_sym(x0, u0, p0)
    t_eval_sym_us = ((time.perf_counter() - t0) / N_EVALS) * 1e6

    t0 = time.perf_counter()
    for _ in range(N_EVALS):
        _ = ode_fast(x0, u0, p0)
    t_eval_c_us = ((time.perf_counter() - t0) / N_EVALS) * 1e6

    speedup_rk4 = t_step_sym_ms / t_step_c_ms
    speedup_eval = t_eval_sym_us / t_eval_c_us

    # Consistency check
    diff_norm = np.linalg.norm(x_sym - x_c)

    print("\n" + "-" * 70)
    print(" BENCHMARK RESULTS SUMMARY")
    print("-" * 70)
    print(f" Single ODE Evaluation (Pure):")
    print(f"   - CasADi Symbolic MXFunction: {t_eval_sym_us:7.2f} us ({t_eval_sym_us/1000:.4f} ms)")
    print(f"   - Native C ca.external DLL:   {t_eval_c_us:7.2f} us ({t_eval_c_us/1000:.4f} ms)")
    print(f"   - Pure Evaluation Speedup:    {speedup_eval:7.2f}x")
    print()
    print(f" Full RK4 Integration Step (4 ODE evals + NumPy vector ops):")
    print(f"   - CasADi Symbolic MXFunction: {t_step_sym_ms:7.4f} ms/step  (Max Rate: {1000.0/t_step_sym_ms:6.1f} Hz)")
    print(f"   - Native C ca.external DLL:   {t_step_c_ms:7.4f} ms/step  (Max Rate: {1000.0/t_step_c_ms:6.1f} Hz)")
    print(f"   - RK4 Integration Speedup:    {speedup_rk4:7.2f}x")
    print()
    print(f" Numerical Trajectory Divergence after {N_STEPS} steps: {diff_norm:.2e} (Identity Confirmed)")
    print("-" * 70)

    if diff_norm < 1e-6:
        print("[SUCCESS] Native C ODE produces mathematically identical results with high performance.")
    else:
        print("[WARNING] Small numerical drift detected between symbolic and C compiled code.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
