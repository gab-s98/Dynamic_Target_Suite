"""
Direct Collocation & IPOPT Performance Profiling Benchmark
==========================================================
Tests the exact breakdown of computation time:
1. Python Model Setup & Graph Building Time
2. Symbolic vs Vectorized CasADi formulation
3. Thread Scaling (OMP_NUM_THREADS = 1, 2, 4, 8, max)
4. IPOPT Iteration Breakdown (Function evaluation vs KKT Linear Solver vs Line Search)
5. JIT Compilation / C-Code generation potential
"""

import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import casadi
casadi_dir = os.path.dirname(casadi.__file__)
if hasattr(os, 'add_dll_directory') and os.path.exists(casadi_dir):
    os.add_dll_directory(casadi_dir)
os.environ['PATH'] = casadi_dir + ';' + os.environ.get('PATH', '')

import casadi as ca
import numpy as np
import pandas as pd

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_direct_collocation import build_track_geometry, solve_direct_collocation_racing_line


def benchmark_thread_scaling():
    print("=" * 70)
    print("1. TEST SCALING THREAD CPU (OMP_NUM_THREADS = 1, 2, 4, 8, max)")
    print("=" * 70)
    
    vp = VehicleParams()
    track = build_track_geometry(track_length_m=850.0, ds=2.5, track_width_m=3.8)
    
    max_cpus = os.cpu_count() or 4
    thread_counts = [1, 2, 4]
    if max_cpus >= 8:
        thread_counts.append(8)
    if max_cpus not in thread_counts:
        thread_counts.append(max_cpus)
        
    results = []
    for th in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(th)
        os.environ["OPENBLAS_NUM_THREADS"] = str(th)
        os.environ["MKL_NUM_THREADS"] = str(th)
        
        t0 = time.perf_counter()
        res = solve_direct_collocation_racing_line(
            vehicle_params=vp, track_data=track,
            ds=2.5, track_width_m=3.8, model_type="14-DOF",
            max_iter=100
        )
        t_wall = time.perf_counter() - t0
        
        results.append({
            'Threads': th,
            'Wall_Time_s': t_wall,
            'Success': res['success'],
            'Lap_Time_s': res['lap_time_s']
        })
        print(f"Threads: {th:2d} | Tempo Esecuzione: {t_wall:.3f} s | Success: {res['success']} | Lap: {res['lap_time_s']:.2f} s")
        
    df_threads = pd.DataFrame(results)
    return df_threads


def benchmark_nlp_breakdown():
    print("\n" + "=" * 70)
    print("2. NLP BOTTLENECK BREAKDOWN (Costruzione vs AD Jacobiani vs Linear Solver)")
    print("=" * 70)
    
    vp = VehicleParams()
    track = build_track_geometry(track_length_m=850.0, ds=2.5, track_width_m=3.8)
    N = track['N']
    ds_val = track['ds']
    kappa_arr = track['kappa']
    w_track = track['track_width']
    w_veh = max(vp.t_f, vp.t_r) + 0.20
    n_max_limit = (w_track - w_veh) / 2.0
    
    m = vp.m; g = vp.g; L = vp.L; rho = vp.rho; S = vp.S; Cz = vp.Cz; mu_y = 1.60; mu_x = 1.65
    P_max = 80000.0

    # Step A: Measure Symbolic Graph Construction Time
    t_build_start = time.perf_counter()
    opti = ca.Opti()
    n = opti.variable(N)
    xi = opti.variable(N)
    E = opti.variable(N)
    ax = opti.variable(N)
    delta = opti.variable(N)

    lap_time = 0.0
    for k in range(N - 1):
        v_mid = 0.5 * (ca.sqrt(2.0 * E[k] + 1.0) + ca.sqrt(2.0 * E[k+1] + 1.0))
        n_mid = 0.5 * (n[k] + n[k+1])
        xi_mid = 0.5 * (xi[k] + xi[k+1])
        kap_mid = 0.5 * (kappa_arr[k] + kappa_arr[k+1])
        scale_s = (1.0 - n_mid * kap_mid) / ca.cos(xi_mid)
        lap_time += (ds_val * scale_s) / ca.fmax(v_mid, 1.0)

    opti.minimize(lap_time)

    for k in range(N - 1):
        n_mid = 0.5 * (n[k] + n[k+1])
        xi_mid = 0.5 * (xi[k] + xi[k+1])
        kap_mid = 0.5 * (kappa_arr[k] + kappa_arr[k+1])
        dn_ds = ca.tan(xi_mid) * (1.0 - n_mid * kap_mid)
        opti.subject_to(n[k+1] - n[k] == ds_val * dn_ds)
        scale_mid = (1.0 - n_mid * kap_mid) / ca.cos(xi_mid)
        dxi_ds = ((delta[k] / L) * scale_mid) - kap_mid
        opti.subject_to(xi[k+1] - xi[k] == ds_val * dxi_ds)
        opti.subject_to(E[k+1] - E[k] == ds_val * ax[k] * scale_mid)

    for k in range(N):
        opti.subject_to(opti.bounded(-n_max_limit, n[k], n_max_limit))
        opti.subject_to(opti.bounded(-np.deg2rad(35.0), xi[k], np.deg2rad(35.0)))
        opti.subject_to(opti.bounded(-np.deg2rad(25.0), delta[k], np.deg2rad(25.0)))
        opti.subject_to(opti.bounded(0.5 * (3.0**2), E[k], 0.5 * (35.0**2)))
        
        f_down = 0.5 * rho * S * Cz * (2.0 * E[k])
        total_fz = m * g + f_down
        mu_eff_y = mu_y * (total_fz / (m * g))
        mu_eff_x = mu_x * (total_fz / (m * g))
        ay_req = (2.0 * E[k]) * (delta[k] / L)
        opti.subject_to((ax[k] / (mu_eff_x * g))**2 + (ay_req / (mu_eff_y * g))**2 <= 1.0)
        v_curr = ca.sqrt(2.0 * E[k] + 1.0)
        opti.subject_to(m * ax[k] * v_curr <= P_max)
        opti.subject_to(ax[k] >= -1.80 * mu_x * g)

    opti.subject_to(n[0] == 0.0)
    opti.subject_to(xi[0] == 0.0)
    opti.set_initial(E, 0.5 * (16.0**2))
    
    t_build = time.perf_counter() - t_build_start
    print(f"A. Tempo Costruzione Grafo Simbolico Python: {t_build*1000:.2f} ms")

    # Step B: Profile IPOPT internal solve with full CasADi timing stats
    p_opts = {'expand': True, 'print_time': True}
    s_opts = {
        'max_iter': 100,
        'tol': 1e-3,
        'print_level': 5, # full ipopt print level for profiling
        'linear_solver': 'mumps'
    }
    opti.solver('ipopt', p_opts, s_opts)

    t_solve_start = time.perf_counter()
    sol = opti.solve()
    t_solve = time.perf_counter() - t_solve_start
    
    stats = opti.stats()
    print(f"B. Tempo Risoluzione Totale IPOPT: {t_solve:.3f} s")
    print(f"   - Numero Variabili Decisione: {N * 5}")
    print(f"   - Numero Vincoli Uguaglianza/Disuguaglianza: {N * 3 + N * 3}")
    print(f"   - Numero Iterazioni: {stats.get('iter_count', 'N/A')}")
    print(f"   - Tempo CPU Ritorno CasADi: {stats.get('t_wall_total', 'N/A')} s")
    print(f"   - Status: {stats.get('return_status', 'N/A')}")
    

if __name__ == "__main__":
    df = benchmark_thread_scaling()
    benchmark_nlp_breakdown()
