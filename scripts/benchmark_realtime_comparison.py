"""
benchmark_realtime_comparison.py
================================
Benchmark quantitativo e generatore di grafici prestazionali per la simulazione 14-DOF.
Confronta:
1. Motore di Calcolo: CasADi Simbolico vs C Nativo DLL vs Zero-Allocation In-Place.
2. Solutori Numerici: Runge-Kutta 4 vs Eulero Simplettico (Latenza e Throughput).
3. Stabilità Temporale & Jitter: Loop Standard (time.sleep) vs Loop Real-Time (Hybrid Spinloop).
4. Profilo di Latenza Percentile (P50, P95, P99, Max).
"""

import os
import sys
import time
import ctypes
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Aggiungi root al path per importare i moduli di progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF


def run_full_benchmark():
    print("=" * 70)
    print(" DYNAMIC TARGET SUITE — BENCHMARK DINAMICA 14-DOF & REAL-TIME ANALYSIS")
    print("=" * 70)

    vp = VehicleParams()
    veh = Vehicle14DOF(vp)
    ode_sym = veh.get_casadi_ode()
    ode_fast = veh.get_fast_ode()

    # Vettori di prova (28 stati, 4 comandi, 37 parametri)
    x0 = np.zeros(28, dtype=np.float64)
    x0[0] = 12.0
    x0[16:20] = 12.0 / 0.203
    u0 = np.array([np.deg2rad(3.5), 160.0, 160.0, 0.0], dtype=np.float64)
    p0 = np.zeros(37, dtype=np.float64)
    p0[0:17] = [
        vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r,
        vp.K_z_f, vp.K_z_r, vp.aero_balance,
        vp.camber_f_static, vp.camber_r_static,
        vp.toe_f_static, vp.toe_r_static,
        vp.L, vp.t_f, vp.t_r, 1.60, 1.60, 1.20
    ]

    dt = 0.001  # 1.0 ms (target 1000 Hz)
    N_BENCH = 2000

    # -------------------------------------------------------------------------
    # TEST 1: VALUTAZIONE SINGOLA ODE
    # -------------------------------------------------------------------------
    print("\n[1/4] Benchmarking Singola Valutazione ODE...")
    # Warmup
    for _ in range(50):
        _ = ode_sym(x0, u0, p0)
        _ = ode_fast(x0, u0, p0)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        _ = ode_sym(x0, u0, p0)
    lat_eval_sym_us = ((time.perf_counter() - t0) / N_BENCH) * 1e6

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        _ = ode_fast(x0, u0, p0)
    lat_eval_fast_us = ((time.perf_counter() - t0) / N_BENCH) * 1e6

    print(f"    - CasADi Simbolico: {lat_eval_sym_us:7.2f} us")
    print(f"    - C Nativo DLL:     {lat_eval_fast_us:7.2f} us (Speedup: {lat_eval_sym_us/lat_eval_fast_us:.2f}x)")

    # -------------------------------------------------------------------------
    # TEST 2: SOLUTORI A CONFRONTO (RK4 Standard, RK4 Zero-Alloc, Symplectic Euler)
    # -------------------------------------------------------------------------
    print("\n[2/4] Benchmarking Solutori Numerici...")

    # A: RK4 Standard (allocazioni NumPy come nel codice attuale)
    def rk4_standard(ode, x, u, p, h):
        k1 = np.array(ode(x, u, p)).flatten()
        k2 = np.array(ode(x + 0.5 * h * k1, u, p)).flatten()
        k3 = np.array(ode(x + 0.5 * h * k2, u, p)).flatten()
        k4 = np.array(ode(x + h * k3, u, p)).flatten()
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # B: RK4 Zero-Allocation (in-place buffer preallocati)
    k1_b = np.empty(28, dtype=np.float64)
    k2_b = np.empty(28, dtype=np.float64)
    k3_b = np.empty(28, dtype=np.float64)
    k4_b = np.empty(28, dtype=np.float64)
    tmp_b = np.empty(28, dtype=np.float64)
    x_out_b = np.empty(28, dtype=np.float64)

    def rk4_zero_alloc(ode, x, u, p, h):
        k1_b[:] = np.asarray(ode(x, u, p)).flatten()
        np.add(x, 0.5 * h * k1_b, out=tmp_b)
        k2_b[:] = np.asarray(ode(tmp_b, u, p)).flatten()
        np.add(x, 0.5 * h * k2_b, out=tmp_b)
        k3_b[:] = np.asarray(ode(tmp_b, u, p)).flatten()
        np.add(x, h * k3_b, out=tmp_b)
        k4_b[:] = np.asarray(ode(tmp_b, u, p)).flatten()
        x_out_b[:] = x + (h / 6.0) * (k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b)
        return x_out_b

    # C: Eulero Simplettico (1 sola valutazione ODE)
    x_dot_symp = np.empty(28, dtype=np.float64)
    x_out_symp = np.empty(28, dtype=np.float64)

    def symplectic_euler(ode, x, u, p, h):
        x_dot_symp[:] = np.asarray(ode(x, u, p)).flatten()
        # Aggiornamento momentum / velocità
        x_out_symp[0:3] = x[0:3] + h * x_dot_symp[0:3]
        x_out_symp[4] = x[4] + h * x_dot_symp[4]
        x_out_symp[6:8] = x[6:8] + h * x_dot_symp[6:8]
        x_out_symp[12:20] = x[12:20] + h * x_dot_symp[12:20]
        # Aggiornamento posizioni con v_{k+1}
        x_out_symp[3] = x[3] + h * x_out_symp[4]
        x_out_symp[5] = x[5] + h * x_out_symp[6]
        x_out_symp[8:12] = x[8:12] + h * x_out_symp[12:16]
        # Relaxations
        x_out_symp[20:28] = x[20:28] + h * x_dot_symp[20:28]
        return x_out_symp

    # Warmup
    x_test = x0.copy()
    for _ in range(50):
        _ = rk4_standard(ode_sym, x_test, u0, p0, dt)
        _ = rk4_standard(ode_fast, x_test, u0, p0, dt)
        _ = rk4_zero_alloc(ode_fast, x_test, u0, p0, dt)
        _ = symplectic_euler(ode_fast, x_test, u0, p0, dt)

    # 1. RK4 Simbolico
    times_rk4_sym = []
    x_curr = x0.copy()
    for _ in range(N_BENCH):
        t0 = time.perf_counter_ns()
        x_curr = rk4_standard(ode_sym, x_curr, u0, p0, dt)
        times_rk4_sym.append((time.perf_counter_ns() - t0) / 1000.0)

    # 2. RK4 C Nativo (Standard)
    times_rk4_c = []
    x_curr = x0.copy()
    for _ in range(N_BENCH):
        t0 = time.perf_counter_ns()
        x_curr = rk4_standard(ode_fast, x_curr, u0, p0, dt)
        times_rk4_c.append((time.perf_counter_ns() - t0) / 1000.0)

    # 3. RK4 C Nativo Zero-Alloc
    times_rk4_za = []
    x_curr = x0.copy()
    for _ in range(N_BENCH):
        t0 = time.perf_counter_ns()
        x_curr = rk4_zero_alloc(ode_fast, x_curr, u0, p0, dt)
        times_rk4_za.append((time.perf_counter_ns() - t0) / 1000.0)

    # 4. Eulero Simplettico C Nativo
    times_symp_c = []
    x_curr = x0.copy()
    for _ in range(N_BENCH):
        t0 = time.perf_counter_ns()
        x_curr = symplectic_euler(ode_fast, x_curr, u0, p0, dt)
        times_symp_c.append((time.perf_counter_ns() - t0) / 1000.0)

    lat_rk4_sym = np.mean(times_rk4_sym)
    lat_rk4_c = np.mean(times_rk4_c)
    lat_rk4_za = np.mean(times_rk4_za)
    lat_symp_c = np.mean(times_symp_c)

    print(f"    - RK4 Simbolico:         {lat_rk4_sym:7.2f} us (Max Rate: {1e6/lat_rk4_sym:6.0f} Hz)")
    print(f"    - RK4 C Nativo:          {lat_rk4_c:7.2f} us (Max Rate: {1e6/lat_rk4_c:6.0f} Hz)")
    print(f"    - RK4 C Zero-Alloc:      {lat_rk4_za:7.2f} us (Max Rate: {1e6/lat_rk4_za:6.0f} Hz)")
    print(f"    - Simplettico C Nativo:  {lat_symp_c:7.2f} us (Max Rate: {1e6/lat_symp_c:6.0f} Hz)")

    # -------------------------------------------------------------------------
    # TEST 3: JITTER TEMPORALE & PRECISIONE DEL PASSO (1000 Hz Target, dt=1.0 ms)
    # -------------------------------------------------------------------------
    print("\n[3/4] Misurazione Jitter Temporale e Stabilità delta-t...")
    N_STEPS_JITTER = 400
    target_dt_s = 0.001  # 1 ms = 1000 Hz
    target_dt_ns = int(target_dt_s * 1e9)

    # CASO A: Approccio Standard (time.sleep senza timeBeginPeriod)
    deltas_sleep = []
    t_prev = time.perf_counter_ns()
    for _ in range(N_STEPS_JITTER):
        # Simula un calcolo rapido
        time.sleep(0.0005)  # sleep 0.5 ms
        t_now = time.perf_counter_ns()
        dt_ms = (t_now - t_prev) / 1e6
        deltas_sleep.append(dt_ms)
        t_prev = t_now

    # CASO B: Approccio Real-Time Hybrid Spinloop con timeBeginPeriod(1)
    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    deltas_hybrid = []
    t_next = time.perf_counter_ns()
    t_prev = t_next
    for _ in range(N_STEPS_JITTER):
        t_next += target_dt_ns
        now = time.perf_counter_ns()
        rem_ns = t_next - now
        if rem_ns > 1_500_000:
            time.sleep((rem_ns - 1_200_000) / 1e9)
        while time.perf_counter_ns() < t_next:
            pass
        t_now = time.perf_counter_ns()
        dt_ms = (t_now - t_prev) / 1e6
        deltas_hybrid.append(dt_ms)
        t_prev = t_now

    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

    std_sleep = np.std(deltas_sleep)
    std_hybrid = np.std(deltas_hybrid)
    print(f"    - Jitter Dev. Std. Standard time.sleep:    {std_sleep:7.3f} ms")
    print(f"    - Jitter Dev. Std. Real-Time Spinloop:      {std_hybrid:7.3f} ms (Abbattimento: {std_sleep/max(std_hybrid, 1e-6):.1f}x)")

    # -------------------------------------------------------------------------
    # TEST 4: GENERAZIONE GRAFICI COMPARATIVI AD ALTA RISOLUZIONE
    # -------------------------------------------------------------------------
    print("\n[4/4] Generazione Grafici Benchmark...")

    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor('#0d1117')

    for ax in axs.flat:
        ax.set_facecolor('#161b22')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    # Grafico 1: Latenza Singolo Step per Solutore
    solvers = ['CasADi\nSimbolico\n(RK4)', 'C Nativo\nStandard\n(RK4)', 'C Nativo\nZero-Alloc\n(RK4)', 'Eulero\nSimplettico\n(C Nativo)']
    latencies = [lat_rk4_sym, lat_rk4_c, lat_rk4_za, lat_symp_c]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    bars = axs[0, 0].bar(solvers, latencies, color=colors, width=0.55, edgecolor='#ffffff', linewidth=0.8)
    axs[0, 0].set_ylabel('Latenza di Calcolo per Step (us)', fontsize=11, fontweight='bold', color='#c9d1d9')
    axs[0, 0].set_title('Latenza di Calcolo per Step (Passo dt=1.0 ms)', fontsize=12, fontweight='bold', color='#58a6ff')
    axs[0, 0].axhline(1000.0, color='#e74c3c', linestyle=':', linewidth=1.5, label='Budget Max 1000 Hz (1000 us)')
    axs[0, 0].legend(loc='upper right', framealpha=0.6)

    for bar, val in zip(bars, latencies):
        axs[0, 0].text(bar.get_x() + bar.get_width()/2.0, val + 15, f"{val:.1f} us", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ffffff')

    # Grafico 2: Throughput Massimo Teorico (Hz)
    rates_hz = [1e6/l for l in latencies]
    bars2 = axs[0, 1].bar(solvers, rates_hz, color=colors, width=0.55, edgecolor='#ffffff', linewidth=0.8)
    axs[0, 1].set_ylabel('Frequenza Massima Sostenibile (Hz)', fontsize=11, fontweight='bold', color='#c9d1d9')
    axs[0, 1].set_title('Throughput Massimo Teorico di Simulazione', fontsize=12, fontweight='bold', color='#58a6ff')
    axs[0, 1].axhline(1000.0, color='#2ecc71', linestyle='--', linewidth=1.5, label='Target 1000 Hz Real-Time')
    axs[0, 1].legend(loc='upper left', framealpha=0.6)

    for bar, val in zip(bars2, rates_hz):
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, val + 300, f"{val:.0f} Hz", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ffffff')

    # Grafico 3: Serie Temporale Delta-t (Jitter: Standard vs Real-Time Spinloop)
    steps_window = 120
    axs[1, 0].plot(deltas_sleep[:steps_window], color='#e74c3c', label=f'Standard sleep (Std: {std_sleep:.2f} ms)', alpha=0.85, linewidth=1.5)
    axs[1, 0].plot(deltas_hybrid[:steps_window], color='#2ecc71', label=f'RT Hybrid Spinloop (Std: {std_hybrid:.4f} ms)', linewidth=2.0)
    axs[1, 0].axhline(1.0, color='#ffffff', linestyle='--', alpha=0.7, label='Target dt = 1.00 ms (1000 Hz)')
    axs[1, 0].set_xlabel('Numero Step di Simulazione', fontsize=11, fontweight='bold', color='#c9d1d9')
    axs[1, 0].set_ylabel('Passo Temporale Effettivo dt (ms)', fontsize=11, fontweight='bold', color='#c9d1d9')
    axs[1, 0].set_title('Confronto Jitter Temporale su Windows (120 Step)', fontsize=12, fontweight='bold', color='#58a6ff')
    axs[1, 0].legend(loc='upper right', framealpha=0.7)
    axs[1, 0].set_ylim([0.0, max(max(deltas_sleep[:steps_window]) * 1.15, 3.0)])

    # Grafico 4: Distribuzione Percentile Latenze (P50, P95, P99, Max)
    percentiles = ['P50 (Mediana)', 'P95', 'P99', 'P99.9', 'Max (Spike)']
    p_sym = [np.percentile(times_rk4_sym, p) for p in [50, 95, 99, 99.9, 100]]
    p_c = [np.percentile(times_rk4_c, p) for p in [50, 95, 99, 99.9, 100]]
    p_za = [np.percentile(times_rk4_za, p) for p in [50, 95, 99, 99.9, 100]]
    p_symp = [np.percentile(times_symp_c, p) for p in [50, 95, 99, 99.9, 100]]

    x_idx = np.arange(len(percentiles))
    width = 0.20
    axs[1, 1].bar(x_idx - 1.5*width, p_sym, width, label='RK4 Simbolico', color='#e74c3c')
    axs[1, 1].bar(x_idx - 0.5*width, p_c, width, label='RK4 C Nativo', color='#f39c12')
    axs[1, 1].bar(x_idx + 0.5*width, p_za, width, label='RK4 C Zero-Alloc', color='#2ecc71')
    axs[1, 1].bar(x_idx + 1.5*width, p_symp, width, label='Simplettico C', color='#3498db')

    axs[1, 1].set_xticks(x_idx)
    axs[1, 1].set_xticklabels(percentiles, fontsize=10, color='#c9d1d9')
    axs[1, 1].set_ylabel('Latenza (us)', fontsize=11, fontweight='bold', color='#c9d1d9')
    axs[1, 1].set_title('Profilo di Latenza Percentile (Resilienza ai Picchi)', fontsize=12, fontweight='bold', color='#58a6ff')
    axs[1, 1].legend(loc='upper left', framealpha=0.7)

    plt.tight_layout()
    output_png = os.path.join(project_root, "docs", "benchmark_realtime_comparison.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\n[SUCCESSO] Grafico salvato in: {output_png}")

    # Salva copia anche nella root per visualizzazione rapida
    root_png = os.path.join(project_root, "benchmark_realtime_comparison.png")
    plt.savefig(root_png, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"[SUCCESSO] Copia salvata in: {root_png}")


if __name__ == "__main__":
    run_full_benchmark()
