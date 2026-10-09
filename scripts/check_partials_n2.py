"""
check_partials_n2.py
--------------------
Standalone diagnostic script:
  1. Runs prob.check_partials(method='cs') and prints a compact report.
  2. Generates the N2 diagram (om.n2) to visualize system coupling.

Usage:
  python check_partials_n2.py

Requirements: run AFTER at least one successful optimization so the problem
architecture is stable. Does NOT interfere with the main optimizer.
"""

import os
import sys
import numpy as np
import openmdao.api as om

# --- Ensure package root is on path when run standalone ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mdo.mdo_steady_state import _build_problem
from src.models.params_vehicle import VehicleParams

def run_diagnostics():
    vp = VehicleParams()
    tir_file = 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir'

    print("Building problem...")
    prob = _build_problem(
        v_x=15.0, delta_deg=3.0,
        wd_f=0.45, lltd=0.50, aero_bal=0.48,
        L=vp.L, t_f=vp.t_f, t_r=vp.t_r,
        tir_file=tir_file,
        verbose=True
    )

    print("Running model (warm-up)...")
    prob.run_model()

    # --- N2 Diagram ---
    n2_path = os.path.join(os.path.dirname(__file__), 'n2_steady_state.html')
    print(f"\nGenerating N2 diagram -> {n2_path}")
    om.n2(prob, outfile=n2_path, show_browser=False)

    # --- Check Partials ---
    print("\nRunning check_partials (complex step)...")
    data = prob.check_partials(method='cs', compact_print=True)

    # Summary
    print("\n=== Jacobian Error Summary ===")
    max_err = 0.0
    for comp_name, comp_data in data.items():
        for (of, wrt), errs in comp_data.items():
            try:
                fwd = errs['rel error'].forward
                if fwd is not None and not np.isnan(fwd):
                    flag = "  *** HIGH ***" if fwd > 1e-4 else ""
                    print(f"  [{comp_name}] d({of})/d({wrt})  rel_err = {fwd:.3e}{flag}")
                    max_err = max(max_err, float(fwd))
            except (AttributeError, KeyError, TypeError):
                pass

    print(f"\nMax relative Jacobian error: {max_err:.3e}")
    if max_err < 1e-6:
        print("Status: EXCELLENT — Analytical Jacobian matches CS within 1e-6")
    elif max_err < 1e-4:
        print("Status: OK — Small discrepancy, acceptable for SLSQP")
    else:
        print("Status: WARNING — Large error, check linearize() implementation")

    prob.cleanup()
    print(f"\nDone. Open '{n2_path}' in your browser for the N2 diagram.")

if __name__ == '__main__':
    run_diagnostics()
