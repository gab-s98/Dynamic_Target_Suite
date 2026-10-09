"""
ci_pipeline.py
--------------
Automated CI & Validation Pipeline for Dynamic Target Suite:
1. Unit Tests Discovery & Execution
2. Headless GUI Smoke Test & Component Validation
3. PyInstaller Standalone Executable Build
4. Executable Process Health-Check & DLL Verification
"""

import os
import sys
import time
import subprocess
import argparse

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  [*] {title.upper()}")
    print("=" * 70)

def print_step(step_name, status, details=""):
    badge = "[PASS]" if status else "[FAIL]"
    print(f"{badge:<8} {step_name:<45} {details}")

def run_stage_1_unit_tests(venv_python):
    print_header("Stage 1: Unit Test Suite Execution")
    test_files = [
        # Core Physics & Tire
        "tests/test_damper_ttx25.py",
        "tests/test_tire_mf61.py",
        "tests/test_tire_relaxation.py",
        "tests/test_3dof.py",
        "tests/test_7dof.py",
        "tests/test_14dof.py",
        "tests/test_camber_steer_symmetry.py",
        "tests/test_straight_line_symmetry.py",
        # Handling Analysis
        "tests/test_vehicle_dna_maps.py",
        "tests/test_mmm_hd_verification.py",
        "tests/test_balance_range.py",
        "tests/test_openmdao.py",
        # Transient & Events
        "tests/test_transient_sensitivity.py",
        "tests/test_dynamic_events.py",
        "tests/test_dynamic_verification.py",
        # Components & Tools
        "tests/test_interactive_driver.py",
        "tests/test_simulator_review_features.py",
        "tests/test_popout_and_real_suspension.py",
        "tests/test_kinematics_digitizer.py",
        # Scoring & Sensitivity
        "tests/test_sensitivity.py",
        # Integration & GUI
        "tests/test_comprehensive_features.py",
        "tests/test_gui_systematic.py",
        "tests/test_mdo_gui_lifecycle.py",
        "tests/test_persistence_and_autox.py",
        "tests/test_arb_diamond_balance.py",
        # Smoke Tests (F-6)
        "tests/test_direct_collocation_smoke.py",
        "tests/test_collocation_ui_and_live_inspector.py",
        "tests/test_uncertainty_smoke.py",
        "tests/test_ackermann_wear.py",
        "tests/test_simulator_quick_setup.py",
        "tests/test_wheel_forces_and_torques.py",
        "tests/test_b_segment_suv.py",
        "tests/test_powertrain_tv_and_dampers.py",
        "tests/test_mdo_neutral_target.py",
        "tests/test_powertrain_electric.py",
    ]
    cmd = [venv_python, "-m", "unittest"] + test_files
    start = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = (p.returncode == 0)
    print_step(f"Core Physics & Logic Unit Tests ({len(test_files)} Suites)", success, f"({elapsed:.2f}s)")
    if not success:
        print("\n--- Test Errors & Output ---")
        print(p.stderr or p.stdout)
    return success

def run_stage_2_gui_smoke_test(venv_python):
    print_header("Stage 2: Headless GUI & Subtab Smoke Test")
    test_code = """
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_gui import MainWindow

app = QApplication(sys.argv)
win = MainWindow()

# 1. Test Transient Sensitivity Tab & Presets
win._apply_sens_preset('rc_2d')
win._update_stability_card()

# 2. Test Vehicle DNA Maps Plotting
win.maps_dna_v_input.setText('15.0')
win.plot_dna_maps()
win.maps_dna_view_combo.setCurrentIndex(1)
win.plot_dna_maps()

# 3. Test Optimal Balance Range & Uncertainty Sweep
win.bal_pts_input.setText('5')
win.plot_optimal_balance_sweep()
win.plot_4_ratios_radar()

# 4. Test YMD Iso-Radius & Multi-Speed Overlay
win.run_mmm_standalone()
win.plot_mmm_multispeed()

# 5. Test TTX25 Damper Dyno Plotting
win.plot_damper_dyno()

# 6. Test Handling Diagram & Stability Frontier Plotting
win.run_constant_parameters()

print('GUI Smoke Test Passed: All subtabs initialized and rendered successfully.')
"""
    cmd = [venv_python, "-c", test_code]
    start = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = (p.returncode == 0 and "GUI Smoke Test Passed" in p.stdout)
    print_step("Headless GUI Subtabs & Plotting Smoke Test", success, f"({elapsed:.2f}s)")
    if not success:
        print("\n--- GUI Smoke Test Error ---")
        print(p.stderr or p.stdout)
    return success

def run_stage_3_build_exe(venv_pyinstaller, spec_file):
    print_header("Stage 3: PyInstaller Standalone Executable Build")
    cmd = [venv_pyinstaller, "--noconfirm", spec_file]
    start = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = (p.returncode == 0)
    print_step("PyInstaller Standalone Build (MDO_Target_Suite.exe)", success, f"({elapsed:.2f}s)")
    if not success:
        print("\n--- Build Error ---")
        print(p.stderr or p.stdout)
    return success

def run_stage_4_exe_health_check(exe_path):
    print_header("Stage 4: Executable Startup & Process Health-Check")
    if not os.path.exists(exe_path):
        print_step("Executable Existence Check", False, f"File not found: {exe_path}")
        return False
    
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    start = time.time()
    p = subprocess.Popen([exe_path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3.5)
    
    is_running = (p.poll() is None)
    if is_running:
        p.terminate()
        try:
            p.wait(timeout=2.0)
        except Exception:
            p.kill()
            
    elapsed = time.time() - start
    print_step("Executable Startup, DLL Loading & Execution Health", is_running, f"PID {p.pid} ({elapsed:.2f}s)")
    return is_running

def main():
    parser = argparse.ArgumentParser(description="Automated CI Pipeline for Dynamic Target Suite")
    parser.add_argument("--skip-build", action="store_true", help="Skip PyInstaller build and run only tests")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    venv_python = os.path.join(base_dir, "venv", "Scripts", "python.exe")
    venv_pyinstaller = os.path.join(base_dir, "venv", "Scripts", "pyinstaller.exe")
    spec_file = os.path.join(base_dir, "MDO_Target_Suite.spec")
    exe_path = os.path.join(base_dir, "dist", "MDO_Target_Suite", "MDO_Target_Suite.exe")

    print("\n" + "#" * 70)
    print("  [CI] DYNAMIC TARGET SUITE -- AUTOMATED CI PIPELINE")
    print("#" * 70)

    stages = []
    
    # 1. Unit tests
    t1 = run_stage_1_unit_tests(venv_python)
    stages.append(("Unit Tests", t1))

    # 2. GUI Smoke Test
    t2 = run_stage_2_gui_smoke_test(venv_python)
    stages.append(("GUI Smoke Test", t2))

    # 3. Build EXE
    if not args.skip_build:
        t3 = run_stage_3_build_exe(venv_pyinstaller, spec_file)
        stages.append(("PyInstaller Build", t3))

        # 4. EXE Health check
        t4 = run_stage_4_exe_health_check(exe_path)
        stages.append(("EXE Health Check", t4))

    print_header("Pipeline Summary")
    all_ok = True
    for name, ok in stages:
        print_step(name, ok)
        if not ok:
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("  [SUCCESS] ALL AUTOMATED STAGES PASSED SUCCESSFULLY!")
    else:
        print("  [WARNING] SOME STAGES FAILED. PLEASE CHECK LOGS ABOVE.")
    print("=" * 70 + "\n")

    sys.exit(0 if all_ok else 1)

if __name__ == '__main__':
    main()
