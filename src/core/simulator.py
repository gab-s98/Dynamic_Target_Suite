"""
simulator.py
------------
Standalone entry point for the Dynamic Target Suite Real-Time 14-DOF Simulator.

Usage:
    python simulator.py                          # default nominal FSAE setup
    python simulator.py --tir tire_tir/MY.tir   # with custom tyre file
    python simulator.py --setup setups/my.json  # with custom vehicle JSON

This entry point does NOT import any MDO modules (mdo_*, vehicle_dna_maps,
mdo_target_optimization, etc.), keeping the simulator boot time minimal and
the standalone executable small.
"""
import sys
import os
import multiprocessing

# CasADi DLL resolution for PyInstaller frozen build
if hasattr(sys, '_MEIPASS'):
    casadi_dir = os.path.join(sys._MEIPASS, 'casadi')
    if os.path.exists(casadi_dir):
        os.add_dll_directory(casadi_dir)

from PyQt5.QtWidgets import QApplication
from src.ui.sim_main_window import Interactive14DOFDrivingWindow
from src.models.params_vehicle import VehicleParams
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic Target Suite – Real-Time Simulator")
    p.add_argument('--tir', default='', help='Path to .tir Pacejka MF6.1 tyre file')
    p.add_argument('--setup', default='', help='Path to vehicle setup JSON file')
    return p.parse_args()


def load_vehicle_params(setup_path: str) -> VehicleParams:
    if setup_path and os.path.isfile(setup_path):
        try:
            with open(setup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            vp = VehicleParams(**{k: v for k, v in data.items() if hasattr(VehicleParams, k)})
            print(f"[Simulator] Loaded setup: {setup_path}")
            return vp
        except Exception as e:
            print(f"[Simulator] Warning: could not load setup '{setup_path}': {e}")
    return VehicleParams()


def main():
    multiprocessing.freeze_support()
    args = parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    vp = load_vehicle_params(args.setup)
    tir = args.tir if args.tir and os.path.isfile(args.tir) else ''

    window = Interactive14DOFDrivingWindow(vehicle_params=vp, tir_file=tir)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
