import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if hasattr(sys, '_MEIPASS'):
    casadi_dir = os.path.join(sys._MEIPASS, 'casadi')
    if os.path.exists(casadi_dir) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(casadi_dir)

from PyQt5.QtWidgets import QApplication

from src.models.params_vehicle import VehicleParams
from src.ui.main_gui import main, MainWindowPro, MainWindow
from src.ui.sim_main_window import Interactive14DOFDrivingWindow


def launch_driving_simulator(vehicle_params=None, tir_file=None):
    app = QApplication.instance()
    standalone = False
    if app is None:
        app = QApplication(sys.argv)
        standalone = True
        
    win = Interactive14DOFDrivingWindow(vehicle_params=vehicle_params, tir_file=tir_file)
    win.show()
    
    if standalone:
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()

