import sys
import os
import multiprocessing

if hasattr(sys, '_MEIPASS'):
    casadi_dir = os.path.join(sys._MEIPASS, 'casadi')
    if os.path.exists(casadi_dir):
        os.add_dll_directory(casadi_dir)

from PyQt5.QtWidgets import QApplication
from src.ui.main_gui_pro import MainWindowPro
MainWindow = MainWindowPro
from src.ui import ui_theme_engine

def main():
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    settings = ui_theme_engine.load_ui_settings()
    
    if "--preview-3d" in sys.argv or "preview_car_3d.py" in " ".join(sys.argv):
        try:
            from scripts.preview_car_3d import CarPreviewStudioWindow
        except ImportError:
            from preview_car_3d import CarPreviewStudioWindow
        v_type = 'b_segment_suv' if ('b_segment_suv' in sys.argv or 'suv' in sys.argv) else 'formula'
        window = CarPreviewStudioWindow(initial_vehicle=v_type)
        window.show()
        sys.exit(app.exec_())

    window = MainWindowPro()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
