"""
test_mdo_gui_lifecycle.py
-------------------------
Verifica automatizzata del ciclo di vita di MainWindowPro:
1. Assenza di distruzione C++ DeferredDelete sui widget ereditati da MainWindow.
2. Corretto instradamento e sincronizzazione dei workspace Pro tramite tabs.setCurrentWidget.
3. Esecuzione corretta di run_constant_parameters() su modelli 3-DOF, 7-DOF e 14-DOF.
4. Controllo integrità delle impostazioni di tema e densità.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

from src.ui import main_gui_pro as gui_pro
from src.ui import main_gui_classic as gui_classic

class TestMDOGUILifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        self.win = gui_pro.MainWindowPro()

    def tearDown(self):
        if hasattr(self, 'win') and self.win:
            self.win.close()
            self.win.deleteLater()
            self.app.processEvents()

    def test_no_deferred_delete_on_widgets(self):
        """Verifica che nessun widget critico venga distrutto dal DeferredDelete di Qt."""
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()

        from PyQt5 import sip

        critical_widgets = [
            self.win.ymd_show_radii_cb,
            self.win.tabs,
            self.win.tab_ymd,
            self.win.tab_ymd_single,
            self.win.tab_ymd_multi,
            self.win.tab_kpi,
            self.win.tab_telemetry,
            self.win.tab_susp_sizing,
            self.win.tab_sweep_res,
            self.win.btn_recalc_ymd,
            self.win.ymd_vx_input,
        ]

        for w in critical_widgets:
            self.assertIsNotNone(w, "Widget critico non deve essere None")
            self.assertFalse(sip.isdeleted(w), f"Widget {w} risulta eliminato in C++!")
            _ = w.isVisible()

    def test_tab_synchronization_and_navigation(self):
        """Verifica che il routing da tabs.setCurrentWidget commuti al workspace Pro corrispondente."""
        # 1. Navigazione a KPI Dashboard -> Workspace 4
        self.win.tabs.setCurrentWidget(self.win.tab_kpi)
        self.assertEqual(self.win.active_workspace_idx, 4)

        # 2. Navigazione a Skidpad -> Workspace 1
        self.win.tabs.setCurrentWidget(self.win.tab_telem_skidpad)
        self.assertEqual(self.win.active_workspace_idx, 1)

        # 3. Navigazione a Sospensioni -> Workspace 2
        self.win.tabs.setCurrentWidget(self.win.tab_susp_sizing)
        self.assertEqual(self.win.active_workspace_idx, 2)

        # 4. Navigazione a Sweep OpenMDAO -> Workspace 3
        self.win.tabs.setCurrentWidget(self.win.tab_sweep_res)
        self.assertEqual(self.win.active_workspace_idx, 3)

        # 5. Navigazione a YMD Singolo -> Workspace 0
        self.win.tabs.setCurrentWidget(self.win.tab_ymd_single)
        self.assertEqual(self.win.active_workspace_idx, 0)

    def test_run_constant_parameters_all_models(self):
        """Verifica che run_constant_parameters() completi con successo su 3-DOF, 7-DOF e 14-DOF."""
        self.win.auto_maps_cb.setChecked(False)

        for model_idx in [0, 1, 2]:
            self.win.model_combo.setCurrentIndex(model_idx)
            self.win._rebuild_params_panel()
            
            try:
                self.win.run_constant_parameters()
            except RuntimeError as e:
                self.fail(f"run_constant_parameters ha sollevato RuntimeError sul modello {model_idx}: {e}")

            self.assertTrue(hasattr(self.win, 'last_optimized_params'))
            self.assertIsNotNone(self.win.last_optimized_params)

    def test_theme_and_density_switching(self):
        """Verifica che la commutazione dinamica di tema e densità avvenga senza errori."""
        self.win.set_theme("dark")
        self.assertEqual(self.win.current_theme, "dark")

        self.win.set_density("compact")
        self.assertEqual(self.win.current_density, "compact")

        self.win.set_theme("light")
        self.assertEqual(self.win.current_theme, "light")

        self.win.set_density("standard")
        self.assertEqual(self.win.current_density, "standard")

if __name__ == '__main__':
    unittest.main()
