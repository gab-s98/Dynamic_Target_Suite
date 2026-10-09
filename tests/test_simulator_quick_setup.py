import unittest
import os
import json
import time
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.models.tire_degradation import TireDegradationModel, TireState, TireDegradationParams
from src.ui.interactive_driver_14dof import PhysicsThread14DOF

try:
    from PyQt5.QtWidgets import QApplication
    from src.ui.interactive_driver_14dof import Interactive14DOFDrivingWindow
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False


class TestSimulatorQuickSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_PYQT5:
            cls.app = QApplication.instance() or QApplication([])
        else:
            cls.app = None

    def test_01_et18_baseline_and_pressures(self):
        vp = VehicleParams()
        self.assertAlmostEqual(vp.aero_balance, 0.5, places=3)
        self.assertTrue(hasattr(vp, 'tire_pressure_f_bar'))
        self.assertTrue(hasattr(vp, 'tire_pressure_r_bar'))
        self.assertAlmostEqual(vp.tire_pressure_f_bar, 0.85, places=2)
        self.assertAlmostEqual(vp.tire_pressure_r_bar, 0.85, places=2)

        setups_path = os.path.join(os.path.dirname(__file__), '..', 'setups', 'et18_2026_kinematics.json')
        if os.path.exists(setups_path):
            with open(setups_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertAlmostEqual(data.get('aero_balance', 0.0), 0.509, places=3)

    def test_02_tire_thermal_recalibration(self):
        p = TireDegradationParams()
        self.assertAlmostEqual(p.thermal_mass_bulk, 2200.0, places=1)
        self.assertAlmostEqual(p.h_conv_bulk, 9.0, places=1)

        model = TireDegradationModel(mode='light', params=p)
        ts = TireState(T_bulk=60.0)
        model.step(ts, F_total=0.0, v_slip=0.0, Fz=300.0, vx=15.0, dt=1.0)
        delta_T = 60.0 - ts.T_bulk
        self.assertGreater(delta_T, 0.0)
        self.assertLess(delta_T, 1.5, f'Cooling too fast: delta_T = {delta_T:.2f} C in 1 sec')

        model_det = TireDegradationModel(mode='detailed', params=p)
        ts_det = TireState(T_surface=70.0, T_tread=50.0, T_carcass=40.0)
        T_eff = model_det.get_display_temperature(ts_det)
        expected_eff = 0.35 * 70.0 + 0.50 * 50.0 + 0.15 * 40.0
        self.assertAlmostEqual(T_eff, expected_eff, places=2)

    def test_03_physics_thread_live_tuning(self):
        vp = VehicleParams()
        th = PhysicsThread14DOF(vehicle_params=vp, target_hz=200.0)

        th.update_live_tuning(
            camber_f=-2.2,
            camber_r=-1.2,
            toe_f=-0.3,
            toe_r=0.4,
            preload_f=300.0,
            preload_r=350.0,
            aero_bal=0.46,
            Cz=1.85,
            Cd=0.72,
            press_f=0.90,
            press_r=0.95
        )

        self.assertTrue(th._tuning_dirty)
        self.assertAlmostEqual(th.vp.camber_f_static, -2.2, places=2)
        self.assertAlmostEqual(th.vp.camber_r_static, -1.2, places=2)
        self.assertAlmostEqual(th.vp.toe_f_static, -0.3, places=2)
        self.assertAlmostEqual(th.vp.toe_r_static, 0.4, places=2)
        self.assertAlmostEqual(th.vp.preload_f_N, 300.0, places=1)
        self.assertAlmostEqual(th.vp.preload_r_N, 350.0, places=1)
        self.assertAlmostEqual(th.vp.aero_balance, 0.46, places=2)
        self.assertAlmostEqual(th.vp.Cz, 1.85, places=2)
        self.assertAlmostEqual(th.vp.Cd, 0.72, places=2)
        self.assertAlmostEqual(th.vp.tire_pressure_f_bar, 0.90, places=2)
        self.assertAlmostEqual(th.vp.tire_pressure_r_bar, 0.95, places=2)

    def test_04_telemetry_stint_archive_on_reset(self):
        vp = VehicleParams()
        th = PhysicsThread14DOF(vehicle_params=vp, target_hz=200.0)

        with th._history_lock:
            th.telemetry_history = [{'vx_kmh': 50.0, 'ay_g': 1.2, 'time_sim': 1.0 + i} for i in range(50)]

        self.assertEqual(len(th.get_telemetry_snapshot()), 50)
        self.assertEqual(len(th.get_last_stint_snapshot()), 0)

        th.reset_state(vx=0.0)

        self.assertEqual(len(th.get_telemetry_snapshot()), 0)
        archived = th.get_last_stint_snapshot()
        self.assertEqual(len(archived), 50)
        self.assertEqual(archived[0]['vx_kmh'], 50.0)

    @unittest.skipUnless(HAS_PYQT5, 'PyQt5 not available')
    def test_05_gui_interactive_setup_tab_and_mdo_sync(self):
        vp = VehicleParams()
        win = Interactive14DOFDrivingWindow(vehicle_params=vp)

        self.assertTrue(hasattr(win, 'spn_camber_f'))
        self.assertTrue(hasattr(win, 'spn_camber_r'))
        self.assertTrue(hasattr(win, 'spn_toe_f'))
        self.assertTrue(hasattr(win, 'spn_toe_r'))
        self.assertTrue(hasattr(win, 'spn_press_f'))
        self.assertTrue(hasattr(win, 'spn_press_r'))
        self.assertTrue(hasattr(win, 'spn_preload_f'))
        self.assertTrue(hasattr(win, 'spn_preload_r'))
        self.assertTrue(hasattr(win, 'spn_aero_cz'))
        self.assertTrue(hasattr(win, 'spn_aero_cx'))
        self.assertTrue(hasattr(win, 'spn_aero_bal'))
        self.assertTrue(hasattr(win, 'lbl_preload_badge'))
        self.assertTrue(hasattr(win, 'lbl_aero_badge'))
        self.assertTrue(hasattr(win, 'combo_aero_preset'))
        self.assertTrue(hasattr(win, 'combo_wing_f'))
        self.assertTrue(hasattr(win, 'combo_wing_r'))

        emitted_vps = []
        win.setup_changed.connect(lambda v: emitted_vps.append(v))

        win.combo_aero_preset.setCurrentIndex(1)
        self.assertAlmostEqual(win.spn_aero_cz.value(), 1.850, places=2)
        self.assertAlmostEqual(win.spn_aero_cx.value(), 0.720, places=2)
        self.assertAlmostEqual(win.spn_aero_bal.value(), 46.0, places=1)
        self.assertGreater(len(emitted_vps), 0)
        self.assertAlmostEqual(emitted_vps[-1].Cz, 1.850, places=2)

        win._reset_to_et18_baseline(show_dialog=False)
        self.assertAlmostEqual(win.spn_aero_cz.value(), 3.421, places=2)
        self.assertAlmostEqual(win.spn_aero_cx.value(), 1.200, places=2)
        self.assertAlmostEqual(win.spn_aero_bal.value(), 50.9, places=1)
        self.assertAlmostEqual(win.spn_camber_f.value(), -2.5, places=1)
        self.assertAlmostEqual(win.spn_press_f.value(), 0.85, places=2)

        win.physics_thread.running = False
        win.physics_thread.wait(500)
        win.close()

if __name__ == '__main__':
    unittest.main()
