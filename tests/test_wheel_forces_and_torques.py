import unittest
import math
import time
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.ui.interactive_driver_14dof import (
    PhysicsThread14DOF,
    FullTelemetryWindow
)

try:
    from PyQt5.QtWidgets import QApplication
    from src.ui.interactive_driver_14dof import Interactive14DOFDrivingWindow
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False


class TestWheelForcesAndTorques(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_PYQT5:
            cls.app = QApplication.instance() or QApplication([])
        else:
            cls.app = None

    def test_01_full_telemetry_window_plots_and_presets(self):
        """Verifies plots 18-23 and Preset 6 exist in FullTelemetryWindow."""
        expected_plots = [
            'wheel_torques',
            'longitudinal_forces',
            'triaxial_forces_fl',
            'triaxial_forces_fr',
            'triaxial_forces_rl',
            'triaxial_forces_rr'
        ]
        for p in expected_plots:
            self.assertIn(p, FullTelemetryWindow.AVAILABLE_PLOTS, f"Missing plot key {p} in AVAILABLE_PLOTS")

        fea_key = next((k for k in FullTelemetryWindow.PRESETS if "Carichi Sospensioni" in k and "Triassiali" in k), None)
        self.assertIsNotNone(fea_key, "Preset 'Carichi Sospensioni & Forze Triassiali (FEA)' not found in PRESETS")
        preset_fea = FullTelemetryWindow.PRESETS[fea_key]
        self.assertIn('longitudinal_forces', preset_fea)
        self.assertIn('vertical_loads', preset_fea)
        self.assertIn('wheel_torques', preset_fea)

    def test_02_physics_thread_live_emission(self):
        """Verifies telem_data produced by PhysicsThread contains all force and torque keys."""
        vp = VehicleParams()
        th = PhysicsThread14DOF(vehicle_params=vp, target_hz=250.0)

        received_states = []
        th.state_updated.connect(lambda s: received_states.append(s))

        th.start()
        start_t = time.time()
        while time.time() - start_t < 3.0 and len(received_states) < 3:
            if self.app:
                self.app.processEvents()
            time.sleep(0.02)
        th.stop()

        self.assertGreater(len(received_states), 0, "No state emitted by physics thread")
        s0 = received_states[-1]

        # Verify Fx, Fy, Fz, F_tot on all 4 wheels
        for corner in ['fl', 'fr', 'rl', 'rr']:
            self.assertIn(f'Fx_{corner}', s0)
            self.assertIn(f'Fy_{corner}', s0)
            self.assertIn(f'Fz_{corner}', s0)
            self.assertIn(f'F_tot_{corner}', s0)
            self.assertIn(f'T_drive_{corner}', s0)
            self.assertIn(f'T_brake_{corner}', s0)
            self.assertIn(f'T_net_{corner}', s0)

            # Normal load Fz must be positive
            self.assertGreater(s0[f'Fz_{corner}'], 0.0)
            # Resultant force |F| must be >= 0.0
            self.assertGreaterEqual(s0[f'F_tot_{corner}'], 0.0)

    @unittest.skipUnless(HAS_PYQT5, 'PyQt5 not available')
    def test_03_gui_loads_and_torques_elements(self):
        """Verifies Interactive14DOFDrivingWindow has all loads and torques UI controls and updates properly."""
        vp = VehicleParams()
        win = Interactive14DOFDrivingWindow(vehicle_params=vp)

        # 1. Triaxial Loads elements
        self.assertTrue(hasattr(win, 'combo_loads_view'))
        self.assertEqual(win.combo_loads_view.count(), 5)
        self.assertTrue(hasattr(win, 'btn_reset_load_peaks'))
        self.assertTrue(hasattr(win, 'widget_loads_table'))
        self.assertTrue(hasattr(win, 'widget_loads_single'))
        self.assertTrue(hasattr(win, 'lbl_fx_fl'))
        self.assertTrue(hasattr(win, 'lbl_fy_fl'))
        self.assertTrue(hasattr(win, 'lbl_fz_fl'))
        self.assertTrue(hasattr(win, 'lbl_ftot_fl'))

        # 2. Torques elements
        self.assertTrue(hasattr(win, 'lbl_tdrv_rl'))
        self.assertTrue(hasattr(win, 'lbl_tdrv_rr'))
        self.assertTrue(hasattr(win, 'lbl_tbrk_fl'))
        self.assertTrue(hasattr(win, 'lbl_tbrk_fr'))
        self.assertTrue(hasattr(win, 'lbl_tnet_fl'))
        self.assertTrue(hasattr(win, 'lbl_tnet_rl'))
        self.assertTrue(hasattr(win, 'lbl_torque_summary'))

        # 3. Test view switching
        win.combo_loads_view.setCurrentIndex(1)  # FL single view
        self.assertTrue(win.widget_loads_table.isHidden())
        self.assertFalse(win.widget_loads_single.isHidden())

        win.combo_loads_view.setCurrentIndex(0)  # Table view
        self.assertFalse(win.widget_loads_table.isHidden())
        self.assertTrue(win.widget_loads_single.isHidden())

        # 4. Test on_physics_frame update with synthetic test state
        dummy_state = {
            'x': 10.0, 'y': 5.0, 'psi_deg': 0.0,
            'physics_hz': 250.0,
            'step_ms': 0.35,
            'rt_ratio': 1.0,
            'vx_kmh': 72.0,
            'lap_time': 12.5,
            'lap_count': 1,
            'lap_time_current': 12.5,
            'lap_time_last': 0.0,
            'lap_time_best': 0.0,
            'ay_g': 1.25,
            'ax_g': 0.45,
            'phi_deg': -1.2,
            'theta_deg': 0.4,
            'z_fl_mm': 12.0, 'z_fr_mm': -8.0,
            'z_rl_mm': 6.0,  'z_rr_mm': -4.0,
            'camber_fl_deg': -2.8, 'camber_fr_deg': -1.1,
            'dFz_jack_N': 15.0,
            'h_rc_f_mm': 45.0, 'y_rc_f_mm': 2.0,
            'h_rc_r_mm': 75.0, 'y_rc_r_mm': -1.0,
            'haptic_brake_lt': 0.0, 'haptic_throt_rt': 0.0,
            'rumble_heavy': 0.0, 'rumble_light': 0.0,
            'understeer_pct': 0.0, 'oversteer_pct': 0.0,
            'beta_deg': 0.5,
            'ctrl_connected': False,
            'steer_deg': 10.0,
            'throttle_pct': 80.0,
            'brake_pct': 20.0,
            'util_fl': 65.0, 'util_fr': 40.0,
            'util_rl': 75.0, 'util_rr': 55.0,
            'Fz_fl': 650.0, 'Fz_fr': 350.0,
            'Fz_rl': 850.0, 'Fz_rr': 550.0,
            'Fy_fl': 520.0, 'Fy_fr': 280.0,
            'Fy_rl': 680.0, 'Fy_rr': 440.0,
            'Fx_fl': -60.0, 'Fx_fr': -60.0,
            'Fx_rl': 320.0, 'Fx_rr': 320.0,
            'F_tot_fl': 834.0, 'F_tot_fr': 452.0,
            'F_tot_rl': 1135.0, 'F_tot_rr': 773.0,
            'T_drive_fl': 0.0, 'T_drive_fr': 0.0,
            'T_drive_rl': 160.0, 'T_drive_rr': 160.0,
            'T_brake_fl': 84.0, 'T_brake_fr': 84.0,
            'T_brake_rl': 56.0, 'T_brake_rr': 56.0,
            'T_net_fl': -84.0, 'T_net_fr': -84.0,
            'T_net_rl': 104.0, 'T_net_rr': 104.0
        }

        win.on_physics_frame(dummy_state)

        # Check updated labels
        self.assertEqual(win.lbl_fx_rl.text(), "+320 N")
        self.assertEqual(win.lbl_fy_fl.text(), "+520 N")
        self.assertEqual(win.lbl_fz_fl.text(), "650 N")
        self.assertEqual(win.lbl_ftot_fl.text(), "834 N")
        self.assertEqual(win.lbl_tdrv_rl.text(), "160.0 Nm")
        self.assertEqual(win.lbl_tbrk_fl.text(), "84.0 Nm")
        self.assertEqual(win.lbl_tnet_rl.text(), "+104.0 Nm")
        self.assertGreater(win.peak_loads['fl'], 800.0)

        # Test peak reset
        win._reset_load_peaks()
        self.assertEqual(win.peak_loads['fl'], 0.0)

        if hasattr(win, 'physics_thread') and win.physics_thread.isRunning():
            win.physics_thread.running = False
            win.physics_thread.wait(500)
        win.close()


if __name__ == '__main__':
    unittest.main()
