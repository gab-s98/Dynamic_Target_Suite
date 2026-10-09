# -*- coding: utf-8 -*-
"""
tests/test_simulator_3d.py
--------------------------
Unit and integration tests for the modular 3D Driving Simulator:
1. Modular imports and entry points (simulator.py and interactive_driver_14dof.py).
2. OpenGL 3D Viewport camera projections, track mesh rendering, and frame buffer capture.
3. Multi-frequency tire thermal & degradation model integration (10, 25, 50, 100 Hz, sync).
4. Physical driving input authority (steering, throttle, brake) and watchdog safety.
"""

import os
import sys
import unittest
import time
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams


class TestSimulator3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_01_modular_exports_and_entrypoint(self):
        """Verify backwards-compatibility proxy and modular package exports."""
        from src.ui import interactive_driver_14dof as id14
        expected_symbols = [
            'GamepadManager', 'FFB_PRESETS', 'XINPUT_VIBRATION_EX',
            'PhysicsThread14DOF',
            'OpenGLViewport3D', '_LegacyViewport3D', 'get_viewport_class',
            'TelemetryPopoutDialog', 'CustomTelemetryLayoutDialog', 'LapSelectionDialog', 'FullTelemetryWindow',
            'TrackEditorCanvas', 'RoadProfileEditorDialog', 'TrackDesignerDialog',
            'Interactive14DOFDrivingWindow'
        ]
        for sym in expected_symbols:
            self.assertTrue(hasattr(id14, sym), f"Missing symbol {sym} in interactive_driver_14dof")

        from src.core import simulator
        self.assertTrue(hasattr(simulator, 'main'))
        self.assertTrue(hasattr(simulator, 'load_vehicle_params'))

    def test_02_viewport_camera_modes_and_rendering(self):
        """Verify that OpenGLViewport3D renders cleanly across all camera and view modes."""
        from src.ui.sim_viewport import OpenGLViewport3D
        vp = OpenGLViewport3D()
        vp.resize(640, 480)
        vp.show()
        self.app.processEvents()
        
        # Feed mock vehicle state
        mock_state = {
            'x': 10.0, 'y': 5.0, 'psi_deg': 15.0,
            'vx_kmh': 60.0, 'ay_g': 0.8, 'ax_g': 0.2,
            'phi_deg': 1.0, 'theta_deg': 0.5,
            'steer_deg': 5.0, 'throttle_pct': 70.0, 'brake_pct': 0.0,
            'z_road': 0.0, 'grade_road': 0.01, 'bank_road_deg': 1.0,
            'lap_time': 12.3, 'lap_count': 1, 'physics_hz': 250.0, 'step_ms': 1.2,
            'rt_ratio': 1.0, 'r_deg_s': 8.0, 'T_sw_Nm': 12.0,
            'dFz_jack_N': 25.0, 'z_fl_mm': 5.0, 'z_fr_mm': 4.5, 'z_rl_mm': 6.0, 'z_rr_mm': 5.8,
            'camber_fl_deg': -2.0, 'camber_fr_deg': -2.0, 'camber_rl_deg': -1.5, 'camber_rr_deg': -1.5,
            'toe_fl_deg': 0.1, 'toe_fr_deg': 0.1, 'toe_rl_deg': 0.0, 'toe_rr_deg': 0.0,
            'Fz_fl': 850.0, 'Fz_fr': 850.0, 'Fz_rl': 950.0, 'Fz_rr': 950.0,
            'Fy_fl': 500.0, 'Fy_fr': 500.0, 'Fy_rl': 600.0, 'Fy_rr': 600.0,
            'Fx_fl': 0.0, 'Fx_fr': 0.0, 'Fx_rl': 400.0, 'Fx_rr': 400.0,
            'alpha_fl': 2.5, 'alpha_fr': 2.5, 'alpha_rl': 2.0, 'alpha_rr': 2.0,
            'kappa_fl': 0.0, 'kappa_fr': 0.0, 'kappa_rl': 0.03, 'kappa_rr': 0.03,
            'util_fl': 65.0, 'util_fr': 65.0, 'util_rl': 75.0, 'util_rr': 75.0
        }
        vp.update_vehicle_state(mock_state)

        # 3D Camera Modes
        for cam in ["CHASE", "COCKPIT", "NOSE", "BUMPER"]:
            vp.camera_mode = cam
            vp.view_mode = "3D"
            vp.makeCurrent()
            vp.paintGL()
            self.app.processEvents()

        # 2D Map View Mode
        vp.view_mode = "2D"
        vp.makeCurrent()
        vp.paintGL()
        self.app.processEvents()
        vp.close()

    def test_03_thermal_frequency_configurations(self):
        """Verify thermal integration frequency switching and physics loop coupling."""
        from src.core.physics_thread import PhysicsThread14DOF
        veh_p = VehicleParams()
        th = PhysicsThread14DOF(veh_p)

        # Test configurable frequencies
        frequencies = [10.0, 25.0, 50.0, 100.0, 0.0]
        for f in frequencies:
            th.set_thermal_hz(f)
            self.assertEqual(th.thermal_hz, f)

        # Test wear acceleration scale
        th.set_tire_wear_scale(5.0)
        self.assertEqual(th.tire_deg_model.params.wear_rate_scale, 5.0)

        # Test thermal modes
        th.set_tire_degradation_mode("detailed")
        self.assertEqual(th.tire_deg_model.mode, "detailed")
        th.set_tire_degradation_mode("light")
        self.assertEqual(th.tire_deg_model.mode, "light")
        th.set_tire_degradation_mode("off")
        self.assertEqual(th.tire_deg_model.mode, "off")

    def test_04_driving_window_headless_execution(self):
        """Verify Interactive14DOFDrivingWindow boots, receives physics frames, and handles key inputs."""
        from src.ui.sim_main_window import Interactive14DOFDrivingWindow
        veh_p = VehicleParams()
        win = Interactive14DOFDrivingWindow(vehicle_params=veh_p)
        win.show()

        try:
            # Check FFB presets count
            self.assertGreaterEqual(win.combo_ffb_presets.count(), 5)
            self.assertGreaterEqual(win.combo_thermal_hz.count(), 5)

            # Wait for initial physics frame
            t0 = time.time()
            while time.time() - t0 < 3.0 and win.viewport.car_state is None:
                self.app.processEvents()
                time.sleep(0.02)

            self.assertIsNotNone(win.viewport.car_state, "Physics thread failed to emit state within 3s")
            self.assertIn('physics_hz', win.viewport.car_state)
            self.assertIn('vx_kmh', win.viewport.car_state)

            # Test keyboard input authority
            ev_throttle = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
            win.keyPressEvent(ev_throttle)
            self.assertTrue(win.physics_thread.key_states.get(Qt.Key_W, False))

            ev_release = QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_W, Qt.NoModifier)
            win.keyReleaseEvent(ev_release)
            self.assertFalse(win.physics_thread.key_states.get(Qt.Key_W, True))

        finally:
            win.physics_thread.running = False
            win.physics_thread.wait(500)
            win.close()


if __name__ == '__main__':
    unittest.main()
