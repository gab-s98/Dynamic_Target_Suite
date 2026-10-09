# -*- coding: utf-8 -*-
"""
Tests for 14-DOF Speed Sweep Convergence and Simulator Camera / 3D Track Fixes.
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.ui.gui_workers import _worker_optimize_single_speed


class TestSweepAndCameraFixes(unittest.TestCase):
    def test_tire_params_safe_fallback(self):
        """Test that TireParams gracefully resolves empty or non-existent paths."""
        t1 = TireParams("")
        self.assertGreater(len(t1.params), 10, "Empty path should fallback to default Hoosier TIR")
        t2 = TireParams(os.path.join(project_root, "tire_tir"))
        self.assertGreater(len(t2.params), 10, "Directory path should fallback to default Hoosier TIR")
        t3 = TireParams("non_existent_tire_file_12345.tir")
        self.assertGreater(len(t3.params), 10, "Non-existent file should fallback to default Hoosier TIR")

    def test_speed_sweep_14dof_convergence(self):
        """Test that 14-DOF speed sweep optimization returns realistic lateral acceleration > 1.0g."""
        bounds = [(1.5, 1.6), (1.2, 1.3), (1.1, 1.3), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6),
                  (-4.0, 0.0), (-4.0, 0.0), (-1.0, 1.0), (-1.0, 1.0),
                  (10000.0, 60000.0), (10000.0, 60000.0), (15000.0, 60000.0), (15000.0, 60000.0)]
        x0 = [1.53, 1.25, 1.20, 0.50, 0.50, 0.50, -2.0, -1.5, 0.0, 0.0, 35000.0, 35000.0, 35000.0, 35000.0]

        # Test at 8.0 m/s with empty tir_file
        args_empty = (8.0, x0, bounds, 1.0, '', 0.0, False, 'max_ay', 15.25, 75.0, '14-DOF', 0.1, 1.0, 0.0)
        res_empty = _worker_optimize_single_speed(args_empty)
        self.assertTrue(res_empty.get('success', False))
        self.assertGreater(res_empty.get('max_ay', 0.0), 1.0, "8.0 m/s max_ay should be > 1.0g with empty tir_file")

        # Test at 15.0 m/s with nominal tir_file
        args_nom = (15.0, x0, bounds, 1.0, 'Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir', 0.0, False, 'max_ay', 15.25, 75.0, '14-DOF', 0.1, 1.0, 0.0)
        res_nom = _worker_optimize_single_speed(args_nom)
        self.assertTrue(res_nom.get('success', False))
        self.assertGreater(res_nom.get('max_ay', 0.0), 1.2, "15.0 m/s max_ay should be > 1.2g")

    def test_camera_heading_unwrap_and_track_invalidation(self):
        """Test circular heading unwrapping and 3D track display list invalidation."""
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5.QtWidgets import QApplication
        from src.ui.interactive_driver_14dof import Interactive14DOFDrivingWindow

        app = QApplication.instance() or QApplication(sys.argv)
        vp = VehicleParams()
        win = Interactive14DOFDrivingWindow(vehicle_params=vp, tir_file="")

        # Check slider limits
        self.assertGreaterEqual(win.sl_dist.maximum(), 150)
        self.assertGreaterEqual(win.sl_height.maximum(), 100)

        # Heading unwrapping
        win.viewport.smooth_heading = 3.10
        state = {'x': 10.0, 'y': 20.0, 'psi_deg': -175.0, 'z_road': 0.0, 'grade_road': 0.05, 'bank_road_deg': 2.0}
        win.viewport.update_vehicle_state(state)
        self.assertGreater(abs(win.viewport.smooth_heading), 2.8, "Heading should not spin across +/- pi")

        # Track display list invalidation
        win.viewport.track_display_list_valid = True
        old_ver = win.physics_thread.track_data.get('_version', 0)
        win.cb_road_macro.setChecked(True)
        win._on_road_profile_params_changed()
        new_ver = win.physics_thread.track_data.get('_version', 0)
        self.assertFalse(win.viewport.track_display_list_valid, "Display list must be invalidated on road profile change")
        self.assertGreater(new_ver, old_ver, "Track version must increment on road profile change")

        # Stop physics thread cleanly
        if hasattr(win, 'physics_thread') and win.physics_thread.isRunning():
            win.physics_thread.running = False
            win.physics_thread.wait(500)
        win.close()


if __name__ == '__main__':
    unittest.main()
