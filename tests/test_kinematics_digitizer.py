import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_global_optimizer import GlobalOptimizer

class TestKinematicsAndDigitizer(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_camber_curve_evaluation(self):
        z_mm, cam_f, cam_r = self.vp.get_camber_curve(z_range_mm=(-30.0, 30.0), n_pts=61)
        self.assertEqual(len(z_mm), 61)
        self.assertAlmostEqual(z_mm[0], -30.0)
        self.assertAlmostEqual(z_mm[-1], 30.0)
        # At z = 0, camber should match static camber
        idx_mid = 30
        self.assertAlmostEqual(cam_f[idx_mid], self.vp.camber_f_static, places=3)
        self.assertAlmostEqual(cam_r[idx_mid], self.vp.camber_r_static, places=3)
        # In bump (z > 0), camber gain is negative, so camber should become more negative
        self.assertLess(cam_f[-1], cam_f[idx_mid])

    def test_toe_and_rollcenter_curves(self):
        z_mm, toe_f, toe_r = self.vp.get_toe_curve(z_range_mm=(-25.0, 25.0), n_pts=51)
        self.assertEqual(len(toe_f), 51)
        self.assertAlmostEqual(toe_f[25], self.vp.toe_f_static, places=3)

        z_mm, rc_f, rc_r = self.vp.get_rollcenter_curve(z_range_mm=(-20.0, 20.0), n_pts=41)
        self.assertEqual(len(rc_f), 41)
        # Roll center at heave=0 in mm:
        self.assertAlmostEqual(rc_f[20], self.vp.rollcenter_f_coeffs[3] * 1000.0, places=2)
        self.assertAlmostEqual(rc_r[20], self.vp.rollcenter_r_coeffs[3] * 1000.0, places=2)

    def test_set_kinematic_map(self):
        new_coeffs = [0.0, 0.0, -18.5, 0.0]
        self.vp.set_kinematic_map('camber_gain_f', new_coeffs)
        self.assertEqual(self.vp.camber_gain_f_coeffs, new_coeffs)

        with self.assertRaises(ValueError):
            self.vp.set_kinematic_map('invalid_map_name', [0, 0, 0, 0])

    def test_digitizer_scaling_logic(self):
        # Suppose raw fit in mm is y(x_mm) = -0.015 * x_mm (deg per mm)
        # For 14-DOF SI (z in meters), gain should be -15.0 deg/m
        c_raw = np.array([0.0, 0.0, -0.015, 0.0])
        x_scale = 1000.0 # mm -> m
        y_scale = 1.0    # deg
        a1 = c_raw[2] * x_scale * y_scale
        self.assertAlmostEqual(a1, -15.0)

    def test_global_optimizer_understeer_tuning(self):
        tir_file = "Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir"
        opt = GlobalOptimizer(
            v_x=15.0, tir_file=tir_file, model_type="3-DOF",
            enforce_understeer=True, understeer_min_deg_g=0.15, understeer_weight=2.0
        )
        self.assertTrue(opt.enforce_understeer)
        self.assertEqual(opt.understeer_min_deg_g, 0.15)
        self.assertEqual(opt.understeer_weight, 2.0)

    def test_rollcenter_vs_roll_angle_phi(self):
        """Verify roll center height and lateral migration as functions of chassis roll angle phi [deg]."""
        phi_deg, rc_f, rc_r = self.vp.get_rollcenter_curve(phi_range_deg=(-2.0, 2.0), n_pts=41)
        self.assertEqual(len(phi_deg), 41)
        self.assertAlmostEqual(phi_deg[0], -2.0)
        self.assertAlmostEqual(phi_deg[-1], 2.0)
        # At phi = 0 deg, static roll center heights
        self.assertAlmostEqual(rc_f[20], self.vp.rollcenter_f_coeffs[3] * 1000.0, places=2)
        self.assertAlmostEqual(rc_r[20], self.vp.rollcenter_r_coeffs[3] * 1000.0, places=2)
        
        # Test lateral migration vs roll angle phi [deg]
        phi_lat_deg, rc_lat_f, rc_lat_r = self.vp.get_rollcenter_lat_curve(phi_range_deg=(-2.0, 2.0), n_pts=41)
        self.assertEqual(len(phi_lat_deg), 41)
        # At phi = 0 deg, lateral migration should be 0
        self.assertAlmostEqual(rc_lat_f[20], 0.0, places=2)
        self.assertAlmostEqual(rc_lat_r[20], 0.0, places=2)
        # Lateral migration gradient is positive with roll
        self.assertGreater(rc_lat_f[-1], 0.0)
        self.assertGreater(rc_lat_r[-1], 0.0)

    def test_active_vehicle_parameters_inspector_dialog(self):
        """Verify that ActiveVehicleParametersInspectorDialog builds and exposes all 5 tabs."""
        from PyQt5.QtWidgets import QApplication
        from src.ui.vehicle_parameters_inspector_dialog import ActiveVehicleParametersInspectorDialog
        app = QApplication.instance() or QApplication([])
        dlg = ActiveVehicleParametersInspectorDialog(self.vp)
        self.assertEqual(dlg.tabs.count(), 5)
        # Check that table rows exist in kinematics tab (all 19 curves)
        self.assertEqual(dlg.tab_kin.rowCount(), 19)
        # Check geometry tab
        self.assertGreater(dlg.tab_geo.rowCount(), 10)

    def test_ackermann_curve_evaluation(self):
        """Verify Ackermann percentage curve and wheel angles evaluation."""
        d_sw, ack_pct = self.vp.get_ackermann_curve(delta_range_deg=(-125.0, 125.0), n_pts=51)
        self.assertEqual(len(d_sw), 51)
        # At delta=0, nominal value around 51%
        mid_idx = 25
        self.assertAlmostEqual(d_sw[mid_idx], 0.0)
        self.assertAlmostEqual(ack_pct[mid_idx], 50.0, places=1)
        # Symmetric: at -125 and +125, same percentage
        self.assertAlmostEqual(ack_pct[0], ack_pct[-1], places=2)
        self.assertGreater(ack_pct[-1], ack_pct[mid_idx])

        # Test wheel steer angles
        d_sw, fl, fr, diff = self.vp.get_steer_wheel_angles(delta_range_deg=(-125.0, 125.0), n_pts=51)
        self.assertAlmostEqual(diff[mid_idx], 0.0, places=2)
        # In a right turn (positive delta), inside right wheel steers sharper than outside left wheel
        self.assertGreater(diff[-1], 2.0)

    def test_preset_kinematics_setups_load(self):
        """Verify kinematics presets load curves successfully."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        f2026 = os.path.join(base_dir, 'data', 'setups', 'et18_2026_kinematics.json') if os.path.exists(os.path.join(base_dir, 'data', 'setups', 'et18_2026_kinematics.json')) else os.path.join(base_dir, 'setups', 'et18_2026_kinematics.json')
        f2025 = os.path.join(base_dir, 'data', 'setups', 'et17_2025_kinematics.json') if os.path.exists(os.path.join(base_dir, 'data', 'setups', 'et17_2025_kinematics.json')) else os.path.join(base_dir, 'setups', 'et17_2025_kinematics.json')
        
        if not os.path.exists(f2026) or not os.path.exists(f2025):
            # In public repository, verify public baseline preset
            f_mock = os.path.join(base_dir, 'data', 'setups', 'fsae_mockup.json')
            self.assertTrue(os.path.exists(f_mock))
            vp_mock = VehicleParams.load_json(f_mock)
            self.assertEqual(len(vp_mock.camber_gain_f_coeffs), 4)
            self.assertEqual(len(vp_mock.rollcenter_f_coeffs), 4)
            return

        vp26 = VehicleParams.load_json(f2026)
        vp25 = VehicleParams.load_json(f2025)
        self.assertEqual(len(vp26.camber_gain_f_coeffs), 4)
        self.assertEqual(len(vp25.camber_gain_f_coeffs), 4)
        self.assertEqual(len(vp26.rollcenter_f_coeffs), 4)
        self.assertEqual(len(vp25.rollcenter_f_coeffs), 4)
        self.assertAlmostEqual(vp26.rollcenter_f_coeffs[3] * 1000.0, 46.3, places=1)
        self.assertAlmostEqual(vp25.rollcenter_f_coeffs[3] * 1000.0, 43.9, places=1)
        # Check Ackermann loaded in presets
        self.assertAlmostEqual(vp26.ackermann_pct * 100.0, 51.0, places=1)
        self.assertAlmostEqual(vp25.ackermann_pct * 100.0, 64.3, places=1)

if __name__ == '__main__':
    unittest.main()
