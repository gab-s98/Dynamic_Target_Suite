"""
test_popout_and_real_suspension.py
----------------------------------
Unit tests for the standalone pop-out dialog cloning, in-app user guide,
and real damper stroke, bump-stop, and installed spring rate models.
"""

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.models.params_vehicle import VehicleParams
from src.utils.suspension_sizing import calculate_suspension_and_arb_sizing


class TestRealSuspensionAndPopout(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_bumpstop_force_evaluation(self):
        gap = self.vp.bumpstop_gap_f_m
        # Deflection below gap: force should be zero
        f_low = self.vp.eval_bumpstop_force(gap - 0.010, corner='f')
        self.assertEqual(f_low, 0.0)

        # Deflection above gap: force should be non-zero and progressive
        f_bump1 = self.vp.eval_bumpstop_force(gap + 0.005, corner='f')
        f_bump2 = self.vp.eval_bumpstop_force(gap + 0.010, corner='f')
        self.assertGreater(f_bump1, 0.0)
        self.assertGreater(f_bump2, f_bump1)

    def test_wheel_rate_installed(self):
        # K_w = K_spring * MR^2 [N/m]
        kw_f = self.vp.eval_wheel_rate_installed(corner='f')
        expected_kw_f = self.vp.spring_rate_installed_f_Nmm * 1000.0 * (self.vp.motion_ratio_f ** 2)
        self.assertAlmostEqual(kw_f, expected_kw_f, places=2)

        kw_r = self.vp.eval_wheel_rate_installed(corner='r')
        expected_kw_r = self.vp.spring_rate_installed_r_Nmm * 1000.0 * (self.vp.motion_ratio_r ** 2)
        self.assertAlmostEqual(kw_r, expected_kw_r, places=2)

    def test_dynamic_ground_clearance(self):
        h_f_nom, h_r_nom, bottom_nom = self.vp.eval_dynamic_ground_clearance(0.0, 0.0)
        self.assertAlmostEqual(h_f_nom, self.vp.static_ride_height_f_m * 1000.0)
        self.assertFalse(bottom_nom)

        # Heavy pitch/heave bottoming out test
        h_f_bot, h_r_bot, bottom_bot = self.vp.eval_dynamic_ground_clearance(0.050, 0.0)
        self.assertLessEqual(h_f_bot, 0.0)
        self.assertTrue(bottom_bot)

    def test_suspension_sizing_with_real_damper(self):
        res = calculate_suspension_and_arb_sizing(
            m_total_kg=300.0,
            wd_f=0.50,
            lltd_target=0.55,
            t_f_m=1.20,
            t_r_m=1.20,
            f_ride_f_hz=3.0,
            f_ride_r_hz=3.3,
            mr_spring_f=0.85,
            mr_spring_r=0.85,
            mr_arb_f=0.65,
            mr_arb_r=0.65,
            arb_material="Steel",
            k_spring_installed_f_Nmm=42.0,
            k_spring_installed_r_Nmm=48.0,
            damper_stroke_bump_mm=35.0,
            damper_stroke_reb_mm=30.0,
            bumpstop_gap_mm=20.0,
            static_ride_height_f_mm=30.0,
            static_ride_height_r_mm=35.0
        )

        self.assertIn('Kw_inst_f_Nmm', res)
        self.assertIn('fn_inst_f_hz', res)
        self.assertIn('z_static_sag_f_mm', res)
        self.assertIn('bump_headroom_f_mm', res)
        self.assertGreater(res['z_static_sag_f_mm'], 0.0)
        self.assertGreater(res['bump_headroom_f_mm'], 0.0)

    def test_user_guide_documentation_presence(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        guide_path = os.path.join(base_dir, "docs", "USER_GUIDE.md")
        self.assertTrue(os.path.exists(guide_path), f"User guide missing at {guide_path}")
        with open(guide_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Dynamic Target Suite", content)
        self.assertIn("Handling Diagram", content)
        self.assertIn("Pacejka", content)
        self.assertIn("Yaw Moment Diagram", content)
        self.assertIn("DNA del Veicolo", content)

    def test_matplotlib_figure_cloning(self):
        import pickle
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 10, 20], label='Test')
        ax.set_title("Test Plot")

        # Emulate PopoutPlotDialog pickle clone
        buf = pickle.dumps(fig)
        cloned_fig = pickle.loads(buf)
        self.assertIsNotNone(cloned_fig)
        self.assertEqual(len(cloned_fig.axes), 1)
        self.assertEqual(cloned_fig.axes[0].get_title(), "Test Plot")
        plt.close(fig)
        plt.close(cloned_fig)


if __name__ == '__main__':
    unittest.main()
