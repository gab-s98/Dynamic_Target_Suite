"""
test_transient_sensitivity.py
-----------------------------
Unit tests for the parametric transient sensitivity module and KPI computation.
"""

import unittest
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.utils import transient_sensitivity as ts


class TestTransientSensitivity(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_scenario_catalogue(self):
        self.assertIn('rc_height_f', ts.SCENARIO_CATALOG)
        self.assertIn('rc_lat_gain_f', ts.SCENARIO_CATALOG)
        self.assertIn('arb_distribution', ts.SCENARIO_CATALOG)
        self.assertIn('h_cg', ts.SCENARIO_CATALOG)

    def test_apply_param(self):
        vp_mod = ts._apply_param(self.vp, 'rc_height_f', 0.08)
        self.assertAlmostEqual(vp_mod.q1, 0.08)
        self.assertAlmostEqual(vp_mod.rollcenter_f_coeffs[3], 0.08)

        vp_mod2 = ts._apply_param(self.vp, 'arb_distribution', 0.60)
        self.assertAlmostEqual(vp_mod2.lltd, 0.60)
        K_tot = self.vp.K_phi_f + self.vp.K_phi_r
        self.assertAlmostEqual(vp_mod2.K_phi_f, 0.60 * K_tot)

    def test_1d_sweep_step_steer(self):
        sweep_vals = [0.03, 0.07]
        res = ts.run_sensitivity_sweep_1d(
            self.vp, 'rc_height_f', sweep_vals,
            maneuver='step_steer', model_type='14-DOF'
        )
        self.assertEqual(len(res['telemetries']), 2)
        self.assertIn('kpi_table', res)
        df_kpi = res['kpi_table']
        self.assertEqual(len(df_kpi), 2)
        self.assertIn('lltd_peak', df_kpi.columns)
        self.assertIn('phi_max_deg', df_kpi.columns)
        self.assertIn('r_ss_deg_s', df_kpi.columns)

        # Higher front roll center should yield higher front steady-state LLTD and lower roll angle peak
        lltd_low = df_kpi['lltd_ss'].iloc[0]
        lltd_high = df_kpi['lltd_ss'].iloc[1]
        self.assertGreater(lltd_high, lltd_low)
        phi_low = df_kpi['phi_max_deg'].iloc[0]
        phi_high = df_kpi['phi_max_deg'].iloc[1]
        self.assertLess(phi_high, phi_low)

    def test_2d_sweep_heatmap(self):
        x_vals = [0.03, 0.08]
        y_vals = [0.04, 0.09]
        res = ts.run_sensitivity_sweep_2d(
            self.vp, 'rc_height_f', x_vals, 'rc_height_r', y_vals,
            kpi_key='lltd_peak', maneuver='step_steer', model_type='14-DOF'
        )
        self.assertEqual(res['Z'].shape, (2, 2))
        self.assertFalse(np.isnan(res['Z']).any())


if __name__ == '__main__':
    unittest.main()
