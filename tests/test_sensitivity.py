"""
test_sensitivity.py — Unit tests for Sensitivity Analysis & FSAE 2026 Event Scoring
"""

import unittest
import numpy as np
from src.models.params_vehicle import VehicleParams
from src.mdo import mdo_sensitivity as ms


class TestSensitivityAndScoring(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_default_mass(self):
        self.assertEqual(self.vp.m, 300.0)

    def test_fs2026_disciplines_constants(self):
        self.assertIn('skidpad', ms.FS2026_DISCIPLINES)
        self.assertIn('acceleration', ms.FS2026_DISCIPLINES)
        self.assertIn('autocross', ms.FS2026_DISCIPLINES)
        self.assertIn('endurance', ms.FS2026_DISCIPLINES)
        self.assertIn('efficiency', ms.FS2026_DISCIPLINES)

    def test_fs2026_score_fastest_vehicle(self):
        # Fastest vehicle (Tteam = Tmin) must get maximum points
        s_accel = ms.compute_fs2026_score_discipline(3.35, 3.35, 'acceleration')
        self.assertAlmostEqual(s_accel, 50.0, places=2)

        s_skid = ms.compute_fs2026_score_discipline(4.75, 4.75, 'skidpad')
        self.assertAlmostEqual(s_skid, 50.0, places=2)

        s_autox = ms.compute_fs2026_score_discipline(53.5, 53.5, 'autocross')
        self.assertAlmostEqual(s_autox, 100.0, places=2)

        s_endur = ms.compute_fs2026_score_discipline(1320.0, 1320.0, 'endurance')
        self.assertAlmostEqual(s_endur, 250.0, places=2)

    def test_fs2026_score_slow_vehicle_cutoff(self):
        # Slower than Tmax (Tteam >= Tmax) must get minimum points or 0
        s_accel_slow = ms.compute_fs2026_score_discipline(3.35 * 1.70, 3.35, 'acceleration')
        self.assertAlmostEqual(s_accel_slow, 2.5, places=2) # Pmin = 0.05 * 50 = 2.5

        s_accel_dq = ms.compute_fs2026_score_discipline(3.35 * 2.0, 3.35, 'acceleration')
        self.assertEqual(s_accel_dq, 0.0)

    def test_evaluate_fsae_2026_full_event(self):
        scores = ms.evaluate_fsae_2026_full_event(
            t_accel=3.60, t_skid=4.90, t_autox=54.5, t_endur=1350.0, energy_kwh=3.0,
            benchmark_event='FS_Germany'
        )
        self.assertIn('total_dynamic_score', scores)
        self.assertGreater(scores['total_dynamic_score'], 350.0)
        self.assertLessEqual(scores['total_dynamic_score'], 525.0)

    def test_sensitivity_1d(self):
        vals = [3.0, 3.5, 4.0]
        df_1d, base_val = ms.run_sensitivity_1d('final_drive', vals, 'accel_time', base_params=self.vp)
        self.assertEqual(len(df_1d), 3)
        self.assertIn('param_value', df_1d.columns)
        self.assertIn('metric_value', df_1d.columns)
        # Higher gear ratio gives faster 75m time (smaller time)
        self.assertLess(df_1d.iloc[2]['metric_value'], df_1d.iloc[0]['metric_value'])

    def test_tornado_sensitivity(self):
        params = ['m', 'final_drive', 'wd_f']
        df_tor, base_m = ms.run_tornado_sensitivity(params, delta_pct=10.0, metric_key='total_fsae_points', base_params=self.vp)
        self.assertEqual(len(df_tor), 3)
        self.assertIn('total_swing', df_tor.columns)
        self.assertTrue(all(df_tor['total_swing'] >= 0))


if __name__ == '__main__':
    unittest.main()
