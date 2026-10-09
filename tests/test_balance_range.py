"""
test_balance_range.py
---------------------
Unit tests for the Optimal Balance Range and Multi-Source Uncertainty Engine.
"""

import unittest
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_balance_range import evaluate_optimal_balance_sweep, SWEEP_PARAMETER_CONFIGS


class TestBalanceRangeSweep(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_parameter_configs_structure(self):
        for key, cfg in SWEEP_PARAMETER_CONFIGS.items():
            self.assertIn('name', cfg)
            self.assertIn('unit', cfg)
            self.assertIn('scale', cfg)
            self.assertIn('default_min', cfg)
            self.assertIn('default_max', cfg)
            self.assertIn('attr', cfg)

    def test_lltd_balance_sweep_with_uncertainty(self):
        res = evaluate_optimal_balance_sweep(
            vp=self.vp,
            param_key='lltd',
            param_min=0.45,
            param_max=0.65,
            n_points=6,
            v_eval=12.0,
            tire_unc_pct=10.0,
            wd_unc_pct=1.5,
            aero_unc_pct=2.0,
            lltd_unc_pct=2.0,
            target_kus_min_degg=0.10,
            target_kus_max_degg=0.80
        )

        self.assertIn('sweep_df', res)
        self.assertIn('optimal_range', res)
        self.assertIn('optimal_range_display', res)
        self.assertIn('robust_score_pct', res)
        
        df = res['sweep_df']
        self.assertEqual(len(df), 6)
        self.assertIn('kus_nom', df.columns)
        self.assertIn('kus_min', df.columns)
        self.assertIn('kus_max', df.columns)
        self.assertIn('ay_nom', df.columns)
        self.assertIn('stab_nom', df.columns)

        # Verification of monotonic or physical trends:
        # Increasing LLTD front increases understeer gradient (K_us)
        kus_first = df.iloc[0]['kus_nom']
        kus_last = df.iloc[-1]['kus_nom']
        self.assertGreater(kus_last, kus_first)

        # Uncertainty envelope verification: kus_min <= kus_nom <= kus_max
        for _, row in df.iterrows():
            self.assertLessEqual(row['kus_min'], row['kus_nom'] + 1e-3)
            self.assertGreaterEqual(row['kus_max'], row['kus_nom'] - 1e-3)

    def test_wd_f_balance_sweep(self):
        res = evaluate_optimal_balance_sweep(
            vp=self.vp,
            param_key='wd_f',
            param_min=0.44,
            param_max=0.56,
            n_points=5,
            v_eval=12.0,
            tire_unc_pct=5.0
        )
        self.assertEqual(len(res['sweep_df']), 5)
        self.assertGreaterEqual(res['robust_score_pct'], 0.0)


if __name__ == '__main__':
    unittest.main()
