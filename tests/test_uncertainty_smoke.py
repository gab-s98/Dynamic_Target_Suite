import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdo.mdo_uncertainty import run_handling_sweep, get_uncertainty_envelope


class TestUncertaintySmoke(unittest.TestCase):
    """Fast smoke tests for the Uncertainty Propagation module."""

    def test_handling_sweep_basic(self):
        """Verify handling sweep returns valid DataFrame."""
        df = run_handling_sweep(v_x=12.0, wd_f=0.5, lmuy_mult=1.0,
                                delta_range_deg=(0.5, 4.0), num_points=5)
        self.assertGreater(len(df), 0)
        self.assertIn('yaw_rate', df.columns)
        self.assertIn('ay_g', df.columns)
        self.assertIn('delta_deg', df.columns)
        # Yaw rate should be monotonically increasing with steering
        if len(df) > 2:
            self.assertTrue(np.all(np.diff(df['yaw_rate'].values) >= -1e-6))

    def test_handling_sweep_grip_effect(self):
        """Verify that reduced grip gives lower lateral acceleration."""
        df_nom = run_handling_sweep(v_x=12.0, delta_range_deg=(0.5, 3.0), num_points=5)
        df_low = run_handling_sweep(v_x=12.0, lmuy_mult=0.85,
                                    delta_range_deg=(0.5, 3.0), num_points=5)
        if len(df_nom) > 1 and len(df_low) > 1:
            ay_nom_max = df_nom['ay_g'].max()
            ay_low_max = df_low['ay_g'].max()
            self.assertGreater(ay_nom_max, ay_low_max * 0.95,
                             "Nominal grip should produce higher ay than reduced grip")

    def test_uncertainty_envelope(self):
        """Verify uncertainty envelope returns three DataFrames."""
        df_nom, df_low, df_high = get_uncertainty_envelope(
            v_x=12.0, wd_f=0.5, uncertainty_pct=15.0,
            delta_range_deg=(0.5, 3.0))
        self.assertGreater(len(df_nom), 0)
        self.assertGreater(len(df_low), 0)
        self.assertGreater(len(df_high), 0)


if __name__ == '__main__':
    unittest.main()
