"""
tests/test_acceleration_and_diamond.py
---------------------------------------
Unit tests covering:
1. Dual-motor acceleration physics and realistic FSAE sprint times (~3.9s for 75m).
2. ARB diamond creation from digitized curves 05 (Ideal) and 06 (Real with chassis compliance).
3. Stability metrics handling of aperiodic unstable yaw regimes (det(A) <= 0 -> fn is None).
4. Sensitivity 1D regex import validation.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_acceleration import simulate_acceleration
from src.mdo.mdo_arb_diamond_balance import ARBDiamondConfig
from src.mdo.mdo_global_optimizer import compute_stability_metrics


class TestAccelerationAndDiamond(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams.nominal_fsae_params()

    def test_dual_motor_acceleration_performance(self):
        """
        Verify that electric prototype dual-motor RWD setup achieves realistic acceleration:
        - 75m sprint time under 4.30 seconds.
        - Peak longitudinal acceleration a_x,max >= 1.00 g.
        """
        res = simulate_acceleration(self.vp, distance=75.0, is_awd=False)
        self.assertTrue(res['success'])
        
        t_75m = res['time']
        a_max_g = res['a_max_g']
        v_final_kmh = res['v_max'] * 3.6
        
        # Check realistic race bounds
        self.assertLess(t_75m, 4.30, f"75m time {t_75m:.3f}s is too slow for dual motor FSAE car!")
        self.assertGreater(t_75m, 3.40, f"75m time {t_75m:.3f}s is unrealistically fast for RWD!")
        self.assertGreater(a_max_g, 1.00, f"Peak ax {a_max_g:.2f}g should exceed 1.00g!")
        self.assertGreater(v_final_kmh, 95.0, f"Terminal speed {v_final_kmh:.1f} km/h should exceed 95 km/h!")

    def test_arb_digitized_curve_05_ideal(self):
        """
        Verify ARB Diamond config generated from digitized curve 05 (Ideal / Rigid chassis).
        """
        cfg = ARBDiamondConfig.from_digitized_curve_05_ideal()
        self.assertIn("Ideale", cfg.name)
        k_min, k_max, r_min, r_max = cfg.get_bounds()
        
        # Roll stiffness range should be within FSAE bounds (~15000 to ~98000 Nm/rad)
        self.assertGreater(k_min, 15000.0)
        self.assertLess(k_max, 100000.0)
        # Front distribution ratio should be around 0.30 to 0.75
        self.assertGreater(r_min, 0.30)
        self.assertLess(r_max, 0.75)
        
        # Center point should be inside
        c_k, c_r = cfg.get_center_point()
        self.assertTrue(cfg.is_inside(c_k, c_r))

    def test_arb_digitized_curve_06_real(self):
        """
        Verify ARB Diamond config generated from digitized curve 06 (Real / Torsional compliance included).
        """
        cfg = ARBDiamondConfig.from_digitized_curve_06_real()
        self.assertIn("Reale", cfg.name)
        k_min, k_max, r_min, r_max = cfg.get_bounds()
        
        # Roll stiffness with chassis compliance is slightly lower than ideal
        self.assertGreater(k_min, 12000.0)
        self.assertLess(k_max, 100000.0)
        self.assertGreater(r_min, 0.30)
        self.assertLess(r_max, 0.75)
        
        # Center point should be inside
        c_k, c_r = cfg.get_center_point()
        self.assertTrue(cfg.is_inside(c_k, c_r))

    def test_aperiodic_unstable_stability_metrics(self):
        """
        Verify that compute_stability_metrics properly handles unstable oversteering regimes
        where det(A) <= 0 (v_x > v_crit). It must NOT return a fake 0.00 Hz oscillatory frequency.
        """
        # Oversteering vehicle with rearward weight distribution at high speed (v_x > v_crit)
        metrics = compute_stability_metrics(
            L=1.529, wd_f=0.25, lltd=0.30, aero_bal=0.30, v_x=40.0
        )
        
        self.assertIsNone(metrics['f_n_hz'], "Oscillatory natural frequency must be None when det(A) <= 0!")
        self.assertIsNone(metrics['omega_n_rads'])
        self.assertIsNone(metrics['zeta'])
        self.assertIn("Aperiodico Instabile", metrics['mode_desc'])

    def test_stable_oscillatory_stability_metrics(self):
        """
        Verify that compute_stability_metrics correctly returns natural frequency and damping
        for a well-behaved understeering vehicle state matrix.
        """
        # Nominal understeering vehicle at moderate speed
        metrics = compute_stability_metrics(
            L=1.529, wd_f=0.55, lltd=0.60, aero_bal=0.52, v_x=15.0
        )
        self.assertIsNotNone(metrics['f_n_hz'])
        self.assertGreater(metrics['f_n_hz'], 0.5)
        self.assertIsNotNone(metrics['zeta'])
        self.assertGreater(metrics['zeta'], 0.0)
        
    def test_sensitivity_1d_execution(self):
        """
        Verify that 1D sensitivity calculation runs smoothly and produces valid metric outputs.
        """
        from src.mdo import mdo_sensitivity as ms
        vals = np.linspace(1.50, 1.55, 3)
        df_res, base_val = ms.run_sensitivity_1d('L', vals, 'skidpad_time', base_params=self.vp)
        self.assertEqual(len(df_res), 3)
        self.assertIn('param_value', df_res.columns)
        self.assertIn('metric_value', df_res.columns)
        self.assertGreater(base_val, 0.0)


if __name__ == '__main__':
    unittest.main()
