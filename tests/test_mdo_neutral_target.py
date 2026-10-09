"""
tests/test_mdo_neutral_target.py
--------------------------------
Comprehensive unit tests for:
1. Pure cornering stiffness and clean Understeer Gradient (calc_understeer_gradient)
2. 14-DOF roll stiffness (K_phi_f, K_phi_r) coupling to dynamic load transfer and Kus
3. Physical peak detection in limit Ay evaluation (_evaluate_limit_ay_at_speed)
4. MDO Target Neutral Handling objective mode across 3-DOF and 14-DOF
5. Smoothly calibrated understeer penalty without boundary pinning
"""

import unittest
import os
import numpy as np
from src.mdo.mdo_global_optimizer import GlobalOptimizer, compute_stability_metrics
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams


class TestMDONeutralTarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.tir_path = os.path.join(base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir")
        cls.vp = VehicleParams()

    def test_clean_understeer_gradient_nominal(self):
        """Verify that calc_understeer_gradient is free from camber thrust pollution and evaluates near neutral."""
        opt3 = GlobalOptimizer(v_x=15.0, tir_file=self.tir_path, model_type="3-DOF")
        x0_3 = [1.529, 1.252, 1.136, 0.45, 0.50, 0.485]
        kus_3 = opt3.calc_understeer_gradient(x0_3)

        # Baseline physical Kus of nominal FSAE vehicle is nearly neutral (~ -0.30 to +0.15 deg/g)
        self.assertGreater(kus_3, -0.40, f"Kus too negative, likely camber thrust corruption: {kus_3:.4f}")
        self.assertLess(kus_3, 0.25, f"Kus unexpectedly large: {kus_3:.4f}")

    def test_14dof_roll_stiffness_coupling_in_kus(self):
        """Verify that 14-DOF roll stiffness distribution (K_phi_f, K_phi_r) directly modulates Kus."""
        opt14 = GlobalOptimizer(v_x=15.0, tir_file=self.tir_path, model_type="14-DOF")
        
        # Nominal: K_phi_f = 8000, K_phi_r = 6000 (front biased roll stiffness)
        x_base = [1.529, 1.252, 1.136, 0.45, 0.50, 0.485, -1.5, -1.0, 0.0, 0.1, 8000.0, 6000.0, 18000.0, 16000.0]
        kus_base = opt14.calc_understeer_gradient(x_base)

        # Stiff front ARB: K_phi_f = 14000, K_phi_r = 3000 -> more front load transfer -> more understeer
        x_stiff_f = list(x_base)
        x_stiff_f[10] = 14000.0
        x_stiff_f[11] = 3000.0
        kus_stiff_f = opt14.calc_understeer_gradient(x_stiff_f)

        # Soft front, stiff rear: K_phi_f = 4000, K_phi_r = 12000 -> more rear load transfer -> more oversteer
        x_stiff_r = list(x_base)
        x_stiff_r[10] = 4000.0
        x_stiff_r[11] = 12000.0
        kus_stiff_r = opt14.calc_understeer_gradient(x_stiff_r)

        self.assertGreater(kus_stiff_f, kus_base, "Stiffer front roll stiffness must increase understeer gradient")
        self.assertLess(kus_stiff_r, kus_base, "Stiffer rear roll stiffness must decrease understeer gradient")

    def test_limit_ay_peak_stop_condition(self):
        """Verify that _evaluate_limit_ay_at_speed stops at steering peak before countersteering."""
        opt3 = GlobalOptimizer(v_x=15.0, tir_file=self.tir_path, model_type="3-DOF")
        x0_3 = [1.529, 1.252, 1.136, 0.45, 0.50, 0.485]
        
        limit_ay = opt3._evaluate_limit_ay(x0_3)
        # Limit Ay must be within physically achievable cornering limit (typically 1.4g to 1.8g)
        self.assertGreater(limit_ay, 1.30)
        self.assertLess(limit_ay, 1.85)

    def test_target_neutral_optimization_3dof(self):
        """Verify that target_neutral objective guides a 3-DOF vehicle to neutral handling (Kus ~ 0.0 deg/g)."""
        opt_neutral = GlobalOptimizer(
            v_x=15.0, tir_file=self.tir_path, model_type="3-DOF",
            objective_mode="target_neutral", target_kus_deg_g=0.0,
            understeer_weight=1.0
        )
        # Starting from an oversteering guess
        x_over = [1.529, 1.252, 1.136, 0.38, 0.40, 0.42]
        bounds = [(1.525, 1.800), (1.000, 1.500), (1.000, 1.500), (0.35, 0.65), (0.35, 0.65), (0.35, 0.65)]
        
        res = opt_neutral.run_optimization(initial_guess=x_over, custom_bounds=bounds, n_starts=1)
        self.assertTrue(res.success or res.fun < 5.0)

        kus_opt = opt_neutral.calc_understeer_gradient(res.x)
        limit_ay_opt = opt_neutral._evaluate_limit_ay(res.x)

        # The optimized car should be very close to neutral (|Kus| < 0.08 deg/g) with high lateral grip
        self.assertAlmostEqual(kus_opt, 0.0, delta=0.08, msg=f"Kus failed to reach neutral target: {kus_opt:.4f}")
        self.assertGreater(limit_ay_opt, 1.45, f"Limit Ay degraded excessively: {limit_ay_opt:.3f}g")

    def test_target_neutral_init_args(self):
        """Verify that target_kus_deg_g is properly retained in GlobalOptimizer attributes."""
        opt = GlobalOptimizer(target_kus_deg_g=0.05, objective_mode="target_neutral")
        self.assertEqual(opt.target_kus_deg_g, 0.05)
        self.assertEqual(opt.objective_mode, "target_neutral")


if __name__ == '__main__':
    unittest.main()
