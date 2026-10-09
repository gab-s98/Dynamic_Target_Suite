import unittest
import numpy as np
import casadi as ca
from src.models import damper_model_ttx25 as dm
from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF


class TestDamperTTX25(unittest.TestCase):

    def test_valving_hierarchy(self):
        v = 0.025  # 25 mm/s
        f_c11 = dm.calculate_damper_force(v, 'C11/R11', lsc=10, hsc=2.0)
        f_c12 = dm.calculate_damper_force(v, 'C12/R12', lsc=10, hsc=2.0)
        f_c13 = dm.calculate_damper_force(v, 'C13/R13', lsc=10, hsc=2.0)
        f_c23 = dm.calculate_damper_force(v, 'C23/R23', lsc=10, hsc=2.0)

        self.assertLess(f_c11, f_c12, "C11 must be softer than C12")
        self.assertLess(f_c12, f_c13, "C12 must be softer than C13")
        self.assertLess(f_c13, f_c23, "C13 must be softer than C23")

    def test_asymmetry_compression_rebound(self):
        v_mag = 0.025
        f_comp = dm.calculate_damper_force(v_mag, 'C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0)
        f_reb  = dm.calculate_damper_force(-v_mag, 'C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0)

        self.assertGreater(f_comp, 0.0, "Compression force must be positive")
        self.assertLess(f_reb, 0.0, "Rebound force must be negative")

    def test_lsc_monotonicity(self):
        v = 0.015  # 15 mm/s low speed
        f_closed = dm.calculate_damper_force(v, 'C12/R12', lsc=0)   # full stiff
        f_mid    = dm.calculate_damper_force(v, 'C12/R12', lsc=10)
        f_open   = dm.calculate_damper_force(v, 'C12/R12', lsc=25)  # full soft

        self.assertGreater(f_closed, f_mid, "LSC 0 clicks must produce higher force than 10 clicks")
        self.assertGreater(f_mid, f_open, "LSC 10 clicks must produce higher force than 25 clicks")

    def test_hsc_monotonicity(self):
        v = 0.200  # 200 mm/s high speed
        f_stiff = dm.calculate_damper_force(v, 'C12/R12', lsc=10, hsc=4.3) # 4.3 turns closed
        f_soft  = dm.calculate_damper_force(v, 'C12/R12', lsc=10, hsc=0.0) # 0.0 turns open

        self.assertGreater(f_stiff, f_soft, "HSC 4.3 turns closed must produce higher force than 0.0 turns open")

    def test_motion_ratio_scaling(self):
        v_wheel = 0.030
        mr = 0.85
        f_wheel = dm.calculate_wheel_damper_force(v_wheel, motion_ratio=mr, valving='C12/R12', lsc=10, hsc=2.0)
        f_damper = dm.calculate_damper_force(v_wheel * mr, valving='C12/R12', lsc=10, hsc=2.0)

        self.assertAlmostEqual(f_wheel, mr * f_damper, places=5)

    def test_casadi_differentiability(self):
        v_sym = ca.SX.sym('v')
        f_ca = dm.calculate_damper_force(v_sym, 'C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, use_casadi=True)
        df_dv = ca.jacobian(f_ca, v_sym)
        f_eval = ca.Function('f_eval', [v_sym], [f_ca, df_dv])

        for v_test in [-0.20, -0.025, 0.0, 0.025, 0.20]:
            f_val, df_val = f_eval(v_test)
            f_num = float(f_val)
            df_num = float(df_val)
            self.assertFalse(np.isnan(f_num), f"NaN force at v={v_test}")
            self.assertFalse(np.isnan(df_num), f"NaN derivative at v={v_test}")
            self.assertGreater(df_num, 0.0, f"Damping slope df/dv must be positive at v={v_test}")

    def test_vehicle_14dof_dual_mode(self):
        # 1. Test Linear Mode (Default fallback)
        vp_linear = VehicleParams()
        vp_linear.damper_mode = 'linear'
        v14_linear = Vehicle14DOF(vp=vp_linear)
        ode_linear = v14_linear.get_casadi_ode()
        self.assertIsNotNone(ode_linear, "14-DOF ODE must compile in linear damper mode")

        # 2. Test TTX25 Non-Linear Mode
        vp_ttx = VehicleParams()
        vp_ttx.damper_mode = 'ttx25_nonlinear'
        vp_ttx.damper_valving_f = 'C12/R12'
        vp_ttx.damper_valving_r = 'C12/R12'
        vp_ttx.damper_lsc_clicks_f = 8.0
        vp_ttx.damper_hsc_turns_f = 2.5
        v14_ttx = Vehicle14DOF(vp=vp_ttx)
        ode_ttx = v14_ttx.get_casadi_ode()
        self.assertIsNotNone(ode_ttx, "14-DOF ODE must compile in TTX25 non-linear damper mode")


if __name__ == '__main__':
    unittest.main()

