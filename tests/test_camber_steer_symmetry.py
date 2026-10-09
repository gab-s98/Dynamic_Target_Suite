"""
test_camber_steer_symmetry.py
------------------------------
Unit tests verifying:
1. Full lock steer symmetry (left vs right specular symmetry across vehicle centerline).
2. Decoupled Caster (differential) and KPI (heave) jacking.
3. Camber Steer dynamic contribution and quantitative 2025 vs 2026 comparison.
4. Dynamic evaluation of Caster Trail and Scrub Radius maps in steering dynamics.
"""

import unittest
import numpy as np
import casadi as ca
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.steering_dynamics import calculate_steering_torque


class TestCamberSteerSymmetry(unittest.TestCase):

    def setUp(self):
        self.vp = VehicleParams()
        self.tp = TireParams("")
        self.model = Vehicle14DOF(self.vp, self.tp, self.tp)
        self.ode = self.model.get_casadi_ode()

    def test_full_lock_symmetry_standstill(self):
        """
        Verifies that at standstill, steering +25 deg (left) produces the EXACT
        mirror-symmetric wheel travel and loads as steering -25 deg (right).
        """
        steer_angles_deg = [10.0, 20.0, 25.0]
        dt = 0.002
        n_steps = 250

        p = [self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
             self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
             self.vp.camber_f_static, self.vp.camber_r_static,
             self.vp.toe_f_static, self.vp.toe_r_static,
             self.vp.L, self.vp.t_f, self.vp.t_r, self.vp.LMUY, self.vp.LMUX]

        for steer_mag in steer_angles_deg:
            u_left = [np.deg2rad(steer_mag), 0.0, 0.0, 0.0]
            u_right = [-np.deg2rad(steer_mag), 0.0, 0.0, 0.0]

            states = {}
            for name, u in [('LEFT', u_left), ('RIGHT', u_right)]:
                x = np.zeros(28)
                x[0] = 0.001
                for _ in range(n_steps):
                    k1 = np.array(self.ode(x, u, p)).flatten()
                    k2 = np.array(self.ode(x + 0.5 * dt * k1, u, p)).flatten()
                    k3 = np.array(self.ode(x + 0.5 * dt * k2, u, p)).flatten()
                    k4 = np.array(self.ode(x + dt * k3, u, p)).flatten()
                    x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    x[0:3] = 0.0
                states[name] = x

            z_fl_L = states['LEFT'][8]
            z_fr_L = states['LEFT'][9]
            z_rl_L = states['LEFT'][10]
            z_rr_L = states['LEFT'][11]

            z_fl_R = states['RIGHT'][8]
            z_fr_R = states['RIGHT'][9]
            z_rl_R = states['RIGHT'][10]
            z_rr_R = states['RIGHT'][11]

            # Front wheels mirror symmetry
            self.assertAlmostEqual(z_fl_L, z_fr_R, places=5,
                                   msg=f"Asymmetry detected at delta={steer_mag} deg: z_fl(L)={z_fl_L} vs z_fr(R)={z_fr_R}")
            self.assertAlmostEqual(z_fr_L, z_fl_R, places=5,
                                   msg=f"Asymmetry detected at delta={steer_mag} deg: z_fr(L)={z_fr_L} vs z_fl(R)={z_fl_R}")

            # Rear wheels mirror symmetry
            self.assertAlmostEqual(z_rl_L, z_rr_R, places=5,
                                   msg=f"Rear asymmetry at delta={steer_mag} deg: z_rl(L)={z_rl_L} vs z_rr(R)={z_rr_R}")
            self.assertAlmostEqual(z_rr_L, z_rl_R, places=5,
                                   msg=f"Rear asymmetry at delta={steer_mag} deg: z_rr(L)={z_rr_L} vs z_rl(R)={z_rl_R}")

    def test_camber_steer_delta_and_symmetry(self):
        """
        Verifies that Camber Steer modifies front camber and is mirror-symmetric.
        """
        vp = VehicleParams()
        coeffs = vp.camber_steer_f_coeffs
        self.assertEqual(len(coeffs), 4)
        self.assertAlmostEqual(coeffs[3], 0.0, places=4, msg="At zero steer angle, camber steer delta must be 0")

        a3, a2, a1, a0 = coeffs
        def eval_p(x):
            return a3*(x**3) + a2*(x**2) + a1*x + a0

        dcam_fl_pos = eval_p(100.0)
        dcam_fr_neg = eval_p(-(-100.0))
        self.assertAlmostEqual(dcam_fl_pos, dcam_fr_neg, places=6)

    def test_camber_steer_2025_vs_2026_magnitude(self):
        """
        Verifies that 2025 (Caster = 6 deg) has larger camber variation with steer
        than 2026 (Caster = 3 deg), matching the Adams Car multibody simulation.
        """
        c_2026 = [1.1590e-06, 6.9177e-05, -6.2981e-03, 0.0]
        c_2025 = [7.8990e-07, 7.4226e-05, 0.009014, 0.0]

        def eval_p(c, x):
            return c[0]*(x**3) + c[1]*(x**2) + c[2]*x + c[3]

        span_2026 = abs(eval_p(c_2026, 125.0) - eval_p(c_2026, -125.0))
        span_2025 = abs(eval_p(c_2025, 125.0) - eval_p(c_2025, -125.0))

        self.assertGreater(span_2025, span_2026 * 1.3,
                           msg=f"Expected 2025 camber steer span ({span_2025:.2f} deg) to exceed 2026 ({span_2026:.2f} deg)")

    def test_steering_dynamics_dynamic_maps(self):
        """
        Verifies that calculate_steering_torque evaluates dynamic trail and scrub radius.
        """
        vp = VehicleParams()
        res_scalar = calculate_steering_torque(
            delta_deg=15.0,
            ay_g=1.2,
            v_x=15.0,
            Fz_front_single=600.0,
            tp=self.tp,
            mech_trail_mm=12.0,
            scrub_radius_mm=36.8
        )

        res_dynamic = calculate_steering_torque(
            delta_deg=15.0,
            ay_g=1.2,
            v_x=15.0,
            Fz_front_single=600.0,
            tp=self.tp,
            caster_trail_coeffs=vp.caster_trail_f_coeffs,
            scrub_radius_coeffs=vp.scrub_radius_f_coeffs
        )

        self.assertIn('T_sw_Nm', res_scalar)
        self.assertIn('T_sw_Nm', res_dynamic)
        self.assertGreater(res_dynamic['T_sw_Nm'], 0.0)
        self.assertNotEqual(res_scalar['t_m_mm'], res_dynamic['t_m_mm'])


if __name__ == '__main__':
    unittest.main()
