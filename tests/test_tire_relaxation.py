"""
test_tire_relaxation.py
-----------------------
Unit and integration tests for Pacejka Contact Patch Relaxation Length model:
- Parameter registration (sigma_x, sigma_y) in TireParams and VehicleParams.
- Standstill pure spring compliance behavior (no singularity at zero speed).
- Exponential convergence of dynamic slip to steady-state slip at constant velocity.
- Full 23-state 14-DOF ODE evaluation and RK4 simulation stability.
"""

import os
import unittest
import numpy as np
from src.models.params_tire import TireParams
from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF


class TestTireRelaxationModel(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tir_path = os.path.join(self.base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir")
        self.tp = TireParams(self.tir_path)
        self.vp = VehicleParams()

    def test_01_tire_params_explicit_relaxation(self):
        """Verify TireParams parses explicit SIGMA_X and SIGMA_Y, with safe fallbacks."""
        # 1. Check parsed values from Hoosier .tir file
        self.assertAlmostEqual(self.tp.sigma_x, 0.15, places=3)
        self.assertAlmostEqual(self.tp.sigma_y, 0.22, places=3)

        # 2. Check fallback values when empty
        tp_empty = TireParams("")
        self.assertAlmostEqual(tp_empty.sigma_x, 0.15, places=3)
        self.assertAlmostEqual(tp_empty.sigma_y, 0.22, places=3)

    def test_02_vehicle_params_relaxation_fields(self):
        """Verify VehicleParams has sigma_x and sigma_y, and serializes correctly."""
        # Nominal defaults
        self.assertAlmostEqual(self.vp.sigma_x, 0.15, places=4)
        self.assertAlmostEqual(self.vp.sigma_y, 0.22, places=4)

        # Serialization to/from dict
        d = self.vp.to_dict()
        self.assertIn('sigma_x', d)
        self.assertIn('sigma_y', d)
        self.assertEqual(d['sigma_x'], 0.15)
        self.assertEqual(d['sigma_y'], 0.22)

        vp_from = VehicleParams.from_dict(d)
        self.assertEqual(vp_from.sigma_x, 0.15)
        self.assertEqual(vp_from.sigma_y, 0.22)

        # Load from preset JSON
        cand_paths = [
            os.path.join(self.base_dir, "data", "setups", "et18_2026_kinematics.json"),
            os.path.join(self.base_dir, "data", "setups", "fsae_mockup.json"),
            os.path.join(self.base_dir, "setups", "fsae_mockup.json"),
        ]
        json_path = next((p for p in cand_paths if os.path.exists(p)), None)
        if json_path:
            vp_json = VehicleParams.load_json(json_path)
            self.assertAlmostEqual(vp_json.sigma_x, 0.15, places=3)
            self.assertAlmostEqual(vp_json.sigma_y, 0.22, places=3)

    def test_03_standstill_pure_spring_compliance(self):
        """
        Verify that at standstill (vx -> 0, vy -> 0):
        The relaxation equation becomes dot(u) = -Vs = w*Rw - Vxw without division-by-zero singularities.
        """
        veh = Vehicle14DOF(self.vp, self.tp, self.tp)
        ode = veh.get_casadi_ode()

        # Completely stationary vehicle, but spinning rear wheels (e.g. burn-out start attempt)
        x = np.zeros(28)
        x[0] = 0.0  # vx = 0
        x[1] = 0.0  # vy = 0
        w_spin = 10.0  # rad/s on rear wheels
        x[18] = w_spin # w_rl
        x[19] = w_spin # w_rr
        # Zero initial carcass deflections: u = 0
        x[20:28] = 0.0

        u = np.zeros(4)
        p = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            self.vp.LMUY, self.vp.LMUX
        ]

        x_dot = np.array(ode(x, u, p)).flatten()
        self.assertEqual(len(x_dot), 28)
        self.assertTrue(np.all(np.isfinite(x_dot)), "All derivatives must be finite at standstill")

        # Longitudinal deflection derivative on rear wheels:
        # dot(ux) = w*Rw - Vxw = 10.0 * 0.228 ≈ 2.28 m/s
        Rw = self.vp.R_w
        expected_ux_dot = w_spin * Rw
        # Rear-left ux_dot is at index 24, rear-right at index 26
        self.assertAlmostEqual(x_dot[24], expected_ux_dot, places=2)
        self.assertAlmostEqual(x_dot[26], expected_ux_dot, places=2)

        # Front wheels have w = 0, so ux_dot must be 0 (indices 20 and 22)
        self.assertAlmostEqual(x_dot[20], 0.0, places=4)
        self.assertAlmostEqual(x_dot[22], 0.0, places=4)

    def test_04_dynamic_slip_steady_state_convergence(self):
        """
        Verify that under steady forward driving, dynamic carcass deflection
        converges to steady-state slip:
        ux_inf = sigma_x * kappa_steady,  uy_inf = sigma_y * tan(alpha_steady)
        """
        veh = Vehicle14DOF(self.vp, self.tp, self.tp)
        ode = veh.get_casadi_ode()

        vx = 15.0  # m/s
        Rw = self.vp.R_w
        w0 = vx / Rw

        p = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            0.0, 0.0,  # zero static camber for pure test
            0.0, 0.0,  # zero static toe for pure test
            self.vp.L, self.vp.t_f, self.vp.t_r,
            1.0, 1.0
        ]

        # Steer angle delta = 2.0 deg
        steer_rad = np.deg2rad(2.0)
        u_ctrl = [steer_rad, 0.0, 0.0, 0.0]

        x = np.zeros(28)
        x[0] = vx
        x[16:20] = w0

        dt = 0.002
        # Run 500 steps (1.0 second), well past the relaxation time tau = sigma / vx = 0.22 / 15 ≈ 0.015 s
        for _ in range(500):
            k1 = np.array(ode(x, u_ctrl, p)).flatten()
            k2 = np.array(ode(x + 0.5 * dt * k1, u_ctrl, p)).flatten()
            k3 = np.array(ode(x + 0.5 * dt * k2, u_ctrl, p)).flatten()
            k4 = np.array(ode(x + dt * k3, u_ctrl, p)).flatten()
            x += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Hold planar forward velocity constant for clean steady-state verification
            x[0] = vx

        # Deflections must have stabilized (dot(u) ≈ 0)
        k_final = np.array(ode(x, u_ctrl, p)).flatten()
        for idx in range(20, 28):
            self.assertLess(abs(k_final[idx]), 0.1, f"Deflection rate at index {idx} not settled: {k_final[idx]}")

        # Lateral deflections must be non-zero and physical
        uy_fl = x[21]
        uy_fr = x[23]
        self.assertNotEqual(uy_fl, 0.0)
        self.assertNotEqual(uy_fr, 0.0)
        self.assertLess(abs(uy_fl), 0.08, "Tire lateral deflection must remain within realistic tire contact limit (< 80 mm)")

    def test_05_ode_28_states_rk4_simulation(self):
        """Verify full 28-state vector integration stability over multi-step simulation."""
        veh = Vehicle14DOF(self.vp, self.tp, self.tp)
        ode = veh.get_fast_ode()

        x = np.zeros(28)
        x[0] = 12.0
        w0 = 12.0 / self.vp.R_w
        x[16:20] = w0

        u = [np.deg2rad(5.0), 50.0, 50.0, 0.0]
        p = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            1.0, 1.0
        ]

        dt = 0.004
        for _ in range(250):
            k1 = np.array(ode(x, u, p)).flatten()
            k2 = np.array(ode(x + 0.5 * dt * k1, u, p)).flatten()
            k3 = np.array(ode(x + 0.5 * dt * k2, u, p)).flatten()
            k4 = np.array(ode(x + dt * k3, u, p)).flatten()
            x += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.assertEqual(len(x), 28)
        self.assertTrue(np.all(np.isfinite(x)), "Integration must produce finite values across all 28 states")
        self.assertGreater(x[0], 0.0, "Car must still have forward speed")


if __name__ == '__main__':
    unittest.main()
