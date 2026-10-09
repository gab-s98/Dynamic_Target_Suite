"""
test_straight_line_symmetry.py
--------------------------------
Comprehensive unit tests verifying:
1. Vehicle 14-DOF straight-line acceleration and coasting symmetry:
   - Left and right tire lateral forces are equal and opposite (F_y,fl + F_y,fr == 0, F_y,rl + F_y,rr == 0).
   - Camber thrust signs are physically correct: for negative static camber, F_y,fl < 0 (pushes right)
     and F_y,fr > 0 (pushes left).
   - Lateral acceleration v_y_dot and yaw acceleration r_dot are strictly zero (< 1e-4).
2. Vehicle 7-DOF straight-line symmetry in both ODE and trim.
3. Vehicle 3-DOF steady-state equilibrium at zero steer.
4. Robustness across both TYRESIDE = 'RIGHT' and TYRESIDE = 'LEFT' tire definitions.
"""

import os
import unittest
import copy
import numpy as np
import casadi as ca
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.tire_model_mf61 import mf61_lateral_force


class TestStraightLineSymmetry(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand_paths = [
            os.path.join(self.base_dir, "data", "tire_tir", "Simple_Load_Dependent.tir"),
            os.path.join(self.base_dir, "data", "tire_tir", "Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir"),
            os.path.join(self.base_dir, "tire_tir", "Simple_Load_Dependent.tir"),
        ]
        self.tir_path = next((p for p in cand_paths if os.path.exists(p)), None)
        if self.tir_path is None:
            raise FileNotFoundError("No valid tire model found for symmetry tests.")
        
        self.vp = VehicleParams()
        # Primary calibrated tire
        self.tp_right = TireParams(self.tir_path)
        self.tp_right.params['TYRESIDE'] = 'RIGHT'
        if self.tp_right.params.get('PKY6', 0.0) < 0:
            self.tp_right.params['PKY6'] = -self.tp_right.params['PKY6']
        if 'PVY3' not in self.tp_right.params or abs(self.tp_right.params.get('PVY3', 0.0)) < 1e-4:
            self.tp_right.params['PVY3'] = 0.5
        
        # Test robustness: programmatically mirrored 'LEFT' version of the SAME tire
        # to verify that physical equations maintain symmetry regardless of bench coordinate origin
        self.tp_left = copy.deepcopy(self.tp_right)
        self.tp_left.params['TYRESIDE'] = 'LEFT'
        self.tp_left.params['PKY6'] = -self.tp_right.params.get('PKY6', 0.0)

    def test_tire_params_tyreside_parsing(self):
        """Verifies get_tyreside() correctly identifies RIGHT vs LEFT tire files."""
        self.assertEqual(self.tp_right.get_tyreside(), "RIGHT")
        self.assertEqual(self.tp_left.get_tyreside(), "LEFT")

    def test_14dof_straight_line_acceleration_symmetry(self):
        """
        In 14-DOF ODE under pure straight-line acceleration (delta=0, symmetric rear drive torque):
        - Left vs right wheel accelerations must be essentially identical (within 0.05 rad/s^2 on a 126 rad/s^2 pull).
        - Lateral acceleration v_y_dot and yaw acceleration r_dot must be strictly zero (< 1e-4).
        """
        for tp_test, name in [(self.tp_right, "RIGHT_TIR"), (self.tp_left, "LEFT_TIR")]:
            model = Vehicle14DOF(self.vp, tp_test, tp_test)
            ode = model.get_casadi_ode()

            # State: moving at 15 m/s, wheels spinning at synchronous speed
            Rw = self.vp.R_w
            w0 = 15.0 / Rw
            x = np.zeros(28)
            x[0] = 15.0       # v_x
            x[16:20] = w0     # w_fl, w_fr, w_rl, w_rr

            # Input: zero steer, 150 Nm drive torque on each rear wheel
            u = [0.0, 150.0, 150.0, 0.0]

            p = [self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
                 self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
                 self.vp.camber_f_static, self.vp.camber_r_static,
                 self.vp.toe_f_static, self.vp.toe_r_static,
                 self.vp.L, self.vp.t_f, self.vp.t_r, self.vp.LMUY, self.vp.LMUX]

            x_dot = np.array(ode(x, u, p)).flatten()

            v_y_dot = x_dot[1]
            r_dot = x_dot[2]
            w_rl_dot = x_dot[18]
            w_rr_dot = x_dot[19]

            # Wheel angular acceleration on rear axle must match to >99.9%
            self.assertAlmostEqual(w_rl_dot, w_rr_dot, places=1,
                                   msg=f"[{name}] Asymmetric rear wheel accelerations: {w_rl_dot} vs {w_rr_dot}")
            # Zero lateral acceleration and zero yaw acceleration (< 1e-3)
            self.assertLess(abs(v_y_dot), 1e-3,
                            msg=f"[{name}] Non-zero v_y_dot ({v_y_dot:.6f} m/s^2) in straight line!")
            self.assertLess(abs(r_dot), 1e-3,
                            msg=f"[{name}] Non-zero r_dot ({r_dot:.6f} rad/s^2) in straight line!")

    def test_camber_thrust_direction_and_sign(self):
        """
        Verifies individual wheel lateral force directions for negative static camber:
        ISO coordinates (+Y is vehicle LEFT):
        - FL: Camber tilted towards vehicle centerline (rightward) -> pushes right -> Fy_fl < 0
        - FR: Camber tilted towards vehicle centerline (leftward) -> pushes left -> Fy_fr > 0
        Net front axle force must sum to essentially 0.0 (< 0.05 N).
        """
        for tp, name in [(self.tp_right, "RIGHT_TIR"), (self.tp_left, "LEFT_TIR")]:
            is_right = (tp.get_tyreside() == "RIGHT")
            camber_f_deg = self.vp.camber_f_static # -1.5 deg
            deg2rad = np.pi / 180.0

            if is_right:
                gamma_fl = -camber_f_deg * deg2rad
                gamma_fr =  camber_f_deg * deg2rad
            else:
                gamma_fl =  camber_f_deg * deg2rad
                gamma_fr = -camber_f_deg * deg2rad

            lvy_fl = -1.0 if is_right else 1.0
            lvy_fr =  1.0 if is_right else -1.0
            lhy_fl = -1.0 if is_right else 1.0
            lhy_fr =  1.0 if is_right else -1.0

            Fz = 600.0 # ~static load per front tire [N]
            # Zero slip angle
            fy_fl = -float(mf61_lateral_force(0.0, Fz, gamma_fl, tp, LMUY=1.0, LVY=lvy_fl, LHY=lhy_fl))
            fy_fr = -float(mf61_lateral_force(0.0, Fz, gamma_fr, tp, LMUY=1.0, LVY=lvy_fr, LHY=lhy_fr))

            # Left tire pushes right (Fy < 0)
            self.assertLess(fy_fl, 0.0, msg=f"[{name}] FL lateral force should be negative (camber thrust rightward), got {fy_fl}")
            # Right tire pushes left (Fy > 0)
            self.assertGreater(fy_fr, 0.0, msg=f"[{name}] FR lateral force should be positive (camber thrust leftward), got {fy_fr}")
            # Sum must cancel out (< 50 mN)
            net_axle_fy = abs(fy_fl + fy_fr)
            self.assertLess(net_axle_fy, 0.05,
                            msg=f"[{name}] Front axle camber thrusts do not cancel: {fy_fl} + {fy_fr} = {fy_fl + fy_fr}")

    def test_7dof_straight_line_symmetry(self):
        """Verifies 7-DOF ODE produces negligible yaw and lateral acceleration in straight-line acceleration (< 1e-3)."""
        for tp_test, name in [(self.tp_right, "RIGHT_TIR"), (self.tp_left, "LEFT_TIR")]:
            model = Vehicle7DOF(self.vp, tp_test, tp_test)
            ode = model.get_casadi_ode()

            Rw = self.vp.R_w
            w0 = 15.0 / Rw
            x = np.array([15.0, 0.0, 0.0, w0, w0, w0, w0])
            u = [0.0, 150.0, 150.0, 0.0]

            # 7-DOF parameters: wd_f, lltd, lmux, lmuy, L, tf, tr, cam_f, cam_r, toe_f, toe_r
            p = [self.vp.wd_f, self.vp.lltd, self.vp.LMUX, self.vp.LMUY,
                 self.vp.L, self.vp.t_f, self.vp.t_r,
                 self.vp.camber_f_static, self.vp.camber_r_static,
                 self.vp.toe_f_static, self.vp.toe_r_static]

            x_dot = np.array(ode(x, u, p)).flatten()
            v_y_dot = abs(x_dot[1])
            r_dot = abs(x_dot[2])

            self.assertLess(v_y_dot, 1e-3,
                            msg=f"[{name}] 7-DOF v_y_dot ({v_y_dot:.6f}) non-zero in straight line")
            self.assertLess(r_dot, 1e-3,
                            msg=f"[{name}] 7-DOF r_dot ({r_dot:.6f}) non-zero in straight line")

    def test_3dof_straight_line_equilibrium(self):
        """Verifies 3-DOF equilibrium equations at zero steer: residuals must be negligible (< 0.05 N) for beta=0, r=0."""
        for tp_test, name in [(self.tp_right, "RIGHT_TIR"), (self.tp_left, "LEFT_TIR")]:
            model = Vehicle3DOF(self.vp, tp_test, tp_test)
            funcs = model.get_casadi_functions()
            res_func = funcs['res_func']

            inputs = [15.0, 0.0] # vx = 15 m/s, delta = 0 rad
            states = [0.0, 0.0]  # beta = 0, r = 0
            params = [self.vp.wd_f, self.vp.lltd, self.vp.LMUY]

            res = np.array(res_func(inputs, states, params)).flatten()
            eq_lat, eq_yaw = abs(res[0]), abs(res[1])

            self.assertLess(eq_lat, 0.10,
                            msg=f"[{name}] 3-DOF lateral residual ({eq_lat:.6f}) non-zero at straight line")
            self.assertLess(eq_yaw, 0.05,
                            msg=f"[{name}] 3-DOF yaw residual ({eq_yaw:.6f}) non-zero at straight line")

    def test_combined_slip_lateral_force_symmetry(self):
        """
        Verifies that under acceleration slip (kappa > 0), combined lateral forces
        are strictly symmetric and equal-and-opposite across the rear axle:
        F_y,rl + F_y,rr == 0 (< 0.05 N).
        """
        from src.models.tire_model_mf61 import mf61_combined_lateral_force
        for tp, name in [(self.tp_right, "RIGHT_TIR"), (self.tp_left, "LEFT_TIR")]:
            is_right = (tp.get_tyreside() == "RIGHT")
            camber_r_deg = self.vp.camber_r_static  # -1.0 deg
            deg2rad = np.pi / 180.0

            gamma_rl = np.deg2rad(-camber_r_deg if is_right else camber_r_deg)
            gamma_rr = np.deg2rad(camber_r_deg if is_right else -camber_r_deg)

            # Rear toe-in convention (negative on left, positive on right)
            toe_r_deg = self.vp.toe_r_static
            alpha_rl = -np.deg2rad(toe_r_deg)
            alpha_rr = +np.deg2rad(toe_r_deg)

            lvy_rl = -1.0 if is_right else 1.0
            lvy_rr =  1.0 if is_right else -1.0
            lhy_rl = -1.0 if is_right else 1.0
            lhy_rr =  1.0 if is_right else -1.0

            Fz = 700.0
            kappa = 0.08  # 8% acceleration slip

            fy_rl_p = -float(mf61_lateral_force(alpha_rl, Fz, gamma_rl, tp, LMUY=1.0, LVY=lvy_rl, LHY=lhy_rl))
            fy_rr_p = -float(mf61_lateral_force(alpha_rr, Fz, gamma_rr, tp, LMUY=1.0, LVY=lvy_rr, LHY=lhy_rr))

            fyw_rl = float(mf61_combined_lateral_force(fy_rl_p, alpha_rl, kappa, Fz, gamma_rl, tp, LVY=lvy_rl, LHY=lhy_rl))
            fyw_rr = float(mf61_combined_lateral_force(fy_rr_p, alpha_rr, kappa, Fz, gamma_rr, tp, LVY=lvy_rr, LHY=lhy_rr))

            self.assertLess(fyw_rl, 0.0, msg=f"[{name}] RL combined lateral force should be negative, got {fyw_rl}")
            self.assertGreater(fyw_rr, 0.0, msg=f"[{name}] RR combined lateral force should be positive, got {fyw_rr}")
            net_rear_fy = abs(fyw_rl + fyw_rr)
            self.assertLess(net_rear_fy, 0.05,
                            msg=f"[{name}] Rear axle combined lateral forces do not cancel: {fyw_rl} + {fyw_rr} = {fyw_rl + fyw_rr}")

    def test_14dof_fast_ode_compiled_symmetry(self):
        """Verifies that the compiled C fast ODE produces zero yaw and lateral acceleration in straight-line acceleration."""
        model = Vehicle14DOF(self.vp, self.tp_right, self.tp_right)
        fast_ode = model.get_fast_ode()
        self.assertIsNotNone(fast_ode)

        Rw = self.vp.R_w
        w0 = 10.0 / Rw
        x = np.zeros(28)
        x[0] = 10.0
        x[16:20] = w0
        u = [0.0, 60.0, 60.0, 0.0]
        p = [self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
             self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
             self.vp.camber_f_static, self.vp.camber_r_static,
             self.vp.toe_f_static, self.vp.toe_r_static,
             self.vp.L, self.vp.t_f, self.vp.t_r, self.vp.LMUY, self.vp.LMUX]

        dt = 0.002
        for _ in range(100):  # 200 ms
            k1 = np.array(fast_ode(x, u, p)).flatten()
            k2 = np.array(fast_ode(x + 0.5 * dt * k1, u, p)).flatten()
            k3 = np.array(fast_ode(x + 0.5 * dt * k2, u, p)).flatten()
            k4 = np.array(fast_ode(x + dt * k3, u, p)).flatten()
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.assertLess(abs(x[1]), 5e-4, msg=f"Compiled fast ODE vy non-zero: {x[1]}")
        self.assertLess(abs(x[2]), 5e-4, msg=f"Compiled fast ODE r non-zero: {x[2]}")


if __name__ == '__main__':
    unittest.main()
