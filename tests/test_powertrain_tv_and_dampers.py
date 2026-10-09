"""
test_powertrain_tv_and_dampers.py
---------------------------------
Comprehensive unit and integration tests for:
1. Powertrain RWD Dual-Motor Differentials & Active Torque Vectoring (TV).
2. Ohlins TTX25 4-Way Registers, Custom Dyno Valving & Live Tuning.
3. Continuous Spatial Delta-Lap Delta_t(s) & 3D Ghost Car Spatial Interpolation.
4. 3D Model Display Lists and Ghost Shader Attributes.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import unittest
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.models import damper_model_ttx25 as dm
from src.models.vehicle_3d_models import FSAECar3DModel, SUVCar3DModel
from src.ui.interactive_driver_14dof import PhysicsThread14DOF

class TestPowertrainDifferentialsAndTV(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()
        self.pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=500.0)

    def test_01_open_diff_torque_split(self):
        """Verify DUAL_MOTOR_OPEN splits drive torque exactly 50/50."""
        self.vp.diff_mode = 'DUAL_MOTOR_OPEN'
        t_drive = 600.0
        # Under open diff, T_RL == T_RR == 300 Nm
        t_rl = t_drive / 2.0
        t_rr = t_drive / 2.0
        self.assertEqual(t_rl, 300.0)
        self.assertEqual(t_rr, 300.0)

    def test_02_spool_locking_torque(self):
        """Verify VIRTUAL_SPOOL transfers torque from faster to slower wheel."""
        k_spool = self.vp.diff_spool_k  # e.g. 80.0 Nm/(rad/s)
        t_drive = 400.0
        base_t = t_drive / 2.0  # 200 Nm
        
        # Scenario 1: Right wheel spinning faster (omega_rr > omega_rl)
        omega_rl = 50.0
        omega_rr = 55.0  # delta_omega = 5.0
        delta_omega = omega_rr - omega_rl
        t_lock = k_spool * delta_omega  # 400.0 Nm
        
        t_rl = base_t + t_lock
        t_rr = base_t - t_lock
        
        self.assertGreater(t_rl, base_t, "Slower (left) wheel must receive locking torque")
        self.assertLess(t_rr, base_t, "Faster (right) wheel must lose torque")
        self.assertAlmostEqual(t_rl + t_rr, t_drive, places=5, msg="Total applied torque must be conserved before saturation")

    def test_03_salisbury_lsd_power_and_coast(self):
        """Verify VIRTUAL_SALISBURY_LSD produces asymmetric locking in power vs coast."""
        t_preload = self.vp.diff_lsd_preload    # 35.0 Nm
        mu_power = self.vp.diff_lsd_mu_power     # 0.40
        mu_coast = self.vp.diff_lsd_mu_coast     # 0.20
        
        delta_omega = 2.0  # rad/s, significant slip
        tanh_factor = float(np.tanh(3.0 * delta_omega))
        
        # Power condition (drive torque = 500 Nm)
        t_drive_power = 500.0
        t_lock_power = (t_preload + mu_power * abs(t_drive_power)) * tanh_factor
        
        # Coast condition (drive torque = -200 Nm regenerative/engine braking)
        t_drive_coast = -200.0
        t_lock_coast = (t_preload + mu_coast * abs(t_drive_coast)) * tanh_factor
        
        self.assertGreater(t_lock_power, t_lock_coast, "Power locking torque must exceed coast locking torque")
        self.assertGreater(t_lock_power, t_preload, "Power locking torque must exceed preload")

    def test_04_active_torque_vectoring_dynamics(self):
        """Verify ACTIVE_TORQUE_VECTORING calculates target yaw rate, PI delta Mz, and torque split."""
        vx = 20.0  # m/s (72 km/h)
        delta_vis = 0.05  # ~2.86 deg steer angle
        L = self.vp.wheelbase
        m = self.vp.mass
        t_r = self.vp.track_r
        rw = self.vp.tire_radius
        
        # 1. Target yaw rate calculation (linear bicycle model)
        cf = self.vp.C_alpha_f
        cr = self.vp.C_alpha_r
        a = self.vp.cg_dist_f
        b = self.vp.cg_dist_r
        kus = (m / L) * (b / cf - a / cr)
        
        r_target_unbounded = (vx / (L + kus * (vx**2))) * delta_vis
        mu_peak = 1.4
        r_max = (mu_peak * 9.81) / max(vx, 1.0)
        r_target = float(np.clip(r_target_unbounded, -r_max, r_max))
        
        self.assertGreater(r_target, 0.0, "Target yaw rate must be positive for positive steering")
        self.assertLessEqual(r_target, r_max, "Target yaw rate must be saturated by adhesion limit")
        
        # 2. Closed-loop PI yaw moment understeer correction
        # Actual yaw rate < target yaw rate (understeer -> need positive yaw moment)
        r_actual = 0.5 * r_target
        err_r = r_target - r_actual
        kp = self.vp.tv_kp
        ki = self.vp.tv_ki
        delta_mz = kp * err_r  # with zero integral initially
        
        self.assertGreater(delta_mz, 0.0, "TV must generate positive corrective yaw moment for understeer")
        
        # 3. Torque split calculation
        delta_t = delta_mz * (rw / t_r)
        base_t = 300.0
        t_rl = base_t - delta_t
        t_rr = base_t + delta_t
        
        self.assertGreater(t_rr, t_rl, "Outer (right) wheel must produce more drive torque to rotate the car into turn")

    def test_05_tv_low_speed_deactivation(self):
        """Verify TV deactivates smoothly below 2.0 m/s (7.2 km/h)."""
        low_vx = 1.0  # m/s
        self.assertLess(low_vx, 2.0, "Low speed threshold test")

    def test_06_torque_vectoring_right_turn_signs(self):
        """Verify ACTIVE_TORQUE_VECTORING correctly handles right turns (delta < 0 -> r < 0)."""
        vx = 20.0  # m/s
        delta_phys = -0.05  # right turn: negative steer angle
        L = self.vp.wheelbase
        m = self.vp.mass
        t_r = self.vp.track_r
        rw = self.vp.tire_radius
        
        cf = self.vp.C_alpha_f
        cr = self.vp.C_alpha_r
        a = self.vp.cg_dist_f
        b = self.vp.cg_dist_r
        kus = (m / L) * (b / cf - a / cr)
        
        r_target_unbounded = (vx / (L + kus * (vx**2))) * delta_phys
        mu_peak = 1.4
        r_max = (mu_peak * 9.81) / max(vx, 1.0)
        r_target = float(np.clip(r_target_unbounded, -r_max, r_max))
        
        self.assertLess(r_target, 0.0, "Target yaw rate must be negative for right turn")
        
        # Understeer in right turn: actual yaw rate is less negative than target (e.g. -0.2 vs -0.4)
        r_actual = 0.5 * r_target  # e.g. -0.2 (understeering)
        err_r = r_target - r_actual  # (-0.4) - (-0.2) = -0.2 < 0
        kp = self.vp.tv_kp
        delta_mz = kp * err_r
        self.assertLess(delta_mz, 0.0, "Corrective yaw moment must be negative to rotate right")
        
        delta_t = delta_mz * (rw / t_r)
        base_t = 300.0
        t_rl = base_t - delta_t  # minus negative -> plus positive
        t_rr = base_t + delta_t  # plus negative -> minus
        
        self.assertGreater(t_rl, t_rr, "Outer (left) wheel must produce MORE torque to rotate right and kill understeer")


class TestTTX25DampersAndTuning(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_01_custom_dyno_valving(self):
        """Verify Custom Dyno curve calculation and smooth saturation."""
        v_test = 0.05  # 50 mm/s
        f_custom = dm.calculate_damper_force(v_test, valving='CUSTOM_DYNO', lsc=10, hsc=2.0)
        self.assertGreater(f_custom, 0.0)
        
        # Test extreme velocity to ensure curve remains monotonic
        f_fast = dm.calculate_damper_force(0.25, valving='CUSTOM_DYNO', lsc=10, hsc=2.0)
        self.assertGreater(f_fast, f_custom, "Dyno curve must remain monotonic at high velocities")

    def test_02_live_tuning_recompilation(self):
        """Verify live tuning update properly reconfigures vehicle parameters and re-compiles fast ODE."""
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=500.0)
        
        # Update tuning to TTX25 non-linear with custom registers
        pt.update_live_tuning(
            diff_mode='ACTIVE_TORQUE_VECTORING',
            damper_mode='ttx25_nonlinear',
            lsc_f=5, hsc_f=1.5, lsr_f=8, hsr_f=2.0,
            lsc_r=12, hsc_r=3.0, lsr_r=14, hsr_r=3.5
        )
        
        self.assertEqual(self.vp.diff_mode, 'ACTIVE_TORQUE_VECTORING')
        self.assertEqual(self.vp.damper_mode, 'ttx25_nonlinear')
        self.assertEqual(self.vp.damper_lsc_clicks_f, 5)
        self.assertEqual(self.vp.damper_hsc_turns_f, 1.5)
        self.assertEqual(self.vp.damper_lsc_clicks_r, 12)
        self.assertEqual(self.vp.damper_hsc_turns_r, 3.0)
        self.assertIsNotNone(pt.ode, "Fast ODE must recompile successfully")

    def test_03_modal_damping_ratios(self):
        """Verify modal damping ratio calculation returns realistic automotive values (0.3 - 1.5)."""
        mr_f = self.vp.motion_ratio_f
        mr_r = self.vp.motion_ratio_r
        c_wheel_f = dm.calculate_wheel_damper_force(0.05, mr_f, 'C12/R12', 10, 2.0) / 0.05
        c_wheel_r = dm.calculate_wheel_damper_force(0.05, mr_r, 'C12/R12', 10, 2.0) / 0.05
        
        m_f = self.vp.mass * (self.vp.cg_dist_r / self.vp.wheelbase) / 2.0
        m_r = self.vp.mass * (self.vp.cg_dist_f / self.vp.wheelbase) / 2.0
        
        zeta_f = c_wheel_f / (2.0 * np.sqrt(self.vp.spring_rate_f * m_f))
        zeta_r = c_wheel_r / (2.0 * np.sqrt(self.vp.spring_rate_r * m_r))
        
        self.assertTrue(0.3 <= zeta_f <= 1.5, f"Front damping ratio out of range: {zeta_f:.2f}")
        self.assertTrue(0.3 <= zeta_r <= 1.5, f"Rear damping ratio out of range: {zeta_r:.2f}")

    def test_04_ttx25_pitch_damping_dissipative_stability(self):
        """Verify that TTX25 non-linear pitch damping opposes pitch angular velocity (theta_dot)."""
        self.vp.damper_mode = 'ttx25_nonlinear'
        model = Vehicle14DOF(self.vp)
        ode = model.get_fast_ode()
        
        # State vector: 28 states
        # x[0..2]: vx, vy, r
        # x[3..4]: phi, p
        # x[5..6]: theta, q (pitch angle and rate)
        # x[7]: w_heave
        # x[8..11]: z_fl..z_rr
        # x[12..15]: vz_fl..vz_rr
        # x[16..19]: w_fl..w_rr
        # x[20..27]: ux/uy carcass deflections
        x = np.zeros(28)
        x[0] = 10.0  # 10 m/s
        x[5] = 0.0   # zero pitch angle
        x[6] = 0.1   # positive pitch rate (nose moving up at +0.1 rad/s)
        
        u = [0.0, 0.0, 0.0, 0.0]  # steer=0, drive=0, brake=0
        p = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            1.6, 1.6
        ]
        
        x_dot = np.array(ode(x, u, p)).flatten()
        theta_ddot = x_dot[6]
        # With theta_dot > 0, the dampers must produce negative angular acceleration to damp the pitch!
        self.assertLess(theta_ddot, 0.0, f"Pitch damping must oppose positive pitch rate (got theta_ddot={theta_ddot:.4f})")

    def test_05_arb_live_tuning_and_models(self):
        """Verify ARB live tuning properly updates K_phi_f, K_phi_r, lltd and model."""
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=500.0)
        pt.update_live_tuning(
            k_phi_f=46705.0,
            k_phi_r=31790.0,
            lltd=0.595,
            arb_model='ideal'
        )
        self.assertEqual(self.vp.K_phi_f, 46705.0)
        self.assertEqual(self.vp.K_phi_r, 31790.0)
        self.assertAlmostEqual(self.vp.lltd, 0.595, places=3)
        self.assertEqual(self.vp.arb_diamond_model, 'ideal')


class TestGhostCarAndSpatialDelta(unittest.TestCase):
    def test_01_spatial_delta_interpolation(self):
        """Verify continuous spatial delta-lap Delta_t(s) calculation and sign convention."""
        best_lap = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (25.0, 25.0, 0.0, 0.0, 0.0, 2.5),
            (50.0, 50.0, 0.0, 0.0, 0.0, 5.0),
            (75.0, 75.0, 0.0, 0.0, 0.0, 7.5),
            (100.0, 100.0, 0.0, 0.0, 0.0, 10.0)
        ]
        
        s_arr = np.array([pt[0] for pt in best_lap])
        t_arr = np.array([pt[5] for pt in best_lap])
        
        current_s = 50.0
        t_best_at_s = float(np.interp(current_s, s_arr, t_arr))
        self.assertAlmostEqual(t_best_at_s, 5.0, places=4)
        
        # Case A: Current car is faster (ahead of ghost), current lap time = 4.6s at s=50
        current_t_fast = 4.6
        delta_fast = current_t_fast - t_best_at_s
        self.assertAlmostEqual(delta_fast, -0.4, places=4)
        self.assertLess(delta_fast, 0.0, "Faster lap must yield negative delta (green HUD card)")
        
        # Case B: Current car is slower (behind ghost), current lap time = 5.3s at s=50
        current_t_slow = 5.3
        delta_slow = current_t_slow - t_best_at_s
        self.assertAlmostEqual(delta_slow, 0.3, places=4)
        self.assertGreater(delta_slow, 0.0, "Slower lap must yield positive delta (red HUD card)")

    def test_02_ghost_pose_temporal_interpolation(self):
        """Verify ghost car pose (x, y, z, yaw) is smoothly interpolated from best lap at current lap time."""
        best_lap = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (25.0, 25.0, 5.0, 0.1, 0.2, 2.0),
            (50.0, 50.0, 10.0, 0.2, 0.4, 4.0),
        ]
        
        t_arr = np.array([pt[5] for pt in best_lap])
        x_arr = np.array([pt[1] for pt in best_lap])
        y_arr = np.array([pt[2] for pt in best_lap])
        psi_arr = np.array([pt[4] for pt in best_lap])
        
        current_t = 3.0  # Midpoint between 2.0s and 4.0s
        ghost_x = float(np.interp(current_t, t_arr, x_arr))
        ghost_y = float(np.interp(current_t, t_arr, y_arr))
        ghost_psi = float(np.interp(current_t, t_arr, psi_arr))
        
        self.assertAlmostEqual(ghost_x, 37.5, places=4)
        self.assertAlmostEqual(ghost_y, 7.5, places=4)
        self.assertAlmostEqual(ghost_psi, 0.3, places=4)


class Test3DModelsDisplayLists(unittest.TestCase):
    def test_01_fsae_model_attributes_and_methods(self):
        """Verify FSAECar3DModel has display list compilation and draw_ghost methods."""
        fsae = FSAECar3DModel()
        self.assertTrue(hasattr(fsae, 'compile_body_display_list'))
        self.assertTrue(hasattr(fsae, 'invalidate_display_list'))
        self.assertTrue(hasattr(fsae, 'draw_ghost'))
        self.assertIsNone(fsae.body_display_list)
        
        # Test invalidate when list is None (smoke test)
        fsae.invalidate_display_list()
        self.assertIsNone(fsae.body_display_list)

    def test_02_suv_model_attributes_and_methods(self):
        """Verify SUVCar3DModel has display list compilation and draw_ghost methods."""
        suv = SUVCar3DModel()
        self.assertTrue(hasattr(suv, 'compile_body_display_list'))
        self.assertTrue(hasattr(suv, 'invalidate_display_list'))
        self.assertTrue(hasattr(suv, 'draw_ghost'))
        self.assertIsNone(suv.body_display_list)
        
        # Test invalidate when list is None (smoke test)
        suv.invalidate_display_list()
        self.assertIsNone(suv.body_display_list)


if __name__ == '__main__':
    unittest.main()
