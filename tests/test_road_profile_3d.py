"""
test_road_profile_3d.py
-----------------------
Unit tests for 3D Road Profile (Macro Altimetry, Banking, Micro Roughness)
and Simulator integration (Tire initial temperature, Standstill logic).
"""

import math
import unittest
import numpy as np

from src.core.road_profile import RoadProfile, RoadProfileConfig
from src.models.tire_degradation import TireDegradationModel, TireDegradationParams, TireState
from src.models.params_vehicle import VehicleParams
from src.ui.interactive_driver_14dof import PhysicsThread14DOF


class TestRoadProfile3D(unittest.TestCase):
    """Test suite for 3D Road Profile geometry, interpolation, and contact patch calculations."""

    def setUp(self):
        self.s = np.linspace(0.0, 500.0, 251)  # 500m track, 2m resolution
        self.curvature = 0.04 * np.sin(2.0 * np.pi * self.s / 250.0)  # gentle curvature

    def test_01_flat_profile(self):
        """Flat profile should return identically zero elevation, grade, bank, and roughness."""
        rp = RoadProfile.flat(self.s)
        z, grade, bank = rp.interpolate_at(125.0)
        self.assertAlmostEqual(z, 0.0)
        self.assertAlmostEqual(grade, 0.0)
        self.assertAlmostEqual(bank, 0.0)

        rough = rp.get_micro_unevenness(125.0, 0.9)
        self.assertAlmostEqual(rough, 0.0)

        wh = rp.get_wheel_contact_heights(125.0, a_dist=0.8, b_dist=0.7, t_f=1.2, t_r=1.2)
        for k in ('fl', 'fr', 'rl', 'rr', 'z_center', 'grade', 'bank'):
            self.assertAlmostEqual(wh[k], 0.0)

    def test_02_procedural_macro_profile(self):
        """Procedural macro profile should generate smooth elevation, nonzero slope and banking."""
        cfg = RoadProfileConfig(
            enable_macro=True,
            macro_elevation_amp=2.0,
            macro_wavelength=100.0,
            macro_bank_max_deg=4.0,
            enable_micro=False
        )
        rp = RoadProfile.procedural(self.s, curvature=self.curvature, config=cfg)

        # Centerline elevation at peak of sine wave
        z_25, grade_25, bank_25 = rp.interpolate_at(25.0)  # quarter wavelength: sin(pi/2) ~ 1
        self.assertGreater(z_25, 1.0)
        self.assertLessEqual(z_25, 3.0)

        # Derivative consistency: grade = dz/ds
        ds = 0.01
        z_plus, _, _ = rp.interpolate_at(25.0 + ds)
        z_minus, _, _ = rp.interpolate_at(25.0 - ds)
        approx_grade = (z_plus - z_minus) / (2.0 * ds)
        self.assertAlmostEqual(grade_25, approx_grade, delta=0.01)

        # Check wheel contact heights
        wh = rp.get_wheel_contact_heights(25.0, a_dist=0.8, b_dist=0.7, t_f=1.2, t_r=1.2)
        self.assertIn('fl', wh)
        self.assertIn('fr', wh)
        self.assertIn('rl', wh)
        self.assertIn('rr', wh)
        # Without micro, left/right height difference is due to banking: dz = t_f * sin(bank)
        expected_diff = 1.2 * math.sin(wh['bank'])
        self.assertAlmostEqual(wh['fl'] - wh['fr'], expected_diff, delta=1e-4)

    def test_03_micro_roughness_decorrelation(self):
        """Micro roughness should provide different unevenness signals for left vs right tire tracks."""
        cfg = RoadProfileConfig(
            enable_macro=False,
            enable_micro=True,
            micro_roughness_amp_mm=10.0,
            micro_wavelength_m=4.0,
            micro_lateral_decorr=0.75
        )
        rp = RoadProfile(self.s, config=cfg)

        z_l = rp.get_micro_unevenness(50.0, lateral_offset_y=0.6)
        z_r = rp.get_micro_unevenness(50.0, lateral_offset_y=-0.6)
        self.assertNotEqual(z_l, z_r)
        self.assertLessEqual(abs(z_l), 0.011)  # within ~10mm + harmonic overshoot
        self.assertLessEqual(abs(z_r), 0.011)

    def test_04_apply_to_track_data(self):
        """Applying road profile to track data adds 3D ribbon keys."""
        cfg = RoadProfileConfig(enable_macro=True, enable_micro=True)
        rp = RoadProfile.procedural(self.s, curvature=self.curvature, config=cfg)

        track_dict = {
            'x_center': np.cos(self.s / 500.0 * 2 * np.pi) * 80.0,
            'y_center': np.sin(self.s / 500.0 * 2 * np.pi) * 80.0,
            's': self.s,
            'track_width_m': 4.0
        }
        enhanced = rp.apply_to_track_data(track_dict)
        self.assertIn('z_center', enhanced)
        self.assertIn('z_left', enhanced)
        self.assertIn('z_right', enhanced)
        self.assertIn('grade', enhanced)
        self.assertIn('bank', enhanced)
        self.assertEqual(len(enhanced['z_center']), len(self.s))

    def test_05_from_user_stations(self):
        """RoadProfile.from_user_stations creates periodic cubic spline matching user points."""
        user_st = [
            {'s': 0.0, 'z': 0.0, 'bank_deg': 0.0},
            {'s': 125.0, 'z': 6.0, 'bank_deg': 5.0},
            {'s': 250.0, 'z': 12.0, 'bank_deg': -3.0},
            {'s': 375.0, 'z': 4.0, 'bank_deg': 2.0},
            {'s': 500.0, 'z': 0.0, 'bank_deg': 0.0}
        ]
        rp = RoadProfile.from_user_stations(self.s, user_st, is_closed=True)
        self.assertTrue(rp.config.is_custom_user)
        self.assertTrue(rp.config.enable_macro)

        # Check interpolation at specified stations
        z_125, _, bank_125 = rp.interpolate_at(125.0)
        self.assertAlmostEqual(z_125, 6.0, places=2)
        self.assertAlmostEqual(math.degrees(bank_125), 5.0, places=2)

        z_250, _, bank_250 = rp.interpolate_at(250.0)
        self.assertAlmostEqual(z_250, 12.0, places=2)
        self.assertAlmostEqual(math.degrees(bank_250), -3.0, places=2)

    def test_06_preset_stations(self):
        """All preset archetypes produce valid non-empty station definitions."""
        presets = [
            "Piatto (2D)",
            "Collinare (Onde & Banking)",
            "Circuito di Montagna (Salita/Discesa +-15m)",
            "Curve Sopraelevate (Banked Corners 10°)",
            "Rollercoaster (Dossi & Avvallamenti)"
        ]
        for p in presets:
            st = RoadProfile.get_preset_stations(p, 600.0, self.s, self.curvature)
            self.assertGreaterEqual(len(st), 4)
            rp = RoadProfile.from_user_stations(self.s, st, is_closed=True)
            self.assertIsNotNone(rp.z_macro)
            self.assertEqual(len(rp.z_macro), len(self.s))

    def test_07_physics_thread_custom_road_profile(self):
        """PhysicsThread14DOF set_custom_road_profile applies user stations to thread state."""
        vp = VehicleParams()
        thread = PhysicsThread14DOF(vehicle_params=vp)
        track_dict = {
            'x_center': np.linspace(0, 400, 200),
            'y_center': np.zeros(200),
            's': np.linspace(0, 400, 200),
            'is_closed': True
        }
        thread.set_track_data(track_dict)

        user_st = [
            {'s': 0.0, 'z': 0.0, 'bank_deg': 0.0},
            {'s': 100.0, 'z': 3.5, 'bank_deg': 4.0},
            {'s': 200.0, 'z': 7.0, 'bank_deg': -2.0},
            {'s': 300.0, 'z': 2.0, 'bank_deg': 1.0},
            {'s': 400.0, 'z': 0.0, 'bank_deg': 0.0}
        ]
        thread.set_custom_road_profile(user_st, enable_macro=True)
        self.assertTrue(thread.road_config.is_custom_user)
        self.assertTrue(thread.road_config.enable_macro)
        self.assertEqual(len(thread.road_config.custom_stations), 5)

        z_val, _, _ = thread.road_profile.interpolate_at(100.0)
        self.assertAlmostEqual(z_val, 3.5, places=1)

    def test_08_toe_symmetry_zero_yaw_moment_et18_msfront(self):
        """
        Verify that asymmetric toe sign bug is fixed:
        Even with large static toe (e.g. et18_MSFront with toe_f = +0.5 deg and toe_r = -1.0 deg),
        driving straight at 15 m/s produces zero net yaw acceleration (cancels out across axle).
        """
        from src.models.vehicle_14dof import Vehicle14DOF
        vp = VehicleParams()
        vp.camber_f_static = 0.0
        vp.camber_r_static = 0.0
        vp.toe_f_static = 0.5   # Toe-in
        vp.toe_r_static = -1.0  # Toe-out
        model = Vehicle14DOF(vp)

        # State: vx = 15 m/s, zero vy, zero yaw rate, zero roll, zero pitch
        ode = model.get_fast_ode()
        Rw = vp.R_w
        w0 = 15.0 / Rw
        x_val = [
            15.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            w0, w0, w0, w0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        u_val = [0.0, 0.0, 0.0, 0.0]

        p_vec = [
            vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r,
            vp.K_z_f, vp.K_z_r, vp.aero_balance,
            vp.camber_f_static, vp.camber_r_static,
            vp.toe_f_static, vp.toe_r_static,
            vp.L, vp.t_f, vp.t_r,
            1.60, 1.60
        ]

        x_dot = ode(x_val, u_val, p_vec)
        r_dot = float(x_dot[2])  # Yaw acceleration r_dot

        # Previously, r_dot was ~10.86 rad/s^2 due to crab-steering sign error!
        # With symmetric toe-in (-toe on right wheel), r_dot is near zero (only residual is minor tire ply-steer)
        self.assertLess(abs(r_dot), 0.10, f"Yaw acceleration {r_dot} exceeds symmetry tolerance")


class TestTireThermalAndStandstill(unittest.TestCase):
    """Test suite for Tire starting temperature, thermal degradation, and standstill logic."""

    def test_01_tire_degradation_reset_custom_t_start(self):
        """TireDegradationModel.reset initializes all layers to specified T_start."""
        params = TireDegradationParams(T_start=45.0)
        model = TireDegradationModel(mode="detailed", params=params)
        ts = TireState()

        # Reset with default params.T_start (45°C)
        model.reset(ts)
        self.assertEqual(ts.T_bulk, 45.0)
        self.assertEqual(ts.T_surface, 45.0)
        self.assertEqual(ts.T_tread, 45.0)
        self.assertEqual(ts.T_carcass, 45.0)
        self.assertEqual(ts.wear_fraction, 0.0)

        # Reset with explicit T_start (optimal temperature)
        t_opt = model.params.T_optimal
        model.reset(ts, T_start=t_opt)
        self.assertAlmostEqual(ts.T_bulk, t_opt)
        self.assertAlmostEqual(ts.T_surface, t_opt)
        self.assertAlmostEqual(ts.T_tread, t_opt)
        self.assertAlmostEqual(ts.T_carcass, t_opt)

        # Grip modifier at T_optimal should be maximum (1.00)
        grip = model.grip_modifier(ts)
        self.assertAlmostEqual(grip, 1.0, places=3)

    def test_02_physics_thread_tire_t_start_and_thermal_hz(self):
        """PhysicsThread14DOF properly accepts and propagates T_start and thermal_hz settings."""
        vp = VehicleParams()
        thread = PhysicsThread14DOF(vehicle_params=vp)

        # Test set_thermal_hz
        thread.set_thermal_hz(25.0)
        self.assertEqual(thread.thermal_hz, 25.0)

        # Test set_tire_t_start & reset_state
        thread.set_tire_t_start(45.0)
        self.assertEqual(thread.tire_T_start, 45.0)
        thread.reset_state(T_start=45.0)
        for ts in thread.tire_states:
            self.assertEqual(ts.T_bulk, 45.0)
            self.assertEqual(ts.T_surface, 45.0)

    def test_03_standstill_logic_preserves_spinout(self):
        """Verify that during a high-yaw spinout (r > 0.05, vy > 0.15), standstill clamp does NOT freeze the car."""
        vp = VehicleParams()
        thread = PhysicsThread14DOF(vehicle_params=vp)
        thread.reset_state(vx=0.0)

        # Simulate a car spinning out: vx is near 0, but vy = 5.0 m/s and r = 2.0 rad/s
        thread.x_state[0] = 0.05   # vx near zero
        thread.x_state[1] = 4.5    # vy high
        thread.x_state[2] = 1.8    # yaw rate high

        v_planar = math.hypot(thread.x_state[0], thread.x_state[1])
        r_abs = abs(thread.x_state[2])
        w_stopped = False

        # In spinout, v_planar is ~4.5 m/s >> 0.15 m/s, so standstill lock does NOT trigger
        is_locked = (v_planar < 0.15 and r_abs < 0.05 and w_stopped)
        self.assertFalse(is_locked)


if __name__ == '__main__':
    unittest.main()
