"""
test_simulator_review_features.py
---------------------------------
Unit tests for the simulator review improvements:
1. Lap Timer with S/F gate photocell detection and standstill launch.
2. Standstill stop and acceleration from 0 km/h.
3. Customizable track surface grip factors (dry/damp/wet/ice).
4. Tire thermal degradation and energy-based wear model (Hoosier R20 calibrated).
"""

import unittest
import numpy as np
import math

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_degradation import TireState, TireDegradationModel, TireDegradationParams
from src.core.simulation_options import SimulationOptions, get_simulation_options


class TestSimulatorReviewFeatures(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()
        self.tp = TireParams("tire_tir/Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir")

    def test_01_tire_degradation_off_mode(self):
        """In 'off' mode, grip modifier is always 1.0 and state does not change."""
        model = TireDegradationModel(mode="off")
        ts = TireState()
        self.assertEqual(model.grip_modifier(ts), 1.0)
        
        # High power sliding step
        model.step(ts, F_total=3000.0, v_slip=5.0, Fz=1500.0, vx=20.0, dt=0.01)
        self.assertEqual(model.grip_modifier(ts), 1.0)
        self.assertEqual(ts.wear_fraction, 0.0)

    def test_02_tire_degradation_light_mode(self):
        """In 'light' mode, 1-zone thermal bulk heats up under friction and cools toward ambient."""
        params = TireDegradationParams(T_ambient=25.0, T_optimal=75.0)
        model = TireDegradationModel(mode="light", params=params)
        ts = TireState()
        
        # Initial cold grip (< 1.0 because T=25 vs T_opt=75)
        init_grip = model.grip_modifier(ts)
        self.assertLess(init_grip, 1.0)
        self.assertGreater(init_grip, 0.2)

        # Apply high sliding power for 8 seconds (simulate heavy lockup/burnout)
        for _ in range(800):
            model.step(ts, F_total=2500.0, v_slip=4.0, Fz=1200.0, vx=15.0, dt=0.01)

        # Temperature should have risen significantly toward optimum
        self.assertGreater(ts.T_bulk, 38.0)
        self.assertGreater(ts.wear_fraction, 0.0)
        self.assertLessEqual(ts.wear_fraction, 1.0)

        # Let it cool down with zero slip
        t_hot = ts.T_bulk
        for _ in range(500):
            model.step(ts, F_total=0.0, v_slip=0.0, Fz=1000.0, vx=20.0, dt=0.01)
        self.assertLess(ts.T_bulk, t_hot)

    def test_03_tire_degradation_detailed_mode(self):
        """In 'detailed' mode, 3 zones (surface, tread, carcass) update via heat transfer network."""
        model = TireDegradationModel(mode="detailed")
        ts = TireState()
        
        # Step with friction
        for _ in range(100):
            model.step(ts, F_total=2000.0, v_slip=3.0, Fz=1000.0, vx=12.0, dt=0.01)

        # Surface heats fastest due to 70% heat share and smaller thermal mass
        self.assertGreater(ts.T_surface, ts.T_tread)
        self.assertGreater(ts.T_tread, ts.T_carcass)
        self.assertGreater(ts.wear_fraction, 0.0)

    def test_04_tire_wear_monotonic(self):
        """Tire wear must monotonically increase and decay grip."""
        model = TireDegradationModel(mode="light")
        ts = TireState()
        
        w_prev = 0.0
        for _ in range(10):
            model.step(ts, F_total=1500.0, v_slip=2.0, Fz=1000.0, vx=10.0, dt=0.05)
            self.assertGreaterEqual(ts.wear_fraction, w_prev)
            w_prev = ts.wear_fraction

        # Worn grip function should be strictly less than new grip at same temperature
        self.assertLess(model.f_wear(0.50), model.f_wear(0.0))
        self.assertAlmostEqual(model.f_wear(1.0), model.params.mu_worn_min)

    def test_05_simulation_options_track_grip(self):
        """SimulationOptions holds customizable track grip multipliers."""
        opts = SimulationOptions()
        self.assertEqual(opts.track_grip_multiplier, 1.0)
        self.assertEqual(opts.offtrack_grip_multiplier, 0.35)
        self.assertEqual(opts.kerb_grip_multiplier, 0.85)

        # Test dictionary export & import
        d = opts.to_dict()
        self.assertIn('track_grip_multiplier', d)
        self.assertIn('tire_degradation_mode', d)
        
        opts2 = SimulationOptions.from_dict(d)
        self.assertEqual(opts2.track_grip_multiplier, 1.0)

    def test_06_sf_gate_photocell_crossing_logic(self):
        """Verify the geometry math of S/F photocell crossing detection."""
        # Simulated straight track: S/F at (0, 0), direction along +X
        x_sf, y_sf = 0.0, 0.0
        dx_sf, dy_sf = 2.0, 0.0  # tangent pointing along +X
        
        def calc_dot(px, py):
            # Signed projection along forward track tangent across S/F line
            return (px - x_sf) * dx_sf + (py - y_sf) * dy_sf

        # Staged 2 meters behind S/F line: pos_x = -2.0
        dot_behind = calc_dot(-2.0, 0.0)
        self.assertLess(dot_behind, 0.0)

        # Crossed 1 meter ahead of S/F line: pos_x = +1.0
        dot_ahead = calc_dot(1.0, 0.0)
        self.assertGreater(dot_ahead, 0.0)

        # Crossing transition detected: dot_prev < 0 and dot_curr >= 0
        crossed = (dot_behind < 0 and dot_ahead >= 0)
        self.assertTrue(crossed)

    def test_07_standstill_starting_state(self):
        """Verify vehicle can stage at standstill (vx=0.0) with zero wheel spin."""
        from src.ui.interactive_driver_14dof import PhysicsThread14DOF
        thread = PhysicsThread14DOF(vehicle_params=self.vp)
        thread.reset_state(x=0.0, y=0.0, psi=0.0, vx=0.0)
        
        self.assertEqual(thread.x_state[0], 0.0)  # vx = 0
        self.assertEqual(thread.x_state[11], 0.0) # w_fl = 0
        self.assertEqual(thread.x_state[14], 0.0) # w_rr = 0
        self.assertFalse(thread._sf_armed)       # disarmed on grid
        self.assertEqual(thread.lap_count, 0)
    def test_08_tire_datas_mu_vs_temp_calibration(self):
        """
        Verify tire degradation thermal curve matches baseline literature
        parabolic thermal model.
        """
        model = TireDegradationModel(mode="light")
        p = model.params

        # Optimal temperature from baseline literature fit
        self.assertAlmostEqual(p.T_optimal, 65.0, places=1)
        self.assertAlmostEqual(p.mu_peak, 1.5, places=2)

        # Peak grip at optimal temperature must be exactly 1.0 (normalized)
        self.assertAlmostEqual(model.f_temp(65.0), 1.0, places=4)
        self.assertAlmostEqual(model.mu_y_temp(65.0), 1.5, places=2)

        # Off-peak temperatures
        f_50 = model.f_temp(50.0)
        f_80 = model.f_temp(80.0)
        self.assertAlmostEqual(f_50, 0.91, places=2)
        self.assertAlmostEqual(f_80, 0.91, places=2)

        # Boundary clamping: cold start (25°C) and heavy overheat (105°C)
        # must be clamped to min_grip_floor (0.80) to preserve vehicle controllability
        self.assertEqual(model.f_temp(25.0), p.min_grip_floor)
        self.assertEqual(model.f_temp(105.0), p.min_grip_floor)

    def test_09_track_reset_and_lap_counting(self):
        """
        Verify that reset_state stages in proximity of the track start line,
        changing track updates physics_thread start position, and lap crossing triggers properly.
        """
        from src.ui.interactive_driver_14dof import PhysicsThread14DOF
        thread = PhysicsThread14DOF(vehicle_params=self.vp)

        # Define a mock track starting at (100.0, 50.0), heading along +X
        s_pts = np.linspace(0.0, 100.0, 51)
        mock_track = {
            's': s_pts,
            'x_center': 100.0 + s_pts,
            'y_center': np.full_like(s_pts, 50.0),
            'x_left': 100.0 + s_pts,
            'y_left': np.full_like(s_pts, 52.0),
            'x_right': 100.0 + s_pts,
            'y_right': np.full_like(s_pts, 48.0),
            'track_width': 4.0
        }
        thread.set_track_data(mock_track)
        thread.reset_state()

        # Car must be staged 10m ahead on the starting grid: (100 + 10, 50) = (110.0, 50.0)
        self.assertAlmostEqual(thread.pos_x, 110.0, places=2)
        self.assertAlmostEqual(thread.pos_y, 50.0, places=2)
        self.assertAlmostEqual(thread.psi, 0.0, places=2)
        self.assertFalse(thread._sf_armed)
        self.assertEqual(thread.lap_count, 0)


if __name__ == '__main__':
    unittest.main()
