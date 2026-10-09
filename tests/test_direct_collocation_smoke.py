import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdo.mdo_direct_collocation import build_track_geometry, compute_backward_forward_velocity_profile
from src.models.params_vehicle import VehicleParams


class TestDirectCollocationSmoke(unittest.TestCase):
    """Fast smoke tests for the Direct Collocation module."""

    def test_build_skidpad_track(self):
        """Verify skidpad track geometry builds with valid arrays."""
        track = build_track_geometry(track_type="fsae_skidpad", ds=5.0, track_width_m=3.8)
        self.assertIn('s', track)
        self.assertIn('kappa', track)
        self.assertTrue(len(track['s']) > 10)
        self.assertTrue(np.all(np.isfinite(track['s'])))
        self.assertTrue(np.all(np.isfinite(track['kappa'])))

    def test_build_fsae_autocross_track(self):
        """Verify FSAE autocross track geometry builds."""
        track = build_track_geometry(track_type="fsae_autocross", ds=5.0)
        self.assertIn('s', track)
        self.assertTrue(len(track['s']) > 50)

    def test_backward_forward_profile(self):
        """Verify backward-forward velocity profile produces physical results."""
        track = build_track_geometry(track_type="fsae_skidpad", ds=5.0)
        vp = VehicleParams()
        result = compute_backward_forward_velocity_profile(track, vp)
    def test_direct_collocation_nlp_solve(self):
        """Verify direct collocation NLP solver converges cleanly with rate regularization."""
        from src.mdo.mdo_direct_collocation import solve_direct_collocation_racing_line, DirectCollocationCostWeights
        track = build_track_geometry(track_type="fsae_skidpad", ds=5.0)
        vp = VehicleParams()
        res = solve_direct_collocation_racing_line(vehicle_params=vp, track_data=track, max_iter=60)
        self.assertTrue(res['success'], "Collocation NLP did not reach optimal/acceptable status")
        self.assertTrue(np.isfinite(res['lap_time_s']))
        self.assertGreater(res['v_max_kmh'], 70.0, "De-conservatism should allow speed > 70 km/h on skidpad straights")
        self.assertLess(res['lap_time_s'], 35.0)
        self.assertIn('cost_breakdown', res)
        self.assertIn('cost_total', res['cost_breakdown'])
        self.assertIn('cost_du2', res['cost_breakdown'])

    def test_direct_collocation_custom_weights(self):
        """Verify DirectCollocationCostWeights dataclass and custom weights execution."""
        from src.mdo.mdo_direct_collocation import solve_direct_collocation_racing_line, DirectCollocationCostWeights
        cw = DirectCollocationCostWeights(
            w_time=1.0,
            T_scale=40.0,
            rdu_ax=2e-4,
            rdu_delta=2e-3,
            rdu2_ax=2e-4,
            rdu2_delta=2e-3,
            ru_ax=1e-5,
            ru_delta=1e-5
        )
        self.assertEqual(cw.T_scale, 40.0)
        self.assertEqual(cw.rdu_ax, 2e-4)

        track = build_track_geometry(track_type="fsae_skidpad", ds=5.0)
        vp = VehicleParams()
        res = solve_direct_collocation_racing_line(vehicle_params=vp, track_data=track, max_iter=60, cost_weights=cw)
        self.assertTrue(res['success'])
        self.assertIn('cost_weights', res)
        self.assertEqual(res['cost_weights'].T_scale, 40.0)
        self.assertGreater(res['cost_breakdown']['cost_total'], 0.0)

    def test_direct_collocation_dict_weights_and_time_rates(self):
        """Verify passing dictionary and temporal rate mode (use_time_rates=True)."""
        from src.mdo.mdo_direct_collocation import solve_direct_collocation_racing_line
        weights_dict = {
            'w_time': 1.0,
            'use_time_rates': True,
            'rdu_ax': 1e-4,
            'rdu_delta': 1e-3
        }
        track = build_track_geometry(track_type="fsae_skidpad", ds=5.0)
        vp = VehicleParams()
        res = solve_direct_collocation_racing_line(vehicle_params=vp, track_data=track, max_iter=60, cost_weights=weights_dict)
        self.assertTrue(res['success'])
        self.assertTrue(res['cost_weights'].use_time_rates)


if __name__ == '__main__':
    unittest.main()

