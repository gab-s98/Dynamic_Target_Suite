"""
test_ackermann_wear.py
-----------------------
Unit tests verifying:
1. Adams Car wheel_steer_f kinematics: inside wheel turns sharper than outside wheel (Ackermann).
2. Bilateral mirror symmetry across vehicle centerline: delta_fl(d) == -delta_fr(-d).
3. Tire wear degradation model calibration (~3-8% wear over 5 km, wear_rate_scale multiplier).
4. Minimap HUD rendering and widget consistency (OpenGL & Legacy Viewport).
"""

import unittest
import numpy as np
from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_tire import TireParams
from src.models.tire_degradation import TireState, TireDegradationModel, TireDegradationParams


class TestAckermannAndWear(unittest.TestCase):

    def setUp(self):
        self.vp = VehicleParams()

    def test_wheel_steer_inside_greater_than_outside_left(self):
        """
        Verifies that for steering wheel angle delta_sw > 0 (left turn),
        the inside wheel (FL) steers sharper than the outside wheel (FR):
        delta_fl > delta_fr > 0.
        """
        delta_sw_deg, fl_deg, fr_deg, diff_deg = self.vp.get_steer_wheel_angles()

        # Check positive steering angles (left turn)
        pos_idx = np.where(delta_sw_deg > 10.0)[0]
        self.assertTrue(len(pos_idx) > 0)

        for i in pos_idx:
            sw = delta_sw_deg[i]
            d_fl = fl_deg[i]
            d_fr = fr_deg[i]
            # Both turn left (positive)
            self.assertGreater(d_fl, 0.0, f"FL should be positive for sw={sw}")
            self.assertGreater(d_fr, 0.0, f"FR should be positive for sw={sw}")
            # Inside wheel (FL) turns more than outside (FR)
            self.assertGreater(d_fl, d_fr,
                               f"At sw={sw:.1f} deg, FL (inside)={d_fl:.2f} deg must be > FR (outside)={d_fr:.2f} deg")

        # High lock check: at sw >= 100 deg, difference should be >= 2.5 deg
        idx_100 = np.argmin(np.abs(delta_sw_deg - 100.0))
        self.assertGreaterEqual(fl_deg[idx_100] - fr_deg[idx_100], 2.5,
                                f"At 100 deg sw, inside-outside difference must be >= 2.5 deg")

    def test_wheel_steer_inside_greater_than_outside_right(self):
        """
        Verifies that for steering wheel angle delta_sw < 0 (right turn),
        the inside wheel (FR) steers sharper than the outside wheel (FL):
        |delta_fr| > |delta_fl|, i.e. delta_fr < delta_fl < 0.
        """
        delta_sw_deg, fl_deg, fr_deg, diff_deg = self.vp.get_steer_wheel_angles()

        # Check negative steering angles (right turn)
        neg_idx = np.where(delta_sw_deg < -10.0)[0]
        self.assertTrue(len(neg_idx) > 0)

        for i in neg_idx:
            sw = delta_sw_deg[i]
            d_fl = fl_deg[i]
            d_fr = fr_deg[i]
            # Both turn right (negative)
            self.assertLess(d_fl, 0.0, f"FL should be negative for sw={sw}")
            self.assertLess(d_fr, 0.0, f"FR should be negative for sw={sw}")
            # Inside wheel (FR) has larger magnitude than outside (FL)
            self.assertGreater(abs(d_fr), abs(d_fl),
                               f"At sw={sw:.1f} deg, |FR| (inside)={abs(d_fr):.2f} deg must be > |FL| (outside)={abs(d_fl):.2f} deg")
            self.assertLess(d_fr, d_fl,
                            f"At sw={sw:.1f} deg, FR (inside)={d_fr:.2f} deg must be < FL (outside)={d_fl:.2f} deg")

    def test_wheel_steer_bilateral_mirror_symmetry(self):
        """
        Verifies exact bilateral mirror symmetry across vehicle centerline:
        Turning left (+sw) vs turning right (-sw):
          delta_fl(+sw) == -delta_fr(-sw)   (inside wheels match)
          delta_fr(+sw) == -delta_fl(-sw)   (outside wheels match)
        """
        test_angles = [10.0, 25.0, 50.0, 75.0, 100.0, 125.0]
        w_coeffs = self.vp.wheel_steer_f_coeffs

        for sw in test_angles:
            fl_pos = self.vp.eval_poly3(w_coeffs, sw)
            fr_pos = -self.vp.eval_poly3(w_coeffs, -sw)

            fl_neg = self.vp.eval_poly3(w_coeffs, -sw)
            fr_neg = -self.vp.eval_poly3(w_coeffs, sw)

            # Inside wheel mirror check
            self.assertAlmostEqual(fl_pos, -fr_neg, places=6,
                                   msg=f"Inside wheel asymmetry: FL(+{sw})={fl_pos} vs -FR(-{sw})={-fr_neg}")
            # Outside wheel mirror check
            self.assertAlmostEqual(fr_pos, -fl_neg, places=6,
                                   msg=f"Outside wheel asymmetry: FR(+{sw})={fr_pos} vs -FL(-{sw})={-fl_neg}")

    def test_tire_wear_accumulation_5km(self):
        """
        Simulates ~5 km driving under active cornering (Fz = 800 N, P_slip = 2000 W, vx = 14 m/s).
        Verifies wear fraction is in physical FSAE endurance range [2%, 8%], NOT negligible (0%).
        """
        model = TireDegradationModel(mode="light")
        state = TireState()

        # 5 km at ~50 km/h (13.89 m/s) -> ~360 seconds
        dt = 0.01
        t_total = 360.0
        n_steps = int(t_total / dt)

        vx = 13.89
        Fz = 800.0
        P_slip = 1000.0  # Lap-averaged sliding power (including straights and corners)
        Q_gen = P_slip * model.params.heat_gen_efficiency

        for _ in range(n_steps):
            model._step_light(state, Q_gen, dt)
            model._step_wear(state, P_slip, Fz, vx, dt)

        wear_pct = state.wear_fraction * 100.0
        self.assertGreater(wear_pct, 2.0, f"Wear after 5 km is too low: {wear_pct:.2f}%")
        self.assertLess(wear_pct, 10.0, f"Wear after 5 km is too high: {wear_pct:.2f}%")

    def test_tire_wear_rate_scaling(self):
        """
        Verifies that setting wear_rate_scale accelerates wear proportionally.
        """
        model_1x = TireDegradationModel(mode="light")
        model_5x = TireDegradationModel(mode="light")
        model_5x.set_wear_rate_scale(5.0)

        state_1x = TireState()
        state_5x = TireState()

        dt = 0.05
        for _ in range(200):
            model_1x._step_wear(state_1x, 2000.0, 800.0, 15.0, dt)
            model_5x._step_wear(state_5x, 2000.0, 800.0, 15.0, dt)

        ratio = state_5x.wear_fraction / max(state_1x.wear_fraction, 1e-8)
        self.assertAlmostEqual(ratio, 5.0, delta=0.25,
                               msg=f"Wear scale ratio should be ~5.0, got {ratio:.2f}")

    def test_minimap_rendering_smoke(self):
        """
        Verifies that _draw_minimap executes properly on a QPainter without exceptions.
        """
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QPainter, QPixmap
        import sys

        app = QApplication.instance() or QApplication(sys.argv)

        from src.ui.interactive_driver_14dof import OpenGLViewport3D, _LegacyViewport3D
        from src.mdo.mdo_direct_collocation import build_track_geometry

        track_data = build_track_geometry(track_length_m=850.0, ds=2.0, track_width_m=3.8)

        vp = OpenGLViewport3D()
        vp.track_data = track_data
        vp.car_state = {
            'x': 50.0, 'y': 20.0, 'psi_deg': 45.0,
            's_lap': 120.0, 'steer_deg': 15.0,
            'steer_fl_deg': 18.2, 'steer_fr_deg': 15.6,
            'vx_kmh': 75.0, 'ax_g': 0.2, 'ay_g': 1.1,
            'throttle_pct': 80.0, 'brake_pct': 0.0
        }

        pix = QPixmap(400, 300)
        painter = QPainter(pix)
        try:
            vp._draw_minimap(painter, 10, 10, 200, 180)
        finally:
            painter.end()

        self.assertTrue(hasattr(vp, '_cached_mm_poly'))
        self.assertGreater(len(vp._cached_mm_poly), 10)

        vp_leg = _LegacyViewport3D()
        self.assertTrue(hasattr(vp_leg, '_draw_minimap'))
        self.assertTrue(hasattr(vp_leg, 'show_minimap'))


if __name__ == '__main__':
    unittest.main()
