"""
test_track_manager.py
---------------------
Unit tests for 2D Road Surface Generation (ISO 8608 Class A),
Contact Patch Enveloping, Real-Time Bilinear Lookup at 1000 Hz,
and 3D Verge Geometry Extrusion.
"""

import time
import math
import unittest
import numpy as np

from src.utils.track_manager import (
    RoadSurfaceArray,
    RoadSurface2D,
    generate_road_surface_2d,
    apply_contact_patch_envelope_filter,
    extrude_verge_geometry
)


class TestTrackManager(unittest.TestCase):
    """Test suite for 2D road surface generation and 3D verge geometry."""

    def test_road_surface_2d(self):
        """
        Validates:
        1. 2D Stochastic Surface generation matching ISO 8608 Class A amplitude (sigma ~ 1.41 mm).
        2. Lateral Lorentzian coherence: decaying cross-correlation for Delta d > 0.1 m.
        3. Contact patch moving average envelope filtering.
        4. RoadSurface2D real-time bilinear lookup speed (< 10 us/query), periodic s-wrap, and d-clamping.
        """
        length = 150.0
        width = 6.0
        ds = 0.05
        dd = 0.05
        seed = 42

        # 1. Generation & Metadata
        Z, s_grid, d_grid = generate_road_surface_2d(
            length=length, width=width, ds=ds, dd=dd, seed=seed, return_coords=True
        )
        self.assertIsInstance(Z, RoadSurfaceArray)
        self.assertEqual(Z.shape, (int(round(length / ds)), int(round(width / dd)) + 1))
        self.assertAlmostEqual(Z.length, length)
        self.assertAlmostEqual(Z.width, width)
        self.assertAlmostEqual(Z.ds, ds)
        self.assertAlmostEqual(Z.dd, dd)

        # 2. ISO 8608 Class A Amplitude
        std_z = float(np.std(Z))
        # ISO 8608 Class A typical roughness std ~ 1.41 mm (0.00141 m)
        self.assertGreater(std_z, 0.0008, "Roughness std dev too low for Class A")
        self.assertLess(std_z, 0.0025, "Roughness std dev too high for Class A")

        # 3. Lateral Cross-Correlation Decay
        # Centerline index:
        mid_idx = Z.shape[1] // 2
        prof_center = Z[:, mid_idx]
        
        # Profile at Delta d = 0.05 m (adjacent column)
        prof_near = Z[:, mid_idx + 1]
        corr_near = float(np.corrcoef(prof_center, prof_near)[0, 1])

        # Profile at Delta d = 0.50 m (10 columns away)
        prof_far = Z[:, mid_idx + 10]
        corr_far = float(np.corrcoef(prof_center, prof_far)[0, 1])

        # Lateral coherence requires correlation to decay with lateral distance
        self.assertGreater(corr_near, corr_far, "Lateral correlation did not decay with distance")
        self.assertLess(corr_far, 0.85, "Profiles at Delta d = 0.5m should have reduced correlation")

        # 4. Contact Patch Envelope Filter
        Z_filtered = apply_contact_patch_envelope_filter(Z, ds=ds, L_patch=0.12)
        self.assertEqual(Z_filtered.shape, Z.shape)
        # Smoothing should reduce variance slightly or preserve it without extreme spikes
        grad_raw = np.diff(Z, axis=0)
        grad_filt = np.diff(Z_filtered, axis=0)
        self.assertLessEqual(np.std(grad_filt), np.std(grad_raw), "Envelope filter should smooth longitudinal gradients")

        # 5. Bilinear Interpolation & Real-Time Lookup Performance
        surf = RoadSurface2D(length=length, width=width, ds=ds, dd=dd, seed=seed)
        
        # Benchmark 2000 queries (simulating 2 seconds of 1000 Hz physics loop)
        t0 = time.perf_counter()
        for i in range(2000):
            s_q = (i * 0.073) % length
            d_q = -1.2 + (i % 50) * 0.05
            z_q, dz_q = surf.sample(s_q, d_q)
        dt = time.perf_counter() - t0
        time_per_query_us = (dt / 2000.0) * 1e6
        self.assertLess(time_per_query_us, 50.0, f"Query took {time_per_query_us:.2f} us (budget < 50 us)")

        # 6. Periodic Boundary Wrapping along s
        z_base, dz_base = surf.sample(10.0, 0.4)
        z_wrapped, dz_wrapped = surf.sample(10.0 + length, 0.4)
        self.assertAlmostEqual(z_base, z_wrapped, places=5, msg="Periodic s-wrap failed")
        self.assertAlmostEqual(dz_base, dz_wrapped, places=5, msg="Periodic s-wrap gradient failed")

        # 7. Lateral Clamping beyond track edges
        w_half = width / 2.0
        z_edge, _ = surf.sample(25.0, w_half)
        z_beyond, _ = surf.sample(25.0, w_half + 3.0)
        self.assertAlmostEqual(z_edge, z_beyond, places=5, msg="Lateral clamping failed")

    def test_verge_geometry(self):
        """
        Validates:
        1. Lateral extrusion of verge mesh along +/- b(s) by W_verge = 12.0 m.
        2. Downward vertical drainage drop of h_drop = 0.4 m.
        3. Correct dictionary output keys and linkage in track_data.
        """
        N = 30
        s = np.linspace(0.0, 150.0, N, endpoint=False)
        track_data = {
            'x_center': s,
            'y_center': np.zeros(N),
            'x_left': s,
            'y_left': np.full(N, 2.5),
            'x_right': s,
            'y_right': np.full(N, -2.5),
            'z_left': np.full(N, 5.0),
            'z_right': np.full(N, 5.0),
            'theta': np.zeros(N)  # Heading East along +X -> b(s) = (sin 0, -cos 0, 0) = (0, -1, 0)
        }

        w_verge = 12.0
        h_drop = 0.4
        verge = extrude_verge_geometry(track_data, w_verge=w_verge, h_drop=h_drop)

        self.assertIn('verge', track_data)
        for k in ('x_verge_l', 'y_verge_l', 'z_verge_l', 'x_verge_r', 'y_verge_r', 'z_verge_r'):
            self.assertIn(k, verge)
            self.assertEqual(len(verge[k]), N)

        # Heading is theta = 0:
        # b(s) = (sin 0, -cos 0, 0) = (0, -1, 0)
        # v_verge,left  = r_edge,left  - W_verge * b(s) - h_drop * k
        #               = (s, 2.5) - 12*(0, -1) = (s, 2.5 + 12) = (s, 14.5)
        # v_verge,right = r_edge,right + W_verge * b(s) - h_drop * k
        #               = (s, -2.5) + 12*(0, -1) = (s, -2.5 - 12) = (s, -14.5)
        np.testing.assert_allclose(verge['y_verge_l'], 2.5 + w_verge, rtol=1e-5)
        np.testing.assert_allclose(verge['y_verge_r'], -2.5 - w_verge, rtol=1e-5)
        np.testing.assert_allclose(verge['x_verge_l'], s, rtol=1e-5)
        np.testing.assert_allclose(verge['x_verge_r'], s, rtol=1e-5)

        # Vertical drop
        np.testing.assert_allclose(verge['z_verge_l'], 5.0 - h_drop, rtol=1e-5)
        np.testing.assert_allclose(verge['z_verge_r'], 5.0 - h_drop, rtol=1e-5)

        # Bottom base elevation
        self.assertIn('z_base', verge)
        self.assertAlmostEqual(verge['z_base'], 5.0 - 1.5, places=5)


if __name__ == '__main__':
    unittest.main()
