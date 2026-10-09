"""
test_vehicle_dna_maps.py
------------------------
Unit tests for Vehicle DNA Stability Maps and Understeer definition.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from src.utils.vehicle_dna_maps import compute_vehicle_dna_fields, plot_vehicle_dna_figures


class TestVehicleDNAMaps(unittest.TestCase):

    def test_dna_fields_computation(self):
        """Tests calculation of steady-state gradients and 6 DNA maps n1...n6."""
        res = compute_vehicle_dna_fields(v_x=15.0, wd_f=0.5, lltd=0.5, delta_max_deg=15.0, ay_max_g=2.0)
        
        # Verify matrices are non-empty and have correct shapes
        self.assertIn('N1', res)
        self.assertIn('N2', res)
        self.assertIn('N3', res)
        self.assertIn('N4', res)
        self.assertIn('N5', res)
        self.assertIn('N6', res)
        self.assertIn('K_us_gradient_degg', res)
        self.assertIn('Rho_y', res)
        
        # Verify D < 0 everywhere (fundamental consistency of bicycle / handling plane)
        D = res['D']
        self.assertTrue(np.all(D < 0), "Denominator determinant D must be negative for valid handling plane.")
        
        # Verify control derivatives are positive
        self.assertGreater(res['Y_hat_delta'], 0.0)
        self.assertGreater(res['N_hat_delta'], 0.0)

    def test_understeer_criterion(self):
        """Tests that Kus and rho_y criterion match theoretical limits."""
        res = compute_vehicle_dna_fields(v_x=12.0)
        
        Kus = res['K_us_gradient_degg']
        Rho_y = res['Rho_y']
        u = res['v_x']
        thresh = 1.0 / (u**2)
        
        # In linear regime (small delta, small ay), vehicle is understeering (Kus > 0, rho_y < 1/u^2)
        self.assertGreater(Kus[2, 2], 0.0, "Linear low-slip regime must be understeering.")
        self.assertLess(Rho_y[2, 2], thresh, "rho_y must be below 1/u^2 threshold in understeering regime.")

    def test_dna_figures_rendering(self):
        """Tests matplotlib rendering for both DNA eigen and Understeer gradient views."""
        res = compute_vehicle_dna_fields(v_x=18.0)
        fig = Figure(figsize=(8, 6))
        
        # Render View 1: 4 Subplots DNA & Eigen
        plot_vehicle_dna_figures(fig, res, view_mode='dna_eigen')
        self.assertEqual(len(fig.axes), 7) # 4 subplots + 3 horizontal colorbars
        
        # Render View 2: 2 Subplots Understeer gradient
        fig.clear()
        plot_vehicle_dna_figures(fig, res, view_mode='understeer_gradient')
        self.assertEqual(len(fig.axes), 3) # 2 subplots + 1 colorbar

    def test_complete_maps_generation(self):
        """Tests generate_complete_maps execution and output structure."""
        from src.mdo.mdo_target_optimization import generate_complete_maps
        res = generate_complete_maps(v_x=12.0, n_speeds=3, n_points=10)
        self.assertIn('speed_curves', res)
        self.assertIn('envelope_df', res)
        self.assertIn('delta_slices', res)
        self.assertIn('iso_ay_curves', res)
        self.assertIn('iso_rho_curves', res)
        self.assertIn('iso_beta_curves', res)
        self.assertGreater(len(res['speed_curves']), 0)
        self.assertFalse(res['envelope_df'].empty)


if __name__ == '__main__':
    unittest.main()
