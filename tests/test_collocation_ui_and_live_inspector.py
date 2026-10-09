"""
test_collocation_ui_and_live_inspector.py
-----------------------------------------
Unit tests verifying:
1. Decoupled CollocationTrajectoryCanvas and TrackConfigPanel.
2. Telemetry Data Inspector multi-plot canvas with channel selection and derived channels.
3. DirectCollocationDialog headless initialization and UI components.
4. LiveVehicleParametersInspector non-modal floating window, live tuning, and telemetry streaming.
5. First-person camera smoothing and altimetric damping logic in OpenGLViewport3D.
"""

import unittest
import sys
import os
import math
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists for headless GUI tests
app = QApplication.instance()
if app is None:
    app = QApplication([sys.argv[0], "-platform", "offscreen"])

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_direct_collocation import build_track_geometry
from src.ui.ui_direct_collocation import (
    CollocationTrajectoryCanvas, DataInspectorCanvas,
    TrackConfigPanel, DirectCollocationDialog
)
from src.ui.vehicle_parameters_inspector_dialog import LiveVehicleParametersInspector
from src.ui.sim_viewport import OpenGLViewport3D


class TestCollocationUIAndInspector(unittest.TestCase):
    """Verifies direct collocation canvas decoupling, data inspector, and live tuning inspector."""

    def setUp(self):
        self.vp = VehicleParams.nominal_fsae_params()
        self.track = build_track_geometry(track_type="fsae_skidpad", ds=5.0, track_width_m=3.5)

    def test_collocation_trajectory_canvas(self):
        """Verifies CollocationTrajectoryCanvas empty state, preview, and results plotting."""
        canvas = CollocationTrajectoryCanvas(theme_name="dark")
        self.assertIsNotNone(canvas.fig)
        self.assertEqual(len(canvas.axes), 5)  # 4 subplots + 1 twinx

        # Preview track
        canvas.plot_track_preview(self.track)
        self.assertEqual(canvas.ax_track.get_title(), f"Anteprima Tracciato: fsae_skidpad (L={self.track['s'][-1]:.1f} m, N={self.track['N']})")

        # Mock optimization results
        n = self.track['N']
        df_mock = pd.DataFrame({
            's_m': self.track['s'],
            'x_opt': self.track['x_center'] + 0.2,
            'y_opt': self.track['y_center'] - 0.1,
            'v_kmh': np.linspace(40, 80, n),
            'v_profiler_kmh': np.linspace(35, 75, n),
            'ay_g': np.full(n, 1.2),
            'ay_profiler_g': np.full(n, 1.1),
            'ax_g': np.full(n, 0.4),
            'ax_profiler_g': np.full(n, 0.3),
            'n_m': np.full(n, 0.15),
            'steer_deg': np.full(n, 5.0),
            't_s': np.linspace(0, 10, n),
        })
        res_mock = {
            'telemetry': df_mock,
            'profiler': {'lap_time_s': 10.5, 'v_kmh': df_mock['v_profiler_kmh']},
            'track': self.track,
            'lap_time_s': 9.8,
            'delta_time_s': -0.7,
            'delta_time_pct': -6.6,
            'v_max_kmh': 80.0,
            'v_avg_kmh': 60.0,
            'max_ay_g': 1.2,
            'max_lateral_dev_m': 0.15,
            'model_type': '14-DOF'
        }
        canvas.plot_optimization_results(res_mock)
        self.assertIn("Traiettoria Ottimale", canvas.ax_track.get_title())

        # Test theme change
        canvas.apply_theme("light")
        self.assertEqual(canvas.theme_name, "light")

    def test_telemetry_data_inspector_canvas(self):
        """Verifies DataInspectorCanvas multi-plot, channel filtering, and derived channels."""
        canvas = DataInspectorCanvas(theme_name="dark")
        self.assertEqual(canvas.layout_mode, "2x1")
        self.assertEqual(len(canvas.subplots), 2)

        # Test switching layout mode
        canvas.set_layout_mode("2x2")
        self.assertEqual(len(canvas.subplots), 4)

        canvas.set_layout_mode("1x1")
        self.assertEqual(len(canvas.subplots), 1)

        # Create sample dataframe
        n = 100
        df = pd.DataFrame({
            's_m': np.linspace(0, 500, n),
            't_s': np.linspace(0, 20, n),
            'v_kmh': np.linspace(30, 90, n),
            'v_mps': np.linspace(30/3.6, 90/3.6, n),
            'v_profiler_kmh': np.linspace(28, 85, n),
            'ax_g': np.sin(np.linspace(0, np.pi, n)) * 0.8,
            'ay_g': np.cos(np.linspace(0, np.pi, n)) * 1.4,
            'steer_deg': np.full(n, 3.5),
        })
        canvas.set_data(df)

        # Check derived channels computed
        self.assertIn('atot_g', canvas.df_telemetry.columns)
        self.assertIn('delta_v_kmh', canvas.df_telemetry.columns)
        self.assertIn('yaw_rate_degs', canvas.df_telemetry.columns)
        self.assertIn('power_long_kw', canvas.df_telemetry.columns)

        # Assign channels to plot #0
        canvas.set_plot_channels(0, ['v_kmh', 'v_profiler_kmh', 'delta_v_kmh'])
        self.assertEqual(canvas.active_channels[0], ['v_kmh', 'v_profiler_kmh', 'delta_v_kmh'])

        # Switch X axis to time
        canvas.set_x_mode("t")
        self.assertEqual(canvas.x_mode, "t")

    def test_track_config_panel(self):
        """Verifies track preset and parameter configuration panel."""
        panel = TrackConfigPanel(theme_name="dark")
        t_data = panel.get_track_data()
        self.assertIsNotNone(t_data)
        self.assertIn('s', t_data)
        self.assertIn('track_width', t_data)

        # Test width and ds adjustments
        panel.spin_width.setValue(4.2)
        panel.spin_ds.setValue(3.0)
        t_updated = panel.get_track_data()
        self.assertAlmostEqual(t_updated['track_width'], 4.2)
        self.assertAlmostEqual(t_updated['ds'], 3.0)

        # Test collocation scheme
        scheme = panel.get_collocation_scheme()
        self.assertIn(scheme, ["radau_3", "legendre_3", "radau_2", "legendre_2"])

    def test_direct_collocation_dialog_headless(self):
        """Verifies DirectCollocationDialog initializes cleanly in headless mode."""
        dlg = DirectCollocationDialog(vehicle_params=self.vp, theme_name="dark")
        self.assertIsNotNone(dlg.track_panel)
        self.assertIsNotNone(dlg.canvas_overview)
        self.assertIsNotNone(dlg.canvas_inspector)
        self.assertEqual(dlg.view_tabs.count(), 3)
        dlg.close()

    def test_live_vehicle_parameters_inspector(self):
        """Verifies LiveVehicleParametersInspector UI, live tuning inputs, and telemetry update."""
        inspector = LiveVehicleParametersInspector(vp=self.vp, theme_name="dark")
        self.assertIsNotNone(inspector.tabs)
        self.assertEqual(inspector.tabs.count(), 7)

        # Verify initial values loaded
        self.assertAlmostEqual(inspector.spin_k_spring_f.value(), getattr(self.vp, 'spring_rate_installed_f_Nmm', 59.5), places=1)
        self.assertAlmostEqual(inspector.spin_cz.value(), getattr(self.vp, 'Cz', 2.8), places=1)

        # Test modifying live parameter
        inspector.spin_cz.setValue(3.2)
        self.assertAlmostEqual(inspector.spin_cz.value(), 3.2)

        # Test nominal setup reset
        vp_nom = VehicleParams.nominal_fsae_params()
        inspector._reset_to_nominal_setup()
        self.assertAlmostEqual(inspector.spin_cz.value(), vp_nom.Cz, places=1)
        inspector.close()

    def test_first_person_camera_smoothing_in_viewport(self):
        """Verifies continuous first-person camera smoothing variables and reset in viewport."""
        viewport = OpenGLViewport3D()
        self.assertIsNone(viewport.onboard_cam_pos)
        self.assertIsNone(viewport.onboard_fwd_vec)
        self.assertIsNone(viewport.onboard_up_vec)

        # Test reset clears states
        viewport.onboard_cam_pos = np.array([10.0, 5.0, 0.5])
        viewport.onboard_theta_smooth = 0.05
        viewport.reset_camera_smoothing()
        self.assertIsNone(viewport.onboard_cam_pos)
        self.assertIsNone(viewport.onboard_theta_smooth)


if __name__ == '__main__':
    unittest.main()
