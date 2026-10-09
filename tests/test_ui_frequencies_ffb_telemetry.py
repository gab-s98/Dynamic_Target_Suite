import os
import sys
import unittest
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5.QtWidgets import QApplication

class TestUIFrequenciesFFBTelemetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_ffb_presets(self):
        from src.ui.interactive_driver_14dof import FFB_PRESETS
        expected_presets = [
            "Bilanciato (Consigliato)",
            "Morbido / Confortevole",
            "Reattivo / Sportivo",
            "Puro Grip (Solo Limite Gomme)",
            "Disattivato"
        ]
        for p in expected_presets:
            self.assertIn(p, FFB_PRESETS)
            cfg = FFB_PRESETS[p]
            self.assertIn('ffb_gain', cfg)
            self.assertIn('road_feel', cfg)
            self.assertIn('grip_loss', cfg)
            self.assertIn('trigger_haptics', cfg)
            self.assertIn('deadzone', cfg)
            self.assertIn('sensitivity', cfg)

    def test_road_profile_editor_dialog_autobank(self):
        from src.ui.interactive_driver_14dof import RoadProfileEditorDialog
        s_arr = np.linspace(0, 500, 100)
        track_data = {
            's_arr': s_arr,
            'x_center': 100 * np.cos(s_arr / 500 * 2 * np.pi),
            'y_center': 100 * np.sin(s_arr / 500 * 2 * np.pi),
            'kappa_arr': np.full_like(s_arr, 0.02),
            'total_length': 500.0,
            'track_width': 4.0
        }
        dlg = RoadProfileEditorDialog(track_data=track_data, current_stations=None)
        self.assertTrue(hasattr(dlg, 'btn_autobank'))
        self.assertTrue(hasattr(dlg, 'ax_map'))
        self.assertTrue(hasattr(dlg, 'ax_z'))
        self.assertTrue(hasattr(dlg, 'ax_bank'))
        
        # Test auto_bank_from_curvature
        dlg.auto_bank_from_curvature()
        stations = dlg.stations
        self.assertGreaterEqual(len(stations), 4)
        max_bank = max(abs(st['bank_deg']) for st in stations)
        self.assertGreater(max_bank, 0.01)

    def test_full_telemetry_window_distance_and_laps(self):
        from src.ui.interactive_driver_14dof import FullTelemetryWindow
        n_pts = 300
        t = np.linspace(0, 30, n_pts)
        dist = t * 20.0
        s_lap = dist % 200.0
        lap_idx = (dist // 200.0).astype(int)

        mock_data = []
        for i in range(n_pts):
            mock_data.append({
                't': float(t[i]),
                'dist_total': float(dist[i]),
                's_lap': float(s_lap[i]),
                'lap_idx': int(lap_idx[i]),
                'lap_count': int(lap_idx[i]),
                'lap_time_last': 10.0 if lap_idx[i] > 0 else 0.0,
                'vx_kmh': float(70.0 + 10.0 * np.sin(t[i] * 0.4)),
                'vy_kmh': float(1.5 * np.cos(t[i] * 0.4)),
                'beta_deg': float(1.0 * np.sin(t[i] * 0.2)),
                'ay_g': float(1.2 * np.sin(t[i] * 0.3)),
                'ax_g': float(0.4 * np.cos(t[i] * 0.3)),
                'steer_deg': float(4.0 * np.sin(t[i] * 0.4)),
                'throttle_pct': 75.0,
                'brake_pct': 0.0,
                'z_fl_mm': 12.0, 'z_fr_mm': 11.5, 'z_rl_mm': 15.0, 'z_rr_mm': 14.8,
                'camber_fl_deg': -2.5, 'camber_fr_deg': -2.5, 'camber_rl_deg': -1.8, 'camber_rr_deg': -1.8,
                'toe_fl_deg': 0.1, 'toe_fr_deg': 0.1, 'toe_rl_deg': 0.2, 'toe_rr_deg': 0.2,
                'Fz_fl': 800.0, 'Fz_fr': 800.0, 'Fz_rl': 900.0, 'Fz_rr': 900.0, 'dFz_jack_N': 30.0,
                'Fy_fl': 500.0, 'Fy_fr': 500.0, 'Fy_rl': 600.0, 'Fy_rr': 600.0,
                'alpha_fl': 3.5, 'alpha_fr': 3.5, 'alpha_rl': 3.0, 'alpha_rr': 3.0,
                'kappa_fl': 2.0, 'kappa_fr': 2.0, 'kappa_rl': 3.0, 'kappa_rr': 3.0,
                'util_fl': 75.0, 'util_fr': 75.0, 'util_rl': 80.0, 'util_rr': 80.0,
                'phi_deg': 1.5, 'theta_deg': 0.8, 'r_deg_s': 10.0,
                'haptic_throt_rt': 10.0, 'haptic_brake_lt': 0.0, 'rumble_heavy': 10.0, 'rumble_light': 5.0,
                'h_rc_f_mm': 40.0, 'h_rc_r_mm': 50.0,
                'tire_T_bulk_fl': 55.0, 'tire_T_bulk_fr': 55.0, 'tire_T_bulk_rl': 60.0, 'tire_T_bulk_rr': 60.0,
                'tire_T_surf_fl': 62.0, 'tire_T_tread_fl': 58.0, 'tire_T_carc_fl': 54.0, 'tire_T_surf_rr': 63.0,
                'tire_grip_fl': 0.99, 'tire_grip_rr': 0.98, 'tire_wear_fl': 0.8, 'tire_wear_rr': 1.0
            })

        dlg = FullTelemetryWindow(mock_data)
        self.assertTrue(hasattr(dlg, 'combo_x_axis'))
        self.assertTrue(hasattr(dlg, 'combo_lap_filter'))
        self.assertTrue(hasattr(dlg, 'lbl_lap_stats'))
        
        # Test lap extraction
        laps, best_id, stats = dlg._extract_laps()
        self.assertGreaterEqual(len(laps), 2)
        self.assertIsNotNone(best_id)

        # Test distance mode
        dlg.combo_x_axis.setCurrentIndex(0) # s_lap
        dlg._render_current_view()
        
        # Test overlay mode
        dlg.combo_lap_filter.setCurrentIndex(1) # overlay
        dlg._render_current_view()

        # Test Best Lap filter
        dlg.combo_lap_filter.setCurrentIndex(2) # best
        dlg._render_current_view()

        # Test single focus
        dlg.combo_focus.setCurrentIndex(1) # Focused plot
        dlg._render_current_view()

    def test_interactive_window_frequencies_and_ffb_ui(self):
        from src.ui.interactive_driver_14dof import Interactive14DOFDrivingWindow
        from src.models.params_vehicle import VehicleParams

        vp = VehicleParams()
        win = Interactive14DOFDrivingWindow(vp)
        try:
            self.assertTrue(hasattr(win, 'combo_hz'))
            self.assertTrue(hasattr(win, 'combo_thermal_hz'))
            self.assertTrue(hasattr(win, 'lbl_freq_status'))
            self.assertTrue(hasattr(win, 'combo_ffb_presets'))
            self.assertTrue(hasattr(win, 'btn_save_ffb'))
            self.assertTrue(hasattr(win, 'btn_load_ffb'))
            self.assertGreaterEqual(win.combo_hz.count(), 4)
            self.assertGreaterEqual(win.combo_thermal_hz.count(), 4)
            self.assertGreaterEqual(win.combo_ffb_presets.count(), 5)
        finally:
            win.physics_thread.running = False
            win.physics_thread.wait(500)

    def test_trigger_haptics_and_test_button(self):
        from src.ui.interactive_driver_14dof import GamepadManager, XINPUT_VIBRATION_EX
        gp = GamepadManager()
        self.assertTrue(hasattr(gp, 'trigger_gain'))
        self.assertTrue(hasattr(gp, 'enable_trigger_haptics'))
        self.assertTrue(hasattr(gp, 'test_trigger_vibration'))

        # Structure definition
        vib_ex = XINPUT_VIBRATION_EX(30000, 20000, 65535, 45000)
        self.assertEqual(vib_ex.wLeftMotorSpeed, 30000)
        self.assertEqual(vib_ex.wRightMotorSpeed, 20000)
        self.assertEqual(vib_ex.wLeftTriggerSpeed, 65535)
        self.assertEqual(vib_ex.wRightTriggerSpeed, 45000)

        # Vibration without crash
        gp.set_vibration(rumble_heavy=0.5, rumble_light=0.4, trigger_left=0.8, trigger_right=0.9)
        gp.test_trigger_vibration()

    def test_lap_selection_dialog_and_export(self):
        from src.ui.interactive_driver_14dof import FullTelemetryWindow, LapSelectionDialog
        import tempfile
        import pandas as pd

        mock_data = []
        for i in range(120):
            lap_idx = 0 if i < 40 else (1 if i < 80 else 2)
            mock_data.append({
                't': i * 0.1,
                'lap_idx': lap_idx,
                's_lap': float((i % 40) * 10),
                'dist_total': float(i * 10),
                'vx_kmh': 80.0,
                'ay_g': 1.0,
                'ax_g': 0.5,
                'z_fl_mm': 2.0, 'z_fr_mm': 2.0, 'z_rl_mm': 1.0, 'z_rr_mm': 1.0,
                'camber_fl_deg': -2.0, 'camber_fr_deg': -2.0, 'camber_rl_deg': -1.5, 'camber_rr_deg': -1.5,
                'toe_fl_deg': 0.1, 'toe_fr_deg': 0.1, 'toe_rl_deg': 0.0, 'toe_rr_deg': 0.0,
                'Fz_fl': 1200.0, 'Fz_fr': 1200.0, 'Fz_rl': 1300.0, 'Fz_rr': 1300.0,
                'Fy_fl': 800.0, 'Fy_fr': 800.0, 'Fy_rl': 900.0, 'Fy_rr': 900.0,
                'alpha_fl': 2.0, 'alpha_fr': 2.0, 'alpha_rl': 1.5, 'alpha_rr': 1.5,
                'kappa_fl': 0.01, 'kappa_fr': 0.01, 'kappa_rl': 0.02, 'kappa_rr': 0.02,
                'util_fl': 60.0, 'util_fr': 60.0, 'util_rl': 70.0, 'util_rr': 70.0,
                'steer_deg': 5.0, 'throttle_pct': 80.0, 'brake_pct': 0.0,
                'phi_deg': 1.0, 'theta_deg': 0.5, 'r_deg_s': 8.0,
                'haptic_throt_rt': 0.0, 'haptic_brake_lt': 0.0, 'rumble_heavy': 10.0, 'rumble_light': 5.0,
                'h_rc_f_mm': 40.0, 'h_rc_r_mm': 50.0,
                'tire_T_bulk_fl': 55.0, 'tire_T_bulk_fr': 55.0, 'tire_T_bulk_rl': 60.0, 'tire_T_bulk_rr': 60.0,
                'tire_T_surf_fl': 62.0, 'tire_T_tread_fl': 58.0, 'tire_T_carc_fl': 54.0, 'tire_T_surf_rr': 63.0,
                'tire_grip_fl': 0.99, 'tire_grip_rr': 0.98, 'tire_wear_fl': 0.8, 'tire_wear_rr': 1.0,
                'lap_time_last': 4.0
            })

        dlg = FullTelemetryWindow(mock_data)
        self.assertTrue(hasattr(dlg, 'btn_select_laps'))
        self.assertTrue(hasattr(dlg, 'btn_export_menu'))
        self.assertTrue(hasattr(dlg, 'selected_laps'))

        lap_dlg = LapSelectionDialog(dlg.laps, dlg.best_lap_idx, dlg.lap_stats, initially_selected=dlg.selected_laps, parent=dlg)
        self.assertEqual(len(lap_dlg.checkboxes), 3)
        lap_dlg._deselect_all()
        lap_dlg._select_best_and_last()
        self.assertTrue(lap_dlg.checkboxes[dlg.best_lap_idx].isChecked())

        # Test overlay_selected mode
        dlg.combo_lap_filter.setCurrentIndex(1) # overlay_selected
        dlg._render_current_view()

        # Test figure saving
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_fig_path = os.path.join(tmp_dir, "test_chart.png")
            dlg.fig.savefig(test_fig_path, dpi=100)
            self.assertTrue(os.path.exists(test_fig_path))

        dlg.close()

    def test_cockpit_camera_mode_and_controls(self):
        from src.ui.interactive_driver_14dof import get_viewport_class, Interactive14DOFDrivingWindow
        from src.models.params_vehicle import VehicleParams

        ViewportClass = get_viewport_class()
        vp = ViewportClass()
        self.assertTrue(hasattr(vp, 'camera_mode'))
        self.assertEqual(vp.camera_mode, "CHASE")
        vp.camera_mode = "COCKPIT"
        self.assertEqual(vp.camera_mode, "COCKPIT")

        # Verify zero-lag lockstep: smooth coordinates snap exactly to car position
        vp.update_vehicle_state({
            'x': 25.0,
            'y': 10.0,
            'psi_deg': 45.0,
            'theta_deg': 0.0,
            'phi_deg': 0.0,
            'z_road': 0.0,
            'grade_road': 0.0,
            'bank_road_deg': 0.0
        })
        self.assertEqual(vp.smooth_x, 25.0)
        self.assertEqual(vp.smooth_y, 10.0)

        veh_p = VehicleParams()
        win = Interactive14DOFDrivingWindow(veh_p)
        try:
            self.assertTrue(hasattr(win, 'btn_test_triggers'))
            self.assertTrue(hasattr(win, 'sb_trigger_gain'))
            # Test camera mode index 1 (Cockpit)
            win.combo_cam.setCurrentIndex(1)
            self.assertEqual(win.viewport.camera_mode, "COCKPIT")
            self.assertEqual(win.viewport.view_mode, "3D")
        finally:
            win.physics_thread.running = False
            win.physics_thread.wait(500)


if __name__ == '__main__':
    unittest.main()

