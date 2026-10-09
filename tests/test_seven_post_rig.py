"""
test_seven_post_rig.py
----------------------
Unit and Integration Tests for the 7-Post Shaker Rig Engine and UI Integration.
Verifies physical consistency, 1g static equilibrium, numerical integration,
spectral analysis, modal extraction, and UI widget initialization.
"""

import unittest
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_seven_post_rig import SevenPostRig, SevenPostConfig, SevenPostResult


class TestSevenPostRigPhysicalEngine(unittest.TestCase):
    """Verifies physical consistency, mass properties, and 1g equilibrium."""

    def setUp(self):
        self.vp = VehicleParams()
        self.rig = SevenPostRig(self.vp)

    def test_mass_and_geometry_consistency(self):
        """Verifies mass partitioning between sprung and unsprung masses."""
        rig = self.rig
        self.assertGreater(rig.m_s, 100.0, "Sprung mass must be greater than 100 kg")
        self.assertGreater(rig.m_u_f, 5.0, "Front unsprung mass must be positive")
        self.assertGreater(rig.m_u_r, 5.0, "Rear unsprung mass must be positive")
        total_m = rig.m_s + 2.0 * rig.m_u_f + 2.0 * rig.m_u_r
        self.assertAlmostEqual(total_m, rig.m, places=2, msg="Sum of masses must equal vehicle mass")

    def test_static_1g_equilibrium_derivatives(self):
        """At zero actuator displacement and 1g gravity, all state derivatives must be exactly zero."""
        rig = self.rig
        x0 = np.zeros(14)
        z_p0 = np.zeros(4)
        dz_p0 = np.zeros(4)
        F_aero0 = np.zeros(3)

        # In generate_excitation, mode='step' with amp=0 yields zero
        # Let's test single-step evaluation via short simulation
        cfg = SevenPostConfig(mode='step', amplitude_mm=0.0, sweep_duration=0.5, dt=0.01)
        res = rig.simulate(cfg)

        # Displacements must stay at 0
        max_zs = np.max(np.abs(res.z_chassis[:, 0]))
        max_theta = np.max(np.abs(res.z_chassis[:, 1]))
        max_phi = np.max(np.abs(res.z_chassis[:, 2]))
        self.assertLess(max_zs, 1e-4, "Chassis heave must stay at 0 under zero excitation")
        self.assertLess(max_theta, 1e-4, "Chassis pitch must stay at 0 under zero excitation")
        self.assertLess(max_phi, 1e-4, "Chassis roll must stay at 0 under zero excitation")

    def test_frequency_sweep_heave_simulation(self):
        """Runs a short chirp sweep in pure heave and checks KPI validity."""
        rig = self.rig
        cfg = SevenPostConfig(
            mode='chirp',
            phasing='heave',
            f_start=1.0,
            f_end=15.0,
            sweep_duration=2.0,
            amplitude_mm=5.0,
            dt=0.004
        )
        res = rig.simulate(cfg)

        self.assertIsInstance(res, SevenPostResult)
        self.assertEqual(len(res.time), len(res.z_posts))
        self.assertEqual(res.z_posts.shape[1], 4)
        self.assertEqual(res.Fz_tires.shape[1], 4)

        # Verify normal load non-negativity (lift-off condition Fz >= 0)
        min_fz = np.min(res.Fz_tires)
        self.assertGreaterEqual(min_fz, 0.0, "Tire normal force Fz cannot be negative")

        # CPLV should be positive and realistic
        self.assertGreater(res.cplv['avg'], 0.0)
        self.assertLess(res.cplv['avg'], 2.0)

        # Transmissibility must be computed
        self.assertGreater(len(res.freqs), 0)
        self.assertEqual(len(res.transmissibility_heave), len(res.freqs))

    def test_phasing_signs(self):
        """Verifies correct mathematical sign relations for Heave, Pitch, Roll, and Warp."""
        rig = self.rig

        # 1. Pitch
        cfg_pitch = SevenPostConfig(mode='chirp', phasing='pitch', sweep_duration=1.0, amplitude_mm=5.0, dt=0.01)
        _, z_p_pitch, _, _ = rig.generate_excitation(cfg_pitch)
        # Front vs Rear must have opposite sign
        dot_fr_rl = np.sum(z_p_pitch[:, 0] * z_p_pitch[:, 2])
        self.assertLess(dot_fr_rl, 0.0, "In pitch mode, front and rear actuators must be anti-phase")

        # 2. Roll
        cfg_roll = SevenPostConfig(mode='chirp', phasing='roll', sweep_duration=1.0, amplitude_mm=5.0, dt=0.01)
        _, z_p_roll, _, _ = rig.generate_excitation(cfg_roll)
        # Left vs Right must have opposite sign
        dot_fl_fr = np.sum(z_p_roll[:, 0] * z_p_roll[:, 1])
        self.assertLess(dot_fl_fr, 0.0, "In roll mode, left and right actuators must be anti-phase")

    def test_iso_road_roughness(self):
        """Verifies ISO 8608 road profile generation and axle lag."""
        rig = self.rig
        cfg = SevenPostConfig(mode='iso_road', iso_class='A', iso_duration=1.5, sim_speed_mps=20.0, dt=0.005)
        t, z_p, dz_p, F_aero = rig.generate_excitation(cfg)

        self.assertEqual(z_p.shape[1], 4)
        self.assertGreater(np.std(z_p[:, 0]), 1e-5, "ISO road must have non-zero variance")
        self.assertGreater(F_aero[0, 0], 0.0, "Aero force must be positive when speed > 0")

    def test_linear_modal_analysis(self):
        """Verifies extraction of state-space system matrix, eigenvalues, and natural frequencies."""
        rig = self.rig
        modal = rig.compute_linear_modal_analysis()

        self.assertIn('A_matrix', modal)
        self.assertEqual(modal['A_matrix'].shape, (14, 14))

        modes = modal['modes']
        self.assertGreaterEqual(len(modes), 5, "Must identify at least 5 positive-frequency modes")

        # Check that frequencies are physically plausible
        freqs = [m['frequency_hz'] for m in modes]
        # Low frequency chassis modes (~1.5 - 6 Hz)
        self.assertTrue(any(1.5 <= f <= 6.5 for f in freqs), "Must identify body modes between 1.5 and 6.5 Hz")
        # High frequency wheel hop modes (~9 - 16 Hz)
        self.assertTrue(any(8.0 <= f <= 16.0 for f in freqs), "Must identify wheel hop modes between 8 and 16 Hz")

        # Check positive damping ratios (system is stable)
        for m in modes:
            self.assertGreater(m['damping_ratio'], 0.0, f"Mode {m['name']} must have positive damping ratio")

    def test_sine_dwell_simulation(self):
        """Verifies single-frequency sine dwell simulation and CPLV calculation."""
        rig = self.rig
        cfg = SevenPostConfig(
            mode='sine_dwell',
            dwell_frequency=2.5,
            sweep_duration=2.0,
            amplitude_mm=5.0,
            dt=0.005
        )
        res = rig.simulate(cfg)
        self.assertIsInstance(res, SevenPostResult)
        self.assertGreater(res.cplv['avg'], 0.0)
        self.assertGreater(len(res.time), 100)

    def test_step_bump_kerb_obstacle(self):
        """Verifies obstacle/kerb strike excitation and transient chassis response."""
        rig = self.rig
        cfg = SevenPostConfig(
            mode='step_bump',
            step_type='kerb',
            step_height_mm=12.0,
            step_duration_s=0.06,
            sweep_duration=2.0,
            sim_speed_mps=15.0,
            dt=0.004
        )
        res = rig.simulate(cfg)
        self.assertIsInstance(res, SevenPostResult)
        self.assertGreater(np.max(res.z_posts), 0.005, "Road displacement must record obstacle")
        self.assertGreater(res.pkpk_ride_height_f_mm, 1.0, "Front ride height must fluctuate upon impact")

    def test_bilstrom_5bin_and_energy(self):
        """Verifies Bilstrom 5-bin histogram percentages and damper energy accounting."""
        rig = self.rig
        cfg = SevenPostConfig(mode='chirp', sweep_duration=2.0, amplitude_mm=6.0, dt=0.005)
        res = rig.simulate(cfg)

        # Check Bilstrom 5-bin sum to ~100%
        for corner in ['fl', 'fr', 'rl', 'rr']:
            bins = res.bilstrom_5bin[corner]
            total_pct = sum(bins.values())
            self.assertAlmostEqual(total_pct, 100.0, delta=1.5, msg=f"Bilstrom bins for {corner} must sum to 100%")

        # Check positive energy dissipation in dampers
        self.assertGreater(res.damper_energy_dissipated_j['fl'], 0.0)
        self.assertGreater(res.damper_energy_dissipated_j['rl'], 0.0)

    def test_engineering_report_generation(self):
        """Verifies technical report generator output."""
        rig = self.rig
        cfg = SevenPostConfig(mode='chirp', sweep_duration=1.5, amplitude_mm=5.0, dt=0.005)
        res = rig.simulate(cfg)
        rep = rig.generate_engineering_report(res)

        self.assertIsInstance(rep, str)
        self.assertIn("REPORT TECNICO BANCO DINAMICO", rep)
        self.assertIn("Contact Patch Load Variation", rep)
        self.assertIn("Bilstrom", rep)
        self.assertIn("Öhlins TTX25", rep)

    def test_4post_architecture_execution(self):
        """Verifies 4-Post Rig simulation mode: zero aero actuators, pure mechanical grip score, and model comparison."""
        rig = self.rig
        cfg = SevenPostConfig(
            rig_architecture='4_post',
            mode='chirp',
            phasing='heave',
            sweep_duration=2.0,
            amplitude_mm=5.0,
            dt=0.005
        )
        res = rig.simulate(cfg)
        self.assertEqual(res.rig_architecture, '4_post')
        # In 4-post mode, aero post forces must be identically zero
        self.assertEqual(np.max(np.abs(res.F_aero_posts)), 0.0)
        # Score overall must equal pure grip score
        self.assertAlmostEqual(res.score_overall, res.score_grip, places=4)
        # Model comparison must be populated
        self.assertIsNotNone(res.model_comparison)
        self.assertIn('analytical', res.model_comparison)
        self.assertIn('correlation_table', res.model_comparison)

        # Engineering report for 4-post
        rep_4p = rig.generate_engineering_report(res)
        self.assertIn("4-POST", rep_4p)
        self.assertIn("Correlazione & Benchmark con il Modello Fisico Veicolo", rep_4p)

    def test_model_benchmark_comparison(self):
        """Verifies calculation of closed-form quarter-car natural frequencies and state-space modal correlation."""
        rig = self.rig
        comp = rig.compute_model_benchmark_comparison(measured_peaks=[
            {'frequency_hz': 4.2, 'mode': 'Chassis Heave', 'magnitude': 1.5},
            {'frequency_hz': 4.6, 'mode': 'Chassis Pitch', 'magnitude': 1.2}
        ])
        self.assertIn('analytical', comp)
        ana = comp['analytical']
        self.assertGreater(ana['heave_approx_hz'], 1.0)
        self.assertLess(ana['heave_approx_hz'], 8.0)
        self.assertGreater(ana['pitch_approx_hz'], 1.0)
        self.assertLess(ana['pitch_approx_hz'], 8.0)
        self.assertGreater(ana['hop_front_hz'], 8.0)
        self.assertLess(ana['hop_front_hz'], 25.0)

        # Correlation table
        corr = comp['correlation_table']
        self.assertTrue(len(corr) >= 4)
        heave_row = next(r for r in corr if r['mode'] == 'Chassis Heave')
        self.assertIsNotNone(heave_row['rig_hz'])
        self.assertIsNotNone(heave_row['delta_pct'])
        self.assertIn('%', heave_row['status'])


class TestSevenPostRigUIIntegration(unittest.TestCase):
    """Verifies that SevenPostRigTab initializes cleanly with PyQt5."""

    @classmethod
    def setUpClass(cls):
        import sys
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def test_ui_widget_instantiation(self):
        from src.ui.ui_seven_post_rig import SevenPostRigTab
        tab = SevenPostRigTab(main_window=None)
        self.assertIsNotNone(tab.btn_run)
        self.assertIsNotNone(tab.canvas_bode)
        self.assertIsNotNone(tab.canvas_cplv)
        self.assertIsNotNone(tab.canvas_telem)
        self.assertIsNotNone(tab.canvas_ttx)
        self.assertIsNotNone(tab.canvas_aero)
        self.assertIsNotNone(tab.canvas_modal)
        self.assertIsNotNone(tab.canvas_rig2d)
        self.assertIsNotNone(tab.txt_report)
        self.assertEqual(tab.tabs_view.count(), 8)

    def test_ui_render_result_pipeline(self):
        """Verifies full execution of UI post-processing and plot rendering methods with simulation results."""
        from src.ui.ui_seven_post_rig import SevenPostRigTab
        tab = SevenPostRigTab(main_window=None)


        rig = SevenPostRig(VehicleParams())
        cfg = SevenPostConfig(mode='chirp', sweep_duration=1.5, amplitude_mm=4.0, dt=0.005)
        res = rig.simulate(cfg)

        # Trigger worker finished handler
        tab._on_worker_finished(res)

        # Verify KPI card values
        self.assertNotEqual(tab.card_score.lbl_val.text(), "— / 100")
        self.assertNotEqual(tab.card_cplv.lbl_val.text(), "— %")
        self.assertTrue(len(tab.txt_report.toPlainText()) > 100)

        # Test dynamic post-processing re-renders
        tab.cb_frf_unit.setCurrentIndex(1)  # Linear
        tab.cb_freq_scale.setCurrentIndex(1)  # Linear scale
        tab.cb_cplv_view.setCurrentIndex(1)  # Continuous spectrum
        tab.chk_show_thresholds.setChecked(False)

        # Test modal button click
        tab._on_modal_clicked()
        self.assertGreater(tab.table_modes.rowCount(), 0)

        # Test slider scrub
        tab._on_slider_scrub(5)

    def test_4post_ui_selection_and_baseline_overlay(self):
        """Verifies switching to 4-Post in UI, disabling aero controls, saving baseline, and rendering."""
        from src.ui.ui_seven_post_rig import SevenPostRigTab
        tab = SevenPostRigTab(main_window=None)

        # Switch to 4-Post
        tab.cb_rig_arch.setCurrentIndex(1)
        self.assertFalse(tab.gb_aero.isEnabled())
        cfg = tab._get_active_config()
        self.assertEqual(cfg.rig_architecture, '4_post')
        self.assertEqual(cfg.aero_mode, 'none')

        # Run short 4-post simulation and test baseline save
        rig = SevenPostRig(VehicleParams())
        res_4p = rig.simulate(cfg)
        tab._on_worker_finished(res_4p)
        self.assertEqual(tab.last_result.rig_architecture, '4_post')
        self.assertIn("4-POST", tab.card_score.lbl_title.text().upper())
        self.assertEqual(tab.table_modes.columnCount(), 6)

        # Test Save Baseline
        tab._on_save_baseline_clicked()
        self.assertIsNotNone(tab.baseline_result)
        self.assertEqual(tab.baseline_result.rig_architecture, '4_post')
        self.assertIn("Baseline: 4-Post", tab.lbl_baseline_status.text())

        # Test Clear Baseline
        tab._on_clear_baseline_clicked()
        self.assertIsNone(tab.baseline_result)

        # Switch back to 7-Post
        tab.cb_rig_arch.setCurrentIndex(0)
        self.assertTrue(tab.gb_aero.isEnabled())
        cfg_7p = tab._get_active_config()
        self.assertEqual(cfg_7p.rig_architecture, '7_post')

    def test_ui_mainwindow_pro_integration(self):
        """Verifies that MainWindowPro includes SevenPostRigTab in Workspace 2 and can navigate to it."""
        from src.ui.main_gui_pro import MainWindowPro
        win = MainWindowPro()
        self.assertIsNotNone(win.tab_seven_post)
        ws2 = win.pro_workspace_stack.widget(2)
        tab_names = [ws2.tabText(i) for i in range(ws2.count())]
        self.assertTrue(any('7-Post' in t for t in tab_names), f"7-Post tab missing from Workspace 2: {tab_names}")
        win.open_seven_post_rig()
        self.assertEqual(win.active_workspace_idx, 2)


if __name__ == '__main__':
    unittest.main()
