"""
tests/test_arb_diamond_balance.py
---------------------------------
Unit tests for the ARB Setting Diamond Mechanical Balance Evaluation Engine (14-DOF).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.params_vehicle import VehicleParams
from src.mdo.mdo_arb_diamond_balance import (
    ARBDiamondConfig,
    convert_diamond_units,
    stiffness_from_coords,
    evaluate_setup_14dof,
    evaluate_diamond_vertices_and_current,
    evaluate_balance_ratio_1d_sweep,
    evaluate_diamond_2d_mesh
)


class TestARBDiamondBalance(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()
        self.diamond = ARBDiamondConfig(
            v_soft_soft=(25000.0, 0.48),
            v_stiff_soft=(55000.0, 0.64),
            v_stiff_stiff=(85000.0, 0.49),
            v_soft_stiff=(55000.0, 0.36),
            name="Test Diamond"
        )
        self.tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')

    def test_diamond_geometry_and_inclusion(self):
        # Center should be inside
        c_k, c_r = self.diamond.get_center_point()
        self.assertTrue(self.diamond.is_inside(c_k, c_r))

        # A point far outside should be false
        self.assertFalse(self.diamond.is_inside(10000.0, 0.20))
        self.assertFalse(self.diamond.is_inside(120000.0, 0.80))

        # Bounds check
        k_min, k_max, r_min, r_max = self.diamond.get_bounds()
        self.assertAlmostEqual(k_min, 25000.0)
        self.assertAlmostEqual(k_max, 85000.0)
        self.assertAlmostEqual(r_min, 0.36)
        self.assertAlmostEqual(r_max, 0.64)

    def test_diamond_from_arb_ranges(self):
        # Springs: 8000 Nm/rad front, 6000 Nm/rad rear
        # ARB front: 10000 to 40000 Nm/rad
        # ARB rear: 8000 to 35000 Nm/rad
        d_calc = ARBDiamondConfig.from_arb_stiffness_ranges(
            k_spring_phi_f=8000.0,
            k_spring_phi_r=6000.0,
            k_arb_f_min=10000.0,
            k_arb_f_max=40000.0,
            k_arb_r_min=8000.0,
            k_arb_r_max=35000.0
        )
        # Soft-Soft: K_f = 18000, K_r = 14000, K_tot = 32000, Ratio = 18/32 = 0.5625
        self.assertAlmostEqual(d_calc.v1[0], 32000.0)
        self.assertAlmostEqual(d_calc.v1[1], 18000.0 / 32000.0)

        # Stiff-Stiff: K_f = 48000, K_r = 41000, K_tot = 89000, Ratio = 48/89 ~ 0.5393
        self.assertAlmostEqual(d_calc.v3[0], 89000.0)
        self.assertAlmostEqual(d_calc.v3[1], 48000.0 / 89000.0)

        # Stiff Front / Soft Rear: Max front ratio
        self.assertGreater(d_calc.v2[1], d_calc.v4[1])

    def test_stiffness_from_coords(self):
        kf, kr = stiffness_from_coords(50000.0, 0.60)
        self.assertAlmostEqual(kf, 30000.0)
        self.assertAlmostEqual(kr, 20000.0)
        self.assertAlmostEqual(kf + kr, 50000.0)

    def test_evaluate_setup_14dof(self):
        # Single point evaluation
        res = evaluate_setup_14dof(self.vp, k_phi_f=35000.0, k_phi_r=25000.0, v_eval=15.0, tir_path=self.tir_path, run_transient=True)
        self.assertIn('ay_max_g', res)
        self.assertIn('t_skidpad_s', res)
        self.assertIn('kus_degg', res)
        self.assertIn('roll_grad_degg', res)
        self.assertIn('t90_s', res)
        self.assertIn('overshoot_pct', res)
        self.assertIn('score_composite', res)

        self.assertGreater(res['ay_max_g'], 1.0)
        self.assertGreater(res['t_skidpad_s'], 4.0)
        self.assertGreater(res['score_composite'], 0.0)

    def test_diamond_vertices_evaluation_trend(self):
        # Comparing V2 (Stiff Front) vs V4 (Stiff Rear):
        # V2 MUST produce higher understeer Kus than V4
        res_all = evaluate_diamond_vertices_and_current(self.vp, self.diamond, v_eval=15.0, tir_path=self.tir_path)
        dict_res = res_all['results_dict']
        v2_res = dict_res["V2 (Ant. Rigido / Post. Morbido - Max Sottosterzo)"]
        v4_res = dict_res["V4 (Ant. Morbido / Post. Rigido - Max Sovrasterzo)"]

        # V2 understeer gradient must be strictly greater than V4
        self.assertGreater(v2_res['kus_degg'], v4_res['kus_degg'])
        # V2 front ratio is greater than V4 front ratio
        self.assertGreater(v2_res['ratio_f'], v4_res['ratio_f'])

    def test_1d_sweep(self):
        df_sweep = evaluate_balance_ratio_1d_sweep(self.vp, k_tot=50000.0, ratio_min=0.40, ratio_max=0.60, n_points=5, v_eval=15.0, tir_path=self.tir_path)
        self.assertEqual(len(df_sweep), 5)
        self.assertIn('kus_degg', df_sweep.columns)
        self.assertIn('ay_max_g', df_sweep.columns)
        # As front ratio increases, Kus should generally increase (more understeer)
        self.assertGreater(df_sweep['kus_degg'].iloc[-1], df_sweep['kus_degg'].iloc[0])

    def test_2d_mesh_fast_and_progress(self):
        progress_events = []
        def p_cb(done, total, msg):
            progress_events.append((done, total, msg))

        mesh_res = evaluate_diamond_2d_mesh(
            self.vp, self.diamond,
            n_k=5, n_ratio=5, v_eval=15.0,
            tir_path=self.tir_path,
            progress_cb=p_cb
        )
        self.assertIn('best_point', mesh_res)
        self.assertIn('AY_mesh', mesh_res)
        self.assertIn('KUS_mesh', mesh_res)
        self.assertEqual(mesh_res['AY_mesh'].shape, (5, 5))
        self.assertGreater(len(progress_events), 0)

    def test_presets_io_and_saving(self):
        # Test serialization to JSON and reload
        test_path = os.path.join(os.path.dirname(__file__), 'temp_test_diamond.json')
        try:
            self.diamond.save_json(test_path)
            loaded = ARBDiamondConfig.load_json(test_path)
            self.assertEqual(loaded.name, self.diamond.name)
            self.assertAlmostEqual(loaded.v1[0], self.diamond.v1[0])
            self.assertAlmostEqual(loaded.v2[1], self.diamond.v2[1])
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_arb_diamond_worker_execution(self):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        from src.ui.gui_workers import ARBDiamondWorker
        worker = ARBDiamondWorker('1d_sweep', self.vp, self.diamond, k_tot=50000.0, n_points=4, v_eval=15.0, tir_path=self.tir_path)
        
        finished_data = {}
        def on_finished(res):
            finished_data.update(res)

        worker.finished.connect(on_finished)
        worker.run()  # Synchronous invocation of run for test stability

        self.assertIn('task_type', finished_data)
        self.assertEqual(finished_data['task_type'], '1d_sweep')
        self.assertIn('sweep_df', finished_data)
        self.assertEqual(len(finished_data['sweep_df']), 4)


if __name__ == '__main__':
    unittest.main()


