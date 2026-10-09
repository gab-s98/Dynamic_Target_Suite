"""
test_mmm_hd_verification.py
----------------------------
Unit tests for high-definition Yaw Moment Diagram (YMD / Moment Method) and Multi-Speed Envelopes.
"""

import os
import unittest
import numpy as np

from src.mdo import mdo_target_optimization as mdo
from src.models.params_vehicle import VehicleParams


class TestMMMAndMultiSpeed(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tir_path = os.path.join(self.base_dir, 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir')

    def test_single_speed_mmm_diagram_3dof(self):
        df_mmm = mdo.generate_mmm_diagram(
            v_x=15.0, wd_f=self.vp.wd_f, lltd=self.vp.lltd, aero_bal=self.vp.aero_balance,
            L=self.vp.L, t_f=self.vp.t_f, t_r=self.vp.t_r,
            delta_deg_list=[-6.0, 0.0, 6.0], num_beta=11, tir_file=self.tir_path,
            model_type="3-DOF"
        )
        self.assertFalse(df_mmm.empty)
        self.assertIn('ay_g', df_mmm.columns)
        self.assertIn('yaw_moment', df_mmm.columns)
        self.assertGreater(df_mmm['ay_g'].max(), 0.5)

    def test_single_speed_mmm_diagram_7dof(self):
        df_mmm = mdo.generate_mmm_diagram(
            v_x=15.0, wd_f=self.vp.wd_f, lltd=self.vp.lltd, aero_bal=self.vp.aero_balance,
            L=self.vp.L, t_f=self.vp.t_f, t_r=self.vp.t_r,
            delta_deg_list=[-6.0, 0.0, 6.0], num_beta=11, tir_file=self.tir_path,
            model_type="7-DOF",
            camber_f_static=-2.0, camber_r_static=-1.5,
            toe_f_static=0.1, toe_r_static=0.1
        )
        self.assertFalse(df_mmm.empty)
        self.assertIn('ay_g', df_mmm.columns)
        self.assertIn('yaw_moment', df_mmm.columns)
        self.assertGreater(df_mmm['ay_g'].max(), 0.5)

    def test_single_speed_mmm_diagram_14dof(self):
        df_mmm = mdo.generate_mmm_diagram(
            v_x=15.0, wd_f=self.vp.wd_f, lltd=self.vp.lltd, aero_bal=self.vp.aero_balance,
            L=self.vp.L, t_f=self.vp.t_f, t_r=self.vp.t_r,
            delta_deg_list=[-6.0, 0.0, 6.0], num_beta=11, tir_file=self.tir_path,
            model_type="14-DOF",
            K_phi_f=8500.0, K_phi_r=6500.0,
            K_z_f=19000.0, K_z_r=17000.0
        )
        self.assertFalse(df_mmm.empty)
        self.assertIn('ay_g', df_mmm.columns)
        self.assertIn('yaw_moment', df_mmm.columns)
        self.assertGreater(df_mmm['ay_g'].max(), 0.5)

    def test_multispeed_mmm_envelopes(self):
        res = mdo.generate_multispeed_mmm_envelopes(
            speeds_list=[10.0, 15.0, 20.0],
            wd_f=self.vp.wd_f, lltd=self.vp.lltd, aero_bal=self.vp.aero_balance,
            L=self.vp.L, t_f=self.vp.t_f, t_r=self.vp.t_r,
            delta_max_deg=8.0, num_beta=9, tir_file=self.tir_path,
            model_type="14-DOF"
        )
        self.assertIn('speed_envelopes', res)
        self.assertIn('derivatives_df', res)
        
        envs = res['speed_envelopes']
        self.assertEqual(len(envs), 3)
        self.assertIn(10.0, envs)
        self.assertIn(15.0, envs)
        self.assertIn(20.0, envs)

        df_derivs = res['derivatives_df']
        self.assertEqual(len(df_derivs), 3)
        self.assertIn('N_delta', df_derivs.columns)
        self.assertIn('N_beta', df_derivs.columns)
        self.assertIn('control_margin', df_derivs.columns)
        self.assertIn('max_ay_g', df_derivs.columns)


if __name__ == '__main__':
    unittest.main()
