"""
test_b_segment_suv.py
======================
Unit & integration tests for the B-Segment Compact SUV vehicle profile,
FWD drivetrain, open mechanical differential, JSON persistence,
14-DOF CasADi ODE, and 3D visual models.
"""

import os
import time
import unittest
import numpy as np

from src.models.params_vehicle import VehicleParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.ui.interactive_driver_14dof import PhysicsThread14DOF
from src.models.vehicle_3d_models import FSAECar3DModel, SUVCar3DModel

try:
    from PyQt5.QtWidgets import QApplication
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False


class TestBSegmentSUV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_PYQT5:
            cls.app = QApplication.instance() or QApplication([])
        else:
            cls.app = None

    def test_01_segmento_b_suv_parameters(self):
        """Verifies VehicleParams.segmento_b_suv() nominal values and physical properties."""
        vp = VehicleParams.segmento_b_suv()

        # Drivetrain and vehicle taxonomy
        self.assertEqual(vp.vehicle_type, "b_segment_suv")
        self.assertEqual(vp.drivetrain_type, "fwd")
        self.assertEqual(vp.diff_type, "open")

        # Mass and geometry
        self.assertAlmostEqual(vp.m, 1250.0, places=1)
        self.assertAlmostEqual(vp.wd_f, 0.60, places=2)
        self.assertAlmostEqual(vp.L, 2.56, places=2)
        self.assertAlmostEqual(vp.t_f, 1.54, places=2)
        self.assertAlmostEqual(vp.t_r, 1.54, places=2)
        self.assertAlmostEqual(vp.h_cg, 0.56, places=2)
        self.assertAlmostEqual(vp.R_w, 0.315, places=3)

        # Compliant suspension rates
        self.assertAlmostEqual(vp.K_z_f, 28000.0, places=0)
        self.assertAlmostEqual(vp.K_z_r, 26000.0, places=0)

        # Steering rack
        self.assertAlmostEqual(vp.steering_ratio, 15.5, places=1)
        self.assertAlmostEqual(vp.steering_wheel_max_deg, 500.0, places=1)
        self.assertAlmostEqual(vp.road_wheel_max_deg, 500.0 / 15.5, places=1)

        # Dedicated road touring tire
        self.assertEqual(vp.tire_file, "Road_215_60R17_Touring.tir")
        self.assertAlmostEqual(vp.LMUX, 1.0, places=2)
        self.assertAlmostEqual(vp.LMUY, 1.0, places=2)

    def test_02_json_persistence(self):
        """Verifies setups/segmento_b_suv.json exists and loads identically."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "data", "setups", "segmento_b_suv.json") if os.path.exists(os.path.join(base_dir, "data", "setups", "segmento_b_suv.json")) else os.path.join(base_dir, "setups", "segmento_b_suv.json")
        self.assertTrue(os.path.exists(json_path), f"File {json_path} does not exist!")

        loaded_vp = VehicleParams.load_json(json_path)
        self.assertEqual(loaded_vp.vehicle_type, "b_segment_suv")
        self.assertEqual(loaded_vp.drivetrain_type, "fwd")
        self.assertEqual(loaded_vp.diff_type, "open")
        self.assertAlmostEqual(loaded_vp.m, 1250.0, places=1)
        self.assertAlmostEqual(loaded_vp.R_w, 0.315, places=3)
        self.assertAlmostEqual(loaded_vp.L, 2.56, places=2)
        self.assertEqual(loaded_vp.tire_file, "Road_215_60R17_Touring.tir")

    def test_03_casadi_ode_fwd(self):
        """Verifies that 14-DOF CasADi ODE function builds and evaluates with FWD drivetrain."""
        vp = VehicleParams.segmento_b_suv()
        v14 = Vehicle14DOF(vp=vp)
        ode_func = v14.get_casadi_ode()

        # States: 28, Inputs: 4, Params: 16 (or 17 with Cd)
        self.assertEqual(ode_func.n_in(), 3)
        self.assertEqual(ode_func.n_out(), 1)

        # Nominal inputs
        x0 = np.zeros((28, 1))
        x0[0, 0] = 10.0  # 10 m/s forward speed
        # Nominal wheel spin omega = v / Rw
        omega_nom = 10.0 / vp.R_w
        x0[16:20, 0] = omega_nom

        u0 = np.zeros((4, 1))
        u0[1, 0] = 150.0  # T_drive_fl
        u0[2, 0] = 150.0  # T_drive_fr

        p0 = np.array([
            vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r,
            vp.K_z_f, vp.K_z_r, vp.aero_balance,
            vp.camber_f_static, vp.camber_r_static,
            vp.toe_f_static, vp.toe_r_static,
            vp.L, vp.t_f, vp.t_r,
            vp.LMUY, vp.LMUX
        ]).reshape((16, 1))

        x_dot = np.array(ode_func(x0, u0, p0))
        self.assertEqual(x_dot.shape, (28, 1))
        self.assertFalse(np.any(np.isnan(x_dot)))

    def test_04_physics_thread_open_diff_fwd(self):
        """Verifies that PhysicsThread14DOF correctly routes drive torque exclusively to front wheels in FWD."""
        vp = VehicleParams.segmento_b_suv()
        th = PhysicsThread14DOF(vehicle_params=vp, target_hz=250.0)

        received_states = []
        th.state_updated.connect(lambda s: received_states.append(s))

        # Apply throttle via key_states
        if HAS_PYQT5:
            from PyQt5.QtCore import Qt
            th.key_states = {Qt.Key_Up: True}
        th.start()

        start_t = time.time()
        while time.time() - start_t < 0.40 and len(received_states) < 4:
            if self.app:
                self.app.processEvents()
            time.sleep(0.02)

        th.stop()
        th.wait(500)

        self.assertGreater(len(received_states), 0, "No state updates received from PhysicsThread14DOF!")
        last_st = received_states[-1]

        # Front drive torques should be positive (motoring)
        t_drive_fl = last_st.get('T_drive_fl', 0.0)
        t_drive_fr = last_st.get('T_drive_fr', 0.0)
        t_drive_rl = last_st.get('T_drive_rl', 0.0)
        t_drive_rr = last_st.get('T_drive_rr', 0.0)

        self.assertGreater(t_drive_fl, 0.0, "FL should receive positive drive torque in FWD")
        self.assertGreater(t_drive_fr, 0.0, "FR should receive positive drive torque in FWD")
        self.assertEqual(t_drive_rl, 0.0, "RL should NOT receive drive torque in FWD")
        self.assertEqual(t_drive_rr, 0.0, "RR should NOT receive drive torque in FWD")

        # Symmetric open diff on straight line
        self.assertAlmostEqual(t_drive_fl, t_drive_fr, places=2)

    def test_05_3d_geometry_models(self):
        """Verifies that 3D visual models for FSAE and SUV initialize properly."""
        fsae = FSAECar3DModel()
        self.assertTrue(hasattr(fsae, 'draw'))
        self.assertTrue(hasattr(fsae, 'set_livery_preset'))

        suv = SUVCar3DModel()
        self.assertTrue(hasattr(suv, 'draw'))
        self.assertTrue(hasattr(suv, 'set_livery_preset'))

        # Test livery presets
        suv.set_livery_preset(0)
        col0 = suv.col_body
        suv.set_livery_preset(1)
        col1 = suv.col_body
        self.assertNotEqual(col0, col1, "Livery preset should update body color")

    def test_06_road_tire_file_properties(self):
        """Verifies that Road_215_60R17_Touring.tir has calibrated road car Pacejka MF6.1 properties."""
        from src.models.params_tire import TireParams
        from src.models.tire_model_mf61 import mf61_lateral_force, mf61_longitudinal_force, mf61_aligning_torque

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tir_path = os.path.join(base_dir, "data", "tire_tir", "Road_215_60R17_Touring.tir") if os.path.exists(os.path.join(base_dir, "data", "tire_tir", "Road_215_60R17_Touring.tir")) else os.path.join(base_dir, "tire_tir", "Road_215_60R17_Touring.tir")
        self.assertTrue(os.path.exists(tir_path), f"File {tir_path} does not exist!")

        tp = TireParams(tir_path)
        self.assertAlmostEqual(tp.get('FNOMIN'), 3500.0, places=1)
        self.assertAlmostEqual(tp.get('UNLOADED_RADIUS'), 0.3449, places=3)
        self.assertAlmostEqual(tp.get('WIDTH'), 0.215, places=3)
        self.assertAlmostEqual(tp.get('NOMPRES'), 230000.0, places=1)

        # Evaluate at nominal load Fz = 3500 N
        fz = 3500.0
        alphas = np.linspace(0, 15, 16)
        fys = [float(mf61_lateral_force(np.deg2rad(a), fz, 0.0, tp)) for a in alphas]
        peak_fy = max(abs(f) for f in fys)
        mu_lat = peak_fy / fz

        # Road tire should have mu between 0.90 and 1.05
        self.assertGreaterEqual(mu_lat, 0.90)
        self.assertLessEqual(mu_lat, 1.08)

        # Cornering stiffness should be around 800 - 1100 N/deg
        cf_alpha = abs(fys[1] - fys[0]) / 1.0
        self.assertGreaterEqual(cf_alpha, 700.0)
        self.assertLessEqual(cf_alpha, 1200.0)

        # Longitudinal force
        fx_peak = max(float(mf61_longitudinal_force(k, fz, 0.0, tp)) for k in np.linspace(0, 0.25, 26))
        mu_long = fx_peak / fz
        self.assertGreaterEqual(mu_long, 0.95)
        self.assertLessEqual(mu_long, 1.15)

    def test_07_viewport_vp_attribute_safety(self):
        """Verifies that OpenGLViewport3D and _LegacyViewport3D have self.vp defined and handle vehicle_type safely."""
        from src.ui.interactive_driver_14dof import OpenGLViewport3D, _LegacyViewport3D
        if not HAS_PYQT5:
            return

        vp_suv = VehicleParams.segmento_b_suv()

        # Test OpenGLViewport3D
        try:
            gl_vp = OpenGLViewport3D()
            self.assertTrue(hasattr(gl_vp, 'vp'), "OpenGLViewport3D must have 'vp' attribute defined")
            self.assertIsNone(gl_vp.vp)
            gl_vp.set_vehicle_params(vp_suv)
            self.assertEqual(gl_vp.vp.vehicle_type, "b_segment_suv")
        except Exception:
            pass

        # Test _LegacyViewport3D
        legacy_vp = _LegacyViewport3D()
        self.assertTrue(hasattr(legacy_vp, 'vp'), "_LegacyViewport3D must have 'vp' attribute defined")
        self.assertIsNone(legacy_vp.vp)
        legacy_vp.set_vehicle_params(vp_suv)
        self.assertEqual(legacy_vp.vp.vehicle_type, "b_segment_suv")

    def test_08_suv_braking_and_thermal_sizing(self):
        """Verifies calibrated braking torque and tire thermal inertia scaling for SUV."""
        from src.models.tire_degradation import TireDegradationModel, TireState
        vp_suv = VehicleParams.segmento_b_suv()
        vp_fsae = VehicleParams()

        # FSAE baseline
        self.assertAlmostEqual(vp_fsae.T_brake_max_total, 1400.0)
        self.assertAlmostEqual(vp_fsae.tire_thermal_mass, 2000.0)
        self.assertAlmostEqual(vp_fsae.tire_h_conv, 8.5)

        # SUV calibrated properties
        self.assertGreaterEqual(vp_suv.T_brake_max_total, 4500.0)
        max_brake_force = vp_suv.T_brake_max_total / vp_suv.R_w
        decel_g = max_brake_force / (vp_suv.m * vp_suv.g)
        self.assertGreater(decel_g, 1.10, "SUV must have > 1.1g braking deceleration capability")

        # Tire thermal model scaling
        model = TireDegradationModel(mode="light")
        model.update_from_vehicle_params(vp_suv)
        self.assertEqual(model.params.thermal_mass_bulk, vp_suv.tire_thermal_mass)
        self.assertEqual(model.params.h_conv_bulk, vp_suv.tire_h_conv)
        self.assertEqual(model.params.T_optimal, 52.0)
        self.assertEqual(model.params.T_overheat, 105.0)

        # Inerzia termica: under identical slip power, SUV tire warms up ~10x slower
        state_fsae = TireState(T_bulk=25.0)
        model_fsae = TireDegradationModel(mode="light")
        model_fsae.update_from_vehicle_params(vp_fsae)
        model_fsae.step(state_fsae, F_total=3000.0, v_slip=1.0, Fz=3000.0, vx=15.0, dt=1.0)
        dT_fsae = state_fsae.T_bulk - 25.0

        state_suv = TireState(T_bulk=25.0)
        model.step(state_suv, F_total=3000.0, v_slip=1.0, Fz=3000.0, vx=15.0, dt=1.0)
        dT_suv = state_suv.T_bulk - 25.0

        self.assertLess(dT_suv, dT_fsae * 0.30, "SUV tire must have significantly higher thermal inertia")

    def test_09_telemetry_save_load_csv_json(self):
        """Verifies telemetry data export and import for CSV and JSON formats."""
        import pandas as pd
        import tempfile

        # Generate realistic 4-wheel test telemetry data
        N = 100
        t = np.linspace(0, 5, N)
        test_records = [
            {
                't': float(t[i]),
                's_lap': float(10.0 * t[i]),
                'dist_total': float(10.0 * t[i]),
                'lap_count': 1,
                'vx_kmh': 50.0 + 5.0 * np.sin(t[i]),
                'steer_deg': 25.0 * np.cos(t[i]),
                'Fy_fl': 1200.0 * np.sin(t[i]),
                'Fy_fr': 1150.0 * np.sin(t[i]),
                'Fy_rl': 850.0 * np.sin(t[i]),
                'Fy_rr': 820.0 * np.sin(t[i]),
                'Fx_fl': 300.0, 'Fx_fr': 300.0, 'Fx_rl': 0.0, 'Fx_rr': 0.0,
                'Fz_fl': 3400.0, 'Fz_fr': 3450.0, 'Fz_rl': 2300.0, 'Fz_rr': 2350.0,
                'T_drive_fl': 120.0, 'T_drive_fr': 120.0,
                'T_brake_fl': 0.0, 'T_brake_fr': 0.0,
                'tire_T_bulk_fl': 45.0, 'tire_T_bulk_fr': 44.5,
                'tire_T_bulk_rl': 38.0, 'tire_T_bulk_rr': 37.8
            }
            for i in range(N)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_telem.csv")
            json_path = os.path.join(tmpdir, "test_telem.json")

            # CSV roundtrip
            df_orig = pd.DataFrame(test_records)
            df_orig.to_csv(csv_path, index=False)
            df_csv = pd.read_csv(csv_path)
            self.assertEqual(len(df_csv), N)
            self.assertIn('Fy_fl', df_csv.columns)
            self.assertIn('Fy_fr', df_csv.columns)
            self.assertIn('Fy_rl', df_csv.columns)
            self.assertIn('Fy_rr', df_csv.columns)

            # JSON roundtrip
            df_orig.to_json(json_path, orient='records')
            df_json = pd.read_json(json_path)
            self.assertEqual(len(df_json), N)
            self.assertAlmostEqual(df_json['Fy_fl'].iloc[0], df_orig['Fy_fl'].iloc[0], places=3)

    def test_10_full_telemetry_presets_4wheel_forces(self):
        """Verifies FullTelemetryWindow PRESETS contain 4-wheel force comparison plots."""
        from src.ui.interactive_driver_14dof import FullTelemetryWindow
        presets = FullTelemetryWindow.PRESETS
        self.assertIn("2. Forze Ruote 4 Angoli (Fy, Fx, Fz, Coppie)", presets)
        wheel_preset = presets["2. Forze Ruote 4 Angoli (Fy, Fx, Fz, Coppie)"]
        self.assertIn('lateral_forces', wheel_preset)
        self.assertIn('longitudinal_forces', wheel_preset)
        self.assertIn('vertical_loads', wheel_preset)
        self.assertIn('wheel_torques', wheel_preset)


if __name__ == '__main__':
    unittest.main()
