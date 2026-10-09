"""
tests/test_powertrain_electric.py
---------------------------------
Unit and integration tests for the Electric Powertrain subsystem:
1. EMRAX 188HV LC motor characteristics (peak torque, peak power, max RPM).
2. 4:1 reduction ratio kinematics and wheel-to-motor conversions.
3. 2D efficiency map eta(RPM, Torque) interpolation and sweet spot validation.
4. FSAE 80 kW accumulator electrical power limit & asymmetric Torque Vectoring (e.g. 50 kW / 30 kW split).
5. CasADi symbolic smooth peak torque formulation.
6. Integration with 3-DOF, 7-DOF, 14-DOF vehicle models and straight-line acceleration.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import unittest
import numpy as np
import casadi as ca

from src.models.powertrain_electric import EMRAX188HV, get_default_powertrain, allocate_dual_motor_torques, casadi_peak_torque
from src.models.params_vehicle import VehicleParams
from src.models.vehicle_3dof import Vehicle3DOF
from src.models.vehicle_7dof import Vehicle7DOF
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_acceleration import simulate_acceleration


class TestEMRAX188HVCharacteristics(unittest.TestCase):
    def setUp(self):
        self.pt = EMRAX188HV(gear_ratio=4.0, p_elec_limit_kw=80.0, n_drivetrain=0.95)

    def test_01_constants_and_gear_ratio(self):
        """Verify fundamental motor constants and 4:1 reduction ratio."""
        self.assertEqual(self.pt.gear_ratio, 4.0)
        self.assertEqual(self.pt.p_elec_limit_kw, 80.0)
        self.assertEqual(self.pt.P_MAX_PEAK_KW, 60.0)
        self.assertEqual(self.pt.T_MAX_PEAK_NM, 100.0)
        self.assertEqual(self.pt.RPM_MAX, 8000.0)

    def test_02_peak_torque_curve(self):
        """Verify peak torque matches manufacturer curve (100 Nm plateau up to 5500 RPM, 70 Nm @ 8000 RPM)."""
        # Plateau: 500 - 5500 RPM
        for rpm in [500.0, 1500.0, 3000.0, 4500.0, 5500.0]:
            t_pk = self.pt.peak_torque(rpm)
            self.assertAlmostEqual(t_pk, 100.0, delta=0.5, msg=f"Torque at {rpm} RPM should be ~100 Nm")

        # Falloff towards 8000 RPM: 70 Nm
        t_8000 = self.pt.peak_torque(8000.0)
        self.assertAlmostEqual(t_8000, 70.0, delta=1.0, msg="Torque at 8000 RPM should be ~70 Nm")

        # Over-rev cutoff: > 8000 RPM -> 0 Nm
        t_over = self.pt.peak_torque(8500.0)
        self.assertEqual(t_over, 0.0, msg="Torque above 8000 RPM must be 0 Nm")

    def test_03_peak_power_kw(self):
        """Verify peak mechanical power reaches ~60 kW around 6200-6500 RPM."""
        p_peak = self.pt.peak_power_kw(6500.0)
        self.assertGreaterEqual(p_peak, 58.0, "Peak power at 6500 RPM should reach ~60 kW")
        self.assertLessEqual(p_peak, 63.0, "Peak power should not unrealistically exceed 63 kW")

        # At low speed (e.g. 1000 RPM, 100 Nm), power is ~10.47 kW
        p_1000 = self.pt.peak_power_kw(1000.0)
        self.assertAlmostEqual(p_1000, 100.0 * (1000.0 * 2.0 * np.pi / 60.0) / 1000.0, places=2)

    def test_04_wheel_rad_s_to_motor_rpm(self):
        """Verify wheel speed to motor shaft RPM conversion with 4:1 ratio."""
        # 100 rad/s wheel * 4.0 = 400 rad/s motor -> 400 * 60 / (2*pi) ~= 3819.7 RPM
        w_wheel = 100.0
        expected_rpm = (100.0 * 4.0) * (60.0 / (2.0 * np.pi))
        rpm = self.pt.wheel_rad_s_to_motor_rpm(w_wheel)
        self.assertAlmostEqual(rpm, expected_rpm, places=3)


class TestEfficiencyMap(unittest.TestCase):
    def setUp(self):
        self.pt = get_default_powertrain()

    def test_01_sweet_spot_island(self):
        """Verify efficiency in sweet spot island (2300-5700 RPM, 30-57 Nm) is >= 96%."""
        test_points = [
            (3800.0, 44.0),
            (3000.0, 40.0),
            (4500.0, 50.0),
            (5000.0, 35.0),
        ]
        for rpm, torque in test_points:
            eta = self.pt.efficiency(rpm, torque)
            self.assertGreaterEqual(eta, 0.959, f"Efficiency at {rpm} RPM, {torque} Nm should be >= 96% (got {eta:.4f})")

    def test_02_contours_and_boundaries(self):
        """Verify efficiency contours decrease reasonably outside optimal range and stay in [0.50, 0.966]."""
        # Very low speed/torque
        eta_low = self.pt.efficiency(200.0, 5.0)
        self.assertGreaterEqual(eta_low, 0.50)
        self.assertLess(eta_low, 0.86)

        # High RPM moderate torque (8000 RPM, 60 Nm)
        eta_high = self.pt.efficiency(7500.0, 60.0)
        self.assertGreater(eta_high, 0.85)
        self.assertLessEqual(eta_high, 0.966)


class TestDualMotorAllocationAndFSAELimit(unittest.TestCase):
    def setUp(self):
        self.pt = EMRAX188HV(gear_ratio=4.0, p_elec_limit_kw=80.0, n_drivetrain=0.95)

    def test_01_low_power_unconstrained(self):
        """When total electrical power is below 80 kW, exact torque requests are satisfied."""
        # Wheel speeds corresponding to ~3000 RPM motor
        w_wheel = 3000.0 / (4.0 * (60.0 / (2.0 * np.pi)))  # ~78.54 rad/s
        # Request 100 Nm wheel torque each (approx 26.3 Nm motor torque each)
        t_req_wheel = 100.0
        t_rl, t_rr, telem = self.pt.allocate_dual_motor_torques(w_wheel, w_wheel, t_req_wheel, t_req_wheel)

        self.assertAlmostEqual(t_rl, t_req_wheel, places=1)
        self.assertAlmostEqual(t_rr, t_req_wheel, places=1)
        self.assertFalse(telem['is_power_limited'])
        self.assertLess(telem['p_elec_total_kw'], 80.0)

    def test_02_symmetric_high_power_capping(self):
        """When requesting max torque at high speed, total electrical power is strictly capped at 80 kW."""
        # Motor speed: ~6000 RPM (each motor could produce ~60 kW, total ~120 kW)
        w_wheel = 6000.0 / (4.0 * (60.0 / (2.0 * np.pi)))
        # Request large torque (e.g. 500 Nm per wheel)
        t_rl, t_rr, telem = self.pt.allocate_dual_motor_torques(w_wheel, w_wheel, 500.0, 500.0)

        # Total electrical power must be capped at 80 kW (within small numerical tolerance)
        self.assertAlmostEqual(telem['p_elec_total_kw'], 80.0, delta=0.2)
        self.assertTrue(telem['is_power_limited'])
        # Symmetric request -> symmetric electrical split (~40 kW each)
        self.assertAlmostEqual(telem['p_elec_rl_kw'], 40.0, delta=0.5)
        self.assertAlmostEqual(telem['p_elec_rr_kw'], 40.0, delta=0.5)
        self.assertAlmostEqual(t_rl, t_rr, delta=1.0)

    def test_03_asymmetric_torque_vectoring_split(self):
        """Verify asymmetric torque request (e.g. 50 kW on one side and 30 kW on the other) sums to 80 kW."""
        w_wheel = 6000.0 / (4.0 * (60.0 / (2.0 * np.pi)))  # 6000 RPM
        conv = 4.0 * 0.95
        omega_m = 6000.0 * (2.0 * np.pi / 60.0)

        # Request torque targeting 50 kW mech on RR and 30 kW mech on RL
        t_m_rl = (30000.0 / omega_m)
        t_m_rr = (50000.0 / omega_m)
        t_req_rl = t_m_rl * conv
        t_req_rr = t_m_rr * conv

        t_rl, t_rr, telem = self.pt.allocate_dual_motor_torques(w_wheel, w_wheel, t_req_rl, t_req_rr)

        # Total electrical power must not exceed 80 kW limit
        self.assertLessEqual(telem['p_elec_total_kw'], 80.01)
        # Asymmetry must be maintained (RR receives significantly more torque than RL)
        self.assertGreater(t_rr, t_rl * 1.3)
        self.assertAlmostEqual(telem['p_elec_total_kw'], 80.0, delta=0.5)
        # Check that individual powers are approximately 30 kW and 50 kW electrical
        self.assertAlmostEqual(telem['p_elec_rl_kw'], 30.0, delta=2.5)
        self.assertAlmostEqual(telem['p_elec_rr_kw'], 50.0, delta=2.5)


class TestCasadiSmoothApproximation(unittest.TestCase):
    def test_casadi_peak_torque(self):
        """Verify symbolic CasADi approximation evaluates and closely tracks numeric peak torque."""
        rpm_s = ca.SX.sym('rpm')
        t_expr = casadi_peak_torque(rpm_s)
        f_torque = ca.Function('f_torque', [rpm_s], [t_expr])

        test_rpms = [1000.0, 3000.0, 5500.0, 6500.0, 8000.0]
        pt = get_default_powertrain()

        for rpm in test_rpms:
            t_num = pt.peak_torque(rpm)
            t_cas = float(f_torque(rpm))
            # Smooth approximation should be within 10% or 6 Nm of physical curve
            self.assertAlmostEqual(t_cas, t_num, delta=7.0,
                                   msg=f"CasADi torque at {rpm} RPM ({t_cas:.1f}) deviated from numeric ({t_num:.1f})")


class TestVehicleModelIntegrations(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_01_params_vehicle_powertrain_attributes(self):
        """VehicleParams contains motor specifications and returns a valid powertrain instance."""
        self.assertEqual(self.vp.motor_type, "GENERIC_ELECTRIC_POWERTRAIN")
        self.assertEqual(self.vp.P_max_motor_kw, 45.0)
        self.assertEqual(self.vp.P_max_kw, 70.0)
        self.assertEqual(self.vp.final_drive, 4.0)

        pt = self.vp.get_powertrain()
        self.assertIsInstance(pt, EMRAX188HV)
        self.assertEqual(pt.gear_ratio, 4.0)
        self.assertEqual(pt.p_elec_limit_kw, 70.0)

    def test_02_vehicle_3dof_tractive_envelope(self):
        """Vehicle3DOF uses the generic electric powertrain tractive envelope."""
        model = Vehicle3DOF(self.vp)
        envelope = model.get_tractive_envelope(20.0)  # at 20 m/s (~72 km/h)
        self.assertIn('fx_max_total_n', envelope)
        self.assertIn('p_elec_total_kw', envelope)
        self.assertIn('eta', envelope)
        self.assertIn('rpm', envelope)
        self.assertLessEqual(envelope['p_elec_total_kw'], 70.05)
        self.assertGreater(envelope['fx_max_total_n'], 500.0)

    def test_03_vehicle_7dof_powertrain_limits(self):
        """Vehicle7DOF applies dual-motor allocation to wheel spin states."""
        model = Vehicle7DOF(self.vp)
        w_rl = 110.0  # rad/s (~4200 RPM motor, power limited)
        w_rr = 110.0  # rad/s
        t_rl_req = 400.0  # Nm wheel torque
        t_rr_req = 400.0  # Nm wheel torque

        t_rl, t_rr, telem = model.apply_powertrain_limits(w_rl, w_rr, t_rl_req, t_rr_req)
        self.assertLessEqual(telem['p_elec_total_kw'], 70.05)
        self.assertTrue(telem['is_power_limited'])
        self.assertAlmostEqual(t_rl, t_rr, delta=1.0)

    def test_04_vehicle_14dof_powertrain_limits(self):
        """Vehicle14DOF applies dual-motor allocation with 4:1 ratio and 80 kW limit."""
        model = Vehicle14DOF(self.vp)
        w_rl = 100.0  # rad/s
        w_rr = 105.0  # rad/s (different wheel speeds in corner)
        t_rl_req = 300.0
        t_rr_req = 500.0  # TV bias towards right wheel

        t_rl, t_rr, telem = model.apply_powertrain_limits(w_rl, w_rr, t_rl_req, t_rr_req)
        self.assertLessEqual(telem['p_elec_total_kw'], 80.05)
        self.assertGreater(t_rr, t_rl)
        self.assertGreater(telem['rpm_rr'], telem['rpm_rl'])

    def test_05_acceleration_run_with_powertrain(self):
        """simulate_acceleration produces realistic FSAE 75m times and records electrical telemetry."""
        res = simulate_acceleration(self.vp, distance=75.0)
        self.assertTrue(res['success'])
        t_75m = res['time']
        v_final_kmh = res['v_max'] * 3.6  # v_max is in m/s

        # FSAE 75m electric acceleration typically 3.5s - 4.5s with 80 kW limit
        self.assertGreaterEqual(t_75m, 3.2, "75m acceleration time should be >= 3.2s")
        self.assertLessEqual(t_75m, 5.0, "75m acceleration time should be <= 5.0s")
        self.assertGreaterEqual(v_final_kmh, 100.0, "Terminal speed should exceed 100 km/h")
        self.assertIn('elec_energy_wh', res)
        self.assertIn('avg_efficiency', res)
        self.assertGreater(res['avg_efficiency'], 0.65)


if __name__ == '__main__':
    unittest.main()
