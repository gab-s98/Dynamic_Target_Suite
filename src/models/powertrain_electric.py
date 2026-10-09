"""
powertrain_electric.py
-----------------------
Physical and numerical model of the Electric Powertrain for Formula SAE / Student prototypes.
Single Source of Truth for:
- EMRAX 188HV LC Motor characteristics (Peak Torque 100 Nm, Peak Power 60 kW, 8000 RPM)
- 2D Bivariate Efficiency Map eta(RPM, Torque) based on manufacturer dynamometer data
- Dual-Motor RWD layout with 4:1 reduction ratio (wheel speed = 1/4 motor speed)
- FSAE Official 80 kW Total Electrical Power Limit (Accumulator energy meter rule)
- Asymmetric Torque Vectoring power allocation (e.g. 50 kW to one motor and 30 kW to the other)
"""

import numpy as np
from typing import Tuple, Dict, Any, Union
import casadi as ca
from scipy.interpolate import RegularGridInterpolator


class EMRAX188HV:
    """
    Physical model of the EMRAX 188HV Liquid-Cooled (LC) Axial Flux Synchronous Motor.
    Configured for peak torque and peak power operating characteristics (non-continuous).
    """
    # Calibration Constants
    GEAR_RATIO: float = 4.0               # Final drive reduction ratio i = 4.0 (wheel speed = motor_speed / 4)
    P_MAX_PEAK_KW: float = 60.0           # Single motor peak power [kW] (~60 kW @ 6200-6500 RPM)
    T_MAX_PEAK_NM: float = 100.0          # Single motor peak torque [Nm] (100 Nm flat 500-5500 RPM)
    RPM_MAX: float = 8000.0               # Maximum operational motor speed [RPM]
    P_ELEC_TOTAL_LIMIT_KW: float = 80.0   # FSAE rule EV maximum accumulator power limit [kW]
    ETA_DRIVETRAIN: float = 0.95          # Mechanical transmission efficiency (gearbox/chain)

    def __init__(self,
                 gear_ratio: float = 4.0,
                 p_elec_limit_kw: float = 80.0,
                 n_drivetrain: float = 0.95):
        self.gear_ratio = float(gear_ratio)
        self.p_elec_limit_kw = float(p_elec_limit_kw)
        self.n_drivetrain = float(n_drivetrain)

        # Build 2D regular grid for efficiency map eta(RPM, Torque)
        self._rpms_grid = np.linspace(0.0, 8000.0, 17)      # 0, 500, 1000, ..., 8000 RPM
        self._torques_grid = np.linspace(0.0, 100.0, 11)    # 0, 10, 20, ..., 100 Nm
        self._eta_table = self._build_efficiency_table()
        self._interpolator = RegularGridInterpolator(
            (self._rpms_grid, self._torques_grid),
            self._eta_table,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def _build_efficiency_table(self) -> np.ndarray:
        """
        Calibrates the 17x11 2D efficiency matrix from 02_efficiency_speed.png:
        - 96% sweet spot island: 2300-5700 RPM, 30-57 Nm (peaking at ~96.5% @ 3800 RPM, 44 Nm)
        - 95% contour: 1900-6700 RPM, 25-69 Nm
        - 94% contour: 1600-8000 RPM, 22-80 Nm (flat top along 80 Nm to 8000 RPM)
        - 90-94% contour: 1200-8000 RPM, 15-90 Nm
        - 86-90% contour: 400-8000 RPM, 7-97 Nm
        """
        table = np.zeros((len(self._rpms_grid), len(self._torques_grid)), dtype=np.float64)

        for i, rpm in enumerate(self._rpms_grid):
            for j, torque in enumerate(self._torques_grid):
                if torque <= 0.0 or rpm <= 0.0:
                    table[i, j] = 0.50
                    continue

                # 1. Sweet spot island >= 96%
                if 2300.0 <= rpm <= 5700.0 and 30.0 <= torque <= 57.0:
                    dr = (rpm - 3800.0) / 1700.0
                    dt = (torque - 44.0) / 14.0
                    eta = 0.966 - 0.006 * (dr**2 + dt**2)
                # 2. 95% contour
                elif 1900.0 <= rpm <= 6700.0 and 25.0 <= torque <= 69.0:
                    dr = (rpm - 3800.0) / 2400.0
                    dt = (torque - 45.0) / 22.0
                    eta = 0.950 + 0.009 * (1.0 - (dr**2 + dt**2))
                # 3. 94% contour (notice 80 Nm continues up to 8000 RPM)
                elif 1600.0 <= rpm <= 8000.0 and 22.0 <= torque <= 80.0:
                    dr = (rpm - 4200.0) / 3200.0
                    dt = (torque - 47.0) / 30.0
                    eta = 0.940 + 0.009 * (1.0 - (dr**2 + dt**2))
                # 4. 90-94% contour
                elif 1200.0 <= rpm <= 8000.0 and 15.0 <= torque <= 90.0:
                    dr = (rpm - 4500.0) / 3500.0
                    dt = (torque - 50.0) / 38.0
                    eta = 0.900 + 0.038 * (1.0 - (dr**2 + dt**2))
                # 5. 86-90% outer boundary
                elif 400.0 <= rpm <= 8000.0 and 7.0 <= torque <= 97.0:
                    dr = (rpm - 4200.0) / 4000.0
                    dt = (torque - 52.0) / 46.0
                    eta = 0.860 + 0.038 * (1.0 - (dr**2 + dt**2))
                else:
                    # Regions outside the 86% contour (low torque, very low speed)
                    if torque < 7.0:
                        eta = max(0.50, 0.86 * (torque / 7.0)**0.8)
                    elif rpm < 400.0:
                        eta = max(0.50, 0.86 * (rpm / 400.0)**0.8)
                    else:
                        eta = 0.84

                table[i, j] = float(np.clip(eta, 0.50, 0.966))

        return table

    def peak_torque(self, rpm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns the single motor peak torque limit [Nm] as a function of RPM from 01_power_torque.png.
        - 0 to 500 RPM: 96.0 -> 100.0 Nm ramp
        - 500 to 5500 RPM: 100.0 Nm flat
        - 5500 to 8000 RPM: constant power envelope peaking at 60 kW and dropping to 70 Nm at 8000 RPM
        """
        is_scalar = np.isscalar(rpm)
        rpm_arr = np.atleast_1d(np.asarray(rpm, dtype=np.float64))
        rpm_clamped = np.clip(rpm_arr, 0.0, self.RPM_MAX)

        # Baseline torque table for interpolation
        rpm_ref = np.array([0.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 5500.0,
                            6000.0, 6500.0, 7000.0, 7500.0, 8000.0], dtype=np.float64)
        t_ref = np.array([96.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                          95.5, 88.2, 81.2, 75.1, 70.0], dtype=np.float64)

        t_peak = np.interp(rpm_clamped, rpm_ref, t_ref)

        # Hard cutoff above max RPM
        over_rev = rpm_arr > self.RPM_MAX
        if np.any(over_rev):
            t_peak[over_rev] = 0.0

        return float(t_peak[0]) if is_scalar else t_peak

    def peak_power_kw(self, rpm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns the single motor peak mechanical power [kW]:
        P_mech = T_peak(RPM) * (RPM * 2*pi / 60) / 1000
        """
        t_pk = self.peak_torque(rpm)
        rpm_arr = np.asarray(rpm, dtype=np.float64)
        omega = rpm_arr * (2.0 * np.pi / 60.0)
        p_kw = (t_pk * omega) / 1000.0
        return float(p_kw) if np.isscalar(rpm) else p_kw

    def efficiency(self, rpm: Union[float, np.ndarray], torque_nm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluates motor efficiency eta in [0.50, 0.966] for given (RPM, Torque).
        """
        is_scalar = np.isscalar(rpm) and np.isscalar(torque_nm)
        r_arr = np.atleast_1d(np.asarray(rpm, dtype=np.float64))
        t_arr = np.atleast_1d(np.asarray(torque_nm, dtype=np.float64))

        pts = np.column_stack([
            np.clip(r_arr, 0.0, self.RPM_MAX),
            np.clip(t_arr, 0.0, 100.0)
        ])

        eta = self._interpolator(pts)
        return float(eta[0]) if is_scalar else eta

    def wheel_rad_s_to_motor_rpm(self, w_wheel: float) -> float:
        """
        Converts wheel rotational velocity [rad/s] to motor shaft [RPM]
        using reduction ratio i = 4.0:
        omega_motor = omega_wheel * 4.0
        RPM = omega_motor * 60 / (2 * pi)
        """
        w_motor = abs(float(w_wheel)) * self.gear_ratio
        return float(w_motor * (60.0 / (2.0 * np.pi)))

    def allocate_dual_motor_torques(self,
                                    w_rl: float,
                                    w_rr: float,
                                    T_req_wheel_rl: float,
                                    T_req_wheel_rr: float,
                                    p_elec_limit_total_w: float = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Allocates driving torques to the two rear wheels, enforcing:
        1. Single motor peak torque envelope T_motor,i <= T_peak(RPM_i) from EMRAX 188 curve.
        2. Maximum total electrical power drawn from accumulator P_elec,total <= 80 kW (FSAE EV rule).
        3. Supports asymmetric torque vectoring (e.g. 50 kW to outer motor and 30 kW to inner motor).
        4. Calculates mechanical & electrical powers and instantaneous motor efficiencies.
        
        Returns:
            (T_wheel_rl, T_wheel_rr, telemetry_info)
        """
        p_limit_w = p_elec_limit_total_w if p_elec_limit_total_w is not None else (self.p_elec_limit_kw * 1000.0)

        # 1. Motor shaft speeds
        omega_m_l = max(abs(float(w_rl)) * self.gear_ratio, 0.05)
        omega_m_r = max(abs(float(w_rr)) * self.gear_ratio, 0.05)
        rpm_l = omega_m_l * (60.0 / (2.0 * np.pi))
        rpm_r = omega_m_r * (60.0 / (2.0 * np.pi))

        # 2. Individual motor peak torque limits at current RPM
        t_peak_l = self.peak_torque(rpm_l)
        t_peak_r = self.peak_torque(rpm_r)

        # 3. Requested motor shaft torque (Wheel torque -> Motor torque)
        # T_wheel = T_motor * gear_ratio * n_drivetrain
        conv_factor = self.gear_ratio * self.n_drivetrain
        t_m_req_l = max(float(T_req_wheel_rl) / max(conv_factor, 1e-4), 0.0)
        t_m_req_r = max(float(T_req_wheel_rr) / max(conv_factor, 1e-4), 0.0)

        # Clamp to single motor peak capability
        t_m_l = min(t_m_req_l, t_peak_l)
        t_m_r = min(t_m_req_r, t_peak_r)

        # 4. Instantaneous efficiencies & electrical power requests
        eta_l = float(self.efficiency(rpm_l, t_m_l))
        eta_r = float(self.efficiency(rpm_r, t_m_r))

        p_mech_l = t_m_l * omega_m_l
        p_mech_r = t_m_r * omega_m_r

        p_elec_l = p_mech_l / max(eta_l, 0.10)
        p_elec_r = p_mech_r / max(eta_r, 0.10)
        p_elec_tot = p_elec_l + p_elec_r

        # 5. Enforce 80 kW total electrical power limit
        if p_elec_tot > p_limit_w and p_elec_tot > 1.0:
            # Preserve Torque Vectoring power differential Delta P = P_elec_r - P_elec_l
            delta_p_elec = p_elec_r - p_elec_l

            # Target allocation: P_l + P_r = P_limit and P_r - P_l = Delta P
            # P_r_tgt = (P_limit + Delta P) / 2
            # P_l_tgt = (P_limit - Delta P) / 2
            p_elec_r_tgt = 0.5 * (p_limit_w + delta_p_elec)
            p_elec_l_tgt = 0.5 * (p_limit_w - delta_p_elec)

            # Max electrical power each motor can physically draw at its current RPM peak:
            p_elec_max_l = (t_peak_l * omega_m_l) / max(eta_l, 0.10)
            p_elec_max_r = (t_peak_r * omega_m_r) / max(eta_r, 0.10)

            # Clamp targets to physically feasible range [0, P_elec_max_i]
            if p_elec_l_tgt < 0.0:
                p_elec_l_final = 0.0
                p_elec_r_final = min(p_limit_w, p_elec_max_r)
            elif p_elec_r_tgt < 0.0:
                p_elec_r_final = 0.0
                p_elec_l_final = min(p_limit_w, p_elec_max_l)
            elif p_elec_r_tgt > p_elec_max_r:
                p_elec_r_final = p_elec_max_r
                p_elec_l_final = min(p_limit_w - p_elec_r_final, p_elec_max_l)
            elif p_elec_l_tgt > p_elec_max_l:
                p_elec_l_final = p_elec_max_l
                p_elec_r_final = min(p_limit_w - p_elec_l_final, p_elec_max_r)
            else:
                p_elec_l_final = p_elec_l_tgt
                p_elec_r_final = p_elec_r_tgt

            # Update mechanical powers and torques from allocated electrical power
            p_mech_l = p_elec_l_final * eta_l
            p_mech_r = p_elec_r_final * eta_r
            t_m_l = min(p_mech_l / omega_m_l, t_peak_l)
            t_m_r = min(p_mech_r / omega_m_r, t_peak_r)
            p_elec_l = p_elec_l_final
            p_elec_r = p_elec_r_final
            p_elec_tot = p_elec_l + p_elec_r

        # 6. Final wheel torques delivered to tire contact patch
        t_wheel_rl = t_m_l * conv_factor
        t_wheel_rr = t_m_r * conv_factor

        telemetry = {
            'rpm_rl': rpm_l,
            'rpm_rr': rpm_r,
            't_motor_rl_nm': t_m_l,
            't_motor_rr_nm': t_m_r,
            't_wheel_rl_nm': t_wheel_rl,
            't_wheel_rr_nm': t_wheel_rr,
            'p_mech_rl_kw': p_mech_l / 1000.0,
            'p_mech_rr_kw': p_mech_r / 1000.0,
            'p_mech_total_kw': (p_mech_l + p_mech_r) / 1000.0,
            'eta_rl': eta_l,
            'eta_rr': eta_r,
            'p_elec_rl_kw': p_elec_l / 1000.0,
            'p_elec_rr_kw': p_elec_r / 1000.0,
            'p_elec_total_kw': p_elec_tot / 1000.0,
            'p_elec_limit_kw': p_limit_w / 1000.0,
            'is_power_limited': bool(p_elec_tot >= p_limit_w - 0.1)
        }

        return t_wheel_rl, t_wheel_rr, telemetry


# Singleton instance for quick module access
_default_powertrain = EMRAX188HV()

def get_default_powertrain() -> EMRAX188HV:
    """Returns the default EMRAX 188HV powertrain model instance."""
    return _default_powertrain

def allocate_dual_motor_torques(w_rl: float,
                                w_rr: float,
                                T_req_wheel_rl: float,
                                T_req_wheel_rr: float,
                                gear_ratio: float = 4.0,
                                p_elec_limit_kw: float = 80.0,
                                n_drivetrain: float = 0.95) -> Tuple[float, float, Dict[str, Any]]:
    """Convenience functional wrapper for torque allocation."""
    pt = EMRAX188HV(gear_ratio=gear_ratio, p_elec_limit_kw=p_elec_limit_kw, n_drivetrain=n_drivetrain)
    return pt.allocate_dual_motor_torques(w_rl, w_rr, T_req_wheel_rl, T_req_wheel_rr)


def casadi_peak_torque(rpm_sym: ca.SX) -> ca.SX:
    """
    CasADi-compatible smooth mathematical approximation of the EMRAX 188HV peak torque curve:
    - 100 Nm flat plateau up to 5500 RPM
    - Smooth constant-power field-weakening decay down to 70 Nm at 8000 RPM
    """
    # Soft transition around 5500 RPM
    t_plateau = 100.0
    # Field weakening curve: 60 kW / (RPM * 2pi / 60)
    omega_sym = ca.fmax(rpm_sym * (2.0 * ca.pi / 60.0), 1.0)
    t_power_lim = 60000.0 / omega_sym
    
    # Smooth minimum using smooth blend
    k = 0.08
    smooth_t = t_plateau - ca.log(1.0 + ca.exp(k * (t_plateau - t_power_lim))) / k
    # Upper RPM cutoff smoothly towards 0 past 8000 RPM
    cutoff = 0.5 * (1.0 + ca.tanh((8200.0 - rpm_sym) / 150.0))
    return ca.fmax(smooth_t * cutoff, 0.0)
