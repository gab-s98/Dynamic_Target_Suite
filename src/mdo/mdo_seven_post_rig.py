"""
mdo_seven_post_rig.py
---------------------
7-Post Dynamic Shaker Rig Simulation and Analysis Engine for Formula SAE & Racing Prototypes.

Features:
- Full 14-state vertical multibody vehicle subsystem:
    Chassis: Heave (z_s, w_h), Pitch (theta, w_theta), Roll (phi, w_phi)
    Hubs: 4x Unsprung Corner Travel & Vertical Velocity (z_w, v_z)
- Actuation Architecture (Classic 7-Post Rig):
    4 Road Actuators (Wheel posts 1-4): FL, FR, RL, RR vertical displacement inputs
    3 Chassis Actuators (Aero posts 5-7): 1 Front Centerline, 2 Rear Corners (Tripod layout)
- Non-linear Suspensions:
    Main springs, progressive bumpstop pack, droop stops, anti-roll bars (ARB)
    Öhlins TTX25 MkII 4-way non-linear dampers (LSC, HSC, LSR, HSR) or linear fallback
- Non-linear Unilateral Tire Contact:
    Radial stiffness K_tz(P), carcass damping C_tz, and dynamic wheel lift-off (Fz >= 0)
- Advanced Excitation Modes:
    1. Frequency Sweep (Chirp: Heave, Pitch, Roll, Warp, Single Corner) from 0.2 to 35 Hz
    2. Sine Dwell (Single Frequency Resonance / Harmonic Work Cycles)
    3. ISO 8608 Stochastic Road Roughness (Classes A, B, C, D) with axle lag
    4. Step / Kerb Strike / Pothole Impulses
    5. Track Profile Replay (telemetry/road elevation with dynamic aero load)
- Comprehensive Post-Processing & Engineering Analytics:
    - FRF Transmissibility (Magnitude [Linear/dB], Phase [deg], Coherence gamma^2 [0-1])
    - Power Spectral Density (PSD) of chassis accelerations
    - Contact Patch Load Variation (CPLV) with Low-Band (1-4 Hz) vs High-Band (8-16 Hz) split
    - Dynamic Load Factors & Estimated Grip Loss (% due to degressivity)
    - Damper Energy Dissipation (Power [W], Work [J], Compression/Rebound Asymmetry Ratio)
    - Bilstrom 5-bin Damper Velocity Histograms
    - Force-Velocity and Force-Displacement Lissajous Work Cycles
    - Aerodynamic Platform Stability (Ride height RMS/pk-pk, Pitch/Roll attitude variance)
    - Synthetic Setup Scores (Grip Score, Aero Platform Score, Composite Index)
    - Instantaneous Linear Modal Analysis (A, B state-space extraction, eigenvalues, mode shapes)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Callable, List
from scipy.signal import welch, csd, coherence, find_peaks

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.damper_model_ttx25 import calculate_wheel_damper_force


@dataclass
class SevenPostConfig:
    """Configuration settings for a 7-Post / 4-Post Rig simulation run."""
    # Rig architecture: '7_post' (4 wheel + 3 aero posts) or '4_post' (4 wheel posts only, pure mechanical)
    rig_architecture: str = '7_post'

    # Simulation mode: 'chirp', 'sine_dwell', 'iso_road', 'step_bump', 'track_replay'
    mode: str = 'chirp'
    
    # Phasing: 'heave', 'pitch', 'roll', 'warp', 'fl_only', 'fr_only', 'rl_only', 'rr_only'
    phasing: str = 'heave'
    
    # Frequency Sweep Parameters (Chirp)
    f_start: float = 0.5           # Starting frequency [Hz]
    f_end: float = 30.0            # Ending frequency [Hz]
    sweep_duration: float = 12.0   # Duration of chirp sweep [s]
    sweep_type: str = 'log'        # 'log' or 'linear'
    amplitude_mm: float = 6.0      # Nominal displacement amplitude [mm]
    v_max_actuator_mps: float = 0.25 # Max actuator velocity clamp [m/s]
    
    # Sine Dwell Parameters (Constant Frequency Test)
    dwell_frequency: float = 2.5   # Frequency for dwell resonance test [Hz]
    
    # Step / Kerb Strike Parameters
    step_type: str = 'kerb'        # 'kerb', 'pothole', 'sine_bump', 'step'
    step_height_mm: float = 10.0   # Height/Depth of bump [mm]
    step_duration_s: float = 0.08  # Duration of obstacle impact [s]
    
    # Aerodynamic / Chassis Load Settings
    aero_mode: str = 'speed_proportional' # 'constant', 'speed_proportional', 'none'
    sim_speed_mps: float = 20.0    # Virtual car speed [m/s] (~72 km/h)
    constant_downforce_n: float = 800.0 # Used if aero_mode == 'constant' [N]
    
    # ISO Road Roughness Settings
    iso_class: str = 'B'           # 'A' (smooth track), 'B' (medium bump), 'C' (rough), 'D' (severe)
    iso_duration: float = 8.0      # Duration of ISO road test [s]
    
    # Custom Track Elevation Inputs (if mode == 'track_replay')
    custom_time: Optional[np.ndarray] = None          # (N,) time array [s]
    custom_z_posts: Optional[np.ndarray] = None       # (N, 4) road elevation [m] [FL, FR, RL, RR]
    custom_speed: Optional[np.ndarray] = None         # (N,) speed profile [m/s]
    
    # Setup Overrides (allows quick parametric sweeps without altering VehicleParams file)
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Numerical Integrator Settings
    dt: float = 0.002              # Simulation time step [s] (500 Hz)


@dataclass
class SevenPostResult:
    """Comprehensive results and post-processed analytics from the 7-Post Rig simulation."""
    # Time history
    time: np.ndarray
    z_posts: np.ndarray       # (N, 4) Actuator road displacement [m] [FL, FR, RL, RR]
    dz_posts: np.ndarray      # (N, 4) Actuator road velocity [m/s]
    F_aero_posts: np.ndarray  # (N, 3) Chassis actuator forces [N] [Front, Rear-Left, Rear-Right]
    
    # Chassis Dynamic States
    z_chassis: np.ndarray     # (N, 3) [z_s (heave), theta (pitch), phi (roll)] [m, rad, rad]
    w_chassis: np.ndarray     # (N, 3) [w_heave, w_pitch, w_roll] [m/s, rad/s, rad/s]
    accel_chassis: np.ndarray # (N, 3) [z_s_ddot, theta_ddot, phi_ddot] [m/s^2, rad/s^2, rad/s^2]
    
    # Hubs / Wheels States
    z_wheels: np.ndarray      # (N, 4) Wheel hub vertical displacement [m]
    v_wheels: np.ndarray      # (N, 4) Wheel hub vertical velocity [m/s]
    
    # Suspension & Tire Loads
    stroke_susp: np.ndarray   # (N, 4) Suspension jounce [m] (+ = bump/compression)
    v_damper: np.ndarray      # (N, 4) Damper velocity [m/s] (+ = compression)
    F_spring: np.ndarray      # (N, 4) Spring + bumpstop forces [N]
    F_damper: np.ndarray      # (N, 4) Damper forces [N]
    F_susp: np.ndarray        # (N, 4) Total suspension forces on chassis [N]
    Fz_tires: np.ndarray      # (N, 4) Dynamic tire normal load at contact patch [N]
    
    # Frequency Domain & Transfer Functions (Bode / FRF)
    freqs: np.ndarray                  # Frequency axis [Hz]
    transmissibility_heave: np.ndarray # Magnitude |Z_s(f) / Z_post(f)| [-]
    phase_heave: np.ndarray            # Phase lag [deg]
    coherence_heave: np.ndarray        # Coherence gamma^2 [0-1]
    
    transmissibility_pitch: np.ndarray # Magnitude |Theta(f) / Z_post(f)| [rad/m]
    phase_pitch: np.ndarray            # Phase lag [deg]
    coherence_pitch: np.ndarray        # Coherence gamma^2 [0-1]
    
    transmissibility_roll: np.ndarray  # Magnitude |Phi(f) / Z_post(f)| [rad/m]
    phase_roll: np.ndarray             # Phase lag [deg]
    coherence_roll: np.ndarray         # Coherence gamma^2 [0-1]
    
    psd_heave: np.ndarray              # Heave acceleration PSD [(m/s^2)^2/Hz]
    psd_pitch: np.ndarray              # Pitch acceleration PSD [(rad/s^2)^2/Hz]
    psd_roll: np.ndarray               # Roll acceleration PSD [(rad/s^2)^2/Hz]
    cplv_spectrum: np.ndarray          # Corner load variation spectrum (4, len(freqs))
    
    # Key Performance Indicators (KPIs)
    cplv: Dict[str, float]             # Overall CPLV (sigma(Fz) / mean(Fz)) per corner & avg
    cplv_low_band: Dict[str, float]    # Low-Frequency Band CPLV (0.5 - 4.5 Hz, body bounce)
    cplv_high_band: Dict[str, float]   # High-Frequency Band CPLV (8.0 - 18.0 Hz, wheel hop)
    delta_fz_pkpk: Dict[str, float]    # Peak-to-Peak dynamic normal force [N]
    estimated_grip_loss_pct: Dict[str, float] # Estimated lateral/long grip loss [%]
    lift_off_duration_pct: Dict[str, float]   # Percentage of test time with Fz <= 0 [%]
    
    rms_heave_accel: float             # Heave RMS acceleration [m/s^2]
    rms_pitch_accel: float             # Pitch RMS acceleration [rad/s^2]
    rms_roll_accel: float              # Roll RMS acceleration [rad/s^2]
    
    # Aerodynamic Platform Control
    rms_ride_height_f_mm: float        # Front ride height RMS [mm]
    rms_ride_height_r_mm: float        # Rear ride height RMS [mm]
    pkpk_ride_height_f_mm: float       # Peak-to-Peak front ride height fluctuation [mm]
    pkpk_ride_height_r_mm: float       # Peak-to-Peak rear ride height fluctuation [mm]
    pkpk_pitch_deg: float              # Peak-to-Peak pitch angle [deg]
    pkpk_roll_deg: float               # Peak-to-Peak roll angle [deg]
    
    # Damper Operating Diagnostics & Energy Dissipation
    damper_power_peak_w: Dict[str, float]       # Peak damping power [W] per corner
    damper_power_rms_w: Dict[str, float]        # RMS damping power [W] per corner
    damper_energy_dissipated_j: Dict[str, float]# Total energy dissipated in dampers [J]
    damper_asymmetry_ratio: Dict[str, float]    # Work ratio Compression / Rebound (E_comp / E_reb)
    bilstrom_5bin: Dict[str, Dict[str, float]]  # Bilstrom 5-bin velocity histogram percentages
    damper_velocity_histogram: Dict[str, Dict[str, float]] # 3-bin legacy format
    bumpstop_strikes: Dict[str, int]            # Number of bumpstop contact events per corner
    wheel_lift_off_events: Dict[str, int]       # Number of zero normal force instances
    
    # Synthetic Performance Scores (0 - 100)
    score_grip: float                  # Mechanical Grip Score (100 = optimal low CPLV)
    score_aero: float                  # Aerodynamic Platform Stability Score
    score_overall: float               # Weighted composite score
    resonance_peaks: List[Dict[str, Any]] # Identified natural resonance peaks
    rig_architecture: str = '7_post'   # '7_post' or '4_post'
    model_comparison: Optional[Dict[str, Any]] = None # Benchmark comparison with linear vehicle model


class SevenPostRig:
    """
    7-Post Shaker Rig Simulation and Analysis Engine.
    """

    def __init__(self, vp: Optional[VehicleParams] = None,
                 tp_f: Optional[TireParams] = None,
                 tp_r: Optional[TireParams] = None):
        self.vp = vp or VehicleParams()
        self.tp_f = tp_f
        self.tp_r = tp_r or tp_f
        self._cache_parameters()

    def _cache_parameters(self, overrides: Optional[Dict[str, Any]] = None):
        """Extract and cache physical parameters, applying any runtime setup overrides."""
        vp = self.vp
        ov = overrides or {}

        self.m = float(ov.get('m', vp.m))
        self.g = float(vp.g)
        self.L = float(ov.get('L', vp.L))
        self.tf = float(ov.get('t_f', vp.t_f))
        self.tr = float(ov.get('t_r', vp.t_r))
        self.wd_f = float(ov.get('weight_dist_f', getattr(vp, 'weight_dist_f', getattr(vp, 'wd_f', 0.46))))
        self.a = self.L * (1.0 - self.wd_f)
        self.b = self.L * self.wd_f

        # Inertias
        self.Ix = float(ov.get('Ix', getattr(vp, 'Ix', 45.0)))
        self.Iy = float(ov.get('Iy', getattr(vp, 'Iy', 125.0)) )
        self.Iz = float(ov.get('Iz', getattr(vp, 'Iz', 140.0)))

        # Unsprung masses [kg] per corner
        self.m_u_f = float(getattr(vp, 'm_us_f', 18.0))
        self.m_u_r = float(getattr(vp, 'm_us_r', 16.0))
        self.m_s = self.m - 2.0 * self.m_u_f - 2.0 * self.m_u_r
        if self.m_s <= 50.0:
            self.m_s = self.m * 0.80

        # Wheel rates / Spring rates [N/m]
        self.K_z_f = float(ov.get('K_z_f', getattr(vp, 'K_z_f', 40000.0)))
        self.K_z_r = float(ov.get('K_z_r', getattr(vp, 'K_z_r', 45000.0)))

        # Anti-roll bar roll stiffness [Nm/rad]
        self.K_phi_f = float(ov.get('K_phi_f', getattr(vp, 'K_phi_f', 600.0)))
        self.K_phi_r = float(ov.get('K_phi_r', getattr(vp, 'K_phi_r', 450.0)))
        self.K_arb_f = max(0.0, self.K_phi_f - self.K_z_f * (self.tf**2) / 2.0)
        self.K_arb_r = max(0.0, self.K_phi_r - self.K_z_r * (self.tr**2) / 2.0)
        self.Kw_arb_f = self.K_arb_f / (self.tf**2 + 1e-6)
        self.Kw_arb_r = self.K_arb_r / (self.tr**2 + 1e-6)

        # Motion ratios
        self.mr_f = float(ov.get('motion_ratio_f', getattr(vp, 'motion_ratio_f', 0.85)))
        self.mr_r = float(ov.get('motion_ratio_r', getattr(vp, 'motion_ratio_r', 0.85)))

        # Tire radial stiffness & damping
        K_tz_12psi = float(getattr(vp, 'tire_vertical_stiffness_12psi', 75831.64))
        p_f_bar = float(getattr(vp, 'tire_pressure_f_bar', 0.82737))
        p_r_bar = float(getattr(vp, 'tire_pressure_r_bar', 0.82737))
        self.K_tz_f = K_tz_12psi * (p_f_bar / 0.82737)
        self.K_tz_r = K_tz_12psi * (p_r_bar / 0.82737)
        self.C_tz = float(getattr(vp, 'C_tire_z', 450.0))

        # Damper configuration
        self.damper_mode = getattr(vp, 'damper_mode', 'ttx25_nonlinear')
        self.valving_f = getattr(vp, 'damper_valving_f', 'C12/R12')
        self.valving_r = getattr(vp, 'damper_valving_r', 'C12/R12')
        self.lsc_f = float(ov.get('lsc_f', getattr(vp, 'damper_lsc_clicks_f', 10.0)))
        self.hsc_f = float(ov.get('hsc_f', getattr(vp, 'damper_hsc_turns_f', 2.0)))
        self.lsr_f = float(ov.get('lsr_f', getattr(vp, 'damper_lsr_clicks_f', 10.0)))
        self.hsr_f = float(ov.get('hsr_f', getattr(vp, 'damper_hsr_turns_f', 2.0)))
        self.lsc_r = float(ov.get('lsc_r', getattr(vp, 'damper_lsc_clicks_r', 10.0)))
        self.hsc_r = float(ov.get('hsc_r', getattr(vp, 'damper_hsc_turns_r', 2.0)))
        self.lsr_r = float(ov.get('lsr_r', getattr(vp, 'damper_lsr_clicks_r', 10.0)))
        self.hsr_r = float(ov.get('hsr_r', getattr(vp, 'damper_hsr_turns_r', 2.0)))
        self.C_z_f_lin = float(getattr(vp, 'C_z_f', 1500.0))
        self.C_z_r_lin = float(getattr(vp, 'C_z_r', 1200.0))

        # Bump stops and pack
        s_shock = float(getattr(vp, 'damper_stroke_total_mm', 57.0)) / 1000.0

        # Sprung mass center of gravity position and arms
        self.a_s = (self.m * self.a - 2.0 * self.m_u_r * self.L) / self.m_s
        self.b_s = self.L - self.a_s

        # Static suspension preloads (supports sprung mass weight at 1g equilibrium)
        self.F_susp_0_f = self.m_s * self.g * (self.b_s / self.L)
        self.F_susp_0_r = self.m_s * self.g * (self.a_s / self.L)
        self.F_susp_0_fl = self.F_susp_0_f / 2.0
        self.F_susp_0_fr = self.F_susp_0_f / 2.0
        self.F_susp_0_rl = self.F_susp_0_r / 2.0
        self.F_susp_0_rr = self.F_susp_0_r / 2.0

        # Static 1g reference ground loads
        self.Fz_0_fl = self.F_susp_0_fl + self.m_u_f * self.g
        self.Fz_0_fr = self.F_susp_0_fr + self.m_u_f * self.g
        self.Fz_0_rl = self.F_susp_0_rl + self.m_u_r * self.g
        self.Fz_0_rr = self.F_susp_0_rr + self.m_u_r * self.g
        self.Fz_0_f = self.Fz_0_fl + self.Fz_0_fr
        self.Fz_0_r = self.Fz_0_rl + self.Fz_0_rr

        # Static tire deflections
        self.defl_0_fl = self.Fz_0_fl / self.K_tz_f
        self.defl_0_fr = self.Fz_0_fr / self.K_tz_f
        self.defl_0_rl = self.Fz_0_rl / self.K_tz_r
        self.defl_0_rr = self.Fz_0_rr / self.K_tz_r

        # Bump stop clearances from 1g static ride height
        z_sag_f = self.F_susp_0_fl / max(self.K_z_f, 1e3)
        z_sag_r = self.F_susp_0_rl / max(self.K_z_r, 1e3)
        self.gap_f = float(ov.get('gap_f', max(0.005, (s_shock - 0.005) / self.mr_f - z_sag_f)))
        self.gap_r = float(ov.get('gap_r', max(0.005, (s_shock - 0.005) / self.mr_r - z_sag_r)))
        self.k_bs_f = float(getattr(vp, 'bumpstop_k_f', 80000.0))
        self.k_bs_r = float(getattr(vp, 'bumpstop_k_r', 80000.0))

        # Aerodynamics
        self.rho = float(getattr(vp, 'rho', 1.225))
        self.S = float(getattr(vp, 'S', 1.139))
        self.Cz = float(getattr(vp, 'Cz', 3.421))
        self.aero_bal = float(ov.get('aero_bal', getattr(vp, 'aero_balance', 0.45)))

    def _calc_spring_force(self, z_jounce: float, k_z: float, gap_b: float, k_bs: float) -> float:
        """Calculates spring force including progressive polyurethane bumpstop engagement."""
        f_lin = k_z * z_jounce
        if z_jounce > gap_b:
            dz = z_jounce - gap_b
            f_bs = k_bs * dz * (1.0 + 15.0 * dz)
            if dz > 0.005:
                dz_pack = dz - 0.005
                f_bs += 1.5e6 * dz_pack * (1.0 + 100.0 * dz_pack)
            return f_lin + f_bs
        elif z_jounce < -0.040:
            dz_d = -z_jounce - 0.040
            return f_lin - k_bs * dz_d * (1.0 + 20.0 * dz_d)
        return f_lin

    def _calc_damper_force(self, v_jounce: float, is_front: bool) -> float:
        """Calculates corner damping force with TTX25 4-way non-linear model or linear fallback."""
        if self.damper_mode in ('ttx25_nonlinear', 'custom_dyno', 'Custom Dyno Curve'):
            mr = self.mr_f if is_front else self.mr_r
            valving = self.valving_f if is_front else self.valving_r
            lsc = self.lsc_f if is_front else self.lsc_r
            hsc = self.hsc_f if is_front else self.hsc_r
            lsr = self.lsr_f if is_front else self.lsr_r
            hsr = self.hsr_f if is_front else self.hsr_r
            return calculate_wheel_damper_force(v_jounce, mr, valving, lsc, hsc, lsr, hsr, use_casadi=False)
        else:
            c_lin = self.C_z_f_lin if is_front else self.C_z_r_lin
            return c_lin * v_jounce

    def generate_excitation(self, config: SevenPostConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates road actuator profiles z_posts(t), dz_posts(t) and chassis aero forces F_aero(t).
        """
        dt = config.dt
        amp = config.amplitude_mm * 1e-3
        v_max = config.v_max_actuator_mps

        is_4post = getattr(config, 'rig_architecture', '7_post') == '4_post'
        if is_4post:
            config.aero_mode = 'none'

        # Phasing signs across [FL, FR, RL, RR]
        phasing_map = {
            'heave': np.array([1.0, 1.0, 1.0, 1.0]),
            'pitch': np.array([1.0, 1.0, -1.0, -1.0]),
            'roll': np.array([1.0, -1.0, 1.0, -1.0]),
            'warp': np.array([1.0, -1.0, -1.0, 1.0]),
            'fl_only': np.array([1.0, 0.0, 0.0, 0.0]),
            'fr_only': np.array([0.0, 1.0, 0.0, 0.0]),
            'rl_only': np.array([0.0, 0.0, 1.0, 0.0]),
            'rr_only': np.array([0.0, 0.0, 0.0, 1.0])
        }
        signs = phasing_map.get(config.phasing, np.array([1.0, 1.0, 1.0, 1.0]))

        if config.mode == 'chirp':
            T = config.sweep_duration
            N = int(math.ceil(T / dt)) + 1
            t = np.linspace(0.0, T, N)
            f0 = config.f_start
            f1 = config.f_end

            if config.sweep_type == 'log':
                beta = (f1 / f0) ** (1.0 / T)
                phase = 2.0 * np.pi * f0 * (beta**t - 1.0) / np.log(beta)
                freq_inst = f0 * (beta**t)
            else:
                k = (f1 - f0) / T
                phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t**2)
                freq_inst = f0 + k * t

            amp_eff = np.minimum(amp, v_max / (2.0 * np.pi * np.maximum(freq_inst, 0.1)))
            taper_len = int(0.05 * N)
            window = np.ones(N)
            if taper_len > 1:
                window[:taper_len] = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / taper_len))
                window[-taper_len:] = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len)[::-1] / taper_len))

            raw_disp = amp_eff * np.sin(phase) * window
            raw_vel = np.gradient(raw_disp, dt)
            z_posts = np.outer(raw_disp, signs)
            dz_posts = np.outer(raw_vel, signs)

            V = config.sim_speed_mps
            F_down = 0.5 * self.rho * self.S * self.Cz * (V**2) if config.aero_mode == 'speed_proportional' else (
                config.constant_downforce_n if config.aero_mode == 'constant' else 0.0
            )
            F_post_f = F_down * self.aero_bal
            F_post_rl = 0.5 * F_down * (1.0 - self.aero_bal)
            F_post_rr = 0.5 * F_down * (1.0 - self.aero_bal)
            F_aero_posts = np.tile([F_post_f, F_post_rl, F_post_rr], (N, 1))

        elif config.mode == 'sine_dwell':
            # Pure single-frequency resonance test for stationary work cycle analysis
            T = max(config.sweep_duration, 4.0)
            N = int(math.ceil(T / dt)) + 1
            t = np.linspace(0.0, T, N)
            f_d = config.dwell_frequency
            omega = 2.0 * np.pi * f_d

            # Smooth 0.5s ramp at start
            ramp_len = int(0.5 / dt)
            envelope = np.ones(N)
            if ramp_len > 1 and ramp_len < N:
                envelope[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

            raw_disp = amp * np.sin(omega * t) * envelope
            raw_vel = amp * omega * np.cos(omega * t) * envelope
            z_posts = np.outer(raw_disp, signs)
            dz_posts = np.outer(raw_vel, signs)

            V = config.sim_speed_mps
            F_down = 0.5 * self.rho * self.S * self.Cz * (V**2) if config.aero_mode != 'none' else 0.0
            F_aero_posts = np.tile([F_down * self.aero_bal,
                                    0.5 * F_down * (1.0 - self.aero_bal),
                                    0.5 * F_down * (1.0 - self.aero_bal)], (N, 1))

        elif config.mode == 'step_bump':
            # Single kerb strike, pothole, or impulse
            T = max(config.sweep_duration, 3.5)
            N = int(math.ceil(T / dt)) + 1
            t = np.linspace(0.0, T, N)
            z_posts = np.zeros((N, 4))
            dz_posts = np.zeros((N, 4))

            # Amplitude resolution: step_height_mm or amplitude_mm
            if config.step_height_mm == 0.0 or config.amplitude_mm == 0.0:
                h_bump = 0.0
            elif config.step_height_mm != 10.0:
                h_bump = config.step_height_mm * 1e-3
            else:
                h_bump = config.amplitude_mm * 1e-3 if config.amplitude_mm != 6.0 else config.step_height_mm * 1e-3

            t_impact = 0.5
            d_impact = config.step_duration_s

            idx_start = int(round(t_impact / dt))
            idx_dur = max(2, int(round(d_impact / dt)))
            idx_end = min(N, idx_start + idx_dur)

            # Rear axle impact lag based on car wheelbase: tau = L / V
            V = max(config.sim_speed_mps, 5.0)
            lag_steps = int(round(self.L / (V * dt)))

            # Front profile
            if config.step_type == 'kerb' or config.step_type == 'sine_bump':
                t_local = np.linspace(0, np.pi, idx_dur)
                bump_shape = h_bump * np.sin(t_local)
            elif config.step_type == 'pothole':
                t_local = np.linspace(0, np.pi, idx_dur)
                bump_shape = -h_bump * np.sin(t_local)
            else:  # step
                bump_shape = np.full(idx_dur, h_bump)

            # Apply to front wheels
            len_f = idx_end - idx_start
            if idx_start < N and len_f > 0:
                if signs[0] != 0: z_posts[idx_start:idx_end, 0] = bump_shape[:len_f] * signs[0]
                if signs[1] != 0: z_posts[idx_start:idx_end, 1] = bump_shape[:len_f] * signs[1]

            # Apply to rear wheels with axle lag
            r_start = idx_start + lag_steps
            r_end = min(N, r_start + idx_dur)
            if r_start < N:
                len_r = r_end - r_start
                if signs[2] != 0: z_posts[r_start:r_end, 2] = bump_shape[:len_r] * signs[2]
                if signs[3] != 0: z_posts[r_start:r_end, 3] = bump_shape[:len_r] * signs[3]

            for i in range(4):
                dz_posts[:, i] = np.gradient(z_posts[:, i], dt)

            F_down = 0.5 * self.rho * self.S * self.Cz * (V**2) if config.aero_mode != 'none' else 0.0
            F_aero_posts = np.tile([F_down * self.aero_bal,
                                    0.5 * F_down * (1.0 - self.aero_bal),
                                    0.5 * F_down * (1.0 - self.aero_bal)], (N, 1))

        elif config.mode == 'iso_road':
            T = config.iso_duration
            N = int(math.ceil(T / dt)) + 1
            t = np.linspace(0.0, T, N)
            V = max(config.sim_speed_mps, 5.0)

            g0_map = {'A': 4e-6, 'B': 16e-6, 'C': 64e-6, 'D': 256e-6}
            G0 = g0_map.get(config.iso_class.upper(), 16e-6)
            n0 = 0.1
            
            np.random.seed(42)
            n_harmonics = 120
            spatial_freqs = np.linspace(0.02, 10.0, n_harmonics)
            delta_n = spatial_freqs[1] - spatial_freqs[0]
            psd = G0 * (spatial_freqs / n0) ** (-2.0)
            amplitudes = np.sqrt(2.0 * psd * delta_n)

            phases_l = np.random.uniform(0, 2.0 * np.pi, n_harmonics)
            phases_r = phases_l + np.random.uniform(-0.65, 0.65, n_harmonics)

            x = V * t
            z_left = np.zeros(N)
            z_right = np.zeros(N)
            for j in range(n_harmonics):
                omega_s = 2.0 * np.pi * spatial_freqs[j]
                z_left += amplitudes[j] * np.sin(omega_s * x + phases_l[j])
                z_right += amplitudes[j] * np.sin(omega_s * x + phases_r[j])

            lag_steps = int(round(self.L / (V * dt)))
            z_fl = z_left
            z_fr = z_right
            z_rl = np.roll(z_left, lag_steps)
            z_rr = np.roll(z_right, lag_steps)
            z_rl[:lag_steps] = 0.0
            z_rr[:lag_steps] = 0.0

            z_posts = np.column_stack([z_fl, z_fr, z_rl, z_rr])
            dz_posts = np.zeros_like(z_posts)
            for i in range(4):
                dz_posts[:, i] = np.gradient(z_posts[:, i], dt)

            F_down = 0.5 * self.rho * self.S * self.Cz * (V**2) if config.aero_mode != 'none' else 0.0
            F_aero_posts = np.tile([F_down * self.aero_bal,
                                    0.5 * F_down * (1.0 - self.aero_bal),
                                    0.5 * F_down * (1.0 - self.aero_bal)], (N, 1))

        elif config.mode == 'track_replay' and config.custom_time is not None:
            t = config.custom_time
            N = len(t)
            z_posts = config.custom_z_posts if config.custom_z_posts is not None else np.zeros((N, 4))
            dz_posts = np.zeros_like(z_posts)
            for i in range(4):
                dz_posts[:, i] = np.gradient(z_posts[:, i], t)

            V_arr = config.custom_speed if config.custom_speed is not None else np.full(N, config.sim_speed_mps)
            F_down_arr = 0.5 * self.rho * self.S * self.Cz * (V_arr**2) if config.aero_mode != 'none' else np.zeros(N)
            F_aero_posts = np.zeros((N, 3))
            F_aero_posts[:, 0] = F_down_arr * self.aero_bal
            F_aero_posts[:, 1] = 0.5 * F_down_arr * (1.0 - self.aero_bal)
            F_aero_posts[:, 2] = 0.5 * F_down_arr * (1.0 - self.aero_bal)

        else:
            T = 3.0
            N = int(math.ceil(T / dt)) + 1
            t = np.linspace(0.0, T, N)
            z_posts = np.zeros((N, 4))
            dz_posts = np.zeros((N, 4))
            step_idx = int(0.5 / dt)
            z_posts[step_idx:, :] = amp
            dz_posts[step_idx:step_idx + 5, :] = amp / (5.0 * dt)
            F_aero_posts = np.zeros((N, 3))

        if is_4post:
            F_aero_posts = np.zeros((len(t), 3))

        return t, z_posts, dz_posts, F_aero_posts

    def simulate(self, config: Optional[SevenPostConfig] = None,
                 progress_callback: Optional[Callable[[int, str], None]] = None) -> SevenPostResult:
        """
        Runs the full 7-Post Rig simulation with advanced spectral analytics and energy accounting.
        """
        cfg = config or SevenPostConfig()
        self._cache_parameters(cfg.overrides)

        if progress_callback:
            progress_callback(5, "Generazione profili eccitazione e precarichi banco...")

        t_arr, z_p_arr, dz_p_arr, F_aero_arr = self.generate_excitation(cfg)
        N = len(t_arr)
        dt = cfg.dt

        x0 = np.zeros(14)

        if progress_callback:
            progress_callback(15, f"Integrazione numerica RK4 a 500 Hz ({N} time steps)...")

        x_history = np.zeros((N, 14))
        accel_history = np.zeros((N, 3))
        F_susp_history = np.zeros((N, 4))
        F_spring_history = np.zeros((N, 4))
        F_damper_history = np.zeros((N, 4))
        Fz_tires_history = np.zeros((N, 4))
        stroke_history = np.zeros((N, 4))
        v_damper_history = np.zeros((N, 4))

        def ode_eval(t, x, z_p, dz_p, F_aero):
            zs, wh, phi, wphi, theta, wtheta, zw_fl, vw_fl, zw_fr, vw_fr, zw_rl, vw_rl, zw_rr, vw_rr = x

            z_c_fl = zs - theta * self.a_s + phi * (self.tf / 2.0)
            z_c_fr = zs - theta * self.a_s - phi * (self.tf / 2.0)
            z_c_rl = zs + theta * self.b_s + phi * (self.tr / 2.0)
            z_c_rr = zs + theta * self.b_s - phi * (self.tr / 2.0)

            v_c_fl = wh - wtheta * self.a_s + wphi * (self.tf / 2.0)
            v_c_fr = wh - wtheta * self.a_s - wphi * (self.tf / 2.0)
            v_c_rl = wh + wtheta * self.b_s + wphi * (self.tr / 2.0)
            v_c_rr = wh + wtheta * self.b_s - wphi * (self.tr / 2.0)

            z_j_fl = zw_fl - z_c_fl; z_j_fr = zw_fr - z_c_fr
            z_j_rl = zw_rl - z_c_rl; z_j_rr = zw_rr - z_c_rr
            v_j_fl = vw_fl - v_c_fl; v_j_fr = vw_fr - v_c_fr
            v_j_rl = vw_rl - v_c_rl; v_j_rr = vw_rr - v_c_rr

            F_sp_fl = self._calc_spring_force(z_j_fl, self.K_z_f, self.gap_f, self.k_bs_f)
            F_sp_fr = self._calc_spring_force(z_j_fr, self.K_z_f, self.gap_f, self.k_bs_f)
            F_sp_rl = self._calc_spring_force(z_j_rl, self.K_z_r, self.gap_r, self.k_bs_r)
            F_sp_rr = self._calc_spring_force(z_j_rr, self.K_z_r, self.gap_r, self.k_bs_r)

            d_z_f = z_j_fl - z_j_fr; d_z_r = z_j_rl - z_j_rr
            F_arb_fl = self.Kw_arb_f * d_z_f; F_arb_fr = -self.Kw_arb_f * d_z_f
            F_arb_rl = self.Kw_arb_r * d_z_r; F_arb_rr = -self.Kw_arb_r * d_z_r

            F_d_fl = self._calc_damper_force(v_j_fl, is_front=True)
            F_d_fr = self._calc_damper_force(v_j_fr, is_front=True)
            F_d_rl = self._calc_damper_force(v_j_rl, is_front=False)
            F_d_rr = self._calc_damper_force(v_j_rr, is_front=False)

            F_su_fl = self.F_susp_0_fl + F_sp_fl + F_d_fl + F_arb_fl
            F_su_fr = self.F_susp_0_fr + F_sp_fr + F_d_fr + F_arb_fr
            F_su_rl = self.F_susp_0_rl + F_sp_rl + F_d_rl + F_arb_rl
            F_su_rr = self.F_susp_0_rr + F_sp_rr + F_d_rr + F_arb_rr

            defl_fl = self.defl_0_fl - (zw_fl - z_p[0])
            defl_fr = self.defl_0_fr - (zw_fr - z_p[1])
            defl_rl = self.defl_0_rl - (zw_rl - z_p[2])
            defl_rr = self.defl_0_rr - (zw_rr - z_p[3])

            Fz_fl = max(0.0, self.K_tz_f * defl_fl + self.C_tz * (dz_p[0] - vw_fl))
            Fz_fr = max(0.0, self.K_tz_f * defl_fr + self.C_tz * (dz_p[1] - vw_fr))
            Fz_rl = max(0.0, self.K_tz_r * defl_rl + self.C_tz * (dz_p[2] - vw_rl))
            Fz_rr = max(0.0, self.K_tz_r * defl_rr + self.C_tz * (dz_p[3] - vw_rr))

            F_aero_tot = F_aero[0] + F_aero[1] + F_aero[2]
            M_pitch_aero = self.a_s * F_aero[0] - self.b_s * (F_aero[1] + F_aero[2])
            M_roll_aero = (self.tr / 2.0) * (F_aero[1] - F_aero[2])

            zs_ddot = (F_su_fl + F_su_fr + F_su_rl + F_su_rr - F_aero_tot - self.m_s * self.g) / self.m_s
            phi_ddot = ((self.tf / 2.0) * (F_su_fl - F_su_fr) +
                        (self.tr / 2.0) * (F_su_rl - F_su_rr) + M_roll_aero) / self.Ix
            theta_ddot = (-self.a_s * (F_su_fl + F_su_fr) +
                          self.b_s * (F_su_rl + F_su_rr) + M_pitch_aero) / self.Iy

            vw_fl_dot = (Fz_fl - F_su_fl - self.m_u_f * self.g) / self.m_u_f
            vw_fr_dot = (Fz_fr - F_su_fr - self.m_u_f * self.g) / self.m_u_f
            vw_rl_dot = (Fz_rl - F_su_rl - self.m_u_r * self.g) / self.m_u_r
            vw_rr_dot = (Fz_rr - F_su_rr - self.m_u_r * self.g) / self.m_u_r

            dx = np.array([
                wh, zs_ddot,
                wphi, phi_ddot,
                wtheta, theta_ddot,
                vw_fl, vw_fl_dot,
                vw_fr, vw_fr_dot,
                vw_rl, vw_rl_dot,
                vw_rr, vw_rr_dot
            ])

            extra = (
                np.array([zs_ddot, theta_ddot, phi_ddot]),
                np.array([F_su_fl, F_su_fr, F_su_rl, F_su_rr]),
                np.array([F_sp_fl, F_sp_fr, F_sp_rl, F_sp_rr]),
                np.array([F_d_fl, F_d_fr, F_d_rl, F_d_rr]),
                np.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]),
                np.array([z_j_fl, z_j_fr, z_j_rl, z_j_rr]),
                np.array([v_j_fl, v_j_fr, v_j_rl, v_j_rr])
            )
            return dx, extra

        curr_x = x0.copy()
        report_step = max(1, N // 20)

        for step in range(N):
            t_curr = t_arr[step]
            z_p, dz_p, F_ae = z_p_arr[step], dz_p_arr[step], F_aero_arr[step]

            dx1, extra = ode_eval(t_curr, curr_x, z_p, dz_p, F_ae)
            accel_history[step] = extra[0]
            F_susp_history[step] = extra[1]
            F_spring_history[step] = extra[2]
            F_damper_history[step] = extra[3]
            Fz_tires_history[step] = extra[4]
            stroke_history[step] = extra[5]
            v_damper_history[step] = extra[6]
            x_history[step] = curr_x

            if step < N - 1:
                k1 = dx1
                k2, _ = ode_eval(t_curr + 0.5 * dt, curr_x + 0.5 * dt * k1, z_p, dz_p, F_ae)
                k3, _ = ode_eval(t_curr + 0.5 * dt, curr_x + 0.5 * dt * k2, z_p, dz_p, F_ae)
                k4, _ = ode_eval(t_curr + dt, curr_x + dt * k3, z_p, dz_p, F_ae)
                curr_x += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if progress_callback and (step % report_step == 0 or step == N - 1):
                pct = int(15 + 65 * (step / N))
                progress_callback(pct, f"Simulazione 7-post: {int(t_curr):2d}/{int(t_arr[-1])} s ({pct}%)")

        if progress_callback:
            progress_callback(80, "Post-processing avanzato: FRF, Bode, Coerenza e Fase...")

        z_s = x_history[:, 0]; w_h = x_history[:, 1]
        phi = x_history[:, 2]; w_phi = x_history[:, 3]
        theta = x_history[:, 4]; w_theta = x_history[:, 5]

        z_wheels = np.column_stack([x_history[:, 6], x_history[:, 8], x_history[:, 10], x_history[:, 12]])
        v_wheels = np.column_stack([x_history[:, 7], x_history[:, 9], x_history[:, 11], x_history[:, 13]])

        # -------------------------------------------------------------
        # 1. Advanced Spectral Processing: PSD, FRF Magnitude, Phase & Coherence
        # -------------------------------------------------------------
        fs = 1.0 / dt
        nperseg = min(N, max(64, int(fs * 2.0)))

        z_p_mean = np.mean(z_p_arr, axis=1)
        z_p_ddot = np.gradient(np.gradient(z_p_mean, dt), dt)

        freqs, P_input = welch(z_p_ddot, fs=fs, nperseg=nperseg)
        _, P_heave = welch(accel_history[:, 0], fs=fs, nperseg=nperseg)
        _, P_pitch = welch(accel_history[:, 1], fs=fs, nperseg=nperseg)
        _, P_roll = welch(accel_history[:, 2], fs=fs, nperseg=nperseg)

        P_input_safe = np.maximum(P_input, 1e-12)
        trans_heave = np.sqrt(P_heave / P_input_safe)
        trans_pitch = np.sqrt(P_pitch / P_input_safe)
        trans_roll = np.sqrt(P_roll / P_input_safe)

        # Cross-Spectral Density for exact phase lag [deg]
        _, C_heave_cross = csd(accel_history[:, 0], z_p_ddot, fs=fs, nperseg=nperseg)
        _, C_pitch_cross = csd(accel_history[:, 1], z_p_ddot, fs=fs, nperseg=nperseg)
        _, C_roll_cross = csd(accel_history[:, 2], z_p_ddot, fs=fs, nperseg=nperseg)
        phase_heave = np.angle(C_heave_cross, deg=True)
        phase_pitch = np.angle(C_pitch_cross, deg=True)
        phase_roll = np.angle(C_roll_cross, deg=True)

        # Coherence function gamma^2 in [0, 1]
        _, coh_heave = coherence(accel_history[:, 0], z_p_ddot, fs=fs, nperseg=nperseg)
        _, coh_pitch = coherence(accel_history[:, 1], z_p_ddot, fs=fs, nperseg=nperseg)
        _, coh_roll = coherence(accel_history[:, 2], z_p_ddot, fs=fs, nperseg=nperseg)

        # CPLV spectrum per corner
        cplv_spec = np.zeros((4, len(freqs)))
        for i in range(4):
            _, P_fz = welch(Fz_tires_history[:, i] - np.mean(Fz_tires_history[:, i]), fs=fs, nperseg=nperseg)
            cplv_spec[i] = np.sqrt(P_fz / P_input_safe)

        # Natural resonance peak detection
        peaks_found = []
        valid_range = (freqs >= 0.5) & (freqs <= 30.0)
        f_sub = freqs[valid_range]
        th_sub = trans_heave[valid_range]
        tp_sub = trans_pitch[valid_range]

        pk_indices, _ = find_peaks(th_sub, prominence=0.15)
        for p in pk_indices:
            peaks_found.append({
                'frequency_hz': float(f_sub[p]),
                'magnitude': float(th_sub[p]),
                'mode': 'Heave / Bounce' if f_sub[p] < 5.0 else 'Wheel Hop'
            })

        pk_pitch_indices, _ = find_peaks(tp_sub, prominence=0.15)
        for p in pk_pitch_indices:
            peaks_found.append({
                'frequency_hz': float(f_sub[p]),
                'magnitude': float(tp_sub[p]),
                'mode': 'Chassis Pitch'
            })

        # -------------------------------------------------------------
        # 2. Key Performance Indicators (KPIs) & Band Decompositions
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(90, "Calcolo CPLV a bande, energia smorzatori e score di aderenza...")

        transient_steps = int(0.5 / dt)
        valid_slice = slice(transient_steps, None) if N > transient_steps * 2 else slice(0, None)
        corners = ['fl', 'fr', 'rl', 'rr']

        cplv_dict = {}
        cplv_low_band = {}
        cplv_high_band = {}
        delta_fz_pkpk = {}
        estimated_grip_loss = {}
        lift_off_pct = {}

        # Frequency band masks for CPLV
        low_band_mask = (freqs >= 0.5) & (freqs <= 4.5)
        high_band_mask = (freqs >= 8.0) & (freqs <= 18.0)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        for i, c in enumerate(corners):
            fz_series = Fz_tires_history[valid_slice, i]
            mean_fz = max(np.mean(fz_series), 1.0)
            std_fz = np.std(fz_series)

            cplv_val = float(std_fz / mean_fz)
            cplv_dict[c] = cplv_val

            # Band integration
            var_low = np.sum(cplv_spec[i, low_band_mask]**2) * df
            var_high = np.sum(cplv_spec[i, high_band_mask]**2) * df
            cplv_low_band[c] = float(np.sqrt(max(0.0, var_low)) / mean_fz)
            cplv_high_band[c] = float(np.sqrt(max(0.0, var_high)) / mean_fz)

            delta_fz_pkpk[c] = float(np.max(fz_series) - np.min(fz_series))
            lift_off_pct[c] = float(np.sum(fz_series < 1.0) / len(fz_series) * 100.0)
            
            # Estimated lateral grip loss from non-linear degressivity (approx. 50 * CPLV^2 %)
            estimated_grip_loss[c] = float(min(45.0, 50.0 * (cplv_val**2)))

        cplv_dict['avg'] = float(np.mean([cplv_dict[c] for c in corners]))
        cplv_low_band['avg'] = float(np.mean([cplv_low_band[c] for c in corners]))
        cplv_high_band['avg'] = float(np.mean([cplv_high_band[c] for c in corners]))
        estimated_grip_loss['avg'] = float(np.mean([estimated_grip_loss[c] for c in corners]))

        rms_heave = float(np.sqrt(np.mean(accel_history[valid_slice, 0]**2)))
        rms_pitch = float(np.sqrt(np.mean(accel_history[valid_slice, 1]**2)))
        rms_roll = float(np.sqrt(np.mean(accel_history[valid_slice, 2]**2)))

        # Aerodynamic platform control
        h_f = z_s[valid_slice] - theta[valid_slice] * self.a_s
        h_r = z_s[valid_slice] + theta[valid_slice] * self.b_s
        rms_rh_f = float(np.std(h_f) * 1e3)
        rms_rh_r = float(np.std(h_r) * 1e3)
        pkpk_rh_f = float((np.max(h_f) - np.min(h_f)) * 1e3)
        pkpk_rh_r = float((np.max(h_r) - np.min(h_r)) * 1e3)
        pkpk_pitch_deg = float(math.degrees(np.max(theta[valid_slice]) - np.min(theta[valid_slice])))
        pkpk_roll_deg = float(math.degrees(np.max(phi[valid_slice]) - np.min(phi[valid_slice])))

        # -------------------------------------------------------------
        # 3. Damper Energy Dissipation & Velocity Regimes (Bilstrom 5-bin)
        # -------------------------------------------------------------
        p_peak_w = {}
        p_rms_w = {}
        e_diss_j = {}
        asym_ratio = {}
        bilstrom_5bin = {}
        damper_hist_3bin = {}
        bump_strikes = {}
        lift_offs = {}

        for i, c in enumerate(corners):
            v_d = v_damper_history[valid_slice, i]
            f_d = F_damper_history[valid_slice, i]
            power_inst = f_d * v_d  # [W] (always positive dissipation)

            p_peak_w[c] = float(np.max(power_inst))
            p_rms_w[c] = float(np.sqrt(np.mean(power_inst**2)))
            e_diss_j[c] = float(np.sum(power_inst) * dt)

            # Compression work vs Rebound work
            mask_comp = v_d > 0
            mask_reb = v_d < 0
            w_comp = float(np.sum(power_inst[mask_comp]) * dt) if np.any(mask_comp) else 1e-3
            w_reb = float(np.sum(power_inst[mask_reb]) * dt) if np.any(mask_reb) else 1e-3
            asym_ratio[c] = float(w_comp / max(w_reb, 1e-6))

            # Bilstrom 5-bin speed distribution [%]
            abs_v_mms = np.abs(v_d) * 1e3
            n_tot = max(1, len(abs_v_mms))
            bilstrom_5bin[c] = {
                'ultra_low': float(np.sum(abs_v_mms < 10.0) / n_tot * 100.0),
                'low': float(np.sum((abs_v_mms >= 10.0) & (abs_v_mms < 25.0)) / n_tot * 100.0),
                'mid': float(np.sum((abs_v_mms >= 25.0) & (abs_v_mms < 50.0)) / n_tot * 100.0),
                'mid_high': float(np.sum((abs_v_mms >= 50.0) & (abs_v_mms < 100.0)) / n_tot * 100.0),
                'high': float(np.sum(abs_v_mms >= 100.0) / n_tot * 100.0)
            }

            # 3-bin legacy format
            damper_hist_3bin[c] = {
                'low_pct': float(np.sum(abs_v_mms < 25.0) / n_tot * 100.0),
                'mid_pct': float(np.sum((abs_v_mms >= 25.0) & (abs_v_mms <= 100.0)) / n_tot * 100.0),
                'high_pct': float(np.sum(abs_v_mms > 100.0) / n_tot * 100.0)
            }

            gap = self.gap_f if i < 2 else self.gap_r
            bump_strikes[c] = int(np.sum(stroke_history[valid_slice, i] > gap))
            lift_offs[c] = int(np.sum(Fz_tires_history[valid_slice, i] < 1.0))

        # Synthetic Performance Scorecard (0 to 100)
        # Mechanical Grip Score: penalizes higher CPLV
        score_grip = float(np.clip(100.0 * (1.0 - 2.2 * cplv_dict['avg']), 0.0, 100.0))
        # Aero Platform Score: penalizes heave and pitch acceleration / stroke variance
        score_aero = float(np.clip(100.0 * np.exp(-(rms_heave / 1.8 + pkpk_pitch_deg / 1.2)), 0.0, 100.0))
        
        is_4post = getattr(cfg, 'rig_architecture', '7_post') == '4_post'
        if is_4post:
            score_overall = float(score_grip)
        else:
            score_overall = float(0.55 * score_grip + 0.45 * score_aero)

        # Correlazione e benchmark con il modello teorico veicolo
        model_comparison = self.compute_model_benchmark_comparison(peaks_found)

        if progress_callback:
            progress_callback(100, f"Simulazione e post-processing {'4-post' if is_4post else '7-post'} completati!")

        return SevenPostResult(
            time=t_arr,
            z_posts=z_p_arr,
            dz_posts=dz_p_arr,
            F_aero_posts=F_aero_arr,
            z_chassis=np.column_stack([z_s, theta, phi]),
            w_chassis=np.column_stack([w_h, w_theta, w_phi]),
            accel_chassis=accel_history,
            z_wheels=z_wheels,
            v_wheels=v_wheels,
            stroke_susp=stroke_history,
            v_damper=v_damper_history,
            F_spring=F_spring_history,
            F_damper=F_damper_history,
            F_susp=F_susp_history,
            Fz_tires=Fz_tires_history,
            freqs=freqs,
            transmissibility_heave=trans_heave,
            phase_heave=phase_heave,
            coherence_heave=coh_heave,
            transmissibility_pitch=trans_pitch,
            phase_pitch=phase_pitch,
            coherence_pitch=coh_pitch,
            transmissibility_roll=trans_roll,
            phase_roll=phase_roll,
            coherence_roll=coh_roll,
            psd_heave=P_heave,
            psd_pitch=P_pitch,
            psd_roll=P_roll,
            cplv_spectrum=cplv_spec,
            cplv=cplv_dict,
            cplv_low_band=cplv_low_band,
            cplv_high_band=cplv_high_band,
            delta_fz_pkpk=delta_fz_pkpk,
            estimated_grip_loss_pct=estimated_grip_loss,
            lift_off_duration_pct=lift_off_pct,
            rms_heave_accel=rms_heave,
            rms_pitch_accel=rms_pitch,
            rms_roll_accel=rms_roll,
            rms_ride_height_f_mm=rms_rh_f,
            rms_ride_height_r_mm=rms_rh_r,
            pkpk_ride_height_f_mm=pkpk_rh_f,
            pkpk_ride_height_r_mm=pkpk_rh_r,
            pkpk_pitch_deg=pkpk_pitch_deg,
            pkpk_roll_deg=pkpk_roll_deg,
            damper_power_peak_w=p_peak_w,
            damper_power_rms_w=p_rms_w,
            damper_energy_dissipated_j=e_diss_j,
            damper_asymmetry_ratio=asym_ratio,
            bilstrom_5bin=bilstrom_5bin,
            damper_velocity_histogram=damper_hist_3bin,
            bumpstop_strikes=bump_strikes,
            wheel_lift_off_events=lift_offs,
            score_grip=score_grip,
            score_aero=score_aero,
            score_overall=score_overall,
            resonance_peaks=peaks_found,
            rig_architecture='4_post' if is_4post else '7_post',
            model_comparison=model_comparison
        )

    def compute_model_benchmark_comparison(self, measured_peaks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calculates theoretical vehicle model benchmarks (both closed-form analytical
        quarter-car/chassis formulas and 14x14 linearized state-space modal analysis),
        and correlates them against peaks measured on the 4-Post / 7-Post rig.
        """
        self._cache_parameters()

        # 1. Closed-form analytical approximations
        # Ride rates
        k_ride_f = (self.K_z_f * self.K_tz_f) / max(self.K_z_f + self.K_tz_f, 1e-6)
        k_ride_r = (self.K_z_r * self.K_tz_r) / max(self.K_z_r + self.K_tz_r, 1e-6)

        m_sf_corner = (self.m_s * self.b_s) / max(2.0 * self.L, 1e-6)
        m_sr_corner = (self.m_s * self.a_s) / max(2.0 * self.L, 1e-6)

        fn_bounce_f = (1.0 / (2.0 * np.pi)) * np.sqrt(k_ride_f / max(m_sf_corner, 1e-6))
        fn_bounce_r = (1.0 / (2.0 * np.pi)) * np.sqrt(k_ride_r / max(m_sr_corner, 1e-6))

        # Overall heave
        k_heave = 2.0 * self.K_z_f + 2.0 * self.K_z_r
        fn_heave_approx = (1.0 / (2.0 * np.pi)) * np.sqrt(k_heave / max(self.m_s, 1e-6))

        # Pitch
        k_pitch = 2.0 * self.K_z_f * (self.a_s ** 2) + 2.0 * self.K_z_r * (self.b_s ** 2)
        fn_pitch_approx = (1.0 / (2.0 * np.pi)) * np.sqrt(k_pitch / max(self.Iy, 1e-6))

        # Roll
        k_roll = (0.5 * self.K_z_f * (self.tf ** 2) + self.Kw_arb_f * (self.tf ** 2) +
                  0.5 * self.K_z_r * (self.tr ** 2) + self.Kw_arb_r * (self.tr ** 2))
        fn_roll_approx = (1.0 / (2.0 * np.pi)) * np.sqrt(k_roll / max(self.Ix, 1e-6))

        # Wheel hop
        fn_hop_f = (1.0 / (2.0 * np.pi)) * np.sqrt((self.K_z_f + self.K_tz_f) / max(self.m_u_f, 1e-6))
        fn_hop_r = (1.0 / (2.0 * np.pi)) * np.sqrt((self.K_z_r + self.K_tz_r) / max(self.m_u_r, 1e-6))

        # 2. Linear 14x14 state-space modal analysis
        modal_res = self.compute_linear_modal_analysis()
        modes = modal_res.get('modes', [])

        # Map state-space modes by primary mode name
        mode_dict = {}
        for m in modes:
            name = m['name']
            if name not in mode_dict or m['frequency_hz'] < mode_dict[name]['frequency_hz']:
                mode_dict[name] = m

        # 3. Correlation table with measured peaks
        peaks = measured_peaks or []
        target_modes = [
            ('Chassis Heave', mode_dict.get('Chassis Heave', {}).get('frequency_hz', fn_heave_approx), mode_dict.get('Chassis Heave', {}).get('damping_ratio', 0.25)),
            ('Chassis Pitch', mode_dict.get('Chassis Pitch', {}).get('frequency_hz', fn_pitch_approx), mode_dict.get('Chassis Pitch', {}).get('damping_ratio', 0.25)),
            ('Chassis Roll', mode_dict.get('Chassis Roll', {}).get('frequency_hz', fn_roll_approx), mode_dict.get('Chassis Roll', {}).get('damping_ratio', 0.25)),
            ('Front Wheel Hop', mode_dict.get('Front Wheel Hop', {}).get('frequency_hz', fn_hop_f), mode_dict.get('Front Wheel Hop', {}).get('damping_ratio', 0.35)),
            ('Rear Wheel Hop', mode_dict.get('Rear Wheel Hop', {}).get('frequency_hz', fn_hop_r), mode_dict.get('Rear Wheel Hop', {}).get('damping_ratio', 0.35)),
        ]

        correlation_table = []
        for name, f_model, zeta in target_modes:
            matched_peak = None
            min_dist = float('inf')
            for p in peaks:
                f_p = p.get('frequency_hz', 0.0)
                # Match window within 35% of model frequency
                if abs(f_p - f_model) / max(f_model, 1e-6) < 0.35:
                    if abs(f_p - f_model) < min_dist:
                        min_dist = abs(f_p - f_model)
                        matched_peak = f_p

            if matched_peak is not None:
                delta_pct = ((matched_peak - f_model) / max(f_model, 1e-6)) * 100.0
                if abs(delta_pct) <= 5.0:
                    status = 'Eccellente (< 5%)'
                elif abs(delta_pct) <= 10.0:
                    status = 'Buono (< 10%)'
                elif abs(delta_pct) <= 20.0:
                    status = 'Accettabile (< 20%)'
                else:
                    status = 'Scostamento (> 20%)'
            else:
                delta_pct = None
                status = 'Non eccitato nel test'

            correlation_table.append({
                'mode': name,
                'model_hz': float(f_model),
                'rig_hz': float(matched_peak) if matched_peak is not None else None,
                'delta_pct': float(delta_pct) if delta_pct is not None else None,
                'damping_ratio': float(zeta),
                'status': status
            })

        return {
            'analytical': {
                'heave_f_hz': float(fn_bounce_f),
                'heave_r_hz': float(fn_bounce_r),
                'heave_approx_hz': float(fn_heave_approx),
                'pitch_approx_hz': float(fn_pitch_approx),
                'roll_approx_hz': float(fn_roll_approx),
                'hop_front_hz': float(fn_hop_f),
                'hop_rear_hz': float(fn_hop_r)
            },
            'state_space_modes': modes,
            'correlation_table': correlation_table
        }

    def generate_engineering_report(self, res: SevenPostResult) -> str:
        """Generates an exhaustive technical markdown summary report from 7-post / 4-post rig results."""
        is_4post = getattr(res, 'rig_architecture', '7_post') == '4_post'
        rig_title = "4-POST SHAKER RIG CLASSICO (4 Attuatori Stradali)" if is_4post else "7-POST SHAKER RIG (4 Stradali + 3 Aero Telaio)"
        lines = [
            f"# REPORT TECNICO BANCO DINAMICO {rig_title}",
            "=" * 60,
            f"Architettura Banco: {'4-Post (Meccanica Pura, Senza Carico Aero)' if is_4post else '7-Post (Full Aero + Road)'}",
            f"Punteggio Sintetico Setup: {res.score_overall:.1f}/100",
            f" - Score Aderenza Meccanica: {res.score_grip:.1f}/100",
        ]
        if not is_4post:
            lines.append(f" - Score Controllo Piattaforma Aero: {res.score_aero:.1f}/100")
        lines.extend([
            "",
            "## 1. Indicatori di Grip & Contact Patch Load Variation (CPLV)",
            "-------------------------------------------------------------",
            f"CPLV Medio Complessivo: {res.cplv['avg']*100:.2f} %",
            f" - Anteriore Sinistro (FL): {res.cplv['fl']*100:.2f} % (Low-Band: {res.cplv_low_band['fl']*100:.2f}%, High-Band: {res.cplv_high_band['fl']*100:.2f}%)",
            f" - Anteriore Destro (FR):   {res.cplv['fr']*100:.2f} % (Low-Band: {res.cplv_low_band['fr']*100:.2f}%, High-Band: {res.cplv_high_band['fr']*100:.2f}%)",
            f" - Posteriore Sinistro (RL):{res.cplv['rl']*100:.2f} % (Low-Band: {res.cplv_low_band['rl']*100:.2f}%, High-Band: {res.cplv_high_band['rl']*100:.2f}%)",
            f" - Posteriore Destro (RR):  {res.cplv['rr']*100:.2f} % (Low-Band: {res.cplv_low_band['rr']*100:.2f}%, High-Band: {res.cplv_high_band['rr']*100:.2f}%)",
            f"Perdita di Aderenza Media Stimata (Degressività): -{res.estimated_grip_loss_pct['avg']:.2f} %",
            f"Tempo in Condizione di Lift-Off (Fz = 0 N): {res.lift_off_duration_pct['fl']:.1f}% FL, {res.lift_off_duration_pct['rl']:.1f}% RL",
            "",
            "## 2. Stabilità Piattaforma Aerodinamica & Dinamica Telaio",
            "---------------------------------------------------------",
            f"Accelerazione RMS Heave: {res.rms_heave_accel:.2f} m/s²",
            f"Accelerazione RMS Pitch: {res.rms_pitch_accel:.2f} rad/s²",
            f"Accelerazione RMS Roll:  {res.rms_roll_accel:.2f} rad/s²",
            f"Escursione Ride Height Ant: RMS = {res.rms_ride_height_f_mm:.1f} mm | Pk-Pk = {res.pkpk_ride_height_f_mm:.1f} mm",
            f"Escursione Ride Height Post: RMS = {res.rms_ride_height_r_mm:.1f} mm | Pk-Pk = {res.pkpk_ride_height_r_mm:.1f} mm",
            f"Variazione Beccheggio Pk-Pk: {res.pkpk_pitch_deg:.2f}°",
            "",
            "## 3. Diagnostica Ammortizzatori Öhlins TTX25 & Bilancio Energetico",
            "------------------------------------------------------------------",
            f"Energia Dissipata Totale: FL={res.damper_energy_dissipated_j['fl']:.1f} J, RL={res.damper_energy_dissipated_j['rl']:.1f} J",
            f"Potenza di Picco Smorzatore: FL={res.damper_power_peak_w['fl']:.1f} W, RL={res.damper_power_peak_w['rl']:.1f} W",
            f"Rapporto Asimmetria Compressione/Estensione (Comp/Reb): FL={res.damper_asymmetry_ratio['fl']:.2f}, RL={res.damper_asymmetry_ratio['rl']:.2f}",
            "",
            "Istogramma Bilstrom a 5 Regimi (FL / RL):",
            f" - Ultra-Low (< 10 mm/s):   {res.bilstrom_5bin['fl']['ultra_low']:.1f}% / {res.bilstrom_5bin['rl']['ultra_low']:.1f}%",
            f" - Low-Speed (10-25 mm/s):  {res.bilstrom_5bin['fl']['low']:.1f}% / {res.bilstrom_5bin['rl']['low']:.1f}%",
            f" - Mid-Speed (25-50 mm/s):  {res.bilstrom_5bin['fl']['mid']:.1f}% / {res.bilstrom_5bin['rl']['mid']:.1f}%",
            f" - Mid-High  (50-100 mm/s): {res.bilstrom_5bin['fl']['mid_high']:.1f}% / {res.bilstrom_5bin['rl']['mid_high']:.1f}%",
            f" - High-Speed (> 100 mm/s): {res.bilstrom_5bin['fl']['high']:.1f}% / {res.bilstrom_5bin['rl']['high']:.1f}%",
            f"Contatti con Bumpstop: FL={res.bumpstop_strikes['fl']} eventi, RL={res.bumpstop_strikes['rl']} eventi"
        ])

        if res.model_comparison and 'correlation_table' in res.model_comparison:
            lines.extend([
                "",
                "## 4. Correlazione & Benchmark con il Modello Fisico Veicolo",
                "-----------------------------------------------------------",
                f"{'Modo Dinamico':<20} | {'Modello [Hz]':<12} | {'Banco [Hz]':<10} | {'Delta %':<10} | {'Smorz. ζ':<10} | {'Correlazione'}",
                "-" * 80
            ])
            for row in res.model_comparison['correlation_table']:
                f_m = f"{row['model_hz']:.2f}"
                f_r = f"{row['rig_hz']:.2f}" if row['rig_hz'] is not None else "N/A"
                d_p = f"{row['delta_pct']:+.1f}%" if row['delta_pct'] is not None else "N/A"
                zeta = f"{row['damping_ratio']:.3f}"
                lines.append(f"{row['mode']:<20} | {f_m:<12} | {f_r:<10} | {d_p:<10} | {zeta:<10} | {row['status']}")

        return "\n".join(lines)

    def compute_linear_modal_analysis(self) -> Dict[str, Any]:
        """
        Extracts linearized state-space matrices A (14x14) and B (14x7) around 1g static equilibrium.
        """
        self._cache_parameters()
        eps = 1e-4

        x0 = np.zeros(14)
        u0_road = np.zeros(4)
        u0_aero = np.zeros(3)

        def eval_deriv(x, z_p, dz_p, F_ae):
            zs, wh, phi, wphi, theta, wtheta, zw_fl, vw_fl, zw_fr, vw_fr, zw_rl, vw_rl, zw_rr, vw_rr = x
            z_c_fl = zs - theta * self.a_s + phi * (self.tf / 2.0)
            z_c_fr = zs - theta * self.a_s - phi * (self.tf / 2.0)
            z_c_rl = zs + theta * self.b_s + phi * (self.tr / 2.0)
            z_c_rr = zs + theta * self.b_s - phi * (self.tr / 2.0)

            v_c_fl = wh - wtheta * self.a_s + wphi * (self.tf / 2.0)
            v_c_fr = wh - wtheta * self.a_s - wphi * (self.tf / 2.0)
            v_c_rl = wh + wtheta * self.b_s + wphi * (self.tr / 2.0)
            v_c_rr = wh + wtheta * self.b_s - wphi * (self.tr / 2.0)

            z_j_fl = zw_fl - z_c_fl; z_j_fr = zw_fr - z_c_fr
            z_j_rl = zw_rl - z_c_rl; z_j_rr = zw_rr - z_c_rr
            v_j_fl = vw_fl - v_c_fl; v_j_fr = vw_fr - v_c_fr
            v_j_rl = vw_rl - v_c_rl; v_j_rr = vw_rr - v_c_rr

            F_sp_fl = self._calc_spring_force(z_j_fl, self.K_z_f, self.gap_f, self.k_bs_f)
            F_sp_fr = self._calc_spring_force(z_j_fr, self.K_z_f, self.gap_f, self.k_bs_f)
            F_sp_rl = self._calc_spring_force(z_j_rl, self.K_z_r, self.gap_r, self.k_bs_r)
            F_sp_rr = self._calc_spring_force(z_j_rr, self.K_z_r, self.gap_r, self.k_bs_r)

            d_z_f = z_j_fl - z_j_fr; d_z_r = z_j_rl - z_j_rr
            F_arb_fl = self.Kw_arb_f * d_z_f; F_arb_fr = -self.Kw_arb_f * d_z_f
            F_arb_rl = self.Kw_arb_r * d_z_r; F_arb_rr = -self.Kw_arb_r * d_z_r

            F_d_fl = self._calc_damper_force(v_j_fl, True)
            F_d_fr = self._calc_damper_force(v_j_fr, True)
            F_d_rl = self._calc_damper_force(v_j_rl, False)
            F_d_rr = self._calc_damper_force(v_j_rr, False)

            F_su_fl = self.F_susp_0_fl + F_sp_fl + F_d_fl + F_arb_fl
            F_su_fr = self.F_susp_0_fr + F_sp_fr + F_d_fr + F_arb_fr
            F_su_rl = self.F_susp_0_rl + F_sp_rl + F_d_rl + F_arb_rl
            F_su_rr = self.F_susp_0_rr + F_sp_rr + F_d_rr + F_arb_rr

            defl_fl = self.defl_0_fl - (zw_fl - z_p[0])
            defl_fr = self.defl_0_fr - (zw_fr - z_p[1])
            defl_rl = self.defl_0_rl - (zw_rl - z_p[2])
            defl_rr = self.defl_0_rr - (zw_rr - z_p[3])

            Fz_fl = max(0.0, self.K_tz_f * defl_fl + self.C_tz * (dz_p[0] - vw_fl))
            Fz_fr = max(0.0, self.K_tz_f * defl_fr + self.C_tz * (dz_p[1] - vw_fr))
            Fz_rl = max(0.0, self.K_tz_r * defl_rl + self.C_tz * (dz_p[2] - vw_rl))
            Fz_rr = max(0.0, self.K_tz_r * defl_rr + self.C_tz * (dz_p[3] - vw_rr))

            F_aero_tot = F_ae[0] + F_ae[1] + F_ae[2]
            M_pitch_aero = self.a_s * F_ae[0] - self.b_s * (F_ae[1] + F_ae[2])
            M_roll_aero = (self.tr / 2.0) * (F_ae[1] - F_ae[2])

            zs_ddot = (F_su_fl + F_su_fr + F_su_rl + F_su_rr - F_aero_tot - self.m_s * self.g) / self.m_s
            phi_ddot = ((self.tf / 2.0) * (F_su_fl - F_su_fr) +
                        (self.tr / 2.0) * (F_su_rl - F_su_rr) + M_roll_aero) / self.Ix
            theta_ddot = (-self.a_s * (F_su_fl + F_su_fr) +
                          self.b_s * (F_su_rl + F_su_rr) + M_pitch_aero) / self.Iy

            vw_fl_dot = (Fz_fl - F_su_fl - self.m_u_f * self.g) / self.m_u_f
            vw_fr_dot = (Fz_fr - F_su_fr - self.m_u_f * self.g) / self.m_u_f
            vw_rl_dot = (Fz_rl - F_su_rl - self.m_u_r * self.g) / self.m_u_r
            vw_rr_dot = (Fz_rr - F_su_rr - self.m_u_r * self.g) / self.m_u_r

            return np.array([
                wh, zs_ddot,
                wphi, phi_ddot,
                wtheta, theta_ddot,
                vw_fl, vw_fl_dot,
                vw_fr, vw_fr_dot,
                vw_rl, vw_rl_dot,
                vw_rr, vw_rr_dot
            ])

        f0 = eval_deriv(x0, u0_road, np.zeros(4), u0_aero)
        A = np.zeros((14, 14))

        for j in range(14):
            x_plus = x0.copy()
            x_plus[j] += eps
            f_plus = eval_deriv(x_plus, u0_road, np.zeros(4), u0_aero)
            A[:, j] = (f_plus - f0) / eps

        eigvals, eigvecs = np.linalg.eig(A)
        modes = []
        for i, val in enumerate(eigvals):
            if val.imag >= 0:
                omega_n = float(np.abs(val))
                fn = float(omega_n / (2.0 * np.pi))
                zeta = float(-val.real / max(omega_n, 1e-6))
                
                v = np.abs(eigvecs[:, i])
                dom_idx = int(np.argmax([v[0], v[2], v[4], v[6] + v[8], v[10] + v[12]]))
                names = ['Chassis Heave', 'Chassis Roll', 'Chassis Pitch', 'Front Wheel Hop', 'Rear Wheel Hop']
                mode_name = names[dom_idx] if dom_idx < len(names) else 'Coupled'

                modes.append({
                    'name': mode_name,
                    'frequency_hz': fn,
                    'damping_ratio': zeta,
                    'eigenvalue': complex(val)
                })

        modes = sorted(modes, key=lambda m: m['frequency_hz'])

        return {
            'A_matrix': A,
            'eigenvalues': eigvals,
            'modes': modes
        }
