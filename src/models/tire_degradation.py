"""
tire_degradation.py
-------------------
Tire thermal and wear degradation models for the Dynamic Target Suite.

Provides three modes:
  - "off":      No degradation ÔÇö constant grip (default)
  - "light":    1-zone bulk temperature model + energy-based wear
  - "detailed": 3-zone (tread surface / tread bulk / carcass) thermal model + wear

Heat generation is based on instantaneous sliding power P_slip = |F ┬À v_slip|.
Grip modifier ┬Á_eff = ┬Á_base ┬À f_temp(T) ┬À f_wear(W) is applied as a multiplier
to both LMUX and LMUY in the MF6.1 tire model.

References:
  - Pacejka, H.B. "Tire and Vehicle Dynamics" Ch. 10 (Thermal effects)
  - Sorniotti, A. "Tire Thermal Model for Enhanced Vehicle Dynamics Simulation"
  - Generic racing slick compound characterization data
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TireState:
    """Per-wheel thermal and wear state."""
    # 1-zone (light mode)
    T_bulk: float = 25.0            # Bulk rubber temperature [°C]

    # 3-zone (detailed mode)
    T_surface: float = 25.0         # Tread surface temperature [°C]
    T_tread: float = 25.0           # Tread bulk temperature [°C]
    T_carcass: float = 25.0         # Carcass temperature [°C]

    # Wear state
    wear_fraction: float = 0.0      # Cumulative wear [0 = new, 1 = fully worn]
    energy_dissipated: float = 0.0  # Total sliding energy [J] (diagnostic)


@dataclass
class TireDegradationParams:
    """
    Thermal and wear model parameters.
    Default values configured for generic prototype racing slick compound.
    """
    # --- Environment ---
    T_ambient: float = 25.0         # Ambient air temperature [°C]
    T_start: float = 25.0           # Starting tire temperature [°C] (25°C ambient, 50°C warm, 65.0°C optimal)

    # --- Thermal (1-zone / light mode) ---
    T_optimal: float = 65.0         # Optimal grip temperature [°C] (generic racing compound)
    T_overheat: float = 95.0        # Overheating threshold [°C]
    thermal_mass_bulk: float = 2200.0   # Effective tread surface/bulk thermal mass [J/°C]
    h_conv_bulk: float = 9.0        # Convective cooling coefficient × area [W/°C]
    heat_gen_efficiency: float = 0.40   # Fraction of slip power converted to heat [-]

    # --- Thermal (3-zone / detailed mode) ---
    thermal_mass_surface: float = 400.0     # Tread surface thermal mass [J/°C]
    thermal_mass_tread: float = 2400.0      # Tread bulk thermal mass [J/°C]
    thermal_mass_carcass: float = 4000.0    # Carcass thermal mass [J/°C]
    h_surface_air: float = 11.5             # Surface-to-air convection [W/°C] (forced convection)
    h_surface_tread: float = 65.0           # Surface-to-tread conduction [W/°C]
    h_tread_carcass: float = 35.0           # Tread-to-carcass conduction [W/°C]
    h_carcass_air: float = 6.0              # Carcass-to-air convection (rim + sidewall) [W/°C]
    heat_gen_surface_frac: float = 0.65     # Fraction of slip heat to surface [-]
    heat_gen_tread_frac: float = 0.28       # Fraction of slip heat to tread bulk [-]
    heat_gen_carcass_frac: float = 0.07     # Fraction of slip heat to carcass [-]

    # --- Grip vs Temperature Calibration (Generic Parabolic Thermal Model) ---
    # mu_y(T) = mu_peak - curvature_a * (T - T_optimal)^2
    # Normalized: f_temp(T) = 1.0 - c_temp * (T - T_optimal)^2
    mu_peak: float = 1.50                   # Peak lateral grip coefficient µ_y at T_optimal [-]
    curvature_a: float = 0.00060            # Parabolic curvature [1/°C²]
    c_temp: float = 0.00040                 # Normalized curvature c = curvature_a / mu_peak [1/°C²]
    min_grip_floor: float = 0.80            # Minimum normalized grip multiplier outside operating range [0..1]
    sigma_temp: float = 25.0                # Temperature spread parameter [°C]

    # --- Wear ---
    k_wear: float = 2.0e-03         # Base wear rate coefficient [-]
    wear_rate_scale: float = 1.0    # User-selectable wear acceleration multiplier (1x = realistic, 2x, 5x, 10x)
    mu_worn_min: float = 0.75       # Minimum µ fraction at full wear [-]
    wear_exponent: float = 1.2      # Wear acceleration exponent (> 1 = progressive) [-]


class TireDegradationModel:
    """
    Tire thermal degradation and wear model.

    Supports three operating modes:
      - "off":      grip_modifier() always returns 1.0, step() is a no-op
      - "light":    1-zone bulk temperature + energy-based wear
      - "detailed": 3-zone thermal network + energy-based wear

    Usage:
        model = TireDegradationModel(mode="light")
        states = [TireState() for _ in range(4)]   # FL, FR, RL, RR

        # In physics loop:
        for i in range(4):
            model.step(states[i], F_total[i], v_slip[i], Fz[i], vx, dt)
            grip_mult = model.grip_modifier(states[i])
            # Apply grip_mult to LMUX / LMUY
    """

    def __init__(self, mode: str = "off", params: Optional[TireDegradationParams] = None):
        self.mode = mode
        self.params = params if params is not None else TireDegradationParams()

    def set_mode(self, mode: str):
        """Change degradation mode at runtime."""
        if mode not in ("off", "light", "detailed"):
            raise ValueError(f"Invalid tire degradation mode: {mode!r}. Use 'off', 'light', or 'detailed'.")
        self.mode = mode

    def set_wear_rate_scale(self, scale: float):
        """Set user wear multiplier (1.0 = realistic endurance, 2.0, 5.0, 10.0 = quick test)."""
        self.params.wear_rate_scale = max(float(scale), 0.0)

    def update_from_vehicle_params(self, vp):
        """
        Dynamically configures thermal inertia, heat dissipation, and operating temperatures
        based on vehicle and tire sizing (FSAE racing slick vs Road SUV tire).
        """
        if vp is None:
            return
        is_suv = (getattr(vp, 'vehicle_type', 'formula') == 'b_segment_suv')
        c_th = float(getattr(vp, 'tire_thermal_mass', 16000.0 if is_suv else 2000.0))
        h_c = float(getattr(vp, 'tire_h_conv', 32.0 if is_suv else 8.5))

        self.params.thermal_mass_bulk = c_th
        self.params.h_conv_bulk = h_c

        ratio_mass = c_th / 2000.0
        ratio_h = h_c / 8.5
        self.params.thermal_mass_surface = 400.0 * ratio_mass
        self.params.thermal_mass_tread = 2400.0 * ratio_mass
        self.params.thermal_mass_carcass = 4000.0 * ratio_mass
        self.params.h_surface_air = 11.5 * ratio_h
        self.params.h_carcass_air = 6.0 * ratio_h

        if is_suv:
            # Road Touring tire: high thermal inertia, gentle thermal sensitivity, lower slip hysteresis
            self.params.T_optimal = 52.0
            self.params.T_overheat = 105.0
            self.params.c_temp = 0.00020
            self.params.min_grip_floor = 0.85
            self.params.heat_gen_efficiency = 0.22
        else:
            # Generic racing slick: rapid heat buildup, tight temperature window
            self.params.T_optimal = 65.0
            self.params.T_overheat = 95.0
            self.params.c_temp = 0.00040
            self.params.min_grip_floor = 0.80
            self.params.heat_gen_efficiency = 0.40

    def reset(self, state: TireState, T_start: Optional[float] = None):
        """Reset a tire state to new condition with specified starting temperature."""
        t_init = float(T_start) if T_start is not None else float(getattr(self.params, 'T_start', self.params.T_ambient))
        state.T_bulk = t_init
        state.T_surface = t_init
        state.T_tread = t_init
        state.T_carcass = t_init
        state.wear_fraction = 0.0
        state.energy_dissipated = 0.0

    # -------------------------------------------------------------------------
    # Grip modifier: f_temp(T) * f_wear(W)
    # -------------------------------------------------------------------------

    def f_temp(self, T: float) -> float:
        """
        Normalized grip vs temperature curve.
        Reference: Parabolic Thermal Model:
            mu_y(T) = mu_peak - a * (T - T_optimal)^2
        Normalized multiplier:
            f(T) = 1.0 - c_temp * (T - T_optimal)^2

        Bounded by min_grip_floor (default 0.80) to prevent unphysical quadratic divergence.
        """
        p = self.params
        delta_T = T - p.T_optimal
        f = 1.0 - p.c_temp * (delta_T ** 2)
        return float(np.clip(f, p.min_grip_floor, 1.0))

    def mu_y_temp(self, T: float) -> float:
        """
        Absolute peak lateral grip mu_y at temperature T [°C] from parabolic model.
        """
        p = self.params
        mu = p.mu_peak - p.curvature_a * ((T - p.T_optimal) ** 2)
        return float(np.clip(mu, p.min_grip_floor * p.mu_peak, p.mu_peak))

    def f_wear(self, W: float) -> float:
        """
        Wear-dependent grip degradation.
        Linear decay from 1.0 (new) to mu_worn_min (fully worn).
        """
        p = self.params
        W_clamped = min(max(W, 0.0), 1.0)
        return 1.0 - (1.0 - p.mu_worn_min) * W_clamped

    def grip_modifier(self, state: TireState) -> float:
        """
        Returns the combined ┬Á multiplier [0..1] based on current tire state.
        In "off" mode, always returns 1.0.
        """
        if self.mode == "off":
            return 1.0

        # Use representative temperature based on mode
        if self.mode == "light":
            T_rep = state.T_bulk
        else:  # detailed
            # Weighted average: tread bulk provides thermal inertia, surface provides transient response
            T_rep = 0.35 * state.T_surface + 0.50 * state.T_tread + 0.15 * state.T_carcass

        return self.f_temp(T_rep) * self.f_wear(state.wear_fraction)

    def get_display_temperature(self, state: TireState) -> float:
        """Returns the representative temperature for HUD display."""
        if self.mode == "light":
            return state.T_bulk
        elif self.mode == "detailed":
            return 0.35 * state.T_surface + 0.50 * state.T_tread + 0.15 * state.T_carcass
        return self.params.T_ambient

    # -------------------------------------------------------------------------
    # Time integration
    # -------------------------------------------------------------------------

    def step(self, state: TireState, F_total: float, v_slip: float,
             Fz: float, vx: float, dt: float):
        """
        Advance tire thermal and wear state by one timestep.

        Parameters:
        -----------
        state    : TireState ÔÇö mutable per-wheel state
        F_total  : float ÔÇö total resultant tire force magnitude [N] (sqrt(Fx┬▓ + Fy┬▓))
        v_slip   : float ÔÇö total sliding velocity [m/s] (sqrt((vx┬À╬║)┬▓ + (vx┬Àtan(╬▒))┬▓))
        Fz       : float ÔÇö vertical load [N]
        vx       : float ÔÇö longitudinal vehicle speed [m/s]
        dt       : float ÔÇö timestep [s]
        """
        if self.mode == "off":
            return

        p = self.params

        # Sliding power [W]
        P_slip = abs(F_total * v_slip)
        Q_gen = P_slip * p.heat_gen_efficiency

        # Track cumulative energy
        state.energy_dissipated += P_slip * dt

        if self.mode == "light":
            self._step_light(state, Q_gen, dt)
        elif self.mode == "detailed":
            self._step_detailed(state, Q_gen, dt)

        # Wear integration (common to both modes)
        self._step_wear(state, P_slip, Fz, vx, dt)

    def _step_light(self, state: TireState, Q_gen: float, dt: float):
        """1-zone bulk temperature model (Euler integration)."""
        p = self.params

        # Heat dissipation: convection to ambient
        Q_cool = p.h_conv_bulk * (state.T_bulk - p.T_ambient)

        # Temperature rate of change
        dT = (Q_gen - Q_cool) / p.thermal_mass_bulk

        state.T_bulk += dT * dt
        # Clamp to physical range
        state.T_bulk = max(state.T_bulk, p.T_ambient - 5.0)

    def _step_detailed(self, state: TireState, Q_gen: float, dt: float):
        """3-zone thermal network model (Euler integration)."""
        p = self.params

        # Heat generation distribution
        Q_surface = Q_gen * p.heat_gen_surface_frac
        Q_tread = Q_gen * p.heat_gen_tread_frac
        Q_carcass = Q_gen * p.heat_gen_carcass_frac

        # Heat flows between zones (conduction) and to environment (convection)
        Q_surf_air = p.h_surface_air * (state.T_surface - p.T_ambient)
        Q_surf_tread = p.h_surface_tread * (state.T_surface - state.T_tread)
        Q_tread_carc = p.h_tread_carcass * (state.T_tread - state.T_carcass)
        Q_carc_air = p.h_carcass_air * (state.T_carcass - p.T_ambient)

        # Temperature rates
        dT_surface = (Q_surface - Q_surf_air - Q_surf_tread) / p.thermal_mass_surface
        dT_tread = (Q_tread + Q_surf_tread - Q_tread_carc) / p.thermal_mass_tread
        dT_carcass = (Q_carcass + Q_tread_carc - Q_carc_air) / p.thermal_mass_carcass

        # Euler integration
        state.T_surface += dT_surface * dt
        state.T_tread += dT_tread * dt
        state.T_carcass += dT_carcass * dt

        # Clamp
        T_min = p.T_ambient - 5.0
        state.T_surface = max(state.T_surface, T_min)
        state.T_tread = max(state.T_tread, T_min)
        state.T_carcass = max(state.T_carcass, T_min)

        # Also update bulk for backwards compatibility
        state.T_bulk = 0.35 * state.T_surface + 0.50 * state.T_tread + 0.15 * state.T_carcass

    def _step_wear(self, state: TireState, P_slip: float, Fz: float,
                   vx: float, dt: float):
        """Energy-based wear integration."""
        p = self.params

        # Normalized sliding power: P_slip / (Fz ┬À max(vx, 1))
        # This makes wear independent of absolute speed and proportional to slip severity
        Fz_safe = max(Fz, 100.0)
        vx_safe = max(abs(vx), 1.0)

        # Wear rate: dimensionless per second
        scale = float(getattr(p, 'wear_rate_scale', 1.0))
        wear_rate = p.k_wear * scale * P_slip / (Fz_safe * vx_safe)

        # Progressive wear acceleration (worn tires wear faster)
        if p.wear_exponent != 1.0 and state.wear_fraction > 0.01:
            wear_rate *= (1.0 + state.wear_fraction) ** (p.wear_exponent - 1.0)

        state.wear_fraction += wear_rate * dt
        state.wear_fraction = min(state.wear_fraction, 1.0)
