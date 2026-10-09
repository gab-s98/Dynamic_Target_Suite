"""
road_profile.py
---------------
3D Road Surface Profile & Elevation Model for the Dynamic Target Suite.

Provides two distinct and independent road profile layers:
  1. Macroscopic 3D Profile:
     - Centerline elevation z(s) [m]
     - Longitudinal slope grade = tan(theta_road) [-]
     - Lateral banking angle phi_bank(s) [rad]
  2. Microscopic Road Roughness & Unevenness:
     - High-frequency waviness & pavement roughness delta_z(s, y) [m]
     - Generates independent road excitation under each tire contact patch (FL, FR, RL, RR)
     - Excites suspension chatter, dynamic wheel loads, and haptics

Backward compatible: flat(s_array) provides flat 2D track (z=0, grade=0, bank=0).
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class RoadProfileConfig:
    """Configuration parameters for 3D road profile generation."""
    # Macro Altimetry & Banking
    enable_macro: bool = False
    macro_elevation_amp: float = 1.50     # Elevation variation amplitude [m]
    macro_wavelength: float = 120.0       # Longitudinal elevation wavelength [m]
    macro_bank_max_deg: float = 3.5       # Maximum corner banking [deg]

    # Micro Roughness & Unevenness
    enable_micro: bool = False
    micro_roughness_amp_mm: float = 8.0   # Unevenness peak amplitude [mm]
    micro_wavelength_m: float = 4.0       # Dominant pavement wave length [m]
    micro_lateral_decorr: float = 0.65    # Left/Right track decorrelation phase [rad]

    # Custom User-Defined Profile Stations
    is_custom_user: bool = False
    custom_stations: Optional[list] = None  # list of {'s': float, 'z': float, 'bank_deg': float}


class RoadProfile:
    """
    Continuous 3D road profile along curvilinear abscissa s.
    """

    def __init__(self, s_array: np.ndarray,
                 z_macro: Optional[np.ndarray] = None,
                 grade: Optional[np.ndarray] = None,
                 bank_rad: Optional[np.ndarray] = None,
                 config: Optional[RoadProfileConfig] = None):
        self.s = np.asarray(s_array, dtype=float)
        self.N = len(self.s)
        self.track_length = float(self.s[-1] - self.s[0]) if self.N > 1 else 100.0
        self.config = config if config is not None else RoadProfileConfig()

        if z_macro is not None:
            self.z_macro = np.asarray(z_macro, dtype=float)
        else:
            self.z_macro = np.zeros(self.N, dtype=float)

        if grade is not None:
            self.grade = np.asarray(grade, dtype=float)
        else:
            self.grade = np.zeros(self.N, dtype=float)

        if bank_rad is not None:
            self.bank_rad = np.asarray(bank_rad, dtype=float)
        else:
            self.bank_rad = np.zeros(self.N, dtype=float)

    @classmethod
    def flat(cls, s_array: np.ndarray, config: Optional[RoadProfileConfig] = None) -> "RoadProfile":
        """Factory for a purely flat 2D track (z=0, grade=0, bank=0)."""
        cfg = config if config is not None else RoadProfileConfig(enable_macro=False, enable_micro=False)
        return cls(s_array, config=cfg)

    @classmethod
    def procedural(cls, s_array: np.ndarray,
                   curvature: Optional[np.ndarray] = None,
                   config: Optional[RoadProfileConfig] = None) -> "RoadProfile":
        """
        Creates a procedural 3D road profile with smooth macro hills and natural corner banking.
        """
        cfg = config if config is not None else RoadProfileConfig(enable_macro=True, enable_micro=False)
        s = np.asarray(s_array, dtype=float)
        N = len(s)
        L = float(s[-1] - s[0]) if N > 1 else 100.0

        if not cfg.enable_macro:
            return cls(s, config=cfg)

        # Macro elevation: harmonic undulating hills that close cleanly for circuits
        n_waves = max(1, int(round(L / max(cfg.macro_wavelength, 20.0))))
        omega_macro = (2.0 * np.pi * n_waves) / max(L, 1.0)

        # Double-harmonic elevation profile for natural terrain variation
        z_macro = (cfg.macro_elevation_amp * np.sin(omega_macro * s) +
                   0.35 * cfg.macro_elevation_amp * np.sin(2.0 * omega_macro * s + 0.4))

        # Grade is the spatial derivative dz/ds = tan(theta)
        grade = (cfg.macro_elevation_amp * omega_macro * np.cos(omega_macro * s) +
                 0.35 * cfg.macro_elevation_amp * 2.0 * omega_macro * np.cos(2.0 * omega_macro * s + 0.4))

        # Lateral banking: corners get positive banking towards inside
        bank_max_rad = math.radians(cfg.macro_bank_max_deg)
        if curvature is not None and len(curvature) == N:
            # Bank inwards proportional to track curvature kappa
            kappa = np.asarray(curvature, dtype=float)
            k_scale = 0.08  # saturation curvature (12.5m radius)
            bank_rad = bank_max_rad * np.clip(kappa / k_scale, -1.0, 1.0)
        else:
            # Gentle sinusoidal banking correlated with undulations
            bank_rad = bank_max_rad * 0.70 * np.sin(omega_macro * s + 0.8)

        return cls(s, z_macro=z_macro, grade=grade, bank_rad=bank_rad, config=cfg)

    @classmethod
    def from_user_stations(cls, s_array: np.ndarray,
                           stations: list,
                           is_closed: bool = True,
                           config: Optional[RoadProfileConfig] = None) -> "RoadProfile":
        """
        Creates a 3D road profile from user-defined stations.
        Each station is either:
          - dict with keys 's', 'z', 'bank_deg'
          - tuple/list of (s, z, bank_deg)
        Uses smooth cubic spline interpolation with periodic boundary condition for closed circuits.
        """
        cfg = config if config is not None else RoadProfileConfig(enable_macro=True, is_custom_user=True)
        cfg.enable_macro = True
        cfg.is_custom_user = True
        cfg.custom_stations = list(stations)

        s = np.asarray(s_array, dtype=float)
        N = len(s)
        if N < 2:
            return cls(s, config=cfg)

        L = float(s[-1] - s[0]) if N > 1 else 100.0

        # Parse stations
        st_list = []
        for item in stations:
            if isinstance(item, dict):
                st_list.append((float(item['s']), float(item['z']), float(item.get('bank_deg', 0.0))))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                b_deg = float(item[2]) if len(item) >= 3 else 0.0
                st_list.append((float(item[0]), float(item[1]), b_deg))

        # Sort by arc length
        st_list.sort(key=lambda x: x[0])
        if len(st_list) < 2:
            return cls(s, config=cfg)

        s_pts = [p[0] for p in st_list]
        z_pts = [p[1] for p in st_list]
        bank_pts = [p[2] for p in st_list]

        from scipy.interpolate import CubicSpline

        if is_closed:
            s_start = s[0]
            s_end = s[-1]
            if s_pts[0] > s_start + 1e-3:
                s_pts.insert(0, s_start)
                z_pts.insert(0, z_pts[-1])
                bank_pts.insert(0, bank_pts[-1])
            if s_pts[-1] < s_end - 1e-3:
                s_pts.append(s_end)
                z_pts.append(z_pts[0])
                bank_pts.append(bank_pts[0])
            else:
                z_pts[-1] = z_pts[0]
                bank_pts[-1] = bank_pts[0]

            try:
                cs_z = CubicSpline(s_pts, z_pts, bc_type='periodic')
                cs_bank = CubicSpline(s_pts, bank_pts, bc_type='periodic')
            except Exception:
                cs_z = CubicSpline(s_pts, z_pts, bc_type='natural')
                cs_bank = CubicSpline(s_pts, bank_pts, bc_type='natural')
        else:
            cs_z = CubicSpline(s_pts, z_pts, bc_type='natural')
            cs_bank = CubicSpline(s_pts, bank_pts, bc_type='natural')

        z_macro = np.asarray(cs_z(s), dtype=float)
        grade = np.asarray(cs_z.derivative(nu=1)(s), dtype=float)
        bank_rad = np.radians(np.asarray(cs_bank(s), dtype=float))

        return cls(s, z_macro=z_macro, grade=grade, bank_rad=bank_rad, config=cfg)

    @staticmethod
    def get_default_stations(track_length: float, n_stations: int = 8) -> list:
        """Generates evenly spaced stations along the circuit with flat default values."""
        ds = track_length / max(n_stations, 2)
        stations = []
        for i in range(n_stations):
            s_val = i * ds
            stations.append({'s': round(s_val, 1), 'z': 0.0, 'bank_deg': 0.0})
        return stations

    @staticmethod
    def get_preset_stations(preset_name: str, track_length: float,
                            s_array: Optional[np.ndarray] = None,
                            curvature: Optional[np.ndarray] = None) -> list:
        """Generates station points for recognizable 3D terrain archetypes."""
        L = max(track_length, 50.0)
        stations = []

        if preset_name == "Piatto (2D)":
            for frac in np.linspace(0.0, 1.0, 9)[:-1]:
                stations.append({'s': round(frac * L, 1), 'z': 0.0, 'bank_deg': 0.0})

        elif preset_name == "Collinare (Onde & Banking)":
            for frac in np.linspace(0.0, 1.0, 9)[:-1]:
                s_val = frac * L
                z_val = 1.8 * np.sin(2.0 * np.pi * frac * 2.0)
                b_deg = 3.5 * np.cos(2.0 * np.pi * frac * 2.0)
                stations.append({'s': round(s_val, 1), 'z': round(float(z_val), 2), 'bank_deg': round(float(b_deg), 1)})

        elif preset_name == "Circuito di Montagna (Salita/Discesa +-15m)":
            fractions = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
            z_profiles = [0.0, 4.5, 10.5, 15.0, 12.0, 6.0, 1.5, -1.0]
            b_profiles = [0.0, 2.5, 4.0, 6.5, 5.0, 3.0, -2.0, 0.0]
            for frac, zv, bv in zip(fractions, z_profiles, b_profiles):
                stations.append({'s': round(frac * L, 1), 'z': float(zv), 'bank_deg': float(bv)})

        elif preset_name == "Curve Sopraelevate (Banked Corners 10°)":
            for frac in np.linspace(0.0, 1.0, 9)[:-1]:
                s_val = frac * L
                if curvature is not None and s_array is not None and len(s_array) == len(curvature):
                    k_val = float(np.interp(s_val, s_array, curvature))
                    b_deg = float(np.clip(k_val / 0.07 * 9.0, -10.0, 10.0))
                else:
                    b_deg = 8.5 * np.sin(2.0 * np.pi * frac * 3.0)
                z_val = 0.5 * np.sin(2.0 * np.pi * frac)
                stations.append({'s': round(s_val, 1), 'z': round(float(z_val), 2), 'bank_deg': round(float(b_deg), 1)})

        elif preset_name == "Rollercoaster (Dossi & Avvallamenti)":
            fractions = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
            z_profiles = [0.0, 5.0, -3.5, 6.5, -2.0, 4.0, -1.5]
            b_profiles = [0.0, 4.0, -5.0, 6.0, -4.0, 3.0, -2.0]
            for frac, zv, bv in zip(fractions, z_profiles, b_profiles):
                stations.append({'s': round(frac * L, 1), 'z': float(zv), 'bank_deg': float(bv)})

        else:
            return RoadProfile.get_default_stations(L, 8)

        return stations

    def interpolate_at(self, s_query: float) -> Tuple[float, float, float]:
        """
        Interpolates macroscopic (z, grade, bank_rad) at arbitrary arc length s.
        Wraps s modulo track length for closed circuits.
        """
        if not self.config.enable_macro or self.N < 2:
            return 0.0, 0.0, 0.0

        # Wrap around track length
        s_mod = (s_query - self.s[0]) % self.track_length + self.s[0]
        z_val = float(np.interp(s_mod, self.s, self.z_macro))
        grade_val = float(np.interp(s_mod, self.s, self.grade))
        bank_val = float(np.interp(s_mod, self.s, self.bank_rad))
        return z_val, grade_val, bank_val

    def get_micro_unevenness(self, s_query: float, lateral_offset_y: float) -> float:
        """
        Computes high-frequency road roughness / micro-waviness delta_z [m]
        at position (s, y).

        Parameters:
        -----------
        s_query          : float — arc length along centerline [m]
        lateral_offset_y : float — lateral position from centerline (+ left, - right) [m]
        """
        if not self.config.enable_micro:
            return 0.0

        amp_m = self.config.micro_roughness_amp_mm * 1e-3
        wl = max(self.config.micro_wavelength_m, 0.5)
        w_lat = self.config.micro_lateral_decorr

        # Multi-harmonic spatial wave field
        k1 = 2.0 * np.pi / wl
        k2 = 2.0 * np.pi / (wl * 0.47)
        k3 = 2.0 * np.pi / (wl * 1.83)

        # Left/Right track decorrelation through lateral offset
        phase_lat = lateral_offset_y * (w_lat / 1.5)

        z_micro = amp_m * (
            0.60 * np.sin(k1 * s_query + phase_lat) +
            0.25 * np.sin(k2 * s_query - phase_lat * 1.3) +
            0.15 * np.sin(k3 * s_query + 0.5)
        )
        return float(z_micro)

    def get_wheel_contact_heights(self, s_center: float,
                                  a_dist: float, b_dist: float,
                                  t_f: float, t_r: float) -> Dict[str, float]:
        """
        Calculates the vertical ground contact height [m] under each of the 4 tires:
        FL (Front-Left), FR (Front-Right), RL (Rear-Left), RR (Rear-Right).

        Combines:
          - Macro elevation z_center
          - Pitch tilt (grade * longitudinal distance)
          - Roll tilt (sin(bank) * half-track)
          - Independent micro-roughness unevenness per corner
        """
        if not self.config.enable_macro and not self.config.enable_micro:
            return {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0,
                    'z_center': 0.0, 'grade': 0.0, 'bank': 0.0}

        z_mid, grade_mid, bank_mid = self.interpolate_at(s_center)

        sin_b = math.sin(bank_mid)
        tf_half = t_f / 2.0
        tr_half = t_r / 2.0

        z_fl_macro = z_mid + a_dist * grade_mid + tf_half * sin_b
        z_fr_macro = z_mid + a_dist * grade_mid - tf_half * sin_b
        z_rl_macro = z_mid - b_dist * grade_mid + tr_half * sin_b
        z_rr_macro = z_mid - b_dist * grade_mid - tr_half * sin_b

        # Micro-roughness per corner
        z_fl_micro = self.get_micro_unevenness(s_center + a_dist, tf_half)
        z_fr_micro = self.get_micro_unevenness(s_center + a_dist, -tf_half)
        z_rl_micro = self.get_micro_unevenness(s_center - b_dist, tr_half)
        z_rr_micro = self.get_micro_unevenness(s_center - b_dist, -tr_half)

        return {
            'fl': float(z_fl_macro + z_fl_micro),
            'fr': float(z_fr_macro + z_fr_micro),
            'rl': float(z_rl_macro + z_rl_micro),
            'rr': float(z_rr_macro + z_rr_micro),
            'z_center': float(z_mid),
            'grade': float(grade_mid),
            'bank': float(bank_mid)
        }

    def apply_to_track_data(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhances track_data dictionary with 3D ribbon coordinates:
        Adds 'z_center', 'z_left', 'z_right', 'grade', 'bank' arrays.
        """
        if track_data is None:
            return track_data

        N = len(track_data.get('x_center', []))
        if N < 2:
            return track_data

        if self.config.enable_macro:
            z_cnt = np.array(self.z_macro, copy=True)
            grade = np.array(self.grade, copy=True)
            bank = np.array(self.bank_rad, copy=True)

            half_w = float(track_data.get('track_width_m', 3.8) / 2.0)
            sin_bank = np.sin(bank)
            z_l = z_cnt + half_w * sin_bank
            z_r = z_cnt - half_w * sin_bank
        else:
            z_cnt = np.zeros(N, dtype=float)
            grade = np.zeros(N, dtype=float)
            bank = np.zeros(N, dtype=float)
            z_l = np.zeros(N, dtype=float)
            z_r = np.zeros(N, dtype=float)

        # Micro-roughness on road edges if micro is enabled
        if self.config.enable_micro:
            s_arr = track_data.get('s', self.s)
            half_w = float(track_data.get('track_width_m', 3.8) / 2.0)
            for i in range(N):
                s_i = float(s_arr[i])
                z_l[i] += self.get_micro_unevenness(s_i, half_w)
                z_r[i] += self.get_micro_unevenness(s_i, -half_w)
                z_cnt[i] += self.get_micro_unevenness(s_i, 0.0)

        track_data['z_center'] = z_cnt
        track_data['z_left'] = z_l
        track_data['z_right'] = z_r
        track_data['grade'] = grade
        track_data['bank'] = bank
        track_data['road_profile'] = self
        track_data['_version'] = track_data.get('_version', 0) + 1
        return track_data
