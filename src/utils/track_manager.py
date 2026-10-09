"""
track_manager.py
----------------
2D Road Surface Generation (ISO 8608 Class A), Contact Patch Enveloping,
Real-Time Bilinear Lookup at 1000 Hz, and 3D Verge Geometry Extrusion.

Mathematical Standards & Specifications:
1. ISO 8608 Class A Road Roughness Power Spectral Density:
   G_d(n) = G_d(n0) * (n / n0)^(-w)
   where G_d(n0) = 4e-6 m^3, n0 = 0.1 cycles/m, w = 2.0.
   Target standard deviation: sigma_z ~ 1.41 mm.
2. 2D Stochastic Lateral Coherence:
   H(n_d) = (2 * d_lat) / (1 + (2*pi * n_d * d_lat)^2)
   with d_lat = 0.20 m, generating exponential lateral cross-correlation:
   rho(Delta d) = exp(-|Delta d| / d_lat), decaying for Delta d > 0.1 m.
3. Tire Contact Patch Spatial Envelope Filter:
   L_patch = 0.12 m spatial moving-average smoothing along s,
   preventing impulsive spikes in vertical contact velocity dot(z_road).
4. Real-Time Damping Rate:
   dot(z_road, i) = (dz_spline/ds + dZ_surf/ds) * v_x,w,i
5. 3D Verge / Grass Mesh Skirt:
   v_verge,left  = r_edge,left  - W_verge * b(s) - h_drop * k
   v_verge,right = r_edge,right + W_verge * b(s) - h_drop * k
   with W_verge = 12.0 m, h_drop = 0.4 m, color #2d4a22.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional, Any


class RoadSurfaceArray(np.ndarray):
    """Numpy ndarray subclass containing spatial grid metadata (s, d, ds, dd, length, width)."""
    def __new__(cls, input_array, s=None, d=None, ds=0.05, dd=0.05, length=100.0, width=4.0):
        obj = np.asarray(input_array).view(cls)
        obj.s = s
        obj.d = d
        obj.ds = ds
        obj.dd = dd
        obj.length = length
        obj.width = width
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.s = getattr(obj, 's', None)
        self.d = getattr(obj, 'd', None)
        self.ds = getattr(obj, 'ds', None)
        self.dd = getattr(obj, 'dd', None)
        self.length = getattr(obj, 'length', None)
        self.width = getattr(obj, 'width', None)


def generate_road_surface_2d(
    length: float,
    width: float,
    ds: float = 0.05,
    dd: float = 0.05,
    seed: Optional[int] = 42,
    return_coords: bool = False,
    is_closed: bool = True
) -> Any:
    """
    Generates a 2D elevation grid Z_surf(s, d) based on ISO 8608 Class A road roughness.

    Parameters:
    -----------
    length : float — Total track length along curvilinear s [m]
    width  : float — Total track width along lateral d [m]
    ds     : float — Longitudinal spatial discretization step [m] (default: 0.05m)
    dd     : float — Lateral spatial discretization step [m] (default: 0.05m)
    seed   : Optional[int] — Random seed for reproducible stochastic synthesis
    return_coords : bool — If True, returns (Z, s_grid, d_grid); otherwise returns Z
    is_closed : bool — If True, enforces periodic boundary conditions

    Returns:
    --------
    RoadSurfaceArray (shape: [Ns, Nd]) of surface elevation heights [m]
    """
    length = max(float(length), 10.0)
    width = max(float(width), 1.0)
    ds = max(float(ds), 0.01)
    dd = max(float(dd), 0.01)

    Ns = max(8, int(round(length / ds)))
    Nd = max(4, int(round(width / dd)) + 1)
    if Nd % 2 == 0:
        Nd += 1  # ensure odd number for exact zero centerline

    s_grid = np.linspace(0.0, length, Ns, endpoint=False, dtype=np.float64)
    d_grid = np.linspace(-width / 2.0, width / 2.0, Nd, dtype=np.float64)

    # Spatial frequencies
    ns = np.fft.fftfreq(Ns, d=ds)
    nd = np.fft.fftfreq(Nd, d=dd)
    NS, ND = np.meshgrid(ns, nd, indexing='ij')

    # ISO 8608 Class A Parameters
    n0 = 0.1           # reference spatial frequency [cycles/m]
    Gd_n0 = 4.0e-6     # Class A roughness coefficient [m^3]
    w = 2.0            # waviness exponent [-]
    d_lat = 0.20       # lateral correlation length [m]

    # Minimum frequency cutoff to prevent infinite variance at DC
    n_min = max(1.0 / length, 0.02)
    n_max = 1.0 / (2.0 * ds)

    # Longitudinal ISO 8608 PSD: G_d(n_s)
    n_eff = np.maximum(np.abs(NS), n_min)
    Gd_s = Gd_n0 * (n_eff / n0) ** (-w)

    # Lateral coherence filter: Lorentzian distribution in n_d
    # Corresponding to exponential decay in space: rho(Delta d) = exp(-|Delta d| / d_lat)
    H_d = (2.0 * d_lat) / (1.0 + (2.0 * np.pi * ND * d_lat) ** 2)

    # 2D Power Spectral Density & Differential elements
    S2D = Gd_s * H_d
    delta_ns = 1.0 / (Ns * ds)
    delta_nd = 1.0 / (Nd * dd)

    amplitude = np.sqrt(2.0 * S2D * delta_ns * delta_nd)

    # Stochastic phase synthesis
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(Ns, Nd))

    # Complex Fourier spectrum and 2D IFFT
    F = Ns * Nd * amplitude * np.exp(1j * phase)
    Z_raw = np.fft.ifft2(F).real

    # Zero-mean centring
    Z_centered = Z_raw - np.mean(Z_raw)

    # Calibrate variance to exact ISO 8608 Class A theoretical integral:
    # Var = int_{n_min}^{n_max} G_d(n) dn = Gd_n0 * n0^2 * (1/n_min - 1/n_max)
    target_var = Gd_n0 * (n0 ** 2) * (1.0 / n_min - 1.0 / n_max)
    target_std = math.sqrt(max(target_var, 1e-12))

    current_std = float(np.std(Z_centered))
    if current_std > 1e-12:
        Z_arr = (Z_centered * (target_std / current_std)).astype(np.float64)
    else:
        Z_arr = Z_centered.astype(np.float64)

    # Enforce smooth periodic loop at boundaries s=0 and s=L
    if is_closed:
        n_blend = min(20, Ns // 10)
        for b in range(n_blend):
            alpha = b / float(n_blend)
            Z_arr[b, :] = alpha * Z_arr[b, :] + (1.0 - alpha) * Z_arr[-(n_blend - b), :]

    # Wrap in RoadSurfaceArray with metadata
    res_surface = RoadSurfaceArray(
        Z_arr, s=s_grid, d=d_grid, ds=ds, dd=dd, length=length, width=width
    )

    if return_coords:
        return res_surface, s_grid, d_grid
    return res_surface


def apply_contact_patch_envelope_filter(
    Z: np.ndarray,
    ds: float,
    L_patch: float = 0.12
) -> np.ndarray:
    """
    Applies a spatial boxcar moving-average envelope filter of length L_patch [m]
    along the longitudinal track axis s.

    Prevents impulsive contact accelerations on unsprung wheel masses.
    """
    M = max(1, int(round(L_patch / ds)))
    if M <= 1:
        return np.array(Z, copy=True)

    Ns, Nd = Z.shape
    Z_out = np.empty_like(Z)

    # Cumulative sum with periodic wrapping along axis 0
    # Pad periodic boundary
    pad_len = M // 2
    Z_padded = np.pad(Z, ((pad_len, pad_len + (M % 2)), (0, 0)), mode='wrap')
    cumsum = np.cumsum(Z_padded, axis=0)

    # Moving average window
    Z_out = (cumsum[M:, :] - cumsum[:-M, :]) / float(M)
    return Z_out[:Ns, :]


class RoadSurface2D:
    """
    High-performance 2D Road Surface lookup engine operating at 1000 Hz.
    Provides sub-microsecond bilinear elevation queries and longitudinal slope rates.
    """

    def __init__(
        self,
        length: float,
        width: float,
        ds: float = 0.05,
        dd: float = 0.05,
        seed: Optional[int] = 42,
        L_patch: float = 0.12,
        Z_surf: Optional[np.ndarray] = None,
        is_closed: bool = True
    ):
        self.length = float(length)
        self.width = float(width)
        self.ds = float(ds)
        self.dd = float(dd)
        self.L_patch = float(L_patch)
        self.is_closed = is_closed

        if Z_surf is not None:
            self.Z = np.asarray(Z_surf, dtype=np.float64)
            self.Ns, self.Nd = self.Z.shape
            self.s_grid = getattr(Z_surf, 's', np.linspace(0.0, self.length, self.Ns, endpoint=False))
            self.d_grid = getattr(Z_surf, 'd', np.linspace(-self.width / 2.0, self.width / 2.0, self.Nd))
        else:
            self.Z, self.s_grid, self.d_grid = generate_road_surface_2d(
                self.length, self.width, self.ds, self.dd, seed=seed, return_coords=True, is_closed=self.is_closed
            )
            self.Ns, self.Nd = self.Z.shape

        # Precompute enveloped grid to ensure 0ms latency per 1000 Hz step
        self.Z_filtered = apply_contact_patch_envelope_filter(self.Z, self.ds, self.L_patch)

        # Precompute longitudinal gradient dZ/ds for fast damping rate dot(z_road)
        self.dZ_ds = np.gradient(self.Z_filtered, self.ds, axis=0)

    def sample(self, s: float, d: float) -> Tuple[float, float]:
        """
        Performs fast bilinear interpolation of ground height z_rough [m]
        and longitudinal slope dZ/ds [-] at curvilinear coordinates (s, d).

        Parameters:
        -----------
        s : float — Curvilinear longitudinal station along centerline [m]
        d : float — Lateral offset from centerline (+ left, - right) [m]

        Returns:
        --------
        (z_rough, dz_ds) : Tuple[float, float]
        """
        # Periodic modulo wrapping along track length (or clamp for open tracks)
        if self.is_closed:
            s_mod = s % self.length
        else:
            s_mod = max(0.0, min(s, self.length - 1e-6))
        # Lateral clamping guard beyond track borders
        w_half = self.width / 2.0
        d_clamped = min(max(d, -w_half), w_half)

        # Longitudinal grid index & bilinear fraction
        fs = s_mod / self.ds
        is0 = int(fs) % self.Ns
        is1 = (is0 + 1) % self.Ns
        ws1 = fs - int(fs)
        ws0 = 1.0 - ws1

        # Lateral grid index & bilinear fraction
        fd = (d_clamped + w_half) / self.dd
        id0 = min(max(int(fd), 0), self.Nd - 1)
        id1 = min(id0 + 1, self.Nd - 1)
        wd1 = min(max(fd - int(fd), 0.0), 1.0)
        wd0 = 1.0 - wd1

        # Bilinear elevation
        z00 = self.Z_filtered[is0, id0]
        z01 = self.Z_filtered[is0, id1]
        z10 = self.Z_filtered[is1, id0]
        z11 = self.Z_filtered[is1, id1]
        z_rough = float(ws0 * (wd0 * z00 + wd1 * z01) + ws1 * (wd0 * z10 + wd1 * z11))

        # Bilinear longitudinal slope dZ/ds
        g00 = self.dZ_ds[is0, id0]
        g01 = self.dZ_ds[is0, id1]
        g10 = self.dZ_ds[is1, id0]
        g11 = self.dZ_ds[is1, id1]
        dz_ds = float(ws0 * (wd0 * g00 + wd1 * g01) + ws1 * (wd0 * g10 + wd1 * g11))

        return z_rough, dz_ds


def extrude_verge_geometry(
    track_data: Dict[str, Any],
    w_verge: float = 8.0,
    h_drop: float = 0.15,
    z_base: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Extrudes a continuous 3D verge / grass ribbon mesh from track borders:
      v_verge,left  = r_edge,left  - W_verge * b(s) - h_drop * k
      v_verge,right = r_edge,right + W_verge * b(s) - h_drop * k

    Parameters:
    -----------
    track_data : Dict[str, Any] — Track geometry dictionary containing
                 'x_center', 'y_center', 'x_left', 'y_left', 'x_right', 'y_right',
                 'z_left', 'z_right', 'theta'.
    w_verge    : float — Verge ribbon horizontal width [m] (default: 12.0m)
    h_drop     : float — Vertical drainage drop at outer verge edge [m] (default: 0.4m)

    Returns:
    --------
    Dict[str, np.ndarray] containing verge vertices:
    'x_verge_l', 'y_verge_l', 'z_verge_l',
    'x_verge_r', 'y_verge_r', 'z_verge_r'
    """
    if 'x_left' not in track_data or 'x_right' not in track_data:
        if 'x_center' in track_data and 'y_center' in track_data:
            xc = np.asarray(track_data['x_center'], dtype=float)
            yc = np.asarray(track_data['y_center'], dtype=float)
            N_pts = len(xc)
            if N_pts < 2:
                return {}
            dx = np.gradient(xc)
            dy = np.gradient(yc)
            norm = np.hypot(dx, dy)
            norm[norm < 1e-6] = 1.0
            tx = dx / norm
            ty = dy / norm
            w_half = float(track_data.get('track_width_m', track_data.get('track_width', 4.0))) / 2.0
            x_left = xc - ty * w_half
            y_left = yc + tx * w_half
            x_right = xc + ty * w_half
            y_right = yc - tx * w_half
            theta = np.arctan2(ty, tx)
        else:
            return {}
    else:
        x_left = np.asarray(track_data['x_left'], dtype=float)
        y_left = np.asarray(track_data['y_left'], dtype=float)
        x_right = np.asarray(track_data['x_right'], dtype=float)
        y_right = np.asarray(track_data['y_right'], dtype=float)
        N_pts = len(x_left)
        theta = np.asarray(track_data.get('theta', np.zeros(N_pts)), dtype=float)

    z_left = np.asarray(track_data.get('z_left', np.zeros(N_pts)), dtype=float)
    z_right = np.asarray(track_data.get('z_right', np.zeros(N_pts)), dtype=float)

    # Unit binormal b(s) in horizontal plane pointing towards the right side of the track:
    # b(s) = (sin(theta), -cos(theta), 0)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Left Verge: v_verge,left = r_edge,left - W_verge * b(s) - h_drop * k
    x_verge_l = x_left - w_verge * sin_theta
    y_verge_l = y_left + w_verge * cos_theta
    z_verge_l = z_left - h_drop

    # Right Verge: v_verge,right = r_edge,right + W_verge * b(s) - h_drop * k
    x_verge_r = x_right + w_verge * sin_theta
    y_verge_r = y_right - w_verge * cos_theta
    z_verge_r = z_right - h_drop

    z_min = float(min(np.min(z_left), np.min(z_right))) if N_pts > 0 else 0.0
    if z_base is None:
        z_base = z_min - 1.5

    verge_dict = {
        'x_verge_l': x_verge_l, 'y_verge_l': y_verge_l, 'z_verge_l': z_verge_l,
        'x_verge_r': x_verge_r, 'y_verge_r': y_verge_r, 'z_verge_r': z_verge_r,
        'w_verge': float(w_verge),
        'h_drop': float(h_drop),
        'z_base': float(z_base)
    }
    track_data['verge'] = verge_dict
    return verge_dict



class TrackCollisionMesh:
    def __init__(self, track_data: dict, road_surface_2d=None, w_verge=12.0):
        self.is_closed = track_data.get('is_closed', True)
        self.w_verge = w_verge
        self.triangles = []  # ((v0, v1, v2), n, surface_type)
        self.cell_size = 2.0
        self.grid = {}
        self.road_surface_2d = road_surface_2d
        self._build_mesh(track_data)
        
    def _build_mesh(self, track_data):
        s_arr = track_data.get('s', np.array([]))
        if len(s_arr) < 2: return
        
        xc = np.asarray(track_data['x_center'])
        yc = np.asarray(track_data['y_center'])
        xl = track_data.get('x_left')
        yl = track_data.get('y_left')
        xr = track_data.get('x_right')
        yr = track_data.get('y_right')
        if xl is None or yl is None or xr is None or yr is None:
            w_half = float(track_data.get('track_width_m', track_data.get('track_width', 3.8))) / 2.0
            dx_c = np.gradient(xc)
            dy_c = np.gradient(yc)
            theta = np.arctan2(dy_c, dx_c)
            xl = xc - w_half * np.sin(theta)
            yl = yc + w_half * np.cos(theta)
            xr = xc + w_half * np.sin(theta)
            yr = yc - w_half * np.cos(theta)
            track_data['x_left'] = xl
            track_data['y_left'] = yl
            track_data['x_right'] = xr
            track_data['y_right'] = yr
        else:
            xl = np.asarray(xl)
            yl = np.asarray(yl)
            xr = np.asarray(xr)
            yr = np.asarray(yr)
        N = len(s_arr)
        
        if not self.is_closed:
            dx = xc[1] - xc[0]
            dy = yc[1] - yc[0]
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                tx, ty = dx/norm, dy/norm
            else:
                tx, ty = 1.0, 0.0
                
            ds = track_data.get('ds', 2.0)
            N_ext = max(1, int(12.0 / ds))
            s_ext = np.linspace(-12.0, 0.0, N_ext, endpoint=False)
            xc_ext = xc[0] + tx * s_ext
            yc_ext = yc[0] + ty * s_ext
            wx = xl[0] - xc[0]
            wy = yl[0] - yc[0]
            xl_ext = xc_ext + wx
            yl_ext = yc_ext + wy
            xr_ext = xc_ext - wx
            yr_ext = yc_ext - wy
            
            s_arr = np.concatenate([s_ext, s_arr])
            xc = np.concatenate([xc_ext, xc])
            yc = np.concatenate([yc_ext, yc])
            xl = np.concatenate([xl_ext, xl])
            yl = np.concatenate([yl_ext, yl])
            xr = np.concatenate([xr_ext, xr])
            yr = np.concatenate([yr_ext, yr])
            N = len(xc)
            
        zc = track_data.get('z_center', None)
        bank = track_data.get('bank', None)
        if zc is not None:
            zc = np.asarray(zc, dtype=float)
            if not self.is_closed and 'N_ext' in locals():
                zc = np.concatenate([np.full(N_ext, zc[0]), zc])
        if bank is not None:
            bank = np.asarray(bank, dtype=float)
            if not self.is_closed and 'N_ext' in locals():
                bank = np.concatenate([np.full(N_ext, bank[0]), bank])

        for i in range(N - 1):
            s_mid = (s_arr[i] + s_arr[i+1]) * 0.5
            z0_c = float(zc[i]) if zc is not None and len(zc) > i else 0.0
            z1_c = float(zc[i+1]) if zc is not None and len(zc) > i+1 else 0.0
            b0 = float(bank[i]) if bank is not None and len(bank) > i else 0.0
            b1 = float(bank[i+1]) if bank is not None and len(bank) > i+1 else 0.0
            tw0 = math.hypot(xl[i] - xr[i], yl[i] - yr[i])
            tw1 = math.hypot(xl[i+1] - xr[i+1], yl[i+1] - yr[i+1])
            z0_l = z0_c + (tw0 / 2.0) * math.sin(b0)
            z0_r = z0_c - (tw0 / 2.0) * math.sin(b0)
            z1_l = z1_c + (tw1 / 2.0) * math.sin(b1)
            z1_r = z1_c - (tw1 / 2.0) * math.sin(b1)
            if self.road_surface_2d:
                r0_l, _ = self.road_surface_2d.sample(s_arr[i], tw0 / 2.0)
                r0_r, _ = self.road_surface_2d.sample(s_arr[i], -tw0 / 2.0)
                r1_l, _ = self.road_surface_2d.sample(s_arr[i+1], tw1 / 2.0)
                r1_r, _ = self.road_surface_2d.sample(s_arr[i+1], -tw1 / 2.0)
                z0_l += r0_l
                z0_r += r0_r
                z1_l += r1_l
                z1_r += r1_r
                
            v0 = np.array([xl[i], yl[i], z0_l])
            v1 = np.array([xr[i], yr[i], z0_r])
            v2 = np.array([xl[i+1], yl[i+1], z1_l])
            v3 = np.array([xr[i+1], yr[i+1], z1_r])
            
            self._add_quad(v0, v1, v2, v3, 0)
            
    def _add_quad(self, v0, v1, v2, v3, surface_type):
        self._add_triangle(v0, v1, v2, surface_type)
        self._add_triangle(v1, v3, v2, surface_type)
        
    def _add_triangle(self, v0, v1, v2, surface_type):
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            n = n / norm
        else:
            n = np.array([0.0, 0.0, 1.0])
            
        tri = ((v0, v1, v2), n, surface_type)
        self.triangles.append(tri)
        
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        
        c_min_x = int(math.floor(min_x / self.cell_size))
        c_max_x = int(math.floor(max_x / self.cell_size))
        c_min_y = int(math.floor(min_y / self.cell_size))
        c_max_y = int(math.floor(max_y / self.cell_size))
        
        tri_idx = len(self.triangles) - 1
        for cx in range(c_min_x, c_max_x + 1):
            for cy in range(c_min_y, c_max_y + 1):
                cell = (cx, cy)
                if cell not in self.grid:
                    self.grid[cell] = []
                self.grid[cell].append(tri_idx)
                
    def raycast_down(self, origin, d_ray):
        ox = float(origin[0])
        oy = float(origin[1])
        oz = float(origin[2])
        cx = int(math.floor(ox / self.cell_size))
        cy = int(math.floor(oy / self.cell_size))
        
        best_t = float('inf')
        best_n = None
        best_stype = 0
        best_z = oz - 1.5

        # Check if ray is vertical (standard downward ground probe)
        is_vertical = (abs(d_ray[0]) < 1e-5 and abs(d_ray[1]) < 1e-5)
        
        neighbor_cells = (
            (cx, cy), (cx-1, cy), (cx+1, cy),
            (cx, cy-1), (cx, cy+1),
            (cx-1, cy-1), (cx+1, cy+1),
            (cx-1, cy+1), (cx+1, cy-1)
        )
        
        if is_vertical:
            # Ultra-fast zero-allocation 2D barycentric projection for vertical raycast
            for cell in neighbor_cells:
                cell_tris = self.grid.get(cell, None)
                if cell_tris is not None:
                    for idx in cell_tris:
                        (v0, v1, v2), n, stype = self.triangles[idx]
                        
                        x0, y0, z0 = float(v0[0]), float(v0[1]), float(v0[2])
                        x1, y1, z1 = float(v1[0]), float(v1[1]), float(v1[2])
                        x2, y2, z2 = float(v2[0]), float(v2[1]), float(v2[2])
                        
                        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                        if abs(denom) < 1e-9:
                            continue
                        inv_denom = 1.0 / denom
                        
                        dx2 = ox - x2
                        dy2 = oy - y2
                        l0 = ((y1 - y2) * dx2 + (x2 - x1) * dy2) * inv_denom
                        if l0 < -1e-4 or l0 > 1.0001:
                            continue
                        l1 = ((y2 - y0) * dx2 + (x0 - x2) * dy2) * inv_denom
                        if l1 < -1e-4 or l1 > 1.0001:
                            continue
                        l2 = 1.0 - l0 - l1
                        if l2 < -1e-4 or l2 > 1.0001:
                            continue
                        
                        z_hit = l0 * z0 + l1 * z1 + l2 * z2
                        t = oz - z_hit
                        if 1e-4 < t < best_t:
                            best_t = t
                            best_z = z_hit
                            best_n = n
                            best_stype = stype
            if best_n is not None:
                return np.array([ox, oy, best_z], dtype=np.float64), best_n, best_stype
        else:
            # General 3D Möller-Trumbore fallback
            for cell in neighbor_cells:
                cell_tris = self.grid.get(cell, None)
                if cell_tris is not None:
                    for idx in cell_tris:
                        (v0, v1, v2), n, stype = self.triangles[idx]
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        h = np.cross(d_ray, edge2)
                        a = np.dot(edge1, h)
                        if -1e-6 < a < 1e-6:
                            continue
                        f = 1.0 / a
                        s = origin - v0
                        u = f * np.dot(s, h)
                        if u < 0.0 or u > 1.0:
                            continue
                        q = np.cross(s, edge1)
                        v = f * np.dot(d_ray, q)
                        if v < 0.0 or u + v > 1.0:
                            continue
                        t = f * np.dot(edge2, q)
                        if 1e-6 < t < best_t:
                            best_t = t
                            best_n = n
                            best_stype = stype
            if best_n is not None:
                r_contact = origin + d_ray * best_t
                return r_contact, best_n, best_stype
            
        # fallback: retain expected road height instead of collapsing to 0.0
        expected_z = float(oz - 1.5)
        return np.array([ox, oy, expected_z], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64), 0
