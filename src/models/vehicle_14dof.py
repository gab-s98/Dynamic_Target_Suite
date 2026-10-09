"""
vehicle_14dof.py
----------------
14-DOF Full Vehicle Model (Planar + Roll + Pitch + 4-Corner Wheel Travel + 4-Wheel Spin).

Features:
- Steady-state Trim with 11 states and exact analytical Jacobians.
- Full 14-DOF Transient ODE with coupled roll/pitch/suspension compliance.
- Kinematic polynomial maps (camber gain, bump-steer, roll centers, anti-features, Ackermann).
- Support for 2D Aero Maps (CFD ride height sensitivity).
- Decoupled ride, pitch, and roll stiffness evaluation.
"""

import os
import hashlib
import casadi as ca
import numpy as np
from src.models.tire_model_mf61 import (
    mf61_lateral_force, 
    mf61_longitudinal_force, 
    mf61_combined_lateral_force, 
    mf61_combined_longitudinal_force
)
from src.models.damper_model_ttx25 import calculate_wheel_damper_force
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams


def _poly3_ca(coeffs, x):
    """Evaluate a degree-3 polynomial with CasADi symbolic variable."""
    a3, a2, a1, a0 = coeffs
    return a3 * (x**3) + a2 * (x**2) + a1 * x + a0


class ODEFunctionWrapper:
    """
    Transparent callable wrapper around CasADi Function / External ODE.
    Guarantees backward compatibility:
      - If caller supplies 16 parameters, Cd is seamlessly appended from default_cd.
      - If caller supplies 23 states, automatically expands to 28 states for ODE evaluation
        and projects derivatives back to 23 states for legacy solvers/tests.
      - Ensures contiguous float64 memory alignment for ctypes / C DLL execution.
    """
    def __init__(self, raw_func, default_cd=1.20):
        self._raw_func = raw_func
        self.default_cd = float(default_cd)

    def __call__(self, x_val, u_val, p_val):
        is_23_state = False
        n_x = getattr(x_val, 'numel', lambda: len(x_val))() if hasattr(x_val, 'numel') else len(x_val)
        if n_x == 23:
            is_23_state = True
            if isinstance(x_val, np.ndarray):
                x_val = np.insert(x_val, 7, 0.0)
                x_val = np.insert(x_val, [12, 12, 12, 12], [0.0, 0.0, 0.0, 0.0])
            elif isinstance(x_val, (list, tuple)):
                x_list = list(x_val)
                x_list.insert(7, 0.0)
                for _ in range(4):
                    x_list.insert(12, 0.0)
                x_val = x_list

        n_p = getattr(p_val, 'numel', lambda: len(p_val))() if hasattr(p_val, 'numel') else len(p_val)
        if n_p < 37:
            pad_len = 37 - n_p
            pad = [0.0] * pad_len
            if n_p == 16:
                pad[0] = self.default_cd
            if 37 - n_p >= 12:
                pad[-10] = 1.0
                pad[-7] = 1.0
                pad[-4] = 1.0
                pad[-1] = 1.0
            if isinstance(p_val, (list, tuple)):
                p_val = list(p_val) + pad
            elif isinstance(p_val, np.ndarray):
                p_val = np.append(p_val, pad)
            elif isinstance(p_val, (ca.MX, ca.SX)):
                p_val = ca.vertcat(p_val, *pad)

        if isinstance(x_val, np.ndarray) and not x_val.flags.c_contiguous:
            x_val = np.ascontiguousarray(x_val, dtype=np.float64)
        if isinstance(u_val, np.ndarray) and not u_val.flags.c_contiguous:
            u_val = np.ascontiguousarray(u_val, dtype=np.float64)
        if isinstance(p_val, np.ndarray) and not p_val.flags.c_contiguous:
            p_val = np.ascontiguousarray(p_val, dtype=np.float64)

        res = self._raw_func(x_val, u_val, p_val)
        if is_23_state:
            res_arr = np.array(res).flatten()
            res_23 = np.delete(res_arr, [7, 12, 13, 14, 15])
            if isinstance(res, (ca.DM, ca.MX, ca.SX)):
                return ca.DM(res_23)
            return res_23
        return res

    def __getattr__(self, item):
        return getattr(self._raw_func, item)

    def __str__(self):
        return str(self._raw_func)

    def __repr__(self):
        return repr(self._raw_func)


class Vehicle14DOF:
    """
    14-DOF Full Vehicle Model (Planar + Roll + Pitch + Wheel Travel + Wheel Spin).

    Kinematic maps (camber, toe, roll-center, anti-features, ackermann)
    are encoded as CasADi symbolic polynomial functions whose coefficients
    are loaded from VehicleParams and can be updated from the Curve Digitizer.
    """

    def __init__(self, vp: VehicleParams = None,
                 tire_params_f: TireParams = None,
                 tire_params_r: TireParams = None):
        self.vp = vp or VehicleParams()
        if tire_params_f is None:
            tir_name = getattr(self.vp, 'tire_file', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir')
            tir_path = tir_name
            if not os.path.isabs(tir_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(base_dir, "tire_tir", tir_name)
                if os.path.exists(candidate):
                    tir_path = candidate
                elif os.path.exists(tir_name):
                    tir_path = tir_name
                else:
                    default_tir = os.path.join(base_dir, "tire_tir", "Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir")
                    if os.path.exists(default_tir):
                        tir_path = default_tir
            self.tp_f = TireParams(tir_path)
        else:
            self.tp_f = tire_params_f

        if tire_params_r is None:
            self.tp_r = self.tp_f
        else:
            self.tp_r = tire_params_r

        self.z_road = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.dz_road = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}

    def set_road_elevations(self, z_fl: float = 0.0, z_fr: float = 0.0, z_rl: float = 0.0, z_rr: float = 0.0,
                            dz_fl: float = 0.0, dz_fr: float = 0.0, dz_rl: float = 0.0, dz_rr: float = 0.0):
        """Sets per-wheel road surface elevation [m] and vertical rate [m/s]."""
        self.z_road['fl'] = float(z_fl)
        self.z_road['fr'] = float(z_fr)
        self.z_road['rl'] = float(z_rl)
        self.z_road['rr'] = float(z_rr)
        self.dz_road['fl'] = float(dz_fl)
        self.dz_road['fr'] = float(dz_fr)
        self.dz_road['rl'] = float(dz_rl)
        self.dz_road['rr'] = float(dz_rr)

    def get_road_elevations(self) -> dict:
        return dict(self.z_road)

    @staticmethod
    def get_equations_markdown():
        return r"""
### Modello 14-DOF — Full Vehicle (Trim & Transient ODE)

**Gradi di Libertà (14):**
- *Moto Planare*: Velocità Longitudinale ($v_x$), Laterale ($v_y$), Imbardata ($r$).
- *Dinamica di Rollio*: Angolo di rollio ($\phi$) e velocità di rollio ($\dot{\phi}$).
- *Dinamica di Beccheggio*: Angolo di beccheggio ($\theta$) e velocità di beccheggio ($\dot{\theta}$).
- *Escursione Sospensioni*: Corsa verticale per ciascuna ruota ($z_{fl}, z_{fr}, z_{rl}, z_{rr}$).
- *Cinematica Ruote*: Velocità angolari delle 4 ruote ($\omega_{fl}, \omega_{fr}, \omega_{rl}, \omega_{rr}$).

**Equazioni di Equilibrio e Dinamica:**

1. **Dinamica Planare Telaio:**
$$\dot{v}_x = \frac{\sum F_x}{m} + v_y r$$
$$\dot{v}_y = \frac{\sum F_y}{m} - v_x r$$
$$\dot{r} = \frac{\sum M_z}{I_z}$$

2. **Dinamica e Trim di Rollio:**
$$I_{x} \ddot{\phi} = M_{roll} - (K_{\phi,f} + K_{\phi,r}) \phi - (C_{\phi,f} + C_{\phi,r}) \dot{\phi}$$
dove $M_{roll} = m a_y (h_{cg} - h_{rc,avg})$ è il momento ribaltante attorno all'asse dei centri di rollio.

3. **Escursione Sospensioni & Carichi Verticali:**
$$K_{z,i} z_i = F_{z,tgt,i} - F_{z,0,i}$$
dove $F_{z,0,i}$ è il carico statico a terra (1g con pilota), garantendo $z_i = 0$ a vettura ferma.
con $K_{heave} = 2K_{z,f} + 2K_{z,r}$ e $K_{\theta} = 2K_{z,f}a^2 + 2K_{z,r}b^2$.

4. **Mappe Cinematiche Non-Lineari (Curve Digitizer):**
- Campanatura dinamica: $\gamma_i(z_i) = f_{camber}(z_i) + \gamma_{static}$
- Convergenza dinamica: $\delta_{toe,i}(z_i) = f_{toe}(z_i) + \delta_{toe,static}$
- Centro di rollio dinamico: $h_{rc}(z_i) = f_{rc}(z_i)$
- Geometria Ackermann: $\Delta\delta_{ack} = f_{ack}(\delta_{steer})$
"""

    def get_trim_casadi_functions(self, K_phi_f=None, K_phi_r=None,
                                   K_z_f=None, K_z_r=None,
                                   L=None, t_f=None, t_r=None,
                                   wd_f=None, lltd=None, aero_bal=None,
                                   camber_f_static=None, camber_r_static=None,
                                   toe_f_static=None, toe_r_static=None):
        """
        Build CasADi trim residual functions for the 14-DOF model.
        Returns dict with res_func, analytical Jacobians, and metadata.
        """
        vp = self.vp

        # ------------------------------------------------------------------
        # Inputs (fixed during trim evaluation)
        # ------------------------------------------------------------------
        v_x     = ca.SX.sym('v_x')
        delta_s = ca.SX.sym('delta_steer')
        inputs  = ca.vertcat(v_x, delta_s)

        # ------------------------------------------------------------------
        # States (11 states solved by Newton in steady-state cornering)
        # ------------------------------------------------------------------
        v_y     = ca.SX.sym('v_y')
        r       = ca.SX.sym('r')
        phi     = ca.SX.sym('phi')          # Roll angle [rad]
        z_fl    = ca.SX.sym('z_fl')         # Wheel travel [m] (+ = compression)
        z_fr    = ca.SX.sym('z_fr')
        z_rl    = ca.SX.sym('z_rl')
        z_rr    = ca.SX.sym('z_rr')
        w_fl    = ca.SX.sym('w_fl')
        w_fr    = ca.SX.sym('w_fr')
        w_rl    = ca.SX.sym('w_rl')
        w_rr    = ca.SX.sym('w_rr')
        states  = ca.vertcat(v_y, r, phi, z_fl, z_fr, z_rl, z_rr,
                             w_fl, w_fr, w_rl, w_rr)

        # ------------------------------------------------------------------
        # Design parameters (15 params exposed for MDO)
        # ------------------------------------------------------------------
        p_wd_f    = ca.SX.sym('wd_f')
        p_lltd    = ca.SX.sym('lltd')
        p_Kphi_f  = ca.SX.sym('K_phi_f')
        p_Kphi_r  = ca.SX.sym('K_phi_r')
        p_Kz_f    = ca.SX.sym('K_z_f')
        p_Kz_r    = ca.SX.sym('K_z_r')
        p_ab      = ca.SX.sym('aero_bal')
        p_camf    = ca.SX.sym('camber_f_static')
        p_camr    = ca.SX.sym('camber_r_static')
        p_toef    = ca.SX.sym('toe_f_static')
        p_toer    = ca.SX.sym('toe_r_static')
        p_L       = ca.SX.sym('L')
        p_tf      = ca.SX.sym('t_f')
        p_tr      = ca.SX.sym('t_r')
        p_lmuy    = ca.SX.sym('lmuy_mult')

        params = ca.vertcat(p_wd_f, p_lltd, p_Kphi_f, p_Kphi_r,
                            p_Kz_f, p_Kz_r, p_ab,
                            p_camf, p_camr, p_toef, p_toer,
                            p_L, p_tf, p_tr, p_lmuy)

        # ------------------------------------------------------------------
        # Kinematic Maps (3rd-order polynomials from Curve Digitizer)
        # ------------------------------------------------------------------
        # Steering Wheel / Pinion angle [deg]
        deg2rad = np.deg2rad(1.0)
        gr_steer = getattr(vp, 'steering_ratio', 5.0)
        delta_sw_deg = (delta_s / deg2rad) * gr_steer

        # Differential Steering Kinematics (Adams Car direct curve or Ackermann blend)
        w_steer_coeffs = getattr(vp, 'wheel_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
        if any(abs(c) > 1e-6 for c in w_steer_coeffs[:3]):
            delta_fl_deg = _poly3_ca(w_steer_coeffs, delta_sw_deg)
            delta_fr_deg = -_poly3_ca(w_steer_coeffs, -delta_sw_deg)
            delta_fl = delta_fl_deg * deg2rad
            delta_fr = delta_fr_deg * deg2rad
        else:
            ack_pct = vp.ackermann_pct
            tan_d = ca.tan(delta_s)
            delta_ack_fl = ca.arctan2(p_L * tan_d, p_L - (p_tf / 2.0) * tan_d)
            delta_ack_fr = ca.arctan2(p_L * tan_d, p_L + (p_tf / 2.0) * tan_d)
            delta_fl = delta_s + ack_pct * (delta_ack_fl - delta_s)
            delta_fr = delta_s + ack_pct * (delta_ack_fr - delta_s)

        # Camber Steer delta from steering angle [deg]
        camber_steer_coeffs = getattr(vp, 'camber_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
        delta_sw_fl_deg = (delta_fl / deg2rad) * gr_steer
        delta_sw_fr_deg = (delta_fr / deg2rad) * gr_steer
        dcam_steer_fl_deg = _poly3_ca(camber_steer_coeffs, delta_sw_fl_deg)
        dcam_steer_fr_deg = _poly3_ca(camber_steer_coeffs, -delta_sw_fr_deg)

        # Check calibrated TYRESIDE from .tir property files
        is_right_f = getattr(self.tp_f, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_f else True
        is_right_r = getattr(self.tp_r, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_r else True

        # Camber angle per wheel [deg -> rad]
        cam_fl_deg = _poly3_ca(vp.camber_gain_f_coeffs, z_fl) + p_camf + dcam_steer_fl_deg
        cam_fr_deg = _poly3_ca(vp.camber_gain_f_coeffs, z_fr) + p_camf + dcam_steer_fr_deg
        cam_rl_deg = _poly3_ca(vp.camber_gain_r_coeffs, z_rl) + p_camr
        cam_rr_deg = _poly3_ca(vp.camber_gain_r_coeffs, z_rr) + p_camr

        if is_right_f:
            gamma_fl = -cam_fl_deg * deg2rad
            gamma_fr =  cam_fr_deg * deg2rad
        else:
            gamma_fl =  cam_fl_deg * deg2rad
            gamma_fr = -cam_fr_deg * deg2rad

        if is_right_r:
            gamma_rl = -cam_rl_deg * deg2rad
            gamma_rr =  cam_rr_deg * deg2rad
        else:
            gamma_rl =  cam_rl_deg * deg2rad
            gamma_rr = -cam_rr_deg * deg2rad

        # Bump steer (Toe angle per wheel [deg -> rad])
        toe_fl_deg = _poly3_ca(vp.toe_gain_f_coeffs, z_fl) + p_toef
        toe_fr_deg = _poly3_ca(vp.toe_gain_f_coeffs, z_fr) + p_toef
        toe_rl_deg = _poly3_ca(vp.toe_gain_r_coeffs, z_rl) + p_toer
        toe_rr_deg = _poly3_ca(vp.toe_gain_r_coeffs, z_rr) + p_toer

        # Dynamic roll center heights & lateral migration (function of chassis roll angle phi [deg])
        phi_deg = phi * (180.0 / ca.pi)
        h_rc_f = _poly3_ca(vp.rollcenter_f_coeffs, phi_deg)
        h_rc_r = _poly3_ca(vp.rollcenter_r_coeffs, phi_deg)
        h_rc_avg = (h_rc_f + h_rc_r) / 2.0

        # Dynamic roll center lateral migration [m]
        y_rc_f = _poly3_ca(getattr(vp, 'rollcenter_lat_f_coeffs', [0.0, 0.0, 0.06453, 0.0]), phi_deg)
        y_rc_r = _poly3_ca(getattr(vp, 'rollcenter_lat_r_coeffs', [0.0, 0.0, 0.04213, 0.0]), phi_deg)
        y_rc_avg = (y_rc_f + y_rc_r) / 2.0

        # Geometry
        a = p_L * (1.0 - p_wd_f)
        b = p_L * p_wd_f

        # Aerodynamics (nominal Cz without ground clearance sensitivity)
        z_f_avg = (z_fl + z_fr) / 2.0
        z_r_avg = (z_rl + z_rr) / 2.0
        
        # Aero downforce
        F_dwnf = 0.5 * vp.rho * vp.S * vp.Cz * (v_x**2)
        Fz_aero_f = F_dwnf * p_ab
        Fz_aero_r = F_dwnf * (1.0 - p_ab)

        Ay = v_x * r

        # ------------------------------------------------------------------
        # Roll Equilibrium (including dynamic CG height, lateral roll center migration and gravity overturning)
        # ------------------------------------------------------------------
        z_cg = (b / p_L) * z_f_avg + (a / p_L) * z_r_avg
        h_cg_dyn = ca.fmax(0.05, vp.h_cg - z_cg)
        K_phi_tot = p_Kphi_f + p_Kphi_r
        h_roll_arm = (h_cg_dyn - h_rc_avg) * ca.cos(phi) - y_rc_avg * ca.sin(phi)
        M_roll = vp.m * Ay * h_roll_arm + vp.m * vp.g * (h_cg_dyn - h_rc_avg) * ca.sin(phi)
        res_roll = K_phi_tot * phi - M_roll

        # ------------------------------------------------------------------
        # Wheel Travel Equilibrium (Reference z_i = 0 at static 1g ground equilibrium)
        # ------------------------------------------------------------------
        Fz_0_f = vp.m * vp.g * (b / p_L)
        Fz_0_r = vp.m * vp.g * (a / p_L)
        Fz_0_fl = Fz_0_f / 2.0
        Fz_0_fr = Fz_0_f / 2.0
        Fz_0_rl = Fz_0_r / 2.0
        Fz_0_rr = Fz_0_r / 2.0

        Fz_stat_f = Fz_0_f + Fz_aero_f
        Fz_stat_r = Fz_0_r + Fz_aero_r

        # Elastic LLT (based on roll stiffness distribution)
        LLTD_elastic = p_Kphi_f / ca.fmax(K_phi_tot, 1.0)
        LLTD_eff = 0.5 * (p_lltd + LLTD_elastic)

        delta_Fz_f_llt = (vp.m * Ay * h_roll_arm * LLTD_eff) / p_tf
        delta_Fz_r_llt = (vp.m * Ay * h_roll_arm * (1.0 - LLTD_eff)) / p_tr

        # Geometric load transfer via roll centers (with lateral migration)
        delta_Fz_f_geo = (vp.m * Ay * (b / p_L) * (h_rc_f + y_rc_f * (h_rc_f / (p_tf / 2.0)))) / p_tf
        delta_Fz_r_geo = (vp.m * Ay * (a / p_L) * (h_rc_r + y_rc_r * (h_rc_r / (p_tr / 2.0)))) / p_tr

        delta_Fz_f_tot = delta_Fz_f_llt + delta_Fz_f_geo
        delta_Fz_r_tot = delta_Fz_r_llt + delta_Fz_r_geo

        Fz_tgt_fl = Fz_stat_f / 2.0 - delta_Fz_f_tot
        Fz_tgt_fr = Fz_stat_f / 2.0 + delta_Fz_f_tot
        Fz_tgt_rl = Fz_stat_r / 2.0 - delta_Fz_r_tot
        Fz_tgt_rr = Fz_stat_r / 2.0 + delta_Fz_r_tot

        # Wheel travel residuals: K_z * z_i = Fz_target - Fz_0 (z_i = 0 at static 1g ground equilibrium)
        res_z_fl = p_Kz_f * z_fl - (Fz_tgt_fl - Fz_0_fl)
        res_z_fr = p_Kz_f * z_fr - (Fz_tgt_fr - Fz_0_fr)
        res_z_rl = p_Kz_r * z_rl - (Fz_tgt_rl - Fz_0_rl)
        res_z_rr = p_Kz_r * z_rr - (Fz_tgt_rr - Fz_0_rr)

        # Clamped contact forces (bounded to avoid exponential blowup: Fz_i in [20.0, 0.90 * m * g])
        Fz_max_trim = vp.m * vp.g * 0.90
        Fz_fl = ca.fmax(ca.fmin(Fz_tgt_fl, Fz_max_trim), 20.0)
        Fz_fr = ca.fmax(ca.fmin(Fz_tgt_fr, Fz_max_trim), 20.0)
        Fz_rl = ca.fmax(ca.fmin(Fz_tgt_rl, Fz_max_trim), 20.0)
        Fz_rr = ca.fmax(ca.fmin(Fz_tgt_rr, Fz_max_trim), 20.0)

        # ------------------------------------------------------------------
        # Kinematics & Slip Angles
        # ------------------------------------------------------------------
        eps = 0.1
        Vx_fl = v_x - r * p_tf / 2.0
        Vy_fl = v_y + r * a
        Vx_fr = v_x + r * p_tf / 2.0
        Vy_fr = v_y + r * a
        Vx_rl = v_x - r * p_tr / 2.0
        Vy_rl = v_y - r * b
        Vx_rr = v_x + r * p_tr / 2.0
        Vy_rr = v_y - r * b

        Vxw_fl =  Vx_fl * ca.cos(delta_fl) + Vy_fl * ca.sin(delta_fl)
        Vyw_fl = -Vx_fl * ca.sin(delta_fl) + Vy_fl * ca.cos(delta_fl)
        Vxw_fr =  Vx_fr * ca.cos(delta_fr) + Vy_fr * ca.sin(delta_fr)
        Vyw_fr = -Vx_fr * ca.sin(delta_fr) + Vy_fr * ca.cos(delta_fr)
        Vxw_rl =  Vx_rl
        Vyw_rl =  Vy_rl
        Vxw_rr =  Vx_rr
        Vyw_rr =  Vy_rr

        alpha_fl = -ca.atan2(Vyw_fl, ca.fmax(ca.fabs(Vxw_fl), 0.1)) + toe_fl_deg * deg2rad
        alpha_fr = -ca.atan2(Vyw_fr, ca.fmax(ca.fabs(Vxw_fr), 0.1)) - toe_fr_deg * deg2rad
        alpha_rl = -ca.atan2(Vyw_rl, ca.fmax(ca.fabs(Vxw_rl), 0.1)) - toe_rl_deg * deg2rad
        alpha_rr = -ca.atan2(Vyw_rr, ca.fmax(ca.fabs(Vxw_rr), 0.1)) + toe_rr_deg * deg2rad

        # Lateral tire forces with TYRESIDE-consistent ply-steer mirroring
        lvy_fl = -1.0 if is_right_f else 1.0
        lvy_fr =  1.0 if is_right_f else -1.0
        lvy_rl = -1.0 if is_right_r else 1.0
        lvy_rr =  1.0 if is_right_r else -1.0

        lhy_fl = -1.0 if is_right_f else 1.0
        lhy_fr =  1.0 if is_right_f else -1.0
        lhy_rl = -1.0 if is_right_r else 1.0
        lhy_rr =  1.0 if is_right_r else -1.0

        fy_fl = -mf61_lateral_force(alpha_fl, Fz_fl, gamma_fl, self.tp_f, LMUY=p_lmuy, LVY=lvy_fl, LHY=lhy_fl)
        fy_fr = -mf61_lateral_force(alpha_fr, Fz_fr, gamma_fr, self.tp_f, LMUY=p_lmuy, LVY=lvy_fr, LHY=lhy_fr)
        fy_rl = -mf61_lateral_force(alpha_rl, Fz_rl, gamma_rl, self.tp_r, LMUY=p_lmuy, LVY=lvy_rl, LHY=lhy_rl)
        fy_rr = -mf61_lateral_force(alpha_rr, Fz_rr, gamma_rr, self.tp_r, LMUY=p_lmuy, LVY=lvy_rr, LHY=lhy_rr)

        # Lateral tire forces with TYRESIDE-consistent ply-steer mirroring (continuous and smooth for trim)

        Fy_fl_ch = fy_fl * ca.cos(delta_fl)
        Fy_fr_ch = fy_fr * ca.cos(delta_fr)
        Fy_f = Fy_fl_ch + Fy_fr_ch
        Fy_r = fy_rl + fy_rr

        # Planar residuals
        res_lat = Fy_f + Fy_r - vp.m * v_x * r
        res_yaw = a * Fy_f - b * Fy_r + (p_tf / 2.0) * (fy_fl * ca.sin(delta_fl) - fy_fr * ca.sin(delta_fr))

        # Pure rolling wheel spin residuals
        Rw = vp.R_w
        res_wfl = w_fl - Vxw_fl / Rw
        res_wfr = w_fr - Vxw_fr / Rw
        res_wrl = w_rl - Vxw_rl / Rw
        res_wrr = w_rr - Vxw_rr / Rw

        residuals = ca.vertcat(
            res_lat, res_yaw, res_roll,
            res_z_fl, res_z_fr, res_z_rl, res_z_rr,
            res_wfl, res_wfr, res_wrl, res_wrr
        )

        in_args = [inputs, states, params]
        jac_states = ca.jacobian(residuals, states)
        jac_inputs = ca.jacobian(residuals, inputs)
        jac_params = ca.jacobian(residuals, params)

        return {
            'res_func':        ca.Function('res14', in_args, [residuals]),
            'jac_states_func': ca.Function('jac14_st', in_args, [jac_states]),
            'jac_inputs_func': ca.Function('jac14_in', in_args, [jac_inputs]),
            'jac_params_func': ca.Function('jac14_p',  in_args, [jac_params]),
            'meta': {
                'inputs': ['v_x', 'delta_steer'],
                'states': ['v_y', 'r', 'phi',
                           'z_fl', 'z_fr', 'z_rl', 'z_rr',
                           'w_fl', 'w_fr', 'w_rl', 'w_rr'],
                'params': ['wd_f', 'lltd', 'K_phi_f', 'K_phi_r',
                           'K_z_f', 'K_z_r', 'aero_bal',
                           'camber_f_static', 'camber_r_static',
                           'toe_f_static', 'toe_r_static',
                           'L', 't_f', 't_r', 'lmuy_mult']
            }
        }

    def get_casadi_ode(self):
        """
        Returns a CasADi Function for the 28-state Full-Vehicle Transient ODE.

        State vector x (28) — ISO sign convention (Z positive up):
          [ 0] v_x        Chassis longitudinal velocity [m/s]
          [ 1] v_y        Chassis lateral velocity [m/s]
          [ 2] r          Yaw rate [rad/s]
          [ 3] phi        Roll angle [rad]
          [ 4] phi_dot    Roll rate [rad/s]
          [ 5] theta      Pitch angle [rad]
          [ 6] theta_dot  Pitch rate [rad/s]
          [ 7] w_heave    Chassis CG heave velocity ISO Z+ [m/s]            ← v9 NEW
          [ 8] z_fl       Suspension stroke FL [m]  (+ = bump/compression from 1g)
          [ 9] z_fr       Suspension stroke FR [m]
          [10] z_rl       Suspension stroke RL [m]
          [11] z_rr       Suspension stroke RR [m]
          [12] vz_fl      Suspension compression rate FL [m/s]               ← v9 NEW
          [13] vz_fr      Suspension compression rate FR [m/s]               ← v9 NEW
          [14] vz_rl      Suspension compression rate RL [m/s]               ← v9 NEW
          [15] vz_rr      Suspension compression rate RR [m/s]               ← v9 NEW
          [16] w_fl       Wheel spin angular velocity FL [rad/s]
          [17] w_fr       Wheel spin angular velocity FR [rad/s]
          [18] w_rl       Wheel spin angular velocity RL [rad/s]
          [19] w_rr       Wheel spin angular velocity RR [rad/s]
          [20] ux_fl      Carcass longitudinal elastic deflection FL [m]
          [21] uy_fl      Carcass lateral elastic deflection FL [m]
          [22] ux_fr      Carcass longitudinal elastic deflection FR [m]
          [23] uy_fr      Carcass lateral elastic deflection FR [m]
          [24] ux_rl      Carcass longitudinal elastic deflection RL [m]
          [25] uy_rl      Carcass lateral elastic deflection RL [m]
          [26] ux_rr      Carcass longitudinal elastic deflection RR [m]
          [27] uy_rr      Carcass lateral elastic deflection RR [m]

        Inputs u (4):   [delta, T_drive_rl, T_drive_rr, T_brake]
        Params p (17):  [wd_f, lltd, K_phi_f, K_phi_r, K_z_f, K_z_r, aero_bal,
                         cam_f, cam_r, toe_f, toe_r, L, t_f, t_r, lmuy, lmux, Cd]

        Physics changes vs. 23-state v8:
          - Fz_i emerges from radial tire spring (VERTICAL_STIFFNESS from .tir) +
            damping (C_tire_z [VehicleParams]), replacing the quasi-static LLT formula.
          - tau_z first-order filter eliminated; each hub (m_us_f/r [kg]) is a 2nd-order
            mass: m_us * vz_dot = (K_tz − K_eff)*z + C_tz*vz − F_damp.
          - Per-corner damper forces driven by vz_i (actual compression rate) rather than
            the modal approximation phi_dot*(t/2) / theta_dot*a.
          - Chassis heave DOF (deviational form, static preload pre-cancelled):
            m_s * w_dot = Σ(K_eff_i*z_i + Fd_i) − F_down.
          - Roll/pitch damping from per-corner forces: M_roll_damp = Σ(arm × Fd_i),
            M_pitch_damp = a*(Fd_fl+Fd_fr) − b*(Fd_rl+Fd_rr).
        """
        # ------------------------------------------------------------------
        # Symbolic State Variables
        # ------------------------------------------------------------------
        v_x       = ca.MX.sym('v_x')
        v_y       = ca.MX.sym('v_y')
        r         = ca.MX.sym('r')
        phi       = ca.MX.sym('phi')
        phi_dot   = ca.MX.sym('phi_dot')
        theta     = ca.MX.sym('theta')
        theta_dot = ca.MX.sym('theta_dot')
        w_heave   = ca.MX.sym('w_heave')   # [7] chassis CG heave velocity [m/s]

        # Suspension strokes (bump-positive; same semantics as former z_fl..z_rr) [8..11]
        z_fl = ca.MX.sym('z_fl')
        z_fr = ca.MX.sym('z_fr')
        z_rl = ca.MX.sym('z_rl')
        z_rr = ca.MX.sym('z_rr')

        # Suspension compression rates — 2nd-order hub states [12..15]
        vz_fl = ca.MX.sym('vz_fl')
        vz_fr = ca.MX.sym('vz_fr')
        vz_rl = ca.MX.sym('vz_rl')
        vz_rr = ca.MX.sym('vz_rr')

        # Wheel spin speeds [16..19]
        w_fl = ca.MX.sym('w_fl')
        w_fr = ca.MX.sym('w_fr')
        w_rl = ca.MX.sym('w_rl')
        w_rr = ca.MX.sym('w_rr')

        # Contact Patch Carcass Elastic Deflection States (Pacejka Relaxation Length Model) [20..27]
        ux_fl = ca.MX.sym('ux_fl');  uy_fl = ca.MX.sym('uy_fl')
        ux_fr = ca.MX.sym('ux_fr');  uy_fr = ca.MX.sym('uy_fr')
        ux_rl = ca.MX.sym('ux_rl');  uy_rl = ca.MX.sym('uy_rl')
        ux_rr = ca.MX.sym('ux_rr');  uy_rr = ca.MX.sym('uy_rr')

        x = ca.vertcat(
            v_x, v_y, r, phi, phi_dot, theta, theta_dot, w_heave,
            z_fl, z_fr, z_rl, z_rr,
            vz_fl, vz_fr, vz_rl, vz_rr,
            w_fl, w_fr, w_rl, w_rr,
            ux_fl, uy_fl, ux_fr, uy_fr,
            ux_rl, uy_rl, ux_rr, uy_rr,
        )


        # Inputs
        delta      = ca.MX.sym('delta')
        T_drive_rl = ca.MX.sym('T_drive_rl')
        T_drive_rr = ca.MX.sym('T_drive_rr')
        T_brake    = ca.MX.sym('T_brake')
        u = ca.vertcat(delta, T_drive_rl, T_drive_rr, T_brake)

        # Parameters
        wd_f     = ca.MX.sym('wd_f')
        lltd     = ca.MX.sym('lltd')
        K_phi_f  = ca.MX.sym('K_phi_f')
        K_phi_r  = ca.MX.sym('K_phi_r')
        K_z_f    = ca.MX.sym('K_z_f')
        K_z_r    = ca.MX.sym('K_z_r')
        aero_bal = ca.MX.sym('aero_bal')
        cam_f    = ca.MX.sym('cam_f')
        cam_r    = ca.MX.sym('cam_r')
        toe_f    = ca.MX.sym('toe_f')
        toe_r    = ca.MX.sym('toe_r')
        L_sym    = ca.MX.sym('L')
        tf_sym   = ca.MX.sym('t_f')
        tr_sym   = ca.MX.sym('t_r')
        lmuy     = ca.MX.sym('lmuy')
        lmux     = ca.MX.sym('lmux')
        cd_sym   = ca.MX.sym('Cd')

        # Road surface 2D elevations and vertical velocities per corner
        z_road_fl = ca.MX.sym('z_road_fl')
        z_road_fr = ca.MX.sym('z_road_fr')
        z_road_rl = ca.MX.sym('z_road_rl')
        z_road_rr = ca.MX.sym('z_road_rr')

        dz_road_fl = ca.MX.sym('dz_road_fl')
        dz_road_fr = ca.MX.sym('dz_road_fr')
        dz_road_rl = ca.MX.sym('dz_road_rl')
        dz_road_rr = ca.MX.sym('dz_road_rr')
        nx_fl, ny_fl, nz_fl = ca.MX.sym('nx_fl'), ca.MX.sym('ny_fl'), ca.MX.sym('nz_fl')
        nx_fr, ny_fr, nz_fr = ca.MX.sym('nx_fr'), ca.MX.sym('ny_fr'), ca.MX.sym('nz_fr')
        nx_rl, ny_rl, nz_rl = ca.MX.sym('nx_rl'), ca.MX.sym('ny_rl'), ca.MX.sym('nz_rl')
        nx_rr, ny_rr, nz_rr = ca.MX.sym('nx_rr'), ca.MX.sym('ny_rr'), ca.MX.sym('nz_rr')

        p = ca.vertcat(wd_f, lltd, K_phi_f, K_phi_r, K_z_f, K_z_r, aero_bal,
                       cam_f, cam_r, toe_f, toe_r, L_sym, tf_sym, tr_sym, lmuy, lmux, cd_sym,
                       z_road_fl, z_road_fr, z_road_rl, z_road_rr,
                       dz_road_fl, dz_road_fr, dz_road_rl, dz_road_rr,
                       nx_fl, ny_fl, nz_fl, nx_fr, ny_fr, nz_fr, nx_rl, ny_rl, nz_rl, nx_rr, ny_rr, nz_rr)

        # Geometry & Inertia
        vp = self.vp
        m  = vp.m
        Iz = vp.Iz
        Ixz = vp.Ixz  # Cross-product of inertia [kg m^2]
        Ix = vp.Ix    # Roll moment of inertia [kg m^2]  — from VehicleParams
        Iy = vp.Iy    # Pitch moment of inertia [kg m^2] — from VehicleParams
        Iw = vp.Iw
        Rw = vp.R_w
        a  = L_sym * (1.0 - wd_f)
        b  = L_sym * wd_f

        # Unsprung masses per corner [kg] (hub + upright + disc; see VehicleParams m_us_f/r)
        m_us_f = float(getattr(vp, 'm_us_f', 18.0))  # Front unsprung mass per corner [kg]
        m_us_r = float(getattr(vp, 'm_us_r', 16.0))  # Rear unsprung mass per corner [kg]
        m_s    = m - 2.0 * m_us_f - 2.0 * m_us_r     # Sprung mass [kg]

        # Radial tire contact stiffness and pressure adaptation
        # Scaling based on nominal 12 PSI (0.82737 bar) reference
        K_tz_12psi = float(getattr(vp, 'tire_vertical_stiffness_12psi', 75831.641))
        p_f_bar = float(getattr(vp, 'tire_pressure_f_bar', 0.82737))
        p_r_bar = float(getattr(vp, 'tire_pressure_r_bar', 0.82737))
        
        K_tz_f = K_tz_12psi * (p_f_bar / 0.82737)
        K_tz_r = K_tz_12psi * (p_r_bar / 0.82737)
        C_tz   = float(getattr(vp, 'C_tire_z', 500.0))

        # Kinematic corrections
        deg2rad = np.deg2rad(1.0)

        # Steering Wheel / Pinion angle [deg]
        gr_steer = getattr(vp, 'steering_ratio', 5.0)
        delta_sw_deg = (delta / deg2rad) * gr_steer

        # Differential Steering Kinematics (Adams Car direct curve or Ackermann blend)
        w_steer_coeffs = getattr(vp, 'wheel_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
        if any(abs(c) > 1e-6 for c in w_steer_coeffs[:3]):
            delta_fl_deg = _poly3_ca(w_steer_coeffs, delta_sw_deg)
            delta_fr_deg = -_poly3_ca(w_steer_coeffs, -delta_sw_deg)
            delta_fl = delta_fl_deg * deg2rad
            delta_fr = delta_fr_deg * deg2rad
        else:
            ack_pct = vp.ackermann_pct
            tan_d = ca.tan(delta)
            delta_ack_fl = ca.arctan2(L_sym * tan_d, L_sym - (tf_sym / 2.0) * tan_d)
            delta_ack_fr = ca.arctan2(L_sym * tan_d, L_sym + (tf_sym / 2.0) * tan_d)
            delta_fl = delta + ack_pct * (delta_ack_fl - delta)
            delta_fr = delta + ack_pct * (delta_ack_fr - delta)

        # Camber Steer delta from steering angle [deg]
        camber_steer_coeffs = getattr(vp, 'camber_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
        delta_sw_fl_deg = (delta_fl / deg2rad) * gr_steer
        delta_sw_fr_deg = (delta_fr / deg2rad) * gr_steer
        dcam_steer_fl_deg = _poly3_ca(camber_steer_coeffs, delta_sw_fl_deg)
        dcam_steer_fr_deg = _poly3_ca(camber_steer_coeffs, -delta_sw_fr_deg)

        # Check calibrated TYRESIDE from .tir property files
        is_right_f = getattr(self.tp_f, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_f else True
        is_right_r = getattr(self.tp_r, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_r else True

        # Aero downforce & drag (Center of Pressure CoP verification):
        # 1. Longitudinal axis (X): CoP location is strictly governed by aero_bal:
        #    Fza_f = F_down * aero_bal, Fza_r = F_down * (1.0 - aero_bal)
        # 2. Vertical axis (Z): Drag aerodynamic application height matches CoG height (z_cop = h_cg),
        #    nullifying fictitious parasite pitch moments from forward resistance:
        #    M_pitch_drag = F_drag * (z_cop - h_cg) = 0.
        # 3. Sprung mass pitch ODE: Downforce pitch balance is already completely accounted for in Fza_f
        #    and Fza_r on suspensions; thus no double counting between aerodynamic pitch moment and Fza.
        rho_val = float(getattr(vp, 'rho', 1.225))
        S_val   = float(getattr(vp, 'S', 1.139))
        Cz_val  = float(getattr(vp, 'Cz', 3.421))
        crr_val = float(getattr(vp, 'crr', 0.015))

        F_down = 0.5 * rho_val * S_val * Cz_val * (v_x**2)
        # Vectorial 2D Planar Aerodynamic Drag (opposes total chassis velocity vector)
        V_planar = ca.sqrt(v_x**2 + v_y**2 + 1e-6)
        F_drag_x = 0.5 * rho_val * S_val * cd_sym * V_planar * v_x
        F_drag_y = 0.5 * rho_val * S_val * cd_sym * V_planar * v_y

        # Rolling resistance force (smooth tanh activation to avoid singularity at standstill)
        F_rr = m * vp.g * crr_val * ca.tanh(2.0 * v_x)

        # ------------------------------------------------------------------
        # Chassis Corner Kinematics & Relative Suspension Jounce
        # ------------------------------------------------------------------
        # Road Mean Plane Kinematics & Relative Chassis Attitude
        # ------------------------------------------------------------------
        L_wb_ca = L_sym
        t_avg_ca = (tf_sym + tr_sym) / 2.0
        z_road_f = (z_road_fl + z_road_fr) / 2.0
        z_road_r = (z_road_rl + z_road_rr) / 2.0
        z_mid = (z_road_f + z_road_r) / 2.0

        # In ISO frame: pitch > 0 nose down. When front road is higher, slope is uphill (nose up -> theta_road < 0)
        theta_road = - (z_road_f - z_road_r) / L_wb_ca
        # Roll > 0 right down. When left road is higher, bank is right down -> phi_road > 0
        phi_road = ((z_road_fl + z_road_rl) - (z_road_fr + z_road_rr)) / (2.0 * t_avg_ca)

        theta_rel = theta - theta_road
        phi_rel = phi - phi_road

        # ------------------------------------------------------------------
        # Chassis Corner Kinematics & Relative Suspension Jounce
        # Attachment point vertical coordinates and velocities in ISO frame:
        # ------------------------------------------------------------------
        K_heave = 2.0 * K_z_f + 2.0 * K_z_r
        z_s = - F_down / K_heave

        z_chassis_fl = z_s - theta_rel * a + phi_rel * (tf_sym / 2.0)
        z_chassis_fr = z_s - theta_rel * a - phi_rel * (tf_sym / 2.0)
        z_chassis_rl = z_s + theta_rel * b + phi_rel * (tr_sym / 2.0)
        z_chassis_rr = z_s + theta_rel * b - phi_rel * (tr_sym / 2.0)

        cos_theta = ca.cos(theta_rel)
        cos_phi = ca.cos(phi_rel)
        
        z_jounce_fl = (z_fl - z_chassis_fl) / (cos_theta * cos_phi)
        z_jounce_fr = (z_fr - z_chassis_fr) / (cos_theta * cos_phi)
        z_jounce_rl = (z_rl - z_chassis_rl) / (cos_theta * cos_phi)
        z_jounce_rr = (z_rr - z_chassis_rr) / (cos_theta * cos_phi)

        z_dot_chassis_fl = w_heave - theta_dot * a + phi_dot * (tf_sym / 2.0)
        z_dot_chassis_fr = w_heave - theta_dot * a - phi_dot * (tf_sym / 2.0)
        z_dot_chassis_rl = w_heave + theta_dot * b + phi_dot * (tr_sym / 2.0)
        z_dot_chassis_rr = w_heave + theta_dot * b - phi_dot * (tr_sym / 2.0)

        v_jounce_fl = (vz_fl - z_dot_chassis_fl) / (cos_theta * cos_phi)
        v_jounce_fr = (vz_fr - z_dot_chassis_fr) / (cos_theta * cos_phi)
        v_jounce_rl = (vz_rl - z_dot_chassis_rl) / (cos_theta * cos_phi)
        v_jounce_rr = (vz_rr - z_dot_chassis_rr) / (cos_theta * cos_phi)

        cam_fl_expr = _poly3_ca(vp.camber_gain_f_coeffs, z_jounce_fl) + cam_f + dcam_steer_fl_deg
        cam_fr_expr = _poly3_ca(vp.camber_gain_f_coeffs, z_jounce_fr) + cam_f + dcam_steer_fr_deg
        cam_rl_expr = _poly3_ca(vp.camber_gain_r_coeffs, z_jounce_rl) + cam_r
        cam_rr_expr = _poly3_ca(vp.camber_gain_r_coeffs, z_jounce_rr) + cam_r

        if is_right_f:
            gamma_fl = -cam_fl_expr * deg2rad
            gamma_fr =  cam_fr_expr * deg2rad
        else:
            gamma_fl =  cam_fl_expr * deg2rad
            gamma_fr = -cam_fr_expr * deg2rad

        if is_right_r:
            gamma_rl = -cam_rl_expr * deg2rad
            gamma_rr =  cam_rr_expr * deg2rad
        else:
            gamma_rl =  cam_rl_expr * deg2rad
            gamma_rr = -cam_rr_expr * deg2rad

        lvy_fl = -1.0 if is_right_f else 1.0
        lvy_fr =  1.0 if is_right_f else -1.0
        lvy_rl = -1.0 if is_right_r else 1.0
        lvy_rr =  1.0 if is_right_r else -1.0

        lhy_fl = -1.0 if is_right_f else 1.0
        lhy_fr =  1.0 if is_right_f else -1.0
        lhy_rl = -1.0 if is_right_r else 1.0
        lhy_rr =  1.0 if is_right_r else -1.0

        toe_fl = (_poly3_ca(vp.toe_gain_f_coeffs, z_jounce_fl) + toe_f) * deg2rad
        toe_fr = (_poly3_ca(vp.toe_gain_f_coeffs, z_jounce_fr) + toe_f) * deg2rad
        toe_rl = (_poly3_ca(vp.toe_gain_r_coeffs, z_jounce_rl) + toe_r) * deg2rad
        toe_rr = (_poly3_ca(vp.toe_gain_r_coeffs, z_jounce_rr) + toe_r) * deg2rad

        phi_deg = phi * (180.0 / ca.pi)
        h_rc_f = _poly3_ca(vp.rollcenter_f_coeffs, phi_deg)
        h_rc_r = _poly3_ca(vp.rollcenter_r_coeffs, phi_deg)
        h_rc_avg = (h_rc_f + h_rc_r) / 2.0

        # ------------------------------------------------------------------
        # Static 1g Reference Vertical Loads (corner weights at z_i = 0, vz_i = 0)
        # Reference: z_i = 0 at static 1g ground equilibrium (SSOT convention)
        # ------------------------------------------------------------------
        Fz_0_f  = m * vp.g * (b / L_sym)
        Fz_0_r  = m * vp.g * (a / L_sym)
        Fz_0_fl = Fz_0_f / 2.0
        Fz_0_fr = Fz_0_f / 2.0
        Fz_0_rl = Fz_0_r / 2.0
        Fz_0_rr = Fz_0_r / 2.0

        # ------------------------------------------------------------------
        # Vertical Tire Contact Force — Emergent from Radial Tire Spring + Damping
        # Strict ground contact law: delta_z = delta_0 + z_road - zw (isolated from chassis roll)
        # ------------------------------------------------------------------
        defl_0_fl = Fz_0_fl / K_tz_f
        defl_0_fr = Fz_0_fr / K_tz_f
        defl_0_rl = Fz_0_rl / K_tz_r
        defl_0_rr = Fz_0_rr / K_tz_r

        # Road plane elevations per corner relative to center
        z_plane_fl = z_mid - theta_road * (L_wb_ca / 2.0) + phi_road * (tf_sym / 2.0)
        z_plane_fr = z_mid - theta_road * (L_wb_ca / 2.0) - phi_road * (tf_sym / 2.0)
        z_plane_rl = z_mid + theta_road * (L_wb_ca / 2.0) + phi_road * (tr_sym / 2.0)
        z_plane_rr = z_mid + theta_road * (L_wb_ca / 2.0) - phi_road * (tr_sym / 2.0)

        # Micro-roughness / bump deviations from planar road slope
        dz_micro_fl = z_road_fl - z_plane_fl
        dz_micro_fr = z_road_fr - z_plane_fr
        dz_micro_rl = z_road_rl - z_plane_rl
        dz_micro_rr = z_road_rr - z_plane_rr

        defl_fl = defl_0_fl - z_fl + dz_micro_fl
        defl_fr = defl_0_fr - z_fr + dz_micro_fr
        defl_rl = defl_0_rl - z_rl + dz_micro_rl
        defl_rr = defl_0_rr - z_rr + dz_micro_rr

        Fz_max = m * vp.g * 1.50

        # Raw contact force when contacting ground (zero tension floor)
        # Includes dynamic road damping: C_tz * (dz_road_i - vz_i)
        Fz_fl_raw = ca.fmax(0.0, K_tz_f * defl_fl + C_tz * (dz_road_fl - vz_fl))
        Fz_fr_raw = ca.fmax(0.0, K_tz_f * defl_fr + C_tz * (dz_road_fr - vz_fr))
        Fz_rl_raw = ca.fmax(0.0, K_tz_r * defl_rl + C_tz * (dz_road_rl - vz_rl))
        Fz_rr_raw = ca.fmax(0.0, K_tz_r * defl_rr + C_tz * (dz_road_rr - vz_rr))

        Fz_fl = ca.fmin(Fz_fl_raw, Fz_max)
        Fz_fr = ca.fmin(Fz_fr_raw, Fz_max)
        Fz_rl = ca.fmin(Fz_rl_raw, Fz_max)
        Fz_rr = ca.fmin(Fz_rr_raw, Fz_max)

        # ------------------------------------------------------------------
        # Suspension Kinematics: Caster & KPI Jacking (PRESERVED UNCHANGED)
        # Applied as small perturbations on top of the emergent spring Fz.
        # ------------------------------------------------------------------
        nu_caster = vp.caster_f * deg2rad
        sigma_kpi = vp.kpi_f   * deg2rad
        d_spindle = vp.spindle_offset_f
        r_scrub   = vp.scrub_radius_f

        # 1. Caster Jacking: Anti-symmetric diagonal corner load transfer
        delta_avg = (delta_fl + delta_fr) / 2.0
        dz_jack_diff = -d_spindle * ca.sin(nu_caster) * ca.sin(delta_avg)
        dFz_jack_diff = K_z_f * dz_jack_diff
        diag_split = (K_z_r * tr_sym) / ca.fmax(K_z_f * tf_sym + K_z_r * tr_sym, 1.0)

        # 2. KPI Jacking: Symmetric heave lift of front axle (even function of steer)
        dz_jack_heave = r_scrub * ca.sin(sigma_kpi) * (1.0 - ca.cos(delta_avg))
        dFz_jack_heave = K_z_f * dz_jack_heave

        Ay = v_x * r
        K_phi_tot = K_phi_f + K_phi_r

        Fz_fl = ca.fmax(ca.fmin(Fz_fl + (dFz_jack_diff + dFz_jack_heave) * ca.tanh(Fz_fl / 100.0), Fz_max), 0.0)
        Fz_fr = ca.fmax(ca.fmin(Fz_fr + (-dFz_jack_diff + dFz_jack_heave) * ca.tanh(Fz_fr / 100.0), Fz_max), 0.0)
        Fz_rl = ca.fmax(ca.fmin(Fz_rl - dFz_jack_diff * diag_split * ca.tanh(Fz_rl / 100.0), Fz_max), 0.0)
        Fz_rr = ca.fmax(ca.fmin(Fz_rr + dFz_jack_diff * diag_split * ca.tanh(Fz_rr / 100.0), Fz_max), 0.0)

        # Velocities at wheel hubs
        Vx_fl = v_x - r * tf_sym / 2.0
        Vy_fl = v_y + r * a
        Vx_fr = v_x + r * tf_sym / 2.0
        Vy_fr = v_y + r * a
        Vx_rl = v_x - r * tr_sym / 2.0

        Vy_rl = v_y - r * b
        Vx_rr = v_x + r * tr_sym / 2.0
        Vy_rr = v_y - r * b

        Vxw_fl =  Vx_fl * ca.cos(delta_fl) + Vy_fl * ca.sin(delta_fl)
        Vyw_fl = -Vx_fl * ca.sin(delta_fl) + Vy_fl * ca.cos(delta_fl)
        Vxw_fr =  Vx_fr * ca.cos(delta_fr) + Vy_fr * ca.sin(delta_fr)
        Vyw_fr = -Vx_fr * ca.sin(delta_fr) + Vy_fr * ca.cos(delta_fr)
        Vxw_rl =  Vx_rl
        Vyw_rl =  Vy_rl
        Vxw_rr =  Vx_rr
        Vyw_rr =  Vy_rr

        # Tire Relaxation Lengths (Contact Patch Transient Model)
        sigma_x = float(getattr(vp, 'sigma_x', 0.15))
        sigma_y = float(getattr(vp, 'sigma_y', 0.22))

        # Real slip velocities at wheel hubs in wheel frame:
        # Vsx = Vxw - w * Rw,  Vsy = Vyw
        Vsx_fl = Vxw_fl - w_fl * Rw
        Vsy_fl = Vyw_fl
        Vsx_fr = Vxw_fr - w_fr * Rw
        Vsy_fr = Vyw_fr
        Vsx_rl = Vxw_rl - w_rl * Rw
        Vsy_rl = Vyw_rl
        Vsx_rr = Vxw_rr - w_rr * Rw
        Vsy_rr = Vyw_rr

        # Total contact patch transport velocity magnitude (incorporates wheel spin speed |w*Rw|
        # to ensure correct relaxation damping rate during high-torque wheelspin transients)
        eps_v2 = 1e-4
        eps_v = 0.8  # [m/s] low-speed floor against standstill singularities
        V_hub_fl = ca.sqrt(Vxw_fl**2 + Vyw_fl**2 + eps_v2)
        V_hub_fr = ca.sqrt(Vxw_fr**2 + Vyw_fr**2 + eps_v2)
        V_hub_rl = ca.sqrt(Vxw_rl**2 + Vyw_rl**2 + eps_v2)
        V_hub_rr = ca.sqrt(Vxw_rr**2 + Vyw_rr**2 + eps_v2)

        V_spin_fl = ca.fabs(w_fl * Rw)
        V_spin_fr = ca.fabs(w_fr * Rw)
        V_spin_rl = ca.fabs(w_rl * Rw)
        V_spin_rr = ca.fabs(w_rr * Rw)

        Vmag_fl_eff = ca.fmax(ca.fmax(V_hub_fl, V_spin_fl), eps_v)
        Vmag_fr_eff = ca.fmax(ca.fmax(V_hub_fr, V_spin_fr), eps_v)
        Vmag_rl_eff = ca.fmax(ca.fmax(V_hub_rl, V_spin_rl), eps_v)
        Vmag_rr_eff = ca.fmax(ca.fmax(V_hub_rr, V_spin_rr), eps_v)

        # Contact patch elastic deflection time derivatives (dot(u) = -Vs - (Vmag_eff / sigma) * u)
        # Governed by total road contact speed Vmag_eff, computed on the free unconstrained state (eliminates integrator windup)
        ux_fl_dot = -Vsx_fl - (Vmag_fl_eff / sigma_x) * ux_fl
        uy_fl_dot = -Vsy_fl - (Vmag_fl_eff / sigma_y) * uy_fl
        ux_fr_dot = -Vsx_fr - (Vmag_fr_eff / sigma_x) * ux_fr
        uy_fr_dot = -Vsy_fr - (Vmag_fr_eff / sigma_y) * uy_fr
        ux_rl_dot = -Vsx_rl - (Vmag_rl_eff / sigma_x) * ux_rl
        uy_rl_dot = -Vsy_rl - (Vmag_rl_eff / sigma_y) * uy_rl
        ux_rr_dot = -Vsx_rr - (Vmag_rr_eff / sigma_x) * ux_rr
        uy_rr_dot = -Vsy_rr - (Vmag_rr_eff / sigma_y) * uy_rr

        # Relaxed dynamic slips to pass to Magic Formula MF6.1:
        # Smooth ca.tanh saturation applied strictly to the output slip ratios (Coulomb sliding limit ~1.5)
        # Preserves the linear slope at small deflections while bounding extreme values without state windup
        kappa_fl = 1.5 * ca.tanh(ux_fl / (1.5 * sigma_x))
        kappa_fr = 1.5 * ca.tanh(ux_fr / (1.5 * sigma_x))
        kappa_rl = 1.5 * ca.tanh(ux_rl / (1.5 * sigma_x))
        kappa_rr = 1.5 * ca.tanh(ux_rr / (1.5 * sigma_x))

        alpha_fl = ca.arctan(1.5 * ca.tanh(uy_fl / (1.5 * sigma_y))) + toe_fl
        alpha_fr = ca.arctan(1.5 * ca.tanh(uy_fr / (1.5 * sigma_y))) - toe_fr
        alpha_rl = ca.arctan(1.5 * ca.tanh(uy_rl / (1.5 * sigma_y))) - toe_rl
        alpha_rr = ca.arctan(1.5 * ca.tanh(uy_rr / (1.5 * sigma_y))) + toe_rr

        # Tire inflation pressure [Pa]
        p_press_f = float(getattr(vp, 'tire_pressure_f_bar', 0.85)) * 1e5
        p_press_r = float(getattr(vp, 'tire_pressure_r_bar', 0.85)) * 1e5

        # Pure forces (with TYRESIDE-consistent ply-steer and camber)
        Fx_fl_p = mf61_longitudinal_force(kappa_fl, Fz_fl, gamma_fl, self.tp_f, LMUX=lmux, pressure=p_press_f)
        Fy_fl_p = -mf61_lateral_force(alpha_fl, Fz_fl, gamma_fl, self.tp_f, LMUY=lmuy, LVY=lvy_fl, LHY=lhy_fl)
        Fx_fr_p = mf61_longitudinal_force(kappa_fr, Fz_fr, gamma_fr, self.tp_f, LMUX=lmux, pressure=p_press_f)
        Fy_fr_p = -mf61_lateral_force(alpha_fr, Fz_fr, gamma_fr, self.tp_f, LMUY=lmuy, LVY=lvy_fr, LHY=lhy_fr)

        Fx_rl_p = mf61_longitudinal_force(kappa_rl, Fz_rl, gamma_rl, self.tp_r, LMUX=lmux, pressure=p_press_r)
        Fy_rl_p = -mf61_lateral_force(alpha_rl, Fz_rl, gamma_rl, self.tp_r, LMUY=lmuy, LVY=lvy_rl, LHY=lhy_rl)
        Fx_rr_p = mf61_longitudinal_force(kappa_rr, Fz_rr, gamma_rr, self.tp_r, LMUX=lmux, pressure=p_press_r)
        Fy_rr_p = -mf61_lateral_force(alpha_rr, Fz_rr, gamma_rr, self.tp_r, LMUY=lmuy, LVY=lvy_rr, LHY=lhy_rr)

        # Combined slip forces (with TYRESIDE-consistent ply-steer and combined shift mirroring)
        Fxw_fl = mf61_combined_longitudinal_force(Fx_fl_p, kappa_fl, alpha_fl, Fz_fl, gamma_fl, self.tp_f, LHY=lhy_fl, pressure=p_press_f)
        Fyw_fl = mf61_combined_lateral_force(Fy_fl_p, alpha_fl, kappa_fl, Fz_fl, gamma_fl, self.tp_f, LVY=lvy_fl, LHY=lhy_fl)
        Fxw_fr = mf61_combined_longitudinal_force(Fx_fr_p, kappa_fr, alpha_fr, Fz_fr, gamma_fr, self.tp_f, LHY=lhy_fr, pressure=p_press_f)
        Fyw_fr = mf61_combined_lateral_force(Fy_fr_p, alpha_fr, kappa_fr, Fz_fr, gamma_fr, self.tp_f, LVY=lvy_fr, LHY=lhy_fr)

        Fxw_rl = mf61_combined_longitudinal_force(Fx_rl_p, kappa_rl, alpha_rl, Fz_rl, gamma_rl, self.tp_r, LHY=lhy_rl, pressure=p_press_r)
        Fyw_rl = mf61_combined_lateral_force(Fy_rl_p, alpha_rl, kappa_rl, Fz_rl, gamma_rl, self.tp_r, LVY=lvy_rl, LHY=lhy_rl)
        Fxw_rr = mf61_combined_longitudinal_force(Fx_rr_p, kappa_rr, alpha_rr, Fz_rr, gamma_rr, self.tp_r, LHY=lhy_rr, pressure=p_press_r)
        Fyw_rr = mf61_combined_lateral_force(Fy_rr_p, alpha_rr, kappa_rr, Fz_rr, gamma_rr, self.tp_r, LVY=lvy_rr, LHY=lhy_rr)

        # Rotate front forces into chassis frame
        Fx_fl =  Fxw_fl * ca.cos(delta_fl) - Fyw_fl * ca.sin(delta_fl)
        Fy_fl =  Fxw_fl * ca.sin(delta_fl) + Fyw_fl * ca.cos(delta_fl)
        Fx_fr =  Fxw_fr * ca.cos(delta_fr) - Fyw_fr * ca.sin(delta_fr)
        Fy_fr =  Fxw_fr * ca.sin(delta_fr) + Fyw_fr * ca.cos(delta_fr)

        Fx_rl = Fxw_rl
        Fy_rl = Fyw_rl
        Fx_rr = Fxw_rr
        Fy_rr = Fyw_rr

        Sum_Fx = Fx_fl + Fx_fr + Fx_rl + Fx_rr - F_drag_x - F_rr
        Sum_Fy = Fy_fl + Fy_fr + Fy_rl + Fy_rr - F_drag_y

        Mz = (a * (Fy_fl + Fy_fr) - b * (Fy_rl + Fy_rr) +
              (tf_sym / 2.0) * (Fx_fr - Fx_fl) +
              (tr_sym / 2.0) * (Fx_rr - Fx_rl))

        # Chassis planar ODEs
        v_x_dot = (Sum_Fx / m) + r * v_y - vp.g * ca.sin(theta)
        v_y_dot = (Sum_Fy / m) - r * v_x + vp.g * ca.sin(phi) * ca.cos(theta)
        # r_dot is coupled with roll dynamics and will be computed below

        # ------------------------------------------------------------------
        # Dynamic CG height & Roll/Pitch Geometry (PRESERVED UNCHANGED)
        # ------------------------------------------------------------------
        z_f_avg = (z_fl + z_fr) / 2.0
        z_r_avg = (z_rl + z_rr) / 2.0
        z_cg_susp = (b / L_sym) * z_f_avg + (a / L_sym) * z_r_avg
        h_cg_dyn = ca.fmax(0.05, vp.h_cg - z_cg_susp)

        phi_deg = phi * (180.0 / ca.pi)
        h_rc_f = _poly3_ca(vp.rollcenter_f_coeffs, phi_deg)
        h_rc_r = _poly3_ca(vp.rollcenter_r_coeffs, phi_deg)
        h_rc_avg = (h_rc_f + h_rc_r) / 2.0
        y_rc_f = _poly3_ca(getattr(vp, 'rollcenter_lat_f_coeffs', [0.0, 0.0, 0.06453, 0.0]), phi_deg)
        y_rc_r = _poly3_ca(getattr(vp, 'rollcenter_lat_r_coeffs', [0.0, 0.0, 0.04213, 0.0]), phi_deg)
        y_rc_avg = (y_rc_f + y_rc_r) / 2.0

        # ------------------------------------------------------------------
        # Anti-Features (Kinematic Anti-Dive / Anti-Squat / Anti-Lift)
        # ------------------------------------------------------------------
        AD_f = ca.fmax(0.0, ca.fmin(_poly3_ca(getattr(vp, 'anti_dive_coeffs',  [0.0, 0.0, 0.0,  0.30]), z_f_avg), 0.80))
        AS_r = ca.fmax(-0.35, ca.fmin(_poly3_ca(getattr(vp, 'anti_squat_coeffs', [0.0, 0.0, 0.0, -0.146]), z_r_avg), 0.50))
        AL_r = ca.fmax(-0.20, ca.fmin(_poly3_ca(getattr(vp, 'anti_lift_coeffs',  [0.0, 0.0, 0.0,  0.098]), z_r_avg), 0.50))

        # ------------------------------------------------------------------
        # Suspension Spring Forces (PRESERVED: bumpstop _calc_k_eff + progressive pack)
        # Applied here to compute F_spring_i = K_eff_i * z_i for hub + chassis ODEs.
        # TTX/linear stroke limits unchanged; vz_i replaces modal velocity approximation.
        # ------------------------------------------------------------------
        s_shock  = float(getattr(vp, 'damper_stroke_total_mm', 57.0)) / 1000.0
        mr_f = ca.fmax(float(getattr(vp, 'motion_ratio_f', 0.8)), 0.01)
        mr_r = ca.fmax(float(getattr(vp, 'motion_ratio_r', 0.8)), 0.01)

        z_sag_f  = ca.fabs(Fz_0_fl) / ca.fmax(K_z_f, 1e3)
        z_sag_r  = ca.fabs(Fz_0_rl) / ca.fmax(K_z_r, 1e3)
        gap_f    = ca.fmax(0.002, (s_shock - 0.005) / mr_f - z_sag_f)
        gap_r    = ca.fmax(0.002, (s_shock - 0.005) / mr_r - z_sag_r)
        reb_gap_f = ca.fmax(0.002, z_sag_f)
        reb_gap_r = ca.fmax(0.002, z_sag_r)
        k_bs_f   = float(getattr(vp, 'bumpstop_k_f', 80000.0))
        k_bs_r   = float(getattr(vp, 'bumpstop_k_r', 80000.0))
        dz_pack_f = 0.0015 / mr_f
        dz_pack_r = 0.0015 / mr_r

        # Smooth progressive CasADi bumpstop with solid pack & droop top-out (Fixed Energy Pumping)
        def _calc_F_spring(z_w, k_z, gap_b, gap_d, k_bs, dz_pack):
            F_lin = k_z * z_w
            act_b = 0.5 * (1.0 + ca.tanh((z_w - gap_b) * 200.0))
            dz_b  = ca.fmax(0.0, z_w - gap_b)
            F_b   = k_bs * act_b * dz_b * (1.0 + 15.0 * dz_b)
            # Impaccamento solido a 3-4mm residui (dz_b >= dz_pack)
            act_pack  = 0.5 * (1.0 + ca.tanh((dz_b - dz_pack) * 500.0))
            dz_pack_pen = ca.fmax(0.0, dz_b - dz_pack)
            F_pack    = 2.0e6 * act_pack * dz_pack_pen * (1.0 + 250.0 * dz_pack_pen)
            # Rebound stop / droop limit a finecorsa estensione
            act_d = 0.5 * (1.0 + ca.tanh((-z_w - gap_d) * 200.0))
            dz_d  = ca.fmax(0.0, -z_w - gap_d)
            F_d   = - k_bs * act_d * dz_d * (1.0 + 25.0 * dz_d)
            return F_lin + F_b + F_pack + F_d

        # Static suspension preloads (supporting sprung mass weight at 1g equilibrium)
        F_susp_0_fl = Fz_0_fl - m_us_f * vp.g
        F_susp_0_fr = Fz_0_fr - m_us_f * vp.g
        F_susp_0_rl = Fz_0_rl - m_us_r * vp.g
        F_susp_0_rr = Fz_0_rr - m_us_r * vp.g

        # Physical spring forces with installed preload and bumpstops
        F_spring_fl = F_susp_0_fl + _calc_F_spring(z_jounce_fl, K_z_f, gap_f, reb_gap_f, k_bs_f, dz_pack_f)
        F_spring_fr = F_susp_0_fr + _calc_F_spring(z_jounce_fr, K_z_f, gap_f, reb_gap_f, k_bs_f, dz_pack_f)
        F_spring_rl = F_susp_0_rl + _calc_F_spring(z_jounce_rl, K_z_r, gap_r, reb_gap_r, k_bs_r, dz_pack_r)
        F_spring_rr = F_susp_0_rr + _calc_F_spring(z_jounce_rr, K_z_r, gap_r, reb_gap_r, k_bs_r, dz_pack_r)

        # Anti-roll bar forces (strictly anti-symmetric, zero net heave force)
        # Direct lumped roll stiffness K_phi_arb [Nm/rad] and differential wheel rate Kw_arb = K_phi_arb / t^2 [N/m]
        # Restoring roll moment: Mx_arb = -K_phi_arb * (phi - phi_axle)
        # Delta_z_susp = z_fl - z_fr: for roll > 0 (right bump, left droop), Delta_z_susp < 0:
        # F_arb_fl = +Kw_arb * Delta_z_susp < 0 (downwards on chassis)
        # F_arb_fr = -Kw_arb * Delta_z_susp > 0 (upwards on chassis)
        K_arb_f = ca.fmax(0.0, K_phi_f - K_z_f * (tf_sym**2) / 2.0)
        K_arb_r = ca.fmax(0.0, K_phi_r - K_z_r * (tr_sym**2) / 2.0)
        Kw_arb_f = K_arb_f / (tf_sym**2 + 1e-6)
        Kw_arb_r = K_arb_r / (tr_sym**2 + 1e-6)
        delta_z_susp_f = z_jounce_fl - z_jounce_fr
        delta_z_susp_r = z_jounce_rl - z_jounce_rr
        F_arb_fl = Kw_arb_f * delta_z_susp_f
        F_arb_fr = - Kw_arb_f * delta_z_susp_f
        F_arb_rl = Kw_arb_r * delta_z_susp_r
        F_arb_rr = - Kw_arb_r * delta_z_susp_r

        d_mode  = getattr(vp, 'damper_mode', 'linear')
        use_ttx = d_mode in ('ttx25_nonlinear', 'custom_dyno', 'Custom Dyno Curve')
        if use_ttx:
            valv_f = 'CUSTOM_DYNO' if d_mode in ('custom_dyno', 'Custom Dyno Curve') else getattr(vp, 'damper_valving_f', 'C12/R12')
            valv_r = 'CUSTOM_DYNO' if d_mode in ('custom_dyno', 'Custom Dyno Curve') else getattr(vp, 'damper_valving_r', 'C12/R12')
            _lsc_f = getattr(vp, 'damper_lsc_clicks_f', 10.0); _hsc_f = getattr(vp, 'damper_hsc_turns_f', 2.0)
            _lsr_f = getattr(vp, 'damper_lsr_clicks_f', 10.0); _hsr_f = getattr(vp, 'damper_hsr_turns_f', 2.0)
            _lsc_r = getattr(vp, 'damper_lsc_clicks_r', 10.0); _hsc_r = getattr(vp, 'damper_hsc_turns_r', 2.0)
            _lsr_r = getattr(vp, 'damper_lsr_clicks_r', 10.0); _hsr_r = getattr(vp, 'damper_hsr_turns_r', 2.0)
            F_damp_fl = calculate_wheel_damper_force(v_jounce_fl, getattr(vp,'motion_ratio_f',0.85), valv_f, _lsc_f, _hsc_f, _lsr_f, _hsr_f, use_casadi=True)
            F_damp_fr = calculate_wheel_damper_force(v_jounce_fr, getattr(vp,'motion_ratio_f',0.85), valv_f, _lsc_f, _hsc_f, _lsr_f, _hsr_f, use_casadi=True)
            F_damp_rl = calculate_wheel_damper_force(v_jounce_rl, getattr(vp,'motion_ratio_r',0.85), valv_r, _lsc_r, _hsc_r, _lsr_r, _hsr_r, use_casadi=True)
            F_damp_rr = calculate_wheel_damper_force(v_jounce_rr, getattr(vp,'motion_ratio_r',0.85), valv_r, _lsc_r, _hsc_r, _lsr_r, _hsr_r, use_casadi=True)
        else:
            # Constant linear damping (default / fallback)
            C_z_f_lin = float(getattr(vp, 'C_z_f', 1500.0))
            C_z_r_lin = float(getattr(vp, 'C_z_r', 1200.0))
            F_damp_fl = C_z_f_lin * v_jounce_fl
            F_damp_fr = C_z_f_lin * v_jounce_fr
            F_damp_rl = C_z_r_lin * v_jounce_rl
            F_damp_rr = C_z_r_lin * v_jounce_rr

        # Total suspension vertical force per corner (+ = UP on chassis, DOWN on hub)
        F_susp_fl = F_spring_fl + F_damp_fl + F_arb_fl
        F_susp_fr = F_spring_fr + F_damp_fr + F_arb_fr
        F_susp_rl = F_spring_rl + F_damp_rl + F_arb_rl
        F_susp_rr = F_spring_rr + F_damp_rr + F_arb_rr

        # ------------------------------------------------------------------
        # Hub (Unsprung Mass) ODEs — 2nd Order Action-Reaction Multibody
        # m_u,i * vz_dot_i = F_z,tire,i - F_susp,i - m_u,i * g
        # ------------------------------------------------------------------
        vz_fl_dot = (Fz_fl * nz_fl - F_susp_fl - m_us_f * vp.g) / m_us_f
        vz_fr_dot = (Fz_fr * nz_fr - F_susp_fr - m_us_f * vp.g) / m_us_f
        vz_rl_dot = (Fz_rl * nz_rl - F_susp_rl - m_us_r * vp.g) / m_us_r
        vz_rr_dot = (Fz_rr * nz_rr - F_susp_rr - m_us_r * vp.g) / m_us_r

        # Kinematic identity: wheel hub vertical displacement derivative = vz
        z_fl_dot = vz_fl
        z_fr_dot = vz_fr
        z_rl_dot = vz_rl
        z_rr_dot = vz_rr

        # ------------------------------------------------------------------
        # Chassis Heave ODE (Action-Reaction from 4 corners)
        # m_s * w_dot = Sum(F_susp,i) - F_down - m_s * g * cos(phi) * cos(theta) + m_s * (u * q - v * p)
        # ------------------------------------------------------------------
        m_s = m - 2.0 * m_us_f - 2.0 * m_us_r
        w_heave_dot = (F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr - F_down - m_s * vp.g * ca.cos(phi) * ca.cos(theta)) / m_s + (v_x * theta_dot - v_y * phi_dot)

        # ------------------------------------------------------------------
        # Roll ODE — True Action-Reaction Moment from Suspensions + ARBs
        # ------------------------------------------------------------------
        h_roll_arm = (h_cg_dyn - h_rc_avg) * ca.cos(phi) - y_rc_avg * ca.sin(phi)
        M_roll_destab = m * Ay * h_roll_arm + m * vp.g * (h_cg_dyn - h_rc_avg) * ca.sin(phi_rel)
        M_roll_susp = - ((tf_sym / 2.0) * (F_susp_fr - F_susp_fl) +
                         (tr_sym / 2.0) * (F_susp_rr - F_susp_rl))
        M_roll_net = M_roll_destab + M_roll_susp

        # ------------------------------------------------------------------
        # Pitch ODE — Elastic + Damping Restoration + Anti-Features
        # Physics: Iy * theta_ddot = -M_destab - K_theta*theta_rel - C_theta*theta_dot + M_grav
        #   M_destab = m*Ax*h_cg*anti_factor  (longitudinal inertial pitch-up/down)
        #   K_theta  = 2*K_z_f*a^2 + 2*K_z_r*b^2  (suspension pitch stiffness)
        #   C_theta  = 2*C_z_f*a^2 + 2*C_z_r*b^2  (suspension pitch damping)
        #   M_grav   = m*g*sin(theta)*h_cg   (static gravity moment on sloped roads)
        # Note: with theta>0 = nose-down convention, -K_theta*theta_rel restores to road slope theta_road.
        # ------------------------------------------------------------------
        Ax = Sum_Fx / m
        blend_acc = 0.5 * (1.0 + ca.tanh(Ax * 2.0))
        blend_brk = 1.0 - blend_acc
        bb = getattr(vp, 'brake_bias', 0.60)
        anti_factor_raw = blend_acc * (1.0 - AS_r) + blend_brk * (1.0 - (AD_f * bb + AL_r * (1.0 - bb)))
        anti_factor = ca.fmax(0.20, ca.fmin(anti_factor_raw, 1.40))
        h_pitch_arm = h_cg_dyn * ca.cos(theta)
        M_pitch_destab = m * Ax * h_pitch_arm * anti_factor
        # Pitch stiffness and damping from suspension (linearised about static equilibrium)
        K_theta = 2.0 * K_z_f * (a**2) + 2.0 * K_z_r * (b**2)
        C_z_f_val = float(getattr(vp, 'C_z_f', 1500.0))
        C_z_r_val = float(getattr(vp, 'C_z_r', 1200.0))
        C_theta = 2.0 * C_z_f_val * (a**2) + 2.0 * C_z_r_val * (b**2)
        # Net pitch moment: inertial destabilisation + elastic restoration to road plane + viscous restoration + gravity
        M_pitch_net = -M_pitch_destab - K_theta * theta_rel - C_theta * theta_dot + m * vp.g * ca.sin(theta) * h_cg_dyn
        theta_ddot = M_pitch_net / Iy + ((Iz - Ix) * phi_dot * r) / Iy

        # ------------------------------------------------------------------
        # Coupled Roll-Yaw Dynamics (Ixz cross-inertia)
        # ------------------------------------------------------------------
        I_det = Ix * Iz - Ixz**2
        phi_ddot = (Iz * M_roll_net + Ixz * Mz) / I_det + ((Iy - Iz) * theta_dot * r) / Ix
        r_dot    = (Ix  * Mz         + Ixz * M_roll_net) / I_det

        # Wheel spin ODEs with Brake Bias (PRESERVED UNCHANGED)
        Tb_f = (T_brake * vp.brake_bias) / 2.0
        Tb_r = (T_brake * (1.0 - vp.brake_bias)) / 2.0

        def calc_Tb(w, Tb_mag):
            return ca.tanh(w * 10.0) * Tb_mag

        drivetrain_type = getattr(vp, 'drivetrain_type', 'rwd')
        Iw_f = float(getattr(vp, 'Iw_f', 0.30 if drivetrain_type == 'rwd' else getattr(vp, 'Iw', 1.2)))
        Iw_r = float(getattr(vp, 'Iw_r', 0.30 if drivetrain_type == 'fwd' else getattr(vp, 'Iw', 1.2)))

        if drivetrain_type == 'fwd':
            w_fl_dot = (T_drive_rl - Fxw_fl * Rw - calc_Tb(w_fl, Tb_f)) / Iw_f
            w_fr_dot = (T_drive_rr - Fxw_fr * Rw - calc_Tb(w_fr, Tb_f)) / Iw_f
            w_rl_dot = (-Fxw_rl * Rw - calc_Tb(w_rl, Tb_r)) / Iw_r
            w_rr_dot = (-Fxw_rr * Rw - calc_Tb(w_rr, Tb_r)) / Iw_r
        elif drivetrain_type == 'awd':
            w_fl_dot = (0.5 * T_drive_rl - Fxw_fl * Rw - calc_Tb(w_fl, Tb_f)) / Iw_f
            w_fr_dot = (0.5 * T_drive_rr - Fxw_fr * Rw - calc_Tb(w_fr, Tb_f)) / Iw_f
            w_rl_dot = (0.5 * T_drive_rl - Fxw_rl * Rw - calc_Tb(w_rl, Tb_r)) / Iw_r
            w_rr_dot = (0.5 * T_drive_rr - Fxw_rr * Rw - calc_Tb(w_rr, Tb_r)) / Iw_r
        else:  # rwd (default)
            w_fl_dot = (-Fxw_fl * Rw - calc_Tb(w_fl, Tb_f)) / Iw_f
            w_fr_dot = (-Fxw_fr * Rw - calc_Tb(w_fr, Tb_f)) / Iw_f
            w_rl_dot = (T_drive_rl - Fxw_rl * Rw - calc_Tb(w_rl, Tb_r)) / Iw_r
            w_rr_dot = (T_drive_rr - Fxw_rr * Rw - calc_Tb(w_rr, Tb_r)) / Iw_r

        # ------------------------------------------------------------------
        # Assemble State Derivative Vector x_dot (28 states)
        # ------------------------------------------------------------------
        x_dot = ca.vertcat(
            v_x_dot, v_y_dot, r_dot,             # [0..2]  planar chassis
            phi_dot, phi_ddot,                    # [3..4]  roll
            theta_dot, theta_ddot,               # [5..6]  pitch
            w_heave_dot,                         # [7]     heave
            z_fl_dot, z_fr_dot, z_rl_dot, z_rr_dot,     # [8..11] suspension strokes
            vz_fl_dot, vz_fr_dot, vz_rl_dot, vz_rr_dot, # [12..15] suspension rates
            w_fl_dot, w_fr_dot, w_rl_dot, w_rr_dot,      # [16..19] wheel spin
            ux_fl_dot, uy_fl_dot,                # [20..21] carcassa FL
            ux_fr_dot, uy_fr_dot,                # [22..23] carcassa FR
            ux_rl_dot, uy_rl_dot,                # [24..25] carcassa RL
            ux_rr_dot, uy_rr_dot,                # [26..27] carcassa RR
        )

        ode_func = ca.Function('ode_14dof', [x, u, p], [x_dot], ['x', 'u', 'p'], ['x_dot'])
        return ODEFunctionWrapper(ode_func, default_cd=float(getattr(vp, 'Cd', 1.20)))


    @staticmethod
    def _find_c_compiler(output_dir=None):
        """
        Locates a suitable C compiler (MSVC 'cl' or GCC/Clang) on the system.
        Checks:
        1. PATH environment variable ('cl', 'gcc', 'clang')
        2. Local project tools (e.g. tools/mingw/bin/gcc.exe)
        3. Common MSYS2 / MinGW / UCRT installation paths
        4. Visual Studio Build Tools / IDE via vswhere
        Returns tuple: (compiler_type: 'msvc' | 'gcc', executable_path: str, extra_env: dict | None)
        """
        import os
        import shutil
        import subprocess
        import platform

        if platform.system().lower() != 'windows':
            gcc = shutil.which('gcc') or shutil.which('clang')
            return ('gcc', gcc, None) if gcc else (None, None, None)

        # 1. Check MSVC in PATH
        if shutil.which('cl'):
            return ('msvc', 'cl', None)

        # 2. Check GCC / Clang in PATH
        gcc_in_path = shutil.which('gcc') or shutil.which('clang')
        if gcc_in_path:
            return ('gcc', gcc_in_path, None)

        # 3. Candidate directories for MinGW / GCC
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, 'tools', 'mingw', 'bin', 'gcc.exe'),
            os.path.join(os.getcwd(), 'tools', 'mingw', 'bin', 'gcc.exe'),
            r'C:\msys64\ucrt64\bin\gcc.exe',
            r'C:\msys64\mingw64\bin\gcc.exe',
            r'C:\msys64\usr\bin\gcc.exe',
            r'C:\MinGW\bin\gcc.exe',
            r'C:\TDM-GCC-64\bin\gcc.exe',
            r'C:\winlibs\bin\gcc.exe',
            r'C:\w64devkit\bin\gcc.exe',
        ]
        if output_dir:
            candidates.insert(0, os.path.join(output_dir, 'tools', 'mingw', 'bin', 'gcc.exe'))

        for cand in candidates:
            if os.path.isfile(cand):
                comp_dir = os.path.dirname(cand)
                env = os.environ.copy()
                env['PATH'] = comp_dir + os.pathsep + env.get('PATH', '')
                return ('gcc', cand, env)

        # 4. Check Visual Studio via vswhere
        vswhere = r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
        if os.path.isfile(vswhere):
            try:
                res = subprocess.run(
                    [vswhere, '-latest', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                     '-find', r'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe'],
                    capture_output=True, text=True
                )
                cl_found = res.stdout.strip().splitlines()
                if cl_found and os.path.isfile(cl_found[0]):
                    return ('msvc', cl_found[0], None)
            except Exception:
                pass

        return (None, None, None)

    def get_model_signature(self) -> str:
        """
        Returns a deterministic MD5 signature uniquely identifying the physical vehicle
        and tire parameters baked as constants into the ODE graph.
        """
        vp = self.vp
        tp_f = self.tp_f
        tp_r = self.tp_r

        v_keys = [
            'v12_fixed_energy_pump',
            round(float(getattr(vp, 'm', 292.0)), 4),
            round(float(getattr(vp, 'm_us_f', 18.0)), 2),
            round(float(getattr(vp, 'm_us_r', 16.0)), 2),
            round(float(getattr(vp, 'C_tire_z', 500.0)), 1),
            round(float(getattr(vp, 'Iz', 120.0)), 4),
            round(float(getattr(vp, 'Ix', 35.0)), 4),
            round(float(getattr(vp, 'Iy', 110.0)), 4),
            round(float(getattr(vp, 'R_w', 0.203)), 5),
            round(float(getattr(vp, 'h_cg', 0.280)), 4),
            round(float(getattr(vp, 'camber_f_static', -1.5)), 4),
            round(float(getattr(vp, 'camber_r_static', -1.0)), 4),
            round(float(getattr(vp, 'toe_f_static', 0.0)), 4),
            round(float(getattr(vp, 'toe_r_static', 0.1)), 4),
            round(float(getattr(vp, 'tire_pressure_f_bar', 0.85)), 3),
            round(float(getattr(vp, 'tire_pressure_r_bar', 0.85)), 3),
            round(float(getattr(vp, 'Iw_f', 0.30)), 3),
            round(float(getattr(vp, 'Iw_r', 1.20)), 3),
            round(float(getattr(vp, 'anti_dive_coeffs', [0,0,0,0.3])[3]), 3),
            round(float(getattr(vp, 'anti_squat_coeffs', [0,0,0,-0.146])[3]), 3),
            round(float(getattr(vp, 'anti_lift_coeffs', [0,0,0,0.098])[3]), 3),
            str(getattr(vp, 'damper_mode', 'linear')),
            str(getattr(vp, 'vehicle_type', 'formula')),
            str(getattr(vp, 'tire_file', '')),
        ]
        t_keys = [
            str(getattr(tp_f, 'get_tyreside', lambda: 'RIGHT')()),
            str(getattr(tp_r, 'get_tyreside', lambda: 'RIGHT')()),
            round(float(tp_f.get('PDX1', 0.0)), 4),
            round(float(tp_f.get('PKX1', 0.0)), 4),
            round(float(tp_f.get('PDY1', 0.0)), 4),
            round(float(tp_f.get('PKY1', 0.0)), 4),
            round(float(tp_r.get('PDX1', 0.0)), 4),
            round(float(tp_r.get('PKX1', 0.0)), 4),
            round(float(tp_r.get('PDY1', 0.0)), 4),
            round(float(tp_r.get('PKY1', 0.0)), 4),
            round(float(tp_f.get('FZ0', 0.0)), 2),
            round(float(tp_r.get('FZ0', 0.0)), 2),
            round(float(tp_f.get('VERTICAL_STIFFNESS', 75831.641)), 1),
            round(float(tp_r.get('VERTICAL_STIFFNESS', 75831.641)), 1),
        ]
        sig_str = "_".join(map(str, v_keys + t_keys))
        return hashlib.md5(sig_str.encode('utf-8')).hexdigest()[:16]

    def compile_c_ode(self, output_dir=None):
        import subprocess
        import os
        import platform
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            
        sig = self.get_model_signature()
        system = platform.system().lower()
        dll_ext = '.dll' if system == 'windows' else '.so'
        
        c_filename = f'ode_14dof_{sig}.c'
        dll_filename = f'ode_14dof_{sig}{dll_ext}'
        c_filepath = os.path.join(output_dir, c_filename)
        dll_path = os.path.join(output_dir, dll_filename)
        
        ode_func = self.get_casadi_ode()
        raw_func = getattr(ode_func, '_raw_func', ode_func)
        
        # Generate C code
        cg = ca.CodeGenerator(c_filename)
        cg.add(raw_func)
        cg.generate(output_dir + os.sep)
        
        comp_type, comp_exe, comp_env = self._find_c_compiler(output_dir)
        if not comp_exe:
            print("[Vehicle14DOF] No C compiler (GCC/MSVC) found. Falling back to symbolic CasADi ODE.")
            return None

        try:
            if comp_type == 'msvc':
                cmd = [comp_exe, '/LD', '/O2', '/fp:fast', c_filepath, '/Fe' + dll_path]
                subprocess.run(cmd, env=comp_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for ext in ['.obj', '.lib', '.exp']:
                    tmp = os.path.join(output_dir, c_filename.replace('.c', ext))
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
            else:
                cmd = [comp_exe, '-shared', '-fPIC', '-O3', '-ffast-math', c_filepath, '-o', dll_path]
                subprocess.run(cmd, env=comp_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            if os.path.exists(c_filepath):
                try:
                    os.remove(c_filepath)
                except OSError:
                    pass

            sig_path = dll_path + '.sig'
            try:
                with open(sig_path, 'w', encoding='utf-8') as f:
                    f.write(sig)
            except Exception as e_sig:
                print(f"[Vehicle14DOF] Warning: could not write signature file: {e_sig}")

            # Also maintain ode_14dof_gen.dll if this is the nominal setup
            nom_sig = Vehicle14DOF(VehicleParams()).get_model_signature()
            if sig == nom_sig:
                gen_dll = os.path.join(output_dir, f'ode_14dof_gen{dll_ext}')
                gen_sig = gen_dll + '.sig'
                try:
                    import shutil
                    shutil.copy2(dll_path, gen_dll)
                    shutil.copy2(sig_path, gen_sig)
                except Exception:
                    pass

            return dll_path
        except Exception as e:
            print(f"[Vehicle14DOF] Failed to compile ODE with {comp_exe}: {e}")
            return None

    def load_compiled_ode(self, dll_path=None):
        import os
        import platform
        if dll_path is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            system = platform.system().lower()
            dll_name = 'ode_14dof_gen.dll' if system == 'windows' else 'ode_14dof_gen.so'
            dll_path = os.path.join(output_dir, dll_name)
            
        if not os.path.exists(dll_path):
            return None

        # Convert to safe 8.3 short path on Windows if path contains non-ASCII characters
        load_path = dll_path
        if platform.system().lower() == 'windows':
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(1024)
                GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
                res_len = GetShortPathName(os.path.abspath(dll_path), buf, 1024)
                if res_len > 0 and os.path.exists(buf.value):
                    load_path = buf.value
            except Exception:
                pass
            
        raw_external = None
        try:
            raw_external = ca.external('ode_14dof', load_path)
        except Exception:
            try:
                rel = os.path.relpath(dll_path)
                raw_external = ca.external('ode_14dof', rel)
            except Exception as e2:
                print(f"[Vehicle14DOF] Failed to load compiled ODE '{dll_path}': {e2}")
                return None
        if raw_external is not None:
            return ODEFunctionWrapper(raw_external, default_cd=float(getattr(self.vp, 'Cd', 1.20)))
        return None

    def get_fast_ode(self):
        import os
        import platform
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        system = platform.system().lower()
        dll_ext = '.dll' if system == 'windows' else '.so'
        current_sig = self.get_model_signature()

        # 1. Check signature-specific DLL first
        sig_dll_name = f'ode_14dof_{current_sig}{dll_ext}'
        sig_dll_path = os.path.join(output_dir, sig_dll_name)
        if os.path.exists(sig_dll_path):
            fast_ode = self.load_compiled_ode(sig_dll_path)
            if fast_ode is not None:
                return fast_ode

        # 2. Check standard ode_14dof_gen.dll with signature match
        gen_dll_name = f'ode_14dof_gen{dll_ext}'
        gen_dll_path = os.path.join(output_dir, gen_dll_name)
        gen_sig_path = gen_dll_path + '.sig'
        if os.path.exists(gen_dll_path) and os.path.exists(gen_sig_path):
            try:
                with open(gen_sig_path, 'r', encoding='utf-8') as f:
                    stored_sig = f.read().strip()
                if stored_sig == current_sig:
                    fast_ode = self.load_compiled_ode(gen_dll_path)
                    if fast_ode is not None:
                        return fast_ode
            except Exception:
                pass

        # 3. If not found, compile signature DLL if compiler is available
        comp_type, comp_exe, _ = self._find_c_compiler(output_dir)
        if comp_exe:
            compiled = self.compile_c_ode(output_dir)
            if compiled and os.path.exists(compiled):
                fast_ode = self.load_compiled_ode(compiled)
                if fast_ode is not None:
                    return fast_ode

        # 4. Fallback to pure CasADi symbolic ODE: 100% dynamic, instantaneous build (21ms),
        # 5100 Hz evaluation capability, zero compiler dependency.
        return self.get_casadi_ode()

    def apply_powertrain_limits(self, w_rl: float, w_rr: float, T_req_rl: float, T_req_rr: float) -> tuple:
        """
        Applies EMRAX 188HV motor curves, 4:1 gear reduction, and 80 kW accumulator electrical
        power limits to rear driven wheels for the 14-DOF full dynamic model.
        """
        pt = getattr(self.vp, 'get_powertrain', lambda: None)()
        if pt is None:
            from src.models.powertrain_electric import get_default_powertrain
            pt = get_default_powertrain()
        return pt.allocate_dual_motor_torques(w_rl, w_rr, T_req_rl, T_req_rr)
