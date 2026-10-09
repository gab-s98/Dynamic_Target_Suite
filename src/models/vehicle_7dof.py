import casadi as ca
import numpy as np
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import (
    mf61_lateral_force, 
    mf61_longitudinal_force, 
    mf61_combined_lateral_force, 
    mf61_combined_longitudinal_force
)

class Vehicle7DOF:
    """
    7-DOF Transient & Steady-State Vehicle Dynamics Model.
    
    Trim States (6): [v_y, r, w_fl, w_fr, w_rl, w_rr]
    Trim Inputs (2): [v_x, delta_steer]
    Trim Params (11): [wd_f, lltd, aero_bal, camber_f_static, camber_r_static,
                       toe_f_static, toe_r_static, L, t_f, t_r, lmuy_mult]
                       
    ODE States (7): [v_x, v_y, r, w_fl, w_fr, w_rl, w_rr]
    ODE Inputs (4): [delta_steer, T_drive_rl, T_drive_rr, T_brake_total]
    """
    def __init__(self, vp: VehicleParams = None, tire_params_f: TireParams = None, tire_params_r: TireParams = None):
        self.vp = vp or VehicleParams()
        self.tp_f = tire_params_f or TireParams("")
        self.tp_r = tire_params_r or TireParams("")
        
    @staticmethod
    def get_equations_markdown():
        return """
### Modello 7-DOF (Trim & Transient ODE)

Questo modello dinamico a 7 gradi di libertà include il moto planare del telaio e la rotazione indipendente delle 4 ruote.
I 7 DOF sono:
- **Corpo Vettura**: Velocità Longitudinale ($v_x$), Velocità Laterale ($v_y$), Velocità di Imbardata ($r$).
- **Cinematica Ruote**: 4 velocità angolari ($\\omega_{fl}, \\omega_{fr}, \\omega_{rl}, \\omega_{rr}$).

#### 1. Equazioni del Moto del Corpo Vettura
$$\\dot{v}_x = \\frac{1}{m} \\sum F_x + v_y \\cdot r$$
$$\\dot{v}_y = \\frac{1}{m} \\sum F_y - v_x \\cdot r$$
$$\\dot{r} = \\frac{1}{I_{z}} \\sum M_z$$

#### 2. Dinamica Angolare Ruote
$$\\dot{\\omega}_i = \\frac{1}{I_w} (T_{drive,i} - T_{brake,i} - F_{x,tire,i} \\cdot R_{eff})$$

#### 3. Forze di Contatto Pneumatico (Pacejka MF6.1 Combined Slip)
Le forze combinate $F_x$ e $F_y$ sono calcolate tramite Pacejka MF6.1 tenendo conto della sensitività al carico $F_z$, dell'angolo di campanatura $\\gamma$ e della convergenza $\\delta_{toe}$.
"""

    def get_trim_casadi_functions(self, L=None, t_f=None, t_r=None,
                                   wd_f=None, lltd=None, aero_bal=None,
                                   camber_f_static=None, camber_r_static=None,
                                   toe_f_static=None, toe_r_static=None):
        """
        Build trim residuals and analytical Jacobians for 7-DOF steady-state cornering.
        """
        vp = self.vp

        # Inputs
        v_x     = ca.SX.sym('v_x')
        delta_s = ca.SX.sym('delta_steer')
        inputs  = ca.vertcat(v_x, delta_s)

        # States (Newton solved)
        v_y  = ca.SX.sym('v_y')
        r    = ca.SX.sym('r')
        w_fl = ca.SX.sym('w_fl')
        w_fr = ca.SX.sym('w_fr')
        w_rl = ca.SX.sym('w_rl')
        w_rr = ca.SX.sym('w_rr')
        states = ca.vertcat(v_y, r, w_fl, w_fr, w_rl, w_rr)

        # Parameters
        p_wd_f = ca.SX.sym('wd_f')
        p_lltd = ca.SX.sym('lltd')
        p_ab   = ca.SX.sym('aero_bal')
        p_camf = ca.SX.sym('camber_f_static')
        p_camr = ca.SX.sym('camber_r_static')
        p_toef = ca.SX.sym('toe_f_static')
        p_toer = ca.SX.sym('toe_r_static')
        p_L    = ca.SX.sym('L')
        p_tf   = ca.SX.sym('t_f')
        p_tr   = ca.SX.sym('t_r')
        p_lmuy = ca.SX.sym('lmuy_mult')
        params = ca.vertcat(p_wd_f, p_lltd, p_ab, p_camf, p_camr, p_toef, p_toer, p_L, p_tf, p_tr, p_lmuy)

        # Geometry
        a = p_L * (1.0 - p_wd_f)
        b = p_L * p_wd_f

        # Aerodynamic Downforce
        Fd = 0.5 * vp.rho * vp.S * vp.Cz * (v_x**2)
        Fza_f = Fd * p_ab
        Fza_r = Fd * (1.0 - p_ab)

        Fz_stat_f = vp.m * vp.g * (b / p_L) + Fza_f
        Fz_stat_r = vp.m * vp.g * (a / p_L) + Fza_r

        Ay = v_x * r
        dFz_f = (vp.m * Ay * vp.h_cg * p_lltd) / p_tf
        dFz_r = (vp.m * Ay * vp.h_cg * (1.0 - p_lltd)) / p_tr

        Fz_fl = ca.fmax(Fz_stat_f / 2.0 - dFz_f, 10.0)
        Fz_fr = ca.fmax(Fz_stat_f / 2.0 + dFz_f, 10.0)
        Fz_rl = ca.fmax(Fz_stat_r / 2.0 - dFz_r, 10.0)
        Fz_rr = ca.fmax(Fz_stat_r / 2.0 + dFz_r, 10.0)

        eps = 0.1
        # Velocities at wheel hubs in chassis frame
        Vx_fl = v_x - r * p_tf / 2.0
        Vy_fl = v_y + r * a
        Vx_fr = v_x + r * p_tf / 2.0
        Vy_fr = v_y + r * a
        Vx_rl = v_x - r * p_tr / 2.0
        Vy_rl = v_y - r * b
        Vx_rr = v_x + r * p_tr / 2.0
        Vy_rr = v_y - r * b

        # Tire coordinate velocities
        Vxw_fl =  Vx_fl * ca.cos(delta_s) + Vy_fl * ca.sin(delta_s)
        Vyw_fl = -Vx_fl * ca.sin(delta_s) + Vy_fl * ca.cos(delta_s)
        Vxw_fr =  Vx_fr * ca.cos(delta_s) + Vy_fr * ca.sin(delta_s)
        Vyw_fr = -Vx_fr * ca.sin(delta_s) + Vy_fr * ca.cos(delta_s)
        Vxw_rl =  Vx_rl
        Vyw_rl =  Vy_rl
        Vxw_rr =  Vx_rr
        Vyw_rr =  Vy_rr

        # Check calibrated TYRESIDE from .tir property files
        is_right_f = getattr(self.tp_f, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_f else True
        is_right_r = getattr(self.tp_r, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_r else True

        # Slip angles with static camber and toe (deg to rad)
        deg_to_rad = ca.pi / 180.0
        if is_right_f:
            gamma_fl = -p_camf * deg_to_rad
            gamma_fr =  p_camf * deg_to_rad
        else:
            gamma_fl =  p_camf * deg_to_rad
            gamma_fr = -p_camf * deg_to_rad

        if is_right_r:
            gamma_rl = -p_camr * deg_to_rad
            gamma_rr =  p_camr * deg_to_rad
        else:
            gamma_rl =  p_camr * deg_to_rad
            gamma_rr = -p_camr * deg_to_rad

        alpha_fl = -ca.arctan2(Vyw_fl, ca.fmax(ca.fabs(Vxw_fl), eps)) + p_toef * deg_to_rad
        alpha_fr = -ca.arctan2(Vyw_fr, ca.fmax(ca.fabs(Vxw_fr), eps)) - p_toef * deg_to_rad
        alpha_rl = -ca.arctan2(Vyw_rl, ca.fmax(ca.fabs(Vxw_rl), eps)) - p_toer * deg_to_rad
        alpha_rr = -ca.arctan2(Vyw_rr, ca.fmax(ca.fabs(Vxw_rr), eps)) + p_toer * deg_to_rad

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

        # Rotate steered tire forces into chassis frame
        Fy_fl_ch = fy_fl * ca.cos(delta_s)
        Fy_fr_ch = fy_fr * ca.cos(delta_s)
        Fy_f = Fy_fl_ch + Fy_fr_ch
        Fy_r = fy_rl + fy_rr

        # Planar residuals
        res_lat = Fy_f + Fy_r - vp.m * v_x * r
        res_yaw = a * Fy_f - b * Fy_r + (p_tf / 2.0) * (fy_fl * ca.sin(delta_s) - fy_fr * ca.sin(delta_s))

        # Pure rolling wheel spin residuals
        Rw = vp.R_w
        res_wfl = w_fl - Vxw_fl / Rw
        res_wfr = w_fr - Vxw_fr / Rw
        res_wrl = w_rl - Vxw_rl / Rw
        res_wrr = w_rr - Vxw_rr / Rw

        residuals = ca.vertcat(res_lat, res_yaw, res_wfl, res_wfr, res_wrl, res_wrr)

        in_args = [inputs, states, params]
        jac_states = ca.jacobian(residuals, states)
        jac_inputs = ca.jacobian(residuals, inputs)
        jac_params = ca.jacobian(residuals, params)

        return {
            'res_func':        ca.Function('res7', in_args, [residuals]),
            'jac_states_func': ca.Function('jac7_st', in_args, [jac_states]),
            'jac_inputs_func': ca.Function('jac7_in', in_args, [jac_inputs]),
            'jac_params_func': ca.Function('jac7_p',  in_args, [jac_params]),
            'meta': {
                'inputs': ['v_x', 'delta_steer'],
                'states': ['v_y', 'r', 'w_fl', 'w_fr', 'w_rl', 'w_rr'],
                'params': ['wd_f', 'lltd', 'aero_bal',
                           'camber_f_static', 'camber_r_static',
                           'toe_f_static', 'toe_r_static',
                           'L', 't_f', 't_r', 'lmuy_mult']
            }
        }

    def get_casadi_ode(self):
        """
        Returns a CasADi Function for the 7-DOF ODE: x_dot = f(x, u, p)
        States (7): [v_x, v_y, r, w_fl, w_fr, w_rl, w_rr]
        Inputs (4): [delta, T_drive_rl, T_drive_rr, T_brake]
        Params (11): [wd_f, lltd, lmux, lmuy, L, t_f, t_r, cam_f, cam_r, toe_f, toe_r]
        """
        # States
        v_x  = ca.MX.sym('v_x')
        v_y  = ca.MX.sym('v_y')
        r    = ca.MX.sym('r')
        w_fl = ca.MX.sym('w_fl')
        w_fr = ca.MX.sym('w_fr')
        w_rl = ca.MX.sym('w_rl')
        w_rr = ca.MX.sym('w_rr')
        x = ca.vertcat(v_x, v_y, r, w_fl, w_fr, w_rl, w_rr)

        # Inputs
        delta      = ca.MX.sym('delta')
        T_drive_rl = ca.MX.sym('T_drive_rl')
        T_drive_rr = ca.MX.sym('T_drive_rr')
        T_brake    = ca.MX.sym('T_brake')
        u = ca.vertcat(delta, T_drive_rl, T_drive_rr, T_brake)

        # Parameters
        wd_f  = ca.MX.sym('wd_f')
        lltd  = ca.MX.sym('lltd')
        lmux  = ca.MX.sym('lmux')
        lmuy  = ca.MX.sym('lmuy')
        L_sym = ca.MX.sym('L')
        tf_sym = ca.MX.sym('t_f')
        tr_sym = ca.MX.sym('t_r')
        cam_f = ca.MX.sym('cam_f')
        cam_r = ca.MX.sym('cam_r')
        toe_f = ca.MX.sym('toe_f')
        toe_r = ca.MX.sym('toe_r')
        p = ca.vertcat(wd_f, lltd, lmux, lmuy, L_sym, tf_sym, tr_sym, cam_f, cam_r, toe_f, toe_r)

        # Vehicle constants
        m = self.vp.m
        Iz = self.vp.Iz
        Iw = self.vp.Iw
        Rw = self.vp.R_w
        h = self.vp.h_cg
        a = L_sym * (1.0 - wd_f)
        b = L_sym * wd_f

        # Aerodynamics
        F_down = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (v_x**2)
        F_drag = 0.5 * self.vp.rho * self.vp.S * self.vp.Cd * (v_x**2)
        Fza_f  = F_down * self.vp.aero_balance
        Fza_r  = F_down * (1.0 - self.vp.aero_balance)

        # Vertical loads
        Fz_stat_f = m * self.vp.g * (b / L_sym) + Fza_f
        Fz_stat_r = m * self.vp.g * (a / L_sym) + Fza_r

        Ay_pseudo = v_x * r
        lat_wt = m * Ay_pseudo * h
        dFz_f = (lat_wt * lltd) / tf_sym
        dFz_r = (lat_wt * (1.0 - lltd)) / tr_sym

        Fz_fl = ca.fmax(Fz_stat_f / 2.0 - dFz_f, 10.0)
        Fz_fr = ca.fmax(Fz_stat_f / 2.0 + dFz_f, 10.0)
        Fz_rl = ca.fmax(Fz_stat_r / 2.0 - dFz_r, 10.0)
        Fz_rr = ca.fmax(Fz_stat_r / 2.0 + dFz_r, 10.0)

        # Velocities at wheel hubs
        Vx_fl = v_x - r * tf_sym / 2.0
        Vy_fl = v_y + r * a
        Vx_fr = v_x + r * tf_sym / 2.0
        Vy_fr = v_y + r * a
        Vx_rl = v_x - r * tr_sym / 2.0
        Vy_rl = v_y - r * b
        Vx_rr = v_x + r * tr_sym / 2.0
        Vy_rr = v_y - r * b

        Vxw_fl =  Vx_fl * ca.cos(delta) + Vy_fl * ca.sin(delta)
        Vyw_fl = -Vx_fl * ca.sin(delta) + Vy_fl * ca.cos(delta)
        Vxw_fr =  Vx_fr * ca.cos(delta) + Vy_fr * ca.sin(delta)
        Vyw_fr = -Vx_fr * ca.sin(delta) + Vy_fr * ca.cos(delta)
        Vxw_rl =  Vx_rl
        Vyw_rl =  Vy_rl
        Vxw_rr =  Vx_rr
        Vyw_rr =  Vy_rr

        eps = 0.1
        def calc_kappa(w, Vxw):
            return (w * Rw - Vxw) / ca.fmax(ca.fabs(Vxw), eps)

        kappa_fl = calc_kappa(w_fl, Vxw_fl)
        kappa_fr = calc_kappa(w_fr, Vxw_fr)
        kappa_rl = calc_kappa(w_rl, Vxw_rl)
        kappa_rr = calc_kappa(w_rr, Vxw_rr)

        d2r = ca.pi / 180.0
        is_right_f = getattr(self.tp_f, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_f else True
        is_right_r = getattr(self.tp_r, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_r else True

        if is_right_f:
            gamma_fl = -cam_f * d2r
            gamma_fr =  cam_f * d2r
        else:
            gamma_fl =  cam_f * d2r
            gamma_fr = -cam_f * d2r

        if is_right_r:
            gamma_rl = -cam_r * d2r
            gamma_rr =  cam_r * d2r
        else:
            gamma_rl =  cam_r * d2r
            gamma_rr = -cam_r * d2r

        alpha_fl = -ca.arctan2(Vyw_fl, ca.fmax(ca.fabs(Vxw_fl), eps)) + toe_f * d2r
        alpha_fr = -ca.arctan2(Vyw_fr, ca.fmax(ca.fabs(Vxw_fr), eps)) - toe_f * d2r
        alpha_rl = -ca.arctan2(Vyw_rl, ca.fmax(ca.fabs(Vxw_rl), eps)) - toe_r * d2r
        alpha_rr = -ca.arctan2(Vyw_rr, ca.fmax(ca.fabs(Vxw_rr), eps)) + toe_r * d2r

        lvy_fl = -1.0 if is_right_f else 1.0
        lvy_fr =  1.0 if is_right_f else -1.0
        lvy_rl = -1.0 if is_right_r else 1.0
        lvy_rr =  1.0 if is_right_r else -1.0

        lhy_fl = -1.0 if is_right_f else 1.0
        lhy_fr =  1.0 if is_right_f else -1.0
        lhy_rl = -1.0 if is_right_r else 1.0
        lhy_rr =  1.0 if is_right_r else -1.0

        # Pure tire forces
        Fx_fl_p = mf61_longitudinal_force(kappa_fl, Fz_fl, gamma_fl, self.tp_f, LMUX=lmux)
        Fy_fl_p = -mf61_lateral_force(alpha_fl, Fz_fl, gamma_fl, self.tp_f, LMUY=lmuy, LVY=lvy_fl, LHY=lhy_fl)
        Fx_fr_p = mf61_longitudinal_force(kappa_fr, Fz_fr, gamma_fr, self.tp_f, LMUX=lmux)
        Fy_fr_p = -mf61_lateral_force(alpha_fr, Fz_fr, gamma_fr, self.tp_f, LMUY=lmuy, LVY=lvy_fr, LHY=lhy_fr)

        Fx_rl_p = mf61_longitudinal_force(kappa_rl, Fz_rl, gamma_rl, self.tp_r, LMUX=lmux)
        Fy_rl_p = -mf61_lateral_force(alpha_rl, Fz_rl, gamma_rl, self.tp_r, LMUY=lmuy, LVY=lvy_rl, LHY=lhy_rl)
        Fx_rr_p = mf61_longitudinal_force(kappa_rr, Fz_rr, gamma_rr, self.tp_r, LMUX=lmux)
        Fy_rr_p = -mf61_lateral_force(alpha_rr, Fz_rr, gamma_rr, self.tp_r, LMUY=lmuy, LVY=lvy_rr, LHY=lhy_rr)

        # Combined slip tire forces
        Fxw_fl = mf61_combined_longitudinal_force(Fx_fl_p, kappa_fl, alpha_fl, Fz_fl, gamma_fl, self.tp_f, LHY=lhy_fl)
        Fyw_fl = mf61_combined_lateral_force(Fy_fl_p, alpha_fl, kappa_fl, Fz_fl, gamma_fl, self.tp_f, LVY=lvy_fl, LHY=lhy_fl)
        Fxw_fr = mf61_combined_longitudinal_force(Fx_fr_p, kappa_fr, alpha_fr, Fz_fr, gamma_fr, self.tp_f, LHY=lhy_fr)
        Fyw_fr = mf61_combined_lateral_force(Fy_fr_p, alpha_fr, kappa_fr, Fz_fr, gamma_fr, self.tp_f, LVY=lvy_fr, LHY=lhy_fr)

        Fxw_rl = mf61_combined_longitudinal_force(Fx_rl_p, kappa_rl, alpha_rl, Fz_rl, gamma_rl, self.tp_r, LHY=lhy_rl)
        Fyw_rl = mf61_combined_lateral_force(Fy_rl_p, alpha_rl, kappa_rl, Fz_rl, gamma_rl, self.tp_r, LVY=lvy_rl, LHY=lhy_rl)
        Fxw_rr = mf61_combined_longitudinal_force(Fx_rr_p, kappa_rr, alpha_rr, Fz_rr, gamma_rr, self.tp_r, LHY=lhy_rr)
        Fyw_rr = mf61_combined_lateral_force(Fy_rr_p, alpha_rr, kappa_rr, Fz_rr, gamma_rr, self.tp_r, LVY=lvy_rr, LHY=lhy_rr)

        # Rotate front forces into chassis frame
        Fx_fl =  Fxw_fl * ca.cos(delta) - Fyw_fl * ca.sin(delta)
        Fy_fl =  Fxw_fl * ca.sin(delta) + Fyw_fl * ca.cos(delta)
        Fx_fr =  Fxw_fr * ca.cos(delta) - Fyw_fr * ca.sin(delta)
        Fy_fr =  Fxw_fr * ca.sin(delta) + Fyw_fr * ca.cos(delta)

        Fx_rl = Fxw_rl
        Fy_rl = Fyw_rl
        Fx_rr = Fxw_rr
        Fy_rr = Fyw_rr

        Sum_Fx = Fx_fl + Fx_fr + Fx_rl + Fx_rr - F_drag
        Sum_Fy = Fy_fl + Fy_fr + Fy_rl + Fy_rr

        # Yaw Moment
        Mz = (a * (Fy_fl + Fy_fr) - b * (Fy_rl + Fy_rr) +
              (tf_sym / 2.0) * (Fx_fr - Fx_fl) +
              (tr_sym / 2.0) * (Fx_rr - Fx_rl))

        # Chassis ODEs
        v_x_dot = (Sum_Fx / m) + r * v_y
        v_y_dot = (Sum_Fy / m) - r * v_x
        r_dot   = Mz / Iz

        # Wheel ODEs with Brake Bias
        Tb_f = (T_brake * self.vp.brake_bias) / 2.0
        Tb_r = (T_brake * (1.0 - self.vp.brake_bias)) / 2.0

        def calc_Tb(w, Tb_mag):
            return ca.tanh(w * 10.0) * Tb_mag

        w_fl_dot = (-Fxw_fl * Rw - calc_Tb(w_fl, Tb_f)) / Iw
        w_fr_dot = (-Fxw_fr * Rw - calc_Tb(w_fr, Tb_f)) / Iw
        w_rl_dot = (T_drive_rl - Fxw_rl * Rw - calc_Tb(w_rl, Tb_r)) / Iw
        w_rr_dot = (T_drive_rr - Fxw_rr * Rw - calc_Tb(w_rr, Tb_r)) / Iw

        x_dot = ca.vertcat(v_x_dot, v_y_dot, r_dot, w_fl_dot, w_fr_dot, w_rl_dot, w_rr_dot)
        return ca.Function('ode_7dof', [x, u, p], [x_dot], ['x', 'u', 'p'], ['x_dot'])

    def apply_powertrain_limits(self, w_rl: float, w_rr: float, T_req_rl: float, T_req_rr: float) -> tuple:
        """
        Applies EMRAX 188HV motor curves, 4:1 gear reduction, and 80 kW accumulator electrical
        power limits to rear driven wheels for the 7-DOF dynamic model.
        """
        pt = getattr(self.vp, 'get_powertrain', lambda: None)()
        if pt is None:
            from src.models.powertrain_electric import get_default_powertrain
            pt = get_default_powertrain()
        return pt.allocate_dual_motor_torques(w_rl, w_rr, T_req_rl, T_req_rr)
