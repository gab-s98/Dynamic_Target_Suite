import casadi as ca
import numpy as np
from src.models.tire_model_mf61 import mf61_lateral_force
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams

class Vehicle3DOF:
    """
    Planar 3-DOF Vehicle Model for Steady-State Handling Optimization.
    Now leveraging decoupled parameters and MF6.1.
    """
    def __init__(self, vehicle_params=None, tire_params_f=None, tire_params_r=None):
        self.vp = vehicle_params or VehicleParams()
        # Default empty tire params if not provided
        self.tp_f = tire_params_f or TireParams("")
        self.tp_r = tire_params_r or TireParams("")
        
    @staticmethod
    def get_equations_markdown():
        return """
### Modello 3-DOF (Gradi di Libertà) Planare

Il modello calcola l'equilibrio a stato stazionario (velocità costante, accelerazione costante, no transitori).
I 3 gradi di libertà sono: **Velocità Longitudinale** ($v_x$), **Velocità Laterale** ($v_y$) e **Velocità di Imbardata** ($r$). Essendo in Steady-State, $v_x$ è fissa, e l'algoritmo cerca i valori di $v_y$ (o angolo di deriva $\\beta$) e $r$ che azzerano i residui di forza laterale e momento d'imbardata.

#### Forze e Trasferimenti di Carico
Il carico verticale statico tiene conto del bilanciamento dei pesi (wd_f) e dell'aerodinamica:
- $F_{z, front} = m \\cdot g \\cdot (1 - wd_f) + \\frac{1}{2} \\rho S C_z v_x^2 \\cdot aero\\_bal$
- $F_{z, rear} = m \\cdot g \\cdot wd_f + \\frac{1}{2} \\rho S C_z v_x^2 \\cdot (1 - aero\\_bal)$

Il trasferimento di carico laterale (LLT) è calcolato dall'accelerazione laterale $a_y = v_x \\cdot r$:
- $\\Delta F_{z, front} = \\frac{m \\cdot a_y \\cdot h_{cg} \\cdot lltd}{t_f}$
- $\\Delta F_{z, rear} = \\frac{m \\cdot a_y \\cdot h_{cg} \\cdot (1 - lltd)}{t_r}$

#### Angoli di Deriva (Slip Angles)
Assumendo angoli piccoli:
- $\\alpha_{front} = \\delta_{steer} - \\arctan\\left(\\frac{v_y + a \\cdot r}{v_x}\\right)$
- $\\alpha_{rear} = - \\arctan\\left(\\frac{v_y - b \\cdot r}{v_x}\\right)$

#### Equazioni di Equilibrio
1. **Equilibrio Laterale**: $\\sum F_y = m \\cdot a_y$
   $F_{y, front} \\cos(\\delta) + F_{y, rear} - m \\cdot v_x \\cdot r = 0$
2. **Equilibrio Momento Imbardata (Yaw)**: $\\sum M_z = 0$
   $a \\cdot F_{y, front} \\cos(\\delta) - b \\cdot F_{y, rear} = 0$
"""

        
    def _compute_physics(self, v_x, delta_steer, wd_f, lltd, lmuy_mult, beta, r, 
                         L=None, t_f=None, t_r=None, aero_bal=None,
                         camber_f=None, camber_r=None, toe_f=None, toe_r=None):
        """
        Core physics equations for the planar steady-state model with load sensitivity,
        camber thrust, and toe angle kinematic offsets.
        Returns the residuals of the steady-state equilibrium.
        """
        L = L if L is not None else self.vp.L
        t_f = t_f if t_f is not None else self.vp.t_f
        t_r = t_r if t_r is not None else self.vp.t_r
        aero_bal = aero_bal if aero_bal is not None else self.vp.aero_balance
        
        # Longitudinal position of CoG from front and rear axles
        a = L * (1.0 - wd_f)
        b = L * wd_f
        
        # Aerodynamic Downforce
        F_downforce = 0.5 * self.vp.rho * self.vp.S * self.vp.Cz * (v_x**2)
        Fz_aero_f = F_downforce * aero_bal
        Fz_aero_r = F_downforce * (1.0 - aero_bal)
        
        # Static vertical loads PER AXLE
        Fz_axle_f = self.vp.m * self.vp.g * (b / L) + Fz_aero_f
        Fz_axle_r = self.vp.m * self.vp.g * (a / L) + Fz_aero_r
        
        # Steady state lateral acceleration
        A_y = v_x * r
        
        # Lateral Load Transfer (total per axle)
        total_lat_weight_transfer_moment = self.vp.m * A_y * self.vp.h_cg
        
        # Front & Rear lateral load transfer
        delta_Fz_f = (total_lat_weight_transfer_moment * lltd) / t_f
        delta_Fz_r = (total_lat_weight_transfer_moment * (1.0 - lltd)) / t_r
        
        # Tire vertical loads (Left and Right). Protect against liftoff (Fz < 1.0)
        fz_fl = ca.fmax(Fz_axle_f / 2.0 - delta_Fz_f, 1.0)
        fz_fr = ca.fmax(Fz_axle_f / 2.0 + delta_Fz_f, 1.0)
        fz_rl = ca.fmax(Fz_axle_r / 2.0 - delta_Fz_r, 1.0)
        fz_rr = ca.fmax(Fz_axle_r / 2.0 + delta_Fz_r, 1.0)
        
        # Kinematic slip angles (small sideslip approximation: v_y ≈ v_x * beta)
        v_y = v_x * beta 
        
        alpha_f_base = delta_steer - ca.arctan2(v_y + a * r, v_x)
        alpha_r_base = - ca.arctan2(v_y - b * r, v_x)
        
        # Static Camber & Toe offsets per axle [deg -> rad]
        cam_f_val = camber_f if camber_f is not None else getattr(self.vp, 'camber_f_static', 0.0)
        cam_r_val = camber_r if camber_r is not None else getattr(self.vp, 'camber_r_static', 0.0)
        toe_f_val = toe_f if toe_f is not None else getattr(self.vp, 'toe_f_static', 0.0)
        toe_r_val = toe_r if toe_r is not None else getattr(self.vp, 'toe_r_static', 0.0)

        # Check calibrated TYRESIDE from .tir property files
        is_right_f = getattr(self.tp_f, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_f else True
        is_right_r = getattr(self.tp_r, 'get_tyreside', lambda: 'RIGHT')() == 'RIGHT' if self.tp_r else True

        deg2rad = np.deg2rad(1.0)
        if is_right_f:
            gamma_fl = -cam_f_val * deg2rad
            gamma_fr =  cam_f_val * deg2rad
        else:
            gamma_fl =  cam_f_val * deg2rad
            gamma_fr = -cam_f_val * deg2rad

        if is_right_r:
            gamma_rl = -cam_r_val * deg2rad
            gamma_rr =  cam_r_val * deg2rad
        else:
            gamma_rl =  cam_r_val * deg2rad
            gamma_rr = -cam_r_val * deg2rad

        toe_f_rad = toe_f_val * deg2rad
        toe_r_rad = toe_r_val * deg2rad

        # Wheel slip angles with toe geometry
        alpha_fl = alpha_f_base + toe_f_rad
        alpha_fr = alpha_f_base - toe_f_rad
        alpha_rl = alpha_r_base - toe_r_rad
        alpha_rr = alpha_r_base + toe_r_rad

        lvy_fl = -1.0 if is_right_f else 1.0
        lvy_fr =  1.0 if is_right_f else -1.0
        lvy_rl = -1.0 if is_right_r else 1.0
        lvy_rr =  1.0 if is_right_r else -1.0

        lhy_fl = -1.0 if is_right_f else 1.0
        lhy_fr =  1.0 if is_right_f else -1.0
        lhy_rl = -1.0 if is_right_r else 1.0
        lhy_rr =  1.0 if is_right_r else -1.0

        # Lateral forces from MF6.1 (assuming SAE convention TIR, invert for ISO)
        fy_fl = -mf61_lateral_force(alpha_fl, fz_fl, gamma_fl, self.tp_f, LMUY=lmuy_mult, LVY=lvy_fl, LHY=lhy_fl)
        fy_fr = -mf61_lateral_force(alpha_fr, fz_fr, gamma_fr, self.tp_f, LMUY=lmuy_mult, LVY=lvy_fr, LHY=lhy_fr)
        fy_f = fy_fl + fy_fr

        fy_rl = -mf61_lateral_force(alpha_rl, fz_rl, gamma_rl, self.tp_r, LMUY=lmuy_mult, LVY=lvy_rl, LHY=lhy_rl)
        fy_rr = -mf61_lateral_force(alpha_rr, fz_rr, gamma_rr, self.tp_r, LMUY=lmuy_mult, LVY=lvy_rr, LHY=lhy_rr)
        fy_r = fy_rl + fy_rr
        
        # Steady-State Equilibrium Equations
        # 1. Lateral force balance: m * v_x * r = Fy_f * cos(delta) + Fy_r
        eq_lat = fy_f * ca.cos(delta_steer) + fy_r - self.vp.m * v_x * r
        
        # 2. Yaw moment balance: 0 = a * Fy_f * cos(delta) - b * Fy_r
        eq_yaw = a * fy_f * ca.cos(delta_steer) - b * fy_r
        
        return eq_lat, eq_yaw


    def get_casadi_functions(self):
        """
        Builds generic CasADi functions for OpenMDAO integration.
        This interface is modular and can be adopted by ANY N-DOF model.
        """
        # --- Define Symbolic Vectors ---
        v_x = ca.SX.sym('v_x')
        delta_steer = ca.SX.sym('delta_steer')
        inputs = ca.vertcat(v_x, delta_steer)
        
        wd_f = ca.SX.sym('wd_f')
        lltd = ca.SX.sym('lltd')
        lmuy_mult = ca.SX.sym('lmuy_mult')
        params = ca.vertcat(wd_f, lltd, lmuy_mult)
        
        beta = ca.SX.sym('beta')
        r = ca.SX.sym('r')
        states = ca.vertcat(beta, r)
        
        # --- Compute Physics ---
        eq_lat, eq_yaw = self._compute_physics(v_x, delta_steer, wd_f, lltd, lmuy_mult, beta, r)
        residuals = ca.vertcat(eq_lat, eq_yaw)
        
        # --- Compute Exact Jacobians ---
        jac_states = ca.jacobian(residuals, states)
        jac_inputs = ca.jacobian(residuals, inputs)
        jac_params = ca.jacobian(residuals, params)
        
        # --- Create CasADi Functions ---
        in_args = [inputs, states, params]
        res_func = ca.Function('res', in_args, [residuals])
        jac_states_func = ca.Function('jac_states', in_args, [jac_states])
        jac_inputs_func = ca.Function('jac_inputs', in_args, [jac_inputs])
        jac_params_func = ca.Function('jac_params', in_args, [jac_params])
        
        return {
            'res_func': res_func,
            'jac_states_func': jac_states_func,
            'jac_inputs_func': jac_inputs_func,
            'jac_params_func': jac_params_func,
            'meta': {
                'inputs': ['v_x', 'delta_steer'],
                'states': ['beta', 'r'],
                'params': ['wd_f', 'lltd', 'lmuy_mult']
            }
        }

    def get_global_casadi_functions(self):
        """
        Builds generic CasADi functions exposing geometry (L, t_f, t_r) as parameters.
        Useful for Global MDO target finding.
        """
        v_x = ca.SX.sym('v_x')
        delta_steer = ca.SX.sym('delta_steer')
        inputs = ca.vertcat(v_x, delta_steer)
        
        wd_f = ca.SX.sym('wd_f')
        lltd = ca.SX.sym('lltd')
        lmuy_mult = ca.SX.sym('lmuy_mult')
        L_sym = ca.SX.sym('L')
        tf_sym = ca.SX.sym('t_f')
        tr_sym = ca.SX.sym('t_r')
        aero_bal_sym = ca.SX.sym('aero_bal')
        params = ca.vertcat(wd_f, lltd, lmuy_mult, L_sym, tf_sym, tr_sym, aero_bal_sym)
        
        beta = ca.SX.sym('beta')
        r = ca.SX.sym('r')
        states = ca.vertcat(beta, r)
        
        eq_lat, eq_yaw = self._compute_physics(v_x, delta_steer, wd_f, lltd, lmuy_mult, beta, r, L=L_sym, t_f=tf_sym, t_r=tr_sym, aero_bal=aero_bal_sym)
        residuals = ca.vertcat(eq_lat, eq_yaw)
        
        jac_states = ca.jacobian(residuals, states)
        jac_inputs = ca.jacobian(residuals, inputs)
        jac_params = ca.jacobian(residuals, params)
        
        in_args = [inputs, states, params]
        res_func = ca.Function('res', in_args, [residuals])
        jac_states_func = ca.Function('jac_states', in_args, [jac_states])
        jac_inputs_func = ca.Function('jac_inputs', in_args, [jac_inputs])
        jac_params_func = ca.Function('jac_params', in_args, [jac_params])
        
        return {
            'res_func': res_func,
            'jac_states_func': jac_states_func,
            'jac_inputs_func': jac_inputs_func,
            'jac_params_func': jac_params_func,
            'meta': {
                'inputs': ['v_x', 'delta_steer'],
                'states': ['beta', 'r'],
                'params': ['wd_f', 'lltd', 'lmuy_mult', 'L', 't_f', 't_r', 'aero_bal']
            }
        }

    def get_trim_casadi_functions(self):
        return self.get_global_casadi_functions()

    def get_tractive_envelope(self, v_x: float) -> dict:
        """
        Calculates the maximum available tractive force and powertrain operating limits
        at longitudinal speed v_x [m/s] for the Dual-Motor EMRAX 188HV powertrain,
        respecting the 4:1 reduction ratio and the 80 kW accumulator electrical power limit.
        """
        pt = getattr(self.vp, 'get_powertrain', lambda: None)()
        if pt is None:
            from src.models.powertrain_electric import get_default_powertrain
            pt = get_default_powertrain()

        v_clean = max(float(v_x), 0.1)
        w_wheel = v_clean / self.vp.R_w
        w_motor = w_wheel * pt.gear_ratio
        rpm = w_motor * (60.0 / (2.0 * np.pi))

        # Peak torque per motor from EMRAX 188 curve
        t_peak = float(pt.peak_torque(rpm))
        eta_peak = float(pt.efficiency(rpm, t_peak))

        # Power per motor constrained by 80 kW total electrical limit (40 kW each in straight line)
        p_elec_lim_per_motor = (pt.p_elec_limit_kw * 1000.0) / 2.0
        p_mech_lim_per_motor = p_elec_lim_per_motor * eta_peak
        t_power_lim = p_mech_lim_per_motor / max(w_motor, 0.1)

        t_motor_avail = min(t_peak, t_power_lim)
        eta_actual = float(pt.efficiency(rpm, t_motor_avail))
        p_mech_actual = t_motor_avail * w_motor
        p_elec_actual = p_mech_actual / max(eta_actual, 0.1)

        # Total tractive force at ground for 2 driven rear wheels
        conv = pt.gear_ratio * pt.n_drivetrain
        fx_max = (2.0 * t_motor_avail * conv) / self.vp.R_w

        return {
            'v_x': v_clean,
            'rpm': rpm,
            't_motor_single_nm': t_motor_avail,
            'p_mech_single_kw': p_mech_actual / 1000.0,
            'p_mech_total_kw': (2.0 * p_mech_actual) / 1000.0,
            'eta': eta_actual,
            'p_elec_single_kw': p_elec_actual / 1000.0,
            'p_elec_total_kw': (2.0 * p_elec_actual) / 1000.0,
            'fx_max_total_n': fx_max,
            'ax_max_g': fx_max / (self.vp.m * self.vp.g)
        }

    def evaluate_operating_point(self, v_x: float, Fx_req: float = None) -> dict:
        """
        Evaluates powertrain operating state (RPM, torque, powers, efficiency)
        to maintain speed v_x [m/s] overcoming drag/resistance Fx_req [N].
        """
        pt = getattr(self.vp, 'get_powertrain', lambda: None)()
        if pt is None:
            from src.models.powertrain_electric import get_default_powertrain
            pt = get_default_powertrain()

        v_clean = max(float(v_x), 0.1)
        if Fx_req is None:
            # Aero drag + FSAE tire rolling resistance
            f_drag = 0.5 * self.vp.rho * self.vp.S * getattr(self.vp, 'Cd', 1.2) * (v_clean**2)
            f_roll = getattr(self.vp, 'crr', 0.015) * self.vp.m * self.vp.g
            Fx_req = f_drag + f_roll

        w_wheel = v_clean / self.vp.R_w
        w_motor = w_wheel * pt.gear_ratio
        rpm = w_motor * (60.0 / (2.0 * np.pi))

        conv = pt.gear_ratio * pt.n_drivetrain
        # Required torque per motor (2 motors sharing total tractive force)
        t_wheel_req = (float(Fx_req) * self.vp.R_w) / 2.0
        t_motor_req = t_wheel_req / max(conv, 1e-4)

        t_peak = float(pt.peak_torque(rpm))
        t_motor_actual = min(t_motor_req, t_peak)
        eta = float(pt.efficiency(rpm, max(t_motor_actual, 1.0)))

        p_mech = t_motor_actual * w_motor
        p_elec = p_mech / max(eta, 0.1)

        return {
            'v_x': v_clean,
            'rpm': rpm,
            'fx_req_n': float(Fx_req),
            't_motor_nm': t_motor_actual,
            'p_mech_single_kw': p_mech / 1000.0,
            'p_mech_total_kw': (2.0 * p_mech) / 1000.0,
            'p_elec_total_kw': (2.0 * p_elec) / 1000.0,
            'eta': eta,
            'is_saturated': bool(t_motor_actual < t_motor_req)
        }
