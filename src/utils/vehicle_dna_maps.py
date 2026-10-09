"""
vehicle_dna_maps.py
-------------------
Implements Vehicle DNA stability maps and the rigorous gradient-based
definition of Understeer / Oversteer based on steady-state curvature gradients
(conforming to SAE J670e and ISO 4138 standards).

Theory Highlights:
------------------
1. Understeer Gradient and Steady-State Gradients:
   - Steering equation: delta = L * rho + K * ay  <=>  rho_p(delta, ay) = delta / L - (K / L) * ay
   - Curvature gradient: rho_y = ∂rho_p/∂ay = -K / L
   - Understeer gradient: K = -L * rho_y  [rad/(m/s^2)] or [deg/g]
     * Understeer (Sottosterzante): rho_y < 0  <=>  K > 0
     * Neutral (Neutro):           rho_y = 0  <=>  K = 0
     * Oversteer (Sovrasterzante):  rho_y > 0  <=>  K < 0
   - Steering sensitivity: rho_delta = ∂rho_p/∂delta = 1 / L
   - Sideslip gradients: beta_y = ∂beta_p/∂ay = D_beta, beta_delta = ∂beta_p/∂delta = b / L

2. Stability Derivatives from Steady-State Gradients:
   - Y_beta = (Y_delta * rho_y + m * rho_delta) / (beta_y * rho_delta - beta_delta * rho_y)
   - Y_rho  = (-Y_delta * beta_y + m * beta_delta) / (beta_y * rho_delta - beta_delta * rho_y)
   - N_beta = (N_delta * rho_y) / (beta_y * rho_delta - beta_delta * rho_y)
   - N_rho  = (-N_delta * beta_y) / (beta_y * rho_delta - beta_delta * rho_y)

3. The 6 Vehicle DNA Stability Maps n1...n6:
   - n1(delta, ay) = 2*zeta*omega_n = -tr(A)  (Equivalent Damping)
   - n2(delta, ay) = omega_n^2 = det(A)       (Equivalent Stiffness / Dynamic Stability)
   - n3(delta, ay) = -N_hat_delta / u^2 * (beta_delta / D + u^2)   (Static Forcing for beta)
   - n4(delta, ay) = Y_hat_delta / u                              (Dynamic Forcing for beta_dot)
   - n5(delta, ay) = -N_hat_delta / u^2 * (rho_delta / D)         (Static Forcing for rho)
   - n6(delta, ay) = N_hat_delta / u                              (Dynamic Forcing for rho_dot)
   where D = beta_y * rho_delta - beta_delta * rho_y < 0.

4. Dynamic Stability Condition & Critical Speed:
   - det(A) = n2 = (N_hat_delta / D) * (rho_y - 1/u^2) > 0  <=>  rho_y < 1/u^2
   - Substituting rho_y = -K/L:  -K/L < 1/u^2  <=>  1 + (K * u^2) / L > 0
   - Critical Speed (for oversteering cars K < 0 <=> rho_y > 0):
     u_crit = sqrt(L / -K) = 1 / sqrt(rho_y)
"""

import os
import numpy as np
import pandas as pd
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import mf61_lateral_force


def compute_vehicle_dna_fields(v_x=15.0, wd_f=0.5, lltd=0.5, aero_bal=0.48,
                               L=1.53, t_f=1.25, t_r=1.15, tir_file=None,
                               delta_max_deg=18.0, ay_max_g=2.4,
                               n_delta=35, n_ay=35,
                               steering_ratio=5.0):
    """
    Computes dense 2D spatial fields in the (delta_v, a_y) plane for:
    - Gradients: rho_y, rho_delta, beta_y, beta_delta
    - The 6 DNA Stability Maps: n1, n2, n3, n4, n5, n6
    - Natural frequency fn [Hz] and damping ratio zeta [-]
    - Curvature-Velocity Understeer Gradient Kus [deg/g]
    - Dynamic instability threshold contour (n2 = 0 <=> rho_y = 1/u^2)

    steering_ratio : float
        Hand-wheel to road-wheel steering ratio [-]. Used for axis labeling.

    Returns
    -------
    dict with meshgrid matrices and scalar metrics.
    """
    tp = TireParams(tir_file)
    vp = VehicleParams()
    m = vp.m
    Iz = vp.Iz
    g = vp.g
    a = L * (1.0 - wd_f)
    b = L * wd_f

    # Aerodynamic vertical downforce
    f_down = 0.5 * vp.rho * vp.S * vp.Cz * (v_x**2)
    Fz_f_nom = max((m * g * (b / L) + f_down * aero_bal) / 2.0, 100.0)
    Fz_r_nom = max((m * g * (a / L) + f_down * (1.0 - aero_bal)) / 2.0, 100.0)

    # Cornering stiffness calculation
    d_alpha = np.deg2rad(1.0)
    Fy_f_1deg = abs(float(mf61_lateral_force(d_alpha, Fz_f_nom, 0.0, tp))) * 2.0
    Fy_r_1deg = abs(float(mf61_lateral_force(d_alpha, Fz_r_nom, 0.0, tp))) * 2.0
    C_alpha_f = max(Fy_f_1deg / d_alpha, 1000.0)  # [N/rad]
    C_alpha_r = max(Fy_r_1deg / d_alpha, 1000.0)  # [N/rad]

    # Control derivatives
    Y_delta = C_alpha_f                         # [N/rad]
    N_delta = a * C_alpha_f                     # [N·m/rad]
    Y_hat_delta = Y_delta / m                   # [N/(rad·kg)]
    N_hat_delta = N_delta / Iz                  # [N·m/(rad·kg·m²)]

    u_a = max(v_x, 1.0)                         # [m/s]

    # Coordinate grids in (delta [rad], ay [m/s^2])
    delta_vec_deg = np.linspace(0.2, delta_max_deg, n_delta)
    ay_vec_g = np.linspace(0.05, ay_max_g, n_ay)

    Delta_deg, Ay_g = np.meshgrid(delta_vec_deg, ay_vec_g)
    Delta_rad = np.deg2rad(Delta_deg)
    Ay_ms2 = Ay_g * g

    # Initialize matrices for steady-state equilibrium states (rho_p, beta_p)
    # Using double track bicycle approximation with nonlinear tire saturation
    Rho_p = np.zeros_like(Delta_rad)
    Beta_p = np.zeros_like(Delta_rad)

    for i in range(n_ay):
        ay_i = Ay_ms2[i, 0]
        # Dynamic load transfer on front and rear axles
        # Full geometric + elastic lateral load transfer (matching vehicle_14dof.py)
        K_phi_f = vp.K_phi_f  # Front roll stiffness [Nm/rad]
        K_phi_r = vp.K_phi_r  # Rear roll stiffness [Nm/rad]
        K_phi_tot = K_phi_f + K_phi_r
        q1 = vp.q1  # Front roll center height [m]
        q2 = vp.q2  # Rear roll center height [m]
        h_rc_avg = (q1 * b + q2 * a) / max(L, 0.5)
        h_roll_arm = vp.h_cg - h_rc_avg

        # Steady-state roll angle
        phi_ss = (m * ay_i * h_roll_arm) / max(K_phi_tot - m * g * h_roll_arm, 10.0)

        # Elastic LLT (through springs/ARBs via roll)
        LLTD_elastic = K_phi_f / max(K_phi_tot, 1.0)
        LLTD_eff = 0.5 * (lltd + LLTD_elastic)
        dFz_f_elastic = (m * ay_i * h_roll_arm * LLTD_eff) / max(t_f, 0.5)
        dFz_r_elastic = (m * ay_i * h_roll_arm * (1.0 - LLTD_eff)) / max(t_r, 0.5)

        # Geometric LLT (direct load transfer through roll centers)
        dFz_f_geo = (m * ay_i * (b / max(L, 0.5)) * q1) / max(t_f, 0.5)
        dFz_r_geo = (m * ay_i * (a / max(L, 0.5)) * q2) / max(t_r, 0.5)

        dFz_f = dFz_f_elastic + dFz_f_geo
        dFz_r = dFz_r_elastic + dFz_r_geo

        Fz_fo = max(Fz_f_nom + dFz_f / 2.0, 20.0)
        Fz_fi = max(Fz_f_nom - dFz_f / 2.0, 20.0)
        Fz_ro = max(Fz_r_nom + dFz_r / 2.0, 20.0)
        Fz_ri = max(Fz_r_nom - dFz_r / 2.0, 20.0)

        # Required lateral forces for equilibrium:
        # Fy_f * a = Fy_r * b (yaw balance) and Fy_f + Fy_r = m * ay_i
        Fy_f_req = m * ay_i * (b / L)
        Fy_r_req = m * ay_i * (a / L)

        def _calc_axle_force_f(al):
            return abs(float(mf61_lateral_force(al, Fz_fo, 0.0, tp))) + abs(float(mf61_lateral_force(al, Fz_fi, 0.0, tp)))
            
        def _calc_axle_force_r(al):
            return abs(float(mf61_lateral_force(al, Fz_ro, 0.0, tp))) + abs(float(mf61_lateral_force(al, Fz_ri, 0.0, tp)))

        # Solve exact non-linear slip angles alpha_f and alpha_r matching equilibrium demand
        try:
            import scipy.optimize as opt
            # Find peak capacity of front and rear axles
            al_bracket = [0.0, np.deg2rad(25.0)]
            max_Fy_f = _calc_axle_force_f(np.deg2rad(15.0))
            max_Fy_r = _calc_axle_force_r(np.deg2rad(15.0))
            
            if Fy_f_req < max_Fy_f * 0.995:
                sol_f = opt.root_scalar(lambda al: _calc_axle_force_f(al) - Fy_f_req, bracket=al_bracket, method='brentq')
                alpha_f = float(sol_f.root)
            else:
                alpha_f = float(np.deg2rad(15.0) * (Fy_f_req / max(max_Fy_f, 1.0)))
                
            if Fy_r_req < max_Fy_r * 0.995:
                sol_r = opt.root_scalar(lambda al: _calc_axle_force_r(al) - Fy_r_req, bracket=al_bracket, method='brentq')
                alpha_r = float(sol_r.root)
            else:
                alpha_r = float(np.deg2rad(15.0) * (Fy_r_req / max(max_Fy_r, 1.0)))
        except Exception:
            C_f_eff = max((abs(float(mf61_lateral_force(d_alpha, Fz_fo, 0.0, tp))) +
                           abs(float(mf61_lateral_force(d_alpha, Fz_fi, 0.0, tp)))) / d_alpha, 500.0)
            C_r_eff = max((abs(float(mf61_lateral_force(d_alpha, Fz_ro, 0.0, tp))) +
                           abs(float(mf61_lateral_force(d_alpha, Fz_ri, 0.0, tp)))) / d_alpha, 500.0)
            alpha_f = Fy_f_req / C_f_eff
            alpha_r = Fy_r_req / C_r_eff

        for j in range(n_delta):
            delta_j = Delta_rad[i, j]

            # Kinematic relationships:
            # delta = L * rho + alpha_f - alpha_r  =>  rho = (delta - (alpha_f - alpha_r)) / L
            # beta = -alpha_r + b * rho
            rho_val = (delta_j - (alpha_f - alpha_r)) / max(L, 0.5)
            beta_val = -alpha_r + b * rho_val

            Rho_p[i, j] = rho_val
            Beta_p[i, j] = beta_val

    # Numerical partial derivatives of steady-state maps (grad rho_p, grad beta_p)
    # rho_y = ∂rho/∂ay [1/(m · m/s^2)], rho_delta = ∂rho/∂delta [1/(m · rad)]
    # beta_y = ∂beta/∂ay [rad / (m/s^2)], beta_delta = ∂beta/∂delta [rad/rad]
    d_ay_step = Ay_ms2[1, 0] - Ay_ms2[0, 0]
    d_delta_step = Delta_rad[0, 1] - Delta_rad[0, 0]

    Rho_y, Rho_delta = np.gradient(Rho_p, d_ay_step, d_delta_step)
    Beta_y, Beta_delta = np.gradient(Beta_p, d_ay_step, d_delta_step)

    # Denominator determinant D = beta_y * rho_delta - beta_delta * rho_y
    # In physically stable vehicle dynamics, D < 0 everywhere
    D = Beta_y * Rho_delta - Beta_delta * Rho_y
    D = np.where(D > -1e-6, -1e-6, D)
    D_safe = D

    # 1. DNA Map n1(delta, ay) = 2*zeta*omega_n = -tr(A) (Equivalent Damping)
    N1 = (1.0 / (u_a * D_safe)) * (N_hat_delta * Beta_y - Y_hat_delta * Rho_y - Rho_delta)

    # 2. DNA Map n2(delta, ay) = omega_n^2 = det(A) (Equivalent Stiffness / Stability)
    # Notice: n2 < 0 occurs when rho_y > 1/u^2 (Dynamic Instability / Oversteer Divergence)
    N2 = (N_hat_delta / D_safe) * (Rho_y - (1.0 / (u_a**2)))

    # 3. DNA Map n3(delta, ay) (Static Forcing on beta)
    N3 = -(N_hat_delta / (u_a**2)) * ((Beta_delta / D_safe) + u_a**2)

    # 4. DNA Map n4(delta, ay) (Dynamic Forcing on beta)
    N4 = np.full_like(Delta_rad, Y_hat_delta / u_a)

    # 5. DNA Map n5(delta, ay) (Static Forcing on rho/r)
    N5 = -(N_hat_delta / (u_a**2)) * (Rho_delta / D_safe)

    # 6. DNA Map n6(delta, ay) (Dynamic Forcing on rho/r)
    N6 = np.full_like(Delta_rad, N_hat_delta / u_a)

    # Modal properties from n1 and n2
    Omega_n = np.sqrt(np.maximum(N2, 0.0))
    F_n_hz = Omega_n / (2.0 * np.pi)
    Zeta = np.where(Omega_n > 1e-4, N1 / (2.0 * np.maximum(Omega_n, 1e-4)), 1.0)

    # Understeer Gradient (rho_y = -K/L  =>  K = -L * rho_y):
    K_intrinsic_rad = -L * Rho_y
    K_intrinsic_degg = K_intrinsic_rad * (180.0 / np.pi) * g

    # Handling trim curve gradient at constant speed: (∂delta/∂ay)_{u=cost} = L/u^2 + K = -(rho_y - 1/u^2) / rho_delta
    d_delta_d_ay_rad = -(Rho_y - (1.0 / (u_a**2))) / np.maximum(Rho_delta, 1e-5)
    d_delta_d_ay_degg = d_delta_d_ay_rad * (180.0 / np.pi) * g

    # Stability condition matrix: True if n2 > 0 and n1 > 0
    Is_Stable_Grid = (N2 > 0.0) & (N1 > 0.0)

    return {
        'v_x': v_x,
        'Delta_deg': Delta_deg,
        'Ay_g': Ay_g,
        'Delta_rad': Delta_rad,
        'Ay_ms2': Ay_ms2,
        'Rho_p': Rho_p,
        'Beta_p': Beta_p,
        'Rho_y': Rho_y,
        'Rho_delta': Rho_delta,
        'Beta_y': Beta_y,
        'Beta_delta': Beta_delta,
        'D': D,
        'N1': N1,
        'N2': N2,
        'N3': N3,
        'N4': N4,
        'N5': N5,
        'N6': N6,
        'Omega_n': Omega_n,
        'F_n_hz': F_n_hz,
        'Zeta': Zeta,
        'K_intrinsic_degg': K_intrinsic_degg,
        'd_delta_d_ay_degg': d_delta_d_ay_degg,
        'K_us_gradient_degg': K_intrinsic_degg,
        'K_us_degg': K_intrinsic_degg,
        'Is_Stable_Grid': Is_Stable_Grid,
        'Y_hat_delta': Y_hat_delta,
        'N_hat_delta': N_hat_delta,
        'C_alpha_f': C_alpha_f,
        'C_alpha_r': C_alpha_r,
        'steering_ratio': steering_ratio,
        'delta_sw_max_deg': delta_max_deg * steering_ratio,
    }


def plot_vehicle_dna_figures(fig, dna_data, view_mode='dna_eigen', language='it'):
    """
    Renders Vehicle Dynamic Stability & Sensitivity Maps and Understeer analysis
    into a matplotlib Figure, supporting both Italian and English formal reporting modes.

    Parameters
    ----------
    fig       : matplotlib.figure.Figure
    dna_data  : dict returned by compute_vehicle_dna_fields()
    view_mode : 'dna_eigen' or 'understeer_gradient'
    language  : 'it' or 'en'
    """
    fig.clear()
    Delta_deg = dna_data['Delta_deg']
    Ay_g = dna_data['Ay_g']
    v_x = dna_data['v_x']
    is_en = (language.lower() == 'en')

    if view_mode == 'dna_eigen':
        # View 1: 4 Subplots
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # 1. Subplot (221): MAP (delta, ay) with iso-zeta and iso-fn
        Zeta = dna_data['Zeta']
        F_n = dna_data['F_n_hz']
        N2 = dna_data['N2']

        # Damping ratio curves (dashed lines)
        zeta_levels = [0.3, 0.5, 0.7, 0.9, 1.1, 1.4]
        cs_zeta = ax1.contour(Delta_deg, Ay_g, Zeta, levels=zeta_levels, colors='#0984e3', linestyles='--', linewidths=1.2)
        ax1.clabel(cs_zeta, inline=True, fontsize=7.5, fmt='ζ=%.1f')

        # Natural frequency curves (solid lines)
        fn_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        cs_fn = ax1.contour(Delta_deg, Ay_g, F_n, levels=fn_levels, colors='#2d3436', linestyles='-', linewidths=1.4)
        ax1.clabel(cs_fn, inline=True, fontsize=7.5, fmt='fn=%.1fHz')

        # Instability boundary (N2 = 0 <=> rho_y = 1/u^2)
        ax1.contour(Delta_deg, Ay_g, N2, levels=[0.0], colors='#d63031', linewidths=2.2, linestyles='-')
        if is_en:
            ax1.set_title("1. Oscillatory Dynamic Modes in $(\\delta, a_y)$ Plane\n(Solid: $f_n$ [Hz], Dashed: $\\zeta$, Red: Limit $N_2=0$)", fontsize=8.5, fontweight='bold')
            ax1.set_xlabel("Steering Angle (RW) $\\delta$ / HW $\\delta_{sw}$ [deg]", fontsize=8)
            ax1.set_ylabel("Lateral Acceleration $a_y$ [g]", fontsize=8)
        else:
            ax1.set_title("1. Modi Oscillatori nel Piano (δ, ay)\n(Continuo: fn [Hz], Tratteggio: ζ, Rosso: Limite N₂=0)", fontsize=8.5, fontweight='bold')
            ax1.set_xlabel("Angolo Sterzo (RW) δ / Volante (HW) δ_sw [deg]", fontsize=8)
            ax1.set_ylabel("Acc. Laterale ay [g]", fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # 2. Subplot (222): Dynamic Stiffness / Stability Field n2(delta, ay) = omega_n^2 = det(A)
        cf_n2 = ax2.contourf(Delta_deg, Ay_g, N2, levels=20, cmap='RdYlGn')
        cb_n2 = fig.colorbar(cf_n2, ax=ax2, orientation='horizontal', pad=0.15)
        if is_en:
            cb_n2.set_label("$n_2(\\delta, a_y) = \\omega_n^2 = \\det(A)$ [rad²/s²] (Green: Stable, Red: Unstable)", fontsize=7.5)
            ax2.set_title("2. Dynamic Stiffness & Stability Field $n_2(\\delta, a_y) = \\det(A)$", fontsize=8.5, fontweight='bold')
            ax2.set_xlabel("Steering Angle (RW) $\\delta$ / HW $\\delta_{sw}$ [deg]", fontsize=8)
            ax2.set_ylabel("Lateral Acceleration $a_y$ [g]", fontsize=8)
        else:
            cb_n2.set_label("n₂(δ, ay) = ωₙ² = det(A) [rad²/s²] (Verde: Stabile, Rosso: Instabile)", fontsize=7.5)
            ax2.set_title("2. Rigidezza Dinamica Eq. n₂(δ, ay) = det(A)", fontsize=8.5, fontweight='bold')
            ax2.set_xlabel("Angolo Sterzo (RW) δ / Volante (HW) δ_sw [deg]", fontsize=8)
            ax2.set_ylabel("Acc. Laterale ay [g]", fontsize=8)
        ax2.contour(Delta_deg, Ay_g, N2, levels=[0.0], colors='black', linewidths=1.8, linestyles='--')
        ax2.grid(True, linestyle=':', alpha=0.4)

        # 3. Subplot (223): Sideslip Forcing Map F_beta (n3, n4)
        N3 = dna_data['N3']
        cf_n3 = ax3.contourf(Delta_deg, Ay_g, N3, levels=15, cmap='viridis')
        cb_n3 = fig.colorbar(cf_n3, ax=ax3, orientation='horizontal', pad=0.15)
        if is_en:
            cb_n3.set_label("$n_3(\\delta, a_y)$ [1/deg] - Static Gain Sideslip $\\beta$", fontsize=7.5)
            ax3.set_title("3. Sideslip Forcing Map $F_\\beta (n_3, n_4)$", fontsize=8.5, fontweight='bold')
            ax3.set_xlabel("Steering Angle (RW) $\\delta$ / HW $\\delta_{sw}$ [deg]", fontsize=8)
            ax3.set_ylabel("Lateral Acceleration $a_y$ [g]", fontsize=8)
        else:
            cb_n3.set_label("n₃(δ, ay) [1/deg] - Guadagno Statico Deriva β", fontsize=7.5)
            ax3.set_title("3. Mappa Forzamento Deriva F_β (n₃, n₄)", fontsize=8.5, fontweight='bold')
            ax3.set_xlabel("Angolo Sterzo (RW) δ / Volante (HW) δ_sw [deg]", fontsize=8)
            ax3.set_ylabel("Acc. Laterale ay [g]", fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.4)

        # 4. Subplot (224): Yaw Forcing Map F_rho (n5, n6)
        N5 = dna_data['N5']
        cf_n5 = ax4.contourf(Delta_deg, Ay_g, N5, levels=15, cmap='plasma')
        cb_n5 = fig.colorbar(cf_n5, ax=ax4, orientation='horizontal', pad=0.15)
        if is_en:
            cb_n5.set_label("$n_5(\\delta, a_y)$ [1/(m·deg)] - Static Gain Curvature $\\rho$", fontsize=7.5)
            ax4.set_title("4. Yaw Forcing Map $F_\\rho (n_5, n_6)$", fontsize=8.5, fontweight='bold')
            ax4.set_xlabel("Steering Angle (RW) $\\delta$ / HW $\\delta_{sw}$ [deg]", fontsize=8)
            ax4.set_ylabel("Lateral Acceleration $a_y$ [g]", fontsize=8)
        else:
            cb_n5.set_label("n₅(δ, ay) [1/(m·deg)] - Guadagno Statico Curvatura ρ", fontsize=7.5)
            ax4.set_title("4. Mappa Forzamento Imbardata F_ρ (n₅, n₆)", fontsize=8.5, fontweight='bold')
            ax4.set_xlabel("Angolo Sterzo (RW) δ / Volante (HW) δ_sw [deg]", fontsize=8)
            ax4.set_ylabel("Acc. Laterale ay [g]", fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.4)

    else:
        # View 2: Understeer Gradient & Stability Threshold
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        K_map = dna_data['K_intrinsic_degg']
        N2 = dna_data['N2']
        Rho_y = dna_data['Rho_y']
        u_a = dna_data['v_x']

        # Subplot 1 (121): 2D Understeer Map K(delta, ay) = -L * rho_y
        K_clipped = np.clip(K_map, -12.0, 12.0)
        cf_kus = ax1.contourf(Delta_deg, Ay_g, K_clipped, levels=25, cmap='coolwarm')
        cb_kus = fig.colorbar(cf_kus, ax=ax1, orientation='horizontal', pad=0.12)
        if is_en:
            cb_kus.set_label("Understeer Gradient $K(\\delta, a_y) = -L\\cdot\\partial\\rho/\\partial a_y$ [°/g] (Blue: Understeer, Red: Oversteer)", fontsize=8)
            cs_zero = ax1.contour(Delta_deg, Ay_g, K_map, levels=[0.0], colors='black', linewidths=2.2, linestyles='--')
            ax1.clabel(cs_zero, inline=True, fontsize=8, fmt='K=0 (Neutral)')
            ax1.contour(Delta_deg, Ay_g, N2, levels=[0.0], colors='#d63031', linewidths=2.5, linestyles='-')
            ax1.set_title("2D Understeer Gradient Map $K(\\delta, a_y) = -L \\cdot \\partial\\rho/\\partial a_y$\n(Solid Red: Stability Boundary $n_2=0$)", fontsize=9.0, fontweight='bold')
            ax1.set_xlabel("Steering Angle (RW) $\\delta$ / HW $\\delta_{sw}$ [deg]", fontsize=8.5)
            ax1.set_ylabel("Lateral Acceleration $a_y$ [g]", fontsize=8.5)
        else:
            cb_kus.set_label("Gradiente Sottosterzo K(δ, ay) = -L·∂ρ/∂ay [°/g] (Blu: Sottosterzo, Rosso: Sovrasterzo)", fontsize=8)
            cs_zero = ax1.contour(Delta_deg, Ay_g, K_map, levels=[0.0], colors='black', linewidths=2.2, linestyles='--')
            ax1.clabel(cs_zero, inline=True, fontsize=8, fmt='K=0 (Neutro)')
            ax1.contour(Delta_deg, Ay_g, N2, levels=[0.0], colors='#d63031', linewidths=2.5, linestyles='-')
            ax1.set_title("Mappa 2D Gradiente di Sottosterzo K(δ, ay) = -L · ∂ρ/∂ay\n(Rosso Continuo: Limite Stabilità n₂=0)", fontsize=9.0, fontweight='bold')
            ax1.set_xlabel("Angolo Sterzo (RW) δ / Volante (HW) δ_sw [deg]", fontsize=8.5)
            ax1.set_ylabel("Accelerazione Laterale ay [g]", fontsize=8.5)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # Subplot 2 (122): Curvature Gradient rho_y vs ay compared to neutral threshold (rho_y=0) and stability threshold (1/u^2)
        ay_col = Ay_g[:, 0]
        rho_y_mid = np.mean(Rho_y[:, 5:15], axis=1)
        thresh_instability = 1.0 / (u_a**2)

        if is_en:
            ax2.plot(ay_col, rho_y_mid, 'b-o', linewidth=2.2, label='$\\partial\\rho/\\partial a_y$ (Effective Curvature Gradient = $-K/L$)')
            ax2.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Neutral Threshold $\\partial\\rho/\\partial a_y = 0$ ($K = 0$)')
            ax2.axhline(thresh_instability, color='#d63031', linestyle='-.', linewidth=2.0, label=f'Instability Threshold $1/u^2 = {thresh_instability:.4f}$')
            ax2.fill_between(ay_col, rho_y_mid, 0.0, where=(rho_y_mid < 0.0),
                             color='#74b9ff', alpha=0.25, label='Understeering Region ($\\partial\\rho/\\partial a_y < 0, K > 0$)')
            ax2.fill_between(ay_col, rho_y_mid, 0.0, where=(rho_y_mid >= 0.0),
                             color='#ff7675', alpha=0.25, label='Oversteering Region ($\\partial\\rho/\\partial a_y > 0, K < 0$)')
            ax2.set_title(f"Curvature Gradient $\\partial\\rho/\\partial a_y$ & Dynamic Stability Threshold\n($v_x = {u_a:.1f}$ m/s, Neutral = 0, Instability Threshold $1/u^2 = {thresh_instability:.4f}$)", fontsize=9.0, fontweight='bold')
            ax2.set_xlabel("Lateral Acceleration $a_y$ [g]", fontsize=8.5)
            ax2.set_ylabel("Curvature Gradient $\\partial\\rho/\\partial a_y$ [1/(m·m/s²)]", fontsize=8.5)
        else:
            ax2.plot(ay_col, rho_y_mid, 'b-o', linewidth=2.2, label='∂ρ/∂ay (Gradiente Curvatura Effettivo = -K/L)')
            ax2.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Soglia Neutra ∂ρ/∂ay = 0 (K = 0)')
            ax2.axhline(thresh_instability, color='#d63031', linestyle='-.', linewidth=2.0, label=f'Soglia Instabilità 1/u² = {thresh_instability:.4f}')
            ax2.fill_between(ay_col, rho_y_mid, 0.0, where=(rho_y_mid < 0.0),
                             color='#74b9ff', alpha=0.25, label='Regione Sottosterzante (∂ρ/∂ay < 0, K > 0)')
            ax2.fill_between(ay_col, rho_y_mid, 0.0, where=(rho_y_mid >= 0.0),
                             color='#ff7675', alpha=0.25, label='Regione Sovrasterzante (∂ρ/∂ay > 0, K < 0)')
            ax2.set_title(f"Gradiente di Curvatura ∂ρ/∂ay & Soglia Stabilità Dinamica\n(v_x = {u_a:.1f} m/s, Soglia Neutra = 0, Soglia Instabilità 1/u² = {thresh_instability:.4f})", fontsize=9.0, fontweight='bold')
            ax2.set_xlabel("Accelerazione Laterale ay [g]", fontsize=8.5)
            ax2.set_ylabel("Gradiente di Curvatura ∂ρ/∂ay [1/(m·m/s²)]", fontsize=8.5)

        ax2.grid(True, linestyle=':', alpha=0.6)
        leg2 = ax2.legend(fontsize='x-small', loc='best')
        if leg2:
            leg2.set_draggable(True)

    fig.tight_layout()
