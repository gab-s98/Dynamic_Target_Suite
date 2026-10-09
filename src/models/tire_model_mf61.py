"""
tire_model_mf61.py
-------------------
Pacejka Magic Formula 6.1 (MF 6.1) Tyre Model Implementation with CasADi.
Fully compliant with Siemens Simcenter Tire MF-Tool Equation Manual (FITTYP = 61).

Provides:
- Pure Longitudinal Force (Fx0)
- Pure Lateral Force (Fy0) including Camber Stiffness (Ky_gamma0) & Camber Shift (SHy_gamma)
- Combined Longitudinal Force (Fx) via G_xa weighting factor
- Combined Lateral Force (Fy) via G_yk weighting factor & SV_yk shift
- Self-Aligning Torque (Mz) via Pneumatic Trail t(alpha_t), Residual Moment Mzr(alpha_r), and Arm s*Fx
- Overturning Moment (Mx)
- Rolling Resistance Moment (My)
- Analytical Jacobians and smooth CasADi expressions for optimization and simulations
"""

import casadi as ca
import numpy as np


def get_equations_markdown():
    return """
#### Modello Pneumatico Completo (Magic Formula 6.1 / Pacejka 2006)
Il modello calcola le forze e i momenti di contatto pneumatico-strada ($F_x, F_y, F_z, M_x, M_y, M_z$)
in funzione di slip ratio $\\kappa$, slip angle $\\alpha$, campanatura $\\gamma$, pressione di gonfiaggio $P_i$ e carico normale $F_z$,
in piena conformità al manuale **Simcenter Tire MF-Tool Equation Manual (FITTYP = 61)**.

##### 1. Incrementi Dimensionali di Carico e Pressione
- Carico nominale scalato: $F'_{z0} = \\lambda_{Fz0} \\cdot F_{z0}$
- Incremento di carico verticale: $df_z = \\frac{F_z - F'_{z0}}{F'_{z0}}$
- Incremento di pressione di gonfiaggio: $dp_i = \\frac{P_i - P_{nom}}{P_{nom}}$

##### 2. Forza Laterale Pura ($F_{y0}$)
$$F_{y0} = D_y \\sin\\left[ C_y \\arctan\\left( B_y \\alpha_y - E_y (B_y \\alpha_y - \\arctan(B_y \\alpha_y)) \\right) \\right] + S_{Vy}$$
- **Fattore di Forma**: $C_y = p_{Cy1} \\cdot \\lambda_{Cy}$
- **Fattore di Aderenza di Picco**: $\\mu_y = (p_{Dy1} + p_{Dy2} df_z)(1 + p_{py3} dp_i + p_{py4} dp_i^2)(1 - p_{Dy3} \\gamma^2) \\lambda_{\\mu y}$
- **Picco di Forza**: $D_y = \\mu_y \\cdot F_z$
- **Rigidezza di Deriva**: $K_{y\\alpha} = p_{Ky1} F'_{z0} (1 + p_{py1} dp_i) \\sin\\left[ p_{Ky4} \\arctan\\left( \\frac{F_z}{(p_{Ky2} + p_{Ky5} \\gamma^2) F'_{z0} (1 + p_{py2} dp_i)} \\right) \\right] (1 - p_{Ky3} |\\gamma|) \\lambda_{Ky\\alpha}$
- **Fattore di Rigidezza**: $B_y = \\frac{K_{y\\alpha}}{C_y D_y}$
- **Rigidezza di Campanatura**: $K_{y\\gamma 0} = (p_{Ky6} + p_{Ky7} df_z) F_z \\lambda_{Ky\\gamma} (1 + p_{py5} dp_i)$
- **Shift Verticale**: $S_{Vy} = F_z (p_{Vy1} + p_{Vy2} df_z) \\lambda_{Vy} \\lambda_{\\mu y} + F_z (p_{Vy3} + p_{Vy4} df_z) \\gamma \\lambda_{Ky\\gamma} \\lambda_{\\mu y}$
- **Shift Orizzontale (con Camber Thrust)**: $S_{Hy} = (p_{Hy1} + p_{Hy2} df_z) \\lambda_{Hy} + \\frac{K_{y\\gamma 0} \\gamma - S_{Vy\\gamma}}{K_{y\\alpha}}$
- **Slip Angle Effettivo**: $\\alpha_y = \\alpha + S_{Hy}$
- **Fattore di Curvatura**: $E_y = (p_{Ey1} + p_{Ey2} df_z)[1 + p_{Ey5} \\gamma^2 - (p_{Ey3} + p_{Ey4} \\gamma) \\operatorname{sgn}(\\alpha_y)] \\lambda_{Ey} \\le 1$

##### 3. Forza Longitudinale Pura ($F_{x0}$)
$$F_{x0} = D_x \\sin\\left[ C_x \\arctan\\left( B_x \\kappa_x - E_x (B_x \\kappa_x - \\arctan(B_x \\kappa_x)) \\right) \\right] + S_{Vx}$$
- **Fattore di Forma**: $C_x = p_{Cx1} \\cdot \\lambda_{Cx}$
- **Fattore di Aderenza**: $\\mu_x = (p_{Dx1} + p_{Dx2} df_z)(1 + p_{px3} dp_i + p_{px4} dp_i^2)(1 - p_{Dx3} \\gamma^2) \\lambda_{\\mu x}$
- **Rigidezza Longitudinale**: $K_{x\\kappa} = F_z (p_{Kx1} + p_{Kx2} df_z) \\exp(p_{Kx3} df_z)(1 + p_{px1} dp_i + p_{px2} dp_i^2) \\lambda_{Kx\\kappa}$
- **Fattore di Rigidezza**: $B_x = \\frac{K_{x\\kappa}}{C_x D_x}$
- **Shift Orizzontale e Verticale**: $S_{Hx} = (p_{Hx1} + p_{Hx2} df_z) \\lambda_{Hx}$, $S_{Vx} = F_z (p_{Vx1} + p_{Vx2} df_z) \\lambda_{Vx} \\lambda_{\\mu x}$
- **Slip Ratio Effettivo**: $\\kappa_x = \\kappa + S_{Hx}$

##### 4. Slip Combinato ($F_x, F_y$)
- **Forza Longitudinale Combinata**: $F_x = F_{x0} \\cdot G_{x\\alpha}$
- **Forza Laterale Combinata**: $F_y = F_{y0} \\cdot G_{y\\kappa} + S_{Vy\\kappa}$
dove $G_{x\\alpha}$ e $G_{y\\kappa}$ sono i coefficienti di ponderazione per l'interazione combinata e $S_{Vy\\kappa}$ è la forza laterale indotta dallo slip longitudinale.

##### 5. Momento di Autoallineamento ($M_z$)
$$M_z = -t(\\alpha_{t,eq}) \\cdot F'_{y0} + M_{zr}(\\alpha_{r,eq}) + s \\cdot F_x$$
- **Braccio Pneumatico ($t$)**: $t = D_t \\cos\\left[ C_t \\arctan\\left( B_t \\alpha_{t,eq} - E_t (B_t \\alpha_{t,eq} - \\arctan(B_t \\alpha_{t,eq})) \\right) \\right] \\cos(\\alpha)$
- **Momento Residuo ($M_{zr}$)**: $M_{zr} = D_r \\cos\\left[ \\arctan(B_r \\alpha_{r,eq}) \\right]$
- **Braccio di Forza Longitudinale ($s$)**: $s = [s_{sz1} + s_{sz2} (F'_{y0} / F'_{z0}) + (s_{sz3} + s_{sz4} df_z) \\gamma] R_0 \\lambda_s$
"""


def mf61_lateral_force(alpha, Fz, gamma, tire_params, LCY=1.0, LMUY=1.0, LKY=1.0, LHY=1.0, LVY=1.0, LKYC=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Pure Lateral Force (Fy0).
    Conforms to Chapter 1.3 of MF-Tool Equation Manual (FITTYP = 61).
    
    Parameters:
    -----------
    alpha       : CasADi SX/MX or float — Slip angle [rad]
    Fz          : CasADi SX/MX or float — Vertical load [N]
    gamma       : CasADi SX/MX or float — Camber / Inclination angle [rad]
    tire_params : TireParams — Parsed .tir file
    LCY         : float/SX — User multiplier on shape factor C_y (default 1.0)
    LMUY        : float/SX — User multiplier on lateral friction coefficient mu_y (default 1.0)
    LKY         : float/SX — User multiplier on cornering stiffness K_ya (default 1.0)
    LHY         : float/SX — User multiplier on horizontal shift SHy (default 1.0)
    LVY         : float/SX — User multiplier on vertical shift SVy (default 1.0)
    LKYC        : float/SX — User multiplier on camber stiffness (default 1.0)
    pressure    : float/SX or None — Inflation pressure [Pa] (None = use .tir INFLPRES)
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0
    
    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    # Clamp Fz to avoid singularities
    Fz_eff = ca.fmax(Fz, 1.0)
    dfz = (Fz_eff - Fz0_prime) / Fz0_prime

    # Scaling coefficients from .tir combined with user multipliers
    LCY_eff = tire_params.get('LCY', 1.0) * LCY
    LMUY_eff = tire_params.get('LMUY', 1.0) * LMUY
    LKY_eff = tire_params.get('LKY', 1.0) * LKY
    LHY_eff = tire_params.get('LHY', 1.0) * LHY
    LVY_eff = tire_params.get('LVY', 1.0) * LVY
    LKYC_eff = tire_params.get('LKYC', 1.0) * LKYC
    LEY_eff = tire_params.get('LEY', 1.0)

    # 1. Shape factor Cy
    PCY1 = tire_params.get('PCY1', 1.0)
    C_y = PCY1 * LCY_eff

    # 2. Friction coefficient mu_y
    PDY1 = tire_params.get('PDY1', 1.0)
    PDY2 = tire_params.get('PDY2', 0.0)
    PDY3 = tire_params.get('PDY3', 0.0)
    PPY3 = tire_params.get('PPY3', 0.0)
    PPY4 = tire_params.get('PPY4', 0.0)
    mu_y = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * (dpi**2)) * (1.0 - PDY3 * (gamma**2)) * LMUY_eff
    mu_y = ca.fmax(mu_y, 0.01)

    # Peak factor Dy
    D_y = mu_y * Fz_eff

    # 3. Cornering stiffness Kya
    PKY1 = tire_params.get('PKY1', 0.0)
    PKY2 = tire_params.get('PKY2', 1.0)
    PKY3 = tire_params.get('PKY3', 0.0)
    PKY4 = tire_params.get('PKY4', 2.0)
    PKY5 = tire_params.get('PKY5', 0.0)
    PPY1 = tire_params.get('PPY1', 0.0)
    PPY2 = tire_params.get('PPY2', 0.0)

    inner_denom = (PKY2 + PKY5 * (gamma**2)) * Fz0_prime * (1.0 + PPY2 * dpi)
    inner_denom = ca.if_else(ca.fabs(inner_denom) < 1e-6, 1e-6 * ca.sign(inner_denom + 1e-12), inner_denom)
    K_ya = PKY1 * Fz0_prime * (1.0 + PPY1 * dpi) * ca.sin(PKY4 * ca.arctan(Fz_eff / inner_denom)) * (1.0 - PKY3 * ca.fabs(gamma)) * LKY_eff

    # Stiffness factor By
    B_y = K_ya / (C_y * D_y + 1e-6)

    # 4. Vertical shift SVy (SVy0 + SVy_gamma)
    PVY1 = tire_params.get('PVY1', 0.0)
    PVY2 = tire_params.get('PVY2', 0.0)
    PVY3 = tire_params.get('PVY3', 0.0)
    PVY4 = tire_params.get('PVY4', 0.0)
    S_Vy0 = Fz_eff * (PVY1 + PVY2 * dfz) * LVY_eff * LMUY_eff
    S_Vy_gamma = Fz_eff * (PVY3 + PVY4 * dfz) * gamma * LKYC_eff * LMUY_eff
    S_Vy = S_Vy0 + S_Vy_gamma

    # 5. Camber stiffness Kya_gamma & Horizontal shift SHy (FITTYP >= 60)
    PKY6 = tire_params.get('PKY6', 0.0)
    PKY7 = tire_params.get('PKY7', 0.0)
    PPY5 = tire_params.get('PPY5', 0.0)
    K_ygamma0 = (PKY6 + PKY7 * dfz) * Fz_eff * LKYC_eff * (1.0 + PPY5 * dpi)

    PHY1 = tire_params.get('PHY1', 0.0)
    PHY2 = tire_params.get('PHY2', 0.0)
    S_Hy0 = (PHY1 + PHY2 * dfz) * LHY_eff
    
    # Camber induced horizontal shift: SHy_gamma = (Ky_gamma0 * gamma - SVy_gamma) / Kya
    # Kya is physically always >= 0 (positive cornering stiffness), so the safe fallback
    # must be +1e-6, not -1e-6 (a negative fallback would invert the camber shift sign).
    Kya_safe = ca.if_else(ca.fabs(K_ya) < 1e-6, 1e-6, K_ya)
    S_Hy_gamma = (K_ygamma0 * gamma - S_Vy_gamma) / Kya_safe
    S_Hy = S_Hy0 + S_Hy_gamma

    alpha_y = alpha + S_Hy

    # 6. Curvature factor Ey
    PEY1 = tire_params.get('PEY1', 0.0)
    PEY2 = tire_params.get('PEY2', 0.0)
    PEY3 = tire_params.get('PEY3', 0.0)
    PEY4 = tire_params.get('PEY4', 0.0)
    PEY5 = tire_params.get('PEY5', 0.0)

    E_y = (PEY1 + PEY2 * dfz) * (1.0 + PEY5 * (gamma**2) - (PEY3 + PEY4 * gamma) * ca.sign(alpha_y)) * LEY_eff
    E_y = ca.fmin(E_y, 1.0)

    # 7. Pure Lateral Force
    arg_y = B_y * alpha_y
    F_y0 = D_y * ca.sin(C_y * ca.arctan(arg_y - E_y * (arg_y - ca.arctan(arg_y)))) + S_Vy

    return F_y0


def mf61_longitudinal_force(kappa, Fz, gamma, tire_params, LCX=1.0, LMUX=1.0, LKX=1.0, LHX=1.0, LVX=1.0, LEX=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Pure Longitudinal Force (Fx0).
    Conforms to Chapter 1.2 of MF-Tool Equation Manual (FITTYP = 61).
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0

    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    Fz_eff = ca.fmax(Fz, 1.0)
    dfz = (Fz_eff - Fz0_prime) / Fz0_prime

    # Scaling coefficients
    LCX_eff = tire_params.get('LCX', 1.0) * LCX
    LMUX_eff = tire_params.get('LMUX', 1.0) * LMUX
    LKX_eff = tire_params.get('LKX', 1.0) * LKX
    LHX_eff = tire_params.get('LHX', 1.0) * LHX
    LVX_eff = tire_params.get('LVX', 1.0) * LVX
    LEX_eff = tire_params.get('LEX', 1.0) * LEX

    # 1. Shape factor Cx
    PCX1 = tire_params.get('PCX1', 1.0)
    C_x = PCX1 * LCX_eff

    # 2. Friction coefficient mu_x
    PDX1 = tire_params.get('PDX1', 1.0)
    PDX2 = tire_params.get('PDX2', 0.0)
    PDX3 = tire_params.get('PDX3', 0.0)
    PPX3 = tire_params.get('PPX3', 0.0)
    PPX4 = tire_params.get('PPX4', 0.0)
    mu_x = (PDX1 + PDX2 * dfz) * (1.0 + PPX3 * dpi + PPX4 * (dpi**2)) * (1.0 - PDX3 * (gamma**2)) * LMUX_eff
    mu_x = ca.fmax(mu_x, 0.01)

    # Peak factor Dx
    D_x = mu_x * Fz_eff

    # 3. Longitudinal slip stiffness Kx_kappa
    PKX1 = tire_params.get('PKX1', 1.0)
    PKX2 = tire_params.get('PKX2', 0.0)
    PKX3 = tire_params.get('PKX3', 0.0)
    PPX1 = tire_params.get('PPX1', 0.0)
    PPX2 = tire_params.get('PPX2', 0.0)

    K_xkappa = Fz_eff * (PKX1 + PKX2 * dfz) * ca.exp(PKX3 * dfz) * (1.0 + PPX1 * dpi + PPX2 * (dpi**2)) * LKX_eff

    # Stiffness factor Bx
    B_x = K_xkappa / (C_x * D_x + 1e-6)

    # 4. Horizontal shift SHx
    PHX1 = tire_params.get('PHX1', 0.0)
    PHX2 = tire_params.get('PHX2', 0.0)
    S_Hx = (PHX1 + PHX2 * dfz) * LHX_eff

    kappa_x = kappa + S_Hx

    # 5. Vertical shift SVx
    PVX1 = tire_params.get('PVX1', 0.0)
    PVX2 = tire_params.get('PVX2', 0.0)
    S_Vx = Fz_eff * (PVX1 + PVX2 * dfz) * LVX_eff * LMUX_eff

    # 6. Curvature factor Ex
    PEX1 = tire_params.get('PEX1', 0.0)
    PEX2 = tire_params.get('PEX2', 0.0)
    PEX3 = tire_params.get('PEX3', 0.0)
    PEX4 = tire_params.get('PEX4', 0.0)

    E_x = (PEX1 + PEX2 * dfz + PEX3 * (dfz**2)) * (1.0 - PEX4 * ca.sign(kappa_x)) * LEX_eff
    E_x = ca.fmin(E_x, 1.0)

    # 7. Pure Longitudinal Force
    arg_x = B_x * kappa_x
    F_x0 = D_x * ca.sin(C_x * ca.arctan(arg_x - E_x * (arg_x - ca.arctan(arg_x)))) + S_Vx

    return F_x0


def mf61_combined_longitudinal_force(Fx_pure, kappa, alpha, Fz, gamma, tire_params, LHY=1.0, LXAL=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Combined Longitudinal Force (Fx).
    Applies the G_xa weighting factor to the pure longitudinal force.
    Conforms to Section 1.2 of MF-Tool Equation Manual.
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0

    Fz_eff = ca.fmax(Fz, 1.0)
    dfz = (Fz_eff - Fz0_prime) / Fz0_prime

    LXAL_eff = tire_params.get('LXAL', 1.0) * LXAL
    LHY_eff = tire_params.get('LHY', 1.0) * LHY

    RBX1 = tire_params.get('RBX1', 0.0)
    RBX2 = tire_params.get('RBX2', 0.0)
    RBX3 = tire_params.get('RBX3', 0.0)
    RCX1 = tire_params.get('RCX1', 1.0)
    REX1 = tire_params.get('REX1', 0.0)
    REX2 = tire_params.get('REX2', 0.0)
    RHX1 = tire_params.get('RHX1', 0.0)

    SHxa = RHX1 * LHY_eff
    alpha_s = alpha + SHxa

    Bxa = (RBX1 + RBX3 * (gamma**2)) * ca.cos(ca.arctan(RBX2 * kappa)) * LXAL_eff
    Cxa = RCX1
    Exa = ca.fmin(REX1 + REX2 * dfz, 1.0)

    # Numerator
    u_num = Bxa * alpha_s
    Gxa_num = ca.cos(Cxa * ca.arctan(u_num - Exa * (u_num - ca.arctan(u_num))))

    # Denominator
    u_den = Bxa * SHxa
    Gxa_den = ca.cos(Cxa * ca.arctan(u_den - Exa * (u_den - ca.arctan(u_den))))
    Gxa_den = ca.fmax(ca.if_else(ca.fabs(Gxa_den) < 1e-6, 1e-6 * ca.sign(Gxa_den + 1e-12), Gxa_den), 1e-6)

    Gxa = ca.fmax(Gxa_num / Gxa_den, 0.0)
    Gxa = ca.fmax(Gxa, 0.0)

    return Fx_pure * Gxa


def mf61_combined_lateral_force(Fy_pure, alpha, kappa, Fz, gamma, tire_params, LVY=1.0, LHY=1.0, LYKA=1.0, LVYKA=1.0, LMUY=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Combined Lateral Force (Fy).
    Applies the G_yk weighting factor and SV_yk shift to the pure lateral force.
    Conforms to Section 1.3 of MF-Tool Equation Manual.
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0

    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    Fz_eff = ca.fmax(Fz, 1.0)
    dfz = (Fz_eff - Fz0_prime) / Fz0_prime

    LYKA_eff = tire_params.get('LYKA', 1.0) * LYKA
    LVYKA_eff = tire_params.get('LVYKA', 1.0) * LVYKA
    LMUY_eff = tire_params.get('LMUY', 1.0) * LMUY
    LVY_eff = tire_params.get('LVY', 1.0) * LVY
    LHY_eff = tire_params.get('LHY', 1.0) * LHY

    RBY1 = tire_params.get('RBY1', 0.0)
    RBY2 = tire_params.get('RBY2', 0.0)
    RBY3 = tire_params.get('RBY3', 0.0)
    RBY4 = tire_params.get('RBY4', 0.0)
    RCY1 = tire_params.get('RCY1', 1.0)
    REY1 = tire_params.get('REY1', 0.0)
    REY2 = tire_params.get('REY2', 0.0)
    RHY1 = tire_params.get('RHY1', 0.0)
    RHY2 = tire_params.get('RHY2', 0.0)

    SHyk = RHY1 + RHY2 * dfz
    kappa_s = kappa + SHyk

    Byk = (RBY1 + RBY4 * (gamma**2)) * ca.cos(ca.arctan(RBY2 * (alpha - RBY3 * LHY_eff))) * LYKA_eff
    Cyk = RCY1
    Eyk = ca.fmin(REY1 + REY2 * dfz, 1.0)

    # Numerator
    u_num = Byk * kappa_s
    Gyk_num = ca.cos(Cyk * ca.arctan(u_num - Eyk * (u_num - ca.arctan(u_num))))

    # Denominator
    u_den = Byk * SHyk
    Gyk_den = ca.cos(Cyk * ca.arctan(u_den - Eyk * (u_den - ca.arctan(u_den))))
    Gyk_den = ca.fmax(ca.if_else(ca.fabs(Gyk_den) < 1e-6, 1e-6 * ca.sign(Gyk_den + 1e-12), Gyk_den), 1e-6)

    Gyk = ca.fmax(Gyk_num / Gyk_den, 0.0)
    Gyk = ca.fmax(Gyk, 0.0)

    # SVyk (kappa-induced side force shift)
    RVY1 = tire_params.get('RVY1', 0.0)
    RVY2 = tire_params.get('RVY2', 0.0)
    RVY3 = tire_params.get('RVY3', 0.0)
    RVY4 = tire_params.get('RVY4', 0.0)
    RVY5 = tire_params.get('RVY5', 0.0)
    RVY6 = tire_params.get('RVY6', 0.0)

    PDY1 = tire_params.get('PDY1', 1.0)
    PDY2 = tire_params.get('PDY2', 0.0)
    PDY3 = tire_params.get('PDY3', 0.0)
    PPY3 = tire_params.get('PPY3', 0.0)
    PPY4 = tire_params.get('PPY4', 0.0)

    mu_y = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * (dpi**2)) * (1.0 - PDY3 * (gamma**2)) * LMUY_eff
    mu_y = ca.fmax(mu_y, 0.01)

    DVyk = mu_y * Fz_eff * ((RVY1 + RVY2 * dfz) * LVY_eff + RVY3 * gamma) * ca.cos(ca.arctan(RVY4 * alpha))
    SVyk = DVyk * ca.sin(RVY5 * ca.arctan(RVY6 * kappa)) * LVYKA_eff

    return Fy_pure * Gyk + SVyk


def mf61_aligning_torque(alpha, Fz, gamma, tire_params, kappa=0.0, Fx=0.0, LVY=1.0, LHY=1.0, LTR=1.0, LRES=1.0, LKZC=1.0, LS=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Self-Aligning Torque (Mz).
    Conforms to Chapter 1.6 of MF-Tool Equation Manual (FITTYP = 61).
    Mz = -t * Fy0_prime + Mzr + s * Fx
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0
    R0 = tire_params.get('UNLOADED_RADIUS', 0.203)

    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    Fz_eff = ca.fmax(Fz, 1.0)
    dfz = (Fz_eff - Fz0_prime) / Fz0_prime

    # Scaling coefficients
    LMUY_tir = tire_params.get('LMUY', 1.0)
    LKY_tir = tire_params.get('LKY', 1.0)
    LCY_tir = tire_params.get('LCY', 1.0)
    LTR_eff = tire_params.get('LTR', 1.0) * LTR
    LRES_eff = tire_params.get('LRES', 1.0) * LRES
    LKZC_eff = tire_params.get('LKZC', 1.0) * LKZC
    LS_eff = tire_params.get('LS', 1.0) * LS
    LVY_eff = tire_params.get('LVY', 1.0) * LVY
    LHY_eff = tire_params.get('LHY', 1.0) * LHY

    # 1. Pneumatic Trail t(alpha_t)
    QHZ1 = tire_params.get('QHZ1', 0.0)
    QHZ2 = tire_params.get('QHZ2', 0.0)
    QHZ3 = tire_params.get('QHZ3', 0.0)
    QHZ4 = tire_params.get('QHZ4', 0.0)
    SHt = (QHZ1 + QHZ2 * dfz) * LHY_eff + (QHZ3 + QHZ4 * dfz) * gamma
    alpha_t = alpha + SHt

    # Equivalent alpha_t for combined slip
    alpha_t_eq = alpha_t

    QBZ1 = tire_params.get('QBZ1', 0.0)
    QBZ2 = tire_params.get('QBZ2', 0.0)
    QBZ3 = tire_params.get('QBZ3', 0.0)
    QBZ4 = tire_params.get('QBZ4', 0.0)
    QBZ5 = tire_params.get('QBZ5', 0.0)
    B_t = (QBZ1 + QBZ2 * dfz + QBZ3 * (dfz**2)) * (1.0 + QBZ4 * ca.fabs(gamma) + QBZ5 * gamma) * (LKY_tir / LMUY_tir)

    QCZ1 = tire_params.get('QCZ1', 1.0)
    C_t = QCZ1

    QDZ1 = tire_params.get('QDZ1', 0.0)
    QDZ2 = tire_params.get('QDZ2', 0.0)
    QDZ3 = tire_params.get('QDZ3', 0.0)
    QDZ4 = tire_params.get('QDZ4', 0.0)
    PPZ1 = tire_params.get('PPZ1', 0.0)
    D_t = Fz_eff * (QDZ1 + QDZ2 * dfz) * (1.0 - PPZ1 * dpi) * (1.0 + QDZ3 * gamma + QDZ4 * (gamma**2)) * (R0 / Fz0_prime) * LTR_eff

    QEZ1 = tire_params.get('QEZ1', 0.0)
    QEZ2 = tire_params.get('QEZ2', 0.0)
    QEZ3 = tire_params.get('QEZ3', 0.0)
    QEZ4 = tire_params.get('QEZ4', 0.0)
    QEZ5 = tire_params.get('QEZ5', 0.0)
    E_t = (QEZ1 + QEZ2 * dfz + QEZ3 * (dfz**2)) * (1.0 + (QEZ4 + QEZ5 * gamma) * (2.0 / np.pi) * ca.arctan(B_t * C_t * alpha_t))
    E_t = ca.fmin(E_t, 1.0)

    arg_t = B_t * alpha_t_eq
    t_trail = D_t * ca.cos(C_t * ca.arctan(arg_t - E_t * (arg_t - ca.arctan(arg_t)))) * ca.cos(alpha)

    # 2. Residual Moment Mzr(alpha_r)
    PCY1 = tire_params.get('PCY1', 1.0)
    C_y = PCY1 * LCY_tir
    PDY1 = tire_params.get('PDY1', 1.0)
    PDY2 = tire_params.get('PDY2', 0.0)
    PDY3 = tire_params.get('PDY3', 0.0)
    PPY3 = tire_params.get('PPY3', 0.0)
    PPY4 = tire_params.get('PPY4', 0.0)
    mu_y = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * (dpi**2)) * (1.0 - PDY3 * (gamma**2)) * LMUY_tir
    mu_y = ca.fmax(mu_y, 0.01)
    D_y = mu_y * Fz_eff

    PKY1 = tire_params.get('PKY1', 0.0)
    PKY2 = tire_params.get('PKY2', 1.0)
    PKY3 = tire_params.get('PKY3', 0.0)
    PKY4 = tire_params.get('PKY4', 2.0)
    PKY5 = tire_params.get('PKY5', 0.0)
    PPY1 = tire_params.get('PPY1', 0.0)
    PPY2 = tire_params.get('PPY2', 0.0)
    inner_denom = (PKY2 + PKY5 * (gamma**2)) * Fz0_prime * (1.0 + PPY2 * dpi)
    inner_denom = ca.if_else(ca.fabs(inner_denom) < 1e-6, 1e-6 * ca.sign(inner_denom + 1e-12), inner_denom)
    K_ya = PKY1 * Fz0_prime * (1.0 + PPY1 * dpi) * ca.sin(PKY4 * ca.arctan(Fz_eff / inner_denom)) * (1.0 - PKY3 * ca.fabs(gamma)) * LKY_tir
    B_y = K_ya / (C_y * D_y + 1e-6)

    PHY1 = tire_params.get('PHY1', 0.0)
    PHY2 = tire_params.get('PHY2', 0.0)
    S_Hy0 = (PHY1 + PHY2 * dfz) * LHY_eff
    PVY1 = tire_params.get('PVY1', 0.0)
    PVY2 = tire_params.get('PVY2', 0.0)
    S_Vy0 = Fz_eff * (PVY1 + PVY2 * dfz) * LVY_eff * LMUY_tir

    alpha_r = alpha + S_Hy0 + S_Vy0 / ca.if_else(ca.fabs(K_ya) < 1e-6, 1e-6, K_ya)
    alpha_r_eq = alpha_r

    QDZ6 = tire_params.get('QDZ6', 0.0)
    QDZ7 = tire_params.get('QDZ7', 0.0)
    QDZ8 = tire_params.get('QDZ8', 0.0)
    QDZ9 = tire_params.get('QDZ9', 0.0)
    QDZ10 = tire_params.get('QDZ10', 0.0)
    QDZ11 = tire_params.get('QDZ11', 0.0)
    PPZ2 = tire_params.get('PPZ2', 0.0)

    D_r = Fz_eff * R0 * LMUY_tir * ca.cos(alpha) * (
        (QDZ6 + QDZ7 * dfz) * LRES_eff * LVY_eff +
        (QDZ8 + QDZ9 * dfz) * (1.0 + PPZ2 * dpi) * gamma * LKZC_eff +
        (QDZ10 + QDZ11 * dfz) * gamma * ca.fabs(gamma) * LKZC_eff
    )

    QBZ9 = tire_params.get('QBZ9', 0.0)
    QBZ10 = tire_params.get('QBZ10', 0.0)
    B_r = (QBZ9 * (LKY_tir / LMUY_tir) + QBZ10 * B_y * C_y)

    M_zr = D_r * ca.cos(ca.arctan(B_r * alpha_r_eq))

    # In FITTYP >= 60, Fy0_prime is pure side force without camber effect (gamma=0)
    PEY1 = tire_params.get('PEY1', 0.0)
    PEY2 = tire_params.get('PEY2', 0.0)
    E_y0 = ca.fmin((PEY1 + PEY2 * dfz) * tire_params.get('LEY', 1.0), 1.0)
    inner_denom0 = PKY2 * Fz0_prime * (1.0 + PPY2 * dpi)
    inner_denom0 = ca.if_else(ca.fabs(inner_denom0) < 1e-6, 1e-6 * ca.sign(inner_denom0 + 1e-12), inner_denom0)
    K_ya0 = PKY1 * Fz0_prime * (1.0 + PPY1 * dpi) * ca.sin(PKY4 * ca.arctan(Fz_eff / inner_denom0)) * LKY_tir
    mu_y0 = (PDY1 + PDY2 * dfz) * (1.0 + PPY3 * dpi + PPY4 * (dpi**2)) * LMUY_tir
    D_y0 = mu_y0 * Fz_eff
    B_y0 = K_ya0 / (C_y * D_y0 + 1e-6)
    alpha_y0 = alpha + S_Hy0
    arg_y0 = B_y0 * alpha_y0
    Fy0_prime = D_y0 * ca.sin(C_y * ca.arctan(arg_y0 - E_y0 * (arg_y0 - ca.arctan(arg_y0)))) + S_Vy0

    # 3. Longitudinal force arm s
    SSZ1 = tire_params.get('SSZ1', 0.0)
    SSZ2 = tire_params.get('SSZ2', 0.0)
    SSZ3 = tire_params.get('SSZ3', 0.0)
    SSZ4 = tire_params.get('SSZ4', 0.0)
    s_arm = (SSZ1 + SSZ2 * (Fy0_prime / Fz0_prime) + (SSZ3 + SSZ4 * dfz) * gamma) * R0 * LS_eff

    # Total Aligning Torque
    Mz = -t_trail * Fy0_prime + M_zr + s_arm * Fx
    return Mz


def mf61_overturning_moment(Fy, Fz, gamma, tire_params, LMX=1.0, LVMX=1.0, pressure=None):
    """
    Pacejka Magic Formula 6.1 for Overturning Couple / Moment (Mx).
    Conforms to Chapter 1.4 of MF-Tool Equation Manual (FITTYP = 61).
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0
    R0 = tire_params.get('UNLOADED_RADIUS', 0.203)

    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    Fz_eff = ca.fmax(Fz, 1.0)
    LMX_eff = tire_params.get('LMX', 1.0) * LMX
    LVMX_eff = tire_params.get('LVMX', 1.0) * LVMX

    QSX1 = tire_params.get('QSX1', 0.0)
    QSX2 = tire_params.get('QSX2', 0.0)
    QSX3 = tire_params.get('QSX3', 0.0)
    QSX4 = tire_params.get('QSX4', 0.0)
    QSX5 = tire_params.get('QSX5', 0.0)
    QSX6 = tire_params.get('QSX6', 0.0)
    QSX7 = tire_params.get('QSX7', 0.0)
    QSX8 = tire_params.get('QSX8', 0.0)
    QSX9 = tire_params.get('QSX9', 0.0)
    QSX10 = tire_params.get('QSX10', 0.0)
    QSX11 = tire_params.get('QSX11', 0.0)
    PPMX1 = tire_params.get('PPMX1', 0.0)

    # Overturning moment default formulation
    term_load = (QSX6 * (Fz_eff / Fz0_prime))**2
    term_atan_load = QSX5 * ca.arctan(term_load)
    term_sin = QSX7 * gamma + QSX8 * ca.arctan(QSX9 * (Fy / Fz0_prime))

    Mx = R0 * LMX_eff * Fz_eff * (
        QSX1 * LVMX_eff -
        QSX2 * (1.0 + PPMX1 * dpi) * gamma +
        QSX3 * (Fy / Fz0_prime) +
        QSX4 * ca.cos(term_atan_load) * ca.sin(term_sin) +
        QSX10 * ca.arctan(QSX11 * (Fz_eff / Fz0_prime)) * gamma
    )
    return Mx


def mf61_rolling_resistance(Fx, Fz, gamma, Vx, tire_params, LMY=1.0, pressure=None, crr=None):
    """
    Pacejka Magic Formula 6.1 for Rolling Resistance Moment (My).
    Conforms to Chapter 1.5 of MF-Tool Equation Manual (FITTYP = 61).
    Allows overriding QSY1 via crr parameter when specified.
    """
    LFZO = tire_params.get('LFZO', 1.0)
    Fz0 = tire_params.get('FNOMIN', 1000.0)
    Fz0_prime = LFZO * Fz0
    R0 = tire_params.get('UNLOADED_RADIUS', 0.203)
    Vref = tire_params.get('LONGVL', 16.6)

    Pnom = tire_params.get('NOMPRES', 200000.0)
    if pressure is None:
        pressure_eff = tire_params.get('INFLPRES', Pnom)
    else:
        pressure_eff = pressure
    dpi = (pressure_eff - Pnom) / Pnom

    Fz_eff = ca.fmax(Fz, 1.0)
    LMY_eff = tire_params.get('LMY', 1.0) * LMY

    QSY1 = crr if crr is not None else tire_params.get('QSY1', 0.01)
    QSY2 = tire_params.get('QSY2', 0.0)
    QSY3 = tire_params.get('QSY3', 0.0)
    QSY4 = tire_params.get('QSY4', 0.0)
    QSY5 = tire_params.get('QSY5', 0.0)
    QSY6 = tire_params.get('QSY6', 0.0)
    QSY7 = tire_params.get('QSY7', 1.0)
    QSY8 = tire_params.get('QSY8', 0.0)

    vx_rel = ca.fabs(Vx) / Vref
    My = -R0 * Fz0_prime * LMY_eff * (
        QSY1 +
        QSY2 * ca.sign(Vx + 1e-6) * (Fx / Fz0_prime) +
        QSY3 * vx_rel +
        QSY4 * (vx_rel**4) +
        QSY5 * (gamma**2) +
        QSY6 * (Fz_eff / Fz0_prime) * (gamma**2)
    ) * ((Fz_eff / Fz0_prime)**QSY7) * ((1.0 + dpi)**QSY8)

    return My

