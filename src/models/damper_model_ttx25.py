"""
damper_model_ttx25.py
----------------------
Non-linear, digressive 4-way damper model based on Öhlins TTX25 MkII FSAE Dyno test data.
Provides both CasADi symbolic expressions (for optimization & 14-DOF simulation)
and pure NumPy evaluations (for GUI plotting and analysis).

Supports:
- 6 Valving specifications: C11/R11, C12/R12 (baseline), C13/R13, C21/R21, C22/R22, C23/R23
- 4-Way independent adjustments:
    * LSC (Low-Speed Compression): 0 to 25 clicks from fully closed (stiff)
    * HSC (High-Speed Compression): 0 to 4.3 turns from fully open (soft)
    * LSR (Low-Speed Rebound): 0 to 25 clicks from fully closed (stiff)
    * HSR (High-Speed Rebound): 0 to 4.3 turns from fully open (soft)
- Motion ratio transformation: v_damper = MR * v_wheel, F_wheel = MR * F_damper
- Equivalent linear damping extraction & modal damping ratios (zeta)
- Dual-mode fallback to linear damping
"""

import numpy as np
import casadi as ca

# Calibration parameters for each valve specification based on Öhlins Dyno N vs mm/s
# Structure: { valving: { 'comp': {...}, 'reb': {...} } }
# Parameters:
#   cls_max: max low-speed damping coefficient [N*s/m] (at LSC/LSR = 0 clicks, fully closed)
#   cls_min: min low-speed damping coefficient [N*s/m] (at LSC/LSR = 25 clicks, fully open)
#   chs_base: base high-speed damping coefficient [N*s/m] (at HSC/HSR = 4.3 turns open, soft)
#   chs_gain: high-speed stiffness gain per turn closed [N*s/(m*turn)]
#   vknee: knee velocity [m/s] (~0.025 m/s = 25 mm/s)
#   fknee_base: base blow-off knee force [N]
#   fknee_gain: knee force gain per turn of HSC/HSR closed [N/turn]

VALVE_SPECS = {
    'C11/R11': {
        'comp': {'cls_max': 10500.0, 'cls_min': 1500.0, 'chs_base': 850.0,  'chs_gain': 180.0, 'vknee': 0.025, 'fknee_base': 120.0, 'fknee_gain': 45.0},
        'reb':  {'cls_max': 9500.0,  'cls_min': 1400.0, 'chs_base': 750.0,  'chs_gain': 160.0, 'vknee': 0.025, 'fknee_base': 105.0, 'fknee_gain': 40.0}
    },
    'C12/R12': {  # Standard FSAE baseline delivery specification
        'comp': {'cls_max': 14000.0, 'cls_min': 2000.0, 'chs_base': 1150.0, 'chs_gain': 240.0, 'vknee': 0.025, 'fknee_base': 160.0, 'fknee_gain': 60.0},
        'reb':  {'cls_max': 12000.0, 'cls_min': 1800.0, 'chs_base': 1000.0, 'chs_gain': 210.0, 'vknee': 0.025, 'fknee_base': 140.0, 'fknee_gain': 55.0}
    },
    'C13/R13': {
        'comp': {'cls_max': 18000.0, 'cls_min': 2600.0, 'chs_base': 1500.0, 'chs_gain': 310.0, 'vknee': 0.025, 'fknee_base': 210.0, 'fknee_gain': 75.0},
        'reb':  {'cls_max': 15500.0, 'cls_min': 2300.0, 'chs_base': 1300.0, 'chs_gain': 270.0, 'vknee': 0.025, 'fknee_base': 180.0, 'fknee_gain': 68.0}
    },
    'C21/R21': {
        'comp': {'cls_max': 15000.0, 'cls_min': 2200.0, 'chs_base': 1300.0, 'chs_gain': 260.0, 'vknee': 0.022, 'fknee_base': 180.0, 'fknee_gain': 65.0},
        'reb':  {'cls_max': 13000.0, 'cls_min': 1900.0, 'chs_base': 1100.0, 'chs_gain': 230.0, 'vknee': 0.022, 'fknee_base': 155.0, 'fknee_gain': 58.0}
    },
    'C22/R22': {
        'comp': {'cls_max': 20000.0, 'cls_min': 2800.0, 'chs_base': 1750.0, 'chs_gain': 350.0, 'vknee': 0.022, 'fknee_base': 240.0, 'fknee_gain': 85.0},
        'reb':  {'cls_max': 17500.0, 'cls_min': 2500.0, 'chs_base': 1500.0, 'chs_gain': 300.0, 'vknee': 0.022, 'fknee_base': 210.0, 'fknee_gain': 75.0}
    },
    'C23/R23': {
        'comp': {'cls_max': 26000.0, 'cls_min': 3500.0, 'chs_base': 2300.0, 'chs_gain': 450.0, 'vknee': 0.020, 'fknee_base': 310.0, 'fknee_gain': 110.0},
        'reb':  {'cls_max': 22500.0, 'cls_min': 3100.0, 'chs_base': 1950.0, 'chs_gain': 390.0, 'vknee': 0.020, 'fknee_base': 270.0, 'fknee_gain': 95.0}
    },
    'CUSTOM_DYNO': {
        'comp': {'cls_max': 16000.0, 'cls_min': 2200.0, 'chs_base': 1300.0, 'chs_gain': 280.0, 'vknee': 0.025, 'fknee_base': 175.0, 'fknee_gain': 65.0},
        'reb':  {'cls_max': 14000.0, 'cls_min': 2000.0, 'chs_base': 1100.0, 'chs_gain': 230.0, 'vknee': 0.025, 'fknee_base': 150.0, 'fknee_gain': 60.0}
    },
    'Custom Dyno Curve': {
        'comp': {'cls_max': 16000.0, 'cls_min': 2200.0, 'chs_base': 1300.0, 'chs_gain': 280.0, 'vknee': 0.025, 'fknee_base': 175.0, 'fknee_gain': 65.0},
        'reb':  {'cls_max': 14000.0, 'cls_min': 2000.0, 'chs_base': 1100.0, 'chs_gain': 230.0, 'vknee': 0.025, 'fknee_base': 150.0, 'fknee_gain': 60.0}
    }
}


def calculate_damper_force(v_damper, valving='C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, use_casadi=False):
    """
    Calculates the non-linear damper force for a given damper velocity.
    
    Parameters:
    -----------
    v_damper   : float or CasADi SX/MX — Damper shaft velocity [m/s] (positive = bump/compression, negative = rebound)
    valving    : str — Valve specification ('C11/R11' .. 'C23/R23')
    lsc        : float/SX — Low Speed Compression clicks from fully closed (0 = stiff, 25 = soft)
    hsc        : float/SX — High Speed Compression turns from fully open (0 = soft, 4.3 = stiff)
    lsr        : float/SX — Low Speed Rebound clicks from fully closed (0 = stiff, 25 = soft)
    hsr        : float/SX — High Speed Rebound turns from fully open (0 = soft, 4.3 = stiff)
    use_casadi : bool — Force CasADi symbolic math if True
    
    Returns:
    --------
    F_damper   : float or CasADi SX/MX — Damper force [N] (positive = bump resistive force, negative = rebound)
    """
    spec = VALVE_SPECS.get(valving, VALVE_SPECS['C12/R12'])
    p_comp = spec['comp']
    p_reb = spec['reb']

    is_ca = use_casadi or isinstance(v_damper, (ca.SX, ca.MX))
    tau_click = 7.5

    if is_ca:
        lsc_c = ca.fmax(0.0, ca.fmin(25.0, lsc))
        lsr_c = ca.fmax(0.0, ca.fmin(25.0, lsr))
        hsc_t = ca.fmax(0.0, ca.fmin(4.3, hsc))
        hsr_t = ca.fmax(0.0, ca.fmin(4.3, hsr))

        decay_c = ca.exp(-lsc_c / tau_click)
        decay_r = ca.exp(-lsr_c / tau_click)

        cls_c = p_comp['cls_min'] + (p_comp['cls_max'] - p_comp['cls_min']) * decay_c
        cls_r = p_reb['cls_min'] + (p_reb['cls_max'] - p_reb['cls_min']) * decay_r

        chs_c = p_comp['chs_base'] + p_comp['chs_gain'] * hsc_t
        chs_r = p_reb['chs_base'] + p_reb['chs_gain'] * hsr_t

        vk_c = p_comp['vknee']
        vk_r = p_reb['vknee']

        v_c = ca.fmax(v_damper, 0.0)
        u_c = v_c / vk_c
        F_comp = cls_c * vk_c * ca.tanh(u_c) + chs_c * (v_c - vk_c * ca.tanh(u_c))

        v_r = ca.fmin(v_damper, 0.0)
        u_r = v_r / vk_r
        F_reb = cls_r * vk_r * ca.tanh(u_r) + chs_r * (v_r - vk_r * ca.tanh(u_r))

        k_blend = 200.0
        blend = 0.5 * (1.0 + ca.tanh(k_blend * v_damper))
        F_total = blend * F_comp + (1.0 - blend) * F_reb
        return F_total
    else:
        lsc_c = np.clip(float(lsc), 0.0, 25.0)
        lsr_c = np.clip(float(lsr), 0.0, 25.0)
        hsc_t = np.clip(float(hsc), 0.0, 4.3)
        hsr_t = np.clip(float(hsr), 0.0, 4.3)

        decay_c = np.exp(-lsc_c / tau_click)
        decay_r = np.exp(-lsr_c / tau_click)

        cls_c = p_comp['cls_min'] + (p_comp['cls_max'] - p_comp['cls_min']) * decay_c
        cls_r = p_reb['cls_min'] + (p_reb['cls_max'] - p_reb['cls_min']) * decay_r

        chs_c = p_comp['chs_base'] + p_comp['chs_gain'] * hsc_t
        chs_r = p_reb['chs_base'] + p_reb['chs_gain'] * hsr_t

        vk_c = p_comp['vknee']
        vk_r = p_reb['vknee']

        v = np.asarray(v_damper, dtype=float)
        u_c = np.maximum(v, 0.0) / vk_c
        F_comp = cls_c * vk_c * np.tanh(u_c) + chs_c * (np.maximum(v, 0.0) - vk_c * np.tanh(u_c))

        u_r = np.minimum(v, 0.0) / vk_r
        F_reb = cls_r * vk_r * np.tanh(u_r) + chs_r * (np.minimum(v, 0.0) - vk_r * np.tanh(u_r))

        F_total = np.where(v >= 0.0, F_comp, F_reb)
        return float(F_total) if np.ndim(v_damper) == 0 else F_total


def calculate_wheel_damper_force(v_wheel, motion_ratio=0.85, valving='C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, use_casadi=False):
    """
    Calculates the damping force effective at the wheel contact/upright.
    F_wheel = MR * F_damper(MR * v_wheel)
    """
    v_damper = motion_ratio * v_wheel
    f_damper = calculate_damper_force(v_damper, valving, lsc, hsc, lsr, hsr, use_casadi=use_casadi)
    return motion_ratio * f_damper


def get_dyno_curve(valving='C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, num_pts=150, v_max=0.255):
    """
    Generates full Dyno Force vs Velocity data arrays for GUI plotting.
    """
    v_arr = np.linspace(-v_max, v_max, num_pts)
    f_arr = calculate_damper_force(v_arr, valving, lsc, hsc, lsr, hsr, use_casadi=False)
    v_mmps = v_arr * 1000.0
    return {
        'v_mmps': v_mmps,
        'f_N': f_arr,
        'v_ms': v_arr,
        'valving': valving,
        'lsc': lsc, 'hsc': hsc, 'lsr': lsr, 'hsr': hsr
    }


def compute_equivalent_linear_damping(valving='C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, motion_ratio=0.85, v_eval=0.025):
    """
    Extracts equivalent linear wheel damping rates around low-speed chassis velocities (v_eval ~ 25 mm/s).
    """
    f_bump_wheel = calculate_wheel_damper_force(v_eval, motion_ratio, valving, lsc, hsc, lsr, hsr)
    f_reb_wheel = calculate_wheel_damper_force(-v_eval, motion_ratio, valving, lsc, hsc, lsr, hsr)

    c_wheel_bump = f_bump_wheel / v_eval
    c_wheel_reb = abs(f_reb_wheel) / v_eval
    c_wheel_avg = 0.5 * (c_wheel_bump + c_wheel_reb)

    return {
        'c_wheel_bump': float(c_wheel_bump),
        'c_wheel_reb': float(c_wheel_reb),
        'c_wheel_avg': float(c_wheel_avg)
    }


def compute_modal_damping_ratios(m_corner_kg, K_wheel_Npm, valving='C12/R12', lsc=10, hsc=2.0, lsr=10, hsr=2.0, motion_ratio=0.85):
    """
    Computes critical damping coefficient and modal damping ratios (zeta = c / c_crit).
    """
    c_crit = 2.0 * np.sqrt(max(K_wheel_Npm, 1000.0) * max(m_corner_kg, 10.0))
    eq = compute_equivalent_linear_damping(valving, lsc, hsc, lsr, hsr, motion_ratio)

    zeta_bump = eq['c_wheel_bump'] / c_crit
    zeta_reb = eq['c_wheel_reb'] / c_crit
    zeta_avg = eq['c_wheel_avg'] / c_crit

    return {
        'c_crit': float(c_crit),
        'zeta_bump_ls': float(zeta_bump),
        'zeta_reb_ls': float(zeta_reb),
        'zeta_avg_ls': float(zeta_avg),
        'c_wheel_avg': eq['c_wheel_avg']
    }


def get_damper_info_markdown():
    return """
### Öhlins TTX25 MkII FSAE — Guida alla Taratura Ammortizzatori 4-Way

L'ammortizzatore **Öhlins TTX25 MkII** è un'unità a doppio tubo (*twin-tube*) con pistone solido e pressurizzazione interna a gas, specificamente progettato per le competizioni Formula SAE / Student.

#### 1. Regolazioni a 4 Vie (4-Way Adjustability)
- **LSC (Low Speed Compression - Esagono 3mm)**:
  - *Funzione*: Regola l'orifizio di bleed per i movimenti lenti della scocca (0 - 25 mm/s).
  - *Effetto*: Controlla la velocità di rollio in inserimento e il beccheggio in frenata. Valori tipici: **8-14 click**.
- **HSC (High Speed Compression - Esagono 12mm)**:
  - *Funzione*: Regola il precarico della molla sul pacco lamellare di compressione.
  - *Effetto*: Controlla la risposta all'impatto sui cordoli e sulle asperità ad alta velocità (> 50 mm/s). Valori tipici: **1.5 - 3.0 giri**.
- **LSR (Low Speed Rebound - Esagono 3mm)**:
  - *Funzione*: Regola la frenatura idraulica in estensione a bassa velocità.
  - *Effetto*: Fondamentale per controllare la riapertura della sospensione interna in uscita curva e stabilizzare il retrotreno in staccata. Valori tipici: **6-12 click**.
- **HSR (High Speed Rebound - Esagono 12mm)**:
  - *Funzione*: Regola la risposta di estensione rapida post-cordolo.
  - *Effetto*: Evita che la ruota rimbalzi o resti sospesa in aria dopo aver saltato un cordolo. Valori tipici: **1.5 - 3.0 giri**.

#### 2. Regole di Assetto Ottimale (Suspension & Damper Tuning Guidelines)
- **Asimmetria Rebound / Bump**: Tipicamente zeta_reb / zeta_bump ~ 1.5 - 2.0. Il ritorno deve essere più frenato della compressione.
- **Target Damping Ratio (zeta)**:
  - *Low Speed Heave / Roll*: zeta_ls ~ 0.65 - 0.85 (controllo transitorio della scocca).
  - *High Speed Bump*: zeta_hs ~ 0.25 - 0.40 (conformità e grip su asfalto irregolare).
"""

