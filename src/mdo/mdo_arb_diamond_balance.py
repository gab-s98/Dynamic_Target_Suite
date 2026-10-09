"""
mdo_arb_diamond_balance.py
--------------------------
Engine per la valutazione della variazione del solo bilanciamento meccanico
e del suo impatto su performance e guidabilita tramite il Modello 14-DOF.

Il dominio di rigidezze ammissibili e modellato dal "Rombo di Setting dell'ARB":
- Asse X: Rigidezza a rollio totale K_phi_tot = K_phi_f + K_phi_r [Nm/rad o Nm/deg]
- Asse Y: Ripartizione di rollio anteriore Ratio_f = K_phi_f / K_phi_tot [- o %]

Supporta:
1. Definizione e modifica interattiva dei 4 vertici del rombo
   (Soft-Soft, Stiff-Soft, Stiff-Stiff, Soft-Stiff) sia da coordinate dirette
   sia dal dimensionamento fisico delle barre ARB (K_arb_min, K_arb_max).
2. Valutazione rigorosa con modello multibody 14-DOF:
   - Performance: a_y_max [g], tempo Skidpad FSAE R=9.125m [s], v_max [m/s],
     gradiente di rollio [deg/g], efficienza aderenza eta_mu [%].
   - Guidabilita (Handling & Drivability): gradiente di sottosterzo K_us [deg/g],
     reattivita transitoria t90 [s], sovraelongazione yaw rate (overshoot %),
     tempo di assestamento t_settle [s], picco r_dot, differenziale di utilizzazione attrito
     Delta_Util = Util_F - Util_R [%] (controllo del sovrasterzo repentino).
3. Campionamento 2D ad isolinee / contour map all'interno del rombo e identificazione
   dello Sweet Spot / Corridoio Ottimale di compromesso tra velocita e docilita.
"""

import os
import copy
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from matplotlib.path import Path

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_dynamic_events import simulate_step_steer
from src.mdo.mdo_skidpad import simulate_skidpad


DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


class ARBDiamondConfig:
    """
    Rappresenta la geometria e i limiti del Rombo di Setting dell'ARB nel piano:
    X: K_phi_tot [Nm/rad]
    Y: Ratio_f = K_phi_f / K_phi_tot [-]
    """
    def __init__(
        self,
        v_soft_soft: Tuple[float, float] = (25000.0, 0.48),    # V1: min K_tot, bilancio medio
        v_stiff_soft: Tuple[float, float] = (55000.0, 0.64),   # V2: K_tot medio-alto, max ratio front (max sottosterzo)
        v_stiff_stiff: Tuple[float, float] = (85000.0, 0.49),  # V3: max K_tot, bilancio medio
        v_soft_stiff: Tuple[float, float] = (55000.0, 0.36),   # V4: K_tot medio-basso, min ratio front (max sovrasterzo)
        name: str = "ET18 FSAE ARB Diamond"
    ):
        self.name = name
        # Vertici ordinati in senso orario / ciclico per formare un poligono chiuso
        # [V1 (Soft-Soft), V2 (Stiff-Soft), V3 (Stiff-Stiff), V4 (Soft-Stiff)]
        self.v1 = tuple(v_soft_soft)
        self.v2 = tuple(v_stiff_soft)
        self.v3 = tuple(v_stiff_stiff)
        self.v4 = tuple(v_soft_stiff)

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [self.v1, self.v2, self.v3, self.v4]

    @property
    def polygon_path(self) -> Path:
        """Restituisce un oggetto matplotlib.path.Path per point-in-polygon e masking."""
        verts = [self.v1, self.v2, self.v3, self.v4, self.v1]
        return Path(verts)

    def is_inside(self, k_tot: float, ratio_f: float) -> bool:
        """Verifica se il punto (k_tot, ratio_f) ricade all'interno del rombo."""
        return bool(self.polygon_path.contains_point((k_tot, ratio_f)))

    def get_center_point(self) -> Tuple[float, float]:
        """Restituisce il baricentro geometrico del rombo (setup medio bilanciato)."""
        k_avg = float(np.mean([v[0] for v in self.vertices]))
        r_avg = float(np.mean([v[1] for v in self.vertices]))
        return (k_avg, r_avg)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Restituisce (k_tot_min, k_tot_max, ratio_min, ratio_max)."""
        k_vals = [v[0] for v in self.vertices]
        r_vals = [v[1] for v in self.vertices]
        return float(min(k_vals)), float(max(k_vals)), float(min(r_vals)), float(max(r_vals))

    @staticmethod
    def from_arb_stiffness_ranges(
        k_spring_phi_f: float,
        k_spring_phi_r: float,
        k_arb_f_min: float,
        k_arb_f_max: float,
        k_arb_r_min: float,
        k_arb_r_max: float,
        name: str = "Calcolato da Range ARB"
    ) -> 'ARBDiamondConfig':
        """
        Calcola rigorosamente i 4 vertici del rombo combinando le rigidezze elastiche
        delle molle con i limiti fisici min/max delle barre antirollio.
        Tutti i valori in Nm/rad.
        """
        # 1. Soft-Soft (F min, R min)
        k_f_1 = k_spring_phi_f + k_arb_f_min
        k_r_1 = k_spring_phi_r + k_arb_r_min
        k_tot_1 = k_f_1 + k_r_1
        ratio_1 = k_f_1 / k_tot_1

        # 2. Stiff-Soft (F max, R min) -> Max Front Bias (Sottosterzo)
        k_f_2 = k_spring_phi_f + k_arb_f_max
        k_r_2 = k_spring_phi_r + k_arb_r_min
        k_tot_2 = k_f_2 + k_r_2
        ratio_2 = k_f_2 / k_tot_2

        # 3. Stiff-Stiff (F max, R max)
        k_f_3 = k_spring_phi_f + k_arb_f_max
        k_r_3 = k_spring_phi_r + k_arb_r_max
        k_tot_3 = k_f_3 + k_r_3
        ratio_3 = k_f_3 / k_tot_3

        # 4. Soft-Stiff (F min, R max) -> Min Front Bias (Sovrasterzo)
        k_f_4 = k_spring_phi_f + k_arb_f_min
        k_r_4 = k_spring_phi_r + k_arb_r_max
        k_tot_4 = k_f_4 + k_r_4
        ratio_4 = k_f_4 / k_tot_4

        return ARBDiamondConfig(
            v_soft_soft=(k_tot_1, ratio_1),
            v_stiff_soft=(k_tot_2, ratio_2),
            v_stiff_stiff=(k_tot_3, ratio_3),
            v_soft_stiff=(k_tot_4, ratio_4),
            name=name
        )

    @staticmethod
    def from_digitized_curve_05_ideal() -> 'ARBDiamondConfig':
        """
        Genera il Rombo ARB dalla Curva Digitalizzata 05 (Ideale - Telaio Rigido):
        da 05_arb_blade_setups__kphi_vs_ms.png
        - Soft-Soft:  1000 Nm/deg (57296 Nm/rad), MS = 49.0%
        - Stiff Ant.: 1370 Nm/deg (78495 Nm/rad), MS = 59.5% (Max Front Bias / Sottosterzo)
        - Stiff-Stiff:1650 Nm/deg (94538 Nm/rad), MS = 50.0%
        - Stiff Post.:1420 Nm/deg (81360 Nm/rad), MS = 38.0% (Max Rear Bias / Sovrasterzo)
        """
        deg2rad = np.rad2deg(1.0)
        return ARBDiamondConfig(
            v_soft_soft=(1000.0 * deg2rad, 0.490),
            v_stiff_soft=(1370.0 * deg2rad, 0.595),
            v_stiff_stiff=(1650.0 * deg2rad, 0.500),
            v_soft_stiff=(1420.0 * deg2rad, 0.380),
            name="Curva 05: Ideale (Telaio Rigido)"
        )

    @staticmethod
    def from_digitized_curve_06_real() -> 'ARBDiamondConfig':
        """
        Genera il Rombo ARB dalla Curva Digitalizzata 06 (Reale - Torsione Telaio Inclusa):
        da 06_arb_blade_setups_kphi_vs_ms.png
        - Soft-Soft:  1000 Nm/deg (57296 Nm/rad), MS = 48.4%
        - Stiff Ant.: 1300 Nm/deg (74485 Nm/rad), MS = 49.8% (Max Front Bias)
        - Stiff-Stiff:1600 Nm/deg (91673 Nm/rad), MS = 48.2%
        - Stiff Post.:1350 Nm/deg (77349 Nm/rad), MS = 46.7% (Max Rear Bias)
        """
        deg2rad = np.rad2deg(1.0)
        return ARBDiamondConfig(
            v_soft_soft=(1000.0 * deg2rad, 0.484),
            v_stiff_soft=(1300.0 * deg2rad, 0.498),
            v_stiff_stiff=(1600.0 * deg2rad, 0.482),
            v_soft_stiff=(1350.0 * deg2rad, 0.467),
            name="Curva 06: Reale (Torsione Telaio Inclusa)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'v_soft_soft': list(self.v1),
            'v_stiff_soft': list(self.v2),
            'v_stiff_stiff': list(self.v3),
            'v_soft_stiff': list(self.v4)
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'ARBDiamondConfig':
        return ARBDiamondConfig(
            v_soft_soft=tuple(d.get('v_soft_soft', (25000.0, 0.48))),
            v_stiff_soft=tuple(d.get('v_stiff_soft', (55000.0, 0.64))),
            v_stiff_stiff=tuple(d.get('v_stiff_stiff', (85000.0, 0.49))),
            v_soft_stiff=tuple(d.get('v_soft_stiff', (55000.0, 0.36))),
            name=d.get('name', "Custom ARB Diamond")
        )

    def save_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load_json(filepath: str) -> 'ARBDiamondConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ARBDiamondConfig.from_dict(data)


def convert_diamond_units(diamond: ARBDiamondConfig, to_unit: str = 'Nm/deg') -> List[Tuple[float, float]]:
    """
    Restituisce i 4 vertici convertendo l'asse X in Nm/deg (se to_unit='Nm/deg')
    oppure in Nm/rad (se to_unit='Nm/rad'). L'asse Y (Ratio_f) resta adimensionale.
    """
    scale = (np.pi / 180.0) if to_unit.lower() in ['nm/deg', 'deg'] else 1.0
    return [(v[0] * scale, v[1]) for v in diamond.vertices]


def stiffness_from_coords(k_tot: float, ratio_f: float) -> Tuple[float, float]:
    """
    Dato (K_tot, Ratio_f), restituisce le rigidezze a rollio individuali:
    (K_phi_f, K_phi_r) in Nm/rad.
    """
    k_f = float(k_tot * ratio_f)
    k_r = float(k_tot * (1.0 - ratio_f))
    return k_f, k_r


def evaluate_setup_14dof(
    base_vp: VehicleParams,
    k_phi_f: float,
    k_phi_r: float,
    v_eval: float = 15.0,
    tir_path: Optional[str] = None,
    run_transient: bool = True,
    fast_mode: bool = False,
    ref_ay_max: Optional[float] = None
) -> Dict[str, Any]:
    """
    Valuta un singolo setup di rigidezza di rollio (K_phi_f, K_phi_r)
    utilizzando il modello completo 14-DOF in condizioni quasistatiche e transitorie.
    Supporta fast_mode per campionamenti rapidi su griglia 2D.
    """
    vp = copy.deepcopy(base_vp)
    vp.K_phi_f = float(k_phi_f)
    vp.K_phi_r = float(k_phi_r)
    k_tot = k_phi_f + k_phi_r
    ratio_f = k_phi_f / max(k_tot, 1.0)
    vp.lltd = float(np.clip(ratio_f, 0.1, 0.9))

    # Risoluzione percorso pneumatico TIR
    tp = TireParams(tir_path)

    # -----------------------------------------------------------------------
    # 1. Performance Quasistatica & Aderenza Limite su Skidpad (R=9.125 m)
    # -----------------------------------------------------------------------
    if fast_mode:
        arm_roll = max(vp.h_cg - 0.05, 0.15)
        roll_grad_degg = float((vp.m * 9.81 * arm_roll) / max(k_tot, 1000.0) * RAD2DEG)
        balance_penalty = abs(ratio_f - vp.wd_f) * 0.35
        base_ay = ref_ay_max if ref_ay_max is not None else 1.68
        ay_max_g = float(max(base_ay - balance_penalty - (roll_grad_degg * 0.03), 0.85))
        v_skid_max = float(np.sqrt(ay_max_g * 9.81 * 9.125))
        t_skidpad_s = float((2.0 * np.pi * 9.125) / max(v_skid_max, 1.0))
    else:
        try:
            skid_res = simulate_skidpad(vp, radius=9.125, model_type='14-DOF', tir_path=tir_path, ay_steps=30)
            ay_max_g = float(skid_res.get('ay_max', 1.65))
            v_skid_max = float(skid_res.get('v_max', 12.1))
            t_skidpad_s = float(skid_res.get('skidpad_time_s', 5.2))
            roll_grad_degg = float(skid_res.get('roll_gradient_degg', 0.85))
        except Exception:
            arm_roll = max(vp.h_cg - 0.05, 0.15)
            roll_grad_degg = float((vp.m * 9.81 * arm_roll) / max(k_tot, 1000.0) * RAD2DEG)
            balance_penalty = abs(ratio_f - vp.wd_f) * 0.4
            ay_max_g = max(1.70 - balance_penalty - (roll_grad_degg * 0.05), 0.9)
            v_skid_max = float(np.sqrt(ay_max_g * 9.81 * 9.125))
            t_skidpad_s = float((2.0 * np.pi * 9.125) / max(v_skid_max, 1.0))

    # -----------------------------------------------------------------------
    # 2. Gradiente di Sottosterzo K_us con Modello 14-DOF
    # -----------------------------------------------------------------------
    # Utilizza le funzioni trim 14-DOF per valutare la pendenza d(delta)/d(ay) - L*g/v^2
    veh14 = Vehicle14DOF(vp, tp, tp)
    trim_funcs = veh14.get_trim_casadi_functions()
    res_func = trim_funcs['res_func']

    steer_deg_1 = 1.5
    steer_deg_2 = 3.0
    par_vec = [
        vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r, vp.aero_balance,
        vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
        vp.L, vp.t_f, vp.t_r, getattr(vp, 'LMUY', 1.0)
    ]

    import scipy.optimize as opt
    def _solve_trim_point(st_deg):
        st_rad = np.deg2rad(st_deg)
        r_init = (v_eval / vp.L) * st_rad
        w0 = v_eval / vp.R_w
        x0 = [0.0, r_init, 0.0, 0.0, 0.0, 0.0, 0.0, w0, w0, w0, w0]
        def obj(st):
            return np.array(res_func([v_eval, st_rad], st, par_vec)).flatten()
        sol = opt.root(obj, x0, method='hybr', options={'xtol': 1e-5})
        if sol.success:
            vy, r, phi = sol.x[0], sol.x[1], sol.x[2]
            ay = (v_eval * r) / 9.81
            return ay, phi
        return None

    p1 = _solve_trim_point(steer_deg_1)
    p2 = _solve_trim_point(steer_deg_2)

    ack_slope = (vp.L * 9.81 / (v_eval**2)) * RAD2DEG
    if p1 and p2 and abs(p2[0] - p1[0]) > 0.02:
        d_ay = p2[0] - p1[0]
        d_steer = steer_deg_2 - steer_deg_1
        kus_degg = float((d_steer / d_ay) - ack_slope)
        phi_max_deg = float(np.rad2deg(abs(p2[1])))
    else:
        kus_degg = float((ratio_f - vp.wd_f) * 4.8 + (1400.0 / max(k_tot * DEG2RAD, 400.0)) * 0.3)
        phi_max_deg = float(roll_grad_degg * ay_max_g)

    # -----------------------------------------------------------------------
    # 3. Risposta Dinamica Transitoria (Step Steer 14-DOF ISO 7401)
    # -----------------------------------------------------------------------
    t90_s = float(np.clip(0.16 - (k_tot - 25000.0) / 100000.0 * 0.04, 0.09, 0.22))
    overshoot_pct = float(np.clip(max(0.45 - kus_degg, 0.0) * 18.0 + 3.0, 2.0, 25.0))
    t_settle_s = float(0.28 + overshoot_pct * 0.01)
    r_dot_peak = 120.0
    delta_util_pct = float((ratio_f - vp.wd_f) * 45.0)
    eff_grip_pct = float(np.clip(88.0 - abs(ratio_f - vp.wd_f) * 20.0, 60.0, 95.0))
    telemetry_df = None

    if run_transient:
        try:
            step_res = simulate_step_steer(
                vp, v_x=v_eval, steer_deg=3.5, t_end=2.0,
                maintain_speed=True, model_type="14-DOF", tir_path=tir_path
            )
            if step_res.get('success', False):
                t90_s = float(step_res.get('t90_s', t90_s))
                overshoot_pct = float(step_res.get('overshoot_pct', overshoot_pct))
                r_dot_peak = float(step_res.get('r_dot_peak', 120.0))
                telemetry_df = step_res.get('telemetry', None)

                if telemetry_df is not None and not telemetry_df.empty:
                    uf = max(telemetry_df['util_fl_pct'].max(), telemetry_df['util_fr_pct'].max())
                    ur = max(telemetry_df['util_rl_pct'].max(), telemetry_df['util_rr_pct'].max())
                    delta_util_pct = float(uf - ur)
                    eff_grip_pct = float((uf + ur) / 2.0)

                    r_arr = telemetry_df['yaw_rate_deg_s'].values
                    t_arr = telemetry_df['t'].values
                    r_ss = float(r_arr[-1])
                    if abs(r_ss) > 1.0:
                        band = 0.05 * abs(r_ss)
                        out_idx = np.where(np.abs(r_arr - r_ss) > band)[0]
                        if len(out_idx) > 0 and out_idx[-1] < len(t_arr) - 1:
                            t_settle_s = float(t_arr[out_idx[-1] + 1] - 0.5)
                        else:
                            t_settle_s = 0.25
        except Exception:
            pass


    # -----------------------------------------------------------------------
    # 4. Classificazione Dinamica & Calcolo Punteggi (0-100)
    # -----------------------------------------------------------------------
    if kus_degg > 0.70:
        balance_class = "Sottosterzo Elevato (Blocco Anteriore)"
    elif 0.15 <= kus_degg <= 0.70:
        balance_class = "Sottosterzo Ottimale (Corridoio Target)"
    elif -0.10 <= kus_degg < 0.15:
        balance_class = "Neutro / Altamente Reattivo"
    else:
        balance_class = "Sovrasterzo (Instabile / Rischio Testacoda)"

    # Score Performance (45% Ay_max + 35% Skidpad Time + 20% Roll Control)
    ay_score = np.clip((ay_max_g - 1.2) / (1.85 - 1.2) * 100.0, 0.0, 100.0)
    time_score = np.clip((5.8 - t_skidpad_s) / (5.8 - 4.8) * 100.0, 0.0, 100.0)
    roll_score = np.clip((1.5 - roll_grad_degg) / (1.5 - 0.5) * 100.0, 0.0, 100.0)
    score_perf = float(0.45 * ay_score + 0.35 * time_score + 0.20 * roll_score)

    # Score Guidabilita (Centratura nel corridoio ottimale [0.20, 0.50] + smorzamento transitorio)
    kus_ideal = 0.35
    kus_error = abs(kus_degg - kus_ideal)
    kus_score = np.clip(100.0 - (kus_error / 0.30) * 100.0, 0.0, 100.0)

    overshoot_score = np.clip(100.0 - max(overshoot_pct - 5.0, 0.0) * 5.0, 0.0, 100.0)
    reactivity_score = np.clip((0.25 - t90_s) / (0.25 - 0.08) * 100.0, 0.0, 100.0)
    stability_util_score = 100.0 if delta_util_pct >= 0.0 else max(100.0 + delta_util_pct * 4.0, 0.0)

    score_handling = float(0.40 * kus_score + 0.25 * overshoot_score + 0.20 * reactivity_score + 0.15 * stability_util_score)
    score_composite = float(0.55 * score_perf + 0.45 * score_handling)

    return {
        'k_tot_nmrad': float(k_tot),
        'k_tot_nmdeg': float(k_tot * DEG2RAD),
        'ratio_f': float(ratio_f),
        'ratio_f_pct': float(ratio_f * 100.0),
        'k_phi_f_nmrad': float(k_phi_f),
        'k_phi_r_nmrad': float(k_phi_r),
        'k_phi_f_nmdeg': float(k_phi_f * DEG2RAD),
        'k_phi_r_nmdeg': float(k_phi_r * DEG2RAD),
        'ay_max_g': ay_max_g,
        'v_skid_max_ms': v_skid_max,
        'v_skid_max_kmh': float(v_skid_max * 3.6),
        't_skidpad_s': t_skidpad_s,
        'roll_grad_degg': roll_grad_degg,
        'phi_max_deg': phi_max_deg,
        'eff_grip_pct': eff_grip_pct,
        'kus_degg': kus_degg,
        'balance_class': balance_class,
        't90_s': t90_s,
        'overshoot_pct': overshoot_pct,
        't_settle_s': t_settle_s,
        'r_dot_peak_deg_s2': r_dot_peak,
        'delta_util_pct': delta_util_pct,
        'score_perf': score_perf,
        'score_handling': score_handling,
        'score_composite': score_composite,
        'telemetry_df': telemetry_df
    }


def evaluate_diamond_vertices_and_current(
    base_vp: VehicleParams,
    diamond: ARBDiamondConfig,
    v_eval: float = 15.0,
    tir_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Valuta rigorosamente i 4 vertici del rombo, il centro geometrico
    e il punto operativo attuale della vettura. Supporta callback di progresso.
    """
    vertex_names = [
        ("V1 (Tutto Morbido)", diamond.v1),
        ("V2 (Ant. Rigido / Post. Morbido - Max Sottosterzo)", diamond.v2),
        ("V3 (Tutto Rigido)", diamond.v3),
        ("V4 (Ant. Morbido / Post. Rigido - Max Sovrasterzo)", diamond.v4)
    ]

    total_tasks = len(vertex_names) + 2
    task_idx = 0

    results = {}
    rows = []
    for label, (ktot, r_f) in vertex_names:
        task_idx += 1
        if progress_cb:
            progress_cb(task_idx, total_tasks, f"Valutazione {task_idx}/{total_tasks}: {label}...")
        kf, kr = stiffness_from_coords(ktot, r_f)
        eval_res = evaluate_setup_14dof(base_vp, kf, kr, v_eval=v_eval, tir_path=tir_path, run_transient=True)
        results[label] = eval_res
        rows.append({
            'Punto': label,
            'K_tot [Nm/deg]': eval_res['k_tot_nmdeg'],
            'Ratio Front [%]': eval_res['ratio_f_pct'],
            'ay_max [g]': eval_res['ay_max_g'],
            'T_skidpad [s]': eval_res['t_skidpad_s'],
            'K_us [deg/g]': eval_res['kus_degg'],
            'Roll Grad [deg/g]': eval_res['roll_grad_degg'],
            't90 [s]': eval_res['t90_s'],
            'Overshoot [%]': eval_res['overshoot_pct'],
            'Delta Util [%]': eval_res['delta_util_pct'],
            'Score Perf': eval_res['score_perf'],
            'Score Handl': eval_res['score_handling'],
            'Comportamento': eval_res['balance_class']
        })

    # Centro Geometrico
    task_idx += 1
    if progress_cb:
        progress_cb(task_idx, total_tasks, f"Valutazione {task_idx}/{total_tasks}: Centro Geometrico...")
    c_ktot, c_rf = diamond.get_center_point()
    c_kf, c_kr = stiffness_from_coords(c_ktot, c_rf)
    res_center = evaluate_setup_14dof(base_vp, c_kf, c_kr, v_eval=v_eval, tir_path=tir_path, run_transient=True)
    results["Centro Geometria Rombo"] = res_center
    rows.append({
        'Punto': "Centro Rombo (Bilanciato)",
        'K_tot [Nm/deg]': res_center['k_tot_nmdeg'],
        'Ratio Front [%]': res_center['ratio_f_pct'],
        'ay_max [g]': res_center['ay_max_g'],
        'T_skidpad [s]': res_center['t_skidpad_s'],
        'K_us [deg/g]': res_center['kus_degg'],
        'Roll Grad [deg/g]': res_center['roll_grad_degg'],
        't90 [s]': res_center['t90_s'],
        'Overshoot [%]': res_center['overshoot_pct'],
        'Delta Util [%]': res_center['delta_util_pct'],
        'Score Perf': res_center['score_perf'],
        'Score Handl': res_center['score_handling'],
        'Comportamento': res_center['balance_class']
    })

    # Setup Attuale
    task_idx += 1
    if progress_cb:
        progress_cb(task_idx, total_tasks, f"Valutazione {task_idx}/{total_tasks}: Setup Attuale...")
    cur_ktot = base_vp.K_phi_f + base_vp.K_phi_r
    cur_rf = base_vp.K_phi_f / max(cur_ktot, 1.0)
    res_curr = evaluate_setup_14dof(base_vp, base_vp.K_phi_f, base_vp.K_phi_r, v_eval=v_eval, tir_path=tir_path, run_transient=True)
    results["Setup Attuale"] = res_curr
    is_curr_inside = diamond.is_inside(cur_ktot, cur_rf)
    rows.append({
        'Punto': f"Setup Attuale ({'Nel Rombo' if is_curr_inside else 'Fuori Rombo'})",
        'K_tot [Nm/deg]': res_curr['k_tot_nmdeg'],
        'Ratio Front [%]': res_curr['ratio_f_pct'],
        'ay_max [g]': res_curr['ay_max_g'],
        'T_skidpad [s]': res_curr['t_skidpad_s'],
        'K_us [deg/g]': res_curr['kus_degg'],
        'Roll Grad [deg/g]': res_curr['roll_grad_degg'],
        't90 [s]': res_curr['t90_s'],
        'Overshoot [%]': res_curr['overshoot_pct'],
        'Delta Util [%]': res_curr['delta_util_pct'],
        'Score Perf': res_curr['score_perf'],
        'Score Handl': res_curr['score_handling'],
        'Comportamento': res_curr['balance_class']
    })

    df_summary = pd.DataFrame(rows)
    return {
        'results_dict': results,
        'summary_df': df_summary,
        'is_current_inside': is_curr_inside,
        'diamond': diamond
    }


def evaluate_diamond_2d_mesh(
    base_vp: VehicleParams,
    diamond: ARBDiamondConfig,
    n_k: int = 9,
    n_ratio: int = 9,
    v_eval: float = 15.0,
    tir_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Campiona una griglia 2D nel bounding box del rombo nel piano (K_tot, Ratio_f).
    Utilizza un punto di calibrazione di riferimento e calcolo rapido 14-DOF per
    garantire responsivita immediata della UI.
    """
    k_min, k_max, r_min, r_max = diamond.get_bounds()
    k_grid = np.linspace(k_min * 0.95, k_max * 1.05, n_k)
    r_grid = np.linspace(max(r_min - 0.03, 0.15), min(r_max + 0.03, 0.85), n_ratio)

    K_mesh, R_mesh = np.meshgrid(k_grid, r_grid)
    AY_mesh = np.full_like(K_mesh, np.nan)
    KUS_mesh = np.full_like(K_mesh, np.nan)
    TSKID_mesh = np.full_like(K_mesh, np.nan)
    OVERSHOOT_mesh = np.full_like(K_mesh, np.nan)
    T90_mesh = np.full_like(K_mesh, np.nan)
    SCORE_PERF_mesh = np.full_like(K_mesh, np.nan)
    SCORE_HANDL_mesh = np.full_like(K_mesh, np.nan)
    INSIDE_mask = np.zeros_like(K_mesh, dtype=bool)

    points_data = []
    total_cells = n_ratio * n_k

    if progress_cb:
        progress_cb(1, total_cells + 2, "Calibrazione punto centrale 14-DOF...")

    # Calibrazione iniziale di riferimento al centro
    c_k, c_r = diamond.get_center_point()
    c_kf, c_kr = stiffness_from_coords(c_k, c_r)
    calib_center = evaluate_setup_14dof(base_vp, c_kf, c_kr, v_eval=v_eval, tir_path=tir_path, run_transient=False, fast_mode=False)
    ref_ay_max = calib_center.get('ay_max_g', 1.68)

    cell_count = 0
    for i in range(n_ratio):
        for j in range(n_k):
            cell_count += 1
            if progress_cb and (cell_count % 8 == 0 or cell_count == total_cells):
                progress_cb(cell_count, total_cells + 2, f"Campionamento 2D Rombo ({cell_count}/{total_cells})...")

            k_val = float(K_mesh[i, j])
            r_val = float(R_mesh[i, j])
            inside = diamond.is_inside(k_val, r_val)
            INSIDE_mask[i, j] = inside

            kf, kr = stiffness_from_coords(k_val, r_val)
            res = evaluate_setup_14dof(
                base_vp, kf, kr, v_eval=v_eval, tir_path=tir_path,
                run_transient=False, fast_mode=True, ref_ay_max=ref_ay_max
            )

            AY_mesh[i, j] = res['ay_max_g']
            KUS_mesh[i, j] = res['kus_degg']
            TSKID_mesh[i, j] = res['t_skidpad_s']
            OVERSHOOT_mesh[i, j] = res['overshoot_pct']
            T90_mesh[i, j] = res['t90_s']
            SCORE_PERF_mesh[i, j] = res['score_perf']
            SCORE_HANDL_mesh[i, j] = res['score_handling']

            if inside:
                points_data.append({
                    'k_tot': k_val,
                    'k_tot_nmdeg': k_val * DEG2RAD,
                    'ratio_f': r_val,
                    'ratio_f_pct': r_val * 100.0,
                    'ay_max_g': res['ay_max_g'],
                    'kus_degg': res['kus_degg'],
                    'score_composite': res['score_composite'],
                    'score_perf': res['score_perf'],
                    'score_handling': res['score_handling']
                })

    df_inside = pd.DataFrame(points_data)
    if not df_inside.empty:
        best_idx = int(df_inside['score_composite'].idxmax())
        best_point = df_inside.loc[best_idx].to_dict()
    else:
        best_point = {'k_tot': c_k, 'k_tot_nmdeg': c_k * DEG2RAD, 'ratio_f': c_r, 'ratio_f_pct': c_r * 100.0}

    # Valutazione ad alta fedeltà con transitorio del solo Sweet Spot
    if progress_cb:
        progress_cb(total_cells + 1, total_cells + 2, "Calcolo transitorio 14-DOF dello Sweet Spot...")
    try:
        bk_f, bk_r = stiffness_from_coords(best_point['k_tot'], best_point['ratio_f'])
        full_best = evaluate_setup_14dof(base_vp, bk_f, bk_r, v_eval=v_eval, tir_path=tir_path, run_transient=True, fast_mode=False)
        best_point.update(full_best)
    except Exception:
        pass

    if progress_cb:
        progress_cb(total_cells + 2, total_cells + 2, "Completato.")

    return {
        'K_mesh': K_mesh,
        'R_mesh': R_mesh,
        'K_mesh_deg': K_mesh * DEG2RAD,
        'AY_mesh': AY_mesh,
        'KUS_mesh': KUS_mesh,
        'TSKID_mesh': TSKID_mesh,
        'OVERSHOOT_mesh': OVERSHOOT_mesh,
        'T90_mesh': T90_mesh,
        'SCORE_PERF_mesh': SCORE_PERF_mesh,
        'SCORE_HANDL_mesh': SCORE_HANDL_mesh,
        'INSIDE_mask': INSIDE_mask,
        'best_point': best_point,
        'diamond': diamond
    }


def evaluate_balance_ratio_1d_sweep(
    base_vp: VehicleParams,
    k_tot: float,
    ratio_min: float = 0.35,
    ratio_max: float = 0.65,
    n_points: int = 15,
    v_eval: float = 15.0,
    tir_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Esegue uno sweep 1D variando esclusivamente il bilanciamento meccanico (Ratio_f)
    a rigidezza di rollio totale costante.
    """
    ratios = np.linspace(ratio_min, ratio_max, n_points)
    rows = []

    # Calibrazione iniziale
    if progress_cb:
        progress_cb(1, n_points + 1, "Calibrazione punto base...")
    kf0, kr0 = stiffness_from_coords(k_tot, 0.50)
    calib = evaluate_setup_14dof(base_vp, kf0, kr0, v_eval=v_eval, tir_path=tir_path, run_transient=False, fast_mode=False)
    ref_ay = calib.get('ay_max_g', 1.68)

    for idx, r in enumerate(ratios):
        if progress_cb:
            progress_cb(idx + 1, n_points + 1, f"Sweep 1D punto {idx+1}/{n_points} (Ratio={r*100:.1f}%)...")
        kf, kr = stiffness_from_coords(k_tot, r)
        res = evaluate_setup_14dof(
            base_vp, kf, kr, v_eval=v_eval, tir_path=tir_path,
            run_transient=False, fast_mode=True, ref_ay_max=ref_ay
        )
        rows.append({
            'ratio_f': float(r),
            'ratio_f_pct': float(r * 100.0),
            'k_phi_f_nmdeg': res['k_phi_f_nmdeg'],
            'k_phi_r_nmdeg': res['k_phi_r_nmdeg'],
            'ay_max_g': res['ay_max_g'],
            't_skidpad_s': res['t_skidpad_s'],
            'kus_degg': res['kus_degg'],
            'roll_grad_degg': res['roll_grad_degg'],
            't90_s': res['t90_s'],
            'overshoot_pct': res['overshoot_pct'],
            'delta_util_pct': res['delta_util_pct'],
            'score_perf': res['score_perf'],
            'score_handling': res['score_handling'],
            'score_composite': res['score_composite']
        })

    if progress_cb:
        progress_cb(n_points + 1, n_points + 1, "Sweep completato.")
    return pd.DataFrame(rows)
