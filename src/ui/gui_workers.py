"""
gui_workers.py
--------------
Dynamic Target Suite ? Background QThread Workers & Multi-Core Subprocesses.
Offloads intensive numerical sweeps, multi-model optimizations, and direct collocation
trajectories to secondary threads, keeping the PyQt5 user interface reactive and stutter-free.
"""

import traceback
from PyQt5 import QtCore

from src.utils.transient_sensitivity import run_sensitivity_sweep_1d, run_sensitivity_sweep_2d
from src.mdo.mdo_arb_diamond_balance import (
    evaluate_diamond_vertices_and_current,
    evaluate_diamond_2d_mesh,
    evaluate_balance_ratio_1d_sweep
)
from src.mdo.mdo_direct_collocation import solve_direct_collocation_racing_line


class SensitivityWorker(QtCore.QThread):
    """
    Background worker for 1D/2D transient sensitivity parameter sweeps.
    Dispatches simulation batches across thread pools and reports progress.
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, mode, base_vp, params_dict):
        super().__init__()
        self.mode = mode
        self.base_vp = base_vp
        self.params_dict = params_dict

    def run(self):
        try:
            def p_cb(done, total):
                self.progress.emit(done, total, f"Calcolo {done}/{total} simulazioni...")

            if self.mode == '1D':
                res = run_sensitivity_sweep_1d(
                    base_params=self.base_vp,
                    scenario_key=self.params_dict['scenario_x'],
                    sweep_values=self.params_dict['sweep_x'],
                    maneuver=self.params_dict['maneuver'],
                    v_x=self.params_dict['v_x'],
                    steer_deg=self.params_dict['steer_deg'],
                    model_type=self.params_dict['model_type'],
                    tir_path=self.params_dict['tir_path'],
                    progress_cb=p_cb
                )
            else:
                res = run_sensitivity_sweep_2d(
                    base_params=self.base_vp,
                    scenario_key_x=self.params_dict['scenario_x'],
                    sweep_values_x=self.params_dict['sweep_x'],
                    scenario_key_y=self.params_dict['scenario_y'],
                    sweep_values_y=self.params_dict['sweep_y'],
                    kpi_key=self.params_dict['kpi_key'],
                    maneuver=self.params_dict['maneuver'],
                    v_x=self.params_dict['v_x'],
                    steer_deg=self.params_dict['steer_deg'],
                    model_type=self.params_dict['model_type'],
                    tir_path=self.params_dict['tir_path'],
                    progress_cb=p_cb
                )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class ARBDiamondWorker(QtCore.QThread):
    """
    Background worker for ARB Balance Diamond envelope evaluations
    (vertices, 2D mesh, and 1D stiffness sweeps).
    """
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, task_type: str, base_vp, diamond, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.base_vp = base_vp
        self.diamond = diamond
        self.kwargs = kwargs
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            def p_cb(done, total, msg):
                if self._cancelled:
                    raise InterruptedError("Operazione annullata dall'utente.")
                self.progress.emit(done, total, msg)

            if self.task_type == 'vertices':
                res = evaluate_diamond_vertices_and_current(
                    self.base_vp, self.diamond,
                    v_eval=self.kwargs.get('v_eval', 15.0),
                    tir_path=self.kwargs.get('tir_path', None),
                    progress_cb=p_cb
                )
                res['task_type'] = 'vertices'
            elif self.task_type == '2d_mesh':
                res = evaluate_diamond_2d_mesh(
                    self.base_vp, self.diamond,
                    n_k=self.kwargs.get('n_k', 9),
                    n_ratio=self.kwargs.get('n_ratio', 9),
                    v_eval=self.kwargs.get('v_eval', 15.0),
                    tir_path=self.kwargs.get('tir_path', None),
                    progress_cb=p_cb
                )
                res['task_type'] = '2d_mesh'
            elif self.task_type == '1d_sweep':
                df_sweep = evaluate_balance_ratio_1d_sweep(
                    self.base_vp,
                    k_tot=self.kwargs.get('k_tot', 50000.0),
                    ratio_min=self.kwargs.get('ratio_min', 0.32),
                    ratio_max=self.kwargs.get('ratio_max', 0.68),
                    n_points=self.kwargs.get('n_points', 12),
                    v_eval=self.kwargs.get('v_eval', 15.0),
                    tir_path=self.kwargs.get('tir_path', None),
                    progress_cb=p_cb
                )
                res = {'task_type': '1d_sweep', 'sweep_df': df_sweep, 'k_tot': self.kwargs.get('k_tot', 50000.0)}
            else:
                res = {}
            if not self._cancelled:
                self.finished.emit(res)
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(f"{e}\n{traceback.format_exc()}")


class CollocationWorker(QtCore.QThread):
    """
    Background worker thread for smooth, non-blocking IPOPT Direct Collocation
    racing line trajectory optimization.
    """
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, vp, ds_val, w_val, model_name, max_it, track_data=None, cost_weights=None,
                 collocation_scheme: str = "radau_3"):
        super().__init__()
        self.vp = vp
        self.ds_val = ds_val
        self.w_val = w_val
        self.model_name = model_name
        self.max_it = max_it
        self.track_data = track_data
        self.cost_weights = cost_weights
        self.collocation_scheme = collocation_scheme

    def run(self):
        try:
            def p_cb(pct, msg):
                self.progress.emit(pct, msg)

            kwargs = {
                'vehicle_params': self.vp,
                'ds': self.ds_val,
                'track_width_m': self.w_val,
                'model_type': self.model_name,
                'max_iter': self.max_it,
                'progress_callback': p_cb,
                'collocation_scheme': self.collocation_scheme,
            }
            if self.track_data is not None:
                kwargs['track_data'] = self.track_data
            if self.cost_weights is not None:
                kwargs['cost_weights'] = self.cost_weights

            res = solve_direct_collocation_racing_line(**kwargs)
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


def _worker_optimize_single_speed(args):
    """
    Top-level worker function for multiprocessing speed sweeps.
    Must reside at module scope to allow pickle serialization across processes.
    """
    try:
        if len(args) >= 14:
            v, initial_guess, custom_bounds, penalty, tir_file, tire_unc, enforce_us, obj_mode, skid_rad, accel_dist, model_type, kus_min, kus_weight, target_kus = args[:14]
        else:
            v, initial_guess, custom_bounds, penalty, tir_file, tire_unc, enforce_us, obj_mode, skid_rad, accel_dist, model_type, kus_min, kus_weight = args[:13]
            target_kus = 0.0
        from src.mdo.mdo_global_optimizer import GlobalOptimizer

        # In speed sweep, keep chassis dimensions (L, t_f, t_r) fixed to nominal/initial values
        # to isolate pure setup variation across velocity
        fixed_bounds = list(custom_bounds)
        L_nom, tf_nom, tr_nom = initial_guess[0], initial_guess[1], initial_guess[2]
        fixed_bounds[0] = (L_nom, L_nom)
        fixed_bounds[1] = (tf_nom, tf_nom)
        fixed_bounds[2] = (tr_nom, tr_nom)

        opt = GlobalOptimizer(v_x=v, penalty_factor=penalty, tir_file=tir_file,
                              objective_mode=obj_mode, skidpad_radius=skid_rad,
                              accel_distance=accel_dist, tire_uncertainty_pct=tire_unc,
                              model_type=model_type, enforce_understeer=enforce_us,
                              understeer_min_deg_g=kus_min, understeer_weight=kus_weight,
                              target_kus_deg_g=target_kus)
        res = opt.run_optimization(initial_guess=initial_guess, custom_bounds=tuple(fixed_bounds),
                                   n_starts=1)
        if res is not None and (res.success or res.fun < 10.0):
            max_ay = opt._evaluate_limit_ay(res.x)
            row = {
                'v_x': v,
                'success': True,
                'max_ay': float(max_ay),
                'x': list(res.x),
                'L': float(res.x[0]),
                't_f': float(res.x[1]),
                't_r': float(res.x[2]),
                'wd_f': float(res.x[3]),
                'lltd': float(res.x[4]),
                'aero_bal': float(res.x[5])
            }
            if len(res.x) >= 10:
                row['camber_f_static'] = float(res.x[6])
                row['camber_r_static'] = float(res.x[7])
                row['toe_f_static'] = float(res.x[8])
                row['toe_r_static'] = float(res.x[9])
            if len(res.x) >= 14:
                row['K_phi_f'] = float(res.x[10])
                row['K_phi_r'] = float(res.x[11])
                row['K_z_f']   = float(res.x[12])
                row['K_z_r']   = float(res.x[13])
            try:
                row['K_us'] = float(opt._evaluate_cornering_stiffness_and_understeer(res.x))
            except Exception:
                pass
            return row
        else:
            return {'v_x': v, 'success': False}
    except Exception as e:
        return {'v_x': v, 'success': False, 'error': str(e)}


class SevenPostRigWorker(QtCore.QThread):
    """
    Background worker thread for 7-Post Rig simulation execution.
    """
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, vp, cfg):
        super().__init__()
        self.vp = vp
        self.cfg = cfg
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from src.mdo.mdo_seven_post_rig import SevenPostRig
            rig = SevenPostRig(self.vp)

            def p_cb(pct, msg):
                if self._cancelled:
                    raise InterruptedError("Simulazione interrotta dall'utente.")
                self.progress.emit(int(pct), str(msg))

            res = rig.simulate(self.cfg, progress_callback=p_cb)
            if not self._cancelled:
                self.finished.emit(res)
        except Exception as e:
            if not self._cancelled:
                import traceback
                self.failed.emit(f"{e}\n{traceback.format_exc()}")