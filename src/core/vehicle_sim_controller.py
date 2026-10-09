import os
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from src.models.params_vehicle import VehicleParams

class SimWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, task_name, func, *args, **kwargs):
        super().__init__()
        self.task_name = task_name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            self.progress.emit(10, f"Avvio {self.task_name}...")
            res = self.func(*self.args, **self.kwargs)
            self.progress.emit(100, f"{self.task_name} completato.")
            self.finished.emit({'task': self.task_name, 'result': res})
        except Exception as e:
            self.error.emit(str(e))


class VehicleSimController(QObject):
    """
    Controller pulito per l'esecuzione di calcoli MDO e dinamica del veicolo,
    completamente disaccoppiato dall'interfaccia grafica.
    """
    simulation_started = pyqtSignal(str)
    simulation_finished = pyqtSignal(dict)
    simulation_error = pyqtSignal(str)
    simulation_progress = pyqtSignal(int, str)
    
    def __init__(self, vehicle_params: VehicleParams = None, parent=None):
        super().__init__(parent)
        self.vp = vehicle_params if vehicle_params else VehicleParams.nominal_fsae_params()
        self._active_worker = None
        
    def _run_async(self, task_name, func, *args, **kwargs):
        if self._active_worker is not None and self._active_worker.isRunning():
            self.simulation_error.emit(f"Un'altra simulazione ({self._active_worker.task_name}) e' in corso.")
            return
            
        self.simulation_started.emit(task_name)
        self._active_worker = SimWorker(task_name, func, *args, **kwargs)
        self._active_worker.finished.connect(self.simulation_finished.emit)
        self._active_worker.error.connect(self.simulation_error.emit)
        self._active_worker.progress.connect(self.simulation_progress.emit)
        self._active_worker.start()

    def run_optimization(self):
        from src.mdo.mdo_global_optimizer import GlobalOptimizer
        def _opt():
            opt = GlobalOptimizer(self.vp)
            return opt.run_optimization()
        self._run_async("Ottimizzazione Globale MDO", _opt)

    def run_constant_parameters(self):
        # Esegue la valutazione MDO statica
        def _eval():
            pass # TODO: import and call the evaluate KPIs function
            return {}
        self._run_async("Valutazione Parametri Statica", _eval)
        
    def update_params(self, new_vp: VehicleParams):
        self.vp = new_vp
