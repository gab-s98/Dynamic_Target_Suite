"""
Dynamic Target Suite - Physics & Vehicle Models Package
"""

from .params_vehicle import VehicleParams, nominal_fsae_params
from .params_tire import TireParams
from .tire_model_mf61 import mf61_lateral_force, mf61_longitudinal_force, mf61_aligning_torque
from .tire_degradation import TireDegradationModel, TireDegradationParams, TireState
from .damper_model_ttx25 import calculate_damper_force, calculate_wheel_damper_force, get_dyno_curve
from .powertrain_electric import EMRAX188HV, get_default_powertrain
from .steering_dynamics import calculate_steering_torque, generate_steering_torque_curve
from .vehicle_3dof import Vehicle3DOF
from .vehicle_7dof import Vehicle7DOF
from .vehicle_14dof import Vehicle14DOF
from .vehicle_3d_models import FSAECar3DModel, SUVCar3DModel

__all__ = [
    'VehicleParams',
    'nominal_fsae_params',
    'TireParams',
    'mf61_lateral_force',
    'mf61_longitudinal_force',
    'mf61_aligning_torque',
    'TireDegradationModel',
    'TireDegradationParams',
    'TireState',
    'calculate_damper_force',
    'calculate_wheel_damper_force',
    'get_dyno_curve',
    'EMRAX188HV',
    'get_default_powertrain',
    'calculate_steering_torque',
    'generate_steering_torque_curve',
    'Vehicle3DOF',
    'Vehicle7DOF',
    'Vehicle14DOF',
    'FSAECar3DModel',
    'SUVCar3DModel',
]
