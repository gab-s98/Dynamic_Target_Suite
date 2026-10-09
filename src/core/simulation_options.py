"""
simulation_options.py
---------------------
Centralized configuration, discretization settings, solver parameters,
and aerodynamic/powertrain defaults for the FSAE Dynamic Target Suite.

Provides JSON serialization and session-wide state persistence.
"""

import json
import os
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SimulationOptions:
    # -----------------------------------------------------------------------
    # 1. Discretization & Numerical Integration
    # -----------------------------------------------------------------------
    dt_ode: float = 0.002             # Integration time step for 7-DOF/14-DOF ODE [s] (e.g. 0.001 - 0.005 s)
    t_end_step: float = 4.0           # Total duration for Step Steer maneuver [s]
    t_end_dlc: float = 5.0            # Total duration for Double Lane Change maneuver [s]
    t_ramp_steer: float = 0.08        # Steering rise time / ramp duration [s] (0.0 = step, >0 = realistic driver)
    ds_track: float = 0.5             # Spatial discretization step for AutoX/Endurance [m]
    hd_num_points: int = 120          # Resolution points for Handling Diagram
    speed_sweep_steps: int = 10       # Number of velocity steps in Speed Sweep
    
    # -----------------------------------------------------------------------
    # 2. Dynamic Maneuver Controls & Torque Vectoring
    # -----------------------------------------------------------------------
    maintain_speed: bool = True       # Apply closed-loop drive torque to maintain target speed Vx during maneuver
    tv_mode: str = "Feedforward"      # "Off", "Feedforward" (fixed dT), "YawRate_Feedback" (closed-loop)
    tv_gain: float = 12.0             # Proportional gain for Yaw Rate Feedback TV [Nm / (rad/s)]
    tv_max_torque: float = 80.0       # Max differential torque per wheel [Nm]
    
    # -----------------------------------------------------------------------
    # 3. Solver & Convergence Settings
    # -----------------------------------------------------------------------
    newton_xtol: float = 1e-6         # Tolerance for state variables in Newton/Root solvers
    newton_ftol: float = 1e-6         # Residual force/moment tolerance [N, Nm]
    newton_maxiter: int = 50          # Maximum iterations for steady-state equilibrium
    bisection_steps: int = 14         # Steps in bisection search for limit Ay
    beta_limit_deg: float = 35.0      # Maximum physical sideslip angle before loss of control [deg]
    delta_limit_deg: float = 35.0     # Maximum physical wheel steer angle [deg]
    
    # -----------------------------------------------------------------------
    # 4. Aerodynamics & Environment
    # -----------------------------------------------------------------------
    rho: float = 1.225                # Air density [kg/m³]
    S: float = 1.139                  # Aerodynamic reference area [m²]
    Cz: float = 3.421                 # Downforce coefficient (negative lift) [-]
    Cd: float = 1.200                 # Drag coefficient [-]
    use_aeromap: bool = False         # Enable ride height / pitch CFD Aero Maps
    
    # -----------------------------------------------------------------------
    # 5. Powertrain & Energy
    # -----------------------------------------------------------------------
    P_max_kw: float = 80.0            # Max electrical motor power limit [kW]
    T_max_motor: float = 100.0        # Max motor shaft torque [Nm]
    final_drive: float = 4.00         # Gearbox reduction ratio [-]
    n_drivetrain: float = 0.95        # Transmission mechanical efficiency [-]
    brake_bias: float = 0.60          # Front axle braking ratio [0-1]
    battery_capacity_kwh: float = 6.8 # Battery energy capacity [kWh]
    regen_max_kw: float = 40.0        # Maximum regenerative braking power [kW]
    regen_efficiency: float = 0.65    # Round-trip regeneration efficiency [-]

    # -----------------------------------------------------------------------
    # 6. Track Surface & Grip Properties
    # -----------------------------------------------------------------------
    track_grip_multiplier: float = 1.0     # Global µ multiplier (1.0=dry, 0.75=damp, 0.50=wet, 0.15=ice)
    offtrack_grip_multiplier: float = 0.35 # µ reduction factor when off-track (grass/gravel)
    kerb_grip_multiplier: float = 0.85     # µ reduction factor on kerbs

    # -----------------------------------------------------------------------
    # 7. Tire Degradation Model
    # -----------------------------------------------------------------------
    tire_degradation_mode: str = "off"     # "off", "light" (1-zone thermal), "detailed" (3-zone thermal)
    tire_T_ambient: float = 25.0           # Ambient temperature [°C]
    tire_T_start: float = 25.0             # Starting tire temperature [°C] (25°C cold, 45°C warm, 56.2°C optimal)
    tire_T_optimal: float = 56.2           # Optimal grip temperature [°C] (calibrated from TTC Hoosier R20 Run 4)
    tire_T_overheat: float = 85.0          # Overheating threshold [°C]
    tire_wear_rate_scale: float = 1.0      # Tire wear rate scale multiplier (1.0 = real endurance, 2.0, 5.0, 10.0)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationOptions":
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def save_to_file(self, filepath: str = None) -> None:
        if filepath is None:
            base_dir = os.path.dirname(__file__)
            filepath = os.path.join(base_dir, "simulation_options.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str = None) -> "SimulationOptions":
        if filepath is None:
            base_dir = os.path.dirname(__file__)
            filepath = os.path.join(base_dir, "simulation_options.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception:
                return cls()
        return cls()


# Global Singleton Instance
_ACTIVE_OPTIONS: SimulationOptions = None


def get_simulation_options() -> SimulationOptions:
    global _ACTIVE_OPTIONS
    if _ACTIVE_OPTIONS is None:
        _ACTIVE_OPTIONS = SimulationOptions.load_from_file()
    return _ACTIVE_OPTIONS


def update_simulation_options(new_opts: SimulationOptions) -> None:
    global _ACTIVE_OPTIONS
    _ACTIVE_OPTIONS = new_opts
    _ACTIVE_OPTIONS.save_to_file()
