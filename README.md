# Dynamic Target Suite

> **Formula SAE / Formula Student Vehicle Dynamics & MDO Suite**  
> Physics-accurate, CasADi-native, open-source.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://python.org)
[![CasADi](https://img.shields.io/badge/CasADi-3.7-orange)](https://web.casadi.org/)
[![OpenMDAO](https://img.shields.io/badge/OpenMDAO-3.x-green)](https://openmdao.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-128%20passing-brightgreen)](tests/)

---

## What is Dynamic Target Suite?

**Dynamic Target Suite** is an integrated vehicle dynamics analysis, multidisciplinary optimization (MDO), and real-time simulation environment for Formula SAE / Formula Student and prototype race cars.

It lets a vehicle dynamics engineer go from a blank-sheet car concept to a validated, optimized setup in a single workflow:

```
Parameters → Optimization → Handling Analysis → Transient Validation → Real-Time Sim
```

The suite is built around three physical fidelity levels, all sharing the same parameter set and Pacejka MF6.1 tire model:

| Model | DOF | Use case |
|---|---|---|
| `vehicle_3dof.py` | 3 | MDO target synthesis, steady-state handling maps |
| `vehicle_7dof.py` | 7 | Transient maneuvers, power/brake events |
| `vehicle_14dof.py` | 14 | Full kinematics, suspension travel, roll/pitch, real-time sim |

---

## Key Features

### Vehicle Models
- **3-DOF Steady-State** — CasADi Newton trim, analytical Jacobians, IPOPT-ready
- **7-DOF Transient** — Yaw + roll + 4-wheel spin dynamics (RK4 / scipy integrate)
- **14-DOF Full Vehicle** — Roll, pitch, 4 independent suspensions, Caster/KPI jacking, bump stops (smooth `tanh`), coupled roll-yaw inertia matrix (Ixz)

### Tire Model
- **Pacejka Magic Formula 6.1** — Full compliance with *Simcenter Tire MF-Tool Equation Manual (FITTYP=61)*
  - Pure Fx, Fy, Mz, Mx, My
  - Combined slip via G_xa / G_yk weighting factors
  - Camber stiffness (PKY6, PKY7), inflation pressure sensitivity (PPY1–PPY5, PPX1–PPX4)
  - CasADi-native (float and SX/MX compatible, no Python `if` branches)

### Optimization
- **MDO Global Optimizer** — Multi-start SLSQP with understeer gradient penalty (K_us ≥ 0) and 4-ratio balance constraints
- **Direct Collocation Racing Line** — CasADi + IPOPT on curvilinear coordinates (FSAE Autocross, Endurance, Skidpad, custom tracks via CSV)
- **Transient Sensitivity 1D/2D** — Parameter sweeps on 14-DOF ODE extracting 21 dynamic KPIs (t90, overshoot, LLTD peak, tire utilization…)

### Handling Analysis
- **Handling Diagram** — Pacejka Handling Diagram with physical roll stiffness LLT (SAE J670e)
- **Map of Achievable Performance (MAPs)** — (δ, u), (u, ay), (δ, ay) steady-state performance maps
- **Vehicle DNA Maps n1…n6** — Modal stability maps, Hurwitz condition contour, critical speed
- **Yaw Moment Diagram (YMD / Moment Method)** — Yaw moment vs lateral force with iso-β / iso-δ contours
- **G-G-V 3D Performance Bowl** — Full speed-dependent traction envelope

### Setup Tools
- **Suspension Sizing** — Spring, ARB tube sizing (hollow steel / titanium), 4-ratio balance radar
- **Steering Effort Calculator** — Pneumatic trail + mechanical trail, pilot torque < 5 Nm
- **Curve Digitizer** — Extract camber/toe kinematic maps from images (pixel → polynomial coefficients)
- **Damper Model** — Non-linear Öhlins TTX25 MkII with 6 valve specs (C11–C23) and 4-way independent adjustment (LSC/HSC/LSR/HSR)

### Simulator
- **Real-Time 14-DOF Driving Simulator** — 250–500 Hz RK4 loop, PyQt5 + OpenGL rendering
  - Xbox/gamepad force feedback (impulse triggers + 4-channel haptics)
  - Tire degradation model, road profile (3D bump/sine/random)
  - G-G diagram, suspension telemetry, 4-wheel slip angles live

---

## Screenshots

> *Screenshots to be added. Run `python main_gui.py` to see the full interface.*

<!-- 
Suggested screenshots:
1. Main window — Handling Diagram + MAPs tabs
2. Vehicle DNA Maps (n1..n6 heatmaps)  
3. Real-time 14-DOF driving simulator (3D view + telemetry)
-->

---

## Physical Parameters (Nominal FSAE Car)

| Parameter | Value | Notes |
|---|---|---|
| Mass | 292 kg | Driver included |
| Wheelbase | 1.529 m | |
| Front track | 1.252 m | |
| Rear track | 1.136 m | |
| CoG height | 0.270 m | |
| Iz (yaw) | 120 kg·m² | |
| Ix (roll) | 35 kg·m² | |
| Iy (pitch) | 110 kg·m² | |
| Downforce Cz | 3.421 | At reference area 1.139 m² |
| Motor | 80 kW peak | FSAE EV regulation limit |
| Damper | Öhlins TTX25 MkII | C12/R12 baseline, 4-way |
| Tire | Hoosier 43075 16×7.5-10 R20 | MF6.1 `.tir` file included |

---

## Quick Start

### Requirements

- Python 3.11 or 3.12
- Windows (primary target — CasADi IPOPT DLLs bundled)
- Linux / macOS: supported, but GUI untested

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Dynamic_Target_Suite.git
cd Dynamic_Target_Suite
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Run the GUI

```bash
python main_gui.py           # Professional interface (default)
python main_gui.py --classic # Classic backup interface
```

### Run Tests

```bash
python -m pytest tests/ -v --ignore=tests/test_gui_systematic.py
```

### Standalone Executable (Windows)

```bash
pyinstaller --noconfirm MDO_Target_Suite.spec
# Output: dist\MDO_Target_Suite\MDO_Target_Suite.exe
```

---

## Project Structure

```
Dynamic_Target_Suite/
│
├── main_gui.py                    # Entry point (--classic flag for backup GUI)
├── main_gui_pro.py                # Professional 5-workspace GUI
├── main_gui_classic.py            # Classic GUI (full feature backup)
│
├── vehicle_3dof.py                # 3-DOF steady-state model (CasADi)
├── vehicle_7dof.py                # 7-DOF transient model
├── vehicle_14dof.py               # 14-DOF full vehicle + ODE (CasADi/IPOPT)
│
├── tire_model_mf61.py             # Pacejka MF6.1 — full implementation
├── damper_model_ttx25.py          # Öhlins TTX25 MkII non-linear model
├── params_vehicle.py              # Single source of truth: VehicleParams dataclass
├── params_tire.py                 # .tir file parser
│
├── mdo_target_optimization.py     # Handling Diagram, MAPs, MMM generator
├── mdo_global_optimizer.py        # Multi-start SLSQP MDO
├── mdo_direct_collocation.py      # CasADi + IPOPT racing line optimizer
├── mdo_dynamic_events.py          # Step steer, DLC ISO 3888-2, AutoX, Endurance
├── mdo_skidpad.py                 # Skidpad simulator
├── mdo_acceleration.py            # Longitudinal acceleration simulator
│
├── transient_sensitivity.py       # 1D/2D parameter sweep, 21 KPIs
├── vehicle_dna_maps.py            # DNA maps n1…n6, stability analysis
├── interactive_driver_14dof.py    # Real-time driving simulator
│
├── tire_tir/                      # Hoosier MF6.1 .tir files
├── tracks/                        # FSAE circuit CSV files (autocross, endurance…)
├── docs/                          # DEVELOPER_GUIDE.md, USER_GUIDE.md
└── tests/                         # 27 unit tests
```

---

## Standards & Engineering Formulations

The suite implements vehicle dynamics models traceable to international automotive standards and established vehicle dynamics formulations:

| Module | Standard / Engineering Formulation |
|---|---|
| Tire MF6.1 | Pacejka Magic Formula MF6.1 / Siemens *MF-Tool Equation Manual FITTYP=61* |
| Handling & Understeer | SAE J670e / ISO 4138 (Steady-State Directional Control) |
| DNA Maps n1…n6 | Modal Stability Analysis & State-Space Curvature Gradients |
| Moment Method / YMD | Yaw Moment vs Lateral Acceleration Balance Diagram |
| Roll Center & LLT | Geometric & Elastic Lateral Load Transfer Formulation |
| ARB & Suspension Sizing | Dual-Rate Series Spring Connection & 4-Ratio Tuning Framework |
| Direct Collocation | Optimal Control & Trajectory Optimization (IPOPT / Curvilinear Coordinates) |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests are welcome. Before submitting:

1. Run `python -m pytest tests/ -v` and ensure all physics tests pass.
2. Follow the code style: English source code, physical units in comments `[unit]`, CasADi-compatible functions.
3. Reference the physical equation and its source when adding new models.
