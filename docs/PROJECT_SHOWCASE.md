# Dynamic Target Suite — Project Showcase & Onboarding Guide

> **Audience:** Engineers, researchers, or team members encountering this project for the first time.  
> **Purpose:** Understand what the suite does, what to run to see it in action, which code is most significant, and which pitfalls to avoid.  
> For deep mathematics and architecture details, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## Table of Contents

1. [What Is Dynamic Target Suite?](#1-what-is-dynamic-target-suite)
2. [Quick Start — Running the Application](#2-quick-start--running-the-application)
3. [Repository Map at a Glance](#3-repository-map-at-a-glance)
4. [Top Simulations to Show](#4-top-simulations-to-show)
   - 4.1 [YMD & Handling Diagram](#41-ymd--handling-diagram)
   - 4.2 [Direct Collocation — Optimal Racing Line](#42-direct-collocation--optimal-racing-line)
   - 4.3 [Transient 1D Sensitivity Sweep](#43-transient-1d-sensitivity-sweep)
   - 4.4 [Interactive 14-DOF Driving Simulator](#44-interactive-14-dof-driving-simulator)
   - 4.5 [Skidpad & Acceleration Event Scoring](#45-skidpad--acceleration-event-scoring)
   - 4.6 [7-Post Rig Frequency Response](#46-7-post-rig-frequency-response)
   - 4.7 [Vehicle DNA Maps (n1–n6)](#47-vehicle-dna-maps-n1n6)
5. [Most Significant Code Sections](#5-most-significant-code-sections)
   - 5.1 [CasADi Symbolic ODE — 14-DOF Compilation Pipeline](#51-casadi-symbolic-ode--14-dof-compilation-pipeline)
   - 5.2 [Pacejka MF6.1 Tire Model — CasADi-Compatible](#52-pacejka-mf61-tire-model--casadi-compatible)
   - 5.3 [VehicleParams — Single Source of Truth](#53-vehicleparams--single-source-of-truth)
   - 5.4 [MDO Global Optimizer — Multi-Start SLSQP](#54-mdo-global-optimizer--multi-start-slsqp)
   - 5.5 [QThread Workers — Async Computation Pattern](#55-qthread-workers--async-computation-pattern)
6. [Critical Conventions & Pitfalls](#6-critical-conventions--pitfalls)
7. [Test Suite — How to Verify Everything Works](#7-test-suite--how-to-verify-everything-works)
8. [Standalone Executable](#8-standalone-executable)

---

## 1. What Is Dynamic Target Suite?

**Dynamic Target Suite** is an integrated engineering workbench for **Formula SAE / Formula Student electric race cars**. It covers the full vehicle setup cycle:

```
Geometric & Physical Parameters
        │
        ▼
   MDO / Trim Optimization          ← SLSQP multi-start, OpenMDAO
        │
        ▼
  Handling Maps & DNA Maps          ← YMD, Moment Method, Achievable Performance arcs
        │
        ▼
  Transient Validation              ← 14-DOF ODE, CasADi + RK4
        │
        ▼
  Real-Time Interactive Simulator   ← 500 Hz physics thread, gamepad/FFB
```

**Key capabilities:**

| Domain | What it does |
|--------|-------------|
| **Vehicle Models** | 3-DOF (quasi-static), 7-DOF (transient yaw/roll), 14-DOF (full suspension + 4-wheel) |
| **Tire** | Pacejka MF6.1 (non-linear, symbolic, CasADi-compatible), thermal degradation |
| **MDO** | SLSQP multi-start optimizer, direct collocation racing line (CasADi + IPOPT) |
| **Events** | Step Steer, Double Lane Change, Skidpad, Acceleration, Endurance, 7-Post Rig |
| **Analysis** | YMD / Handling diagrams, Vehicle DNA maps, 1D sensitivity sweep (21 KPIs), uncertainty propagation |
| **Simulator** | Real-time 14-DOF at 250–500 Hz with gamepad / force-feedback wheel |

---

## 2. Quick Start — Running the Application

> [!IMPORTANT]
> Always use the **local virtual environment** interpreter. Never use a globally installed Python.

```powershell
# 1. Activate environment (optional, for interactive shell)
.\venv\Scripts\Activate.ps1

# 2. Launch the full GUI application
.\venv\Scripts\python.exe main.py
```

The GUI that opens is **DYNAMIC TARGET SUITE PRO** — a 5-workspace docked interface:

| Workspace | Purpose |
|-----------|---------|
| **Setup** | Edit vehicle parameters (mass, geometry, suspension, powertrain) |
| **MDO** | Run optimizations and view Pareto frontiers |
| **Analysis** | Handling diagrams, YMD, DNA maps, sensitivity sweeps |
| **Simulator** | Launch real-time 14-DOF interactive simulator |
| **Results** | KPI dashboards, event scoring, export |

**Entrypoint chain:**
[`main.py`](../main.py) → [`src/ui/main_gui.py`](../src/ui/main_gui.py) → [`MainWindowPro`](../src/ui/main_gui_pro.py)

---

## 3. Repository Map at a Glance

```
Dynamic_Target_Suite/
├── main.py                          ← Application entrypoint
├── src/
│   ├── models/                      ← Physical models (3/7/14-DOF, tires, powertrain)
│   ├── mdo/                         ← Optimization & event simulation engines
│   ├── core/                        ← Real-time physics thread, road profiles, OpenMDAO
│   ├── ui/                          ← PyQt5 GUI (Pro + Classic), workers, dialogs
│   └── utils/                       ← Track manager, DNA maps, sensitivity, curve digitizer
├── data/
│   ├── tire_tir/                    ← Pacejka .tir files (Hoosier 43075, SC24)
│   ├── tracks/                      ← Circuit coordinates (.csv)
│   ├── setups/                      ← Exported vehicle setups (.json)
│   └── motor_curve/                 ← EMRAX 188HV efficiency maps
├── tests/                           ← 218 unit tests (pytest / unittest)
├── docs/                            ← Developer guide, showcase, math notes
└── MDO_Target_Suite.spec            ← PyInstaller build spec
```

---

## 4. Top Simulations to Show

### 4.1 YMD & Handling Diagram

**What it shows:** The complete vehicle stability envelope — yaw moment *N* vs. lateral acceleration *ay* at fixed speed, with control margin, stability margin, and Ackermann reference lines overlaid.

**How to run:** GUI → Analysis Workspace → *Handling / YMD* tab → set speed + tire → *Run Sweep*.

**Key file:** [`src/mdo/mdo_target_optimization.py`](../src/mdo/mdo_target_optimization.py) — `generate_handling_sweep_classic()`

**What makes it impressive:**
- Full 4-wheel Pacejka non-linear sweep: front/rear slip angles iterated to equilibrium at each (β, δ) point
- Control margin CM = ∂N/∂δ and stability margin SM = −∂N/∂β automatically extracted and plotted
- Iso-radius arcs (Skidpad 9.125 m, Autocross turn) overlaid for direct FSAE scoring context
- Post-peak tracking via pseudo-arc-length continuation (no solution drops past peak lateral)

**Output to highlight:** The divergence of *N–ay* curves from ideal Ackermann, the neutral-steer point, and the crossover from understeer to oversteer gradient.

---

### 4.2 Direct Collocation — Optimal Racing Line

**What it shows:** The minimum-time racing line and velocity profile for any FSAE circuit track, solved as a continuous optimal control problem.

**How to run:** GUI → MDO Workspace → *Direct Collocation* → select track → *Solve*.

**Key file:** [`src/mdo/mdo_direct_collocation.py`](../src/mdo/mdo_direct_collocation.py) — `solve_direct_collocation_racing_line()`

**What makes it impressive:**
- Formulated in curvilinear coordinates (track progress *s*, lateral offset *n*, heading error *ξ*)
- State vector scaled to O(1) — energy state E = ½v² instead of speed; avoids IPOPT numerical scaling issues
- Combined friction ellipse constraint with aero downforce scaling
- 80 kW power budget constraint (FSAE EV regulation)
- Warm-start: backward-forward quasi-static velocity profiler gives IPOPT a feasible initial guess → typically converges in < 60 iterations

**Output to highlight:** The colored velocity map over track geometry, the curvature-forced braking zones, and the side-by-side comparison with the quasi-static profile.

---

### 4.3 Transient 1D Sensitivity Sweep

**What it shows:** How each setup parameter (camber, toe, ARB stiffness, spring rate, etc.) affects 21 transient KPIs in a step-steer manoeuvre.

**How to run:** GUI → Analysis Workspace → *Sensitivity* tab → select scenario + parameter range → *Run Sweep*.

**Key file:** [`src/utils/transient_sensitivity.py`](../src/utils/transient_sensitivity.py) — `run_sensitivity_sweep_1d()`

**KPIs captured per sweep (21 total):**

| Category | KPIs |
|----------|------|
| Yaw response | r_ss, r_peak, OS%, t90, Δt_yaw |
| Lateral dynamics | ay_peak, Δt_ay, TB-Factor, r_dot_peak |
| Stability | Gr, Gay, t_settle, β_peak |
| Load transfer | LLTD_peak, LLTD_ss, ΔFz_asym |
| Chassis | RollGrad, φ_max, η_μ |
| Tire utilization | Util_F_max, Util_R_max |

**Output to highlight:** The tornado chart showing parameter influence ranking, and how understeer gradient *K* changes with front/rear spring split.

---

### 4.4 Interactive 14-DOF Driving Simulator

**What it shows:** Real-time physics-based driving with full suspension, 4-wheel tire forces, camber, toe, and torque vectoring — playable with gamepad or force-feedback wheel.

**How to run:** GUI → Simulator Workspace → *Launch Interactive Simulator* → connect gamepad.

**Key files:**
- [`src/ui/interactive_driver_14dof.py`](../src/ui/interactive_driver_14dof.py) — OpenGL/PyQt5 viewport
- [`src/core/physics_thread.py`](../src/core/physics_thread.py) — `PhysicsThread14DOF` at 250–500 Hz
- [`src/models/vehicle_14dof.py`](../src/models/vehicle_14dof.py) — 28-state ODE

**What makes it impressive:**
- The ODE is compiled to a **native C shared library (.dll) via CasADi → C → MSVC** at first run, then cached with a SHA-256 hash of the physical parameters — subsequent launches use the cached DLL for near-zero overhead
- Full RK4 integration at 500 Hz in a dedicated `QThread` — GUI never blocks
- Torque vectoring (yaw-rate PI controller) runs in the same physics thread
- Live telemetry: tire slip angles, load transfer, camber, lateral acceleration — all streamed to PyQt5 plots at 60 Hz

**Output to highlight:** The live tire utilization ellipse and how it saturates in oversteer. The real-time load transfer bar between front and rear axles.

---

### 4.5 Skidpad & Acceleration Event Scoring

**What it shows:** Full FSAE Skidpad and 75 m Acceleration event simulation with FS2026 scoring formula applied.

**How to run:** GUI → MDO Workspace → *Events* → *Skidpad* / *Acceleration*.

**Key files:**
- [`src/mdo/mdo_skidpad.py`](../src/mdo/mdo_skidpad.py) — steady-state + transient Skidpad solver
- [`src/mdo/mdo_acceleration.py`](../src/mdo/mdo_acceleration.py) — longitudinal dynamics with traction control
- [`src/mdo/mdo_sensitivity.py`](../src/mdo/mdo_sensitivity.py) — FS2026 scoring constants + world ranking benchmarks

**What makes it impressive:**
- Skidpad uses the 9.125 m radius figure-8 with regulated entry/exit sequencing
- Acceleration integrates the EMRAX 188HV motor torque map against rolling resistance, aero drag, and tire traction limit — outputting a predicted time with TCS engagement history
- FS2026 scoring formula applied immediately: points out of 75 (Skidpad) and 100 (Acceleration) using world ranking benchmarks

---

### 4.6 7-Post Rig Frequency Response

**What it shows:** Suspension modal analysis under prescribed multi-axis shaker inputs (heave, pitch, roll, warp), computing contact patch load variation (CPLV) and body transmissibility.

**How to run:** GUI → MDO Workspace → *7-Post Rig* tab → configure frequency sweep → *Run*.

**Key files:**
- [`src/mdo/mdo_seven_post_rig.py`](../src/mdo/mdo_seven_post_rig.py) — core physics
- [`src/ui/ui_seven_post_rig.py`](../src/ui/ui_seven_post_rig.py) — dedicated workspace UI

**What makes it impressive:**
- Decoupled modal inputs: Heave (all 4 posts in-phase), Pitch (front vs. rear), Roll (left vs. right), Warp (diagonal pairs)
- Outputs natural frequencies, damping ratios, CPLV per corner at each frequency
- Direct correlation with Öhlins TTX25 damper maps ([`src/models/damper_model_ttx25.py`](../src/models/damper_model_ttx25.py)) — shows whether chosen damper settings produce optimal isolation

---

### 4.7 Vehicle DNA Maps (n1–n6)

**What it shows:** Six normalised handling maps (n1–n6) that characterise the vehicle's full nonlinear understeer–oversteer DNA across speed, lateral acceleration, and wheel load.

**How to run:** GUI → Analysis Workspace → *DNA Maps* tab → *Compute n1–n6*.

**Key file:** [`src/utils/vehicle_dna_maps.py`](../src/utils/vehicle_dna_maps.py)

**What makes it impressive:**
- Maps formulated from nonlinear handling theory — including the fundamental relation ρ_y = −K/L
- Each map is normalised to [−1, +1] so different vehicle configurations compare on the same scale
- Pseudo-arc-length continuation used to track branches past the limit-handling regime without losing the solution

---

## 5. Most Significant Code Sections

### 5.1 CasADi Symbolic ODE — 14-DOF Compilation Pipeline

**File:** [`src/models/vehicle_14dof.py`](../src/models/vehicle_14dof.py)

The most technically demanding piece of the project. The 28-state vehicle ODE is written in CasADi's **symbolic** framework (`casadi.SX`), which allows:

1. **Automatic differentiation** — exact Jacobians for IPOPT, no finite differences
2. **C code generation** — the symbolic graph is emitted as C source via `casadi.CodeGenerator`
3. **Native DLL compilation** — C is compiled with MSVC/`cl.exe` to a `.dll` that runs at near-C speed inside Python via `ctypes`
4. **Hash-based caching** — the compiled DLL is stored in `.cache/c_build/ode_14dof_<sha256>.dll` keyed on all physics parameters; parameter changes trigger automatic recompilation

```python
# Conceptual pipeline (see vehicle_14dof.py for full implementation)
x    = ca.SX.sym('x', 28)  # state vector (positions, velocities, tire states)
u    = ca.SX.sym('u',  4)  # inputs (steering, throttle, brake, TV torque)
xdot = vehicle_14dof_rhs(x, u, params)  # symbolic expression tree

ode_fn = ca.Function('ode', [x, u], [xdot])
gen = ca.CodeGenerator('ode_14dof.c')
gen.add(ode_fn)
gen.generate()   # → ode_14dof.c  (CasADi C runtime)
# then: cl.exe → ode_14dof.dll  (native, cached in .cache/c_build/)
```

> [!NOTE]
> The ODE is **always** written with CasADi-smooth primitives. No Python `if/else` on symbolic variables — use `ca.if_else`, `ca.tanh`, `ca.fmax` and smooth regularisation sqrt(x² + ε²) for magnitude operations.

---

### 5.2 Pacejka MF6.1 Tire Model — CasADi-Compatible

**File:** [`src/models/tire_model_mf61.py`](../src/models/tire_model_mf61.py)  
**Data:** [`src/models/params_tire.py`](../src/models/params_tire.py) + `.tir` files in [`data/tire_tir/`](../data/tire_tir/)

The full MF6.1 formulation (combined-slip, camber effects, normalised coefficients) implemented in CasADi symbolic math so it can be:
- Embedded inside the 14-DOF ODE (for compiled DLL execution)
- Differentiated exactly for IPOPT in direct collocation
- Evaluated in pure NumPy for quick offline analysis

Key non-linearities handled smoothly:
- Slip saturation: μ·Fz limit approached via smooth `tanh` shaping
- Combined slip: the Brush/MF weighting factors G_xα, G_yκ maintain correct coupling
- TYRESIDE mirroring: a single set of coefficients fits both left and right wheels by sign-flipping camber and lateral velocity

> [!IMPORTANT]
> **TYRESIDE convention:** The `.tir` files in this project use `RIGHT` side as the reference (Hoosier 43075). For left-side wheels, camber is negated (γ_left = −γ_raw) and the lateral velocity sign is flipped. **Do not modify this mirroring logic** without re-validating all test cases.

---

### 5.3 VehicleParams — Single Source of Truth

**File:** [`src/models/params_vehicle.py`](../src/models/params_vehicle.py)

Every physical and geometric property of the car lives in the `VehicleParams` dataclass — mass, wheelbase, track widths, inertias, suspension geometry, aerodynamic coefficients, powertrain parameters.

```python
from src.models.params_vehicle import VehicleParams, nominal_fsae_params

p = nominal_fsae_params()   # FSAE E18 nominal configuration
p.mass_total = 230.0        # override any field
```

**Rule:** No module is allowed to hardcode a geometric or physical constant. All values come from a `VehicleParams` instance. The SHA-256 hash used for DLL caching is computed from this dataclass — any parameter change triggers automatic ODE recompilation without manual intervention.

---

### 5.4 MDO Global Optimizer — Multi-Start SLSQP

**File:** [`src/mdo/mdo_global_optimizer.py`](../src/mdo/mdo_global_optimizer.py)

The top-level optimizer wraps `scipy.optimize.minimize(method='SLSQP')` with:
- **Multi-start Latin Hypercube Sampling** — explores the parameter space before local convergence
- **Stability derivatives** computed from YMD gradients — ∂N/∂δ, ∂N/∂β used as constraints
- **Standard understeer gradient** K = −L·ρ_y fed as a soft constraint with target band
- **OpenMDAO** used for disciplinary coupling when wheel loads and aerodynamics interact non-trivially

The objective function is a weighted sum of event scores (Skidpad, Acceleration, Endurance lap time) computed by calling the respective MDO modules.

---

### 5.5 QThread Workers — Async Computation Pattern

**File:** [`src/ui/gui_workers.py`](../src/ui/gui_workers.py)

Every heavy computation is wrapped in a `QThread`-based worker that:
1. Receives parameters via Python signals
2. Runs the blocking computation (IPOPT, SLSQP, ODE integration) in a **separate thread**
3. Emits `progress(int, str)` every N iterations for live progress bars
4. Emits `finished(result_dict)` on completion
5. Emits `error(str)` on exception — never crashes the GUI

```python
# Pattern used everywhere in the project
class DirectCollocationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def run(self):
        try:
            result = solve_direct_collocation_racing_line(
                self.track, self.params,
                progress_cb=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

> [!CAUTION]
> **Never** call `mdo_*`, ODE integration, or optimization functions directly from a slot connected to a GUI button. Always dispatch via a worker. Blocking the main thread freezes the Qt event loop and makes the window unresponsive.

---

## 6. Critical Conventions & Pitfalls

### Python Environment
```powershell
# Correct
.\venv\Scripts\python.exe main.py
.\venv\Scripts\python.exe -m pytest tests/

# Wrong — misses CasADi, OpenMDAO, PyQt5
python main.py
```

### Curvature-Velocity Understeer Gradient Sign Convention
The understeer gradient is defined as:

    K = −L · ρ_y     where  ρ_y = ∂ρ/∂ay

- **Understeer:** K > 0 (positive K, negative ρ_y)
- **Oversteer:** K < 0
- **Neutral steer:** K = 0
- **Stability condition:** ρ_y < 1/u²

> [!WARNING]
> **Never invert** the sign of ρ_y or K. Several older references use the opposite convention. The entire DNA map and handling diagram system is built on this sign convention — a sign flip would silently produce wrong stability assessments.

### CasADi Symbolic Variables
```python
# Correct — smooth CasADi primitives
abs_v   = ca.sqrt(v**2 + 1e-6)                      # regularised |v|
clamped = ca.fmax(0.0, ca.fmin(1.0, x))             # clamp [0, 1]
sign_v  = ca.tanh(v / 0.01)                         # smooth sign

# Wrong — Python control flow on symbolic values
if v > 0:      # raises CasADi exception or silently evaluates wrong
    ...
abs_v = abs(v) # Python abs() does not dispatch to CasADi
```

### Tire File Discovery
`TireParams._resolve_path()` searches in this order:
1. Exact path as given
2. `data/tire_tir/<filename>`
3. `tire_tir/<filename>` (legacy layout)
4. Current working directory, repo root, model source directory

If a `.tir` file is not found, the constructor raises `FileNotFoundError` with the full search path listed — check this list before reporting a missing file.

### DLL Cache Location
Compiled ODE DLLs land in `.cache/c_build/`. If the cache grows large or a DLL becomes corrupt:
```powershell
Remove-Item .cache\c_build\*.dll -Force
# Next run recompiles automatically (adds ~10–15 s at first launch)
```

### openmdao Resource Files
When running from the compiled EXE, `openmdao` must exist as a raw directory in `_internal/` — not compressed in the PYZ archive. This is already configured in [`MDO_Target_Suite.spec`](../MDO_Target_Suite.spec). If you see `FileNotFoundError: unit_library.ini`, the spec lost its `exclude_module` / `collect_all('openmdao')` configuration.

### TYRESIDE in `.tir` Files
```
TYRESIDE = 'RIGHT'   ← Hoosier 43075 reference
```
Left-wheel physics is computed by negating camber and flipping the lateral velocity sign **automatically** inside `tire_model_mf61.py` based on an `is_left_wheel` flag. Do not pre-negate values before passing them to the model.

---

## 7. Test Suite — How to Verify Everything Works

The test suite has **218 tests** covering all physical models, events, and analysis utilities.

```powershell
# Run all tests (takes ~7 minutes — ODE compilation included)
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"

# Run only fast model tests
.\venv\Scripts\python.exe -m pytest tests/test_tire_mf61.py tests/test_3dof.py tests/test_damper_ttx25.py -v

# Run the unified CI pipeline (tests + headless GUI + health check)
.\venv\Scripts\python.exe scripts/ci_pipeline.py
```

**Key test files to know:**

| Test file | What it validates |
|-----------|------------------|
| `tests/test_14dof.py` | 14-DOF ODE: state propagation, DLL cache, camber/toe effects |
| `tests/test_tire_mf61.py` | MF6.1: combined slip, camber stiffness, TYRESIDE symmetry |
| `tests/test_mmm_hd_verification.py` | YMD / Handling diagram: CM/SM signs, neutral-steer point |
| `tests/test_direct_collocation_smoke.py` | DC optimizer: IPOPT convergence, constraint satisfaction |
| `tests/test_transient_sensitivity.py` | All 21 KPIs: sign, magnitude, monotonicity checks |
| `tests/test_vehicle_dna_maps.py` | DNA maps n1–n6: standard understeer gradient sign convention |
| `tests/test_dynamic_events.py` | Step Steer, DLC, Skidpad: physics sanity bounds |
| `tests/test_powertrain_electric.py` | EMRAX torque map, regeneration, TV allocation |
| `tests/test_acceleration_and_diamond.py` | 75 m event timing, ARB Diamond map |

---

## 8. Standalone Executable

For demos without a Python environment:

```powershell
# Build (from repo root, takes ~5–10 minutes)
.\venv\Scripts\pyinstaller.exe --noconfirm MDO_Target_Suite.spec

# Run
.\dist\MDO_Target_Suite\MDO_Target_Suite.exe
```

The built EXE bundles: PyQt5, CasADi, OpenMDAO, Matplotlib, SciPy, NumPy, Pygame, Pandas.  
Tire data files (`data/tire_tir/`) are copied into `_internal/tire_tir/` automatically during the build.

> [!TIP]
> If you distribute the EXE, always ship the entire `dist/MDO_Target_Suite/` folder — the executable alone without `_internal/` will crash immediately.

---

*Guide maintained by the E18 Vehicle Dynamics team. Last updated: October 2026.*  
*For mathematical derivations, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md). For contribution rules, see [`AGENTS.md`](AGENTS.md) and [`GEMINI.md`](GEMINI.md).*
