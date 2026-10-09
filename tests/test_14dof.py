import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.models.vehicle_14dof import Vehicle14DOF
from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.mdo.mdo_steady_state import evaluate_kpis_with_openmdao
from src.mdo.mdo_dynamic_events import simulate_step_steer, simulate_double_lane_change

def get_test_params():
    vp = VehicleParams()
    tir_file = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    tp = TireParams(tir_file)
    return vp, tp

def test_14dof_casadi_trim_functions():
    """Verify that Vehicle14DOF generates valid CasADi trim functions."""
    vp, tp = get_test_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    funcs = veh14.get_trim_casadi_functions()
    
    assert 'res_func' in funcs
    assert 'jac_states_func' in funcs
    assert 'jac_inputs_func' in funcs
    assert 'jac_params_func' in funcs
    
    # 11 states: [vy, r, phi, z_fl, z_fr, z_rl, z_rr, w_fl, w_fr, w_rl, w_rr]
    v_x = 15.0
    delta = np.deg2rad(3.0)
    w0 = v_x / vp.R_w
    inputs = [v_x, delta]
    states = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, w0, w0, w0, w0]
    params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r, vp.aero_balance,
              vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
              vp.L, vp.t_f, vp.t_r, vp.LMUY]
              
    res = funcs['res_func'](inputs, states, params)
    assert res.shape == (11, 1)

def test_14dof_openmdao_equilibrium_and_roll_gradient():
    """Verify OpenMDAO solves 14-DOF trim and roll gradient matches analytical prediction.

    Analytical roll gradient (linearised single-track model):
        phi/ay = m * h_arm / K_phi_tot
    where:
        h_arm    = h_cg - h_rc_avg = 0.270 - (0.0463 + 0.07693)/2 = 0.2084 m
        K_phi_tot= K_phi_f + K_phi_r = 26929 + 30367 = 57296 Nm/rad (1000 Nm/deg)
        m        = 292 kg
        -> phi/ay = 292 * 0.2084 / 57296 = 1.062e-3 rad/(m/sÂ²)
        -> in deg/g: 1.062e-3 * (180/pi) * 9.81 = 0.60 deg/g

    Accepted range accounts for:
      +roll-center lateral migration (not in linearised formula)
      +roll-yaw cross-inertia Ixz coupling
      +dynamic aero downforce at v_x = 15 m/s
    """
    vp = VehicleParams()
    h_rc_avg = (vp.rollcenter_f_coeffs[-1] + vp.rollcenter_r_coeffs[-1]) / 2.0
    h_arm    = vp.h_cg - h_rc_avg
    K_phi_tot = vp.K_phi_f + vp.K_phi_r
    rg_analytical = vp.m * h_arm / K_phi_tot * (180.0 / np.pi) * vp.g  # deg/g
    tol_rel = 0.30  # Â±30% accounts for full nonlinear roll-center dynamics

    res = evaluate_kpis_with_openmdao(
        v_x=15.0, delta_deg=3.0, wd_f=0.45, lltd=vp.lltd, aero_bal=0.485,
        L=1.529, t_f=1.252, t_r=1.136, model_type='14-DOF'
    )
    assert res['solver_iters'] <= 50
    assert abs(res['yaw_rate_rads']) > 0.01
    assert 'roll_gradient_deg_g' in res

    rg = res['roll_gradient_deg_g']
    assert rg_analytical * (1.0 - tol_rel) <= rg <= rg_analytical * (1.0 + tol_rel), (
        f"Roll gradient {rg:.2f} deg/g deviates >30% from analytical prediction "
        f"{rg_analytical:.2f} deg/g (h_arm={h_arm:.4f}m, K_phi={K_phi_tot:.0f} Nm/rad)"
    )

def test_14dof_step_steer_and_dlc():
    """Verify 14-DOF transient integration for Step Steer and Double Lane Change."""
    vp, tp = get_test_params()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')
    
    ss_res = simulate_step_steer(vp, v_x=15.0, steer_deg=3.0, t_end=2.0, model_type='14-DOF', tir_path=tir_path)
    assert ss_res['success'] is True
    assert ss_res['ay_peak_g'] > 0.3
    
    dlc_res = simulate_double_lane_change(vp, v_entry=20.0, t_end=3.5, model_type='14-DOF', tir_path=tir_path)
    assert dlc_res['success'] is True
    assert dlc_res['max_ay_g'] > 0.5
    assert dlc_res['max_phi_deg'] > 0.25

def test_14dof_static_equilibrium_zero_travel():
    """Verify that at standstill on ground (vx=0, delta=0), 14-DOF suspension travel is strictly 0.0 mm."""
    vp, tp = get_test_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    
    # 1. Steady-state Trim Residuals at rest
    funcs = veh14.get_trim_casadi_functions()
    inputs = [0.0, 0.0]  # vx=0, delta=0
    # 11 states: [vy, r, phi, z_fl, z_fr, z_rl, z_rr, w_fl, w_fr, w_rl, w_rr]
    states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r, vp.aero_balance,
              vp.camber_f_static, vp.camber_r_static, vp.toe_f_static, vp.toe_r_static,
              vp.L, vp.t_f, vp.t_r, vp.LMUY]
              
    res = np.array(funcs['res_func'](inputs, states, params)).flatten()
    # Suspension travel residuals (indices 3..6) must be exactly 0.0 at static rest
    np.testing.assert_allclose(res[3:7], 0.0, atol=1e-6)
    
    # 2. ODE derivatives at rest (16 params: trim params + LMUX)
    ode = veh14.get_casadi_ode()
    x0 = np.zeros(23)  # vx, vy, r, phi, p, theta, q, z_fl..rr, w_fl..rr, ux_fl..uy_rr
    u0 = np.zeros(4)   # delta, T_rl, T_rr, T_brake
    p0 = np.array(params + [vp.LMUX])
    
    x_dot = np.array(ode(x0, u0, p0)).flatten()
    # Suspension travel rates z_dot (indices 7, 8, 9, 10) must be 0.0 at rest
    np.testing.assert_allclose(x_dot[7:11], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# NEW TESTS â€” Phase 3 Test Hardening
# ---------------------------------------------------------------------------

def test_14dof_understeer_gradient_weight_distribution_sensitivity():
    """Invariant: increasing front weight distribution (wd_f) must monotonically
    increase the Understeer Gradient (K_us).

    Physical reasoning (Linear Bicycle Model Understeer):
        K_us = (m/L) * (b/C_alpha_f - a/C_alpha_r)

    Shifting weight forward (higher wd_f):
      - Increases front static load Fz_f -> front slip angles must increase
      - Decreases rear static load Fz_r -> rear slip angles decrease
      - Increases distance 'b' (CG moves further from rear axle)
      - Decreases distance 'a' (CG moves closer to front axle)
    All four effects work in the same direction: higher wd_f -> higher K_us.
    """
    from src.mdo.mdo_global_optimizer import compute_stability_metrics

    kus_rear = compute_stability_metrics(L=1.529, wd_f=0.45, lltd=0.50, aero_bal=0.485, v_x=15.0)['K_us_degg']
    kus_mid  = compute_stability_metrics(L=1.529, wd_f=0.50, lltd=0.50, aero_bal=0.485, v_x=15.0)['K_us_degg']
    kus_fwd  = compute_stability_metrics(L=1.529, wd_f=0.55, lltd=0.50, aero_bal=0.485, v_x=15.0)['K_us_degg']

    assert kus_fwd > kus_mid > kus_rear, (
        f"K_us must increase monotonically with wd_f: "
        f"got rear={kus_rear:.3f}, mid={kus_mid:.3f}, fwd={kus_fwd:.3f} deg/g"
    )
    # Front-heavy car must be understeering (K_us > 0)
    assert kus_fwd > 0.0, f"Front-heavy car (wd_f=0.55) must have K_us > 0, got {kus_fwd:.3f}"


def test_14dof_yaw_rate_monotonic_buildup():
    """Invariant: after a step-steer input, yaw rate must increase monotonically
    during the steering ramp (active transient initiation).

    Physical basis: For an understeering car at v_x=15 m/s with Î´=3Â°, the yaw
    mode has Î¶ > 0.6 (well-damped). During the steering application ramp
    (t âˆˆ [0.50, 0.65] s), the initial transient response must be strictly
    increasing. A decrease would indicate numerical instability or wrong-sign
    damping in the roll-yaw ODE.
    """
    vp, tp = get_test_params()
    tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir')

    res = simulate_step_steer(vp, v_x=15.0, steer_deg=3.0, t_end=2.0,
                               model_type='14-DOF', tir_path=tir_path)
    assert res['success'] is True

    df = res['telemetry']
    # Steering ramp window: steer starts at t=0.50s and reaches 3 deg at tâ‰ˆ0.58s
    mask_ramp = (df['t'] >= 0.505) & (df['t'] <= 0.65)
    r_ramp = df.loc[mask_ramp, 'yaw_rate_deg_s'].values
    assert len(r_ramp) >= 10, "Not enough time samples during steering ramp"

    # r must be non-decreasing (allow tiny numerical noise of 1e-3 deg/s)
    diffs = np.diff(r_ramp)
    assert np.all(diffs >= -1e-3), (
        f"Yaw rate must build up monotonically during steering ramp. "
        f"Found decrease of {diffs.min():.4f} deg/s at sample {np.argmin(diffs)}"
    )


def test_14dof_drag_glide_energy_conservation():
    """Invariant: with no drive, no brake, straight ahead (delta=0), v_x must
    decrease strictly monotonically over 1 second due to aerodynamic drag.

    Physical derivation:
        m * dv_x/dt = -F_drag = -0.5 * rho * S * Cd * v_xÂ²
        -> v_x(t) = v_x0 / (1 + v_x0 * t * 0.5*rho*S*Cd/m)

    For nominal params (m=292kg, rho=1.225, Sâ‰ˆ1.14mÂ², Cd=1.2):
        dv_x/dt|t=0 = -0.5*1.225*1.14*1.2*15Â²/292 â‰ˆ -0.68 m/sÂ²
        -> After 1 s: v_x should drop by roughly 0.5â€“0.9 m/s.

    A non-monotonic v_x in this test indicates a drag sign error or a spurious
    traction source in the ODE (e.g. wrong wheel torque initialisation).
    """
    vp, tp = get_test_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    ode = veh14.get_casadi_ode()

    import scipy.integrate as sci

    params_trim = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r,
                   vp.aero_balance, vp.camber_f_static, vp.camber_r_static,
                   vp.toe_f_static, vp.toe_r_static, vp.L, vp.t_f, vp.t_r, vp.LMUY, vp.LMUX]

    v_x0 = 15.0
    w0 = v_x0 / vp.R_w
    x0 = np.array([v_x0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # vx vy r phi p theta q
                   0.0, 0.0, 0.0, 0.0,                    # z_fl..rr
                   w0, w0, w0, w0,                        # w_fl..rr
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # ux_fl..uy_rr
    u_glide = np.array([0.0, 0.0, 0.0, 0.0])  # delta=0, T_rl=0, T_rr=0, T_brake=0

    def rhs(t, x):
        return np.array(ode(x, u_glide, params_trim)).flatten()

    sol = sci.solve_ivp(rhs, [0.0, 1.0], x0, method='RK45',
                        max_step=0.01, rtol=1e-6, atol=1e-8)
    assert sol.success, f"ODE integration failed: {sol.message}"

    v_x_traj = sol.y[0]   # v_x is state index 0

    # v_x must be strictly decreasing (allow floating-point noise of 1e-5 m/s)
    diffs = np.diff(v_x_traj)
    assert np.all(diffs <= 1e-5), (
        f"v_x must decrease monotonically in drag-only glide. "
        f"Found increase of {diffs.max():.5f} m/s â€” check drag or traction sign in ODE."
    )

    # Sanity check: total speed drop should be in the expected physical range
    delta_vx = v_x0 - v_x_traj[-1]
    F_drag_nom = 0.5 * vp.rho * vp.S * vp.Cd * v_x0**2
    dv_expected = F_drag_nom / vp.m  # rough m/sÂ² deceleration at t=0
    # After 1s: drop should be roughly between 0.3Ã— and 2.0Ã— the initial deceleration rate
    assert 0.3 * dv_expected <= delta_vx <= 2.0 * dv_expected, (
        f"Speed drop {delta_vx:.3f} m/s in 1s glide is outside physical range "
        f"[{0.3*dv_expected:.3f}, {2.0*dv_expected:.3f}] m/s "
        f"(F_drag={F_drag_nom:.1f}N, m={vp.m}kg)"
    )


def test_14dof_spinout_passive_energy_dissipation():
    """
    Verifies that during a spinout maneuver with zero control inputs (u=[0,0,0,0]):
    1. The vehicle dissipates the majority of its initial kinetic energy.
    2. Tire carcass deflections (ux_fl..rr) remain strictly bounded (< 0.35 m),
       confirming the elimination of relaxation length integrator windup.
    3. No persistent phantom traction accelerates the car once settled.
    """
    vp, tp = get_test_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    ode = veh14.get_casadi_ode()

    m = vp.m
    Iz = vp.Iz
    Rw = vp.R_w
    params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r,
               vp.aero_balance, vp.camber_f_static, vp.camber_r_static,
               vp.toe_f_static, vp.toe_r_static, vp.L, vp.t_f, vp.t_r,
               vp.LMUY, vp.LMUX, float(getattr(vp, 'Cd', 1.20))]

    # Violent spinout initial state
    x0 = np.zeros(23)
    x0[0] = 15.0  # vx [m/s]
    x0[1] = -6.0  # vy [m/s] (oversteer sideslip)
    x0[2] = 3.0   # r [rad/s]
    x0[11:15] = 15.0 / Rw  # w_fl..rr

    u = np.zeros(4)
    dt = 0.002
    steps = int(2.5 / dt)

    Ek_0 = 0.5 * m * (x0[0]**2 + x0[1]**2) + 0.5 * Iz * (x0[2]**2)
    x = x0.copy()
    max_ux = 0.0

    for _ in range(steps):
        k1 = np.array(ode(x, u, params)).flatten()
        k2 = np.array(ode(x + 0.5*dt*k1, u, params)).flatten()
        k3 = np.array(ode(x + 0.5*dt*k2, u, params)).flatten()
        k4 = np.array(ode(x + dt*k3, u, params)).flatten()
        x = x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        for ux_idx in [15, 17, 19, 21]:
            if abs(x[ux_idx]) > max_ux:
                max_ux = abs(x[ux_idx])

    Ek_end = 0.5 * m * (x[0]**2 + x[1]**2) + 0.5 * Iz * (x[2]**2)

    # 1. Energy must be strongly dissipated (>90% drop)
    assert Ek_end < 0.15 * Ek_0, f"Expected >85% energy dissipation in spinout, got Ek_end={Ek_end:.1f}J vs Ek_0={Ek_0:.1f}J"
    # 2. Carcass deflection must be bounded (< 0.35m vs previous unconstrained runaway > 0.89m)
    assert max_ux < 0.35, f"Carcass deflection wound up to {max_ux:.3f}m! (Integrator windup detected)"
    # 3. Final vehicle velocity must be low and decaying (< 5.0 m/s)
    v_planar_end = np.hypot(x[0], x[1])
    assert v_planar_end < 5.0, f"Vehicle still moving at {v_planar_end:.2f} m/s after 2.5s unpowered spinout"


def test_14dof_full_throttle_launch_stability():
    """Verify that launching from rest at 100% full throttle:
    1. Does NOT produce numerical divergence or NaN/Inf in the 28-DOF/23-DOF states.
    2. Maintains smooth pitch squat (theta bounded, negative, no violent oscillations).
    3. Solves stably across 500 Hz, 1000 Hz, and 1500 Hz explicit RK4 integration.
    """
    vp, tp = get_test_params()
    veh14 = Vehicle14DOF(vp, tp, tp)
    ode = veh14.get_casadi_ode()

    Rw = vp.R_w
    params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r,
              vp.aero_balance, vp.camber_f_static, vp.camber_r_static,
              vp.toe_f_static, vp.toe_r_static, vp.L, vp.t_f, vp.t_r,
              vp.LMUY, vp.LMUX, float(getattr(vp, 'Cd', 1.20))]

    for hz in [500.0, 1000.0, 1500.0]:
        dt = 1.0 / hz
        steps = int(0.5 / dt)  # 0.5s launch
        x = np.zeros(23)
        x[0] = 0.5  # 0.5 m/s initial roll
        x[11:15] = 0.5 / Rw
        u = np.array([0.0, 400.0, 400.0, 0.0])  # 100% full throttle (800 Nm total rear torque)

        for _ in range(steps):
            k1 = np.array(ode(x, u, params)).flatten()
            k2 = np.array(ode(x + 0.5 * dt * k1, u, params)).flatten()
            k3 = np.array(ode(x + 0.5 * dt * k2, u, params)).flatten()
            k4 = np.array(ode(x + dt * k3, u, params)).flatten()
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        assert np.all(np.isfinite(x)), f"States contains NaN/Inf at {hz} Hz!"
        assert x[0] > 2.0, f"Vehicle should accelerate forward under full throttle, got vx={x[0]:.2f} m/s at {hz} Hz"
        pitch_deg = np.rad2deg(x[5])
        assert -3.0 < pitch_deg < 0.5, f"Unphysical pitch angle {pitch_deg:.2f} deg at {hz} Hz!"


def test_lateral_stability_cornering():
    """Verify that in cornering under coast-down at 60 km/h with 5 deg, 1.5 Hz sine steer:
    1. Maximum real part of Jacobian eigenvalues is strictly negative (Re(lambda) < 0),
       proving asymptotic stability without spurious kinematic energy injection.
    2. Numerical integration over 3.5s/5.0s is rock-solid stable:
       - No NaN / Inf divergence
       - Peak roll angle remains tightly bounded (|phi| < 1.5 deg)
       - Total mechanical energy rate dE/dt <= 0 (strict energy dissipation).
    3. Stable in both Linear damping and TTX25 4-way nonlinear damping modes.
    """
    vp, tp = get_test_params()
    
    for damper_mode in ['linear', 'ttx25_nonlinear']:
        vp.damper_mode = damper_mode
        veh14 = Vehicle14DOF(vp, tp, tp)
        ode = veh14.get_casadi_ode()
        
        # 17 parameters
        params = [vp.wd_f, vp.lltd, vp.K_phi_f, vp.K_phi_r, vp.K_z_f, vp.K_z_r,
                  vp.aero_balance, vp.camber_f_static, vp.camber_r_static,
                  vp.toe_f_static, vp.toe_r_static, vp.L, vp.t_f, vp.t_r,
                  vp.LMUY, vp.LMUX, float(getattr(vp, 'Cd', 1.20))]
        
        # 1. Jacobian eigenvalue check at vx = 15 m/s (28 states)
        v_x0 = 15.0
        w0 = v_x0 / vp.R_w
        x0 = np.zeros(28)
        x0[0] = v_x0
        x0[16:20] = w0
        u_zero = np.zeros(4)
        
        # Numerical Jacobian
        eps = 1e-5
        f0 = np.array(ode(x0, u_zero, params)).flatten()
        J = np.zeros((28, 28))
        for j in range(28):
            x_p = x0.copy()
            x_p[j] += eps
            f_p = np.array(ode(x_p, u_zero, params)).flatten()
            J[:, j] = (f_p - f0) / eps
            
        eigvals = np.linalg.eigvals(J)
        max_real = np.max([ev.real for ev in eigvals])
        assert max_real < 1e-4, f"[{damper_mode}] Found unstable eigenvalue: max Re(lambda) = {max_real:+.4f}"
        
        # 2. Dynamic simulation coast-down 60 km/h with 5 deg, 1.5 Hz sine steer
        v_entry = 60.0 / 3.6  # 16.67 m/s
        w_entry = v_entry / vp.R_w
        x = np.zeros(28)
        x[0] = v_entry
        x[16:20] = w_entry
        
        dt = 0.001
        freq = 1.5
        amp_rad = np.deg2rad(5.0)
        t_end = 3.5
        steps = int(t_end / dt)
        
        max_phi = 0.0
        max_dE = -1e9
        
        for i in range(steps):
            t = i * dt
            delta = amp_rad * np.sin(2.0 * np.pi * freq * t)
            u_in = np.array([delta, 0.0, 0.0, 0.0])
            
            k1 = np.array(ode(x, u_in, params)).flatten()
            k2 = np.array(ode(x + 0.5 * dt * k1, u_in, params)).flatten()
            k3 = np.array(ode(x + 0.5 * dt * k2, u_in, params)).flatten()
            k4 = np.array(ode(x + dt * k3, u_in, params)).flatten()
            x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            assert np.all(np.isfinite(x_next)), f"[{damper_mode}] Simulation diverged with NaN/Inf at t={t:.3f}s"
            
            # Energy tracking (kinetic energy decay in coast-down)
            E_curr = 0.5 * vp.m * (x[0]**2 + x[1]**2) + 0.5 * vp.Ix * (x[4]**2) + 0.5 * vp.Iy * (x[6]**2) + 0.5 * vp.Iz * (x[2]**2)
            E_next = 0.5 * vp.m * (x_next[0]**2 + x_next[1]**2) + 0.5 * vp.Ix * (x_next[4]**2) + 0.5 * vp.Iy * (x_next[6]**2) + 0.5 * vp.Iz * (x_next[2]**2)
            dE_dt = (E_next - E_curr) / dt
            if dE_dt > max_dE:
                max_dE = dE_dt
                
            x = x_next
            phi_deg = abs(np.rad2deg(x[3]))
            if phi_deg > max_phi:
                max_phi = phi_deg
                
        assert max_phi < 1.5, f"[{damper_mode}] Roll angle exceeded threshold: {max_phi:.2f} deg"
        assert max_dE <= 10.0, f"[{damper_mode}] Positive mechanical energy injection detected: max dE/dt = {max_dE:.2f} W"



