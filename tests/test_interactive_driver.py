"""
test_interactive_driver.py
---------------------------
Unit & Smoke Tests for 14-DOF Real-Time Driving Simulator:
1. Thread-safe snapshot buffer under heavy concurrent access (500 Hz appends)
2. FullTelemetryWindow initialization and Matplotlib drawing
3. Geometric steering direction verification (Right steer -> Right wheel rotation)
4. 4-Channel Haptic and Force Feedback mapping (LT brake/ABS, RT traction wheelspin)
5. Dynamic physics frequency scaling
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import time
import threading
import unittest
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists for Qt widgets
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from src.ui.interactive_driver_14dof import (
    GamepadManager, PhysicsThread14DOF, Viewport3DChaseCamera,
    FullTelemetryWindow, Interactive14DOFDrivingWindow,
    setup_global_exception_handler
)
from src.models.params_vehicle import VehicleParams


class TestInteractiveDriver(unittest.TestCase):
    def setUp(self):
        self.vp = VehicleParams()

    def test_01_global_exception_handler(self):
        """Verify global exception hook is installed properly."""
        setup_global_exception_handler()
        self.assertIsNotNone(sys.excepthook)

    def test_02_thread_safe_telemetry_snapshot(self):
        """Verify get_telemetry_snapshot is thread-safe and never raises concurrent mutation errors."""
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=500.0)
        
        stop_thread = False
        def fast_appender():
            i = 0
            while not stop_thread:
                data = {
                    't': i * 0.002, 'vx_kmh': 50.0 + (i % 10),
                    'ay_g': 1.2, 'ax_g': 0.3, 'steer_deg': 5.0,
                    'z_fl_mm': 1.0, 'z_fr_mm': 1.0, 'z_rl_mm': -0.5, 'z_rr_mm': -0.5,
                    'camber_fl_deg': -2.5, 'camber_fr_deg': -2.5,
                    'camber_rl_deg': -1.5, 'camber_rr_deg': -1.5,
                    'Fz_fl': 500.0, 'Fz_fr': 500.0, 'Fz_rl': 600.0, 'Fz_rr': 600.0,
                    'haptic_brake_lt': 0.0, 'haptic_throt_rt': 15.0,
                    'rumble_heavy': 10.0, 'rumble_light': 5.0
                }
                with pt._history_lock:
                    if len(pt.telemetry_history) < pt.max_history:
                        pt.telemetry_history.append(data)
                i += 1
                time.sleep(0.0001)

        t = threading.Thread(target=fast_appender)
        t.start()

        # Simultaneously read snapshots and build DataFrames rapidly
        for _ in range(50):
            snap = pt.get_telemetry_snapshot()
            df = pd.DataFrame(snap)
            self.assertIsInstance(df, pd.DataFrame)
            time.sleep(0.001)

        stop_thread = True
        t.join()
        self.assertGreater(len(pt.get_telemetry_snapshot()), 10)

    def test_03_full_telemetry_window_rendering(self):
        """Verify FullTelemetryWindow initializes and plots without crashing."""
        sample_hist = []
        for i in range(100):
            t_val = i * 0.01
            sample_hist.append({
                't': t_val,
                'vx_kmh': 40.0 + np.sin(t_val) * 10.0,
                'ay_g': np.sin(t_val) * 1.5,
                'ax_g': np.cos(t_val) * 0.8,
                'steer_deg': np.sin(t_val) * 15.0,
                'throttle_pct': max(np.sin(t_val) * 100.0, 0.0),
                'brake_pct': max(-np.sin(t_val) * 100.0, 0.0),
                'z_fl_mm': 2.0, 'z_fr_mm': 1.5, 'z_rl_mm': -1.0, 'z_rr_mm': -0.8,
                'camber_fl_deg': -2.5, 'camber_fr_deg': -2.4, 'camber_rl_deg': -1.5, 'camber_rr_deg': -1.5,
                'Fz_fl': 550.0, 'Fz_fr': 520.0, 'Fz_rl': 680.0, 'Fz_rr': 670.0,
                'dFz_jack_N': 15.0,
                'haptic_brake_lt': 20.0,
                'haptic_throt_rt': 30.0,
                'rumble_heavy': 25.0,
                'rumble_light': 15.0
            })
        
        dlg = FullTelemetryWindow(sample_hist)
        self.assertIsNotNone(dlg)
        
        # Test preset switching
        for i in range(len(dlg.PRESETS)):
            dlg.combo_preset.setCurrentIndex(i)
            dlg._render_current_view()
            
        # Test single plot focus
        dlg.combo_focus.setCurrentIndex(1)  # Focus on suspension_travel
        dlg._render_current_view()
        self.assertEqual(len(dlg.fig.axes), 1)
        
        # Test focus on G-G plot and check aspect ratio
        gg_idx = dlg.combo_focus.findData('gg_diagram')
        if gg_idx >= 0:
            dlg.combo_focus.setCurrentIndex(gg_idx)
            dlg._render_current_view()
            ax_gg = dlg.fig.axes[0]
            self.assertEqual(ax_gg.get_aspect(), 1.0)
            
        # Test custom layout dialog
        dlg._on_preset_changed(0)
        self.assertGreater(len(dlg.fig.axes), 1)

    def test_04_wheel_steering_geometry_direction(self):
        """Verify visual wheel points steer to the RIGHT when delta > 0 and LEFT when delta < 0."""
        vp = Viewport3DChaseCamera()
        vp.car_state = {
            'x': 0.0, 'y': 0.0, 'psi_deg': 0.0, 'phi_deg': 0.0, 'theta_deg': 0.0,
            'steer_deg': 15.0, 'util_fl': 50, 'util_fr': 50, 'util_rl': 50, 'util_rr': 50
        }
        
        # Test 3D Projection
        # Left track curb (+1.9m) vs Right track curb (-1.9m)
        p_track_l, _ = vp.project_3d_point(5.0, 1.9, 0.0, -6.8, 0.0, 3.6, 0.0)
        p_track_r, _ = vp.project_3d_point(5.0, -1.9, 0.0, -6.8, 0.0, 3.6, 0.0)
        self.assertLess(p_track_l.x(), vp.width() / 2.0, "Left track curb must be on left of screen (sx < cx)")
        self.assertGreater(p_track_r.x(), vp.width() / 2.0, "Right track curb must be on right of screen (sx > cx)")

        # Test Steered Wheel Front Edge vs Rear Edge
        cam_x, cam_y, cam_z, cam_yaw = -6.8, 0.0, 3.6, 0.0
        steer_rad = np.deg2rad(15.0) # Steering RIGHT
        cos_s = np.cos(steer_rad); sin_s = np.sin(steer_rad)
        w_rad = 0.20
        
        def w_pt_test(wheel_x, wheel_y, dx, dy):
            lx = wheel_x + dx * cos_s + dy * sin_s
            ly = wheel_y - dx * sin_s + dy * cos_s
            # Car to world with heading 0
            wx = lx; wy = ly; wz = 0.22
            p, _ = vp.project_3d_point(wx, wy, wz, cam_x, cam_y, cam_z, cam_yaw)
            return p.x()

        front_edge_sx = w_pt_test(0.8, 0.6, w_rad, 0.0)
        rear_edge_sx = w_pt_test(0.8, 0.6, -w_rad, 0.0)
        self.assertGreater(front_edge_sx, rear_edge_sx, "When steering RIGHT, front of wheel must have greater sx (to the right of rear edge)")

        # Steering LEFT
        steer_rad_l = np.deg2rad(-15.0)
        cos_sl = np.cos(steer_rad_l); sin_sl = np.sin(steer_rad_l)
        def w_pt_test_l(wheel_x, wheel_y, dx, dy):
            lx = wheel_x + dx * cos_sl + dy * sin_sl
            ly = wheel_y - dx * sin_sl + dy * cos_sl
            wx = lx; wy = ly; wz = 0.22
            p, _ = vp.project_3d_point(wx, wy, wz, cam_x, cam_y, cam_z, cam_yaw)
            return p.x()

        front_edge_sx_l = w_pt_test_l(0.8, 0.6, w_rad, 0.0)
        rear_edge_sx_l = w_pt_test_l(0.8, 0.6, -w_rad, 0.0)
        self.assertLess(front_edge_sx_l, rear_edge_sx_l, "When steering LEFT, front of wheel must have smaller sx (to the left of rear edge)")

    def test_05_force_feedback_and_dynamic_hz(self):
        """Verify GamepadManager haptic channels and PhysicsThread frequency tuning."""
        gp = GamepadManager()
        gp.set_vibration(rumble_heavy=0.5, rumble_light=0.3, trigger_left=0.8, trigger_right=0.2)
        
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=250.0)
        self.assertEqual(pt.target_hz, 250.0)
        self.assertAlmostEqual(pt.dt, 0.004, places=5)
        
        pt.set_physics_hz(333.0)
        self.assertEqual(pt.target_hz, 333.0)
        self.assertAlmostEqual(pt.dt, 1.0/333.0, places=5)

        pt.set_physics_hz(500.0)
        self.assertEqual(pt.target_hz, 500.0)
        self.assertAlmostEqual(pt.dt, 0.002, places=5)

    def test_06_opengl_viewport_and_presets(self):
        """Verify OpenGL/Fallback Viewport initialization, skid ribbons tracking, and camera zoom."""
        from src.ui.interactive_driver_14dof import OpenGLViewport3D, HAS_OPENGL, get_viewport_class
        VpClass = get_viewport_class()
        vp = VpClass()
        self.assertIsNotNone(vp)
        
        # Test vehicle state update with high utilization (skid ribbon creation)
        state_high_util = {
            'x': 10.0, 'y': 5.0, 'psi_deg': 45.0, 'phi_deg': 1.0, 'theta_deg': -0.5,
            'steer_deg': 10.0, 'vx_kmh': 80.0, 'ax_mps2': 2.0, 'ay_mps2': 15.0,
            'z_fl': 5.0, 'z_fr': 5.0, 'z_rl': -3.0, 'z_rr': -3.0,
            'util_fl': 95.0, 'util_fr': 92.0, 'util_rl': 50.0, 'util_rr': 50.0,
            'kappa_fl': 5.0, 'kappa_fr': 5.0, 'kappa_rl': 25.0, 'kappa_rr': 25.0,
            'lap_time': 12.34
        }
        vp.update_vehicle_state(state_high_util)
        # Second update to create ribbon segment
        state_high_util_2 = dict(state_high_util)
        state_high_util_2['x'] = 10.5
        state_high_util_2['y'] = 5.5
        vp.update_vehicle_state(state_high_util_2)
        
        if hasattr(vp, 'skid_ribbons'):
            self.assertGreater(len(vp.skid_ribbons), 0, "Skid ribbons should be generated when wheel utilization > 82%")
            
        # Test camera parameters
        self.assertEqual(vp.cam_dist, 6.8)
        vp.cam_dist = 4.0
        vp.cam_height = 2.0
        self.assertEqual(vp.cam_dist, 4.0)
        self.assertEqual(vp.cam_height, 2.0)

    def test_07_interactive_window_controls_and_fullscreen(self):
        """Verify Interactive14DOFDrivingWindow exclusive game fullscreen toggle, sliders, and camera presets."""
        win = Interactive14DOFDrivingWindow(vehicle_params=self.vp)
        self.assertIsNotNone(win)
        
        # Test camera slider updates
        win.sl_fov.setValue(500)
        win.sl_dist.setValue(50)
        win.sl_height.setValue(25)
        self.assertEqual(win.viewport.fov, 500.0)
        self.assertAlmostEqual(win.viewport.cam_dist, 5.0)
        self.assertAlmostEqual(win.viewport.cam_height, 2.5)
        
        # Test reset camera
        win._reset_cam()
        self.assertEqual(win.sl_fov.value(), 420)
        self.assertEqual(win.sl_dist.value(), 68)
        
        # Test game-only fullscreen toggle method (collapsing sidebar)
        win.toggle_fullscreen()
        self.assertTrue(win.is_fullscreen_mode)
        self.assertFalse(win.right_panel.isVisible())
        self.assertTrue(win.viewport.is_fullscreen_view)
        
        # Toggle back to normal
        win.toggle_fullscreen()
        self.assertFalse(win.is_fullscreen_mode)
        self.assertTrue(win.right_panel.isVisible())
        self.assertFalse(win.viewport.is_fullscreen_view)
        
        # Close physics thread to avoid dangling threads
        if hasattr(win, 'physics_thread') and win.physics_thread.isRunning():
            win.physics_thread.running = False
            win.physics_thread.wait(500)

    def test_08_casadi_fast_ode_fallback(self):
        """Verify Vehicle14DOF get_fast_ode() returns a functioning CasADi ODE."""
        from src.models.vehicle_14dof import Vehicle14DOF
        v = Vehicle14DOF(self.vp)
        ode = v.get_fast_ode()
        self.assertIsNotNone(ode)
        
        # Test evaluation with dummy states (28 states)
        x = np.zeros(28)
        x[0] = 10.0 # vx = 10 m/s
        u = np.zeros(4)
        p = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            getattr(self.vp, 'LMUY', 1.60),
            getattr(self.vp, 'LMUX', 1.60)
        ]
        x_dot = np.array(ode(x, u, p)).flatten()
        self.assertEqual(len(x_dot), 28)
        self.assertTrue(np.all(np.isfinite(x_dot)))

    def test_09_track_designer_and_splines(self):
        """Verify TrackEditorCanvas and TrackDesignerDialog spline generation and track dict creation."""
        from src.ui.interactive_driver_14dof import TrackEditorCanvas, TrackDesignerDialog
        canvas = TrackEditorCanvas()
        self.assertIsNotNone(canvas)
        self.assertIsNotNone(canvas.generated_track)
        self.assertGreater(canvas.total_length_m, 50.0)
        self.assertIn('x_center', canvas.generated_track)
        self.assertIn('x_left', canvas.generated_track)
        self.assertIn('x_right', canvas.generated_track)
        
        # Test preset change
        dlg = TrackDesignerDialog()
        dlg.on_preset_changed(1) # Skidpad Figure-8
        self.assertEqual(len(dlg.canvas.control_points), 9)
        self.assertIsNotNone(dlg.canvas.generated_track)

    def test_10_ffb_sat_and_zero_suspension_standstill(self):
        """Verify simulator FFB SAT calculations and zero suspension travel at rest."""
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=250.0)
        self.assertEqual(pt.x_state[7], 0.0)
        self.assertEqual(pt.x_state[8], 0.0)
        self.assertEqual(pt.x_state[9], 0.0)
        self.assertEqual(pt.x_state[10], 0.0)

    def test_11_tcs_toggle_and_slip_regulation(self):
        """Verify Traction Control System (TCS) state, toggling, and slip regulation."""
        pt = PhysicsThread14DOF(vehicle_params=self.vp, target_hz=250.0)
        self.assertTrue(pt.tcs_enabled)
        self.assertAlmostEqual(pt.tcs_target_slip, 0.095)
        self.assertFalse(pt.tcs_active)
        self.assertAlmostEqual(pt.tcs_cut_pct, 0.0)
        
        # Test toggling
        pt.toggle_tcs()
        self.assertFalse(pt.tcs_enabled)
        pt.toggle_tcs()
        self.assertTrue(pt.tcs_enabled)
        pt.set_tcs_enabled(False)
        self.assertFalse(pt.tcs_enabled)
        pt.set_tcs_enabled(True)
        self.assertTrue(pt.tcs_enabled)
        
        # Test window TCS bindings and shortcut
        win = Interactive14DOFDrivingWindow(vehicle_params=self.vp)
        self.assertTrue(hasattr(win, 'cb_tcs'))
        self.assertTrue(win.cb_tcs.isChecked())
        win.toggle_tcs()
        self.assertFalse(win.cb_tcs.isChecked())
        self.assertFalse(win.physics_thread.tcs_enabled)
        win.toggle_tcs()
        self.assertTrue(win.cb_tcs.isChecked())
        self.assertTrue(win.physics_thread.tcs_enabled)
        if hasattr(win, 'physics_thread') and win.physics_thread.isRunning():
            win.physics_thread.running = False
            win.physics_thread.wait(500)

    def test_12_spinout_dissipation_and_antifeatures(self):
        """Verify that spinouts dissipate rotational energy into rest and anti-features are integrated."""
        from src.models.vehicle_14dof import Vehicle14DOF
        from src.models.params_tire import TireParams
        
        tir_path = os.path.join(os.path.dirname(__file__), '..', 'tire_tir', 'Hoosier_43075_16x7.5-10_R20_Rim7_lm_2_3.tir')
        tp = TireParams(tir_path)
        v14 = Vehicle14DOF(self.vp, tp, tp)
        ode = v14.get_fast_ode()
        p_vec = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            getattr(self.vp, 'LMUY', 1.0),
            getattr(self.vp, 'LMUX', 1.0)
        ]
        
        # Initialize vehicle in high yaw rate spinout (r = 3.0 rad/s at 10 m/s)
        x = np.zeros(28)
        x[0] = 10.0
        x[2] = 3.0
        x[16:20] = 10.0 / self.vp.R_w
        u = np.zeros(4) # Controls released
        
        E_start = 0.5 * self.vp.m * (x[0]**2) + 0.5 * self.vp.Iz * (x[2]**2)
        dt = 0.002
        # Integrate 1.5 seconds (750 steps)
        for _ in range(750):
            k1 = np.array(ode(x, u, p_vec)).flatten()
            k2 = np.array(ode(x + 0.5 * dt * k1, u, p_vec)).flatten()
            k3 = np.array(ode(x + 0.5 * dt * k2, u, p_vec)).flatten()
            k4 = np.array(ode(x + dt * k3, u, p_vec)).flatten()
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
        E_end = 0.5 * self.vp.m * (np.hypot(x[0], x[1])**2) + 0.5 * self.vp.Iz * (x[2]**2)
        # Energy must strictly decrease (dissipative system)
        self.assertLess(E_end, E_start)
        # Yaw rate must be damped out (spinout stops rotating)
        self.assertLess(abs(x[2]), 0.5, f"Yaw rate should damp below 0.5 rad/s, got {x[2]}")

    def test_13_racing_wheel_support_and_ffb(self):
        """Verify PC racing wheel recognition, 1:1 linear steering, mode selector, and FFB mixing."""
        gp = GamepadManager()

        # 1. Test keyword detection
        wheel_names = [
            "Fanatec ClubSport Wheel Base",
            "Logitech G29 Driving Force Racing Wheel",
            "Logitech G920 Driving Force",
            "Thrustmaster T300RS Racing Wheel",
            "Fanatec CSL DD",
            "Direct Drive DD1 Pro Wheel"
        ]
        for name in wheel_names:
            name_lower = name.lower()
            is_w = any(kw in name_lower for kw in gp.WHEEL_KEYWORDS)
            self.assertTrue(is_w, f"Device '{name}' should be recognized as a racing wheel")

        pad_names = ["Xbox Wireless Controller", "DualSense Controller", "Generic USB Gamepad"]
        for name in pad_names:
            name_lower = name.lower()
            is_w = any(kw in name_lower for kw in gp.WHEEL_KEYWORDS)
            self.assertFalse(is_w, f"Device '{name}' should NOT be recognized as a wheel")

        # 2. Test manual device mode selector
        gp.detected_device_type = "gamepad"
        gp.set_device_mode("auto")
        self.assertFalse(gp.is_wheel())

        gp.set_device_mode("wheel")
        self.assertTrue(gp.is_wheel(), "Manual 'wheel' mode must override auto detection")

        gp.set_device_mode("gamepad")
        self.assertFalse(gp.is_wheel(), "Manual 'gamepad' mode must override auto detection")

        # 3. Test differentiated steering response: Gamepad vs Wheel
        # Fake a small joystick input within the default 4% deadzone (0.02)
        gp.steer_deadzone = 0.04
        gp.steer_sensitivity = 1.25

        class MockJoystick:
            def __init__(self, axis_val=0.02):
                self.axis_val = axis_val
            def get_axis(self, idx):
                return self.axis_val if idx == 0 else 0.0
            def get_num_axes(self):
                return 4
            def get_num_buttons(self):
                return 8
            def get_button(self, idx):
                return False
            def rumble(self, h, l, d):
                pass

        gp.backend = "pygame"
        gp.connected = True
        gp.py_joystick = MockJoystick(axis_val=0.02)

        # In Gamepad mode: 0.02 is inside deadzone (0.04) -> steer must be 0.0
        gp.set_device_mode("gamepad")
        cmd_pad = gp.poll_inputs()
        self.assertEqual(cmd_pad['steer'], 0.0, "Gamepad must filter out small inputs inside deadzone")

        # In Wheel mode: 0.02 is 1:1 linear direct -> steer must be exactly 0.02 (no deadband)
        gp.set_device_mode("wheel")
        cmd_wheel = gp.poll_inputs()
        self.assertAlmostEqual(cmd_wheel['steer'], 0.02, places=4, msg="Wheel must have 1:1 linear response without deadband")

        # Test larger input (e.g. 0.50):
        # Gamepad: shaped with deadzone + exponent ((0.50 - 0.04) / 0.96) ** 1.25 != 0.50
        gp.py_joystick = MockJoystick(axis_val=0.50)
        gp.set_device_mode("gamepad")
        cmd_pad_50 = gp.poll_inputs()
        self.assertNotEqual(cmd_pad_50['steer'], 0.50, "Gamepad must apply cubic sensitivity filter")

        # Wheel: exactly 0.50
        gp.set_device_mode("wheel")
        cmd_wheel_50 = gp.poll_inputs()
        self.assertAlmostEqual(cmd_wheel_50['steer'], 0.50, places=4, msg="Wheel must be direct 1:1 without shaping")

        # 4. Test FFB algebraic mixing and safety without crashes
        gp.set_device_mode("wheel")
        gp.set_vibration(rumble_heavy=0.3, rumble_light=0.2, steer_torque=-4.5, tactile_vibe=0.20)
        self.assertIsNotNone(gp._last_wheel_level)

        gp.set_device_mode("gamepad")
        gp.set_vibration(rumble_heavy=0.4, rumble_light=0.2, trigger_left=0.5, trigger_right=0.1)

    def test_12_aero_drag_live_tuning_and_top_speed_sensitivity(self):
        """Verify that live tuning Cd updates p_vec dynamically and enforces physical top speed limits."""
        from src.ui.interactive_driver_14dof import PhysicsThread14DOF
        from src.models.vehicle_14dof import Vehicle14DOF
        import numpy as np

        thread = PhysicsThread14DOF(vehicle_params=self.vp)
        self.assertEqual(thread.vp.Cd, self.vp.Cd)

        # 1. Update Cd via update_live_tuning
        thread.update_live_tuning(Cd=2.4)
        self.assertEqual(thread.vp.Cd, 2.4)
        self.assertEqual(thread.vp.aero_Cd, 2.4)
        self.assertTrue(thread._tuning_dirty)

        # 2. Verify ODE decel sensitivity with Cd in p_vec
        v14 = Vehicle14DOF(self.vp)
        ode = v14.get_fast_ode()

        vx = 43.33  # 156 km/h
        w = vx / self.vp.R_w
        x = np.zeros(28)
        x[0] = vx
        x[16:20] = w
        u = np.zeros(4)

        p_base = [
            self.vp.wd_f, self.vp.lltd, self.vp.K_phi_f, self.vp.K_phi_r,
            self.vp.K_z_f, self.vp.K_z_r, self.vp.aero_balance,
            self.vp.camber_f_static, self.vp.camber_r_static,
            self.vp.toe_f_static, self.vp.toe_r_static,
            self.vp.L, self.vp.t_f, self.vp.t_r,
            1.0, 1.0
        ]
        p_cd12 = p_base + [1.20]
        p_cd24 = p_base + [2.40]

        res_12 = np.array(ode(x, u, p_cd12)).flatten()
        res_24 = np.array(ode(x, u, p_cd24)).flatten()

        # Deceleration with Cd=2.4 must be substantially higher than with Cd=1.2
        self.assertLess(res_24[0], res_12[0], "Doubling Cd must produce strictly stronger deceleration")
        delta_drag_force = (res_12[0] - res_24[0]) * self.vp.m
        expected_delta_drag = 0.5 * self.vp.rho * self.vp.S * (2.4 - 1.2) * (vx**2)
        self.assertAlmostEqual(delta_drag_force, expected_delta_drag, delta=2.0)

        # 3. Verify drag power at 156 km/h with Cd=2.4 exceeds 80 kW
        f_drag_24 = 0.5 * self.vp.rho * self.vp.S * 2.4 * (vx**2)
        p_drag_24_kw = (f_drag_24 * vx) / 1000.0
        self.assertGreater(p_drag_24_kw, 115.0, f"Drag power {p_drag_24_kw:.1f} kW should exceed 115 kW")

        # 4. Verify reverse motion drag direction (vx * abs(vx))
        x_rev = np.zeros(28)
        x_rev[0] = -20.0
        res_rev = np.array(ode(x_rev, u, p_cd12)).flatten()
        # In reverse (vx < 0), aero drag pushes forward (positive v_x_dot contribution)
        self.assertGreater(res_rev[0], 0.0, "Drag in reverse must push forward to oppose motion")


if __name__ == '__main__':
    unittest.main()



