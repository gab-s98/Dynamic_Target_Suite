"""
Interactive 14-DOF Real-Time Driving Simulator with 3D Bird's-Eye View & Full Kinematics Telemetry
=================================================================================================
Features:
- Physical Caster & KPI Wheel Jacking (Standstill diagonal corner weight redistribution):
  * Caster angle (nu = 5.5 deg), KPI (sigma = 6.5 deg), Mechanical Trail & Scrub Radius.
  * Steering from standstill dynamically redistributes Fz_FL, Fz_FR, Fz_RL, Fz_RR cross-diagonally!
- Exact 3D & 2D Wheel & Chassis Visual Rendering:
  * Front wheels rotate visually in the exact direction of steering (Right -> Right, Left -> Left).
  * 3D Depth-sorted track rendering with grass green outfield and horizon sky gradient.
  * 2D Top-down track map with steered wheels and dynamic grip coloring.
- Complete 14-DOF Telemetry Suite & Crash-Proof Analysis:
  * Thread-safe snapshot buffer preventing concurrent mutation crashes.
  * Suspension wheel travel (z_fl..z_rr) [mm], dynamic camber & toe, roll centers (h_rc_f, h_rc_r)
  * 4-Wheel tire forces (Fz, Fy, Fx), slip angles (alpha_i), and slip ratios (kappa_i)
  * G-G diagram with velocity coloring and friction circle.
- Advanced Force Feedback & Haptic Trigger Feedback (Impulse Triggers & 4-Channel Haptics):
  * Left Trigger (LT): Brake lockup threshold & ABS pulsation haptics.
  * Right Trigger (RT): Traction loss, throttle wheelspin & longitudinal slip buzz.
  * Low-Frequency Heavy Rumble: Chassis inertia, oversteer snap, tail drift & kerb impacts.
  * High-Frequency Light Rumble: Front tire understeer scrub & road surface grip limit.
- Universal controller auto-detection (Windows Gaming Input + Native XInput + Pygame DirectInput + Keyboard).
- Configurable RK4 Physics loop (250 Hz default / 333 Hz / 500 Hz) with real-time performance budget monitor.
- Global text crash logger saving traceback to crash_log.txt and displaying user-friendly dialogs.
"""

import sys
import os
import time
import math
import ctypes
from ctypes import wintypes
import threading
import traceback
import numpy as np
import pandas as pd
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QSplitter, QMessageBox, QFileDialog,
    QSlider, QDoubleSpinBox, QSpinBox, QAbstractSpinBox, QLineEdit,
    QTextEdit, QPlainTextEdit, QTabWidget, QScrollArea, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QToolButton, QMenu, QFrame,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPolygonF,
    QPainterPath, QLinearGradient
)

try:
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import QSurfaceFormat
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True

    # Configure high-performance OpenGL surface format (zero VSync latency / tear-free double-buffering)
    _gl_fmt = QSurfaceFormat()
    _gl_fmt.setDepthBufferSize(24)
    _gl_fmt.setStencilBufferSize(8)
    _gl_fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(_gl_fmt)
except ImportError:
    HAS_OPENGL = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.models.tire_model_mf61 import (
    mf61_lateral_force, mf61_longitudinal_force,
    mf61_combined_lateral_force, mf61_combined_longitudinal_force,
    mf61_aligning_torque
)
from src.models.vehicle_14dof import Vehicle14DOF
from src.mdo.mdo_direct_collocation import build_track_geometry
from src.core.road_profile import RoadProfile, RoadProfileConfig
from src.models.tire_degradation import TireState, TireDegradationModel, TireDegradationParams


# =============================================================================
# 0. GLOBAL EXCEPTION HOOK & CRASH LOGGER
# =============================================================================
def setup_global_exception_handler():
    """Installs a global exception hook to catch unhandled crashes, write to crash_log.txt, and show a Qt dialog."""
    def exception_hook(exc_type, exc_value, exc_tb):
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(f"\n[CRASH DETECTED]\n{err_msg}\n")
        try:
            with open("crash_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- CRASH LOG [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
                f.write(err_msg)
                f.write("\n" + "="*60 + "\n")
        except Exception:
            pass

        app = QApplication.instance()
        if app is not None:
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Errore Imprevisto (Crash Log)")
                msg_box.setText(f"<b>Si è verificato un errore imprevisto:</b><br><br><code>{exc_value}</code><br><br>I dettagli sono stati registrati in <code>crash_log.txt</code>.")
                msg_box.setDetailedText(err_msg)
                msg_box.exec_()
            except Exception:
                pass
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_hook

setup_global_exception_handler()


# =============================================================================
# 1. NATIVE WINDOWS XINPUT & CONTROLLER DEFINITIONS
# =============================================================================


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', wintypes.WORD),
        ('bLeftTrigger', wintypes.BYTE),
        ('bRightTrigger', wintypes.BYTE),
        ('sThumbLX', wintypes.SHORT),
        ('sThumbLY', wintypes.SHORT),
        ('sThumbRX', wintypes.SHORT),
        ('sThumbRY', wintypes.SHORT),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ('dwPacketNumber', wintypes.DWORD),
        ('Gamepad', XINPUT_GAMEPAD)
    ]


class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [
        ('wLeftMotorSpeed', wintypes.WORD),
        ('wRightMotorSpeed', wintypes.WORD)
    ]


class XINPUT_VIBRATION_EX(ctypes.Structure):
    _fields_ = [
        ('wLeftMotorSpeed', wintypes.WORD),
        ('wRightMotorSpeed', wintypes.WORD),
        ('wLeftTriggerSpeed', wintypes.WORD),
        ('wRightTriggerSpeed', wintypes.WORD)
    ]


# DirectInput / SDL2 Haptic Force Feedback Structures (for PC Racing Wheels)

class SDL_HapticDirection(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint8),
        ('dir', ctypes.c_int32 * 3),
    ]


class SDL_HapticConstant(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint16),
        ('direction', SDL_HapticDirection),
        ('length', ctypes.c_uint32),
        ('delay', ctypes.c_uint16),
        ('button', ctypes.c_uint16),
        ('interval', ctypes.c_uint16),
        ('level', ctypes.c_int16),
        ('attack_length', ctypes.c_uint16),
        ('attack_level', ctypes.c_uint16),
        ('fade_length', ctypes.c_uint16),
        ('fade_level', ctypes.c_uint16),
    ]


class SDL_HapticEffect(ctypes.Union):
    _fields_ = [
        ('type', ctypes.c_uint16),
        ('constant', SDL_HapticConstant),
        ('pad', ctypes.c_uint8 * 72),
    ]


# =============================================================================

FFB_PRESETS = {
    "Bilanciato (Consigliato)": {
        "ffb_gain": 80.0,
        "road_feel": 20.0,
        "grip_loss": 75.0,
        "trigger_haptics": True,
        "deadzone": 4.0,
        "sensitivity": 1.25,
    },
    "Morbido / Confortevole": {
        "ffb_gain": 50.0,
        "road_feel": 10.0,
        "grip_loss": 50.0,
        "trigger_haptics": True,
        "deadzone": 5.0,
        "sensitivity": 1.10,
    },
    "Reattivo / Sportivo": {
        "ffb_gain": 100.0,
        "road_feel": 35.0,
        "grip_loss": 100.0,
        "trigger_haptics": True,
        "deadzone": 3.0,
        "sensitivity": 1.35,
    },
    "Puro Grip (Solo Limite Gomme)": {
        "ffb_gain": 70.0,
        "road_feel": 0.0,
        "grip_loss": 85.0,
        "trigger_haptics": False,
        "deadzone": 4.0,
        "sensitivity": 1.20,
    },
    "Disattivato": {
        "ffb_gain": 0.0,
        "road_feel": 0.0,
        "grip_loss": 0.0,
        "trigger_haptics": False,
        "deadzone": 4.0,
        "sensitivity": 1.0,
    }
}


# =============================================================================
# 2. UNIVERSAL GAMEPAD & WHEEL INPUT MANAGER WITH TRIGGER HAPTICS
# =============================================================================

class GamepadManager:
    """Universal Gamepad & PC Racing Wheel Input Manager with DirectInput / SDL2 Constant Force FFB."""
    WHEEL_KEYWORDS = ["fanatec", "logitech", "wheel", "g29", "g920", "thrustmaster", "csl", "dd"]

    def __init__(self):
        self.backend = "none"
        self.gamepad_name = "Tastiera (WASD / Frecce)"
        self.connected = False
        
        # Device mode selector: "auto" (default), "gamepad", "wheel"
        self.device_mode = "auto"
        self.detected_device_type = "keyboard"  # "gamepad", "wheel", "keyboard"
        
        self.steer_deadzone = 0.04
        self.steer_sensitivity = 1.25
        self.invert_steer = False
        self.trigger_deadzone = 0.02
        self.max_steer_deg = 25.0
        
        self.enable_ffb = True
        self.enable_trigger_haptics = True
        self.ffb_gain = 0.80
        self.road_feel_gain = 0.20
        self.grip_loss_gain = 0.75
        self.trigger_gain = 1.00
        self.trigger_blend_gain = 0.75

        # Anti-Jerk & Responsive Smoothing State
        self._smooth_heavy = 0.0
        self._smooth_light = 0.0
        self._last_vibration_time = 0.0
        self._last_left_val = -1
        self._last_right_val = -1
        self._last_lt_trig = -1
        self._last_rt_trig = -1
        self._last_wheel_level = -999999
        
        # Progressive Keyboard Input Smoothing State
        self._kb_steer = 0.0
        self._kb_throttle = 0.0
        self._kb_brake = 0.0
        
        self.xinput_dll = None
        self.xinput_user_index = 0
        self._init_xinput()
        
        self.has_pygame = False
        self.py_joystick = None
        self.sdl_dll = None
        self._sdl_gc_rumble_triggers = None
        self._sdl_gc_handle = None
        
        # SDL2 / DirectInput Constant Force Haptic Handle
        self.sdl_haptic = None
        self.sdl_haptic_effect_id = -1
        self.supports_constant_ffb = False
        
        self._init_pygame()
        self.scan_controllers()
        self.last_scan_time = time.time()

    def is_wheel(self) -> bool:
        """Returns True if the active input device operates in PC Racing Wheel mode."""
        if self.device_mode == "wheel":
            return True
        if self.device_mode == "gamepad":
            return False
        # "auto": rely on keyword detection
        return self.detected_device_type == "wheel"

    def set_device_mode(self, mode: str):
        """Sets peripheral mode: 'auto', 'gamepad', or 'wheel'."""
        if mode in ("auto", "gamepad", "wheel"):
            self.device_mode = mode

    def _close_haptic(self):
        """Safely stops and closes SDL2 haptic device."""
        if self.sdl_haptic is not None:
            try:
                if self.sdl_haptic_effect_id >= 0 and hasattr(self, '_sdl_haptic_stop_effect'):
                    self._sdl_haptic_stop_effect(self.sdl_haptic, self.sdl_haptic_effect_id)
                if self.sdl_haptic_effect_id >= 0 and hasattr(self, '_sdl_haptic_destroy_effect'):
                    self._sdl_haptic_destroy_effect(self.sdl_haptic, self.sdl_haptic_effect_id)
                if hasattr(self, '_sdl_haptic_close'):
                    self._sdl_haptic_close(self.sdl_haptic)
            except Exception:
                pass
            self.sdl_haptic = None
            self.sdl_haptic_effect_id = -1
            self.supports_constant_ffb = False

    def set_vibration(self, rumble_heavy=0.0, rumble_light=0.0, trigger_left=0.0, trigger_right=0.0,
                      steer_torque=0.0, tactile_vibe=0.0):
        """
        Applies physical Force Feedback to controller or PC racing wheel:
        - Gamepad:
            * rumble_heavy (0.0 to 1.0): Low-frequency heavy motor (Oversteer snap, tail drift, curb impacts)
            * rumble_light (0.0 to 1.0): High-frequency light motor (Tire scrub, road texture, grip limit)
            * trigger_left (0.0 to 1.0): Left trigger impulse motor (Brake lockup / ABS pulsation)
            * trigger_right (0.0 to 1.0): Right trigger impulse motor (Traction loss / Wheelspin)
        - Volante PC (DirectInput / SDL2 Haptic Constant Force):
            * steer_torque (signed float in Nm): Restoring torque (SAT Mz + Fy * tm) from front axle
            * tactile_vibe (signed float [-1.0, 1.0]): Superimposed kerb rumble, road texture, and slip oscillations
        """
        now_wall = time.time()
        if hasattr(self, '_test_vibration_until') and now_wall < self._test_vibration_until:
            rumble_heavy = max(rumble_heavy, 0.85)
            rumble_light = max(rumble_light, 0.85)
            trigger_left = max(trigger_left, 0.85)
            trigger_right = max(trigger_right, 0.85)
            if steer_torque == 0.0:
                steer_torque = 3.5

        if not self.enable_ffb:
            rumble_heavy = 0.0
            rumble_light = 0.0
            trigger_left = 0.0
            trigger_right = 0.0
            steer_torque = 0.0
            tactile_vibe = 0.0

        now = time.perf_counter()

        # =====================================================================
        # ROUTE A: PC RACING WHEEL (DIRECTINPUT / SDL2 CONSTANT FORCE)
        # =====================================================================
        if self.is_wheel():
            # Reference maximum handwheel torque rating for Formula Student [Nm]
            T_ref = 8.0
            u_sat = float(np.clip(steer_torque / T_ref, -1.0, 1.0))
            u_tactile = float(np.clip(tactile_vibe, -0.60, 0.60))
            
            # Algebraic sum of constant restoring force + procedural tactile vibrations, scaled by FFB gain
            u_wheel = float(np.clip((u_sat + u_tactile) * self.ffb_gain, -1.0, 1.0))
            target_level = int(u_wheel * 32767.0)

            # Hardware USB rate-limiting to ~100 Hz (max once per 10ms)
            if (now - self._last_vibration_time >= 0.010) or abs(target_level - self._last_wheel_level) > 250:
                self._last_vibration_time = now
                self._last_wheel_level = target_level

                if self.sdl_haptic and self.sdl_haptic_effect_id >= 0 and self.supports_constant_ffb:
                    eff = SDL_HapticEffect()
                    eff.type = 1  # SDL_HAPTIC_CONSTANT
                    eff.constant.type = 1
                    eff.constant.direction.type = 1  # SDL_HAPTIC_CARTESIAN
                    eff.constant.direction.dir[0] = 1  # X steering axis
                    eff.constant.direction.dir[1] = 0
                    eff.constant.direction.dir[2] = 0
                    eff.constant.length = 0xFFFFFFFF  # Infinite
                    eff.constant.level = target_level
                    try:
                        self._sdl_haptic_update_effect(self.sdl_haptic, self.sdl_haptic_effect_id, ctypes.byref(eff))
                    except Exception:
                        pass
                elif self.backend == "pygame" and self.py_joystick and hasattr(self.py_joystick, 'rumble'):
                    try:
                        mag = float(min(max(abs(u_wheel), 0.0), 1.0))
                        self.py_joystick.rumble(mag, mag, 40)
                    except Exception:
                        pass
            return

        # =====================================================================
        # ROUTE B: GAMEPAD (XINPUT SET STATE / SDL RUMBLE MOTORS)
        # =====================================================================
        gain = self.ffb_gain
        h_heavy = min(max(rumble_heavy * gain, 0.0), 1.0)
        h_light = min(max(rumble_light * gain, 0.0), 1.0)

        t_gain = getattr(self, 'trigger_gain', 1.00)
        t_left_val = min(max(trigger_left * t_gain, 0.0), 1.0) if self.enable_trigger_haptics else 0.0
        t_right_val = min(max(trigger_right * t_gain, 0.0), 1.0) if self.enable_trigger_haptics else 0.0

        if self.enable_trigger_haptics:
            blend_mult = getattr(self, 'trigger_blend_gain', 0.85)
            h_heavy = min(max(h_heavy + t_left_val * blend_mult, 0.0), 1.0)
            h_light = min(max(h_light + t_right_val * blend_mult, 0.0), 1.0)

        rate_up = 0.50
        rate_dn = 0.35
        if h_heavy > self._smooth_heavy:
            self._smooth_heavy += rate_up * (h_heavy - self._smooth_heavy)
        else:
            self._smooth_heavy += rate_dn * (h_heavy - self._smooth_heavy)

        if h_light > self._smooth_light:
            self._smooth_light += rate_up * (h_light - self._smooth_light)
        else:
            self._smooth_light += rate_dn * (h_light - self._smooth_light)

        left_int = int(self._smooth_heavy * 65535)
        right_int = int(self._smooth_light * 65535)
        t_left_int = int(t_left_val * 65535)
        t_right_int = int(t_right_val * 65535)

        # Strict rate-limiting at 50-60 Hz (Delta_t >= 16 ms)
        # Decoupled from 250-500 Hz physics loop to prevent USB HID buffer overflows
        dt_vibe = now - getattr(self, '_last_vibration_time', 0.0)
        if dt_vibe < 0.016:
            return

        # Minimum delta intensity threshold: Delta_intensity > 0.02 (1310 out of 65535)
        thresh_delta = int(0.02 * 65535)
        is_significant_change = (
            abs(left_int - getattr(self, '_last_left_val', 0)) > thresh_delta or
            abs(right_int - getattr(self, '_last_right_val', 0)) > thresh_delta or
            abs(t_left_int - getattr(self, '_last_lt_trig', 0)) > thresh_delta or
            abs(t_right_int - getattr(self, '_last_rt_trig', 0)) > thresh_delta
        )
        is_stop = (left_int == 0 and right_int == 0 and t_left_int == 0 and t_right_int == 0 and
                   (getattr(self, '_last_left_val', 0) != 0 or getattr(self, '_last_right_val', 0) != 0 or
                    getattr(self, '_last_lt_trig', 0) != 0 or getattr(self, '_last_rt_trig', 0) != 0))
        if not is_stop and not is_significant_change:
            return

        self._last_vibration_time = now
        self._last_left_val = left_int
        self._last_right_val = right_int
        self._last_lt_trig = t_left_int
        self._last_rt_trig = t_right_int

        if self.backend == "xinput" and self.xinput_dll:
            try:
                vib = XINPUT_VIBRATION(left_int, right_int)
                self.xinput_dll.XInputSetState(self.xinput_user_index, ctypes.byref(vib))
            except Exception:
                pass
        elif self.backend == "pygame" and self.py_joystick and hasattr(self.py_joystick, 'rumble'):
            try:
                self.py_joystick.rumble(float(self._smooth_heavy), float(self._smooth_light), 40)
            except Exception:
                pass

        if self._sdl_gc_rumble_triggers and self._sdl_gc_handle and self.enable_trigger_haptics:
            try:
                self._sdl_gc_rumble_triggers(self._sdl_gc_handle, t_left_int, t_right_int, 40)
            except Exception:
                pass

    def test_vibration(self):
        """Sends an unmistakable 500ms test vibration burst to controller motors or steering wheel."""
        self._test_vibration_until = time.time() + 0.50
        if self.is_wheel():
            if self.sdl_haptic and self.sdl_haptic_effect_id >= 0 and self.supports_constant_ffb:
                eff = SDL_HapticEffect()
                eff.type = 1
                eff.constant.type = 1
                eff.constant.direction.type = 1
                eff.constant.direction.dir[0] = 1
                eff.constant.length = 0xFFFFFFFF
                eff.constant.level = int(0.50 * 32767)  # 50% torque pulse
                try:
                    self._sdl_haptic_update_effect(self.sdl_haptic, self.sdl_haptic_effect_id, ctypes.byref(eff))
                except Exception:
                    pass
            elif self.backend == "pygame" and self.py_joystick and hasattr(self.py_joystick, 'rumble'):
                try:
                    self.py_joystick.rumble(0.85, 0.85, 500)
                except Exception:
                    pass
        else:
            if self.backend == "xinput" and self.xinput_dll:
                try:
                    vib = XINPUT_VIBRATION(55000, 55000)
                    self.xinput_dll.XInputSetState(self.xinput_user_index, ctypes.byref(vib))
                except Exception:
                    pass
            elif self.backend == "pygame" and self.py_joystick and hasattr(self.py_joystick, 'rumble'):
                try:
                    self.py_joystick.rumble(0.85, 0.85, 500)
                except Exception:
                    pass
            if self._sdl_gc_rumble_triggers and self._sdl_gc_handle:
                try:
                    self._sdl_gc_rumble_triggers(self._sdl_gc_handle, 55000, 55000, 500)
                except Exception:
                    pass
        self.set_vibration(rumble_heavy=0.85, rumble_light=0.85, trigger_left=0.85, trigger_right=0.85,
                           steer_torque=4.0, tactile_vibe=0.35)

    def test_trigger_vibration(self):
        """Compatibility alias for test_vibration."""
        self.test_vibration()

    def _init_xinput(self):
        for dll_name in ['xinput1_4.dll', 'xinput1_3.dll', 'xinput9_1_0.dll']:
            try:
                self.xinput_dll = ctypes.windll.LoadLibrary(dll_name)
                break
            except Exception:
                continue

    def _init_pygame(self):
        self.sdl_dll = None
        self._sdl_gc_rumble_triggers = None
        self._sdl_gc_handle = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            self.has_pygame = True
            try:
                sdl_path = os.path.join(os.path.dirname(pygame.__file__), 'SDL2.dll')
                if os.path.exists(sdl_path):
                    self.sdl_dll = ctypes.cdll.LoadLibrary(sdl_path)
                    if hasattr(self.sdl_dll, 'SDL_GameControllerRumbleTriggers'):
                        self._sdl_gc_rumble_triggers = self.sdl_dll.SDL_GameControllerRumbleTriggers
                        self._sdl_gc_rumble_triggers.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32]
                        self._sdl_gc_rumble_triggers.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_GameControllerOpen'):
                        self._sdl_gc_open = self.sdl_dll.SDL_GameControllerOpen
                        self._sdl_gc_open.argtypes = [ctypes.c_int]
                        self._sdl_gc_open.restype = ctypes.c_void_p

                    # DirectInput / SDL2 Haptic Force Feedback API setup
                    SDL_INIT_HAPTIC = 0x00001000
                    if hasattr(self.sdl_dll, 'SDL_InitSubSystem'):
                        self.sdl_dll.SDL_InitSubSystem(SDL_INIT_HAPTIC)
                    if hasattr(self.sdl_dll, 'SDL_NumHaptics'):
                        self._sdl_num_haptics = self.sdl_dll.SDL_NumHaptics
                        self._sdl_num_haptics.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticOpen'):
                        self._sdl_haptic_open = self.sdl_dll.SDL_HapticOpen
                        self._sdl_haptic_open.argtypes = [ctypes.c_int]
                        self._sdl_haptic_open.restype = ctypes.c_void_p
                    if hasattr(self.sdl_dll, 'SDL_JoystickOpen'):
                        self._sdl_joy_open = self.sdl_dll.SDL_JoystickOpen
                        self._sdl_joy_open.argtypes = [ctypes.c_int]
                        self._sdl_joy_open.restype = ctypes.c_void_p
                    if hasattr(self.sdl_dll, 'SDL_JoystickIsHaptic'):
                        self._sdl_joy_is_haptic = self.sdl_dll.SDL_JoystickIsHaptic
                        self._sdl_joy_is_haptic.argtypes = [ctypes.c_void_p]
                        self._sdl_joy_is_haptic.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticOpenFromJoystick'):
                        self._sdl_haptic_open_from_joy = self.sdl_dll.SDL_HapticOpenFromJoystick
                        self._sdl_haptic_open_from_joy.argtypes = [ctypes.c_void_p]
                        self._sdl_haptic_open_from_joy.restype = ctypes.c_void_p
                    if hasattr(self.sdl_dll, 'SDL_HapticClose'):
                        self._sdl_haptic_close = self.sdl_dll.SDL_HapticClose
                        self._sdl_haptic_close.argtypes = [ctypes.c_void_p]
                        self._sdl_haptic_close.restype = None
                    if hasattr(self.sdl_dll, 'SDL_HapticQuery'):
                        self._sdl_haptic_query = self.sdl_dll.SDL_HapticQuery
                        self._sdl_haptic_query.argtypes = [ctypes.c_void_p]
                        self._sdl_haptic_query.restype = ctypes.c_uint32
                    if hasattr(self.sdl_dll, 'SDL_HapticNewEffect'):
                        self._sdl_haptic_new_effect = self.sdl_dll.SDL_HapticNewEffect
                        self._sdl_haptic_new_effect.argtypes = [ctypes.c_void_p, ctypes.POINTER(SDL_HapticEffect)]
                        self._sdl_haptic_new_effect.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticUpdateEffect'):
                        self._sdl_haptic_update_effect = self.sdl_dll.SDL_HapticUpdateEffect
                        self._sdl_haptic_update_effect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(SDL_HapticEffect)]
                        self._sdl_haptic_update_effect.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticRunEffect'):
                        self._sdl_haptic_run_effect = self.sdl_dll.SDL_HapticRunEffect
                        self._sdl_haptic_run_effect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
                        self._sdl_haptic_run_effect.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticStopEffect'):
                        self._sdl_haptic_stop_effect = self.sdl_dll.SDL_HapticStopEffect
                        self._sdl_haptic_stop_effect.argtypes = [ctypes.c_void_p, ctypes.c_int]
                        self._sdl_haptic_stop_effect.restype = ctypes.c_int
                    if hasattr(self.sdl_dll, 'SDL_HapticDestroyEffect'):
                        self._sdl_haptic_destroy_effect = self.sdl_dll.SDL_HapticDestroyEffect
                        self._sdl_haptic_destroy_effect.argtypes = [ctypes.c_void_p, ctypes.c_int]
                        self._sdl_haptic_destroy_effect.restype = None
            except Exception:
                pass
        except Exception:
            self.has_pygame = False

    def scan_controllers(self):
        self._close_haptic()

        if self.xinput_dll:
            state = XINPUT_STATE()
            for idx in range(4):
                try:
                    res = self.xinput_dll.XInputGetState(idx, ctypes.byref(state))
                    if res == 0:
                        self.backend = "xinput"
                        self.xinput_user_index = idx
                        self.connected = True
                        self.detected_device_type = "gamepad"

                        # Inizializza l'handle SDL2 per i motorini dei grilletti fisici LT/RT
                        if self.sdl_dll and hasattr(self, '_sdl_gc_open'):
                            try:
                                self._sdl_gc_handle = self._sdl_gc_open(idx)
                            except Exception:
                                self._sdl_gc_handle = None

                        if self.is_wheel():
                            self.gamepad_name = f"Xbox Wheel Emulation (Porta {idx+1})"
                        else:
                            self.gamepad_name = f"Xbox Controller (Porta {idx+1} + Force Feedback)"
                        return True
                except Exception:
                    pass

        if self.has_pygame:
            try:
                import pygame
                pygame.joystick.quit()
                pygame.joystick.init()
                cnt = pygame.joystick.get_count()
                if cnt > 0:
                    self.py_joystick = pygame.joystick.Joystick(0)
                    self.py_joystick.init()
                    self.backend = "pygame"
                    self.connected = True
                    
                    # Keyword-based steering wheel detection
                    name_raw = self.py_joystick.get_name()
                    name_lower = name_raw.lower()
                    if any(kw in name_lower for kw in self.WHEEL_KEYWORDS):
                        self.detected_device_type = "wheel"
                    else:
                        self.detected_device_type = "gamepad"

                    # Open SDL GameController handle if available
                    if self.sdl_dll and hasattr(self, '_sdl_gc_open'):
                        try:
                            self._sdl_gc_handle = self._sdl_gc_open(0)
                        except Exception:
                            self._sdl_gc_handle = None

                    # Open DirectInput / SDL2 Haptic device for Force Feedback
                    if self.sdl_dll:
                        try:
                            if hasattr(self, '_sdl_joy_open') and hasattr(self, '_sdl_haptic_open_from_joy'):
                                joy_ptr = self._sdl_joy_open(0)
                                if joy_ptr and hasattr(self, '_sdl_joy_is_haptic') and self._sdl_joy_is_haptic(joy_ptr):
                                    self.sdl_haptic = self._sdl_haptic_open_from_joy(joy_ptr)
                            if not self.sdl_haptic and hasattr(self, '_sdl_num_haptics') and self._sdl_num_haptics() > 0:
                                self.sdl_haptic = self._sdl_haptic_open(0)

                            if self.sdl_haptic and hasattr(self, '_sdl_haptic_query'):
                                query = self._sdl_haptic_query(self.sdl_haptic)
                                if query & 1:  # SDL_HAPTIC_CONSTANT = 1
                                    self.supports_constant_ffb = True
                                    eff = SDL_HapticEffect()
                                    eff.type = 1
                                    eff.constant.type = 1
                                    eff.constant.direction.type = 1  # CARTESIAN
                                    eff.constant.direction.dir[0] = 1  # X steering axis
                                    eff.constant.direction.dir[1] = 0
                                    eff.constant.direction.dir[2] = 0
                                    eff.constant.length = 0xFFFFFFFF  # Infinite
                                    eff.constant.level = 0
                                    self.sdl_haptic_effect_id = self._sdl_haptic_new_effect(self.sdl_haptic, ctypes.byref(eff))
                                    if self.sdl_haptic_effect_id >= 0 and hasattr(self, '_sdl_haptic_run_effect'):
                                        self._sdl_haptic_run_effect(self.sdl_haptic, self.sdl_haptic_effect_id, 1)
                        except Exception:
                            self.sdl_haptic = None
                            self.sdl_haptic_effect_id = -1
                            self.supports_constant_ffb = False

                    if self.is_wheel():
                        ffb_desc = "Constant FFB" if self.supports_constant_ffb else "DirectInput"
                        self.gamepad_name = f"🏎️ {name_raw} ({ffb_desc})"
                    else:
                        self.gamepad_name = f"🎮 {name_raw}"
                    return True
            except Exception:
                pass

        self.backend = "keyboard"
        self.connected = False
        self.detected_device_type = "keyboard"
        self.gamepad_name = "Tastiera (WASD / Frecce)"
        return False

    def poll_inputs(self, key_states=None):
        now = time.time()
        if not self.connected and (now - self.last_scan_time > 1.5):
            self.scan_controllers()
            self.last_scan_time = now

        steer = 0.0
        throttle = 0.0
        brake = 0.0
        tv_bias = 0.0
        reset_req = False
        raw_steer_val = 0.0

        if self.backend == "xinput" and self.xinput_dll:
            try:
                state = XINPUT_STATE()
                res = self.xinput_dll.XInputGetState(self.xinput_user_index, ctypes.byref(state))
                if res == 0:
                    gp = state.Gamepad
                    raw_lx = gp.sThumbLX / 32768.0
                    raw_steer_val = raw_lx
                    
                    if self.is_wheel():
                        # Volante: 1:1 lineare diretta su [-125°, +125°], zero filtri esponenziali o deadband
                        steer = -raw_lx if self.invert_steer else raw_lx
                    else:
                        # Gamepad: deadzone + filtro cubico per controllo a pollice
                        if abs(raw_lx) > self.steer_deadzone:
                            sign = math.copysign(1.0, raw_lx)
                            norm = (abs(raw_lx) - self.steer_deadzone) / (1.0 - self.steer_deadzone)
                            norm = min(max(norm, 0.0), 1.0)
                            shaped = sign * (norm ** self.steer_sensitivity)
                            steer = -shaped if self.invert_steer else shaped
                        
                    raw_rt = gp.bRightTrigger / 255.0
                    raw_lt = gp.bLeftTrigger / 255.0
                    
                    if raw_rt > self.trigger_deadzone:
                        throttle = (raw_rt - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                    if raw_lt > self.trigger_deadzone:
                        brake = (raw_lt - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                        
                    if gp.wButtons & 0x0100:  # LB
                        tv_bias = -1.0
                    elif gp.wButtons & 0x0200:  # RB
                        tv_bias = 1.0
                        
                    if (gp.wButtons & 0x0020) or (gp.wButtons & 0x1000 and throttle < 0.05 and brake > 0.8):
                        reset_req = True
                else:
                    self.connected = False
                    self.backend = "keyboard"
            except Exception:
                self.connected = False

        elif self.backend == "pygame" and self.py_joystick:
            try:
                import pygame
                pygame.event.pump()
                
                raw_steer = self.py_joystick.get_axis(0)
                raw_steer_val = raw_steer
                
                if self.is_wheel():
                    # Volante PC: 1:1 lineare diretta su [-125°, +125°], zero filtri o deadband
                    steer = -raw_steer if self.invert_steer else raw_steer
                else:
                    # Gamepad: deadzone + filtro cubico per controllo a pollice
                    if abs(raw_steer) > self.steer_deadzone:
                        sign = math.copysign(1.0, raw_steer)
                        norm = (abs(raw_steer) - self.steer_deadzone) / (1.0 - self.steer_deadzone)
                        norm = min(max(norm, 0.0), 1.0)
                        shaped = sign * (norm ** self.steer_sensitivity)
                        steer = -shaped if self.invert_steer else shaped

                num_axes = self.py_joystick.get_num_axes()
                if self.is_wheel():
                    # Racing wheel pedals (Logitech G29/G920, Fanatec, Thrustmaster)
                    # Axis 1 = Gas/Throttle, Axis 2 = Brake (unpressed: 1.0, fully pressed: -1.0)
                    if num_axes >= 3:
                        raw_a1 = self.py_joystick.get_axis(1)
                        raw_a2 = self.py_joystick.get_axis(2)
                        val_throt = max((1.0 - raw_a1) / 2.0, 0.0) if raw_a1 != 0.0 else 0.0
                        val_brake = max((1.0 - raw_a2) / 2.0, 0.0) if raw_a2 != 0.0 else 0.0
                        
                        # Fallback to axes 4 & 5 if combined/gamepad layout
                        if num_axes >= 6 and val_throt < 0.02 and val_brake < 0.02:
                            val_brake = max((self.py_joystick.get_axis(4) + 1.0) / 2.0, 0.0)
                            val_throt = max((self.py_joystick.get_axis(5) + 1.0) / 2.0, 0.0)

                        if val_throt > self.trigger_deadzone:
                            throttle = (val_throt - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                        if val_brake > self.trigger_deadzone:
                            brake = (val_brake - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                else:
                    # Standard gamepad triggers
                    if num_axes >= 6:
                        raw_lt = self.py_joystick.get_axis(4)
                        raw_rt = self.py_joystick.get_axis(5)
                        lt_norm = max((raw_lt + 1.0) / 2.0, 0.0)
                        rt_norm = max((raw_rt + 1.0) / 2.0, 0.0)
                        if lt_norm > self.trigger_deadzone:
                            brake = (lt_norm - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                        if rt_norm > self.trigger_deadzone:
                            throttle = (rt_norm - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
                    elif num_axes >= 3:
                        raw_trig = self.py_joystick.get_axis(2)
                        if raw_trig > self.trigger_deadzone:
                            throttle = raw_trig
                        elif raw_trig < -self.trigger_deadzone:
                            brake = -raw_trig

                if self.py_joystick.get_num_buttons() >= 6:
                    if self.py_joystick.get_button(4):
                        tv_bias = -1.0
                    elif self.py_joystick.get_button(5):
                        tv_bias = 1.0
                if self.py_joystick.get_num_buttons() >= 7:
                    if self.py_joystick.get_button(6) or self.py_joystick.get_button(0):
                        reset_req = True
            except Exception:
                self.connected = False

        if key_states:
            tgt_steer = 0.0
            tgt_throt = 0.0
            tgt_brake = 0.0

            if key_states.get(Qt.Key_Left, False) or key_states.get(Qt.Key_A, False):
                tgt_steer = -0.85 if not self.invert_steer else 0.85
                raw_steer_val = -1.0
            elif key_states.get(Qt.Key_Right, False) or key_states.get(Qt.Key_D, False):
                tgt_steer = 0.85 if not self.invert_steer else -0.85
                raw_steer_val = 1.0

            if key_states.get(Qt.Key_Up, False) or key_states.get(Qt.Key_W, False):
                tgt_throt = 1.0
            if key_states.get(Qt.Key_Down, False) or key_states.get(Qt.Key_S, False):
                tgt_brake = 1.0
            if key_states.get(Qt.Key_Q, False):
                tv_bias = -1.0
            elif key_states.get(Qt.Key_E, False):
                tv_bias = 1.0
            if key_states.get(Qt.Key_R, False) or key_states.get(Qt.Key_Space, False):
                reset_req = True

            # Smooth progressive keyboard ramping (rate-limiting key transitions)
            rate_steer = 0.18
            rate_pedal = 0.15
            self._kb_steer += rate_steer * (tgt_steer - self._kb_steer)
            self._kb_throttle += rate_pedal * (tgt_throt - self._kb_throttle)
            self._kb_brake += rate_pedal * (tgt_brake - self._kb_brake)

            if abs(self._kb_steer) > 0.005:
                steer = self._kb_steer
            if self._kb_throttle > 0.01:
                throttle = self._kb_throttle
            if self._kb_brake > 0.01:
                brake = self._kb_brake
        else:
            self._kb_steer *= 0.85
            self._kb_throttle *= 0.70
            self._kb_brake *= 0.70

        return {
            'steer': float(np.clip(steer, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0)),
            'brake': float(np.clip(brake, 0.0, 1.0)),
            'tv_bias': float(np.clip(tv_bias, -1.0, 1.0)),
            'reset': bool(reset_req),
            'raw_steer': float(raw_steer_val),
            'connected': bool(self.connected),
            'device_name': str(self.gamepad_name),
            'is_wheel': bool(self.is_wheel())
        }


# =============================================================================
# 3. REAL-TIME 14-DOF VEHICLE DYNAMICS THREAD (CONFIGURABLE 250-500 Hz)
# =============================================================================

