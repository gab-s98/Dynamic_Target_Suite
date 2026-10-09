"""
interactive_driver_14dof.py
---------------------------
BACKWARDS COMPATIBILITY PROXY MODULE.

This 10,500-line monolith has been split into clean, modular files.
All classes have been moved to `core/` and `ui/` directories.
This file now simply re-exports them to avoid breaking imports in other parts
of the Dynamic Target Suite (e.g. tests, main_gui, MDO modules).

DO NOT add new code here. Add it to the specific modules instead.
"""

from src.core.gamepad_manager import GamepadManager, FFB_PRESETS, XINPUT_VIBRATION_EX
from src.core.physics_thread import PhysicsThread14DOF
from src.ui.sim_viewport import (
    OpenGLViewport3D, _LegacyViewport3D, get_viewport_class, HAS_OPENGL,
    setup_global_exception_handler
)
Viewport3DChaseCamera = _LegacyViewport3D

from src.ui.sim_telemetry import TelemetryPopoutDialog, CustomTelemetryLayoutDialog, LapSelectionDialog, FullTelemetryWindow
from src.ui.sim_controls import TrackEditorCanvas, RoadProfileEditorDialog, TrackDesignerDialog
from src.ui.sim_main_window import Interactive14DOFDrivingWindow

__all__ = [
    'GamepadManager', 'FFB_PRESETS', 'XINPUT_VIBRATION_EX',
    'PhysicsThread14DOF',
    'OpenGLViewport3D', '_LegacyViewport3D', 'Viewport3DChaseCamera',
    'get_viewport_class', 'HAS_OPENGL', 'setup_global_exception_handler',
    'TelemetryPopoutDialog', 'CustomTelemetryLayoutDialog', 'LapSelectionDialog', 'FullTelemetryWindow',
    'TrackEditorCanvas', 'RoadProfileEditorDialog', 'TrackDesignerDialog',
    'Interactive14DOFDrivingWindow'
]


if __name__ == '__main__':
    from src.core.simulator import main
    main()
