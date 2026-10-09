import ast
import os

MONOLITH = "interactive_driver_14dof.py"

def split_monolith():
    with open(MONOLITH, "r", encoding="utf-8") as f:
        source = f.read()
    
    lines = source.split("\n")
    tree = ast.parse(source)

    # Find the top imports block (everything before the first class or function)
    # Actually, we can just grab all imports via AST and generate an import block.
    # But to be safe and preserve comments, let's find the first node that is a ClassDef or FunctionDef
    # (excluding setup_global_exception_handler)
    first_major_node_line = 0
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name == "setup_global_exception_handler":
                continue
            first_major_node_line = node.lineno - 1
            break
            
    top_imports = "\n".join(lines[:first_major_node_line])

    # Now map every node in the body to its source text
    sections = {}
    
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign)):
            # determine end line by looking at next node
            start_line = node.lineno - 1
            # Try to grab decorators
            if hasattr(node, "decorator_list") and node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1
                
            end_line = len(lines)
            if i + 1 < len(tree.body):
                next_node = tree.body[i+1]
                end_line = next_node.lineno - 1
                if hasattr(next_node, "decorator_list") and next_node.decorator_list:
                    end_line = next_node.decorator_list[0].lineno - 1
            
            # Identify name
            name = None
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                name = node.name
            elif isinstance(node, ast.Assign):
                if node.targets and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id

            if name:
                sections[name] = "\n".join(lines[start_line:end_line])

    def write_module(filepath, names, extra_imports=""):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(top_imports)
            f.write("\n" + extra_imports + "\n\n")
            for name in names:
                if name in sections:
                    f.write(sections[name])
                    f.write("\n\n")

    # Mapping
    gamepad_classes = ["XINPUT_GAMEPAD", "XINPUT_STATE", "XINPUT_VIBRATION", "XINPUT_VIBRATION_EX", 
                       "SDL_HapticDirection", "SDL_HapticConstant", "SDL_HapticEffect", "FFB_PRESETS", "GamepadManager"]
    
    physics_classes = ["PhysicsThread14DOF"]
    
    viewport_classes = ["car_2d_pt", "draw_2d_wheel", "OpenGLViewport3D", "get_viewport_class", "_LegacyViewport3D"]
    
    telemetry_classes = ["TelemetryPopoutDialog", "CustomTelemetryLayoutDialog", "LapSelectionDialog", "FullTelemetryWindow"]
    
    controls_classes = ["TrackEditorCanvas", "RoadProfileEditorDialog", "TrackDesignerDialog"]
    
    main_window_classes = ["Interactive14DOFDrivingWindow"]

    write_module("core/gamepad_manager.py", gamepad_classes)
    write_module("core/physics_thread.py", physics_classes, "from core.gamepad_manager import GamepadManager")
    write_module("ui/sim_viewport.py", viewport_classes, "from core.physics_thread import PhysicsThread14DOF")
    write_module("ui/sim_telemetry.py", telemetry_classes)
    write_module("ui/sim_controls.py", controls_classes, "from ui.sim_viewport import get_viewport_class")
    write_module("ui/sim_main_window.py", main_window_classes, 
        "from core.physics_thread import PhysicsThread14DOF\n"
        "from ui.sim_viewport import get_viewport_class\n"
        "from ui.sim_telemetry import FullTelemetryWindow, TelemetryPopoutDialog, LapSelectionDialog, CustomTelemetryLayoutDialog\n"
        "from ui.sim_controls import TrackDesignerDialog, RoadProfileEditorDialog\n"
        "from core.gamepad_manager import GamepadManager\n"
    )

    print("Splitting complete.")

if __name__ == "__main__":
    split_monolith()


