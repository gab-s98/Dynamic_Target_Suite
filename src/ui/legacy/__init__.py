"""
Dynamic Target Suite - Legacy UI Package
"""
try:
    from ..main_gui_classic import MainWindowClassic
except ImportError:
    pass

try:
    from ..main_gui_pro import MainWindowPro
except ImportError:
    pass
