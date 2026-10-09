"""
gui_canvas.py
-------------
Dynamic Target Suite ? Matplotlib Canvas Integration for PyQt5.
Provides high-performance, resizable FigureCanvas widgets optimized for
real-time vehicle dynamics telemetry and multi-subplot engineering diagrams.
"""

from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MplCanvas(FigureCanvas):
    """
    Standard Matplotlib canvas widget with default responsive layout,
    expanding size policy, and single subplot initialization.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(80, 60)
        self.updateGeometry()