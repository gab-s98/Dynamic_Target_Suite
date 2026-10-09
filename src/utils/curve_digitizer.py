import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, QFormLayout, 
                             QGroupBox, QMessageBox, QListWidget, QComboBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.points = []
        self.calib_points = {'origin': None, 'x_max': None, 'y_max': None}
        self.current_action = None
        self.pixmap_orig = None
        
    def set_pixmap(self, pixmap):
        self.pixmap_orig = pixmap
        self.setPixmap(pixmap)
        
    def mousePressEvent(self, event):
        if self.pixmap_orig is None:
            return
            
        pos = event.pos()
        # Scale position to original image size if label is scaled
        # (Assuming the label is not scaled for simplicity, but handled via ScrollArea ideally)
        
        if self.current_action == 'origin':
            self.calib_points['origin'] = pos
        elif self.current_action == 'x_max':
            self.calib_points['x_max'] = pos
        elif self.current_action == 'y_max':
            self.calib_points['y_max'] = pos
        elif self.current_action == 'gather':
            self.points.append(pos)
            
        self.update_drawing()
        self.parent().parent().update_status()

    def update_drawing(self):
        if self.pixmap_orig is None:
            return
        temp_pixmap = self.pixmap_orig.copy()
        painter = QPainter(temp_pixmap)
        
        # Draw Calibration points
        if self.calib_points['origin']:
            painter.setPen(QPen(QColor('red'), 5))
            painter.drawPoint(self.calib_points['origin'])
        if self.calib_points['x_max']:
            painter.setPen(QPen(QColor('green'), 5))
            painter.drawPoint(self.calib_points['x_max'])
        if self.calib_points['y_max']:
            painter.setPen(QPen(QColor('blue'), 5))
            painter.drawPoint(self.calib_points['y_max'])
            
        # Draw Data points
        painter.setPen(QPen(QColor('yellow'), 4))
        for p in self.points:
            painter.drawPoint(p)
            
        painter.end()
        self.setPixmap(temp_pixmap)
        
    def clear_data_points(self):
        self.points = []
        self.update_drawing()


class CurveDigitizerWindow(QMainWindow):
    def __init__(self):
        super(CurveDigitizerWindow, self).__init__()
        self.setWindowTitle("Curve Digitizer (Polyfit)")
        self.setGeometry(150, 150, 1000, 700)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left Panel (Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        main_layout.addWidget(left_panel)
        
        # Image Loading
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(self.btn_load)
        
        # Calibration Group
        calib_group = QGroupBox("1. Calibration")
        calib_layout = QVBoxLayout(calib_group)
        
        form_layout = QFormLayout()
        self.val_origin_x = QLineEdit("0.0")
        self.val_origin_y = QLineEdit("0.0")
        form_layout.addRow("Origin Physical X:", self.val_origin_x)
        form_layout.addRow("Origin Physical Y:", self.val_origin_y)
        
        self.val_xmax = QLineEdit("30.0")
        form_layout.addRow("X-Max Physical Val:", self.val_xmax)
        
        self.val_ymax = QLineEdit("2.0")
        form_layout.addRow("Y-Max Physical Val:", self.val_ymax)

        self.x_unit_combo = QComboBox()
        self.x_unit_combo.addItems([
            "Wheel Travel [mm] (Auto-scaled to [m] for 14-DOF)",
            "Wheel Travel [m]",
            "Steer Angle [deg]"
        ])
        form_layout.addRow("X-Axis Physical Unit:", self.x_unit_combo)

        self.y_unit_combo = QComboBox()
        self.y_unit_combo.addItems([
            "Angle [deg] (Camber / Toe)",
            "Height / Displacement [mm] (Auto-scaled to [m])",
            "Height / Displacement [m]",
            "Fraction / Ratio [-] (Anti-dive/squat)"
        ])
        form_layout.addRow("Y-Axis Physical Unit:", self.y_unit_combo)

        calib_layout.addLayout(form_layout)
        
        self.btn_set_origin = QPushButton("Pick Origin (Red)")
        self.btn_set_origin.clicked.connect(lambda: self.set_action('origin'))
        calib_layout.addWidget(self.btn_set_origin)
        
        self.btn_set_xmax = QPushButton("Pick X-Max (Green)")
        self.btn_set_xmax.clicked.connect(lambda: self.set_action('x_max'))
        calib_layout.addWidget(self.btn_set_xmax)
        
        self.btn_set_ymax = QPushButton("Pick Y-Max (Blue)")
        self.btn_set_ymax.clicked.connect(lambda: self.set_action('y_max'))
        calib_layout.addWidget(self.btn_set_ymax)
        
        left_layout.addWidget(calib_group)
        
        # Data Gathering Group
        data_group = QGroupBox("2. Gather Points")
        data_layout = QVBoxLayout(data_group)
        
        self.btn_gather = QPushButton("Start Picking Curve Points (Yellow)")
        self.btn_gather.clicked.connect(lambda: self.set_action('gather'))
        data_layout.addWidget(self.btn_gather)
        
        self.btn_clear = QPushButton("Clear Data Points")
        self.btn_clear.clicked.connect(self.clear_data)
        data_layout.addWidget(self.btn_clear)
        
        left_layout.addWidget(data_group)
        
        # Analysis Group
        analysis_group = QGroupBox("3. Polynomial Fit (Degree 3)")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.btn_fit = QPushButton("Compute 3rd-Degree PolyFit")
        self.btn_fit.clicked.connect(self.compute_fit)
        analysis_layout.addWidget(self.btn_fit)
        
        self.lbl_result = QLabel("Coefficients:\nN/A")
        analysis_layout.addWidget(self.lbl_result)
        
        left_layout.addWidget(analysis_group)

        # Export Group
        export_group = QGroupBox("4. Export to 14-DOF")
        export_layout = QVBoxLayout(export_group)

        self.curve_name_combo = QComboBox()
        self.curve_name_combo.addItems([
            "camber_gain_f",
            "camber_gain_r",
            "toe_gain_f",
            "toe_gain_r",
            "rollcenter_f",
            "rollcenter_r",
            "anti_dive",
            "anti_squat",
            "ackermann",
        ])
        export_layout.addWidget(QLabel("Curve Name:"))
        export_layout.addWidget(self.curve_name_combo)

        self.btn_export = QPushButton("Export Coefficients to JSON")
        self.btn_export.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_to_json)
        export_layout.addWidget(self.btn_export)

        left_layout.addWidget(export_group)

        
        # Status
        self.lbl_status = QLabel("Status: Idle")
        left_layout.addWidget(self.lbl_status)
        
        left_layout.addStretch()
        
        # Right Panel (Image Area)
        from PyQt5.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.image_label = ImageLabel(self.scroll_area)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)
        
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.set_pixmap(pixmap)
            self.image_label.resize(pixmap.width(), pixmap.height())
            self.lbl_status.setText(f"Loaded: {file_name}")

    def set_action(self, action):
        self.image_label.current_action = action
        self.update_status()
        
    def clear_data(self):
        self.image_label.clear_data_points()
        
    def update_status(self):
        pts = len(self.image_label.points)
        c_pts = sum(1 for v in self.image_label.calib_points.values() if v is not None)
        action = self.image_label.current_action
        self.lbl_status.setText(f"Action: {action} | Calib: {c_pts}/3 | Points: {pts}")

    def compute_fit(self):
        # 1. Validate Calibration
        c = self.image_label.calib_points
        if not c['origin'] or not c['x_max'] or not c['y_max']:
            QMessageBox.warning(self, "Error", "Please pick Origin, X-Max, and Y-Max first.")
            return
            
        if len(self.image_label.points) < 4:
            QMessageBox.warning(self, "Error", "Please pick at least 4 points for a 3rd degree fit.")
            return
            
        try:
            p_orig_x = float(self.val_origin_x.text())
            p_orig_y = float(self.val_origin_y.text())
            p_xmax = float(self.val_xmax.text())
            p_ymax = float(self.val_ymax.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Physical values must be numeric.")
            return
            
        # 2. Pixel mappings
        # Origin
        px_ox = c['origin'].x()
        px_oy = c['origin'].y()
        
        # Pixels per unit
        # Note: Y axis in pixels goes DOWN, physical Y usually goes UP
        px_per_x = (c['x_max'].x() - px_ox) / (p_xmax - p_orig_x) if p_xmax != p_orig_x else 1
        px_per_y = (c['y_max'].y() - px_oy) / (p_ymax - p_orig_y) if p_ymax != p_orig_y else 1
        
        if px_per_x == 0 or px_per_y == 0:
            QMessageBox.warning(self, "Error", "Calibration points overlap on axis.")
            return
            
        # 3. Transform points
        phys_x = []
        phys_y = []
        for p in self.image_label.points:
            x_val = p_orig_x + (p.x() - px_ox) / px_per_x
            y_val = p_orig_y + (p.y() - px_oy) / px_per_y
            phys_x.append(x_val)
            phys_y.append(y_val)
            
        phys_x = np.array(phys_x)
        phys_y = np.array(phys_y)
        
        # 4. Polyfit (Degree 3)
        # Raw fit: y(x_phys) = c3*x^3 + c2*x^2 + c1*x + c0
        coeffs_raw = np.polyfit(phys_x, phys_y, 3)
        self._last_raw_coeffs = coeffs_raw

        # Auto-scaling to 14-DOF SI Standards (wheel travel z in [m], angles in [deg], lengths in [m]):
        x_scale = 1000.0 if self.x_unit_combo.currentIndex() == 0 else 1.0
        y_scale = 0.001 if self.y_unit_combo.currentIndex() == 1 else 1.0

        a3 = float(coeffs_raw[0] * (x_scale**3) * y_scale)
        a2 = float(coeffs_raw[1] * (x_scale**2) * y_scale)
        a1 = float(coeffs_raw[2] * (x_scale**1) * y_scale)
        a0 = float(coeffs_raw[3] * y_scale)

        self._last_coeffs = np.array([a3, a2, a1, a0])  # stored for export to 14-DOF

        res_str = (f"Fit su Dati ({self.x_unit_combo.currentText().split()[0]}):\n"
                   f"  y = {coeffs_raw[0]:.4e}x³ + {coeffs_raw[1]:.4e}x² + {coeffs_raw[2]:.4e}x + {coeffs_raw[3]:.4e}\n\n"
                   f"Polinomio Scalato 14-DOF SI (z [m]):\n"
                   f"  y(z) = {a3:.4e}z³ + {a2:.4e}z² + {a1:.4e}z + {a0:.4e}")
        self.lbl_result.setText(res_str)

        # Plot verification with matplotlib
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(phys_x, phys_y, 'ro', label='Punti Digitalizzati')
        x_fit = np.linspace(min(phys_x), max(phys_x), 100)
        y_fit = np.polyval(coeffs_raw, x_fit)
        ax.plot(x_fit, y_fit, 'b-', label='Fit 3° Grado')
        ax.set_xlabel(self.x_unit_combo.currentText())
        ax.set_ylabel(self.y_unit_combo.currentText())
        ax.set_title('Digitizer Result vs PolyFit')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def export_to_json(self):
        """Save the last-fitted polynomial coefficients (scaled for 14-DOF) as a named JSON file."""
        import json
        result_text = self.lbl_result.text()
        if "N/A" in result_text or not hasattr(self, '_last_coeffs'):
            QMessageBox.warning(self, "Error", "Compute a PolyFit first before exporting.")
            return

        curve_name = self.curve_name_combo.currentText()
        coeffs = self._last_coeffs.tolist()

        data = {
            "curve_name": curve_name,
            "degree": 3,
            "coeffs": coeffs,   # [a3, a2, a1, a0] ready for Vehicle14DOF
            "units": {
                "input_x": "Wheel Travel z [m]" if "ackermann" not in curve_name else "Steer angle delta [rad]",
                "output_y": "deg" if "camber" in curve_name or "toe" in curve_name or "ackermann" in curve_name else "m"
            },
            "description": f"y(z) = {coeffs[0]:.4e}*z^3 + {coeffs[1]:.4e}*z^2 + {coeffs[2]:.4e}*z + {coeffs[3]:.4e}"
        }

        from PyQt5.QtWidgets import QFileDialog
        options = QFileDialog.Options()
        default_name = f"{curve_name}_poly3.json"
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Kinematic Map", default_name,
            "JSON Files (*.json)", options=options
        )
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(data, f, indent=2)
            self.lbl_status.setText(f"Exported: {file_name}")
            QMessageBox.information(self, "Export OK",
                f"Coefficients for '{curve_name}' saved to:\n{file_name}\n\n"
                f"Load it in vehicle_14dof.py or VehicleParams.set_kinematic_map().")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CurveDigitizerWindow()
    win.show()
    sys.exit(app.exec_())
