"""
gui_dialogs.py
--------------
Dynamic Target Suite — Specialized Engineering Dialogs & Chart Viewers.
Includes:
- ChartGuideDialog: in-depth physical and mathematical guide reader for plots.
- PopoutPlotDialog: resizable, fullscreen-capable high-definition plot inspector.
- translate_figure_language & FIGURE_IT_TO_EN: dynamic translation engine
  for academic papers and SAE reports (Italian <-> English).
"""

import io
import pickle
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QCheckBox, QFileDialog, QMessageBox
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)


class ChartGuideDialog(QDialog):
    """Dialog for displaying in-app physical and mathematical plot interpretations."""
    def __init__(self, title, html_content, parent=None):
        super(ChartGuideDialog, self).__init__(parent)
        self.setWindowTitle(f"Guida Fisica & Interpretazione: {title}")
        self.resize(850, 650)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("background-color: #ffffff; color: #2d3436; font-size: 13px; padding: 12px;")
        browser.setHtml(html_content)
        layout.addWidget(browser)

        btn_close = QPushButton("Chiudi Guida")
        btn_close.setStyleSheet("font-weight: bold; padding: 8px 16px; background-color: #2d3436; color: white; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)


FIGURE_IT_TO_EN = {
    # Headers & Standards
    "Handling Diagram (Piano Cartesiano)": "Handling Diagram ($a_y/g$ vs $L\\cdot\\rho - \\delta$)",
    "Handling Diagram (SAE / ISO)": "Vehicle Handling Diagram",
    "Handling Diagram": "Vehicle Handling Diagram",
    "Dinamica del Veicolo": "Vehicle Dynamics",
    "Mappa di Equilibrio": "Equilibrium Map",

    # Handling Diagram
    "Angolo di Deriva / Deformazione": "Equivalent Slip / Handling Metric",
    "Angolo di Deriva": "Sideslip Angle",
    "Angolo Volante": "Steering Angle",
    "Angolo di Sterzo": "Steering Angle",
    "Angolo Sterzo": "Steering Angle",
    "Accelerazione Laterale": "Lateral Acceleration",
    "Acc. Laterale": "Lateral Acc.",
    "Accelerazione Longitudinale": "Longitudinal Acceleration",
    "Acc. Longitudinale": "Longitudinal Acc.",
    "Linea Neutra": "Neutral Line",
    "Sottosterzante": "Understeering",
    "Sovrasterzante": "Oversteering",
    "Sottosterzo": "Understeer",
    "Sovrasterzo": "Oversteer",
    "Neutro": "Neutral",

    # YMD & Moment Method
    "Momento Imbardata Totale": "Total Yaw Moment",
    "Momento Imbardata": "Yaw Moment",
    "Momento d'Imbardata": "Yaw Moment",
    "Inviluppi YMD Multi-Velocità": "Multi-Speed YMD Envelopes",
    "Derivate di Controllo & Stabilità vs": "Control & Stability Derivatives vs",
    "Grip Limite & Margine di Controllo vs": "Peak Lateral Grip & Control Margin vs",
    "Derivata Controllo": "Control Derivative",
    "Derivata Stabilità": "Stability Derivative",
    "Margine Controllo": "Control Margin",
    "Margine di Controllo": "Control Margin",
    "Grip Limite": "Limit Lateral Grip",
    "Autorità Controllo": "Control Authority",
    "Stabilità Direzionale": "Directional Stability",

    # G-G-V
    "G-G-V Performance Bowl: Baseline vs Target Ottimizzato": "G-G-V Performance Bowl: Baseline vs Optimized Target",
    "Inviluppo di Accelerazione": "Acceleration Envelope",
    "Velocità di Avanzamento": "Forward Velocity",
    "Velocità": "Forward Speed",

    # MAPs
    "Regioni di Ingresso": "Input Equilibrium Regions",
    "Regioni di Uscita": "Output Equilibrium Regions",
    "Regione Mista": "Mixed Equilibrium Region",
    "Regione di Ingresso": "Input Region",
    "Regione di Uscita": "Output Region",
    "Velocità vs Grip": "Speed vs Lateral Grip",
    "Curve di Trim": "Trim Curves",
    "Regione Ammissibile": "Admissible Region",

    # Dynamic Stability & Sensitivity Maps (DNA)
    "Mappe DNA del Veicolo": "Vehicle Dynamic Stability & Sensitivity Maps",
    "Mappe del DNA": "Dynamic Stability & Sensitivity Maps",
    "Modi Oscillatori nel Piano": "Oscillatory Dynamic Modes in",
    "Modi Oscillatori": "Oscillatory Dynamic Modes",
    "Rigidezza Dinamica Eq.": "Equivalent Dynamic Stiffness",
    "Rigidezza Dinamica": "Dynamic Stiffness",
    "Forzamento Deriva": "Sideslip Forcing Map",
    "Forzamento Imbardata": "Yaw Forcing Map",
    "Mappa 2D Gradiente di Sottosterzo": "2D Understeer Gradient Map",
    "Gradiente Sottosterzo": "Understeer Gradient",
    "Gradiente di Sottosterzo": "Understeer Gradient",
    "Gradiente di Curvatura": "Curvature Gradient",
    "Gradiente Curvatura Effettivo": "Effective Curvature Gradient",
    "Soglia Neutra": "Neutral Threshold",
    "Soglia Instabilità": "Dynamic Instability Threshold",
    "Limite Stabilità": "Stability Boundary",
    "Limite Instabilità": "Instability Boundary",
    "Regione Sottosterzante": "Understeering Region",
    "Regione Sovrasterzante": "Oversteering Region",
    "Continuo:": "Solid:",
    "Tratteggio:": "Dashed:",
    "Rosso:": "Red:",
    "Verde: Stabile, Rosso: Instabile": "Green: Stable, Red: Unstable",
    "Blu: Sottosterzo, Rosso: Sovrasterzo": "Blue: Understeer, Red: Oversteer",

    # Steering Torque
    "Coppia allo Sterzo Totale & Feedback Pilota": "Total Steering Torque & Driver Feedback",
    "Coppia allo Sterzo": "Steering Wheel Torque",
    "Coppia allo Sterzo Risultante": "Net Steering Wheel Torque",
    "Coppia di Autoallineamento": "Self-Aligning Torque",
    "Coppia al Volante": "Steering Wheel Torque",
    "Braccio Meccanico": "Mechanical Trail",
    "Braccio Pneumatico": "Pneumatic Trail",

    # Sizing & Suspension
    "Ripartizione Rigidezza Rollio": "Roll Stiffness Breakdown",
    "Rigidezza di Rollio": "Roll Stiffness",
    "Rigidezza di Beccheggio": "Pitch Stiffness",
    "Molle Sospensione": "Suspension Springs",
    "Barra Antirollio ARB": "Anti-Roll Bar (ARB)",
    "Barra Antirollio": "Anti-Roll Bar",
    "Corsa Ammortizzatore TTX25": "TTX25 Damper Stroke",
    "Corsa TTX25": "TTX25 Damper Stroke",
    "Corsa Sospensione / Damper": "Suspension / Damper Travel",
    "Corsa Damper & Tamponi Finecorsa": "Damper Stroke & Bumpstops",
    "Static Sag / Droop Disp. (1g)": "Static Sag / Droop (1g)",
    "Compressione Statica a 1g": "Static 1g Sag",
    "Corsa Libera prima Tampone": "Free Bumpstop Headroom",
    "Corsa Max TTX25": "Max TTX25 Stroke",
    "Finecorsa Bump": "Bump Travel Limit",
    "Finecorsa Rebound": "Rebound Travel Limit",
    "Damper Ant.": "Front Damper",
    "Damper Post.": "Rear Damper",
    "Avantreno (Front)": "Front Axle",
    "Retrotreno (Rear)": "Rear Axle",
    "Avantreno": "Front",
    "Retrotreno": "Rear",

    # Damper Dyno
    "Curva Caratteristica Forza-Velocità F-V": "Damper Characteristic Force-Velocity (F-V) Curve",
    "Caratteristica Forza-Velocità": "Force-Velocity (F-V) Characteristic",
    "Rapporto di Smorzamento Modale Effettivo": "Effective Modal Damping Ratio",
    "Rapporto di Smorzamento Modale": "Modal Damping Ratio",
    "Rapporto di Smorzamento": "Damping Ratio",
    "Smorzamento": "Damping",
    "Velocità Stelo": "Damper Shaft Velocity",
    "Forza Smorzamento": "Damping Force",
    "Forza di Smorzamento": "Damping Force",
    "Compressione": "Compression",
    "Estensione": "Rebound",
    "Smorzamento Heave/Ride": "Ride / Heave Damping",
    "Smorzamento Rollio": "Roll Damping",

    # Transient & Bode
    "Risposta al Gradino di Sterzo": "Step-Steer Transient Response",
    "Risposta in Frequenza": "Frequency Response (Bode)",
    "Valori Propri e Modi di Rollio/Imbardata": "Chassis Eigenvalues & Modal Poles",
    "Analisi di Sensibilità": "Sensitivity Analysis",
    "Tempo": "Time",
    "Frequenza": "Frequency",
    "Ampiezza": "Magnitude",
    "Fase": "Phase",
    "Parte Reale": "Real Part",
    "Parte Immaginaria": "Imaginary Part"
}


def translate_figure_language(fig, to_english=True):
    """
    Translates all titles, axes labels, colorbar labels, legends, and text annotations
    in a matplotlib Figure between Italian and formal SAE/Academic English.
    """
    if fig is None:
        return
    mapping = FIGURE_IT_TO_EN if to_english else {v: k for k, v in FIGURE_IT_TO_EN.items()}

    for ax in fig.axes:
        # 1. Subplot Title
        title = ax.get_title()
        if title:
            for k, v in mapping.items():
                if k in title:
                    title = title.replace(k, v)
            ax.set_title(title)

        # 2. X and Y Labels
        xlabel = ax.get_xlabel()
        if xlabel:
            for k, v in mapping.items():
                if k in xlabel:
                    xlabel = xlabel.replace(k, v)
            ax.set_xlabel(xlabel)

        ylabel = ax.get_ylabel()
        if ylabel:
            for k, v in mapping.items():
                if k in ylabel:
                    ylabel = ylabel.replace(k, v)
            ax.set_ylabel(ylabel)

        if hasattr(ax, 'get_zlabel'):
            zlabel = ax.get_zlabel()
            if zlabel:
                for k, v in mapping.items():
                    if k in zlabel:
                        zlabel = zlabel.replace(k, v)
                ax.set_zlabel(zlabel)

        # 3. Legends
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                txt = t.get_text()
                if txt:
                    for k, v in mapping.items():
                        if k in txt:
                            txt = txt.replace(k, v)
                    t.set_text(txt)

        # 4. Text Annotations / Labels on Axes
        for child in ax.get_children():
            if hasattr(child, 'get_text') and hasattr(child, 'set_text'):
                txt = child.get_text()
                if txt and isinstance(txt, str):
                    for k, v in mapping.items():
                        if k in txt:
                            txt = txt.replace(k, v)
                    child.set_text(txt)


class PopoutPlotDialog(QDialog):
    """
    Independent resizable & fullscreen-capable dialog to view, inspect, and export
    any Matplotlib figure/canvas from the Dynamic Target Suite in high definition,
    supporting real-time English / Report Mode translation on the fly.
    """
    def __init__(self, source_canvas, title="Grafico Ingrandito", parent=None):
        super(PopoutPlotDialog, self).__init__(parent)
        self.setWindowTitle(f"{title} — Dynamic Target Suite")
        self.resize(1150, 780)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top Action Bar
        top_bar = QHBoxLayout()
        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_title.setStyleSheet("font-size: 13px; color: #2d3436;")
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()

        # English Translation Toggle Checkbox
        self.cb_english = QCheckBox("English Mode (Paper / Report)")
        self.cb_english.setToolTip("Switch all chart titles, axis labels, and legends to formal SAE / Academic English for papers and reports.")
        self.cb_english.setStyleSheet("font-weight: bold; color: #2d3436; margin-right: 12px;")
        init_en = parent.is_english_mode() if parent and hasattr(parent, 'is_english_mode') else False
        self.cb_english.setChecked(init_en)
        self.cb_english.stateChanged.connect(self.on_language_toggled)
        top_bar.addWidget(self.cb_english)

        self.btn_fullscreen = QPushButton("Schermo Intero (F11)")
        self.btn_fullscreen.setStyleSheet("font-weight: bold; padding: 5px 12px; background-color: #0984e3; color: white; border-radius: 4px;")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)

        btn_save = QPushButton("Salva Immagine HD")
        btn_save.setStyleSheet("font-weight: bold; padding: 5px 12px; background-color: #27ae60; color: white; border-radius: 4px;")
        btn_save.clicked.connect(self.save_hd_image)
        top_bar.addWidget(btn_save)

        btn_close = QPushButton("Chiudi (Esc)")
        btn_close.setStyleSheet("padding: 5px 12px; border-radius: 4px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)
        top_bar.addWidget(btn_close)
        layout.addLayout(top_bar)

        # Matplotlib Canvas & Toolbar (cloned from source)
        try:
            buf = io.BytesIO()
            pickle.dump(source_canvas.figure, buf)
            buf.seek(0)
            self.fig = pickle.load(buf)
        except Exception:
            self.fig = source_canvas.figure

        if init_en:
            translate_figure_language(self.fig, to_english=True)

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        for b in self.toolbar.findChildren(QtWidgets.QToolButton):
            b.setAutoRaise(False)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._is_fullscreen = False

    def on_language_toggled(self, state):
        is_en = (state == QtCore.Qt.Checked)
        translate_figure_language(self.fig, to_english=is_en)
        self.canvas.draw()

    def toggle_fullscreen(self):
        if not self._is_fullscreen:
            self.showFullScreen()
            self.btn_fullscreen.setText("Esci da Schermo Intero (F11)")
            self._is_fullscreen = True
        else:
            self.showNormal()
            self.btn_fullscreen.setText("Schermo Intero (F11)")
            self._is_fullscreen = False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == QtCore.Qt.Key_Escape:
            if self._is_fullscreen:
                self.toggle_fullscreen()
            else:
                self.accept()
        else:
            super().keyPressEvent(event)

    def save_hd_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva Grafico ad Alta Risoluzione", "chart_hd_export.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)", options=options
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Salvataggio", f"Grafico esportato con successo in:\n{file_path}")


class ConsolePopoutDialog(QDialog):
    """Finestra dedicata e ridimensionabile per la console di log del solutore MDO."""
    def __init__(self, source_text_edit, parent=None):
        super(ConsolePopoutDialog, self).__init__(parent)
        self.setWindowTitle("Console Output Solutore MDO — Dynamic Target Suite")
        self.resize(980, 680)
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        
        # Header bar
        head_layout = QHBoxLayout()
        lbl_title = QLabel("Registro Iterazioni Solutore (Live Console):")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 11pt; color: #2d3436;")
        head_layout.addWidget(lbl_title)
        head_layout.addStretch()
        
        btn_copy = QPushButton("Copia Tutto")
        btn_copy.setStyleSheet("font-size: 9pt; padding: 4px 10px; font-weight: bold;")
        btn_copy.clicked.connect(self._copy_all)
        head_layout.addWidget(btn_copy)
        
        btn_clear = QPushButton("Pulisci")
        btn_clear.setStyleSheet("font-size: 9pt; padding: 4px 10px;")
        btn_clear.clicked.connect(source_text_edit.clear)
        head_layout.addWidget(btn_clear)
        
        layout.addLayout(head_layout)
        
        # Text edit sharing the exact same document
        self.text_view = QtWidgets.QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setDocument(source_text_edit.document())
        self.text_view.setStyleSheet("""
            QTextEdit {
                background-color: #1e272e;
                color: #d2dae2;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                line-height: 1.3;
                border: 1px solid #485460;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.text_view, stretch=1)
        
        # Bottom bar with Close button
        bot_layout = QHBoxLayout()
        self.lbl_lines = QLabel()
        self._update_status()
        bot_layout.addWidget(self.lbl_lines)
        bot_layout.addStretch()
        
        btn_close = QPushButton("Chiudi Finestra")
        btn_close.setStyleSheet("font-weight: bold; padding: 6px 16px; background-color: #2d3436; color: white; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        bot_layout.addWidget(btn_close)
        layout.addLayout(bot_layout)
        
        source_text_edit.textChanged.connect(self._update_status)
        
    def _update_status(self):
        doc = self.text_view.document()
        block_count = doc.blockCount() if doc else 0
        self.lbl_lines.setText(f"Righe di Log: {block_count} | Sincronizzato con il solutore in tempo reale")
        
    def _copy_all(self):
        self.text_view.selectAll()
        self.text_view.copy()
        cursor = self.text_view.textCursor()
        cursor.clearSelection()
        self.text_view.setTextCursor(cursor)


class CollapsibleGroupBox(QtWidgets.QGroupBox):
    """
    QGroupBox con pulsante di riduzione/espansione [-] / [+] nell'header e
    supporto al doppio click per miniaturizzare la sezione e liberare spazio verticale nell'inspector.
    """
    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super(CollapsibleGroupBox, self).__init__(title, parent)
        self._is_collapsed = collapsed
        
        self._toggle_btn = QtWidgets.QToolButton(self)
        self._toggle_btn.setText("[-]" if not collapsed else "[+]")
        self._toggle_btn.setToolTip("Comprimi / Espandi questa sottosezione per liberare spazio")
        self._toggle_btn.setFixedSize(26, 18)
        self._toggle_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            "font-size: 10px; font-weight: bold; border: 1px solid #b2bec3; border-radius: 2px; "
            "background-color: #dfe6e9; color: #2d3436; padding: 0px;"
        )
        self._toggle_btn.clicked.connect(self.toggle_collapse)
        
        self._content_widget = QtWidgets.QWidget(self)
        self._outer_layout = QtWidgets.QVBoxLayout(self)
        self._outer_layout.setContentsMargins(4, 18, 4, 4)
        self._outer_layout.setSpacing(2)
        self._outer_layout.addWidget(self._content_widget)
        
        if collapsed:
            self._content_widget.hide()

    @property
    def content_widget(self):
        return self._content_widget

    def resizeEvent(self, event):
        super(CollapsibleGroupBox, self).resizeEvent(event)
        self._toggle_btn.move(max(self.width() - self._toggle_btn.width() - 8, 10), 2)

    def mouseDoubleClickEvent(self, event):
        if event.pos().y() < 24:
            self.toggle_collapse()
        else:
            super(CollapsibleGroupBox, self).mouseDoubleClickEvent(event)

    def toggle_collapse(self):
        self._is_collapsed = not self._is_collapsed
        self._content_widget.setVisible(not self._is_collapsed)
        self._toggle_btn.setText("[+]" if self._is_collapsed else "[-]")

    def set_collapsed(self, collapsed: bool):
        self._is_collapsed = collapsed
        self._content_widget.setVisible(not self._is_collapsed)
        self._toggle_btn.setText("[+]" if self._is_collapsed else "[-]")

    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def setContentLayout(self, layout: QtWidgets.QLayout):
        self._content_widget.setLayout(layout)