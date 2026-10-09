"""
vehicle_parameters_inspector_dialog.py
======================================
Finestra Esterna Fluttuante Non-Bloccante:
Ispezione Parametri Veicolo, Telemetria Real-Time & Live-Tuning a Caldo (Hot-Reload).

Caratteristiche:
- ActiveVehicleParametersInspectorDialog: Classe base per ispezione analitica completa a 5 schede
  (19 Curve Adams Car, Geometria e Masse, Sospensioni, Aerodinamica & Powertrain, Pneumatici MF6.1).
- LiveVehicleParametersInspector: Finestra indipendente fluttuante (non-modal floating window, Qt.Window)
  con esecuzione non bloccante (.show()): consente al pilota di continuare a guidare mentre l'ispettore
  e aperto su un secondo monitor o a lato.
- Tab Live-Tuning: Regolazione a caldo di rigidezze molle, barre antirollio, precarichi, altezze da terra,
  smorzatori TTX25 (click LSC/HSC/LSR/HSR), coefficienti aerodinamici (Cz, Cd, Bal), pressioni gomme,
  grip e angoli statici di camber e convergenza.
- Hot-Reload istantaneo e thread-safe su PhysicsThread14DOF.
- Tab Telemetria Real-Time: Streaming a 25 Hz dei canali dinamici attivi (v, ax, ay, yaw rate, rollio,
  beccheggio, corse sospensioni z_i, carichi verticali Fz_i, slip angles e slip ratios).
- Gestione Setup: Salvataggio / Esportazione JSON, Caricamento JSON e Ripristino Setup Nominale.
- Pieno allineamento cromatico al motore di temi dell'applicazione, privo di emoji.
"""

import os
import json
import time
import numpy as np
from typing import Optional, Dict, Any, List

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QApplication, QGroupBox,
    QGridLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QScrollArea,
    QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont

from src.models.params_vehicle import VehicleParams
from src.models.params_tire import TireParams
from src.ui.ui_theme_engine import get_theme_palette, get_stylesheet


class ActiveVehicleParametersInspectorDialog(QDialog):
    """
    Dialogo completo di ispezione parametri veicolo e 19 curve cinematiche Adams Car.
    """
    def __init__(self, vp=None, tp=None,
                 title="Parametri Veicolo & Cinematiche Attive",
                 parent=None, theme_name="dark"):
        super().__init__(parent)
        self.vp = vp if vp is not None else VehicleParams.nominal_fsae_params()
        self.tp = tp
        self.theme_name = theme_name

        self.setWindowTitle(title)
        self.resize(1060, 750)
        self._apply_dialog_styling()
        self._init_base_ui()

    def _apply_dialog_styling(self):
        t = get_theme_palette(self.theme_name)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t['bg_root']};
                color: {t['text_primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {t['border']};
                background-color: {t['bg_card']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {t['tab_inactive_bg']};
                color: {t['text_secondary']};
                padding: 7px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background-color: {t['tab_active_bg']};
                color: {t['text_primary']};
                border-bottom: 2px solid {t['accent_primary']};
            }}
            QGroupBox {{
                background-color: {t['bg_card']};
                border: 1px solid {t['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 11px;
                color: {t['text_primary']};
            }}
            QTableWidget {{
                background-color: {t['bg_card']};
                color: {t['text_primary']};
                gridline-color: {t['border']};
                border: none;
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: {t['bg_input']};
                color: {t['accent_primary']};
                padding: 6px;
                font-weight: bold;
                border: 1px solid {t['border']};
            }}
            QLineEdit, QDoubleSpinBox, QSpinBox {{
                background-color: {t['bg_input']};
                color: {t['text_primary']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QPushButton {{
                background-color: {t['accent_primary']};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
                border: 1px solid {t['border_focus']};
            }}
            QPushButton:hover {{
                background-color: {t['accent_primary_hover']};
            }}
        """)

    def _init_base_ui(self):
        t = get_theme_palette(self.theme_name)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(8)

        # Header Status Banner
        top_card = QFrame()
        top_card.setStyleSheet(f"background-color: {t['bg_card']}; border-radius: 6px; border: 1px solid {t['border']}; padding: 6px;")
        tc_layout = QHBoxLayout(top_card)

        info_layout = QVBoxLayout()
        self.lbl_header_title = QLabel("<b>Parametri Attivi in Memoria (Stato Corrente del Modello)</b>")
        self.lbl_header_title.setStyleSheet(f"font-size: 13px; color: {t['accent_primary']};")
        info_layout.addWidget(self.lbl_header_title)

        rc_f_mm = self.vp.rollcenter_f_coeffs[3] * 1000.0 if hasattr(self.vp, 'rollcenter_f_coeffs') and len(self.vp.rollcenter_f_coeffs) > 3 else 0.0
        rc_r_mm = self.vp.rollcenter_r_coeffs[3] * 1000.0 if hasattr(self.vp, 'rollcenter_r_coeffs') and len(self.vp.rollcenter_r_coeffs) > 3 else 0.0
        self.lbl_sub_info = QLabel(
            f"[OK] 19 Curve Cinematiche Adams Car Attive | "
            f"Centri di Rollio: Funzione di Rollio phi [deg] (h_rc,f0={rc_f_mm:.1f} mm, h_rc,r0={rc_r_mm:.1f} mm) | "
            f"Ackermann: {self.vp.ackermann_pct*100.0:.1f}% nominale"
        )
        self.lbl_sub_info.setStyleSheet(f"font-size: 11px; color: {t['text_secondary']};")
        info_layout.addWidget(self.lbl_sub_info)
        tc_layout.addLayout(info_layout, 1)

        # Search Bar
        search_layout = QVBoxLayout()
        search_layout.addWidget(QLabel("<b>Filtro Rapido:</b>"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cerca parametro, unita o simbolo...")
        self.search_input.textChanged.connect(self._filter_active_table)
        search_layout.addWidget(self.search_input)
        tc_layout.addLayout(search_layout)

        self.main_layout.addWidget(top_card)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(lambda: self._filter_active_table(self.search_input.text()))

        self.tab_kin = self._build_kinematics_table()
        self.tab_geo = self._build_geometry_table()
        self.tab_susp = self._build_suspension_table()
        self.tab_aero_pow = self._build_aero_powertrain_table()
        self.tab_tires = self._build_tires_table()

        self.tabs.addTab(self.tab_kin, "19 Curve Cinematiche (Adams Car)")
        self.tabs.addTab(self.tab_geo, "Assetto, Masse & Geometria")
        self.tabs.addTab(self.tab_susp, "Sospensioni & Rigidezze")
        self.tabs.addTab(self.tab_aero_pow, "Aerodinamica & Powertrain")
        self.tabs.addTab(self.tab_tires, "Pneumatici (Pacejka MF6.1)")

        self.main_layout.addWidget(self.tabs, 1)

        # Bottom Buttons
        bottom_layout = QHBoxLayout()
        
        btn_copy = QPushButton("Copia Tabella Corrente (TSV)")
        btn_copy.setStyleSheet(f"background-color: {t['bg_input']}; color: {t['text_primary']}; border: 1px solid {t['border']};")
        btn_copy.clicked.connect(self._copy_current_table)
        bottom_layout.addWidget(btn_copy)

        btn_export = QPushButton("Esporta Setup Attivo JSON")
        btn_export.setStyleSheet("background-color: #238636; color: white; border: 1px solid #2ea043;")
        btn_export.clicked.connect(self._export_active_json)
        bottom_layout.addWidget(btn_export)

        bottom_layout.addStretch()

        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet("background-color: #da3633; color: white; border: 1px solid #f85149;")
        btn_close.clicked.connect(self.accept)
        bottom_layout.addWidget(btn_close)

        self.main_layout.addLayout(bottom_layout)

    def _create_table(self, headers, rows):
        t = get_theme_palette(self.theme_name)
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if c_idx == 0:
                    item.setFont(QFont("Segoe UI", 9, QFont.Bold))
                    item.setForeground(QColor(t['accent_primary']))
                elif c_idx == 2 or (len(row) > 3 and c_idx == 3):
                    item.setForeground(QColor("#3fb950"))
                table.setItem(r_idx, c_idx, item)

        return table

    def _build_kinematics_table(self):
        vp = self.vp
        headers = [
            "Curva Cinematica",
            "Variabile Indipendente",
            "Valore Statico",
            "Coefficienti Polinomio [a3, a2, a1, a0]",
            "Pendenza / Gain Principale (a1)",
            "Stato"
        ]

        def fmt_poly(p):
            if not p or len(p) != 4:
                return "Non definito"
            return f"[{p[0]:.4e}, {p[1]:.4e}, {p[2]:.4f}, {p[3]:.4f}]"

        def fmt_val(v, unit=""):
            return f"{v:+.3f} {unit}" if v is not None else "N/D"

        rc_f_sval = vp.rollcenter_f_coeffs[3] * 1000.0 if len(vp.rollcenter_f_coeffs) > 3 else 0.0
        rc_r_sval = vp.rollcenter_r_coeffs[3] * 1000.0 if len(vp.rollcenter_r_coeffs) > 3 else 0.0
        rc_lat_f_sval = vp.rollcenter_lat_f_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_lat_f_coeffs') and len(vp.rollcenter_lat_f_coeffs) > 3 else 0.0
        rc_lat_r_sval = vp.rollcenter_lat_r_coeffs[3] * 1000.0 if hasattr(vp, 'rollcenter_lat_r_coeffs') and len(vp.rollcenter_lat_r_coeffs) > 3 else 0.0

        rows = [
            ["1. Campanatura Anteriore gamma_f(z)", "z corsa ruota [m]", fmt_val(vp.camber_f_static, "deg"), fmt_poly(vp.camber_gain_f_coeffs), f"{vp.camber_gain_f_coeffs[2]:+.2f} deg/m", "[OK] Attiva (Adams Car)"],
            ["2. Campanatura Posteriore gamma_r(z)", "z corsa ruota [m]", fmt_val(vp.camber_r_static, "deg"), fmt_poly(vp.camber_gain_r_coeffs), f"{vp.camber_gain_r_coeffs[2]:+.2f} deg/m", "[OK] Attiva (Adams Car)"],
            ["3. Convergenza Anteriore delta_toe,f(z)", "z corsa ruota [m]", fmt_val(vp.toe_f_static, "deg"), fmt_poly(vp.toe_gain_f_coeffs), f"{vp.toe_gain_f_coeffs[2]:+.2f} deg/m", "[OK] Attiva (Adams Car)"],
            ["4. Convergenza Posteriore delta_toe,r(z)", "z corsa ruota [m]", fmt_val(vp.toe_r_static, "deg"), fmt_poly(vp.toe_gain_r_coeffs), f"{vp.toe_gain_r_coeffs[2]:+.2f} deg/m", "[OK] Attiva (Adams Car)"],
            ["5. Altezza Centro Rollio Ant. h_rc,f(phi)", "phi angolo di rollio [deg]", f"{rc_f_sval:.1f} mm", fmt_poly(vp.rollcenter_f_coeffs), f"{vp.rollcenter_f_coeffs[1]:+.6f} m/deg2", "[OK] Attiva (f(phi) Adams Car)"],
            ["6. Altezza Centro Rollio Post. h_rc,r(phi)", "phi angolo di rollio [deg]", f"{rc_r_sval:.1f} mm", fmt_poly(vp.rollcenter_r_coeffs), f"{vp.rollcenter_r_coeffs[1]:+.6f} m/deg2", "[OK] Attiva (f(phi) Adams Car)"],
            ["7. Migrazione Laterale RC Ant. y_rc,f(phi)", "phi angolo di rollio [deg]", f"{rc_lat_f_sval:.1f} mm", fmt_poly(getattr(vp, 'rollcenter_lat_f_coeffs', [0, 0, 0.06453, 0])), f"{getattr(vp, 'rollcenter_lat_f_coeffs', [0, 0, 0.06453, 0])[2]*1000.0:+.1f} mm/deg", "[OK] Attiva (f(phi) Adams Car)"],
            ["8. Migrazione Laterale RC Post. y_rc,r(phi)", "phi angolo di rollio [deg]", f"{rc_lat_r_sval:.1f} mm", fmt_poly(getattr(vp, 'rollcenter_lat_r_coeffs', [0, 0, 0.04213, 0])), f"{getattr(vp, 'rollcenter_lat_r_coeffs', [0, 0, 0.04213, 0])[2]*1000.0:+.1f} mm/deg", "[OK] Attiva (f(phi) Adams Car)"],
            ["9. Geometria Anti-Dive Ant.", "z corsa ruota [m]", f"{vp.anti_dive_coeffs[3]*100.0:.1f} %", fmt_poly(vp.anti_dive_coeffs), f"{vp.anti_dive_coeffs[2]:+.2f} 1/m", "[OK] Attiva (Adams Car)"],
            ["10. Geometria Anti-Squat Post.", "z corsa ruota [m]", f"{vp.anti_squat_coeffs[3]*100.0:.1f} %", fmt_poly(vp.anti_squat_coeffs), f"{vp.anti_squat_coeffs[2]:+.2f} 1/m", "[OK] Attiva (Adams Car)"],
            ["11. Geometria Anti-Lift Post.", "z corsa ruota [m]", f"{vp.anti_lift_coeffs[3]*100.0:.1f} %", fmt_poly(vp.anti_lift_coeffs), f"{vp.anti_lift_coeffs[2]:+.2f} 1/m", "[OK] Attiva (Adams Car)"],
            ["12. Motion Ratio Damper Ant.", "z corsa ruota [m]", f"{vp.motion_ratio_f:.3f}", fmt_poly(vp.damper_motion_ratio_f_coeffs), f"{vp.damper_motion_ratio_f_coeffs[2]:+.3f} 1/m", "[OK] Attiva (Adams Car)"],
            ["13. Motion Ratio Damper Post.", "z corsa ruota [m]", f"{vp.motion_ratio_r:.3f}", fmt_poly(vp.damper_motion_ratio_r_coeffs), f"{vp.damper_motion_ratio_r_coeffs[2]:+.3f} 1/m", "[OK] Attiva (Adams Car)"],
            ["14. Motion Ratio ARB Ant.", "z corsa ruota [m]", f"{vp.motion_ratio_arb_f:.3f}", fmt_poly(vp.arb_motion_ratio_f_coeffs), f"{vp.arb_motion_ratio_f_coeffs[2]:+.3f} 1/m", "[OK] Attiva (Adams Car)"],
            ["15. Motion Ratio ARB Post.", "z corsa ruota [m]", f"{vp.motion_ratio_arb_r:.3f}", fmt_poly(vp.arb_motion_ratio_r_coeffs), f"{vp.arb_motion_ratio_r_coeffs[2]:+.3f} 1/m", "[OK] Attiva (Adams Car)"],
            ["16. Geometria di Ackermann %(delta)", "delta_sw angolo volante [deg]", f"{vp.ackermann_pct * 100.0:.1f} %", fmt_poly(vp.ackermann_coeffs), f"{vp.ackermann_coeffs[1]:+.6f} %/deg2", "[OK] Attiva (Adams Car)"],
            ["17. Braccio a Terra Meccanico t_m(delta)", "delta_sw angolo volante [deg]", f"{getattr(vp, 'mechanical_trail_f', 0.012)*1000.0:.1f} mm", fmt_poly(getattr(vp, 'caster_trail_f_coeffs', [0, 0, 0, 0.012])), f"{getattr(vp, 'caster_trail_f_coeffs', [0, 0, 0.000023, 0.012])[2]*1000.0:+.3f} mm/deg", "[OK] Attiva (Adams Car)"],
            ["18. Raggio di Scrub Ant. r_scrub(delta)", "delta_sw angolo volante [deg]", f"{getattr(vp, 'scrub_radius_f', 0.0368)*1000.0:.1f} mm", fmt_poly(getattr(vp, 'scrub_radius_f_coeffs', [0, 0, 0, 0.0368])), f"{getattr(vp, 'scrub_radius_f_coeffs', [0, 0, -0.000014, 0.0368])[2]*1000.0:+.3f} mm/deg", "[OK] Attiva (Adams Car)"],
            ["19. Delta Camber Steer Ant. delta_gamma(delta)", "delta_sw angolo volante [deg]", "0.000 deg (a delta=0)", fmt_poly(getattr(vp, 'camber_steer_f_coeffs', [0, 0, -0.0063, 0])), f"{getattr(vp, 'camber_steer_f_coeffs', [0, 0, -0.0063, 0])[2]:+.4f} deg/deg", "[OK] Attiva (Adams Car)"],
        ]
        return self._create_table(headers, rows)

    def _build_geometry_table(self):
        vp = self.vp
        headers = ["Parametro", "Simbolo", "Valore Attivo", "Unita", "Descrizione"]
        a_dist = vp.L * (1.0 - vp.wd_f)
        b_dist = vp.L * vp.wd_f

        rows = [
            ["Passo Veicolo", "L", f"{vp.L:.4f}", "m", "Interasse longitudinale ruote anteriori e posteriori"],
            ["Carreggiata Anteriore", "t_f", f"{vp.t_f:.4f}", "m", "Distanza trasversale centri impronta ruote anteriori"],
            ["Carreggiata Posteriore", "t_r", f"{vp.t_r:.4f}", "m", "Distanza trasversale centri impronta ruote posteriori"],
            ["Massa Totale Veicolo + Pilota", "m", f"{vp.m:.1f}", "kg", "Massa complessiva in ordine di marcia"],
            ["Ripartizione Pesi Anteriore", "wd_f", f"{vp.wd_f*100.0:.2f}", "%", "Frazione di peso gravante sull'assale anteriore"],
            ["Distanza CG da Asse Ant.", "a", f"{a_dist:.4f}", "m", "Posizione longitudinale baricentro"],
            ["Distanza CG da Asse Post.", "b", f"{b_dist:.4f}", "m", "Posizione longitudinale baricentro"],
            ["Altezza Baricentro Statica", "h_cg", f"{vp.h_cg:.3f}", "m", "Altezza del baricentro da terra a vuoto"],
            ["Inerzia di Imbardata", "Iz", f"{getattr(vp, 'Iz', 120.0):.2f}", "kg m^2", "Momento d'inerzia asse verticale Z"],
            ["Inerzia di Rollio", "Ix", f"{getattr(vp, 'Ix', 35.0):.2f}", "kg m^2", "Momento d'inerzia asse longitudinale X"],
            ["Inerzia di Beccheggio", "Iy", f"{getattr(vp, 'Iy', 110.0):.2f}", "kg m^2", "Momento d'inerzia asse trasversale Y"],
        ]
        return self._create_table(headers, rows)

    def _build_suspension_table(self):
        vp = self.vp
        headers = ["Parametro Sospensione", "Simbolo", "Valore Attivo", "Unita", "Descrizione"]
        rows = [
            ["Rigidezza Molla Anteriore", "k_spring_f", f"{getattr(vp, 'spring_rate_installed_f_Nmm', 59.5):.1f}", "N/mm", "Taratura molla installata ant."],
            ["Rigidezza Molla Posteriore", "k_spring_r", f"{getattr(vp, 'spring_rate_installed_r_Nmm', 61.0):.1f}", "N/mm", "Taratura molla installata post."],
            ["Rigidezza Antirollio Ant.", "K_phi_f", f"{vp.K_phi_f:.0f}", "Nm/rad", "Rigidezza barra ARB e geometria ant."],
            ["Rigidezza Antirollio Post.", "K_phi_r", f"{vp.K_phi_r:.0f}", "Nm/rad", "Rigidezza barra ARB e geometria post."],
            ["Ripartizione LLTD", "lltd", f"{vp.lltd*100.0:.1f}", "%", "Frazione trasferimento carico lat. all'anteriore"],
            ["Precarico Molla Ant.", "preload_f", f"{getattr(vp, 'preload_f_N', 180.0):.0f}", "N", "Carico precompressione molla ruota ant."],
            ["Precarico Molla Post.", "preload_r", f"{getattr(vp, 'preload_r_N', 220.0):.0f}", "N", "Carico precompressione molla ruota post."],
        ]
        return self._create_table(headers, rows)

    def _build_aero_powertrain_table(self):
        vp = self.vp
        headers = ["Parametro Aero / Power", "Simbolo", "Valore Attivo", "Unita", "Descrizione"]
        s_val = getattr(vp, 'S', getattr(vp, 'A_front', 1.139))
        rho_val = getattr(vp, 'rho', getattr(vp, 'rho_air', 1.225))
        rows = [
            ["Superficie Frontale Veicolo", "S", f"{s_val:.3f}", "m^2", "Area di riferimento aerodinamica"],
            ["Portanza Negativa (Downforce)", "Cz", f"{vp.Cz:.3f}", "-", "Coefficiente di portanza aero globale"],
            ["Resistenza Aerodinamica (Drag)", "Cd", f"{vp.Cd:.3f}", "-", "Coefficiente di penetrazione aerodinamica"],
            ["Bilanciamento Aerodinamico", "aero_balance", f"{vp.aero_balance*100.0:.1f}", "%", "Quota carico aero gravante sull'anteriore"],
            ["Densita Aria Nominale", "rho", f"{rho_val:.3f}", "kg/m^3", "Condizioni atmosferiche standard"],
        ]
        return self._create_table(headers, rows)

    def _build_tires_table(self):
        tp = self.tp
        headers = ["Coefficiente Pneumatico", "Simbolo TIR", "Valore Attivo", "Unita", "Descrizione MF6.1"]
        rows = []
        if tp is not None:
            tir_attrs = [
                ("Raggio Nominale a Vuoto", "UNLOADED_RADIUS", "m", "Raggio geometrico non deformato"),
                ("Rigidezza Verticale Pneumatico", "VERTICAL_STIFFNESS", "N/m", "Costante elastica carcassa pneumatico"),
                ("Pressione Nominale Riferimento", "NOMPRES", "Pa", "Pressione di calibrazione file TIR"),
                ("Carico Verticale Nominale", "FNOMIN", "N", "Carico verticale di riferimento"),
                ("Attrito Laterale di Base (Peak Mu)", "PDY1", "-", "Coefficiente di aderenza laterale puro"),
                ("Dipendenza Carico Attrito Lat.", "PDY2", "1/N", "Declassamento coefficiente di attrito con Fz"),
                ("Rigidezza Angolare di Deriva (Cornering)", "PKY1", "N/rad", "Pendenza iniziale forza laterale Fy vs alpha"),
                ("Attrito Longitudinale di Base (Peak Mu)", "PDX1", "-", "Coefficiente di aderenza longitudinale puro"),
            ]
            for desc, key, unit, comment in tir_attrs:
                val = getattr(tp, key, None)
                if val is not None:
                    rows.append([desc, key, f"{val:.4f}" if isinstance(val, float) else str(val), unit, comment])
        else:
            rows.append(["File Dati Pneumatico TIR", "tp", "Pacejka MF6.1 Nominale", "-", "Modello non-lineare attivo"])

        return self._create_table(headers, rows)

    def _filter_active_table(self, query: str):
        curr_table = self.tabs.currentWidget()
        if not isinstance(curr_table, QTableWidget):
            return
        query = query.strip().lower()
        for row in range(curr_table.rowCount()):
            matched = False
            for col in range(curr_table.columnCount()):
                item = curr_table.item(row, col)
                if item and query in item.text().lower():
                    matched = True
                    break
            curr_table.setRowHidden(row, not matched if query else False)

    def _copy_current_table(self):
        curr_table = self.tabs.currentWidget()
        if not isinstance(curr_table, QTableWidget):
            return
        lines = []
        headers = [curr_table.horizontalHeaderItem(c).text() for c in range(curr_table.columnCount())]
        lines.append("\t".join(headers))
        for r in range(curr_table.rowCount()):
            if not curr_table.isRowHidden(r):
                row_text = [curr_table.item(r, c).text() if curr_table.item(r, c) else "" for c in range(curr_table.columnCount())]
                lines.append("\t".join(row_text))
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))
        QMessageBox.information(self, "Copiato", "Tabella copiata negli appunti con successo (formato TSV)!")

    def _export_active_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salva Parametri Attivi", "active_vehicle_setup.json", "JSON Setup Files (*.json)")
        if path:
            try:
                self.vp.save_json(path)
                QMessageBox.information(self, "Esportazione Riuscita", f"Setup esportato con successo in:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file:\n{e}")


# =============================================================================
# LIVE TUNING & TELEMETRY FLOATING WINDOW
# =============================================================================
class LiveVehicleParametersInspector(ActiveVehicleParametersInspectorDialog):
    """
    Finestra non-modale fluttuante per ispezione live, telemetria in tempo reale
    e taratura dell'assetto a caldo (Hot-Reload) durante la guida.
    """
    def __init__(self, target_or_vp=None, tp=None,
                 title="Live-Tuning Assetto & Ispezione Telemetria Real-Time",
                 parent=None, theme_name="dark", physics_thread=None, vp=None):
        
        # Gestione polimorfica dei parametri di inizializzazione
        if target_or_vp is not None and hasattr(target_or_vp, 'vp'):
            self.physics_thread = target_or_vp
            vp_target = target_or_vp.vp
            tp_target = getattr(target_or_vp, 'tp', tp)
        elif target_or_vp is not None:
            self.physics_thread = physics_thread
            vp_target = target_or_vp
            tp_target = tp
        elif physics_thread is not None:
            self.physics_thread = physics_thread
            vp_target = physics_thread.vp
            tp_target = getattr(physics_thread, 'tp', tp)
        elif vp is not None:
            self.physics_thread = None
            vp_target = vp
            tp_target = tp
        else:
            self.physics_thread = None
            vp_target = VehicleParams.nominal_fsae_params()
            tp_target = tp

        super().__init__(vp=vp_target, tp=tp_target, title=title, parent=parent, theme_name=theme_name)

        # Configura finestra non bloccante indipendente fluttuante
        self.setWindowFlags(Qt.Window)

        # Aggiungi le schede Live Tuning e Telemetria in testa al TabWidget
        self.tab_tuning = self._build_live_tuning_tab()
        self.tab_telemetry = self._build_live_telemetry_tab()
        self.tabs.insertTab(0, self.tab_tuning, "Live Tuning & Assetto a Caldo")
        self.tabs.insertTab(1, self.tab_telemetry, "Telemetria Real-Time")
        self.tabs.setCurrentIndex(0)

        # Timer streaming telemetria a 25 Hz
        self.timer_telemetry = QTimer(self)
        self.timer_telemetry.setInterval(40)
        self.timer_telemetry.timeout.connect(self._update_telemetry_ui)
        self.timer_telemetry.start()

    def set_physics_thread(self, phys_thread):
        """Aggiorna il riferimento al thread fisico attivo."""
        self.physics_thread = phys_thread
        if phys_thread is not None:
            self.vp = phys_thread.vp
            self.tp = getattr(phys_thread, 'tp', self.tp)
            self._sync_inputs_from_vp()

    # =========================================================================
    # LIVE TUNING TAB
    # =========================================================================
    def _build_live_tuning_tab(self) -> QWidget:
        t = get_theme_palette(self.theme_name)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        main_vbox = QVBoxLayout(container)
        main_vbox.setSpacing(10)

        # Action Toolbar
        tb_layout = QHBoxLayout()
        self.btn_apply_hot = QPushButton("Applica Modifiche a Caldo (Hot-Reload)")
        self.btn_apply_hot.setStyleSheet("font-weight: bold; background-color: #1f6feb; color: white; padding: 8px 18px; border-radius: 4px; border: 1px solid #388bfd;")
        self.btn_apply_hot.clicked.connect(self._apply_hot_reload)
        tb_layout.addWidget(self.btn_apply_hot)

        self.chk_auto_apply = QCheckBox("Applica automaticamente a ogni modifica")
        self.chk_auto_apply.setChecked(True)
        tb_layout.addWidget(self.chk_auto_apply)

        btn_default = QPushButton("Ripristina Setup Nominale")
        btn_default.setStyleSheet(f"background-color: {t['bg_input']}; color: {t['accent_warning']}; border: 1px solid {t['border']};")
        btn_default.clicked.connect(self._reset_to_nominal_setup)
        tb_layout.addWidget(btn_default)

        btn_import = QPushButton("Importa Setup JSON...")
        btn_import.setStyleSheet(f"background-color: {t['bg_input']}; color: {t['text_primary']}; border: 1px solid {t['border']};")
        btn_import.clicked.connect(self._import_setup_json)
        tb_layout.addWidget(btn_import)

        tb_layout.addStretch()
        self.lbl_apply_status = QLabel("[OK] Parametri sincronizzati con il motore fisico.")
        self.lbl_apply_status.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")
        tb_layout.addWidget(self.lbl_apply_status)
        main_vbox.addLayout(tb_layout)

        grid_top = QGridLayout()
        grid_top.setSpacing(10)

        # 1. Molle & Barre Antirollio
        gb_susp = QGroupBox("Molle, Barre Antirollio & Assetto")
        susp_layout = QGridLayout(gb_susp)

        susp_layout.addWidget(QLabel("Molla Ant. [N/mm]:"), 0, 0)
        self.spin_k_spring_f = self._create_dsp(10.0, 250.0, getattr(self.vp, 'spring_rate_installed_f_Nmm', 59.5), 1.0)
        susp_layout.addWidget(self.spin_k_spring_f, 0, 1)

        susp_layout.addWidget(QLabel("Molla Post. [N/mm]:"), 0, 2)
        self.spin_k_spring_r = self._create_dsp(10.0, 250.0, getattr(self.vp, 'spring_rate_installed_r_Nmm', 61.0), 1.0)
        susp_layout.addWidget(self.spin_k_spring_r, 0, 3)

        susp_layout.addWidget(QLabel("Barra ARB Ant. [Nm/rad]:"), 1, 0)
        self.spin_k_phi_f = self._create_dsp(0.0, 2500.0, getattr(self.vp, 'K_phi_f', 425.0), 25.0)
        susp_layout.addWidget(self.spin_k_phi_f, 1, 1)

        susp_layout.addWidget(QLabel("Barra ARB Post. [Nm/rad]:"), 1, 2)
        self.spin_k_phi_r = self._create_dsp(0.0, 2500.0, getattr(self.vp, 'K_phi_r', 375.0), 25.0)
        susp_layout.addWidget(self.spin_k_phi_r, 1, 3)

        susp_layout.addWidget(QLabel("Ripartizione LLTD [%]:"), 2, 0)
        self.spin_lltd = self._create_dsp(10.0, 90.0, getattr(self.vp, 'lltd', 0.53) * 100.0, 1.0)
        susp_layout.addWidget(self.spin_lltd, 2, 1)

        susp_layout.addWidget(QLabel("Precarico Ant. [N]:"), 3, 0)
        self.spin_preload_f = self._create_dsp(0.0, 2000.0, getattr(self.vp, 'preload_f_N', 180.0), 20.0)
        susp_layout.addWidget(self.spin_preload_f, 3, 1)

        susp_layout.addWidget(QLabel("Precarico Post. [N]:"), 3, 2)
        self.spin_preload_r = self._create_dsp(0.0, 2000.0, getattr(self.vp, 'preload_r_N', 220.0), 20.0)
        susp_layout.addWidget(self.spin_preload_r, 3, 3)

        susp_layout.addWidget(QLabel("Ride Height Ant. [mm]:"), 4, 0)
        rh_f_mm = getattr(self.vp, 'ride_height_f', 0.030) * 1000.0
        self.spin_rh_f = self._create_dsp(15.0, 80.0, rh_f_mm, 1.0)
        susp_layout.addWidget(self.spin_rh_f, 4, 1)

        susp_layout.addWidget(QLabel("Ride Height Post. [mm]:"), 4, 2)
        rh_r_mm = getattr(self.vp, 'ride_height_r', 0.032) * 1000.0
        self.spin_rh_r = self._create_dsp(15.0, 80.0, rh_r_mm, 1.0)
        susp_layout.addWidget(self.spin_rh_r, 4, 3)
        grid_top.addWidget(gb_susp, 0, 0)

        # 2. Ammortizzatori TTX25
        gb_damp = QGroupBox("Ammortizzatori Öhlins TTX25 MkII (Click & Giri)")
        damp_layout = QGridLayout(gb_damp)

        damp_layout.addWidget(QLabel("LSC Ant. [click]:"), 0, 0)
        self.spin_lsc_f = self._create_sp(0, 40, int(getattr(self.vp, 'damper_lsc_clicks_f', 12)))
        damp_layout.addWidget(self.spin_lsc_f, 0, 1)

        damp_layout.addWidget(QLabel("LSC Post. [click]:"), 0, 2)
        self.spin_lsc_r = self._create_sp(0, 40, int(getattr(self.vp, 'damper_lsc_clicks_r', 12)))
        damp_layout.addWidget(self.spin_lsc_r, 0, 3)

        damp_layout.addWidget(QLabel("HSC Ant. [giri]:"), 1, 0)
        self.spin_hsc_f = self._create_dsp(0.0, 5.0, getattr(self.vp, 'damper_hsc_turns_f', 2.0), 0.25)
        damp_layout.addWidget(self.spin_hsc_f, 1, 1)

        damp_layout.addWidget(QLabel("HSC Post. [giri]:"), 1, 2)
        self.spin_hsc_r = self._create_dsp(0.0, 5.0, getattr(self.vp, 'damper_hsc_turns_r', 2.0), 0.25)
        damp_layout.addWidget(self.spin_hsc_r, 1, 3)

        damp_layout.addWidget(QLabel("LSR Ant. [click]:"), 2, 0)
        self.spin_lsr_f = self._create_sp(0, 40, int(getattr(self.vp, 'damper_lsr_clicks_f', 14)))
        damp_layout.addWidget(self.spin_lsr_f, 2, 1)

        damp_layout.addWidget(QLabel("LSR Post. [click]:"), 2, 2)
        self.spin_lsr_r = self._create_sp(0, 40, int(getattr(self.vp, 'damper_lsr_clicks_r', 14)))
        damp_layout.addWidget(self.spin_lsr_r, 2, 3)

        damp_layout.addWidget(QLabel("HSR Ant. [giri]:"), 3, 0)
        self.spin_hsr_f = self._create_dsp(0.0, 5.0, getattr(self.vp, 'damper_hsr_turns_f', 2.0), 0.25)
        damp_layout.addWidget(self.spin_hsr_f, 3, 1)

        damp_layout.addWidget(QLabel("HSR Post. [giri]:"), 3, 2)
        self.spin_hsr_r = self._create_dsp(0.0, 5.0, getattr(self.vp, 'damper_hsr_turns_r', 2.0), 0.25)
        damp_layout.addWidget(self.spin_hsr_r, 3, 3)
        grid_top.addWidget(gb_damp, 0, 1)

        main_vbox.addLayout(grid_top)

        grid_bot = QGridLayout()
        grid_bot.setSpacing(10)

        # 3. Aerodinamica
        gb_aero = QGroupBox("Aerodinamica & Bilanciamento")
        aero_layout = QGridLayout(gb_aero)

        aero_layout.addWidget(QLabel("Portanza Cz (Downforce):"), 0, 0)
        self.spin_cz = self._create_dsp(0.0, 6.0, getattr(self.vp, 'Cz', 2.8), 0.05)
        aero_layout.addWidget(self.spin_cz, 0, 1)

        aero_layout.addWidget(QLabel("Resistenza Cd (Drag):"), 1, 0)
        self.spin_cd = self._create_dsp(0.2, 3.0, getattr(self.vp, 'Cd', 1.15), 0.02)
        aero_layout.addWidget(self.spin_cd, 1, 1)

        aero_layout.addWidget(QLabel("Bilanciamento Aero Ant [%]:"), 2, 0)
        self.spin_aero_bal = self._create_dsp(20.0, 80.0, getattr(self.vp, 'aero_balance', 0.48) * 100.0, 0.5)
        aero_layout.addWidget(self.spin_aero_bal, 2, 1)
        grid_bot.addWidget(gb_aero, 0, 0)

        # 4. Pneumatici & Angoli Statici
        gb_tires = QGroupBox("Pneumatici & Angoli Statici")
        tire_layout = QGridLayout(gb_tires)

        tire_layout.addWidget(QLabel("Camber Statico Ant. [deg]:"), 0, 0)
        self.spin_camber_f = self._create_dsp(-6.0, 2.0, getattr(self.vp, 'camber_f_static', -2.5), 0.1)
        tire_layout.addWidget(self.spin_camber_f, 0, 1)

        tire_layout.addWidget(QLabel("Camber Statico Post. [deg]:"), 0, 2)
        self.spin_camber_r = self._create_dsp(-6.0, 2.0, getattr(self.vp, 'camber_r_static', -1.5), 0.1)
        tire_layout.addWidget(self.spin_camber_r, 0, 3)

        tire_layout.addWidget(QLabel("Toe Statico Ant. [deg]:"), 1, 0)
        self.spin_toe_f = self._create_dsp(-2.0, 2.0, getattr(self.vp, 'toe_f_static', 0.0), 0.05)
        tire_layout.addWidget(self.spin_toe_f, 1, 1)

        tire_layout.addWidget(QLabel("Toe Statico Post. [deg]:"), 1, 2)
        self.spin_toe_r = self._create_dsp(-2.0, 2.0, getattr(self.vp, 'toe_r_static', 0.2), 0.05)
        tire_layout.addWidget(self.spin_toe_r, 1, 3)

        tire_layout.addWidget(QLabel("Pressione Gonf. Ant. [bar]:"), 2, 0)
        self.spin_press_f = self._create_dsp(0.5, 2.5, getattr(self.vp, 'tire_pressure_f_bar', 0.82), 0.02)
        tire_layout.addWidget(self.spin_press_f, 2, 1)

        tire_layout.addWidget(QLabel("Pressione Gonf. Post. [bar]:"), 2, 2)
        self.spin_press_r = self._create_dsp(0.5, 2.5, getattr(self.vp, 'tire_pressure_r_bar', 0.82), 0.02)
        tire_layout.addWidget(self.spin_press_r, 2, 3)

        tire_layout.addWidget(QLabel("Moltiplicatore Grip Battistrada:"), 3, 0)
        self.spin_grip = self._create_dsp(0.5, 1.5, getattr(self.physics_thread, 'track_grip_multiplier', 1.0) if self.physics_thread else 1.0, 0.05)
        tire_layout.addWidget(self.spin_grip, 3, 1)

        tire_layout.addWidget(QLabel("Rigidezza Carcassa kz [N/m]:"), 3, 2)
        self.spin_kz = self._create_dsp(50000.0, 250000.0, getattr(self.vp, 'tire_vertical_stiffness_12psi', 120000.0), 5000.0)
        tire_layout.addWidget(self.spin_kz, 3, 3)
        grid_bot.addWidget(gb_tires, 0, 1)

        main_vbox.addLayout(grid_bot)
        main_vbox.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_dsp(self, min_val, max_val, val, step) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(min_val, max_val)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.valueChanged.connect(self._on_input_auto_change)
        return sp

    def _create_sp(self, min_val, max_val, val) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(min_val, max_val)
        sp.setValue(val)
        sp.valueChanged.connect(self._on_input_auto_change)
        return sp

    def _on_input_auto_change(self):
        if hasattr(self, 'chk_auto_apply') and self.chk_auto_apply.isChecked():
            self._apply_hot_reload()

    def _apply_hot_reload(self):
        """Invia i parametri modificati al thread fisico in modo sicuro e senza lag."""
        if self.physics_thread is None:
            if hasattr(self, 'lbl_apply_status'):
                self.lbl_apply_status.setText("[ERR] Nessun motore fisico associato.")
            return

        try:
            k_spring_f = float(self.spin_k_spring_f.value())
            k_spring_r = float(self.spin_k_spring_r.value())
            k_phi_f = float(self.spin_k_phi_f.value())
            k_phi_r = float(self.spin_k_phi_r.value())
            lltd = float(self.spin_lltd.value()) / 100.0
            preload_f = float(self.spin_preload_f.value())
            preload_r = float(self.spin_preload_r.value())
            rh_f = float(self.spin_rh_f.value())
            rh_r = float(self.spin_rh_r.value())

            lsc_f = float(self.spin_lsc_f.value())
            lsc_r = float(self.spin_lsc_r.value())
            hsc_f = float(self.spin_hsc_f.value())
            hsc_r = float(self.spin_hsc_r.value())
            lsr_f = float(self.spin_lsr_f.value())
            lsr_r = float(self.spin_lsr_r.value())
            hsr_f = float(self.spin_hsr_f.value())
            hsr_r = float(self.spin_hsr_r.value())

            cz = float(self.spin_cz.value())
            cd = float(self.spin_cd.value())
            aero_bal = float(self.spin_aero_bal.value()) / 100.0

            camber_f = float(self.spin_camber_f.value())
            camber_r = float(self.spin_camber_r.value())
            toe_f = float(self.spin_toe_f.value())
            toe_r = float(self.spin_toe_r.value())
            press_f = float(self.spin_press_f.value())
            press_r = float(self.spin_press_r.value())
            k_tire_z = float(self.spin_kz.value())
            grip = float(self.spin_grip.value())

            self.physics_thread.update_live_tuning(
                camber_f=camber_f, camber_r=camber_r,
                toe_f=toe_f, toe_r=toe_r,
                preload_f=preload_f, preload_r=preload_r,
                aero_bal=aero_bal, Cz=cz, Cd=cd,
                press_f=press_f, press_r=press_r,
                k_tire_z=k_tire_z,
                k_phi_f=k_phi_f, k_phi_r=k_phi_r, lltd=lltd,
                rh_f=rh_f, rh_r=rh_r,
                k_spring_f=k_spring_f, k_spring_r=k_spring_r,
                lsc_f=lsc_f, hsc_f=hsc_f, lsr_f=lsr_f, hsr_f=hsr_f,
                lsc_r=lsc_r, hsc_r=hsc_r, lsr_r=lsr_r, hsr_r=hsr_r
            )
            if hasattr(self.physics_thread, 'set_track_grip'):
                self.physics_thread.set_track_grip(grip)

            if hasattr(self, 'lbl_apply_status'):
                self.lbl_apply_status.setText(f"[{time.strftime('%H:%M:%S')}] [OK] Hot-Reload completato con successo!")
                self.lbl_apply_status.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")
        except Exception as e:
            if hasattr(self, 'lbl_apply_status'):
                self.lbl_apply_status.setText(f"[ERR] Errore Hot-Reload: {e}")
                self.lbl_apply_status.setStyleSheet("color: #f85149; font-size: 11px; font-weight: bold;")

    # =========================================================================
    # LIVE TELEMETRY TAB
    # =========================================================================
    def _build_live_telemetry_tab(self) -> QWidget:
        t = get_theme_palette(self.theme_name)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(8)

        def make_card(title, unit):
            card = QFrame()
            card.setStyleSheet(f"background-color: {t['bg_card']}; border: 1px solid {t['border']}; border-radius: 5px; padding: 6px;")
            l = QVBoxLayout(card)
            l.setContentsMargins(4, 4, 4, 4)
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet(f"font-size: 10px; color: {t['text_secondary']};")
            lbl_v = QLabel("0.0")
            lbl_v.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {t['accent_primary']};")
            lbl_u = QLabel(unit)
            lbl_u.setStyleSheet(f"font-size: 9px; color: {t['text_muted']};")
            l.addWidget(lbl_t)
            l.addWidget(lbl_v)
            l.addWidget(lbl_u)
            return card, lbl_v

        self.telemetry_labels = {}
        channels = [
            ('v_kmh', "Velocita", "km/h", 0, 0),
            ('v_mps', "Velocita", "m/s", 0, 1),
            ('ax_g', "Accelerazione Long. (ax)", "g", 0, 2),
            ('ay_g', "Accelerazione Lat. (ay)", "g", 0, 3),
            ('yaw_rate', "Velocita Imbardata (r)", "deg/s", 1, 0),
            ('roll_deg', "Angolo Rollio (phi)", "deg", 1, 1),
            ('pitch_deg', "Angolo Beccheggio (theta)", "deg", 1, 2),
            ('steer_sw', "Angolo Volante", "deg", 1, 3),
            ('z_fl', "Corsa Sosp. FL", "mm", 2, 0),
            ('z_fr', "Corsa Sosp. FR", "mm", 2, 1),
            ('z_rl', "Corsa Sosp. RL", "mm", 2, 2),
            ('z_rr', "Corsa Sosp. RR", "mm", 2, 3),
            ('fz_fl', "Carico Verticale Fz FL", "N", 3, 0),
            ('fz_fr', "Carico Verticale Fz FR", "N", 3, 1),
            ('fz_rl', "Carico Verticale Fz RL", "N", 3, 2),
            ('fz_rr', "Carico Verticale Fz RR", "N", 3, 3),
            ('alpha_fl', "Slip Angle alpha FL", "deg", 4, 0),
            ('alpha_fr', "Slip Angle alpha FR", "deg", 4, 1),
            ('alpha_rl', "Slip Angle alpha RL", "deg", 4, 2),
            ('alpha_rr', "Slip Angle alpha RR", "deg", 4, 3),
        ]

        for key, name, unit, r, c in channels:
            card, lbl = make_card(name, unit)
            self.telemetry_labels[key] = lbl
            grid.addWidget(card, r, c)

        layout.addLayout(grid)
        layout.addStretch()
        return w

    def _update_telemetry_ui(self):
        """Aggiorna le etichette di telemetria live leggendo lo stato fisico istantaneo."""
        if not self.isVisible() or self.physics_thread is None:
            return

        state = getattr(self.physics_thread, 'car_state', None)
        if not state:
            return

        vx = float(state.get('vx', 0.0))
        ax = float(state.get('ax', 0.0)) / 9.81
        ay = float(state.get('ay', 0.0)) / 9.81
        r_deg = np.rad2deg(float(state.get('r', 0.0)))
        phi = float(state.get('phi_deg', 0.0))
        theta = float(state.get('theta_deg', 0.0))
        steer = float(state.get('steer_deg', 0.0))

        z_fl = float(state.get('z_fl', 0.0)) * 1000.0
        z_fr = float(state.get('z_fr', 0.0)) * 1000.0
        z_rl = float(state.get('z_rl', 0.0)) * 1000.0
        z_rr = float(state.get('z_rr', 0.0)) * 1000.0

        fz_fl = float(state.get('Fz_fl', 0.0))
        fz_fr = float(state.get('Fz_fr', 0.0))
        fz_rl = float(state.get('Fz_rl', 0.0))
        fz_rr = float(state.get('Fz_rr', 0.0))

        alpha_fl = float(state.get('alpha_fl_deg', 0.0))
        alpha_fr = float(state.get('alpha_fr_deg', 0.0))
        alpha_rl = float(state.get('alpha_rl_deg', 0.0))
        alpha_rr = float(state.get('alpha_rr_deg', 0.0))

        val_map = {
            'v_kmh': f"{vx * 3.6:.1f}",
            'v_mps': f"{vx:.2f}",
            'ax_g': f"{ax:+.2f}",
            'ay_g': f"{ay:+.2f}",
            'yaw_rate': f"{r_deg:+.1f}",
            'roll_deg': f"{phi:+.2f}",
            'pitch_deg': f"{theta:+.2f}",
            'steer_sw': f"{steer:+.1f}",
            'z_fl': f"{z_fl:+.1f}",
            'z_fr': f"{z_fr:+.1f}",
            'z_rl': f"{z_rl:+.1f}",
            'z_rr': f"{z_rr:+.1f}",
            'fz_fl': f"{fz_fl:.0f}",
            'fz_fr': f"{fz_fr:.0f}",
            'fz_rl': f"{fz_rl:.0f}",
            'fz_rr': f"{fz_rr:.0f}",
            'alpha_fl': f"{alpha_fl:+.2f}",
            'alpha_fr': f"{alpha_fr:+.2f}",
            'alpha_rl': f"{alpha_rl:+.2f}",
            'alpha_rr': f"{alpha_rr:+.2f}",
        }

        for k, text in val_map.items():
            lbl = self.telemetry_labels.get(k)
            if lbl:
                lbl.setText(text)

    def _sync_inputs_from_vp(self):
        vp = self.vp
        self.spin_k_spring_f.setValue(getattr(vp, 'spring_rate_installed_f_Nmm', 59.5))
        self.spin_k_spring_r.setValue(getattr(vp, 'spring_rate_installed_r_Nmm', 61.0))
        self.spin_k_phi_f.setValue(getattr(vp, 'K_phi_f', 425.0))
        self.spin_k_phi_r.setValue(getattr(vp, 'K_phi_r', 375.0))
        self.spin_lltd.setValue(getattr(vp, 'lltd', 0.53) * 100.0)
        self.spin_preload_f.setValue(getattr(vp, 'preload_f_N', 180.0))
        self.spin_preload_r.setValue(getattr(vp, 'preload_r_N', 220.0))
        self.spin_rh_f.setValue(getattr(vp, 'ride_height_f', 0.030) * 1000.0)
        self.spin_rh_r.setValue(getattr(vp, 'ride_height_r', 0.032) * 1000.0)

        self.spin_cz.setValue(getattr(vp, 'Cz', 2.8))
        self.spin_cd.setValue(getattr(vp, 'Cd', 1.15))
        self.spin_aero_bal.setValue(getattr(vp, 'aero_balance', 0.48) * 100.0)

        self.spin_camber_f.setValue(getattr(vp, 'camber_f_static', -2.5))
        self.spin_camber_r.setValue(getattr(vp, 'camber_r_static', -1.5))
        self.spin_toe_f.setValue(getattr(vp, 'toe_f_static', 0.0))
        self.spin_toe_r.setValue(getattr(vp, 'toe_r_static', 0.2))
        self.spin_press_f.setValue(getattr(vp, 'tire_pressure_f_bar', 0.82))
        self.spin_press_r.setValue(getattr(vp, 'tire_pressure_r_bar', 0.82))

    def _reset_to_nominal_setup(self):
        vp_nom = VehicleParams.nominal_fsae_params()
        self.vp = vp_nom
        if self.physics_thread is not None:
            self.physics_thread.set_vehicle_params(vp_nom)
        self._sync_inputs_from_vp()
        if hasattr(self, 'lbl_apply_status'):
            self.lbl_apply_status.setText("[OK] Setup ripristinato ai valori nominali FSAE.")

    def _import_setup_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importa Setup Veicolo JSON", "", "File JSON (*.json);;Tutti i File (*.*)")
        if path:
            try:
                self.vp.load_json(path)
                if self.physics_thread is not None:
                    self.physics_thread.set_vehicle_params(self.vp)
                self._sync_inputs_from_vp()
                QMessageBox.information(self, "Importa Setup", f"Setup caricato e applicato con successo:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Importazione", f"Impossibile caricare il setup:\n{e}")
