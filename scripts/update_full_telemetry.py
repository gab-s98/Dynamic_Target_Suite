import re

NEW_FULL_TELEMETRY = """# =============================================================================
# 5.4 FULL TELEMETRY DATA VIEWER (SUSPENSION, CAMBER, FORCES, G-G, WEAR)
# =============================================================================
class FullTelemetryWindow(QDialog):
    \"\"\"
    Comprehensive interactive engineering telemetry analysis window.
    Features:
    - Distance-based (s [m]) and Time-based (t [s]) horizontal axis.
    - Lap-by-Lap Comparison (Giro vs Giro Overlay, Best Lap, Last Lap).
    - Multi-plot grid & focused single plot views with smooth zoom/pan.
    - Fullscreen (F11) and pop-out analysis.
    \"\"\"
    AVAILABLE_PLOTS = {
        'suspension_travel': {'name': "1. Escursioni Sospensioni z [mm]",     'title': "1. Escursioni Sospensioni z [mm]"},
        'camber':            {'name': "2. Campanatura Dinamica gamma [deg]",    'title': "2. Campanatura Dinamica gamma [deg]"},
        'toe':               {'name': "3. Convergenza Dinamica Toe [deg]",     'title': "3. Convergenza Dinamica Toe [deg]"},
        'vertical_loads':    {'name': "4. Carichi Verticali Fz [N] & Jacking", 'title': "4. Carichi Verticali Fz [N] & Jacking"},
        'lateral_forces':    {'name': "5. Forze Laterali Gomme Fy [N]",        'title': "5. Forze Laterali Gomme Fy [N]"},
        'slip_angles':       {'name': "6. Angoli Deriva Pneumatici alpha [deg]", 'title': "6. Angoli di Deriva Pneumatici alpha [deg]"},
        'slip_ratios':       {'name': "7. Rapporti Slittamento kappa [%]",     'title': "7. Slittamento Longitudinale kappa [%]"},
        'grip_util':         {'name': "8. Utilizzo Aderenza Pneumatici [%]",   'title': "8. Grip Utilization 4 Ruote [%]"},
        'gg_diagram':        {'name': "9. Diagramma G-G (Ay vs Ax) [1:1 Equal]", 'title': "9. Diagramma G-G (Ay vs Ax)"},
        'chassis_attitude':  {'name': "10. Assetto Cassa (Rollio, Beccheggio)", 'title': "10. Assetto Cassa (Roll, Pitch, Yaw)"},
        'chassis_velocities':{'name': "11. Velocita Chassis (vx, vy, Deriva beta)", 'title': "11. Velocita Chassis & Deriva beta"},
        'driver_inputs':     {'name': "12. Input Pilota (Sterzo, Gas, Freno)", 'title': "12. Input Pilota (Sterzo, Gas, Freno)"},
        'haptics':           {'name': "13. Feedback Aptico & Rumble [%]",      'title': "13. Feedback Aptico Grilletti & Rumble"},
        'roll_centers':      {'name': "14. Centri di Rollio Dinamici h_rc [mm]", 'title': "14. Centri di Rollio Dinamici h_rc [mm]"},
        'tire_temps_bulk':   {'name': "15. Temp. Gomme Bulk (4 Ruote) [°C]",   'title': "15. Temperature Gomme Bulk (FL, FR, RL, RR) [°C]"},
        'tire_temps_detailed': {'name': "16. Strati Termici 3-Zone (Surf, Tread, Carc) [°C]", 'title': "16. Strati Termici Gomme (3 Strati) [°C]"},
        'tire_wear_grip':    {'name': "17. Usura Pneumatici & Moltiplicatori Grip [%]", 'title': "17. Usura Gomme [%] & Grip Multiplier"}
    }
    
    PRESETS = {
        "1. Setup & Dinamica Veicolo": ['suspension_travel', 'camber', 'vertical_loads', 'gg_diagram', 'haptics', 'driver_inputs'],
        "2. Pneumatici & Aderenza 4 Ruote": ['lateral_forces', 'slip_angles', 'vertical_loads', 'grip_util', 'slip_ratios', 'gg_diagram'],
        "3. Geometria, Assetto & Cassa": ['suspension_travel', 'camber', 'toe', 'roll_centers', 'chassis_attitude', 'chassis_velocities'],
        "4. Guida & Performance Pilota": ['driver_inputs', 'gg_diagram', 'chassis_velocities', 'haptics', 'grip_util', 'chassis_attitude'],
        "5. Termica & Degrado Pneumatici": ['tire_temps_bulk', 'tire_temps_detailed', 'tire_wear_grip', 'grip_util', 'lateral_forces', 'gg_diagram']
    }

    LAP_COLORS = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c', '#fdcb6e']

    def __init__(self, history_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analisi Telemetria 14-DOF Completa (Distanza, Tempo & Giro vs Giro)")
        self.resize(1260, 820)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet("background-color: #1e272e; color: #ecf0f1;")
        
        self.history_data = history_data or []
        self.df = pd.DataFrame(self.history_data) if len(self.history_data) > 0 else pd.DataFrame()
        self.t = self.df.get('t', np.arange(len(self.df))) if not self.df.empty else np.array([])
        
        self.current_layout_keys = list(self.PRESETS["1. Setup & Dinamica Veicolo"])
        self.custom_layout_keys = list(self.current_layout_keys)
        
        # Extract laps and stats
        self.laps, self.best_lap_idx, self.lap_stats = self._extract_laps()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # --- Top Action Bar ---
        top_bar = QHBoxLayout()
        
        top_bar.addWidget(QLabel("<b>Layout:</b>"))
        self.combo_preset = QComboBox()
        for p_name in self.PRESETS.keys():
            self.combo_preset.addItem(p_name)
        self.combo_preset.addItem("Layout Personalizzato...")
        self.combo_preset.setStyleSheet("font-size: 11px; padding: 3px 8px; background-color: #2c3e50; color: white;")
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        top_bar.addWidget(self.combo_preset)
        
        self.btn_config_layout = QPushButton("Configura Layout...")
        self.btn_config_layout.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #6c5ce7; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_config_layout.clicked.connect(self._open_custom_layout_dialog)
        top_bar.addWidget(self.btn_config_layout)
        
        top_bar.addSpacing(15)
        top_bar.addWidget(QLabel("<b>Focus:</b>"))
        self.combo_focus = QComboBox()
        self.combo_focus.addItem("Tutti i Grafici (Griglia)")
        for k, info in self.AVAILABLE_PLOTS.items():
            self.combo_focus.addItem(info['name'], k)
        self.combo_focus.setStyleSheet("font-size: 11px; padding: 3px 8px; background-color: #2c3e50; color: white;")
        self.combo_focus.currentIndexChanged.connect(self._render_current_view)
        top_bar.addWidget(self.combo_focus)
        
        self.btn_popout_selected = QPushButton("Pop-out Ingrandito")
        self.btn_popout_selected.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #0984e3; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_popout_selected.clicked.connect(self._popout_current_plot)
        top_bar.addWidget(self.btn_popout_selected)
        
        top_bar.addSpacing(10)
        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        btn_reset_zoom.clicked.connect(self._reset_zoom)
        top_bar.addWidget(btn_reset_zoom)

        self.btn_z1 = QPushButton("Finestra Breve")
        self.btn_z1.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        self.btn_z1.clicked.connect(lambda: self._set_window_span(1))
        top_bar.addWidget(self.btn_z1)

        self.btn_z2 = QPushButton("Finestra Media")
        self.btn_z2.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #34495e; color: white; border-radius: 3px;")
        self.btn_z2.clicked.connect(lambda: self._set_window_span(2))
        top_bar.addWidget(self.btn_z2)
        
        top_bar.addStretch()
        
        self.btn_fullscreen = QPushButton("Fullscreen (F11)")
        self.btn_fullscreen.setStyleSheet("font-size: 11px; font-weight: bold; background-color: #00b894; color: white; padding: 4px 10px; border-radius: 3px;")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        top_bar.addWidget(self.btn_fullscreen)
        
        btn_close = QPushButton("Chiudi")
        btn_close.setStyleSheet("font-size: 11px; padding: 4px 10px; background-color: #7f8c8d; color: white; border-radius: 3px;")
        btn_close.clicked.connect(self.accept)
        top_bar.addWidget(btn_close)
        
        main_layout.addLayout(top_bar)

        # --- Secondary Action Bar: Asse X & Analisi Giri ---
        analysis_bar = QHBoxLayout()
        analysis_bar.addWidget(QLabel("<b>Asse X:</b>"))
        self.combo_x_axis = QComboBox()
        self.combo_x_axis.addItem("Distanza Pista s [m]", "s_lap")
        self.combo_x_axis.addItem("Tempo t [s]", "t")
        self.combo_x_axis.addItem("Distanza Totale [m]", "dist_total")
        self.combo_x_axis.setStyleSheet("font-size: 11px; font-weight: bold; padding: 3px 8px; background-color: #2c3e50; color: #00cec9;")
        self.combo_x_axis.currentIndexChanged.connect(self._on_analysis_mode_changed)
        analysis_bar.addWidget(self.combo_x_axis)

        analysis_bar.addSpacing(15)
        analysis_bar.addWidget(QLabel("<b>Modalità Giri:</b>"))
        self.combo_lap_filter = QComboBox()
        self.combo_lap_filter.addItem("Tutti i Dati (Sequenziale)", "all")
        self.combo_lap_filter.addItem("Confronto Giro vs Giro (Overlay)", "overlay")
        self.combo_lap_filter.addItem("Solo Best Lap", "best")
        self.combo_lap_filter.addItem("Solo Ultimo Giro", "last")
        self.combo_lap_filter.setStyleSheet("font-size: 11px; font-weight: bold; padding: 3px 8px; background-color: #2c3e50; color: #f1c40f;")
        self.combo_lap_filter.currentIndexChanged.connect(self._on_analysis_mode_changed)
        analysis_bar.addWidget(self.combo_lap_filter)

        self.lbl_lap_stats = QLabel("")
        self.lbl_lap_stats.setStyleSheet("font-size: 11px; color: #f39c12; font-style: italic; padding-left: 10px;")
        analysis_bar.addWidget(self.lbl_lap_stats)

        analysis_bar.addStretch()
        main_layout.addLayout(analysis_bar)
        
        # --- Matplotlib Canvas & Toolbar ---
        self.fig = Figure(figsize=(12, 7.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        # Connect interactive mouse wheel zoom
        self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll_zoom)
        
        self._is_fullscreen = False
        self._update_lap_stats_label()
        self._render_current_view()

    def _extract_laps(self):
        \"\"\"Groups telemetry points by lap, returns dictionary {lap_idx: sub_df}, best_lap_idx, and lap_stats.\"\"\"
        if self.df.empty:
            return {}, None, {}
            
        laps = {}
        lap_col = 'lap_idx' if 'lap_idx' in self.df.columns else ('lap_count' if 'lap_count' in self.df.columns else None)
        if lap_col:
            for l_id, g in self.df.groupby(lap_col):
                if len(g) >= 5:
                    laps[int(l_id)] = g.copy()
        else:
            laps[0] = self.df.copy()
            
        if not laps:
            laps[0] = self.df.copy()

        lap_stats = {}
        best_id = None
        best_time = 1e9
        
        for l_id, g in sorted(laps.items()):
            dt = float(g['t'].iloc[-1] - g['t'].iloc[0]) if 't' in g.columns and len(g) > 1 else 0.0
            dist = float(g['dist_total'].iloc[-1] - g['dist_total'].iloc[0]) if 'dist_total' in g.columns and len(g) > 1 else 0.0
            v_avg = float(g['vx_kmh'].mean()) if 'vx_kmh' in g.columns else 0.0
            rec_lap_time = float(g['lap_time_last'].iloc[-1]) if 'lap_time_last' in g.columns and g['lap_time_last'].iloc[-1] > 0 else dt
            lap_stats[l_id] = {'dt': dt, 'dist': dist, 'v_avg': v_avg, 'rec_time': rec_lap_time}
            
            if (dist > 50.0 or dt > 5.0) and dt < best_time:
                best_time = dt
                best_id = l_id
                
        if best_id is None:
            best_id = list(laps.keys())[0]
            
        return laps, best_id, lap_stats

    def _update_lap_stats_label(self):
        n_laps = len(self.laps)
        if n_laps <= 1:
            self.lbl_lap_stats.setText("1 giro registrato (completa ulteriori giri sul circuito per il confronto overlay)")
        else:
            best_info = self.lap_stats.get(self.best_lap_idx, {})
            best_t = best_info.get('dt', 0.0)
            self.lbl_lap_stats.setText(f"Giri registrati: {n_laps} | Best Lap: Giro {self.best_lap_idx + 1} ({best_t:.2f} s)")

    def _on_analysis_mode_changed(self):
        self._update_lap_stats_label()
        self._render_current_view()

    def _on_mouse_scroll_zoom(self, event):
        \"\"\"Smoothly zooms the X-axis (and Y-axis for G-G) around mouse cursor.\"\"\"
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 0.8 if event.button == 'up' else 1.25
        
        cur_xlim = ax.get_xlim()
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_x = (event.xdata - cur_xlim[0]) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
        ax.set_xlim([event.xdata - new_w * rel_x, event.xdata + new_w * (1.0 - rel_x)])
        
        if getattr(ax, 'name', '') == 'rectilinear' and ax.get_aspect() == 1.0:
            cur_ylim = ax.get_ylim()
            new_h = (cur_ylim[1] - cur_ylim[0]) * scale
            rel_y = (event.ydata - cur_ylim[0]) / max(cur_ylim[1] - cur_ylim[0], 1e-6)
            ax.set_ylim([event.ydata - new_h * rel_y, event.ydata + new_h * (1.0 - rel_y)])
            
        self.canvas.draw_idle()

    def _reset_zoom(self):
        for ax in self.fig.axes:
            ax.autoscale()
            if hasattr(ax, '_is_gg') and ax._is_gg:
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.set_aspect('equal', 'box')
        self.canvas.draw_idle()

    def _set_window_span(self, mode_idx):
        x_mode = self.combo_x_axis.currentData() if hasattr(self, 'combo_x_axis') else 's_lap'
        if x_mode == 's_lap':
            span = 250.0 if mode_idx == 1 else 600.0
        elif x_mode == 'dist_total':
            span = 500.0 if mode_idx == 1 else 1500.0
        else:
            span = 10.0 if mode_idx == 1 else 30.0

        for ax in self.fig.axes:
            if not (hasattr(ax, '_is_gg') and ax._is_gg):
                lines = ax.get_lines()
                if lines:
                    xdata = lines[0].get_xdata()
                    if len(xdata) > 0:
                        max_x = xdata[-1]
                        min_x = max(0.0, max_x - span)
                        ax.set_xlim(min_x, max_x)
        self.canvas.draw_idle()
        
    def _on_preset_changed(self, index):
        p_name = self.combo_preset.currentText()
        if p_name in self.PRESETS:
            self.current_layout_keys = list(self.PRESETS[p_name])
            self.combo_focus.setCurrentIndex(0)
            self._render_current_view()
        else:
            self._open_custom_layout_dialog()

    def _open_custom_layout_dialog(self):
        dlg = CustomTelemetryLayoutDialog(self.custom_layout_keys, self.AVAILABLE_PLOTS, self)
        if dlg.exec_() == QDialog.Accepted:
            self.custom_layout_keys = list(dlg.selected_layout)
            self.current_layout_keys = list(self.custom_layout_keys)
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(self.combo_preset.count() - 1)
            self.combo_preset.blockSignals(False)
            self.combo_focus.setCurrentIndex(0)
            self._render_current_view()

    def toggle_fullscreen(self):
        if not self._is_fullscreen:
            self.showFullScreen()
            self.btn_fullscreen.setText("Esci Fullscreen (F11)")
            self._is_fullscreen = True
        else:
            self.showNormal()
            self.btn_fullscreen.setText("Fullscreen (F11)")
            self._is_fullscreen = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            return
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
                return
            else:
                self.accept()
                return
        super().keyPressEvent(event)

    def _render_current_view(self):
        self.fig.clear()
        if self.df.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Nessun dato di telemetria disponibile.\\nGuida l'auto per raccogliere dati.",
                    ha='center', va='center', fontsize=12, color='white')
            self.canvas.draw()
            return
            
        focus_idx = self.combo_focus.currentIndex()
        if focus_idx == 0:
            keys = self.current_layout_keys
            n_plots = len(keys)
            
            if n_plots == 1:
                rows, cols = 1, 1
            elif n_plots == 2:
                rows, cols = 2, 1
            elif n_plots <= 4:
                rows, cols = 2, 2
            elif n_plots <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 4, 2
                
            for i, k in enumerate(keys):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                self._draw_single_plot(ax, k, is_grid=True)
        else:
            key = self.combo_focus.currentData()
            ax = self.fig.add_subplot(111)
            self._draw_single_plot(ax, key, is_grid=False)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def _popout_current_plot(self):
        focus_idx = self.combo_focus.currentIndex()
        fig_pop = Figure(figsize=(10, 6), dpi=100)
        
        if focus_idx == 0:
            title = f"Griglia Telemetria ({self.combo_preset.currentText()})"
            keys = self.current_layout_keys
            n_plots = len(keys)
            rows = 2 if n_plots <= 6 else 4
            cols = 3 if n_plots == 6 else (2 if n_plots <= 4 else 2)
            if n_plots == 1: rows, cols = 1, 1
            elif n_plots == 2: rows, cols = 2, 1
            
            for i, k in enumerate(keys):
                ax = fig_pop.add_subplot(rows, cols, i + 1)
                self._draw_single_plot(ax, k, is_grid=True, fig=fig_pop)
        else:
            key = self.combo_focus.currentData()
            title = self.AVAILABLE_PLOTS.get(key, {}).get('title', "Grafico Telemetria")
            ax = fig_pop.add_subplot(111)
            self._draw_single_plot(ax, key, is_grid=False, fig=fig_pop)
            
        fig_pop.tight_layout()
        dlg = TelemetryPopoutDialog(fig_pop, title=title, parent=self)
        dlg.exec_()

    def _draw_single_plot(self, ax, plot_key, is_grid=False, fig=None):
        target_fig = fig if fig is not None else self.fig
        fs_title = 8.5 if is_grid else 11.0
        fs_lbl = 8.0 if is_grid else 10.0
        fs_leg = 7.0 if is_grid else 9.0

        x_mode = self.combo_x_axis.currentData() if hasattr(self, 'combo_x_axis') else 's_lap'
        lap_mode = self.combo_lap_filter.currentData() if hasattr(self, 'combo_lap_filter') else 'all'

        if x_mode == 's_lap':
            x_axis_label = "Distanza Pista s [m]"
        elif x_mode == 'dist_total':
            x_axis_label = "Distanza Totale [m]"
        else:
            x_axis_label = "Tempo [s]"

        # Special Case: G-G Diagram
        if plot_key == 'gg_diagram':
            ax._is_gg = True
            if lap_mode == 'overlay' and len(self.laps) > 1:
                for l_id, lap_df in sorted(self.laps.items()):
                    ay = lap_df.get('ay_g', np.zeros(len(lap_df)))
                    ax_g = lap_df.get('ax_g', np.zeros(len(lap_df)))
                    is_best = (l_id == self.best_lap_idx)
                    c = '#f1c40f' if is_best else self.LAP_COLORS[l_id % len(self.LAP_COLORS)]
                    lbl = f"★ Best Lap (G{l_id+1})" if is_best else f"Giro {l_id+1}"
                    ax.scatter(ay, ax_g, color=c, s=12 if is_grid else 22, alpha=0.75 if is_best else 0.45, label=lbl)
            else:
                if lap_mode == 'best' and self.best_lap_idx in self.laps:
                    sub_df = self.laps[self.best_lap_idx]
                elif lap_mode == 'last' and self.laps:
                    sub_df = self.laps[sorted(self.laps.keys())[-1]]
                else:
                    sub_df = self.df
                    
                ay = sub_df.get('ay_g', np.zeros(len(sub_df)))
                ax_g = sub_df.get('ax_g', np.zeros(len(sub_df)))
                v = sub_df.get('vx_kmh', np.zeros(len(sub_df)))
                if len(ay) > 0:
                    sc = ax.scatter(ay, ax_g, c=v, cmap='turbo', s=12 if is_grid else 20, alpha=0.8, edgecolors='none')
                    cbar = target_fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('v [km/h]', fontsize=fs_lbl)
                    cbar.ax.tick_params(labelsize=fs_leg)
                    
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal', 'box')
            ax.axhline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.axvline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.7)
            ax.set_title("9. Diagramma G-G (Ay vs Ax) [1:1 Equal]", fontsize=fs_title, fontweight='bold')
            ax.set_xlabel("Acc. Lat. Ay [g]", fontsize=fs_lbl)
            ax.set_ylabel("Acc. Long. Ax [g]", fontsize=fs_lbl)
            ax.grid(True, linestyle=':', alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=fs_leg, loc='best')
            return

        # General Signal Dictionary for the remaining 16 plots
        plot_configs = {
            'suspension_travel': {
                'title': "1. Escursioni Sospensioni z [mm]", 'ylabel': "Corsa [mm]",
                'signals': [('z_fl_mm', 'z_FL [mm]', '#0984e3', '-'), ('z_fr_mm', 'z_FR [mm]', '#00cec9', '-'),
                            ('z_rl_mm', 'z_RL [mm]', '#e17055', '-'), ('z_rr_mm', 'z_RR [mm]', '#d63031', '-')],
                'overlay_primary': ('z_fl_mm', 'z_FL')
            },
            'camber': {
                'title': "2. Campanatura Dinamica gamma [deg]", 'ylabel': "Camber [deg]",
                'signals': [('camber_fl_deg', 'Camber FL [deg]', '#0984e3', '-'), ('camber_fr_deg', 'Camber FR [deg]', '#00cec9', '-'),
                            ('camber_rl_deg', 'Camber RL [deg]', '#e17055', '-'), ('camber_rr_deg', 'Camber RR [deg]', '#d63031', '-')],
                'overlay_primary': ('camber_fl_deg', 'Camber FL')
            },
            'toe': {
                'title': "3. Convergenza Dinamica Toe [deg]", 'ylabel': "Toe [deg]",
                'signals': [('toe_fl_deg', 'Toe FL [deg]', '#0984e3', '-'), ('toe_fr_deg', 'Toe FR [deg]', '#00cec9', '-'),
                            ('toe_rl_deg', 'Toe RL [deg]', '#e17055', '-'), ('toe_rr_deg', 'Toe RR [deg]', '#d63031', '-')],
                'overlay_primary': ('toe_fl_deg', 'Toe FL')
            },
            'vertical_loads': {
                'title': "4. Carichi Verticali Fz [N] & Jacking", 'ylabel': "Carico Fz [N]",
                'signals': [('Fz_fl', 'Fz FL [N]', '#0984e3', '-'), ('Fz_fr', 'Fz FR [N]', '#00cec9', '-'),
                            ('Fz_rl', 'Fz RL [N]', '#e17055', '-'), ('Fz_rr', 'Fz RR [N]', '#d63031', '-'),
                            ('dFz_jack_N', 'Jacking dFz [N]', '#f1c40f', '--')],
                'overlay_primary': ('Fz_fl', 'Fz FL')
            },
            'lateral_forces': {
                'title': "5. Forze Laterali Gomme Fy [N]", 'ylabel': "Forza Fy [N]",
                'signals': [('Fy_fl', 'Fy FL [N]', '#0984e3', '-'), ('Fy_fr', 'Fy FR [N]', '#00cec9', '-'),
                            ('Fy_rl', 'Fy RL [N]', '#e17055', '-'), ('Fy_rr', 'Fy RR [N]', '#d63031', '-')],
                'overlay_primary': ('Fy_fl', 'Fy FL')
            },
            'slip_angles': {
                'title': "6. Angoli di Deriva Pneumatici alpha [deg]", 'ylabel': "Deriva alpha [deg]",
                'signals': [('alpha_fl', 'alpha FL [deg]', '#0984e3', '-'), ('alpha_fr', 'alpha FR [deg]', '#00cec9', '-'),
                            ('alpha_rl', 'alpha RL [deg]', '#e17055', '-'), ('alpha_rr', 'alpha RR [deg]', '#d63031', '-')],
                'overlay_primary': ('alpha_fl', 'alpha FL')
            },
            'slip_ratios': {
                'title': "7. Slittamento Longitudinale kappa [%]", 'ylabel': "Slittamento kappa [%]",
                'signals': [('kappa_fl', 'kappa FL [%]', '#0984e3', '-'), ('kappa_fr', 'kappa FR [%]', '#00cec9', '-'),
                            ('kappa_rl', 'kappa RL [%]', '#e17055', '-'), ('kappa_rr', 'kappa RR [%]', '#d63031', '-')],
                'overlay_primary': ('kappa_fl', 'kappa FL')
            },
            'grip_util': {
                'title': "8. Grip Utilization 4 Ruote [%]", 'ylabel': "Utilizzo Aderenza [%]",
                'signals': [('util_fl', 'Util FL [%]', '#0984e3', '-'), ('util_fr', 'Util FR [%]', '#00cec9', '-'),
                            ('util_rl', 'Util RL [%]', '#e17055', '-'), ('util_rr', 'Util RR [%]', '#d63031', '-')],
                'overlay_primary': ('util_fl', 'Util FL')
            },
            'chassis_attitude': {
                'title': "10. Assetto Cassa (Roll, Pitch, Yaw)", 'ylabel': "Angolo [deg] / [deg/s]",
                'signals': [('phi_deg', 'Rollio phi [deg]', '#e74c3c', '-'), ('theta_deg', 'Beccheggio theta [deg]', '#f39c12', '-'),
                            ('r_deg_s', 'Yaw Rate r [deg/s]', '#9b59b6', '--')],
                'overlay_primary': ('phi_deg', 'Rollio phi')
            },
            'chassis_velocities': {
                'title': "11. Velocita Chassis & Deriva beta", 'ylabel': "Velocita [km/h] / beta [deg]",
                'signals': [('vx_kmh', 'vx [km/h]', '#2ecc71', '-'), ('vy_kmh', 'vy [km/h]', '#3498db', '-'),
                            ('beta_deg', 'Deriva beta [deg]', '#e67e22', '--')],
                'overlay_primary': ('vx_kmh', 'vx')
            },
            'driver_inputs': {
                'title': "12. Input Pilota (Sterzo, Gas, Freno)", 'ylabel': "Comando [% / deg]",
                'signals': [('steer_deg', 'Sterzo delta [deg]', '#00cec9', '-'), ('throttle_pct', 'Gas [%]', '#2ecc71', '-'),
                            ('brake_pct', 'Freno [%]', '#e74c3c', '-')],
                'overlay_primary': ('steer_deg', 'Sterzo delta')
            },
            'haptics': {
                'title': "13. Feedback Aptico Grilletti & Rumble", 'ylabel': "Intensita [%]",
                'signals': [('haptic_throt_rt', 'Haptic Gas RT [%]', '#f1c40f', '-'), ('haptic_brake_lt', 'Haptic Freno LT [%]', '#e74c3c', '-'),
                            ('rumble_heavy', 'Rumble Telaio [%]', '#9b59b6', ':'), ('rumble_light', 'Rumble Gomme [%]', '#1abc9c', ':')],
                'overlay_primary': ('rumble_heavy', 'Rumble')
            },
            'roll_centers': {
                'title': "14. Centri di Rollio Dinamici [mm]", 'ylabel': "Altezza RC [mm]",
                'signals': [('h_rc_f_mm', 'RC Ant. [mm]', '#0984e3', '-'), ('h_rc_r_mm', 'RC Post. [mm]', '#e17055', '-')],
                'overlay_primary': ('h_rc_f_mm', 'RC Ant')
            },
            'tire_temps_bulk': {
                'title': "15. Temperature Gomme Bulk (4 Ruote) [°C]", 'ylabel': "Temp [°C]",
                'signals': [('tire_T_bulk_fl', 'T_bulk FL [°C]', '#0984e3', '-'), ('tire_T_bulk_fr', 'T_bulk FR [°C]', '#00cec9', '-'),
                            ('tire_T_bulk_rl', 'T_bulk RL [°C]', '#e84393', '-'), ('tire_T_bulk_rr', 'T_bulk RR [°C]', '#e17055', '-')],
                'overlay_primary': ('tire_T_bulk_fl', 'T_bulk FL')
            },
            'tire_temps_detailed': {
                'title': "16. Strati Termici 3-Zone (Surf, Tread, Carc) [°C]", 'ylabel': "Temp [°C]",
                'signals': [('tire_T_surf_fl', 'FL Superficie [°C]', '#e74c3c', '-'), ('tire_T_tread_fl', 'FL Battistrada [°C]', '#e67e22', '--'),
                            ('tire_T_carc_fl', 'FL Carcassa [°C]', '#3498db', ':'), ('tire_T_surf_rr', 'RR Superficie [°C]', '#9b59b6', '-')],
                'overlay_primary': ('tire_T_surf_fl', 'T_surf FL')
            },
            'tire_wear_grip': {
                'title': "17. Usura Gomme [%] & Grip Multiplier", 'ylabel': "Valore [%]",
                'signals': [('tire_grip_fl_pct', 'Grip FL [%]', '#2ecc71', '-'), ('tire_grip_rr_pct', 'Grip RR [%]', '#27ae60', '-'),
                            ('tire_wear_fl', 'Usura FL [%]', '#e74c3c', '--'), ('tire_wear_rr', 'Usura RR [%]', '#e67e22', '--')],
                'overlay_primary': ('tire_grip_fl_pct', 'Grip FL')
            }
        }

        if plot_key not in plot_configs:
            return

        cfg = plot_configs[plot_key]
        title = cfg['title']
        ylabel = cfg['ylabel']

        # --- LAP OVERLAY MODE ---
        if lap_mode == 'overlay' and len(self.laps) > 1:
            col, short_lbl = cfg['overlay_primary']
            for l_id, lap_df in sorted(self.laps.items()):
                if x_mode == 's_lap' and 's_lap' in lap_df.columns:
                    lap_df_sorted = lap_df.sort_values('s_lap')
                    x_lap = lap_df_sorted['s_lap']
                    x_axis_label = "Distanza Pista s [m]"
                elif x_mode == 't':
                    lap_df_sorted = lap_df
                    x_lap = lap_df['t'] - lap_df['t'].iloc[0]
                    x_axis_label = "Tempo Trascorso nel Giro dt [s]"
                else:
                    lap_df_sorted = lap_df
                    x_lap = lap_df.get('dist_total', lap_df['t'])
                    x_axis_label = "Distanza Totale [m]"

                if col == 'tire_grip_fl_pct' and 'tire_grip_fl' in lap_df_sorted.columns:
                    y_lap = lap_df_sorted['tire_grip_fl'] * 100.0
                elif col in lap_df_sorted.columns:
                    y_lap = lap_df_sorted[col]
                else:
                    continue

                is_best = (l_id == self.best_lap_idx)
                lap_dt = self.lap_stats.get(l_id, {}).get('dt', 0.0)
                lap_t_str = f"{lap_dt:.2f}s"
                color = '#f1c40f' if is_best else self.LAP_COLORS[l_id % len(self.LAP_COLORS)]
                lw = 2.4 if is_best else 1.3
                lbl = f"★ Best Lap (G{l_id+1}: {lap_t_str})" if is_best else f"Giro {l_id+1} ({lap_t_str})"
                ax.plot(x_lap, y_lap, label=lbl, color=color, linewidth=lw)

                # In focused view, add auxiliary secondary channels per lap if applicable
                if not is_grid and plot_key == 'chassis_velocities' and 'vy_kmh' in lap_df_sorted.columns:
                    ax.plot(x_lap, lap_df_sorted['vy_kmh'], color=color, linewidth=1.0, linestyle='--', alpha=0.55)
                if not is_grid and plot_key == 'driver_inputs' and 'throttle_pct' in lap_df_sorted.columns:
                    ax.plot(x_lap, lap_df_sorted['throttle_pct'], color=color, linewidth=1.0, linestyle=':', alpha=0.55)

            if plot_key in ['tire_temps_bulk', 'tire_temps_detailed']:
                ax.axhline(56.2, color='#2ecc71', linestyle=':', label='Ottimale TTC (56.2°C)')

            ax.set_title(f"{title} [Overlay Giro vs Giro]", fontsize=fs_title, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=fs_lbl)
            ax.set_xlabel(x_axis_label, fontsize=fs_lbl)

        # --- STANDARD / FILTERED VIEW (ALL, BEST, LAST) ---
        else:
            if lap_mode == 'best' and self.best_lap_idx in self.laps:
                sub_df = self.laps[self.best_lap_idx].copy()
                suffix = f" [Best Lap: Giro {self.best_lap_idx + 1}]"
            elif lap_mode == 'last' and self.laps:
                last_id = sorted(self.laps.keys())[-1]
                sub_df = self.laps[last_id].copy()
                suffix = f" [Ultimo Giro: Giro {last_id + 1}]"
            else:
                sub_df = self.df.copy()
                suffix = "" if len(self.laps) > 1 else " (Giro 1)"

            if x_mode == 's_lap' and 's_lap' in sub_df.columns:
                if lap_mode in ['best', 'last']:
                    sub_df = sub_df.sort_values('s_lap')
                x_vals = sub_df['s_lap']
                x_axis_label = "Distanza Pista s [m]"
            elif x_mode == 'dist_total' and 'dist_total' in sub_df.columns:
                x_vals = sub_df['dist_total']
                x_axis_label = "Distanza Totale [m]"
            else:
                x_vals = sub_df['t'] if 't' in sub_df.columns else np.arange(len(sub_df))
                x_axis_label = "Tempo [s]"

            for col, lbl, color, ls in cfg['signals']:
                if col == 'tire_grip_fl_pct' and 'tire_grip_fl' in sub_df.columns:
                    y_vals = sub_df['tire_grip_fl'] * 100.0
                elif col == 'tire_grip_rr_pct' and 'tire_grip_rr' in sub_df.columns:
                    y_vals = sub_df['tire_grip_rr'] * 100.0
                elif col in sub_df.columns:
                    y_vals = sub_df[col]
                else:
                    continue
                ax.plot(x_vals, y_vals, label=lbl, color=color, linestyle=ls)

            if plot_key in ['tire_temps_bulk', 'tire_temps_detailed']:
                ax.axhline(56.2, color='#2ecc71', linestyle=':', label='Ottimale TTC (56.2°C)')

            ax.set_title(f"{title}{suffix}", fontsize=fs_title, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=fs_lbl)
            ax.set_xlabel(x_axis_label, fontsize=fs_lbl)

        ax.grid(True, linestyle=':', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=fs_leg, loc='best')
"""

def main():
    with open('interactive_driver_14dof.py', 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'class FullTelemetryWindow\(QDialog\):.*?(?=# =+\n# 5\.5 INTERACTIVE 2D TRACK DESIGNER)', re.DOTALL)
    m = pattern.search(content)
    if not m:
        print("ERROR: Could not find FullTelemetryWindow in interactive_driver_14dof.py")
        return 1

    print(f"Matched FullTelemetryWindow from {m.start()} to {m.end()}")
    updated_content = content[:m.start()] + NEW_FULL_TELEMETRY + "\n\n\n" + content[m.end():]

    with open('interactive_driver_14dof.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print("Successfully replaced FullTelemetryWindow!")
    return 0

if __name__ == '__main__':
    exit(main())
