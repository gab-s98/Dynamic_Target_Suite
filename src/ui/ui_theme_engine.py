"""
ui_theme_engine.py
------------------
Motore di styling e gestione temi per Dynamic Target Suite Pro.
Fornisce schemi cromatici ingegneristici professionali (Motorsport Dark,
Engineering Light, High Contrast), sincronizzazione automatica dei canvas
Matplotlib, calibrazione densità interfaccia e persistenza preferenze.
"""

import os
import json
import numpy as np
from matplotlib.figure import Figure

THEMES = {
    'dark': {
        'name': 'Motorsport Scuro',
        'bg_root': '#0d1117',
        'bg_card': '#161b22',
        'bg_input': '#21262d',
        'bg_input_focus': '#28313e',
        'border': '#30363d',
        'border_focus': '#388bfd',
        'text_primary': '#f0f6fc',
        'text_secondary': '#8b949e',
        'text_muted': '#6e7681',
        'accent_primary': '#1f6feb',
        'accent_primary_hover': '#388bfd',
        'accent_secondary': '#238636',
        'accent_secondary_hover': '#2ea043',
        'accent_warning': '#d29922',
        'accent_danger': '#f85149',
        'selection_bg': '#1f6feb',
        'selection_text': '#ffffff',
        'header_bg': '#12161f',
        'tab_active_bg': '#21262d',
        'tab_inactive_bg': '#141820',
        'plot_fig_bg': '#0d1117',
        'plot_ax_bg': '#161b22',
        'plot_spines': '#30363d',
        'plot_ticks': '#8b949e',
        'plot_labels': '#c9d1d9',
        'plot_title': '#f0f6fc',
        'plot_grid': '#30363d',
        'plot_colors': ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff', '#ff7b72', '#d29922', '#39c5cf', '#e3b341']
    },
    'light': {
        'name': 'Ingegneristico Chiaro',
        'bg_root': '#f1f5f9',
        'bg_card': '#ffffff',
        'bg_input': '#ffffff',
        'bg_input_focus': '#f8fafc',
        'border': '#94a3b8',
        'border_focus': '#0052cc',
        'text_primary': '#000000',
        'text_secondary': '#0f172a',
        'text_muted': '#334155',
        'accent_primary': '#0052cc',
        'accent_primary_hover': '#003d99',
        'accent_secondary': '#15803d',
        'accent_secondary_hover': '#166534',
        'accent_warning': '#9a3412',
        'accent_danger': '#b91c1c',
        'selection_bg': '#0052cc',
        'selection_text': '#ffffff',
        'header_bg': '#e2e8f0',
        'tab_active_bg': '#ffffff',
        'tab_inactive_bg': '#e2e8f0',
        'plot_fig_bg': '#ffffff',
        'plot_ax_bg': '#ffffff',
        'plot_spines': '#000000',
        'plot_ticks': '#000000',
        'plot_labels': '#000000',
        'plot_title': '#000000',
        'plot_grid': '#cbd5e1',
        'plot_colors': ['#0052cc', '#c2410c', '#15803d', '#7e22ce', '#b91c1c', '#a16207', '#0e7490', '#334155']
    },
    'contrast': {
        'name': 'Alto Contrasto Monocromatico',
        'bg_root': '#000000',
        'bg_card': '#0d0d0d',
        'bg_input': '#1a1a1a',
        'bg_input_focus': '#262626',
        'border': '#404040',
        'border_focus': '#ffffff',
        'text_primary': '#ffffff',
        'text_secondary': '#b3b3b3',
        'text_muted': '#808080',
        'accent_primary': '#2563eb',
        'accent_primary_hover': '#3b82f6',
        'accent_secondary': '#16a34a',
        'accent_secondary_hover': '#22c55e',
        'accent_warning': '#eab308',
        'accent_danger': '#dc2626',
        'selection_bg': '#ffffff',
        'selection_text': '#000000',
        'header_bg': '#050505',
        'tab_active_bg': '#262626',
        'tab_inactive_bg': '#0d0d0d',
        'plot_fig_bg': '#000000',
        'plot_ax_bg': '#0a0a0a',
        'plot_spines': '#555555',
        'plot_ticks': '#cccccc',
        'plot_labels': '#ffffff',
        'plot_title': '#ffffff',
        'plot_grid': '#333333',
        'plot_colors': ['#38bdf8', '#fb923c', '#4ade80', '#f43f5e', '#a78bfa', '#facc15', '#2dd4bf', '#ffffff']
    }
}

DENSITIES = {
    'compact': {
        'font_family': 'Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif',
        'font_size_base': '9pt',
        'font_size_small': '8pt',
        'font_size_header': '10.5pt',
        'padding_btn': '3px 8px',
        'padding_input': '2px 5px',
        'min_height_btn': '22px',
        'spacing_layout': '4px',
        'margin_card': '3px',
    },
    'standard': {
        'font_family': 'Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif',
        'font_size_base': '9.5pt',
        'font_size_small': '8.5pt',
        'font_size_header': '11pt',
        'padding_btn': '4px 10px',
        'padding_input': '2px 6px',
        'min_height_btn': '25px',
        'spacing_layout': '5px',
        'margin_card': '4px',
    }
}

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_settings.json')

def load_ui_settings():
    defaults = {
        'theme': 'light',
        'density': 'standard',
        'active_workspace': 0,
        'sidebar_visible': True,
        'preferred_gui': 'pro'
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                defaults.update(data)
        except Exception:
            pass
    return defaults

def save_ui_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

def get_theme_palette(theme_name='dark'):
    return THEMES.get(theme_name, THEMES['dark'])

def get_density_config(density_name='standard'):
    return DENSITIES.get(density_name, DENSITIES['standard'])

def get_stylesheet(theme_name='dark', density_name='standard'):
    t = get_theme_palette(theme_name)
    d = get_density_config(density_name)
    
    qss = f"""
    /* Base Application Stylesheet */
    QMainWindow, QDialog {{
        background-color: {t['bg_root']};
        color: {t['text_primary']};
        font-family: {d['font_family']};
        font-size: {d['font_size_base']};
    }}
    
    QWidget {{
        background-color: transparent;
        color: {t['text_primary']};
        font-family: {d['font_family']};
        font-size: {d['font_size_base']};
    }}
    
    /* Menu Bar */
    QMenuBar {{
        background-color: {t['header_bg']};
        color: {t['text_primary']};
        border-bottom: 1px solid {t['border']};
        padding: 2px 4px;
        font-size: {d['font_size_base']};
    }}
    QMenuBar::item {{
        background-color: transparent;
        padding: 4px 10px;
        border-radius: 3px;
    }}
    QMenuBar::item:selected {{
        background-color: {t['bg_input']};
    }}
    QMenu {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        padding: 4px 0px;
    }}
    QMenu::item {{
        padding: 5px 24px 5px 16px;
    }}
    QMenu::item:selected {{
        background-color: {t['selection_bg']};
        color: {t['selection_text']};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {t['border']};
        margin: 4px 8px;
    }}
    
    /* GroupBoxes */
    QGroupBox {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        margin-top: 14px;
        padding-top: 10px;
        padding-left: 6px;
        padding-right: 6px;
        padding-bottom: 6px;
        font-weight: 600;
        font-size: {d['font_size_base']};
        color: {t['text_primary']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 4px;
        background-color: {t['bg_card']};
        color: {t['accent_primary']};
        font-weight: bold;
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        padding: {d['padding_btn']};
        min-height: {d['min_height_btn']};
        font-weight: 500;
        font-size: {d['font_size_base']};
    }}
    QPushButton:hover {{
        background-color: {t['bg_input_focus']};
        border-color: {t['border_focus']};
    }}
    QPushButton:pressed {{
        background-color: {t['border']};
    }}
    QPushButton:disabled {{
        background-color: {t['bg_root']};
        color: {t['text_muted']};
        border-color: {t['border']};
    }}
    
    /* Ruoli Ingegneristici Pulsanti */
    QPushButton[role="primary"] {{
        background-color: {t['accent_primary']};
        color: #ffffff;
        border: 1px solid {t['accent_primary']};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{
        background-color: {t['accent_primary_hover']};
        border-color: {t['accent_primary_hover']};
    }}
    QPushButton[role="success"] {{
        background-color: {t['accent_secondary']};
        color: #ffffff;
        border: 1px solid {t['accent_secondary']};
        font-weight: 600;
    }}
    QPushButton[role="success"]:hover {{
        background-color: {t['accent_secondary_hover']};
        border-color: {t['accent_secondary_hover']};
    }}
    
    /* Ruoli Speciali Esecuzione & Run */
    QPushButton[role="primary_run"] {{
        background-color: {t['accent_primary']};
        color: #ffffff;
        border: 2px solid {t['accent_primary_hover']};
        border-radius: 4px;
        font-weight: 800;
        font-size: 10pt;
        padding: 5px 16px;
        min-height: 28px;
    }}
    QPushButton[role="primary_run"]:hover {{
        background-color: {t['accent_primary_hover']};
    }}
    QPushButton[role="primary_run"]:disabled {{
        background-color: {t['border']};
        border-color: {t['border']};
        color: {t['text_muted']};
    }}
    
    QPushButton[role="secondary_run"] {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 2px solid {t['accent_primary']};
        border-radius: 4px;
        font-weight: 700;
        font-size: 9.5pt;
        padding: 5px 12px;
        min-height: 28px;
    }}
    QPushButton[role="secondary_run"]:hover {{
        background-color: {t['header_bg']};
        border-color: {t['accent_primary_hover']};
        color: {t['accent_primary']};
    }}
    QPushButton[role="secondary_run"]:disabled {{
        background-color: {t['bg_root']};
        border-color: {t['border']};
        color: {t['text_muted']};
    }}
    
    QPushButton[role="layout_switch"] {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        font-weight: 600;
        font-size: 8.5pt;
        padding: 4px 10px;
    }}
    QPushButton[role="layout_switch"]:hover {{
        background-color: {t['header_bg']};
        border-color: {t['border_focus']};
    }}
    
    /* Header Tecnico */
    QFrame#pro_header {{
        background-color: {t['header_bg']};
        border-bottom: 1px solid {t['border']};
        border-radius: 4px;
    }}
    
    /* Navigation Rail Workspaces */
    QFrame#workspace_nav_bar {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 4px;
    }}
    QFrame#workspace_nav_bar QPushButton {{
        background-color: {t['bg_input']};
        color: {t['text_secondary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        padding: 6px 14px;
        font-weight: 600;
    }}
    QFrame#workspace_nav_bar QPushButton:hover {{
        background-color: {t['bg_input_focus']};
        color: {t['text_primary']};
    }}
    QFrame#workspace_nav_bar QPushButton:checked {{
        background-color: {t['accent_primary']};
        color: #ffffff;
        border: 1px solid {t['border_focus']};
    }}
    
    /* Inspector Panel */
    QFrame#pro_inspector_panel {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 4px;
    }}
    
    /* Status Bar */
    QFrame#pro_statusbar {{
        background-color: {t['header_bg']};
        border-top: 1px solid {t['border']};
        border-radius: 3px;
    }}
    
    /* Inputs & Combos */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        padding: {d['padding_input']};
        font-family: 'Segoe UI', 'Consolas', monospace, sans-serif;
        font-size: {d['font_size_base']};
        min-height: 22px;
        min-width: 50px;
        selection-background-color: {t['selection_bg']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {t['border_focus']};
        background-color: {t['bg_input_focus']};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {t['bg_root']};
        color: {t['text_muted']};
    }}
    
    QComboBox {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        padding: {d['padding_input']};
        min-height: {d['min_height_btn']};
        font-size: {d['font_size_base']};
    }}
    QComboBox:focus {{
        border: 1px solid {t['border_focus']};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        selection-background-color: {t['selection_bg']};
        selection-color: {t['selection_text']};
    }}
    
    /* Labels */
    QLabel {{
        color: {t['text_primary']};
        font-size: {d['font_size_base']};
    }}
    
    /* Tab Widgets */
    QTabWidget::pane {{
        border: 1px solid {t['border']};
        background-color: {t['bg_card']};
        top: -1px;
    }}
    QTabBar {{
        qproperty-usesScrollButtons: true;
        qproperty-drawBase: 0;
    }}
    QTabBar::tab {{
        background-color: {t['tab_inactive_bg']};
        color: {t['text_secondary']};
        border: 1px solid {t['border']};
        border-bottom-color: {t['border']};
        padding: 4px 8px;
        margin-right: 2px;
        font-weight: 500;
        font-size: {d['font_size_small']};
        min-height: 22px;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
    }}
    QTabBar::tab:selected {{
        background-color: {t['tab_active_bg']};
        color: {t['text_primary']};
        border-bottom-color: {t['tab_active_bg']};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
    }}
    QTabBar::scroller {{
        width: 20px;
    }}
    QTabBar QToolButton {{
        background-color: {t['bg_input']};
        border: 1px solid {t['border']};
        border-radius: 2px;
        margin: 1px;
    }}
    QTabBar QToolButton:hover {{
        background-color: {t['bg_input_focus']};
        border-color: {t['border_focus']};
    }}
    
    /* Tables */
    QTableWidget {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        gridline-color: {t['border']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        font-size: {d['font_size_small']};
        selection-background-color: {t['selection_bg']};
        selection-color: {t['selection_text']};
    }}
    QHeaderView::section {{
        background-color: {t['bg_input']};
        color: {t['text_secondary']};
        padding: 4px 6px;
        border: 1px solid {t['border']};
        font-weight: 600;
        font-size: {d['font_size_small']};
    }}
    QTableCornerButton::section {{
        background-color: {t['bg_input']};
        border: 1px solid {t['border']};
    }}
    
    /* Scrollbars */
    QScrollBar:vertical {{
        background: {t['bg_root']};
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {t['border']};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {t['text_muted']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {t['bg_root']};
        height: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {t['border']};
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {t['text_muted']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* Matplotlib Navigation Toolbars & Graphic Controls */
    QToolBar {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        padding: 2px 6px;
        spacing: 4px;
        min-height: 32px;
    }}
    
    QToolBar QToolButton {{
        background-color: {t['bg_input']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        padding: 3px 6px;
        margin: 1px 2px;
        min-width: 26px;
        min-height: 24px;
        font-weight: 500;
    }}
    
    QToolBar QToolButton:hover {{
        background-color: {t['bg_input_focus']};
        border-color: {t['border_focus']};
    }}
    
    QToolBar QToolButton:checked {{
        background-color: {t['accent_primary']};
        border-color: {t['border_focus']};
        color: #ffffff;
    }}
    
    QToolBar QToolButton:pressed {{
        background-color: {t['border']};
    }}
    
    QToolBar::separator {{
        width: 1px;
        background-color: {t['border']};
        margin: 3px 6px;
    }}
    
    /* Splitters */
    QSplitter::handle {{
        background-color: {t['border']};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    
    /* Checkboxes */
    QCheckBox {{
        color: {t['text_primary']};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {t['border']};
        border-radius: 2px;
        background-color: {t['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {t['accent_primary']};
        border-color: {t['accent_primary']};
    }}
    
    /* Text Editors / Browsers */
    QTextEdit, QTextBrowser {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 3px;
        font-family: 'Consolas', 'Cascadia Code', monospace;
        font-size: {d['font_size_small']};
    }}
    """
    return qss

def apply_plot_theme(fig: Figure, axes=None, theme_name='dark'):
    """Synchronizes a Matplotlib Figure and its Axes with the chosen theme palette."""
    t = get_theme_palette(theme_name)
    fig.patch.set_facecolor(t['plot_fig_bg'])
    
    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
        
    for ax in axes:
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(t['plot_ax_bg'])
        
        # Spines
        if hasattr(ax, 'spines'):
            for spine_name, spine in ax.spines.items():
                spine.set_color(t['plot_spines'])
                spine.set_linewidth(0.9)
            
        # Ticks
        if hasattr(ax, 'tick_params'):
            ax.tick_params(colors=t['plot_ticks'], which='both', labelsize=8.5)
        
        # Labels and title
        if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'label'):
            ax.xaxis.label.set_color(t['plot_labels'])
        if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'label'):
            ax.yaxis.label.set_color(t['plot_labels'])
        if hasattr(ax, 'title'):
            ax.title.set_color(t['plot_title'])
            
        # Grid
        if hasattr(ax, 'grid'):
            ax.grid(True, linestyle='--', alpha=0.22, color=t['plot_grid'])

def apply_app_palette(target, theme_name='light'):
    """Applica un QPalette corrispondente al target (QApplication o QWidget) per garantire che Qt e Matplotlib rilevino accuratamente la modalita dark/light."""
    from PyQt5 import QtGui
    t = get_theme_palette(theme_name)
    pal = target.palette()
    bg_color = QtGui.QColor(t['bg_card'])
    btn_color = QtGui.QColor(t['bg_input'])
    fg_color = QtGui.QColor(t['text_primary'])
    pal.setColor(QtGui.QPalette.Window, bg_color)
    pal.setColor(QtGui.QPalette.Background, bg_color)
    pal.setColor(QtGui.QPalette.Base, bg_color)
    pal.setColor(QtGui.QPalette.Button, btn_color)
    pal.setColor(QtGui.QPalette.WindowText, fg_color)
    pal.setColor(QtGui.QPalette.Text, fg_color)
    pal.setColor(QtGui.QPalette.ButtonText, fg_color)
    target.setPalette(pal)

