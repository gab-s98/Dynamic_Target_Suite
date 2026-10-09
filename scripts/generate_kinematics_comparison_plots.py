"""
generate_kinematics_comparison_plots.py
---------------------------------------
Generates high-resolution side-by-side and overlay comparison plots
between the digitized 3rd-order polynomials and the original Adams curves
from the suspension design presentation.
"""

import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_EXTRACTED = os.path.join(PROJECT_ROOT, 'design_suspension', 'extracted_curves')
OUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, 'design_suspension', 'verification_plots')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'runs', 'verification_plots')

os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def get_plot_box(arr):
    is_dark = (arr[:, :, 0] < 50) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
    col_counts = np.sum(is_dark, axis=0)
    row_counts = np.sum(is_dark, axis=1)

    top_cols = [x for x in np.argsort(col_counts)[::-1] if col_counts[x] > 180 and x < 150]
    right_cols = [x for x in np.argsort(col_counts)[::-1] if col_counts[x] > 180 and x > 1000]
    top_rows = [y for y in np.argsort(row_counts)[::-1] if row_counts[y] > 350 and y < 80]
    bottom_rows = [y for y in np.argsort(row_counts)[::-1] if row_counts[y] > 350 and y > 280]

    x_left = min(top_cols) if top_cols else 75
    x_right = max(right_cols) if right_cols else (1135 if arr.shape[1] > 1165 else 1122)
    y_top = min(top_rows) if top_rows else 38
    y_bottom = max(bottom_rows) if bottom_rows else (414 if arr.shape[0] > 450 else 346)

    return x_left, x_right, y_top, y_bottom


def extract_raw_points(arr, xl, xr, yt, yb, x_min_phys, x_max_phys, y_min_phys, y_max_phys, color='blue', legend_box=None):
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    if color == 'blue':
        mask = (b > 110) & (b - r > 35) & (b - g > 25)
    elif color == 'red':
        mask = (r > 120) & (r - b > 40) & (r - g > 30)
    else:
        raise ValueError("Unknown color")

    mask[:yt + 1, :] = False
    mask[yb:, :] = False
    mask[:, :xl + 1] = False
    mask[:, xr:] = False

    if legend_box:
        lx1, lx2, ly1, ly2 = legend_box
        mask[ly1:ly2, lx1:lx2] = False

    xs_phys, ys_phys = [], []
    for x in range(xl + 1, xr):
        col_pts = np.where(mask[:, x])[0]
        if len(col_pts) > 0:
            y_val = np.median(col_pts)
            x_phys = x_min_phys + (x - xl) / (xr - xl) * (x_max_phys - x_min_phys)
            y_phys = y_max_phys - (y_val - yt) / (yb - yt) * (y_max_phys - y_min_phys)
            xs_phys.append(x_phys)
            ys_phys.append(y_phys)

    return np.array(xs_phys), np.array(ys_phys)


def evaluate_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    rmse = np.sqrt(np.mean(residuals**2))
    max_err = np.max(np.abs(residuals))
    return r2, rmse, max_err


# Curve specifications dictionary
CURVES_SPECS = [
    {
        'id': 'camber_f',
        'title': 'Front Camber vs Wheel Travel',
        'file': r'01_Front_Suspension\01_camber_gain_f__camber_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-3.0, 0.0),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Camber Angle [deg]',
        'color': 'blue',
        'leg': (260, 445, 50, 115),
        'scale_poly_x': 0.001,  # mm -> m for poly eval
        'scale_poly_y': 1.0,
        'poly_raw': [-87.4468, -53.3962, -33.8889, -1.50067],
    },
    {
        'id': 'camber_r',
        'title': 'Rear Camber vs Wheel Travel',
        'file': r'02_Rear_Suspension\01_camber_gain_r__camber_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-3.5, 1.5),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Camber Angle [deg]',
        'color': 'blue',
        'leg': (900, 1135, 40, 115),
        'scale_poly_x': 0.001,
        'scale_poly_y': 1.0,
        'poly_raw': [-397.785, -123.972, -70.7528, -1.00488],
    },
    {
        'id': 'toe_f',
        'title': 'Front Bump Steer / Toe vs Wheel Travel',
        'file': r'01_Front_Suspension\04_toe_gain_f__bump_steer_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-0.01, 0.03),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Toe Angle [deg]',
        'color': 'blue',
        'leg': (95, 310, 40, 110),
        'scale_poly_x': 0.001,
        'scale_poly_y': 1.0,
        'poly_raw': [-50.4274, 22.4794, -0.0847, 0.00018],
    },
    {
        'id': 'toe_r',
        'title': 'Rear Bump Steer / Toe vs Wheel Travel',
        'file': r'02_Rear_Suspension\03_toe_gain_r__bump_steer_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-0.015, 0.085),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Toe Angle [deg]',
        'color': 'blue',
        'leg': (100, 640, 40, 110),
        'scale_poly_x': 0.001,
        'scale_poly_y': 1.0,
        'poly_raw': [4.8981, 34.3142, 0.2507, 0.00021],
    },
    {
        'id': 'anti_dive',
        'title': 'Front Anti-Dive vs Wheel Travel',
        'file': r'01_Front_Suspension\14_anti_dive__antidive_percent_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (20.0, 40.0),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Anti-Dive [%]',
        'color': 'blue',
        'leg': (85, 275, 40, 110),
        'scale_poly_x': 0.001,
        'scale_poly_y': 100.0,
        'poly_raw': [-6.6852, -0.4540, -2.1650, 0.2999],
    },
    {
        'id': 'anti_squat',
        'title': 'Rear Anti-Squat vs Wheel Travel',
        'file': r'02_Rear_Suspension\08_anti_squat__antisquat_percent_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-16.0, -10.0),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Anti-Squat [%]',
        'color': 'blue',
        'leg': (155, 350, 40, 110),
        'scale_poly_x': 0.001,
        'scale_poly_y': 100.0,
        'poly_raw': [-6.8919, 0.4834, -0.4616, -0.1463],
    },
    {
        'id': 'anti_lift',
        'title': 'Rear Anti-Lift vs Wheel Travel',
        'file': r'02_Rear_Suspension\07_anti_lift_r__antilift_percent_vs_wheel_travel.jpeg',
        'x_lim': (-25.0, 25.0),
        'y_lim': (-5.0, 10.0),
        'x_label': 'Wheel Travel [mm]',
        'y_label': 'Anti-Lift [%]',
        'color': 'blue',
        'leg': (70, 240, 160, 230),
        'scale_poly_x': 0.001,
        'scale_poly_y': 100.0,
        'poly_raw': [1.009, -0.163, -0.052, 0.100],
    },
    {
        'id': 'rc_lat_f',
        'title': 'Front Roll Center Lateral Disp. vs Roll Angle',
        'file': r'01_Front_Suspension\13_rollcenter_lateral_f__roll_center_lat_disp_vs_roll_angle.jpeg',
        'x_lim': (0.0, 1.5),
        'y_lim': (0.0, 150.0),
        'x_label': 'Roll Angle [deg]',
        'y_label': 'Roll Center Lat. Disp. [mm]',
        'color': 'blue',
        'leg': (100, 350, 38, 105),
        'scale_poly_x': 1.0,
        'scale_poly_y': 1.0,
        'poly_raw': None,
    },
    {
        'id': 'rc_lat_r',
        'title': 'Rear Roll Center Lateral Disp. vs Roll Angle',
        'file': r'02_Rear_Suspension\06_rollcenter_lateral_r__roll_center_lat_vs_roll_angle.jpeg',
        'x_lim': (0.0, 1.5),
        'y_lim': (0.0, 100.0),
        'x_label': 'Roll Angle [deg]',
        'y_label': 'Roll Center Lat. Disp. [mm]',
        'color': 'blue',
        'leg': (100, 420, 38, 105),
        'scale_poly_x': 1.0,
        'scale_poly_y': 1.0,
        'poly_raw': None,
    }
]


def process_single_curve(spec):
    img_path = os.path.join(BASE_EXTRACTED, spec['file'])
    img = Image.open(img_path)
    arr = np.array(img)
    xl, xr, yt, yb = get_plot_box(arr)
    cropped = arr[yt:yb, xl:xr]

    xs, ys = extract_raw_points(
        arr, xl, xr, yt, yb,
        spec['x_lim'][0], spec['x_lim'][1],
        spec['y_lim'][0], spec['y_lim'][1],
        color=spec['color'],
        legend_box=spec['leg']
    )

    x_dense = np.linspace(spec['x_lim'][0], spec['x_lim'][1], 300)

    if spec['poly_raw'] is not None:
        p = np.poly1d(spec['poly_raw'])
        # Evaluate at raw points
        x_eval_raw = xs * spec['scale_poly_x']
        y_fit_at_raw = p(x_eval_raw) * spec['scale_poly_y']
        
        # Evaluate on dense grid
        x_eval_dense = x_dense * spec['scale_poly_x']
        y_fit_dense = p(x_eval_dense) * spec['scale_poly_y']
    else:
        # Roll center lateral linear fits vs roll angle
        deg2rad = np.pi / 180.0
        if spec['id'] == 'rc_lat_f':
            dz = 1.252 * np.sin(xs * deg2rad)
            slope = 2.94004
            y_fit_at_raw = (slope * dz) * 1000.0
            dz_dense = 1.252 * np.sin(x_dense * deg2rad)
            y_fit_dense = (slope * dz_dense) * 1000.0
        else:
            dz = 1.136 * np.sin(xs * deg2rad)
            slope = 2.13100
            y_fit_at_raw = (slope * dz) * 1000.0
            dz_dense = 1.136 * np.sin(x_dense * deg2rad)
            y_fit_dense = (slope * dz_dense) * 1000.0

    r2, rmse, max_err = evaluate_metrics(ys, y_fit_at_raw)

    return {
        'spec': spec,
        'cropped': cropped,
        'xs': xs,
        'ys': ys,
        'x_dense': x_dense,
        'y_fit_dense': y_fit_dense,
        'r2': r2,
        'rmse': rmse,
        'max_err': max_err
    }


def create_detailed_figure(data, out_filename):
    spec = data['spec']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

    # 1. Direct Overlay onto original Adams plot
    ax1.imshow(
        data['cropped'],
        extent=[spec['x_lim'][0], spec['x_lim'][1], spec['y_lim'][0], spec['y_lim'][1]],
        aspect='auto',
        alpha=0.90
    )
    # Bright high-contrast overlay
    ax1.plot(
        data['x_dense'], data['y_fit_dense'],
        color='#FF1493', linestyle='--', linewidth=2.8,
        label='Polinomio Fittato 2026 (Fit 3° ordine)'
    )
    ax1.set_xlim(spec['x_lim'])
    ax1.set_ylim(spec['y_lim'])
    ax1.set_xlabel(spec['x_label'], fontsize=11, fontweight='bold')
    ax1.set_ylabel(spec['y_label'], fontsize=11, fontweight='bold')
    ax1.set_title(f"[OVERLAY DIRETTO] {spec['title']}", fontsize=12, fontweight='bold', color='#111111')
    ax1.legend(loc='lower right' if spec['id'] in ['anti_dive', 'camber_f'] else 'best', framealpha=0.9)
    ax1.grid(True, alpha=0.35, color='#888888')

    # 2. Scatter comparison & metrics
    ax2.scatter(
        data['xs'], data['ys'],
        s=12, color='#0066cc', alpha=0.55, label='Pixel campionati da curva blu 2026'
    )
    ax2.plot(
        data['x_dense'], data['y_fit_dense'],
        color='#d62728', linestyle='-', linewidth=2.2, label='Curva analitica polinomio fittato'
    )
    ax2.set_xlim(spec['x_lim'])
    ax2.set_ylim(spec['y_lim'])
    ax2.set_xlabel(spec['x_label'], fontsize=11, fontweight='bold')
    ax2.set_ylabel(spec['y_label'], fontsize=11, fontweight='bold')
    ax2.set_title(f"[VERIFICA STATISTICA] R² = {data['r2']:.5f} | RMSE = {data['rmse']:.4f}", fontsize=12, fontweight='bold', color='#111111')
    
    stat_box = (
        f"Accuratezza Fitting:\n"
        f"• R²: {data['r2']:.6f}\n"
        f"• RMSE: {data['rmse']:.4f}\n"
        f"• Max Error: {data['max_err']:.4f}\n"
        f"• Campioni N: {len(data['xs'])}"
    )
    ax2.text(
        0.04, 0.10, stat_box, transform=ax2.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92)
    )

    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUT_PLOTS_DIR, out_filename)
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    artifact_copy = os.path.join(ARTIFACTS_DIR, out_filename)
    shutil.copy2(out_path, artifact_copy)
    print(f"Saved & copied: {out_filename}")


def create_master_dashboard(all_results):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), dpi=150)
    fig.suptitle('Dynamic Target Suite — Quadro di Verifica Completo: Polinomi 2026 vs Curve Adams', fontsize=18, fontweight='bold', y=0.99)
    
    axes_flat = axes.flatten()
    for idx, res in enumerate(all_results):
        ax = axes_flat[idx]
        spec = res['spec']
        
        ax.imshow(
            res['cropped'],
            extent=[spec['x_lim'][0], spec['x_lim'][1], spec['y_lim'][0], spec['y_lim'][1]],
            aspect='auto',
            alpha=0.65
        )
        ax.scatter(res['xs'], res['ys'], s=6, color='#0055ff', alpha=0.6, label='Curve 2026 (Adams)')
        ax.plot(res['x_dense'], res['y_fit_dense'], color='#d60000', linewidth=2.2, linestyle='-', label=f"Polinomio (R²={res['r2']:.4f})")
        
        ax.set_xlim(spec['x_lim'])
        ax.set_ylim(spec['y_lim'])
        ax.set_title(spec['title'], fontsize=11, fontweight='bold')
        ax.set_xlabel(spec['x_label'], fontsize=9)
        ax.set_ylabel(spec['y_label'], fontsize=9)
        ax.legend(loc='best', fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dash_filename = 'dashboard_confronto_cinematica_2026.png'
    out_path = os.path.join(OUT_PLOTS_DIR, dash_filename)
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    artifact_copy = os.path.join(ARTIFACTS_DIR, dash_filename)
    shutil.copy2(out_path, artifact_copy)
    print(f"Saved master dashboard: {dash_filename}")


def main():
    print("Processing all curves for comparison plots...")
    all_results = []
    for spec in CURVES_SPECS:
        print(f"Processing {spec['id']}...")
        res = process_single_curve(spec)
        all_results.append(res)
        
        out_name = f"confronto_{spec['id']}.png"
        create_detailed_figure(res, out_name)

    create_master_dashboard(all_results)
    print("All comparison plots generated successfully!")


if __name__ == '__main__':
    main()
