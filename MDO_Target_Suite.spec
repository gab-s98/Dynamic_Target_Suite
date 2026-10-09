# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
import casadi
import openmdao
from PyInstaller.utils.hooks import collect_all

# Preserve any user setups from dist directory before rebuild
dist_setups = os.path.join(os.getcwd(), 'dist', 'MDO_Target_Suite', 'data', 'setups')
root_setups = os.path.join(os.getcwd(), 'data', 'setups')
os.makedirs(root_setups, exist_ok=True)
if os.path.exists(dist_setups):
    for root, dirs, files in os.walk(dist_setups):
        rel_dir = os.path.relpath(root, dist_setups)
        target_dir = os.path.join(root_setups, rel_dir) if rel_dir != '.' else root_setups
        os.makedirs(target_dir, exist_ok=True)
        for fname in files:
            if fname.endswith('.json') and fname != 'default_vehicle.json':
                src_f = os.path.join(root, fname)
                dst_f = os.path.join(target_dir, fname)
                if not os.path.exists(dst_f) or os.path.getmtime(src_f) >= os.path.getmtime(dst_f):
                    try:
                        shutil.copy2(src_f, dst_f)
                    except Exception:
                        pass

datas = [
    ('data', 'data'),
    ('data/tire_tir', 'tire_tir'),
    ('docs', 'docs'),
    ('data/tracks', 'tracks'),
    ('data/setups', 'setups'),
    ('data/tire_datas', 'tire_datas'),
    ('data/design_suspension', 'design_suspension'),
    ('scripts', 'scripts'),
    ('data/motor_curve', 'motor_curve'),
]
binaries = []
import glob
for dll_f in glob.glob('.cache/c_build/ode_14dof*.dll') + glob.glob('ode_14dof*.dll'):
    binaries.append((dll_f, '.'))
    binaries.append((dll_f, '.cache/c_build'))
for sig_f in glob.glob('.cache/c_build/ode_14dof*.sig') + glob.glob('ode_14dof*.sig'):
    datas.append((sig_f, '.'))
    datas.append((sig_f, '.cache/c_build'))

hiddenimports = [
    'scipy.optimize', 'scipy.spatial', 'scipy.interpolate', 'matplotlib', 
    'PyQt5', 'casadi',
    'src', 'src.models', 'src.mdo', 'src.core', 'src.ui', 'src.utils',
    'src.models.vehicle_3dof', 'src.models.vehicle_7dof', 'src.models.vehicle_14dof',
    'src.models.tire_model_mf61', 'src.models.tire_degradation', 'src.models.damper_model_ttx25',
    'src.models.powertrain_electric', 'src.models.steering_dynamics', 'src.models.vehicle_3d_models',
    'src.models.params_vehicle', 'src.models.params_tire',
    'src.mdo.mdo_acceleration', 'src.mdo.mdo_skidpad', 'src.mdo.mdo_global_optimizer',
    'src.mdo.mdo_direct_collocation', 'src.mdo.mdo_dynamic_events', 'src.mdo.mdo_seven_post_rig',
    'src.mdo.mdo_arb_diamond_balance', 'src.mdo.mdo_balance_range', 'src.mdo.mdo_sensitivity',
    'src.mdo.mdo_uncertainty', 'src.mdo.mdo_target_optimization', 'src.mdo.mdo_steady_state',
    'src.core.simulator', 'src.core.om_vehicle_equilibrium', 'src.core.road_profile',
    'src.core.simulation_options', 'src.core.gamepad_manager', 'src.core.physics_thread',
    'src.core.vehicle_sim_controller',
    'src.ui.main_gui', 'src.ui.gui_canvas', 'src.ui.gui_dialogs', 'src.ui.gui_workers',
    'src.ui.ui_theme_engine', 'src.ui.ui_seven_post_rig', 'src.ui.vehicle_parameters_inspector_dialog',
    'src.ui.ui_direct_collocation',
    'src.ui.interactive_driver_14dof', 'src.ui.sim_controls', 'src.ui.sim_main_window',
    'src.ui.sim_telemetry', 'src.ui.sim_viewport', 'src.ui.telemetry_viewer', 'src.ui.track_editor',
    'src.ui.main_window', 'src.ui.main_gui_pro', 'src.ui.main_gui_classic',
    'src.utils.track_manager', 'src.utils.curve_digitizer', 'src.utils.vehicle_dna_maps',
    'src.utils.optimized_params_manager', 'src.utils.suspension_sizing', 'src.utils.transient_sensitivity',
    'src.utils.logger',
    'scripts.preview_car_3d'
]

# Collect CasADi
tmp_ret = collect_all('casadi')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Ensure complete CasADi directory is included in datas as raw files
casadi_pkg_dir = os.path.dirname(casadi.__file__)
datas.append((casadi_pkg_dir, 'casadi'))

# Collect OpenMDAO
tmp_om = collect_all('openmdao')
datas += tmp_om[0]
binaries += tmp_om[1]
hiddenimports += tmp_om[2]

# Ensure complete OpenMDAO directory is included in datas as raw files (contains unit_library.ini)
om_pkg_dir = os.path.dirname(openmdao.__file__)
datas.append((om_pkg_dir, 'openmdao'))

# Collect PyOpenGL
try:
    tmp_ret = collect_all('OpenGL')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except Exception:
    hiddenimports += ['OpenGL', 'OpenGL.GL', 'OpenGL.GLU', 'OpenGL.platform', 'OpenGL.platform.win32']

a = Analysis(
    ['main.py'],
    pathex=['.', 'src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Remove casadi and openmdao from pure python modules in PYZ archive to prevent
# zlib MemoryError during startup extraction and ensure relative resource files
# (such as openmdao/utils/unit_library.ini) are loaded directly from disk.
a.pure = [item for item in a.pure if not item[0].startswith('casadi') and not item[0].startswith('openmdao')]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MDO_Target_Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MDO_Target_Suite',
)

# Copy user-facing folders next to the exe for user convenience
dist_root = os.path.join(os.getcwd(), 'dist', 'MDO_Target_Suite')
for folder_name in ['setups', 'design_suspension', 'tracks', 'tire_tir', 'motor_curve']:
    src_dir = os.path.join(os.getcwd(), 'data', folder_name) if os.path.exists(os.path.join(os.getcwd(), 'data', folder_name)) else os.path.join(os.getcwd(), folder_name)
    dst_dir = os.path.join(dist_root, folder_name)
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
data_src = os.path.join(os.getcwd(), 'data')
data_dst = os.path.join(dist_root, 'data')
if os.path.exists(data_src):
    shutil.copytree(data_src, data_dst, dirs_exist_ok=True)

# Ensure OpenMDAO unit_library.ini and resources are directly present in _internal/openmdao
om_internal = os.path.join(dist_root, '_internal', 'openmdao')
if os.path.exists(om_pkg_dir):
    shutil.copytree(om_pkg_dir, om_internal, dirs_exist_ok=True)



