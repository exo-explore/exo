# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path.cwd()
SOURCE_ROOT = PROJECT_ROOT / "src"
ENTRYPOINT = SOURCE_ROOT / "exo" / "__main__.py"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard" / "build"
EXO_SHARED_MODELS_DIR = SOURCE_ROOT / "exo" / "shared" / "models"

if not ENTRYPOINT.is_file():
    raise SystemExit(f"Unable to locate Exo entrypoint: {ENTRYPOINT}")

if not DASHBOARD_DIR.is_dir():
    raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_DIR}")

if not EXO_SHARED_MODELS_DIR.is_dir():
    raise SystemExit(f"Shared model assets are missing: {EXO_SHARED_MODELS_DIR}")

block_cipher = None


def _module_directory(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(f"Module '{module_name}' is not available in the current environment.")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    raise SystemExit(f"Unable to determine installation directory for '{module_name}'.")


MLX_PACKAGE_DIR = _module_directory("mlx")
MLX_LIB_DIR = MLX_PACKAGE_DIR / "lib"
if not MLX_LIB_DIR.is_dir():
    raise SystemExit(f"mlx Metal libraries are missing: {MLX_LIB_DIR}")


def _safe_collect(package_name: str) -> list[str]:
    try:
        return collect_submodules(package_name)
    except ImportError:
        return []


HIDDEN_IMPORTS = sorted(
    set(
        collect_submodules("mlx")
        + _safe_collect("mlx_lm")
        + _safe_collect("transformers")
    )
)

DATAS: list[tuple[str, str]] = [
    (str(DASHBOARD_DIR), "dashboard"),
    (str(MLX_LIB_DIR), "mlx/lib"),
    (str(EXO_SHARED_MODELS_DIR), "exo/shared/models"),
]

MACMON_PATH = shutil.which("macmon")
if MACMON_PATH is None:
    raise SystemExit(
        "macmon binary not found in PATH. "
        "Install it via: brew install macmon"
    )

BINARIES: list[tuple[str, str]] = [
    (MACMON_PATH, "."),
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="exo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="exo",
)

