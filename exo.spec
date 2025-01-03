# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import shutil
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# Basic Configuration
block_cipher = None
name = os.environ.get('EXO_NAME', 'exo')
spec_dir = os.path.dirname(os.path.abspath(SPEC))
root_dir = os.path.abspath(os.path.join(spec_dir))

# Get Python library path dynamically
python_exec = sys.executable
python_prefix = sys.prefix
if sys.platform.startswith('darwin'):
    # On macOS, construct library path based on current Python version
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lib_patterns = [
        os.path.join(python_prefix, 'lib', f'libpython{version}.dylib'),
        os.path.join(python_prefix, 'lib', f'libpython{version}m.dylib'),
        os.path.join(os.path.dirname(python_exec), '..', 'lib', f'libpython{version}.dylib'),
    ]
    
    PYTHON_LIB = None
    for pattern in lib_patterns:
        if os.path.exists(pattern):
            PYTHON_LIB = pattern
            break
    
    if not PYTHON_LIB:
        raise FileNotFoundError(f"Could not find Python library for Python {version}")
    print(f"Using Python library: {PYTHON_LIB}")

# Create a local copy of the Python library
local_lib_dir = os.path.join(spec_dir, 'lib')
os.makedirs(local_lib_dir, exist_ok=True)
local_python_lib = os.path.join(local_lib_dir, 'libpython3.12.dylib')
shutil.copy2(PYTHON_LIB, local_python_lib)
print(f"Copied Python library to: {local_python_lib}")

# Model Collection
models_dir = os.path.join(root_dir, 'exo', 'inference', 'mlx', 'models')
model_files = []
for root, dirs, files in os.walk(models_dir):
    for file in files:
        if file.endswith('.py') and file not in ['__init__.py', 'base.py']:
            model_files.append(os.path.join(root, file))

model_imports = [
    f"exo.inference.mlx.models.{os.path.basename(f)[:-3]}"
    for f in model_files
    if '__pycache__' not in f
]

# Data Collection
datas = [
    (os.path.join(root_dir, 'exo/tinychat'), 'exo/tinychat'),
    (os.path.join(root_dir, 'exo'), 'exo'),
    (local_python_lib, '.'),
]

# Collect Transformers Data
print("Collecting transformers data...")
try:
    trans_datas, _, _ = collect_all('transformers')
    filtered_datas = [(src, dst) for src, dst in trans_datas 
                     if not any(x in dst.lower() for x in ['.git', 'test', 'examples'])]
    datas.extend(filtered_datas)
    datas.extend(copy_metadata('transformers'))
except Exception as e:
    print(f"Warning: Could not collect transformers data: {e}")

# MLX Integration
if sys.platform.startswith('darwin'):
    print("Configuring macOS specific settings...")
    mlx_locations = [
        '/opt/homebrew/Caskroom/miniconda/base/envs/exo/lib/python3.12/site-packages/mlx',
        os.path.join(root_dir, 'venv/lib/python3.12/site-packages/mlx'),
        os.path.join(python_prefix, 'lib/python3.12/site-packages/mlx'),  # Added new search path
    ]
    
    mlx_path = None
    for loc in mlx_locations:
        if os.path.exists(loc):
            mlx_path = os.path.abspath(loc)
            print(f"Found MLX at: {mlx_path}")
            break

    if mlx_path:
        datas.append((mlx_path, 'mlx'))
        # Search for metallib in multiple possible locations
        metallib_locations = [
            os.path.join(mlx_path, 'backend/metal/kernels/mlx.metallib'),
            os.path.join(mlx_path, 'mlx/backend/metal/kernels/mlx.metallib'),
            os.path.join(python_prefix, 'lib/python3.12/site-packages/mlx/backend/metal/kernels/mlx.metallib'),
        ]
        
        metallib_found = False
        for metallib_path in metallib_locations:
            if os.path.exists(metallib_path):
                print(f"Found metallib at: {metallib_path}")
                # Add metallib to both the root and the MLX directory structure
                datas.extend([
                    (metallib_path, '.'),
                    (metallib_path, 'mlx/backend/metal/kernels')
                ])
                metallib_found = True
                break
                
        if not metallib_found:
            print("ERROR: Could not find mlx.metallib in any expected location!")
            print("Searched locations:", "\n".join(metallib_locations))
            sys.exit(1)
    else:
        print("ERROR: MLX package not found in expected locations")
        sys.exit(1)

# Initial binaries list with Python library
binaries = []
if sys.platform.startswith('darwin'):
    binaries.append((local_python_lib, '.'))

# Analysis Configuration
a = Analysis(
    [os.path.join(root_dir, 'exo/main.py')],
    pathex=[root_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'transformers',
        'safetensors',
        'safetensors.torch',
        'exo',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
        'packaging.markers',
        'charset_normalizer',
        'requests',
        'urllib3',
        'certifi',
        'idna',
        'mlx',
        'mlx.core',
        'mlx.nn',
        'mlx.backend',
        'mlx.backend.metal',
        'mlx.backend.metal.kernels',
        '_sysconfigdata__darwin_darwin',
    ] + model_imports,
    hookspath=[],
    hooksconfig={
        'urllib3': {'ssl': True},
        'transformers': {'module': True}
    },
    runtime_hooks=[],
    excludes=[
        'pytest',
        'sentry_sdk'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Make sure Python library is included in both datas and binaries
a.datas = list(dict.fromkeys(a.datas + [(os.path.basename(local_python_lib), local_python_lib, 'DATA')]))
a.binaries = list(dict.fromkeys(a.binaries + [(os.path.basename(local_python_lib), local_python_lib, 'BINARY')]))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=False,  # Enable debug mode
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)
# Create the collection
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=name,
)
