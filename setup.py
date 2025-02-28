import sys
import platform
import subprocess

from setuptools import find_packages, setup

# Base requirements for all platforms
install_requires = [
  "aiohttp==3.10.11",
  "aiohttp_cors==0.7.0",
  "aiofiles==24.1.0",
  "grpcio==1.67.0",
  "grpcio-tools==1.67.0",
  "Jinja2==3.1.4",
  "numpy==2.0.0",
  "nuitka==2.5.1",
  "nvidia-ml-py==12.560.30",
  "opencv-python==4.10.0.84",
  "pillow==10.4.0",
  "prometheus-client==0.20.0",
  "protobuf==5.28.1",
  "psutil==6.0.0",
  "pydantic==2.9.2",
  "requests==2.32.3",
  "rich==13.7.1",
  "scapy==2.6.1",
  "tqdm==4.66.4",
  "transformers==4.46.3",
  "uuid==1.30",
  "uvloop==0.21.0",
  "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@ec120ce6b9ce8e4ff4b5692566a683ef240e8bc8",
]

extras_require = {
  "formatting": ["yapf==0.40.2",],
  "apple_silicon": [
    "mlx==0.22.0",
    "mlx-lm==0.21.1",
  ],
  "windows": ["pywin32==308",],
  "nvidia-gpu": ["nvidia-ml-py==12.560.30",],
  "amd-gpu": ["pyrsmi==0.2.0"],
}

# Check if running on macOS with Apple Silicon
if sys.platform.startswith("darwin") and platform.machine() == "arm64":
  install_requires.extend(extras_require["apple_silicon"])

# Check if running Windows
if sys.platform.startswith("win32"):
  install_requires.extend(extras_require["windows"])


def _add_gpu_requires():
  global install_requires
  # Add Nvidia-GPU
  try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["nvidia-gpu"])
  except subprocess.CalledProcessError:
    pass

  # Add AMD-GPU
  # This will mostly work only on Linux, amd/rocm-smi is not yet supported on Windows
  try:
    out = subprocess.run(['amd-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["amd-gpu"])
  except:
    out = subprocess.run(['rocm-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["amd-gpu"])
  finally:
    pass


_add_gpu_requires()

setup(
  name="exo",
  version="0.0.1",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={"exo": ["tinychat/**/*"]},
  entry_points={"console_scripts": ["exo = exo.main:run"]},
)
