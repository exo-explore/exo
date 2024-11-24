import sys
import platform

from setuptools import find_packages, setup

# Base requirements for all platforms
install_requires = [
  "aiohttp==3.10.11",
  "aiohttp_cors==0.7.0",
  "aiofiles==24.1.0",
  "grpcio==1.68.0",
  "grpcio-tools==1.68.0",
  "Jinja2==3.1.4",
  "netifaces==0.11.0",
  "numpy==2.0.0",
  "nuitka==2.5.1",
  "nvidia-ml-py==12.560.30",
  "pillow==10.4.0",
  "prometheus-client==0.20.0",
  "protobuf==5.28.1",
  "psutil==6.0.0",
  "pydantic==2.9.2",
  "requests==2.32.3",
  "rich==13.7.1",
  "tenacity==9.0.0",
  "tqdm==4.66.4",
  "transformers==4.46.3",
  "uuid==1.30",
  "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@3b26e51fcebfc6576f4e0f99693e6f1406d61d79",
  "torch==2.4.0",
  "accelerate==0.34.2",
  "torchtune==0.4.0",
  "torchao==0.6.1",
  "pytest==8.3.3",
  "pytest-asyncio==0.24.0"
]

extras_require = {
  "formatting": [
    "yapf==0.40.2",
  ],
  "apple_silicon": [
    "mlx==0.20.0",
    "mlx-lm==0.19.3",
  ],
}

# Check if running on macOS with Apple Silicon
if sys.platform.startswith("darwin") and platform.machine() == "arm64":
  install_requires.extend(extras_require["apple_silicon"])

setup(
  name="exo",
  version="0.0.1",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={"exo": ["tinychat/**/*"]},
  entry_points={"console_scripts": ["exo = exo.main:run"]},
)
