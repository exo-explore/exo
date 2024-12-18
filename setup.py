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
  "opencv-python==4.10.0.84",
  "opentelemetry-api==1.29.0",
  "opentelemetry-sdk==1.29.0",
  "opentelemetry-exporter-otlp==1.29.0",
  "opentelemetry-instrumentation==0.50b0",
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
  "uvloop==0.21.0",
  "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@3b26e51fcebfc6576f4e0f99693e6f1406d61d79",
]

extras_require = {
  "formatting": [
    "yapf==0.40.2",
  ],
  "apple_silicon": [
    "mlx==0.21.1",
    "mlx-lm==0.20.4",
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
