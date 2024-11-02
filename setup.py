import sys
from setuptools import find_packages, setup

# Base requirements for all platforms
install_requires = [
  "aiohttp==3.10.2",
  "aiohttp_cors==0.7.0",
  "aiofiles==24.1.0",
  "blobfile==2.1.1",
  "grpcio==1.64.1",
  "grpcio-tools==1.64.1",
  "hf-transfer==0.1.8",
  "huggingface-hub==0.24.5",
  "Jinja2==3.1.4",
  "netifaces==0.11.0",
  "numpy>=1.21.0,<1.26.0",
  "pillow==10.4.0",
  "prometheus-client==0.20.0",
  "protobuf==5.27.1",
  "psutil==6.0.0",
  "pynvml==11.5.3",
  "requests==2.32.3",
  "rich==13.7.1",
  "safetensors==0.4.3",
  "tenacity==9.0.0",
  "tiktoken==0.7.0",
  "tokenizers==0.19.1",
  "tqdm==4.66.4",
  "transformers==4.43.3",
  "uuid==1.30",
  "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@639af3f823cf242a1945dc24183e52a9df0af2b7",
  "fastapi==0.100.0",
  "uvicorn==0.23.0",
  "pandas==1.5.3",
  "scikit-learn==1.3.0",
]

# Add macOS-specific packages if on Darwin (macOS)
if sys.platform.startswith("darwin"):
  install_requires.extend([
    "mlx==0.17.1",
    "mlx-lm==0.17.0",
  ])

extras_require = {
  "linting": [
    "pylint==3.2.6",
    "ruff==0.5.5",
    "mypy==1.11.0",
    "yapf==0.40.2",
  ],
}

setup(
  name="exo",
  version="0.0.1",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
)
