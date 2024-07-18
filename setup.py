from setuptools import setup, find_packages
import sys

# Base requirements for all platforms
install_requires = [
    "aiohttp==3.9.5",
    "aiohttp_cors==0.7.0",
    "blobfile==2.1.1",
    "grpcio==1.64.1",
    "grpcio-tools==1.64.1",
    "huggingface-hub==0.23.4",
    "Jinja2==3.1.4",
    "numpy==2.0.0",
    "protobuf==5.27.1",
    "psutil==6.0.0",
    "pynvml==11.5.3",
    "requests==2.32.3",
    "safetensors==0.4.3",
    "tiktoken==0.7.0",
    "tokenizers==0.19.1",
    "tqdm==4.66.4",
    "transformers==4.41.2",
    "uuid==1.30",
    "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@a9f5a764dc640a5e5cbaaeeee21df7c8ca37da38",
]

# Add macOS-specific packages if on Darwin (macOS)
if sys.platform.startswith("darwin"):
    install_requires.extend(
        [
            "mlx==0.15.1",
            "mlx-lm==0.14.3",
        ]
    )


setup(
    name="exo",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
)
