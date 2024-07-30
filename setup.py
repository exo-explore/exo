import sys

from setuptools import find_packages, setup

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
    "prometheus-client==0.20.0",
    "protobuf==5.27.1",
    "psutil==6.0.0",
    "pynvml==11.5.3",
    "requests==2.32.3",
    "rich==13.7.1",
    "safetensors==0.4.3",
    "tiktoken==0.7.0",
    "tokenizers==0.19.1",
    "tqdm==4.66.4",
    "transformers==4.41.2",
    "uuid==1.30",
    "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@639af3f823cf242a1945dc24183e52a9df0af2b7",
]

# Add macOS-specific packages if on Darwin (macOS)
if sys.platform.startswith("darwin"):
    install_requires.extend(
        [
            "mlx==0.16.0",
            "mlx-lm==0.16.1",
        ]
    )

extras_require = {
    "linting": [
        "pylint==3.2.6",
        "ruff==0.5.5",
        "mypy==1.11.0",
    ]
}

setup(
    name="exo",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
