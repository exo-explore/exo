#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinygrad',
      version='0.9.1',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.runtime.autogen', 'tinygrad.codegen', 'tinygrad.nn', 'tinygrad.renderer', 'tinygrad.engine',
                  'tinygrad.runtime', 'tinygrad.runtime.support', 'tinygrad.runtime.graph', 'tinygrad.shape'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=["numpy",
                        "pyobjc-framework-Metal; platform_system=='Darwin'",
                        "pyobjc-framework-libdispatch; platform_system=='Darwin'"],
      python_requires='>=3.8',
      extras_require={
        'llvm': ["llvmlite"],
        'arm': ["unicorn"],
        'triton': ["triton-nightly>=2.1.0.dev20231014192330"],
        'linting': [
            "pylint",
            "mypy",
            "typing-extensions",
            "pre-commit",
            "ruff",
            "types-tqdm",
        ],
        #'mlperf': ["mlperf-logging @ git+https://github.com/mlperf/logging.git@4.0.0-rc2"],
        'testing': [
            "torch",
            "pillow",
            "pytest",
            "pytest-xdist",
            "onnx==1.16.0",
            "onnx2torch",
            "opencv-python",
            "tabulate",
            "tqdm",
            "safetensors",
            "transformers",
            "sentencepiece",
            "tiktoken",
            "blobfile",
            "librosa",
            "networkx",
            "hypothesis",
            "nibabel",
            "bottle",
        ],
        'docs': [
            "mkdocs-material",
            "mkdocstrings[python]",
            "markdown-callouts",
            "markdown-exec[ansi]",
            "black"
        ],
        'testing_tf': [
            "tensorflow==2.15.1",
            "tensorflow_addons",
        ]
      },
      include_package_data=True)
