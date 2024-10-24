#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
# ggml has to be installed seperately due to shared library dependency
pip install ggml-python --config-settings=cmake.args='-DGGML_CUDA=ON;-DGGML_METAL=ON'
pip install -e .
