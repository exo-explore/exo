#!/usr/bin/env bash

if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed, proceeding with python3.12..."
    python3.12 -m venv .venv
else
    echo "The recommended version of Python to run exo with is Python 3.12, but $(python3 --version) is installed. Proceeding with $(python3 --version)"
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e .
