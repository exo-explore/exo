#!/usr/bin/env bash

if [[ "$OSTYPE" == "msys" ]]; then
    echo "Detected Windows environment. Using python instead of python3.12..."
    if command -v python &>/dev/null; then
        echo "Python is installed, proceeding with python..."
        python -m venv .venv
    else
        echo "Python is not installed. Please install Python and try again."
        exit 1
    fi
else
    if command -v python3.12 &>/dev/null; then
        echo "Python 3.12 is installed, proceeding with python3.12..."
        python3.12 -m venv .venv
    else
        echo "The recommended version of Python to run exo with is Python 3.12, but $(python3 --version) is installed. Proceeding with $(python3 --version)"
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate
pip install -e .
