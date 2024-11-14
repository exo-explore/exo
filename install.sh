#!/bin/bash

# Check if Python 3.12 is installed
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed, proceeding..."
else
    echo "This repo currently only works with Python 3.12, which is not installed on your machine. Please install Python 3.12 and try again."
    exit 1
fi

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
