#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate

# Check if running on macOS with Apple Silicon
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Installing with Apple Silicon support..."
    pip install -e ".[apple_silicon]"
else
    echo "Installing without Apple Silicon support..."
    pip install -e .
fi
