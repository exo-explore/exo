#!/bin/bash
# Wrapper script to run MoE test with proper environment

# Change to the exo directory
cd /Users/$(whoami)/Movies/exo

# Activate uv environment and run the test
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the Python script with all arguments
python test_moe_distributed.py "$@"