#!/bin/bash

# SIMPLE EXO LAUNCH - No complex config needed!

echo "Starting Exo on mini1..."
source .venv/bin/activate
exo &
MINI1_PID=$!

echo "Starting Exo on mini2..."
ssh mini2@192.168.5.2 "cd ~/Movies/exo && source .venv/bin/activate && export PATH='/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH' && exo" &
MINI2_PID=$!

echo "Exo cluster starting..."
echo "mini1 PID: $MINI1_PID"
echo "mini2 PID: $MINI2_PID"
echo ""
echo "API Endpoints:"
echo "  mini1: http://localhost:52415"
echo "  mini2: http://192.168.5.2:52415"
echo ""
echo "Press Ctrl+C to stop"

wait