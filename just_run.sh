#!/bin/bash

# EXACTLY as the docs say - just run exo on both devices!

echo "Starting exo on mini1..."
source .venv/bin/activate
exo &

echo "Starting exo on mini2..."
ssh mini2@192.168.5.2 "cd ~/Movies/exo && source .venv/bin/activate && PATH='/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH' exo" &

echo ""
echo "Exo is starting - it should auto-discover peers"
echo "WebUI: http://localhost:52415"
echo "API: http://localhost:52415/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop"

wait