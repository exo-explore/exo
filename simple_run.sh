#!/bin/bash

echo "Killing existing exo processes..."
pkill -f exo
ssh mini2@192.168.5.2 "pkill -f exo"

echo "Starting exo on mini1 (simplest way)..."
source .venv/bin/activate
exo &
MINI1_PID=$!

echo "Starting exo on mini2 (simplest way)..."
ssh mini2@192.168.5.2 "cd ~/Movies/exo && source .venv/bin/activate && /Users/mini2/Movies/exo/.venv/bin/exo" &
MINI2_PID=$!

sleep 5

echo ""
echo "Testing if nodes are running:"
curl -s http://192.168.5.1:52415/v1/models | jq -r '.data[0].id' 2>/dev/null && echo "✓ mini1 is running" || echo "✗ mini1 not ready"
curl -s http://192.168.5.2:52415/v1/models | jq -r '.data[0].id' 2>/dev/null && echo "✓ mini2 is running" || echo "✗ mini2 not ready"

echo ""
echo "API endpoints:"
echo "  http://192.168.5.1:52415/v1/chat/completions"
echo "  http://192.168.5.2:52415/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $MINI1_PID 2>/dev/null; ssh mini2@192.168.5.2 'pkill -f exo'" INT
wait