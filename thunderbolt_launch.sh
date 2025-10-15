#!/bin/bash

# Kill any existing exo processes
pkill -f exo

echo "Starting Exo cluster on Thunderbolt network (192.168.5.x)..."

# Start mini1 on Thunderbolt IP
source .venv/bin/activate
exo --node-id mini1 --node-host 192.168.5.1 --node-port 50051 &
MINI1_PID=$!
echo "Started mini1 (PID: $MINI1_PID) on 192.168.5.1:50051"

# Start mini2 on Thunderbolt IP  
ssh mini2@192.168.5.2 "pkill -f exo; cd ~/Movies/exo && source .venv/bin/activate && export PATH='/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH' && exo --node-id mini2 --node-host 192.168.5.2 --node-port 50051" &
MINI2_PID=$!
echo "Started mini2 (PID: $MINI2_PID) on 192.168.5.2:50051"

echo ""
echo "Waiting for nodes to discover each other..."
sleep 5

echo ""
echo "Cluster should be running on Thunderbolt network:"
echo "  mini1 API: http://192.168.5.1:52415/v1/chat/completions"  
echo "  mini2 API: http://192.168.5.2:52415/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop"

# Trap Ctrl+C to kill both processes
trap "kill $MINI1_PID $MINI2_PID 2>/dev/null; ssh mini2@192.168.5.2 'pkill -f exo' 2>/dev/null" INT

wait