#!/bin/bash

# Force both nodes to use ONLY Thunderbolt network IPs

echo "Starting exo cluster on Thunderbolt network only..."

# mini1 - bind to Thunderbolt IP
source .venv/bin/activate
EXO_NODE_HOST=192.168.5.1 exo &

# mini2 - bind to Thunderbolt IP  
ssh mini2@192.168.5.2 "cd ~/Movies/exo && source .venv/bin/activate && PATH='/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH' EXO_NODE_HOST=192.168.5.2 exo" &

echo ""
echo "Nodes starting on Thunderbolt network..."
echo "They should auto-discover each other via UDP broadcast on 192.168.5.x"
echo ""
echo "WebUI: http://192.168.5.1:52415"
echo "API: http://192.168.5.1:52415/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop"

wait