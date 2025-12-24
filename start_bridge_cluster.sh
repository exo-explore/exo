#!/bin/bash
# Start Exo P2P Bridge Cluster
# This script starts the bridge on all 3 nodes using FIXED ports

set -e

# Configuration
BRIDGE_SCRIPT="/home/sophia/exo_cluster/exo_p2p_bridge.py"

# Node 1: localhost (4x V100)
echo "Starting bridge on localhost (port 7000)..."
python3 "$BRIDGE_SCRIPT" \
    --fixed-port 7000 \
    --log-file ~/.cache/exo/exo.log \
    > /tmp/exo_bridge_localhost.log 2>&1 &

echo "Bridge PID: $!"

# Node 2: .106 (RTX 5070) - via SSH
echo "Starting bridge on .106 (port 7001)..."
sshpass -p "Elyanlabs12@" ssh sophia5070node@192.168.0.106 \
    "cd /home/sophia5070node/exo_cluster && \
     nohup python3 exo_p2p_bridge.py \
     --fixed-port 7001 \
     --log-file /tmp/exo_106_new.log \
     > /tmp/exo_bridge_106.log 2>&1 &"

# Node 3: .134 (M2 Mac) - via SSH
echo "Starting bridge on .134 M2 Mac (port 7002)..."
sshpass -p "Elyanlabs12@" ssh sophimac@192.168.0.134 \
    "cd ~/exo_cluster && \
     nohup python3 exo_p2p_bridge.py \
     --fixed-port 7002 \
     --log-file /tmp/exo.log \
     > /tmp/exo_bridge_134.log 2>&1 &"

echo ""
echo "✅ All bridges started!"
echo ""
echo "Bridge ports:"
echo "  localhost: 7000 -> (dynamic Exo port)"
echo "  .106:      7001 -> (dynamic Exo port)"
echo "  .134:      7002 -> (dynamic Exo port)"
echo ""
echo "Monitor logs:"
echo "  localhost: tail -f /tmp/exo_bridge_localhost.log"
echo "  .106:      ssh sophia5070node@192.168.0.106 'tail -f /tmp/exo_bridge_106.log'"
echo "  .134:      ssh sophimac@192.168.0.134 'tail -f /tmp/exo_bridge_134.log'"
echo ""
echo "Now update STATIC_PEERS in discovery.rs to use fixed ports 7000/7001/7002"
