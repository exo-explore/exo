#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 3-node M4 cluster.

# ── Tunable defaults (override via environment before running) ──
: "${EXO_FAST_SYNCH:=1}"
: "${EXO_DISABLE_METAL_TIMEOUT:=1}"
: "${EXO_LIBP2P_NAMESPACE:=MAC_STUDIO_CLUSTER}"
: "${EXO_PP_DRAFT_MODEL:=$HOME/.exo/models/mlx-community--Qwen3.5-0.8B-MLX-8bit}"
: "${LOG_LEVEL:=DEBUG}"

export IBV_FORK_SAFE=1
export PYTHONUNBUFFERED=1

# Define Node Constants
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"
MBP_IP="192.168.86.203"
MBP_PEER_ID="12D3KooWGtRYJcQpFLQBc3AFbES1A3BrFy55GyNLMNLNm64bHv16"

# Get current IPs (check all interfaces to correctly identify the node)
CURRENT_IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

IS_M4_1=false
IS_M4_2=false
IS_MBP=false

for IP in $CURRENT_IPS; do
    if [ "$IP" == "$M4_1_IP" ]; then
        IS_M4_1=true
        break
    fi
    if [ "$IP" == "$M4_2_IP" ]; then
        IS_M4_2=true
        break
    fi
    if [ "$IP" == "$MBP_IP" ]; then
        IS_MBP=true
        break
    fi
done

if [ "$IS_M4_1" = true ]; then
    echo "Detected M4-1 ($M4_1_IP)"
    # Peer with M4-2
    export EXO_DISCOVERY_PEERS="/ip4/$M4_2_IP/tcp/52415/p2p/$M4_2_PEER_ID"
elif [ "$IS_M4_2" = true ]; then
    echo "Detected M4-2 ($M4_2_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
elif [ "$IS_MBP" = true ]; then
    echo "Detected MacBook Pro ($MBP_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
else
    echo "Unknown host IPs: $CURRENT_IPS. Running as remote controller."
fi

# Full cluster setup — always runs regardless of which machine launches the script
echo "Starting cluster setup..."
echo "-----------------------------------------------------"

# Define nodes to start (using SSH config aliases)
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")

    # Thunderbolt Connectivity Check
    echo "Discovering active Thunderbolt IPs..."

get_node_tb_ips() {
    local node=$1
    # 1. Ask the node for its Thunderbolt device names (e.g., en1, en2)
    local devices=$(ssh "$node" "networksetup -listallhardwareports" | awk '/Hardware Port: Thunderbolt/{getline; print $2}')
    
    # 2. Iterate through them locally, asking the node about each one individually
    for dev in $devices; do
        if ssh "$node" "ifconfig $dev" 2>/dev/null | grep -q "status: active"; then
            ssh "$node" "ifconfig $dev" | awk '/inet / && !/127\.0\.0\.1/{print $2}'
        fi
    done
}

find_shared_ip() {
    local target_ips=$1
    local peer_ips=$2
    for tip in $target_ips; do
        local t_subnet=$(echo "$tip" | awk -F. '{print $1"."$2"."$3}')
        for pip in $peer_ips; do
            local p_subnet=$(echo "$pip" | awk -F. '{print $1"."$2"."$3}')
            if [ "$t_subnet" == "$p_subnet" ]; then
                echo "$tip"
                return 0
            fi
        done
    done
    return 1
}

echo "Fetching active Thunderbolt IPs from all nodes..."
TB_M4_1_IPS=$(get_node_tb_ips "macstudio-m4-1")
TB_M4_2_IPS=$(get_node_tb_ips "macstudio-m4-2")
TB_MBP_IPS=$(get_node_tb_ips "macbook-m4")

# Match IPs by their shared broadcast domains
M4_1_TO_M4_2=$(find_shared_ip "$TB_M4_1_IPS" "$TB_M4_2_IPS")
M4_1_TO_MBP=$(find_shared_ip "$TB_M4_1_IPS" "$TB_MBP_IPS")

M4_2_TO_M4_1=$(find_shared_ip "$TB_M4_2_IPS" "$TB_M4_1_IPS")
M4_2_TO_MBP=$(find_shared_ip "$TB_M4_2_IPS" "$TB_MBP_IPS")

MBP_TO_M4_1=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_1_IPS")
MBP_TO_M4_2=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_2_IPS")

echo "macstudio-m4-1 routes: -> M4-2 ($M4_1_TO_M4_2), -> MBP ($M4_1_TO_MBP)"
echo "macstudio-m4-2 routes: -> M4-1 ($M4_2_TO_M4_1), -> MBP ($M4_2_TO_MBP)"
echo "macbook-m4   routes: -> M4-1 ($MBP_TO_M4_1), -> M4-2 ($MBP_TO_M4_2)"

# Verify all 6 connection points were successfully discovered
if [ -z "$M4_1_TO_M4_2" ] || [ -z "$M4_1_TO_MBP" ] || [ -z "$M4_2_TO_M4_1" ] || [ -z "$M4_2_TO_MBP" ] || [ -z "$MBP_TO_M4_1" ] || [ -z "$MBP_TO_M4_2" ]; then
    echo "CRITICAL ERROR: Could not map a full 3-way Thunderbolt mesh subnet topology!"
    exit 1
fi

# Validate each node has 2 active Thunderbolt interfaces (catches loose cables)
echo "Verifying direct Thunderbolt links..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    active_count=$(echo "$(get_node_tb_ips "$node")" | grep -c '.')
    if [ "$active_count" -lt 2 ]; then
        echo "CRITICAL ERROR: $node has only $active_count active Thunderbolt interface(s) — expected 2."
        echo "Check physical Thunderbolt cable connections!"
        exit 1
    fi
    echo "  $node: $active_count active TB interfaces ✓"
done

# Direct-link pings — clear any stale cross-subnet routes from previous runs first,
# then ping. Without routes, pings can only succeed over direct physical links — no relay.
echo "Testing direct-link connectivity (clearing stale routes first)..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$node" "for r in \$(netstat -rn | awk '/192\.168\.(200|201|202)\./{print \$1}' | sort -u); do sudo route delete -net \$r 2>/dev/null; done" &> /dev/null
done

# M4-1 ↔ M4-2 (direct link)
if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $M4_2_TO_M4_1" &> /dev/null; then echo "ERROR: macstudio-m4-1 cannot directly reach M4-2 ($M4_2_TO_M4_1). Check cable!"; exit 1; fi
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $M4_1_TO_M4_2" &> /dev/null; then echo "ERROR: macstudio-m4-2 cannot directly reach M4-1 ($M4_1_TO_M4_2). Check cable!"; exit 1; fi

# M4-1 ↔ MBP (direct link)
if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $MBP_TO_M4_1" &> /dev/null; then echo "ERROR: macstudio-m4-1 cannot directly reach MBP ($MBP_TO_M4_1). Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_1_TO_MBP" &> /dev/null; then echo "ERROR: macbook-m4 cannot directly reach M4-1 ($M4_1_TO_MBP). Check cable!"; exit 1; fi

# M4-2 ↔ MBP (direct link)
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $MBP_TO_M4_2" &> /dev/null; then echo "ERROR: macstudio-m4-2 cannot directly reach MBP ($MBP_TO_M4_2). Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_2_TO_MBP" &> /dev/null; then echo "ERROR: macbook-m4 cannot directly reach M4-2 ($M4_2_TO_MBP). Check cable!"; exit 1; fi

echo "All 6 direct Thunderbolt links verified ✓"

# RoCEv2 (RDMA) Per-Device Port State Check
# Only checks RDMA ports corresponding to TB interfaces that have active IPs (in use).
# PORT_DOWN on unconnected TB ports is expected and harmless.
echo "Checking per-device RDMA port states (active TB interfaces only)..."
RDMA_HEALTHY=true
for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    echo -n "  $NODE: "
    # Get only the RDMA devices whose underlying interface has an active IP
    PORT_STATUS=$(ssh "$NODE" 'for dev in rdma_en1 rdma_en2 rdma_en3 rdma_en4 rdma_en5; do
        iface=${dev#rdma_}
        # Only check if this interface has a 192.168.x.x IP (active TB link)
        ip=$(ifconfig "$iface" 2>/dev/null | awk "/inet / && /192\.168\./{print \$2}")
        if [ -n "$ip" ]; then
            state=$(ibv_devinfo -d $dev 2>/dev/null | grep "state:" | head -1 | awk "{print \$2}")
            if [ -n "$state" ]; then
                echo -n "$dev($iface=$ip)=$state "
            fi
        fi
    done' 2>/dev/null)

    if [ -z "$PORT_STATUS" ]; then
        echo "no active RDMA devices found"
        RDMA_HEALTHY=false
    else
        echo "$PORT_STATUS"
        if echo "$PORT_STATUS" | grep -q "PORT_DOWN"; then
            echo "    ERROR: PORT_DOWN on an active TB interface on $NODE! Check cable."
            RDMA_HEALTHY=false
        fi
    fi
done

if [ "$RDMA_HEALTHY" = false ]; then
    echo ""
    if [ "${SKIP_RDMA_PORT_CHECK:-0}" = "1" ]; then
        echo "WARNING: RDMA ports are DOWN but SKIP_RDMA_PORT_CHECK=1 is set. Continuing (fresh boot assumed)."
    else
        echo "ERROR: One or more active-link RDMA ports are DOWN. Fix cables and retry."
        exit 1
    fi
else
    echo "  All active-link RDMA ports active \u2713"
fi

# RoCEv2 (RDMA) Protection Domain Allocation Check
# A degraded Thunderbolt cable will pass `ping` (using USB-C fallback Ethernet), but fail to allocate
# an RDMA Protection Domain, causing `jaccl` to instantly crash when Exo starts.
echo "Verifying RoCEv2 (RDMA) support over Thunderbolt..."
for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    echo -n "  Testing RDMA allocation on $NODE... "
    # We use `timeout 2` because a successful PD allocation will hang waiting for a coordinator.
    # We run it within the uv environment to ensure mlx is available.
    # If the Thunderbolt cable is degraded, allocating the Protection Domain crashes immediately.
    RDMA_CHECK=$(ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && timeout 2 uv run python -c \"import mlx.core as mx; mx.distributed.init(strict=False, backend='\\''jaccl'\\'')\" 2>&1'" || true)
    
    if echo "$RDMA_CHECK" | grep -q "Couldn't allocate protection domain"; then
        echo "FAIL ❌"
        echo "CRITICAL ERROR: Failed to allocate RDMA Protection Domain on $NODE!"
        echo "One of your Thunderbolt cables has fallen back to standard USB-C Ethernet."
        echo "Please re-seat the cables on $NODE."
        exit 1
    else
        echo "OK ✓"
    fi
done

# Enable IP forwarding and add cross-subnet routes
# Each node has 2 direct links, but needs a route for the 3rd subnet it's not on.
echo "Enabling IP forwarding and configuring cross-subnet routes..."

SUBNET_M4_1_M4_2=$(echo "$M4_1_TO_M4_2" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_1_MBP=$(echo "$M4_1_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_2_MBP=$(echo "$M4_2_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')

for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$NODE" "sudo sysctl -w net.inet.ip.forwarding=1" &> /dev/null
done

# M4-1 is NOT on the M4-2<->MBP subnet → route via M4-2
ssh macstudio-m4-1 "sudo route delete -net $SUBNET_M4_2_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_2_MBP $M4_2_TO_M4_1" &> /dev/null
# M4-2 is NOT on the M4-1<->MBP subnet → route via M4-1
ssh macstudio-m4-2 "sudo route delete -net $SUBNET_M4_1_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_1_MBP $M4_1_TO_M4_2" &> /dev/null
# MBP is NOT on the M4-1<->M4-2 subnet → route via M4-1
ssh macbook-m4 "sudo route delete -net $SUBNET_M4_1_M4_2 2>/dev/null; sudo route add -net $SUBNET_M4_1_M4_2 $M4_1_TO_MBP" &> /dev/null

echo "Cross-subnet routes configured."

# 0. Pre-deploy push check — verify local HEAD is on origin/main
echo "Verifying local commits are pushed to origin/main..."
LOCAL_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "none")
git fetch origin --quiet 2>/dev/null || true
ORIGIN_MAIN=$(git rev-parse origin/main 2>/dev/null || echo "none")
if [ "$LOCAL_HEAD" = "none" ]; then
    echo "WARNING: Not in a git repo on controller. Skipping push check."
elif [ "$ORIGIN_MAIN" = "none" ]; then
    echo "WARNING: Could not fetch origin/main. Skipping push check."
elif ! git merge-base --is-ancestor "$LOCAL_HEAD" origin/main 2>/dev/null; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  WARNING: Local HEAD is NOT on origin/main!                  ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Local HEAD:   $LOCAL_HEAD"
    echo "║  origin/main:  $(git rev-parse --short origin/main)"
    echo "║                                                              ║"
    echo "║  Cluster nodes will reset to origin/main, so your local      ║"
    echo "║  commits will NOT be deployed. Push first:                   ║"
    echo "║    git push origin main                                      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "  Local HEAD ($LOCAL_HEAD) is on origin/main ✓"
fi

# 1. Cleanup, Update, and Build
for NODE in "${NODES[@]}"; do
    echo "Preparing $NODE..."
    echo "Setting Metal memory limit on $NODE..."
    if [[ "$NODE" == *"macbook"* ]]; then
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=32000"
    else
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=124000"
    fi
    
    echo "Killing existing Exo processes on $NODE..."
    for i in {1..5}; do
        ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
        ssh "$NODE" "pkill -9 -f 'exo.main' || true"
        ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
        
        if ssh "$NODE" "pgrep -f 'exo.main'" > /dev/null; then
            sleep 1
        else
            break
        fi
    done
    
    ssh "$NODE" "screen -wipe || true"

    echo "Ensuring Xcode developer directory on $NODE..."
    ssh "$NODE" "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer || true"
    
    # Update and Build Logic
    TARGET_BRANCH="main"
    ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && git fetch origin && git reset --hard && git checkout $TARGET_BRANCH && git reset --hard origin/$TARGET_BRANCH'" || { echo "Failed to update repo on $NODE"; exit 1; }
    
    echo "Ensuring build dependencies on $NODE..."
    ssh "$NODE" "/opt/homebrew/bin/brew install cmake 2>/dev/null || true"

    # Sync dependencies (mlx and mlx-lm are pulled from git via uv sources)
    echo "Syncing dependencies on $NODE..."
    ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv sync'" || { echo "Failed to sync on $NODE"; exit 1; }

    # Rebuild Rust pyo3 bindings from source (uv sync installs a stale pre-compiled version)
    echo "Rebuilding Rust pyo3 bindings on $NODE..."
    ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && uv pip install maturin 2>/dev/null && uv run maturin develop --release -m rust/exo_pyo3_bindings/Cargo.toml'" || { echo "Failed to rebuild Rust bindings on $NODE"; exit 1; }


    echo "Building dashboard on $NODE..."
    ssh "$NODE" "zsh -l -c 'source ~/.zshrc; cd ~/repos/exo/dashboard && npm install && npm run build'" || { echo "Failed to build dashboard on $NODE"; exit 1; }

done

# 2. Inter-Node Git Sync Check (M4-1 vs M4-2 vs MBP)
echo "Verifying commit consistency between nodes..."
COMMIT_M4_1=$(ssh macstudio-m4-1 "cd ~/repos/exo && git rev-parse --short HEAD")
COMMIT_M4_2=$(ssh macstudio-m4-2 "cd ~/repos/exo && git rev-parse --short HEAD")
COMMIT_MBP=$(ssh macbook-m4 "cd ~/repos/exo && git rev-parse --short HEAD")

if [ "$COMMIT_M4_1" != "$COMMIT_M4_2" ] || [ "$COMMIT_M4_1" != "$COMMIT_MBP" ]; then
    echo "CRITICAL ERROR: Cluster out of sync!"
    echo "macstudio-m4-1: $COMMIT_M4_1"
    echo "macstudio-m4-2: $COMMIT_M4_2"
    echo "macbook-m4: $COMMIT_MBP"
    exit 1
fi
echo "Nodes synchronized on commit $COMMIT_M4_1."

# 3. Start Exo on each node
for NODE in "${NODES[@]}"; do
    echo "Starting Exo on $NODE..."
    
    # Build the node environment string
    EXO_ENV="PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 IBV_FORK_SAFE=1"
    EXO_ENV="$EXO_ENV EXO_LIBP2P_NAMESPACE=$EXO_LIBP2P_NAMESPACE"
    EXO_ENV="$EXO_ENV EXO_FAST_SYNCH=$EXO_FAST_SYNCH"
    EXO_ENV="$EXO_ENV EXO_PP_DRAFT_MODEL=$EXO_PP_DRAFT_MODEL"
    EXO_ENV="$EXO_ENV LOG_LEVEL=$LOG_LEVEL"

    # Metal GPU timeout mitigations
    if [ "$EXO_DISABLE_METAL_TIMEOUT" == "1" ]; then
        EXO_ENV="$EXO_ENV MTL_DISABLE_TIMEOUT=1 MTL_COMMAND_BUFFER_TIMEOUT=0 EXO_DISABLE_METAL_TIMEOUT=1"
    fi


    # Rotate log: keep previous run as exo.log.prev, then append
    ssh "$NODE" "cp ~/exo.log ~/exo.log.prev 2>/dev/null; : > ~/exo.log"

    if [ "$NODE" == "macstudio-m4-1" ]; then
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_2_TO_M4_1/tcp/52415/p2p/$M4_2_PEER_ID .venv/bin/python -m exo >> ~/exo.log 2>&1'"
    elif [ "$NODE" == "macstudio-m4-2" ]; then
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_TO_M4_2/tcp/52415/p2p/$M4_1_PEER_ID .venv/bin/python -m exo >> ~/exo.log 2>&1'"
    else
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_TO_MBP/tcp/52415/p2p/$M4_1_PEER_ID .venv/bin/python -m exo >> ~/exo.log 2>&1'"
    fi
done

# 4. Health Check / Topology Verification
# Wait for all 3 nodes AND their identities (friendlyName) to be populated.
API="http://$M4_1_IP:52415"

echo -n "Waiting for cluster to stabilize..."
CLUSTER_READY=false
for i in {1..90}; do
    response=$(curl -s "$API/state")
    node_count=$(echo "$response" | jq '.topology.nodes | length' 2>/dev/null)
    identity_count=$(echo "$response" | jq '.nodeIdentities | length' 2>/dev/null)

    # Handle null or empty counts
    if [ -z "$node_count" ] || [ "$node_count" == "null" ]; then node_count=0; fi
    if [ -z "$identity_count" ] || [ "$identity_count" == "null" ]; then identity_count=0; fi

    if [ "$node_count" -ge 3 ] && [ "$identity_count" -ge 3 ]; then
        echo " HEALTHY! (Nodes: $node_count, Identities: $identity_count)"
        CLUSTER_READY=true
        break
    fi
    echo -n "."
    sleep 2
done

if [ "$CLUSTER_READY" = false ]; then
    echo ""
    # Check for the specific pyo3 initialization panic that happens when uv.lock goes out of sync
    PYO3_PANIC=$(ssh macstudio-m4-1 "grep -i 'The Python interpreter is not initialized' ~/exo.log" 2>/dev/null || true)

    if [ -n "$PYO3_PANIC" ]; then
        echo "CRITICAL ERROR: Detected a corrupted Rust pyo3 binding state on the primary node!"
        echo "This usually happens when 'uv.lock' changes (e.g. from switching git branches) and the virtual environment gets out of sync."
        echo ""
        echo "AUTOMATIC FIX: Run the following command on ALL nodes to repair the bindings:"
        echo "  zsh -l -c 'cd ~/repos/exo && uv sync --reinstall-package exo_pyo3_bindings'"
        echo ""
        echo "Exiting."
        exit 1
    fi

    echo "TIMEOUT: Cluster did not stabilize."
    echo "Fetching logs from macstudio-m4-1:"
    ssh macstudio-m4-1 "tail -n 20 ~/exo.log"
    exit 1
fi


# 5. Create MiniMax instance on the Mac Studios

# Look up Mac Studio node IDs from cluster state (these differ from libp2p peer IDs).
echo "Looking up node IDs from cluster state..."
M4_1_NODE_ID=""
M4_2_NODE_ID=""
MBP_NODE_ID=""
for i in {1..15}; do
    NODE_STATE=$(curl -s "$API/state")
    M4_1_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("Studio.*M4-1")) | .key')
    M4_2_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("Studio.*M4-2")) | .key')
    MBP_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("MacBook")) | .key')

    if [ -n "$M4_1_NODE_ID" ] && [ -n "$M4_2_NODE_ID" ] && [ -n "$MBP_NODE_ID" ]; then
        break
    fi
    echo "  Waiting for node identities to propagate..."
    sleep 2
done

echo "  Mac Studio M4-1: $M4_1_NODE_ID"
echo "  Mac Studio M4-2: $M4_2_NODE_ID"
echo "  MacBook Pro:     $MBP_NODE_ID"

if [ -z "$M4_1_NODE_ID" ] || [ -z "$M4_2_NODE_ID" ] || [ -z "$MBP_NODE_ID" ]; then
    echo "ERROR: Could not resolve all node IDs. Skipping instance creation."
    echo "Create instances manually from the dashboard."
    exit 1
fi

place_instance_with_retry() {
    local label="$1"
    local model_id="$2"
    local payload="$3"
    local expected_count="${4:-1}"  # how many instances of this model_id are expected
    local max_attempts=20
    for attempt in $(seq 1 $max_attempts); do
        # Before each attempt, check if enough instances for this model already exist
        local existing
        existing=$(curl -s "$API/state" | jq -r --arg m "$model_id" \
            '[.. | objects | select(has("shardAssignments")) | select(.shardAssignments.modelId == $m)] | length' 2>/dev/null)
        if [ -n "$existing" ] && [ "$existing" != "null" ] && [ "$existing" -ge "$expected_count" ] 2>/dev/null; then
            echo "  Instance for $label already exists ($existing/$expected_count), skipping."
            return 0
        fi

        local http_code raw_response
        raw_response=$(curl -s -w '\n%{http_code}' -X POST "$API/place_instance" \
            -H "Content-Type: application/json" \
            -d "$payload")
        http_code=$(echo "$raw_response" | tail -1)
        raw_response=$(echo "$raw_response" | sed '$d')

        # Parse message from response (handles both top-level and nested error formats)
        local msg
        msg=$(echo "$raw_response" | jq -r '.message // .error.message // .detail // empty' 2>/dev/null)

        # Success
        if [ -n "$http_code" ] && [ "$http_code" -ge 200 ] 2>/dev/null && [ "$http_code" -lt 300 ] 2>/dev/null; then
            echo "  ${msg:-OK}"
            return 0
        fi

        # These 400s happen when node_memory/node_network/RDMA aren't populated yet — retryable
        if echo "$msg" | grep -qi "sufficient memory\|devices to be able to communicate\|RDMA connections\|jaccl backend\|No running instance found for draft model\|No tensor sharding found"; then
            if [ "$attempt" -lt "$max_attempts" ]; then
                echo "  Attempt $attempt/$max_attempts: cluster state still propagating, retrying in 5s..."
                sleep 5
                continue
            fi
        fi

        # Any other 4xx = real client error, don't retry
        if [ -n "$http_code" ] && [ "$http_code" -ge 400 ] 2>/dev/null && [ "$http_code" -lt 500 ] 2>/dev/null; then
            echo "  ERROR (HTTP $http_code): ${msg:-$raw_response}"
            return 1
        fi

        # 5xx or no response — API not ready, safe to retry
        if [ "$attempt" -lt "$max_attempts" ]; then
            echo "  Attempt $attempt/$max_attempts: API not ready (HTTP $http_code), retrying in 3s..."
            sleep 3
        else
            echo "  ERROR: Failed after $max_attempts attempts. HTTP $http_code: $raw_response"
            return 1
        fi
    done
}

EXPECTED_RUNNERS=0

# ── Instance 1: Primary model (2-node Pipeline Parallel over RDMA) ──
# MacBook excluded — OOMs at ~5K context with this model.
# TODO: implement remote KV cache so MacBook serves as RDMA-backed
# KV prefix cache server without running any transformer layers.
# Instance creation disabled — use the dashboard to load models manually.
# echo "Creating Qwen3.5-397B instance (Studios PP / RDMA)..."
# if place_instance_with_retry "Qwen3.5-397B" "mlx-community/Qwen3.5-397B-A17B-4bit" "{
#     \"model_id\": \"mlx-community/Qwen3.5-397B-A17B-4bit\",
#     \"sharding\": \"Pipeline\",
#     \"instance_meta\": \"MlxJaccl\",
#     \"min_nodes\": 2,
#     \"node_ids\": [\"$M4_1_NODE_ID\", \"$M4_2_NODE_ID\"],
#     \"max_context_tokens\": 262144
# }"; then
#     EXPECTED_RUNNERS=$((EXPECTED_RUNNERS + 2))
# fi
#
# echo "Creating Qwen3.5-35B instance (MacBook)..."
# if place_instance_with_retry "Qwen3.5-35B" "mlx-community/Qwen3.5-35B-A3B-4bit" "{
#     \"model_id\": \"mlx-community/Qwen3.5-35B-A3B-4bit\",
#     \"sharding\": \"Pipeline\",
#     \"min_nodes\": 1,
#     \"node_ids\": [\"$MBP_NODE_ID\"],
#     \"max_context_tokens\": 50000
# }"; then
#     EXPECTED_RUNNERS=$((EXPECTED_RUNNERS + 1))
# fi

echo "Cluster is up. Load models via the dashboard."
INSTANCES_READY=true

if false; then
# (dead code — kept for reference)
echo -n "Waiting for $EXPECTED_RUNNERS runner(s) to load..."
INSTANCES_READY=false
for i in {1..120}; do
    READY_COUNT=$(curl -s "$API/state" | jq '[.runners | to_entries[] | .value | to_entries[] | .key | select(. == "RunnerReady" or . == "RunnerRunning")] | length' 2>/dev/null || echo 0)
    if [ "$READY_COUNT" -ge "$EXPECTED_RUNNERS" ]; then
        echo " All instances ready! ($READY_COUNT runners)"
        INSTANCES_READY=true
        break
    fi
    echo -n "."
    sleep 3
done

if [ "$INSTANCES_READY" = false ]; then
    echo ""
    echo "WARNING: Only $READY_COUNT/$EXPECTED_RUNNERS runners loaded within timeout. Check the dashboard."
fi
fi  # end dead code block

# Final environment export
export IBV_FORK_SAFE=${IBV_FORK_SAFE:-1}
