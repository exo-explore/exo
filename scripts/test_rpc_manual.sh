#!/bin/bash
# ==============================================================================
# Manual RPC Distributed Inference Test
# ==============================================================================
# This script documents the manual test commands to verify llama.cpp RPC
# distributed inference works BEFORE running through EXO.
#
# Run this to isolate whether issues are in EXO orchestration or llama.cpp itself.
# ==============================================================================

set -e

# Configuration - UPDATE THESE VALUES
WORKER_IP="192.168.1.51"       # Worker device IP
MASTER_IP="192.168.1.126"      # Master device IP  
RPC_PORT="60000"               # RPC port for worker
MODEL_PATH="$HOME/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
TENSOR_SPLIT="0.59,0.41"       # Adjust based on device RAM ratios

echo "=============================================="
echo "llama.cpp RPC Manual Test"
echo "=============================================="
echo ""
echo "Before running this test:"
echo "1. Ensure llama.cpp is built with GGML_RPC=ON on both devices"
echo "2. Ensure the model file exists on the master device"
echo "3. Run the WORKER commands first, then MASTER commands"
echo ""

# Detect if we're on Android/Termux
if [[ "$PREFIX" == *"termux"* ]] || [[ -d "/data/data/com.termux" ]]; then
    LLAMA_CPP_DIR="$HOME/llama.cpp/build/bin"
else
    LLAMA_CPP_DIR="$HOME/llama.cpp/build/bin"
fi

echo "=============================================="
echo "STEP 1: Start WORKER on $WORKER_IP"
echo "=============================================="
echo ""
echo "SSH into the worker device and run:"
echo ""
echo "  # Acquire wake lock (Android only)"
echo "  termux-wake-lock"
echo ""
echo "  # Set debug environment"
echo "  export GGML_RPC_DEBUG=1"
echo ""
echo "  # Start RPC server"
echo "  $LLAMA_CPP_DIR/rpc-server --host 0.0.0.0 --port $RPC_PORT"
echo ""
echo "=============================================="
echo "STEP 2: Verify Worker is Listening"
echo "=============================================="
echo ""
echo "From master device, test connectivity:"
echo ""
echo "  nc -zv $WORKER_IP $RPC_PORT"
echo ""
echo "Expected: Connection to $WORKER_IP $RPC_PORT port [tcp/*] succeeded!"
echo ""

echo "=============================================="
echo "STEP 3: Start MASTER on $MASTER_IP"
echo "=============================================="
echo ""
echo "Run on the master device:"
echo ""
echo "  # Set debug environment"
echo "  export GGML_RPC_DEBUG=1"
echo "  export LLAMA_LOG_VERBOSITY=0"
echo "  export LLAMA_LOG_TIMESTAMPS=1"
echo ""
echo "  # Start distributed llama-server"
echo "  $LLAMA_CPP_DIR/llama-server \\"
echo "    -m $MODEL_PATH \\"
echo "    --port 8080 \\"
echo "    --host 127.0.0.1 \\"
echo "    --rpc $WORKER_IP:$RPC_PORT \\"
echo "    --tensor-split $TENSOR_SPLIT \\"
echo "    --no-mmap \\"
echo "    --no-flash-attn \\"
echo "    -c 1024 \\"
echo "    --batch-size 128 \\"
echo "    --verbose"
echo ""

echo "=============================================="
echo "STEP 4: Verify Server Health"
echo "=============================================="
echo ""
echo "In another terminal on master, run:"
echo ""
echo "  # Watch health status"
echo "  watch -n 2 'curl -s http://127.0.0.1:8080/health | head -c 200'"
echo ""
echo "Expected: {\"status\":\"ok\"}"
echo ""
echo "If you see {\"error\":{\"code\":503,...}}, the model is still loading."
echo "Wait up to 10 minutes for large models over slow networks."
echo ""

echo "=============================================="
echo "STEP 5: Test Inference"
echo "=============================================="
echo ""
echo "Once health returns 200, test inference:"
echo ""
echo "  curl http://127.0.0.1:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo '      "model": "model",'
echo '      "messages": [{"role": "user", "content": "Hello!"}],'
echo '      "max_tokens": 50'
echo "    }'"
echo ""

echo "=============================================="
echo "Troubleshooting"
echo "=============================================="
echo ""
echo "If worker connection fails:"
echo "  - Check firewall: sudo iptables -L -n | grep $RPC_PORT"
echo "  - Verify port is listening: netstat -tlnp | grep $RPC_PORT"
echo "  - Test with different port if blocked"
echo ""
echo "If model loading hangs:"
echo "  - Check worker logs for tensor transfer activity"
echo "  - Verify network bandwidth: iperf3 -c $WORKER_IP"
echo "  - Try smaller model (Qwen2.5-0.5B is ~400MB)"
echo "  - Ensure --no-mmap is set"
echo ""
echo "If server crashes:"
echo "  - Check memory: free -h"
echo "  - Reduce tensor split for device with less RAM"
echo "  - Try Q3_K_M quantization instead of Q4_K_M"
echo ""

