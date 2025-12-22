# Distributed LLM Inference on Android: Success Guide

> **How we achieved distributed llama.cpp inference across two Android phones**

This document captures the working setup for running distributed LLM inference using llama.cpp's RPC backend across multiple Android devices running Termux.

---

## Table of Contents

1. [Summary](#summary)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Verification](#verification)
6. [Key Learnings](#key-learnings)
7. [Performance Results](#performance-results)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

---

## Summary

We successfully ran a Qwen 2.5 0.5B model distributed across two Samsung Galaxy Z Fold phones:

| Device | Role | Model Portion | Function |
|--------|------|---------------|----------|
| Phone 1 (Hotspot Host) | MASTER | 89 MiB (19%) | Runs `llama-server`, coordinates inference |
| Phone 2 (Hotspot Client) | WORKER | 374 MiB (81%) | Runs `rpc-server`, computes tensors |

**Result:** Working chat completions API with both devices computing together.

---

## The Problem

### Initial Symptoms

When attempting distributed inference on a regular WiFi network (Eero + Unifi):

```bash
nc -zv 10.69.1.51 60000
# Result: Ncat: TIMEOUT.
```

Both devices could see each other on the network, but TCP connections to the RPC port timed out.

### Root Cause Analysis

1. **Not the router** - Unifi showed "Isolate Network" was OFF
2. **Not the Eero** - Eero was in bridge mode, no isolation settings
3. **The real cause: Android blocks incoming TCP connections**

Even when testing with mobile hotspot (bypassing all router/network equipment), the hotspot HOST device could not accept incoming connections:

```bash
# On device connected to hotspot, trying to reach hotspot host:
nc -zv 192.168.43.1 60000
# Result: Ncat: TIMEOUT.
```

But localhost worked fine on the device running the server:

```bash
nc -zv 127.0.0.1 60000
# Result: 127.0.0.1 (127.0.0.1:60000) open
```

**Conclusion:** Android's network stack blocks incoming TCP connections from external devices, even on the hotspot interface.

---

## The Solution

### The Key Insight

The device that can only make **outgoing** connections should be the **MASTER**.
The device that can **accept incoming** connections should be the **WORKER**.

### Hotspot Configuration

When using Android's mobile hotspot:

| Device | Hotspot Role | Can Accept Incoming? | llama.cpp Role |
|--------|--------------|---------------------|----------------|
| Hotspot Host | Provides WiFi | ❌ No | MASTER (`llama-server`) |
| Hotspot Client | Connects to hotspot | ✅ Yes | WORKER (`rpc-server`) |

The hotspot host can make outgoing connections to devices on its network, but cannot receive incoming connections. The device connected TO the hotspot can accept connections.

---

## Step-by-Step Setup

### Prerequisites

- Two Android devices with Termux installed
- llama.cpp built with RPC support (`-DGGML_RPC=ON`)
- A GGUF model file on the MASTER device

### Step 1: Enable Mobile Hotspot on MASTER Device

On the device that will be the MASTER:
1. Go to Android Settings → Hotspot & Tethering
2. Enable Mobile Hotspot
3. Note: This device will run `llama-server`

### Step 2: Connect WORKER to Hotspot

On the device that will be the WORKER:
1. Connect to the MASTER's hotspot WiFi
2. Get the assigned IP address:

```bash
ip addr show wlan0 | grep "inet "
# Example output: inet 192.168.37.219/24 ...
```

### Step 3: Start RPC Server on WORKER

On the WORKER device (connected to hotspot):

```bash
# Optional: Enable debug logging
export GGML_RPC_DEBUG=1

# Start the RPC server
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

Expected output:
```
Starting RPC server v3.6.0
  endpoint       : 0.0.0.0:60000
  local cache    : n/a
Devices:
  CPU: CPU (11358 MiB, 11358 MiB free)
```

### Step 4: Verify Connectivity from MASTER

On the MASTER device (hotspot host), test connectivity:

```bash
nc -zv 192.168.37.219 60000
# Expected: 192.168.37.219 (192.168.37.219:60000) open
```

### Step 5: Start Distributed llama-server on MASTER

On the MASTER device:

```bash
# Enable debug logging
export GGML_RPC_DEBUG=1
export LLAMA_LOG_VERBOSITY=0

# Start distributed llama-server
~/llama.cpp/build/bin/llama-server \
  -m ~/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc 192.168.37.219:60000 \
  --tensor-split 0.5,0.5 \
  --no-mmap \
  -c 1024 \
  --batch-size 128 \
  --verbose
```

### Step 6: Wait for Model Loading

You'll see:
- MASTER: Connection attempts and tensor transfer progress (`....`)
- WORKER: `Null buffer for tensor passed to init_tensor function` (repeated)

Eventually MASTER shows:
```
main: model loaded
main: server is listening on http://127.0.0.1:8080
```

---

## Verification

### Test Health Endpoint

On MASTER device, open another terminal:

```bash
curl http://127.0.0.1:8080/health
# Expected: {"status":"ok"}
```

### Test Chat Completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

Expected response:
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"2+2 equals 4."}}],...}
```

### Verify Worker Activity

With `GGML_RPC_DEBUG=1` on the WORKER, you'll see during inference:

```
[set_tensor] buffer: 0xb400007c9969efd0, data: 0xb400007bc87b9000, offset: 0, size: 3584
[graph_recompute] device: 0
[get_tensor] buffer: 0xb400007c9969efd0, data: 0xb400007bc8829000, offset: 0, size: 607744
```

The `[graph_recompute]` message confirms the WORKER is actively computing.

---

## Key Learnings

### 1. Android Blocks Incoming Connections

Android's network stack (likely iptables/netfilter rules) blocks incoming TCP connections by default. This is a security feature, not a bug.

### 2. Hotspot Asymmetry

- **Hotspot host** → Can make outgoing connections, CANNOT accept incoming
- **Hotspot client** → CAN accept incoming connections

### 3. Role Assignment Matters

| If device can... | It should be... | Running... |
|------------------|-----------------|------------|
| Only make outgoing connections | MASTER | `llama-server --rpc` |
| Accept incoming connections | WORKER | `rpc-server` |

### 4. Router/Network Isolation is Often Not the Issue

We spent time checking Eero and Unifi settings, but the blocking was on the Android device itself.

### 5. Debug Logging is Essential

```bash
export GGML_RPC_DEBUG=1  # On rpc-server
export LLAMA_LOG_VERBOSITY=0  # On llama-server
```

---

## Performance Results

**Model:** Qwen 2.5 0.5B Instruct (Q4_K_M quantization)

**Distribution:**
- MASTER: 89.26 MiB (CPU buffer)
- WORKER: 373.71 MiB (RPC buffer)

**Inference Speed:**
- Prompt processing: 26.6 tokens/second
- Generation: 5.5 tokens/second

**Network:** Mobile hotspot (WiFi direct between phones)

---

## Troubleshooting

### Connection Timeout

```bash
nc -zv <worker-ip> 60000
# Ncat: TIMEOUT.
```

**Solution:** Swap roles. The device you're trying to connect TO should be the one connected to the other's hotspot.

### "invalid argument: --no-flash-attn"

Some llama.cpp versions don't support this flag. Remove it from the command.

### Model Not Found

Ensure the model path exists on the MASTER device:
```bash
ls -la ~/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/
```

### Worker Shows No Activity During Inference

Enable debug logging:
```bash
export GGML_RPC_DEBUG=1
```

---

## Next Steps

### For Regular Network (Without Hotspot)

#### Option 1: Tailscale VPN

Install Tailscale on both devices to get reliable IPs that bypass all NAT/firewall issues:

```bash
pkg install tailscale
tailscale up
# Each device gets a 100.x.x.x IP
```

Then use Tailscale IPs for `--rpc` flag.

#### Option 2: Determine Acceptable Device

Test which device can accept connections on your regular network and assign roles accordingly.

### EXO Integration

The EXO framework needs to be updated to:

1. Detect which devices can accept incoming connections
2. Assign MASTER role to devices that can only make outgoing connections
3. Assign WORKER role to devices that can accept connections
4. Handle the Android incoming-connection limitation automatically

---

## Command Reference

### Worker (RPC Server)

```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

### Master (llama-server)

```bash
export GGML_RPC_DEBUG=1
export LLAMA_LOG_VERBOSITY=0

~/llama.cpp/build/bin/llama-server \
  -m <model-path>.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc <worker-ip>:60000 \
  --tensor-split 0.5,0.5 \
  --no-mmap \
  -c 1024 \
  --batch-size 128 \
  --verbose
```

### Test Connectivity

```bash
nc -zv <ip> <port>
```

### Test Inference

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER DEVICE (Hotspot Host)                  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  llama-server                                            │   │
│   │  - Loads model metadata                                  │   │
│   │  - Keeps 89 MiB of tensors locally                       │   │
│   │  - Coordinates inference                                 │   │
│   │  - Serves HTTP API on :8080                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ Outgoing TCP                      │
│                              │ (--rpc 192.168.37.219:60000)      │
│                              ▼                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               │ WiFi (Hotspot)
                               │
                               ▼
┌──────────────────────────────┼───────────────────────────────────┐
│                    WORKER DEVICE (Hotspot Client)                │
│                              │                                   │
│   ┌──────────────────────────▼──────────────────────────────┐   │
│   │  rpc-server (:60000)                                     │   │
│   │  - Accepts incoming connections                          │   │
│   │  - Holds 374 MiB of tensors                              │   │
│   │  - Executes [graph_compute] for inference                │   │
│   │  - Returns results to master                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

*Document created: December 2024*
*Tested on: Samsung Galaxy Z Fold (SM-F926U) x2*
*llama.cpp version: 3.6.0*

