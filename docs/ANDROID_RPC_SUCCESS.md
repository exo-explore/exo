# Distributed LLM Inference on Android: Success Guide

> **How we achieved distributed llama.cpp inference across multiple Android phones**

This document captures the working setup for running distributed LLM inference using llama.cpp's RPC backend across multiple Android devices running Termux.

---

## Table of Contents

1. [Summary](#summary)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Dedicated Network Setup (Recommended)](#dedicated-network-setup-recommended)
5. [Hotspot Setup (Alternative)](#hotspot-setup-alternative)
6. [3+ Phone Cluster Setup](#3-phone-cluster-setup)
7. [Verification](#verification)
8. [Key Learnings](#key-learnings)
9. [Performance Results](#performance-results)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

---

## Summary

We successfully ran distributed LLM inference across **2 and 3 Android phones**:

### 2-Phone Setup (Hotspot)

| Device | Role | Model Portion | Function |
|--------|------|---------------|----------|
| Phone 1 (Hotspot Host) | MASTER | 89 MiB (19%) | Runs `llama-server`, coordinates inference |
| Phone 2 (Hotspot Client) | WORKER | 374 MiB (81%) | Runs `rpc-server`, computes tensors |

### 3-Phone Setup (Dedicated Network)

| Device | Role | IP | Model Buffer |
|--------|------|-----|--------------|
| Phone 1 | MASTER | 10.99.0.x | 89.26 MiB |
| Phone 2 | WORKER 1 | 10.99.0.14 | 244.33 MiB |
| Phone 3 | WORKER 2 | 10.99.0.152 | 129.37 MiB |

**Result:** Working chat completions API with all devices computing together.

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

After extensive testing, we discovered **two separate issues**:

#### Issue 1: Eero Access Points Block Device-to-Device Traffic

The Eero access points (even in bridge mode) were blocking device-to-device communication. When we tested on native Unifi access points, connections worked **both ways**:

```bash
# On Phone A → Phone B
nc -zv 10.69.1.68 60000
# Result: Connected to 10.69.1.68:60000

# On Phone B → Phone A
nc -zv 10.69.1.69 60000
# Result: Connected to 10.69.1.69:60000
```

**Conclusion:** Android does NOT inherently block incoming TCP connections. The issue was Eero's hidden client isolation.

#### Issue 2: Android Hotspot NAT Blocks Incoming Connections

When using Android's mobile hotspot, the hotspot HOST device cannot accept incoming connections due to NAT:

```bash
# On device connected to hotspot, trying to reach hotspot host:
nc -zv 192.168.43.1 60000
# Result: Ncat: TIMEOUT.
```

But localhost works fine on the device running the server:

```bash
nc -zv 127.0.0.1 60000
# Result: 127.0.0.1 (127.0.0.1:60000) open
```

**Conclusion:** Hotspot host acts as a NAT gateway and blocks incoming connections to itself.

---

## The Solution

### Two Working Approaches

| Approach | Best For | Setup Complexity |
|----------|----------|------------------|
| **Dedicated Network (VLAN)** | Multiple phones, regular use | Medium (one-time) |
| **Mobile Hotspot** | Quick testing, 2 phones | Easy |

---

## Dedicated Network Setup (Recommended)

For reliable multi-device inference, create a dedicated network with no client isolation.

### Why a Dedicated Network?

- ✅ No client isolation blocking device-to-device traffic
- ✅ Stable IPs via DHCP
- ✅ Works with 3+ devices
- ✅ Better performance than hotspot
- ✅ Isolated from other home traffic

### Step 1: Create Unifi Network (VLAN)

1. Open **Unifi Controller** → **Settings** → **Networks**
2. Click **+ Create New Network**

| Setting | Value |
|---------|-------|
| **Name** | `AI-Phone-Cluster` |
| **Host Address** | `10.99.0.1` |
| **Netmask** | `24` |
| **VLAN ID** | `99` (or any unused) |
| **Isolate Network** | **OFF** ⚠️ Critical! |
| **Allow Internet Access** | ON |
| **DHCP Mode** | DHCP Server |
| **mDNS** | ON (optional, helps discovery) |

### Step 2: Create Dedicated WiFi SSID

1. Go to **Settings** → **WiFi** → **+ Create New WiFi**

| Setting | Value |
|---------|-------|
| **Name (SSID)** | `AI-Phone-Cluster` |
| **Network** | Select `AI-Phone-Cluster` |
| **WiFi Band** | 5 GHz (faster) |
| **Security** | WPA2/WPA3 |

**Advanced Settings (Critical):**

| Setting | Value | Why |
|---------|-------|-----|
| **Client Device Isolation** | **OFF** ⚠️ | Devices must communicate |
| **L2 Isolation** | **OFF** ⚠️ | Same reason |
| **Proxy ARP** | OFF | |
| **BSS Transition** | ON | Better roaming |

### Step 3: Connect All Phones

1. Connect all Android phones to `AI-Phone-Cluster` WiFi
2. Verify IPs:

```bash
ip addr show wlan0 | grep "inet "
# Should show: 10.99.0.x
```

### Step 4: Test Connectivity

From any phone to any other:

```bash
nc -zv 10.99.0.152 60000
# Should show: Connected
```

### Step 5: Start Distributed Inference

**WORKER devices:**
```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

**MASTER device:**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc 10.99.0.152:60000 \
  --tensor-split 0.5,0.5 \
  --no-mmap \
  -c 1024 \
  --verbose
```

---

## Hotspot Setup (Alternative)

For quick testing without network configuration.

### The Key Insight

When using hotspot, the device that can only make **outgoing** connections should be the **MASTER**.

| Device | Hotspot Role | Can Accept Incoming? | llama.cpp Role |
|--------|--------------|---------------------|----------------|
| Hotspot Host | Provides WiFi | ❌ No (NAT blocks) | MASTER (`llama-server`) |
| Hotspot Client | Connects to hotspot | ✅ Yes | WORKER (`rpc-server`) |

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
export GGML_RPC_DEBUG=1
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

On the MASTER device (hotspot host):

```bash
nc -zv 192.168.37.219 60000
# Expected: 192.168.37.219 (192.168.37.219:60000) open
```

### Step 5: Start Distributed llama-server on MASTER

```bash
export GGML_RPC_DEBUG=1
export LLAMA_LOG_VERBOSITY=0

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

Eventually MASTER shows:
```
main: model loaded
main: server is listening on http://127.0.0.1:8080
```

---

## 3+ Phone Cluster Setup

Adding more phones to the cluster for larger models.

### Why Use 3+ Phones?

| Model Size | Recommended Phones | Benefit |
|------------|-------------------|---------|
| 0.5B - 1B | 1-2 phones | Overhead not worth it for small models |
| 3B - 7B | 2-3 phones | Good balance of speed and memory |
| 13B+ | 3-4+ phones | Required to fit model in memory |

⚠️ **Note:** For small models (0.5B), 3 phones may be **slower** than 2 due to network overhead.

### Step 1: Start rpc-server on ALL Worker Phones

**Worker 1 (e.g., 10.99.0.14):**
```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

**Worker 2 (e.g., 10.99.0.152):**
```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

### Step 2: Test Connectivity from Master

```bash
nc -zv 10.99.0.14 60000   # Worker 1
nc -zv 10.99.0.152 60000  # Worker 2
# Both should show: Connected
```

### Step 3: Start llama-server with Multiple Workers

**On MASTER device:**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc 10.99.0.14:60000,10.99.0.152:60000 \
  --tensor-split 0.33,0.33,0.34 \
  --no-mmap \
  -c 1024 \
  --verbose
```

**Key changes for 3 devices:**
- `--rpc` contains **comma-separated** addresses of all workers
- `--tensor-split 0.33,0.33,0.34` splits across 3 devices (must sum to 1.0)

### 3-Phone Model Distribution Example

```
load_tensors:          CPU model buffer size =    89.26 MiB
load_tensors: RPC0[10.99.0.14:60000] model buffer size =   244.33 MiB
load_tensors: RPC0[10.99.0.152:60000] model buffer size =   129.37 MiB
```

| Device | Model Buffer | Layers |
|--------|--------------|--------|
| Master (CPU) | 89.26 MiB | Embeddings + output |
| Worker 1 (10.99.0.14) | 244.33 MiB | Layers 0-12 |
| Worker 2 (10.99.0.152) | 129.37 MiB | Layers 13-23 |

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

### 1. Android Does NOT Block Incoming Connections (Usually)

Contrary to initial assumptions, Android itself does NOT block incoming TCP connections. The issue was specific network equipment (Eero access points) with hidden client isolation.

**When tested on Unifi access points:** Connections worked both ways between Android devices.

### 2. Eero Access Points Have Hidden Client Isolation

Even in bridge mode, Eero access points may block device-to-device traffic. This was the root cause of our initial failures.

**Solution:** Use Unifi access points with client isolation explicitly disabled, or create a dedicated VLAN.

### 3. Hotspot NAT Blocks Incoming to Host

When using Android's mobile hotspot:
- **Hotspot host** → CANNOT accept incoming (NAT blocks)
- **Hotspot client** → CAN accept incoming connections

| If device can... | It should be... | Running... |
|------------------|-----------------|------------|
| Only make outgoing connections | MASTER | `llama-server --rpc` |
| Accept incoming connections | WORKER | `rpc-server` |

### 4. Dedicated VLAN is the Best Solution

Creating a dedicated network (VLAN + WiFi SSID) with:
- **Isolate Network: OFF**
- **Client Device Isolation: OFF**
- **L2 Isolation: OFF**

Provides reliable device-to-device communication for distributed inference.

### 5. More Phones ≠ Always Faster

For small models (0.5B-1B), adding more phones can actually **slow down** inference due to network overhead:

| Test | Phones | Generation Speed |
|------|--------|-----------------|
| Small prompt | 2 phones | 9.4 tok/s |
| Same prompt | 3 phones | 7.0 tok/s |
| Long code gen | 3 phones | 4.05 tok/s |

**Rule of thumb:** Only add more phones when the model is too large for fewer devices.

### 6. Debug Logging is Essential

```bash
export GGML_RPC_DEBUG=1  # On rpc-server (shows graph_compute)
export LLAMA_LOG_VERBOSITY=0  # On llama-server
```

---

## Performance Results

**Model:** Qwen 2.5 0.5B Instruct (Q4_K_M quantization)

### 2-Phone Setup (Hotspot)

| Component | Size |
|-----------|------|
| MASTER (CPU) | 89.26 MiB |
| WORKER (RPC) | 373.71 MiB |

| Metric | Value |
|--------|-------|
| Prompt processing | 20.1 tok/s |
| Generation | 9.4 tok/s |

### 3-Phone Setup (Dedicated Network)

| Component | Size | Layers |
|-----------|------|--------|
| MASTER (CPU) | 89.26 MiB | Embeddings + output |
| WORKER 1 (10.99.0.14) | 244.33 MiB | Layers 0-12 |
| WORKER 2 (10.99.0.152) | 129.37 MiB | Layers 13-23 |

**Memory Breakdown:**

| Device | Model Buffer | KV Cache | Compute Buffer |
|--------|--------------|----------|----------------|
| Master | 89.26 MiB | - | 3.76 MiB |
| Worker 1 | 244.33 MiB | 5.50 MiB | 298.50 MiB |
| Worker 2 | 129.37 MiB | 6.50 MiB | 35.01 MiB |

### Benchmark Results

| Test | Phones | Prompt Tokens | Generated Tokens | Prompt Speed | Gen Speed | Total Time |
|------|--------|---------------|------------------|--------------|-----------|------------|
| Haiku | 2 | 37 | 13 | 20.1 tok/s | 9.4 tok/s | 3.2 sec |
| Story | 3 | 60 | 105 | 18.0 tok/s | 7.0 tok/s | 18.4 sec |
| Fibonacci code | 3 | 46 | 250 | 1.1 tok/s* | 4.05 tok/s | 62.6 sec |

*Prompt was mostly cached from previous requests.

### Key Observations

1. **2 phones faster than 3 for small models** - Network overhead outweighs parallelization benefit
2. **Generation slows with length** - KV cache sync adds latency per token
3. **Prompt caching works** - Repeated system prompts are cached, reducing prompt processing
4. **WiFi latency accumulates** - Each token requires round-trip to all workers

**Network:** Unifi VLAN 99 (5GHz WiFi, dedicated SSID)

---

## Troubleshooting

### Connection Timeout

```bash
nc -zv <worker-ip> 60000
# Ncat: TIMEOUT.
```

**Possible causes:**

| Cause | Solution |
|-------|----------|
| Eero access point | Use Unifi AP or create dedicated VLAN |
| Client isolation enabled | Disable in router/WiFi settings |
| Hotspot NAT | Swap roles - hotspot host must be MASTER |
| Wrong IP | Verify with `ip addr show wlan0` |

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

You should see `[graph_recompute]` messages during inference.

### Slow Performance with 3+ Phones

This is **expected** for small models. The network overhead outweighs the benefit of parallel computation.

| Model Size | Recommended Phones |
|------------|-------------------|
| < 1B | 1-2 phones |
| 3B-7B | 2-3 phones |
| 13B+ | 3-4+ phones |

### "No route to host"

Check that:
1. Both devices are on the same network/VLAN
2. WiFi is connected (not mobile data)
3. IP addresses are in the same subnet

---

## Next Steps

### Try Larger Models

The real benefit of distributed inference comes with larger models:

```bash
# Example: 3B model across 3 phones
~/llama.cpp/build/bin/llama-server \
  -m ~/models/phi-3-mini-4k-instruct-q4_k_m.gguf \
  --rpc 10.99.0.14:60000,10.99.0.152:60000 \
  --tensor-split 0.33,0.33,0.34 \
  --no-mmap \
  -c 2048 \
  --verbose
```

### Add More Phones

For 4+ phones:
```bash
--rpc 10.99.0.14:60000,10.99.0.152:60000,10.99.0.153:60000
--tensor-split 0.25,0.25,0.25,0.25
```

### Tailscale for Remote Devices

For phones on different networks:

```bash
pkg install tailscale
tailscale up
# Each device gets a 100.x.x.x IP
```

Use Tailscale IPs for `--rpc` flag.

### EXO Integration

The EXO framework could be updated to:

1. Auto-detect devices on the same network
2. Test connectivity before assigning roles
3. Automatically select optimal tensor split based on device memory
4. Handle network configuration automatically

---

## Architecture Diagrams

### 2-Phone Setup (Hotspot)

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

### 3-Phone Setup (Dedicated Network)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     AI-Phone-Cluster Network                          │
│                        VLAN 99 (10.99.0.x)                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │   WORKER 1      │  │   WORKER 2      │  │    MASTER       │       │
│  │  10.99.0.14     │  │  10.99.0.152    │  │  10.99.0.x      │       │
│  │  rpc-server     │  │  rpc-server     │  │  llama-server   │       │
│  │  :60000         │  │  :60000         │  │  :8080          │       │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤       │
│  │ Model: 244 MiB  │  │ Model: 129 MiB  │  │ Model: 89 MiB   │       │
│  │ Layers: 0-12    │  │ Layers: 13-23   │  │ Embeddings      │       │
│  │ KV: 5.5 MiB     │  │ KV: 6.5 MiB     │  │ Output layer    │       │
│  │ Compute: 298 MiB│  │ Compute: 35 MiB │  │ Compute: 4 MiB  │       │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘       │
│           │                    │                    │                │
│           └────────────────────┼────────────────────┘                │
│                                │                                      │
│                         RPC Communication                             │
│                    (--rpc 10.99.0.14:60000,10.99.0.152:60000)        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘

Token Generation Flow:
═══════════════════════════════════════════════════════════════════════

  Input     ┌──────────┐   ┌──────────┐   ┌──────────┐   Output
  Prompt ──►│ Master   │──►│ Worker 1 │──►│ Worker 2 │──► Token
            │ Embed    │   │ Layers   │   │ Layers   │
            │ + Output │   │ 0-12     │   │ 13-23    │
            └──────────┘   └──────────┘   └──────────┘
                 ▲                              │
                 └──────────────────────────────┘
                         (repeat for each token)
```

---

## Command Reference

### Worker (RPC Server)

```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

### Master (2 Workers)

```bash
~/llama.cpp/build/bin/llama-server \
  -m <model-path>.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc <worker1-ip>:60000,<worker2-ip>:60000 \
  --tensor-split 0.33,0.33,0.34 \
  --no-mmap \
  -c 1024 \
  --verbose
```

### Test Inference

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

*Document created: December 2024*
*Updated: December 2024 (3-phone setup, dedicated network)*
*Tested on: Samsung Galaxy Z Fold (SM-F926U) x3*
*llama.cpp version: 3.6.0*
*Network: Unifi Dream Machine Pro + Unifi APs (VLAN 99)*

