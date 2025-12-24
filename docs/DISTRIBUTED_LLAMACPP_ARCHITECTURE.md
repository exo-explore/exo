# Distributed llama.cpp RPC Architecture

This document explains how EXO orchestrates distributed LLM inference across multiple Android devices using llama.cpp's RPC (Remote Procedure Call) system.

## Overview

EXO enables multiple devices to work together as a single LLM by:
1. Electing a **master node** to coordinate inference
2. Starting **RPC worker servers** on other devices
3. Distributing model **tensors** across all devices
4. Running **parallel computation** during inference

```
┌─────────────────────────────────────────────────────────────────┐
│                         EXO Cluster                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐      RPC        ┌──────────────┐             │
│   │   Worker 1   │◄───────────────►│              │             │
│   │  rpc-server  │   TCP:60000     │              │             │
│   │  (layers 0-11)                 │    Master    │             │
│   └──────────────┘                 │              │             │
│                                    │ llama-server │             │
│   ┌──────────────┐      RPC        │   :8080      │             │
│   │   Worker 2   │◄───────────────►│              │             │
│   │  rpc-server  │   TCP:60000     │ (orchestrates│             │
│   │ (layers 12-23)                 │  + inference)│             │
│   └──────────────┘                 └──────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. EXO Master Election

When EXO starts on multiple devices, they automatically discover each other and elect a master:

```
Device A: Node elected Master
Device B: Started worker process
Device C: Started worker process
```

The master node becomes the **llama-server host** and coordinates all inference requests.

### 2. RPC Worker Servers

Worker devices run `rpc-server` to expose their compute resources:

```bash
# Automatically started by EXO on workers
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

Each worker:
- Binds to port **60000** on all interfaces
- Waits for tensor operations from the master
- Performs local computation and returns results

### 3. Master llama-server

The master runs `llama-server` with RPC connections to all workers:

```bash
# Automatically started by EXO on master
~/llama.cpp/build/bin/llama-server \
  -m ~/.exo/models/<model>.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc <worker1_ip>:60000,<worker2_ip>:60000 \
  --tensor-split 0.5,0.5 \
  --no-mmap \
  -c 1024 \
  --verbose
```

---

## Tensor Split Strategy

The `--tensor-split` parameter controls how model layers are distributed:

### For 2 Workers
```
--tensor-split 0.5,0.5
```
- Worker 1: 50% of layers
- Worker 2: 50% of layers

### For 3 Workers
```
--tensor-split 0.34,0.33,0.33
```
- Worker 1: 34% of layers
- Worker 2: 33% of layers  
- Worker 3: 33% of layers

### Memory-Based Calculation

EXO calculates splits based on available RAM on each device:

```python
# From placement_utils.py
def get_tensor_split_for_llamacpp(selected_cycle):
    workers = selected_cycle[1:]  # Skip master
    
    total_worker_memory = sum(node.memory for node in workers)
    
    tensor_split = []
    for node in workers:
        ratio = node.memory / total_worker_memory
        tensor_split.append(ratio)
    
    return tensor_split  # e.g., [0.52, 0.48]
```

**Important**: The tensor split has one value per **worker**, NOT per total device. The master is excluded from the split calculation.

---

## Startup Sequence

When you click **Launch** in the EXO dashboard:

### Phase 1: Worker RPC Servers Start (5-10 seconds)

```
[Worker 1] STARTING RPC WORKER SERVER
[Worker 1] rpc-server started successfully on 0.0.0.0:60000
[Worker 1] External IPs for RPC: 10.99.0.78:60000

[Worker 2] STARTING RPC WORKER SERVER  
[Worker 2] rpc-server started successfully on 0.0.0.0:60000
[Worker 2] External IPs for RPC: 10.99.0.152:60000
```

### Phase 2: Master Verifies Workers (1-5 seconds)

```
[Master] Checking connectivity to 2 RPC worker(s)...
[Master] Worker 10.99.0.152:60000: READY
[Master] Worker 10.99.0.78:60000: READY
[Master] All 2 RPC workers ready after 1s
```

### Phase 3: Master Starts llama-server (5 seconds)

```
[Master] Starting llama-server with distributed inference...
[Master] Full command: llama-server -m <model> --rpc 10.99.0.152:60000,10.99.0.78:60000 --tensor-split 0.50,0.50 --no-mmap
```

### Phase 4: RPC Handshake (30-60 seconds)

The master connects to each worker multiple times to negotiate:

```
[Worker] Accepted client connection
[Worker] [hello] version: 3.6.0
[Worker] [get_alloc_size] device: 0, buffer: 0x0
[Worker] Client connection closed

(repeats many times - this is normal!)
```

### Phase 5: Tensor Distribution (30-60 seconds)

Model weights are distributed to workers:

```
[Master] load_tensors: layer 0 assigned to device RPC0
[Master] load_tensors: layer 12 assigned to device RPC1
...
[Worker 1] [set_tensor] buffer: 0xb400007..., size: 551936
[Worker 2] [set_tensor] buffer: 0xb400007..., size: 551936
```

### Phase 6: Ready! (Total: 2-3 minutes)

```
[Master] main: model loaded
[Master] server is listening on http://127.0.0.1:8080
[Master] DETECTED: Model loaded successfully!
[Master] DISTRIBUTED LLAMA-SERVER READY
```

---

## Inference Flow

When a chat message is sent:

```
┌────────┐    HTTP     ┌────────┐    RPC     ┌──────────┐
│  User  │────────────►│ Master │───────────►│ Worker 1 │
│        │◄────────────│        │◄───────────│          │
└────────┘   Response  │        │            └──────────┘
                       │        │    RPC     ┌──────────┐
                       │        │───────────►│ Worker 2 │
                       │        │◄───────────│          │
                       └────────┘            └──────────┘
```

1. **User sends message** → EXO dashboard
2. **Dashboard forwards** → Master's llama-server (:8080)
3. **Master coordinates** → Sends tensor ops to workers via RPC
4. **Workers compute** → Return results to master
5. **Master aggregates** → Streams response back to user

### What Workers Do

During inference, workers perform:

```
[Worker] [set_tensor] - Receive input tensors
[Worker] [graph_compute] device: 0, n_nodes: 443, n_tensors: 630
[Worker] [get_tensor] - Return computed results
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/exo/worker/engines/llamacpp/utils.py` | `DistributedLlamaServer` class - manages master llama-server |
| `src/exo/worker/engines/llamacpp/rpc_server.py` | `RpcServerManager` class - manages worker rpc-server |
| `src/exo/master/placement_utils.py` | `get_tensor_split_for_llamacpp()` - calculates layer distribution |
| `src/exo/worker/runner/runner.py` | Orchestrates worker/master processes based on role |

---

## Critical Configuration

### llama-server (Master)

| Flag | Value | Purpose |
|------|-------|---------|
| `--port` | 8080 | HTTP API port |
| `--host` | 127.0.0.1 | Bind locally (EXO proxies requests) |
| `--rpc` | ip1:60000,ip2:60000 | Worker addresses |
| `--tensor-split` | 0.5,0.5 | Layer distribution ratios |
| `--no-mmap` | (flag) | Required for Android/Termux |
| `-c` | 1024 | Context window size |
| `--verbose` | (flag) | Detailed logging |

### rpc-server (Workers)

| Flag | Value | Purpose |
|------|-------|---------|
| `--host` | 0.0.0.0 | Accept connections from any IP |
| `--port` | 60000 | RPC listen port |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `GGML_RPC_DEBUG` | 1 | Enable RPC debug logging |
| `LD_LIBRARY_PATH` | ~/llama.cpp/build/src | Shared library path |

---

## Troubleshooting

### Workers Not Receiving Tensors

**Symptom**: Workers show only `[hello]` and `[get_alloc_size]`, no `[set_tensor]`

**Cause**: Tensor split has wrong number of values

**Fix**: Ensure `--tensor-split` has exactly N values where N = number of workers (NOT total devices)

### Health Check 503 Forever

**Symptom**: Master shows repeated "Health check 503: Loading model"

**Cause**: Distributed llama-server doesn't return /health=200 until after first inference

**Fix**: EXO now detects "model loaded" in stderr instead of relying on HTTP health checks

### Connection Refused

**Symptom**: Master can't connect to workers

**Causes**:
- Workers not started yet
- Wrong IP addresses
- Firewall blocking port 60000

**Fix**: EXO now waits for workers to be verified ready before starting master

---

## Building llama.cpp with RPC Support

RPC support must be compiled in:

```bash
cd ~/llama.cpp
rm -rf build
cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4
```

Verify RPC is enabled:

```bash
~/llama.cpp/build/bin/llama-server --help | grep rpc
# Should show: --rpc SERVERS
```

---

## Performance Notes

- **Startup time**: 2-3 minutes for initial load (includes RPC handshake + tensor distribution)
- **Inference**: Slightly slower than single-device due to network overhead
- **Scaling**: Works with 2+ devices; more devices = larger models possible
- **Memory**: Each worker only holds its assigned layers, not the full model

---

## Summary

EXO + llama.cpp RPC enables:

✅ **Automatic device discovery** - No manual IP configuration  
✅ **Dynamic tensor distribution** - Based on device memory  
✅ **Fault-tolerant startup** - Retries and health checks  
✅ **Transparent to users** - Just click Launch and chat  
✅ **Scalable** - Add more devices for larger models  

The system transforms multiple Android phones into a unified AI compute cluster!

