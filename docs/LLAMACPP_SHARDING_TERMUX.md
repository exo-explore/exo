# llama.cpp Sharding for Android/Termux

> **Distributing LLM inference across multiple Android devices using llama.cpp**

This guide covers how llama.cpp handles model sharding (distributing layers across multiple devices), and provides best practices for setting up distributed inference on Android devices running Termux.

---

## Table of Contents

1. [Understanding llama.cpp Sharding](#understanding-llamacpp-sharding)
2. [Sharding Mechanisms in llama.cpp](#sharding-mechanisms-in-llamacpp)
3. [RPC-Based Distributed Inference](#rpc-based-distributed-inference)
4. [Termux Setup for Distributed Inference](#termux-setup-for-distributed-inference)
5. [Building with RPC Support](#building-with-rpc-support)
6. [Running a Distributed Cluster](#running-a-distributed-cluster)
7. [Tensor Split Configuration](#tensor-split-configuration)
8. [Memory and Performance Optimization](#memory-and-performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices Summary](#best-practices-summary)

---

## Understanding llama.cpp Sharding

### What is Sharding?

Sharding in the context of LLMs refers to **distributing model layers and computations across multiple devices**. This enables running larger models than a single device could handle, by splitting:

- **Model Weights**: Different layers reside on different devices
- **Tensor Operations**: Computations are parallelized across devices
- **Memory Load**: Each device only holds a portion of the model

### Types of Parallelism

| Type | Description | llama.cpp Support |
|------|-------------|-------------------|
| **Tensor Parallelism (TP)** | Split individual tensors across devices | ✓ via `--tensor-split` + `--rpc` |
| **Pipeline Parallelism (PP)** | Different layers on different devices | ✓ via RPC backend |
| **Data Parallelism (DP)** | Same model, different batches | Not natively supported |

### llama.cpp's Approach

llama.cpp uses **RPC-based tensor parallelism** where:

1. A **master node** (rank 0) coordinates inference
2. **Worker nodes** (rank > 0) run `rpc-server` to handle tensor operations
3. The master connects to workers via TCP using the `--rpc` flag
4. Tensor splits are defined with `--tensor-split` to control memory distribution

---

## Sharding Mechanisms in llama.cpp

### The GGML RPC Backend

llama.cpp's sharding relies on the **GGML RPC backend**, which enables offloading tensor operations to remote devices over the network.

**Key Components:**

| Component | Purpose | Binary |
|-----------|---------|--------|
| **rpc-server** | Listens for tensor operations from master | `rpc-server` |
| **--rpc flag** | Connects master to worker RPC servers | Used by `llama-server`, `llama-cli` |
| **--tensor-split** | Defines how to distribute layers | Used by `llama-server`, `llama-cli` |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      MASTER NODE (Device 0)                      │
│                                                                  │
│  llama-server -m model.gguf --rpc 192.168.1.101:50052,          │
│               192.168.1.102:50052 --tensor-split 0.4,0.3,0.3    │
│                                                                  │
│  - Loads model metadata                                          │
│  - Coordinates inference                                         │
│  - Handles API requests                                          │
│  - Distributes tensor ops to workers                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ TCP (RPC Protocol)
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Worker 1   │   │  Worker 2   │   │  Worker 3   │
│             │   │             │   │             │
│ rpc-server  │   │ rpc-server  │   │ rpc-server  │
│ :50052      │   │ :50052      │   │ :50052      │
│             │   │             │   │             │
│ 40% tensors │   │ 30% tensors │   │ 30% tensors │
└─────────────┘   └─────────────┘   └─────────────┘
```

### Tensor Split Explained

The `--tensor-split` flag controls how model layers are distributed:

```bash
# Format: --tensor-split ratio1,ratio2,ratio3,...
# Each ratio represents the fraction of the model for each device

# Example: 3 devices with equal split
--tensor-split 0.33,0.33,0.34

# Example: Master has more RAM, workers less
--tensor-split 0.5,0.25,0.25

# Example: Heterogeneous devices (flagship + mid-range)
--tensor-split 0.6,0.4  # Flagship gets 60%
```

**Important**: The ratios should sum to 1.0 and match the number of devices in the cluster.

---

## RPC-Based Distributed Inference

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    EXO Distributed System                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐                                          │
│  │  EXO Master     │◄────── API Requests (/v1/chat/...)      │
│  │  (Coordinator)  │                                          │
│  └────────┬────────┘                                          │
│           │                                                   │
│           │ Spawn Instance                                    │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Distributed llama-server                    │ │
│  │                                                          │ │
│  │  Master (rank 0):                                        │ │
│  │    llama-server -m model.gguf                            │ │
│  │                 --rpc worker1:50052,worker2:50052        │ │
│  │                 --tensor-split 0.4,0.3,0.3               │ │
│  │                                                          │ │
│  │  Workers (rank > 0):                                     │ │
│  │    rpc-server --host 0.0.0.0 --port 50052                │ │
│  │                                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### RPC Server Ports

| Device Rank | Role | Port | Notes |
|-------------|------|------|-------|
| 0 | Master | N/A (uses --rpc) | Runs `llama-server` |
| 1 | Worker | 50052 | Runs `rpc-server` |
| 2 | Worker | 50053 | Runs `rpc-server` |
| N | Worker | 50051 + N | Runs `rpc-server` |

Port assignment (as implemented in EXO):

```python
def assign_rpc_port(device_rank: int) -> int:
    """Assign RPC port based on device rank."""
    if device_rank == 0:
        return 0  # Master doesn't need RPC port
    return 50052 + device_rank - 1  # Workers get sequential ports
```

---

## Termux Setup for Distributed Inference

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Android Version | 8.0+ | 10.0+ |
| RAM per device | 4 GB | 8+ GB |
| Storage | 10 GB free | 50+ GB free |
| Network | Same WiFi | 5 GHz WiFi / wired |
| Termux | F-Droid version | Latest from F-Droid |

### Directory Structure

Termux follows a specific file system structure. Store all files in appropriate locations:

```
/data/data/com.termux/files/
├── home/                          # ~ (your home directory)
│   ├── llama.cpp/                 # llama.cpp source and binaries
│   │   └── build/
│   │       └── bin/
│   │           ├── llama-server   # API server (master)
│   │           ├── llama-cli      # CLI interface
│   │           └── rpc-server     # RPC worker server
│   ├── models/                    # Store GGUF models here
│   │   └── model.gguf
│   └── exo-termux/                # EXO installation
└── usr/                           # System binaries, libraries
    ├── bin/
    └── lib/
```

**Best Practice**: Store models in `~/models/` or `~/.exo/models/` for consistent access.

---

## Building with RPC Support

### Standard Build (Without RPC)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4
```

This builds `llama-server` and `llama-cli` but **NOT** `rpc-server`.

### Build with RPC Support (Required for Distributed Inference)

```bash
cd ~/llama.cpp

# Clean previous build if exists
rm -rf build

# Configure with RPC enabled
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_RPC=ON

# Build main binaries
cmake --build build --config Release -j4

# Build rpc-server explicitly
cmake --build build --target rpc-server -j4

# Verify all binaries exist
ls -la build/bin/llama-server build/bin/llama-cli build/bin/rpc-server
```

**Critical**: The `GGML_RPC=ON` flag is **required** for distributed inference. Without it, `rpc-server` won't be built.

### Verify Build

```bash
# Check binaries exist
ls ~/llama.cpp/build/bin/

# Expected output:
# llama-cli
# llama-server
# rpc-server      # <-- This is critical for workers

# Test rpc-server
~/llama.cpp/build/bin/rpc-server --help
```

### Environment Configuration

Add to `~/.bashrc`:

```bash
# llama.cpp library path
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so

# Convenient aliases
alias llama-server='$HOME/llama.cpp/build/bin/llama-server'
alias llama-cli='$HOME/llama.cpp/build/bin/llama-cli'
alias rpc-server='$HOME/llama.cpp/build/bin/rpc-server'
```

---

## Running a Distributed Cluster

### Step 1: Identify Device IPs

On each device:

```bash
# Get WiFi IP address
ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1

# Example output: 192.168.1.101
```

### Step 2: Start Worker Nodes (Rank > 0)

On **each worker device**, start `rpc-server`:

```bash
# Worker 1 (Device at 192.168.1.101)
termux-wake-lock  # Prevent sleep
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052

# Worker 2 (Device at 192.168.1.102)
termux-wake-lock
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052
```

**Important Notes:**

- Use `--host 0.0.0.0` to listen on all interfaces (required for network access)
- Workers **don't load the model** - they receive tensor operations from master
- Keep workers running in foreground or use `screen`/`tmux`

### Step 3: Start Master Node (Rank 0)

On the **master device**:

```bash
# Acquire wake lock
termux-wake-lock

# Start distributed llama-server
~/llama.cpp/build/bin/llama-server \
    -m ~/models/model.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --rpc 192.168.1.101:50052,192.168.1.102:50052 \
    --tensor-split 0.4,0.3,0.3 \
    -c 2048 \
    -t $(nproc) \
    --no-mmap
```

### Step 4: Verify Cluster

```bash
# Check server health
curl http://localhost:8080/health

# Test completion
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }'
```

---

## Tensor Split Configuration

### Calculating Optimal Splits

The tensor split should reflect **each device's available memory** for the model:

```bash
# Formula:
# ratio = device_available_ram / total_cluster_ram

# Example: 3 devices with 8GB, 6GB, 6GB RAM
# Total: 20GB
# Splits: 8/20=0.4, 6/20=0.3, 6/20=0.3
--tensor-split 0.4,0.3,0.3
```

### Memory Considerations

| Device RAM | Usable for Model | Notes |
|------------|------------------|-------|
| 4 GB | ~2-2.5 GB | Android needs ~1.5-2 GB |
| 6 GB | ~3.5-4 GB | Good for 3B models |
| 8 GB | ~5-6 GB | Good for 7B models |
| 12 GB | ~8-9 GB | Good for 13B models |

### Common Split Configurations

| Cluster Size | Configuration | Use Case |
|--------------|---------------|----------|
| 2 devices (equal) | `0.5,0.5` | Two similar phones |
| 2 devices (flagship + mid) | `0.6,0.4` | One faster device |
| 3 devices (equal) | `0.33,0.33,0.34` | Three similar phones |
| 3 devices (1 flagship) | `0.5,0.25,0.25` | One flagship, two budget |
| 4 devices (equal) | `0.25,0.25,0.25,0.25` | Four similar phones |

---

## Memory and Performance Optimization

### Quantization Selection

For Android devices, use aggressive quantization:

| Quantization | Size vs FP16 | Quality | Recommended For |
|--------------|--------------|---------|-----------------|
| Q2_K | ~3x smaller | Lower | 4GB RAM devices |
| Q3_K_M | ~2.5x smaller | OK | 4-6GB RAM devices |
| **Q4_K_M** | ~2x smaller | **Good** | **6-8GB RAM (recommended)** |
| Q5_K_M | ~1.5x smaller | Better | 8+GB RAM devices |
| Q8_0 | ~1.1x smaller | Best | 12+GB RAM devices |

### Recommended Models by Cluster RAM

| Total Cluster RAM | Model Size | Recommended Models |
|-------------------|------------|-------------------|
| 12-16 GB (3×4GB) | 3B-7B | Llama-3.2-3B-Q4_K_M, Qwen2.5-3B-Q4_K_M |
| 18-24 GB (3×6GB) | 7B-13B | Llama-3.1-8B-Q4_K_M, Mistral-7B-Q4_K_M |
| 24-32 GB (4×6GB) | 13B-20B | Llama-2-13B-Q4_K_M, Qwen-14B-Q4_K_M |
| 32+ GB (4×8GB) | 20B-30B | Mixtral-8x7B-Q3_K_M, Qwen-32B-Q3_K_M |

### Performance Flags

```bash
# Master node optimal flags
llama-server \
    -m ~/models/model.gguf \
    --rpc worker1:50052,worker2:50052 \
    --tensor-split 0.4,0.3,0.3 \
    -c 2048 \              # Context size (reduce if OOM)
    -t $(nproc) \          # Use all CPU threads
    -ngl 0 \               # No GPU layers (Android has no CUDA)
    --no-mmap \            # Disable mmap (more stable on Android)
    --batch-size 256 \     # Reduce if memory-constrained
    --flash-attn           # Enable flash attention if supported
```

### Network Optimization

For best performance:

1. **Use 5 GHz WiFi** - Higher bandwidth, lower latency
2. **Same router** - All devices on the same network
3. **Minimize interference** - Close to router, few obstacles
4. **Static IPs** - Configure via router DHCP settings

### Thermal Management

Create `~/thermal_cluster.sh`:

```bash
#!/bin/bash
# Monitor and throttle cluster when too hot

TEMP_LIMIT=42  # Celsius

while true; do
    TEMP=$(termux-battery-status 2>/dev/null | jq -r '.temperature' 2>/dev/null || echo "0")
    
    if [ "$TEMP" -gt "$TEMP_LIMIT" ]; then
        echo "Temperature $TEMP°C too high, pausing..."
        # Pause RPC server if running
        pkill -STOP -f rpc-server
        sleep 60
        pkill -CONT -f rpc-server
    fi
    
    sleep 30
done
```

---

## Troubleshooting

### Common Issues

#### RPC Server Not Found

```bash
# Error: rpc-server not found
# Solution: Rebuild with GGML_RPC=ON
cd ~/llama.cpp
rm -rf build
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_RPC=ON
cmake --build build --target rpc-server -j4
```

#### Worker Connection Refused

```bash
# Check worker is running
curl -v telnet://192.168.1.101:50052

# Common causes:
# 1. rpc-server not running
# 2. Firewall blocking port
# 3. Wrong IP address
# 4. rpc-server bound to 127.0.0.1 (use 0.0.0.0)
```

#### Workers Not Reachable

```bash
# On worker, verify binding
netstat -tlnp | grep 50052

# Check external IP is correct
ip addr show wlan0

# Test from master
ping 192.168.1.101
nc -zv 192.168.1.101 50052
```

#### Out of Memory

```bash
# Symptoms: Process killed, segfault
# Solutions:
# 1. Use smaller model
# 2. Use more aggressive quantization (Q3_K_M, Q2_K)
# 3. Reduce context size: -c 1024
# 4. Reduce batch size: --batch-size 128
# 5. Close other apps
```

#### Slow Inference

```bash
# Check network latency
ping -c 10 worker_ip

# If latency > 10ms, network is bottleneck
# Solutions:
# 1. Move devices closer to router
# 2. Use 5 GHz WiFi
# 3. Use wired connection if possible
# 4. Reduce tensor-split for slower devices
```

### Debug Logging

Enable verbose output:

```bash
# Master with verbose
llama-server -m model.gguf \
    --rpc worker1:50052,worker2:50052 \
    --tensor-split 0.4,0.3,0.3 \
    --verbose

# Worker with logging
rpc-server --host 0.0.0.0 --port 50052 2>&1 | tee ~/rpc.log
```

---

## Best Practices Summary

### ✅ Do

| Practice | Reason |
|----------|--------|
| Build with `GGML_RPC=ON` | Required for `rpc-server` |
| Use 5 GHz WiFi | Lower latency, higher bandwidth |
| Use `--no-mmap` on Android | More stable memory handling |
| Store models in `~/` | Termux has full access |
| Use `termux-wake-lock` | Prevents Android from killing process |
| Use Q4_K_M quantization | Best balance for mobile |
| Calculate tensor-split from RAM | Optimal memory distribution |
| Start workers before master | Master waits for workers |
| Monitor thermal status | Prevent throttling |

### ❌ Don't

| Avoid | Reason |
|-------|--------|
| Building without RPC flag | No `rpc-server` binary |
| Binding to 127.0.0.1 | Workers unreachable from network |
| Using mmap on Android | Memory issues, segfaults |
| Running without wake lock | Android kills background processes |
| Ignoring thermal limits | Devices throttle and crash |
| Mismatched tensor-split ratios | Memory imbalance, OOM |
| Using 2.4 GHz WiFi | Higher latency, interference |

### Quick Start Checklist

1. [ ] Build llama.cpp with `GGML_RPC=ON`
2. [ ] Verify `rpc-server` binary exists
3. [ ] Configure `LD_LIBRARY_PATH`
4. [ ] Identify all device IPs
5. [ ] Calculate tensor-split ratios
6. [ ] Download quantized model (Q4_K_M)
7. [ ] Start workers with `--host 0.0.0.0`
8. [ ] Start master with `--rpc` and `--tensor-split`
9. [ ] Test with `/health` endpoint
10. [ ] Enable thermal monitoring

---

## EXO Integration

EXO handles distributed llama.cpp inference automatically. The key components:

| Component | File | Purpose |
|-----------|------|---------|
| `RpcServerManager` | `src/exo/worker/engines/llamacpp/rpc_server.py` | Manages worker `rpc-server` |
| `DistributedLlamaServer` | `src/exo/worker/engines/llamacpp/utils.py` | Manages master `llama-server` |
| `build_rpc_address_list` | `src/exo/worker/engines/llamacpp/utils.py` | Builds `--rpc` addresses |
| `build_tensor_split_string` | `src/exo/worker/engines/llamacpp/utils.py` | Builds `--tensor-split` |

When you run EXO:

1. Workers automatically start `rpc-server` on their assigned port
2. Master builds the RPC address list from cluster topology
3. Master starts `llama-server` with `--rpc` pointing to all workers
4. Tensor split is calculated based on device capabilities
5. Inference requests are distributed across the cluster

---

## References

- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [GGML RPC Backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-rpc)
- [llama.cpp Android Tutorial](https://github.com/JackZeng0208/llama.cpp-android-tutorial)
- [PRIMA.CPP: Distributed LLM Inference](https://arxiv.org/abs/2504.08791)
- [EXO Android Setup](./ANDROID_SETUP.md)
- [Termux Advanced Topics](./TERMUX_ADVANCED.md)
- [ARM Optimization Guide](./ARM_OPTIMIZATION.md)

---

*Last updated: December 2024*

