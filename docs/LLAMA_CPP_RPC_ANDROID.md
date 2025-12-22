# llama.cpp Distributed RPC on Android/Termux

> **Comprehensive guide for distributed inference using llama.cpp RPC on Android/Termux**

This document covers known limitations, required build flags, performance optimizations, stability notes, and troubleshooting for running llama.cpp with RPC (Remote Procedure Call) offloading on ARM/Android platforms.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & How RPC Works](#architecture--how-rpc-works)
3. [Known Limitations](#known-limitations)
4. [Required Build Flags](#required-build-flags)
5. [Building llama.cpp with RPC on Termux](#building-llamacpp-with-rpc-on-termux)
6. [RPC Server Configuration](#rpc-server-configuration)
7. [Client Configuration](#client-configuration)
8. [Memory Management](#memory-management)
9. [Performance Optimization](#performance-optimization)
10. [Stability & Troubleshooting](#stability--troubleshooting)
11. [32-bit ARM Considerations](#32-bit-arm-considerations)
12. [Security Considerations](#security-considerations)
13. [Network Requirements](#network-requirements)
14. [Best Practices Checklist](#best-practices-checklist)
15. [GitHub Issues Reference](#github-issues-reference)

---

## Overview

llama.cpp supports distributed inference via RPC, allowing you to split model layers across multiple devices. This is particularly useful for:

- Running large models that don't fit in a single device's memory
- Utilizing multiple Android devices as a compute cluster
- Offloading computation from a less powerful client to powerful servers

**Key Components:**

| Component | Description |
|-----------|-------------|
| `rpc-server` | Runs on worker nodes, handles tensor computations |
| `llama-cli` / `llama-server` | Client that coordinates inference, can use `--rpc` flag |
| GGML_RPC | Build flag to enable RPC functionality |

---

## Architecture & How RPC Works

### Distributed Inference Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT DEVICE                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  llama-cli / llama-server                               │    │
│  │  - Loads model metadata                                  │    │
│  │  - Coordinates tensor splits                             │    │
│  │  - Handles tokenization/detokenization                   │    │
│  │  - Runs layers assigned to local device                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              │ RPC calls                         │
│                              ▼                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────────┐                 ┌─────────────────────┐
│   RPC SERVER 1      │                 │   RPC SERVER 2      │
│   (Android Device)  │                 │   (Android Device)  │
│                     │                 │                     │
│   - Runs rpc-server │                 │   - Runs rpc-server │
│   - Holds tensors   │                 │   - Holds tensors   │
│   - Executes ops    │                 │   - Executes ops    │
│   - Port: 50052     │                 │   - Port: 50052     │
└─────────────────────┘                 └─────────────────────┘
```

### Tensor Split Concept

The model's layers are distributed across devices using `--tensor-split` or `-ts`:

```bash
# Example: 3 devices, split 40% / 30% / 30%
llama-cli --rpc 192.168.1.101:50052,192.168.1.102:50052 \
          --tensor-split 0.4,0.3,0.3 \
          -m model.gguf -p "Hello"
```

---

## Known Limitations

### 1. Memory Constraints (Critical)

**Issue:** Offloading large models via RPC can crash if model size exceeds ~75% of server RAM.

| Condition | Result |
|-----------|--------|
| Model < 75% RAM | ✅ Stable |
| Model 75-85% RAM | ⚠️ May crash during tensor loading |
| Model > 85% RAM | ❌ Almost certain crash |

**Root Cause:** During tensor loading, both the model weights AND temporary buffers must fit in memory simultaneously. The tensor loading process creates memory pressure that causes OOM.

**GitHub Issue:** [#15055](https://github.com/ggml-org/llama.cpp/issues/15055)

**Workaround:**
- Use `--tensor-split` to ensure each server handles < 70% of its RAM
- Use smaller quantizations (Q4_K_M instead of Q5_K_M)
- Close other apps on the device

---

### 2. Performance Regressions on Android aarch64

**Issue:** NEON optimizations and i8mm detection have known issues in newer llama.cpp versions.

**Symptoms:**
- Slower inference than expected
- Q4_0_4_4 format shows drastically reduced performance
- Repacking for Q4_0 and IQ4_NL may be ineffective

**GitHub Issue:** [#10662](https://github.com/ggml-org/llama.cpp/issues/10662)

**Workaround:**
- Test different llama.cpp versions (some older versions may perform better)
- Use Q4_K_M or Q5_K_M quantizations instead of Q4_0_4_4
- Check CPU feature detection with `cat /proc/cpuinfo | grep Features`

---

### 3. Android 15 Linker Crashes

**Issue:** Building complex binaries on Android 15 + Termux can result in linker crashes due to Memory Tagging Extension (MTE) tagged pointer truncation.

**Affected Binaries:**
- `llama-quantize`
- Some test binaries
- Larger/more complex targets

**Simpler binaries like `llama-cli` and `rpc-server` usually build successfully.**

**Source:** [Weekly GitHub Report - October 2025](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-october-20-2025-3191/)

**Workaround:**
- Use targeted builds (`cmake --build build --target rpc-server`)
- Add `-fno-sanitize=memtag` to CFLAGS if available
- Use older Termux/NDK versions if possible

---

### 4. Local Model Copy Requirement

**Issue:** RPC offloading requires the model to be transmitted over the network to each server during initialization.

**Impact:**
- 100 Mbit connection → Very slow initialization (minutes for large models)
- 1 Gbit connection → Reasonable initialization times

**GitHub Discussion:** [#9740](https://github.com/ggml-org/llama.cpp/discussions/9740)

**Workaround:**
- Store a local copy of the model on each RPC server
- Use the same model path on all devices
- Pre-download models before starting the cluster

---

### 5. Single-Threaded GPU Offload

**Issue:** When the entire model is offloaded to GPU, llama.cpp uses only a single thread regardless of `--threads` argument.

**GitHub Issue:** [#8684](https://github.com/ggml-org/llama.cpp/issues/8684)

**Note:** This primarily affects GPU offload, not CPU-based RPC workers.

---

### 6. Limited NPU/DSP Integration

**Issue:** llama.cpp lacks deep integration with mobile NPU/DSP accelerators.

**Impact:**
- Cannot leverage Qualcomm Hexagon DSP
- No MediaTek APU support
- No Samsung NPU acceleration

**Current Status:** Partial Vulkan support exists but performance is device/driver dependent.

---

## Required Build Flags

### Essential CMake Flags

| Flag | Purpose | Default | Recommended |
|------|---------|---------|-------------|
| `GGML_RPC=ON` | Enable RPC server/client | OFF | **ON** (required) |
| `BUILD_SHARED_LIBS=ON` | Build shared libraries | OFF | ON |
| `GGML_NO_MMAP=ON` | Disable memory mapping | OFF | Test both |
| `LLAMA_NATIVE=OFF` | Disable -march=native | ON | **OFF** (safer) |

### Build Command for Termux

```bash
# Clone
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure with RPC enabled
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_RPC=ON

# Build all targets
cmake --build build --config Release -j4

# Or build specific targets
cmake --build build --target llama-cli rpc-server -j4
```

### Memory Mapping (mmap) Considerations

| Configuration | Pros | Cons | When to Use |
|---------------|------|------|-------------|
| **mmap ON** (default) | Faster loading, lower peak RAM | May cause page faults, stability issues on low RAM | Devices with 8GB+ RAM |
| **mmap OFF** (`--no-mmap`) | More stable on low RAM devices | Slower loading, higher RAM usage | Devices with <6GB RAM |

**To disable mmap at build time:**
```bash
cmake -B build -DGGML_NO_MMAP=ON ...
```

**To disable mmap at runtime:**
```bash
llama-cli --no-mmap -m model.gguf ...
```

---

## Building llama.cpp with RPC on Termux

### Prerequisites

```bash
pkg update && pkg upgrade -y
pkg install git cmake ninja clang
```

### Standard Build (64-bit ARM)

```bash
cd ~/llama.cpp

# Clean previous build
rm -rf build

# Configure with RPC
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_RPC=ON

# Build (use -j2 if OOM, -j4 if enough RAM)
cmake --build build --config Release -j4

# Verify binaries
ls -la build/bin/llama-cli build/bin/rpc-server
```

### Optimized Build with Architecture Flags

For better performance, specify your device's CPU architecture:

```bash
# Detect CPU features
cat /proc/cpuinfo | grep Features

# Example for Cortex-A76 class (Snapdragon 855/865)
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_RPC=ON \
    -DCMAKE_C_FLAGS="-O3 -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16" \
    -DCMAKE_CXX_FLAGS="-O3 -mcpu=cortex-a76 -march=armv8.2-a+dotprod+fp16"

cmake --build build --config Release -j4
```

### CPU Flag Reference

| Device Class | SoC Example | Recommended Flags |
|--------------|-------------|-------------------|
| 2024 Flagship | Snapdragon 8 Gen 3 | `-mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16` |
| 2023 Flagship | Snapdragon 8 Gen 2 | `-mcpu=cortex-x3 -march=armv9-a+sve2` |
| 2022 Flagship | Snapdragon 8 Gen 1 | `-mcpu=cortex-x2 -march=armv9-a+sve2` |
| 2021 Flagship | Snapdragon 888 | `-mcpu=cortex-x1 -march=armv8.2-a+dotprod+fp16` |
| 2020 Flagship | Snapdragon 865 | `-mcpu=cortex-a77 -march=armv8.2-a+dotprod+fp16` |
| Mid-range | Snapdragon 778G | `-mcpu=cortex-a78 -march=armv8.2-a+dotprod` |
| Budget | Helio G80 | `-mcpu=cortex-a75 -march=armv8.2-a` |
| Generic safe | Any ARMv8.2+ | `-march=armv8.2-a+dotprod` |

---

## RPC Server Configuration

### Starting the RPC Server

```bash
# Basic usage
./rpc-server -H 0.0.0.0 -p 50052

# With specific thread count (match your big cores)
./rpc-server -H 0.0.0.0 -p 50052 -t 4

# With memory limit (experimental)
./rpc-server -H 0.0.0.0 -p 50052 --mem 6G
```

### RPC Server Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-H, --host` | Host/IP to bind | `127.0.0.1` |
| `-p, --port` | Port to listen on | `50052` |
| `-t, --threads` | Number of CPU threads | Auto-detect |
| `-c` | CPU backend only | - |

### Running as Background Service

```bash
# Using nohup
nohup ./rpc-server -H 0.0.0.0 -p 50052 > rpc.log 2>&1 &

# Using screen
pkg install screen
screen -dmS rpc ./rpc-server -H 0.0.0.0 -p 50052

# Using tmux
pkg install tmux
tmux new-session -d -s rpc './rpc-server -H 0.0.0.0 -p 50052'
```

### Auto-Start on Boot (Termux:Boot)

```bash
mkdir -p ~/.termux/boot

cat > ~/.termux/boot/01-rpc-server.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
sleep 10  # Wait for network
cd ~/llama.cpp/build/bin
./rpc-server -H 0.0.0.0 -p 50052 >> ~/rpc.log 2>&1 &
EOF

chmod +x ~/.termux/boot/01-rpc-server.sh
```

---

## Client Configuration

### Basic RPC Client Usage

```bash
# Single RPC server
llama-cli --rpc 192.168.1.101:50052 \
          -m ~/models/model.gguf \
          -p "Hello, how are you?"

# Multiple RPC servers
llama-cli --rpc 192.168.1.101:50052,192.168.1.102:50052 \
          -m ~/models/model.gguf \
          -p "Hello, how are you?"
```

### Tensor Split Configuration

Distribute model layers across devices:

```bash
# 3 devices: 40% local, 30% server1, 30% server2
llama-cli --rpc 192.168.1.101:50052,192.168.1.102:50052 \
          --tensor-split 0.4,0.3,0.3 \
          -m ~/models/model.gguf \
          -p "Hello"

# 2 devices: 50% each
llama-cli --rpc 192.168.1.101:50052 \
          --tensor-split 0.5,0.5 \
          -m ~/models/model.gguf \
          -p "Hello"
```

### Using with llama-server (API mode)

```bash
# Start llama-server with RPC backends
llama-server --rpc 192.168.1.101:50052,192.168.1.102:50052 \
             --tensor-split 0.4,0.3,0.3 \
             -m ~/models/model.gguf \
             --host 0.0.0.0 \
             --port 8080
```

### Client Command Line Options for RPC

| Option | Description |
|--------|-------------|
| `--rpc HOST:PORT,...` | Comma-separated list of RPC servers |
| `--tensor-split N,N,...` | Fraction of model for each device |
| `-ngl, --n-gpu-layers` | Layers to offload (per device) |
| `--no-mmap` | Disable memory mapping |
| `-t, --threads` | Number of threads for local compute |

---

## Memory Management

### RAM Guidelines for Android Devices

| Device RAM | Max Model Size (Single Device) | Recommended Quantization |
|------------|-------------------------------|-------------------------|
| 4 GB | ~2.5 GB model | Q4_K_M, Q3_K_M |
| 6 GB | ~4 GB model | Q4_K_M, Q5_K_M |
| 8 GB | ~5.5 GB model | Q4_K_M, Q5_K_M, Q6_K |
| 12 GB | ~8 GB model | Q5_K_M, Q6_K, Q8_0 |
| 16 GB | ~11 GB model | Any |

### Tensor Split Memory Calculation

```
Each device needs: (Total Model Size × Split Fraction) + Overhead (~500MB)
```

**Example: 7B Q4_K_M model (~4 GB) across 3 devices:**
```
Device 1 (40%): 4 GB × 0.4 + 0.5 GB = ~2.1 GB
Device 2 (30%): 4 GB × 0.3 + 0.5 GB = ~1.7 GB
Device 3 (30%): 4 GB × 0.3 + 0.5 GB = ~1.7 GB
```

### Enabling Swap (Not Recommended but Possible)

```bash
# Create swap file (requires root or specific Termux setup)
# This is generally NOT recommended for performance
dd if=/dev/zero of=$HOME/.swapfile bs=1M count=2048
mkswap $HOME/.swapfile
swapon $HOME/.swapfile

# Verify
free -h
```

**Warning:** Swap on Android/Termux severely degrades performance and may cause system instability.

---

## Performance Optimization

### Thread Configuration

Match thread count to your device's big/performance cores:

| SoC Type | Big Cores | Recommended `-t` |
|----------|-----------|------------------|
| Snapdragon 8 Gen 3 | 1 X4 + 5 A720 | `-t 4` to `-t 6` |
| Snapdragon 888 | 1 X1 + 3 A78 | `-t 4` |
| Snapdragon 778G | 4 A78 | `-t 4` |
| Mid-range | 2-4 big cores | `-t 2` to `-t 4` |

### Batch Size and Context

```bash
# Reduce batch size for memory-constrained devices
llama-cli -b 256 ...  # Default is 512

# Reduce context size
llama-cli -c 2048 ...  # Default is often 4096+
```

### Quantization Performance Comparison

| Quantization | Size (7B) | Speed | Quality | RAM Usage |
|--------------|-----------|-------|---------|-----------|
| Q2_K | ~2.5 GB | Fastest | Lower | Lowest |
| Q3_K_M | ~3.0 GB | Fast | Fair | Low |
| Q4_K_M | ~4.0 GB | Good | Good | Medium |
| Q5_K_M | ~5.0 GB | Slower | Better | Higher |
| Q6_K | ~5.5 GB | Slowest | Best | High |
| Q8_0 | ~7.0 GB | Slow | Excellent | Highest |

**Recommended for Android:** Q4_K_M (best balance of size/speed/quality)

### Network Performance Impact

| Connection Speed | Init Time (7B model) | Inference Impact |
|------------------|---------------------|------------------|
| WiFi 5 (~500 Mbps) | ~20-30 seconds | Minimal |
| WiFi 4 (~150 Mbps) | ~60-90 seconds | Noticeable |
| 100 Mbps Ethernet | ~90-120 seconds | High |
| 5 GHz WiFi (ideal) | ~10-20 seconds | Minimal |

**Recommendation:** Use 5 GHz WiFi or better for RPC. Store local model copies on each device.

---

## Stability & Troubleshooting

### Common Issues and Solutions

#### 1. RPC Server Crashes During Model Load

**Symptoms:**
```
rpc-server: Out of memory
Connection reset by peer
```

**Solutions:**
- Reduce tensor split allocation for that server
- Use smaller quantization (Q4_K_M → Q3_K_M)
- Close other apps, free up RAM
- Add `--no-mmap` flag

#### 2. Client Hangs During Initialization

**Symptoms:**
```
Loading model...
(hangs indefinitely)
```

**Solutions:**
- Check network connectivity: `ping server_ip`
- Verify RPC server is running: `netstat -tlnp | grep 50052`
- Check firewall settings
- Try with single RPC server first

#### 3. Slow Token Generation

**Symptoms:**
- < 1 token/second when expecting more

**Solutions:**
- Check CPU frequency: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
- Verify threads setting matches big cores
- Check for thermal throttling
- Reduce context length

#### 4. NEON/SIMD Not Used

**Symptoms:**
- Very slow inference
- No SIMD instructions in CPU usage

**Solutions:**
- Verify build flags include NEON: `strings llama-cli | grep -i neon`
- Rebuild with explicit architecture flags
- Check `/proc/cpuinfo` for `asimd` feature

#### 5. Model Corrupted Error

**Symptoms:**
```
error: model file corrupted
invalid magic
```

**Solutions:**
- Re-download the model
- Verify model SHA256 hash
- Check if model was fully transferred over RPC

### Diagnostic Commands

```bash
# Check RPC server status
netstat -tlnp | grep 50052

# Check available memory
free -h

# Check CPU frequency/throttling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Check thermal status
cat /sys/class/thermal/thermal_zone*/temp

# Check CPU features
cat /proc/cpuinfo | grep Features

# Network connectivity test
ping -c 3 192.168.1.101

# Check llama.cpp version
./llama-cli --version
```

---

## 32-bit ARM Considerations

### Building on 32-bit ARM (armv7l/armv8l)

32-bit ARM builds require special handling:

```bash
# Disable native flags that may not work
make clean
make CC=clang CXX=clang++ \
    CFLAGS="-O3 -mcpu=cortex-a8 -mfpu=neon -mfloat-abi=hard" \
    CXXFLAGS="-O3 -mcpu=cortex-a8 -mfpu=neon -mfloat-abi=hard" \
    LLAMA_NATIVE=OFF
```

### Known 32-bit Limitations

| Issue | Impact | Workaround |
|-------|--------|------------|
| 4GB address space limit | Cannot load models > ~3GB | Use highly quantized models |
| Limited SIMD | Slower inference | Optimize for NEON |
| Unsupported compiler flags | Build failures | Use explicit arch flags |
| Memory pressure | More crashes | Use smaller models |

### Recommended Model Sizes for 32-bit

| Model | Quantization | Size | Viable? |
|-------|--------------|------|---------|
| TinyLlama 1.1B | Q4_K_M | ~700 MB | ✅ Yes |
| Qwen 2.5 0.5B | Q4_K_M | ~400 MB | ✅ Yes |
| Qwen 2.5 1.5B | Q4_K_M | ~1 GB | ✅ Yes |
| Llama 3.2 1B | Q4_K_M | ~750 MB | ✅ Yes |
| Llama 3.2 3B | Q4_K_M | ~2 GB | ⚠️ Marginal |
| 7B models | Any | ~4+ GB | ❌ No |

---

## Security Considerations

### RPC Is NOT Secure by Default

**Warning:** The llama.cpp RPC protocol has NO built-in authentication or encryption.

**Risks:**
- Anyone on the network can connect to your RPC server
- Model weights can be intercepted
- Prompts/responses are transmitted in plaintext

**Mitigations:**

1. **Network Isolation**
   ```bash
   # Only bind to trusted interface
   ./rpc-server -H 192.168.1.0 -p 50052  # Specific interface only
   ```

2. **Firewall Rules (if available)**
   ```bash
   # Allow only specific IPs (requires iptables access)
   iptables -A INPUT -p tcp --dport 50052 -s 192.168.1.100 -j ACCEPT
   iptables -A INPUT -p tcp --dport 50052 -j DROP
   ```

3. **VPN/SSH Tunnel**
   ```bash
   # SSH tunnel from client to server
   ssh -L 50052:localhost:50052 user@rpc-server
   
   # Then connect to localhost
   llama-cli --rpc localhost:50052 ...
   ```

4. **Private WiFi Network**
   - Use a dedicated router for your AI cluster
   - Enable WPA3 encryption
   - Disable guest access

### GPU Memory Leak Risks

Research has shown potential for GPU memory leaks to expose sensitive data. While less relevant for CPU-only Android, be aware if using Vulkan GPU acceleration.

**Reference:** [arxiv.org/abs/2401.16603](https://arxiv.org/abs/2401.16603)

---

## Network Requirements

### Minimum Requirements

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Bandwidth | 100 Mbps | 500+ Mbps |
| Latency | < 50ms | < 10ms |
| Connection | WiFi 4 | WiFi 5/6 or Ethernet |
| Network Type | Same subnet | Same switch/AP |

### Bandwidth Calculation

```
Approximate bandwidth needed = (Model Size × Layers on Server) / Acceptable Init Time

Example:
- 7B Q4_K_M model = 4 GB
- Server handles 50% = 2 GB
- Acceptable init = 30 seconds
- Required bandwidth = 2 GB / 30s ≈ 70 MB/s ≈ 560 Mbps
```

### Network Optimization Tips

1. **Use 5 GHz WiFi** - Less interference, higher bandwidth
2. **Same Access Point** - Minimize routing hops
3. **Wired Connection (if possible)** - USB-C to Ethernet adapter
4. **Static IPs** - Prevent DHCP issues
5. **QoS Priority** - If router supports, prioritize RPC traffic

---

## Best Practices Checklist

### Before Starting

- [ ] All devices on same WiFi network (5 GHz preferred)
- [ ] Static IPs assigned or known
- [ ] Termux battery optimization disabled
- [ ] Other apps closed to free RAM
- [ ] Model downloaded to all devices (if using local copies)
- [ ] Wake lock acquired: `termux-wake-lock`

### Build Configuration

- [ ] Built with `GGML_RPC=ON`
- [ ] Using architecture-optimized flags
- [ ] Tested on all target devices
- [ ] `rpc-server` binary exists and works

### Memory Setup

- [ ] Each device has RAM for its tensor split + 500MB overhead
- [ ] Using appropriate quantization for device RAM
- [ ] `--no-mmap` tested on low-RAM devices
- [ ] Swap disabled (for stability)

### Network Setup

- [ ] RPC servers reachable from client
- [ ] Port 50052 open on all devices
- [ ] Tested with single server before multi-server
- [ ] Network stable (no drops during test)

### Runtime

- [ ] RPC servers started before client
- [ ] Tensor split totals to 1.0
- [ ] Thread count matches big cores
- [ ] Monitoring for thermal throttling

---

## GitHub Issues Reference

### Critical Issues to Watch

| Issue | Title | Impact |
|-------|-------|--------|
| [#15055](https://github.com/ggml-org/llama.cpp/issues/15055) | RPC tensor loading crash at 75%+ RAM | High |
| [#10662](https://github.com/ggml-org/llama.cpp/issues/10662) | Android aarch64 NEON performance regression | High |
| [#8684](https://github.com/ggml-org/llama.cpp/issues/8684) | Single-threaded GPU offload | Medium |
| [#9740](https://github.com/ggml-org/llama.cpp/discussions/9740) | RPC local model copy proposal | Feature |
| [#13801](https://github.com/ggml-org/llama.cpp/issues/13801) | Vulkan build failures in Termux | Medium |

### Related Discussions

- [Distributed inference with llama.cpp (ARM)](https://learn.arm.com/learning-paths/servers-and-cloud-computing/distributed-inference-with-llama-cpp/)
- [Android phone running Gemma 270M](https://medium.com/@itsme_rahul/android-phone-running-gemma-3-270m-in-llama-cpp-for-tiny-sentiment-tasks-6dbf76e1f034)

---

## Quick Reference Commands

### Start RPC Server

```bash
cd ~/llama.cpp/build/bin
./rpc-server -H 0.0.0.0 -p 50052 -t 4
```

### Start Client with RPC

```bash
# Single server
./llama-cli --rpc 192.168.1.101:50052 -m model.gguf -p "Hello"

# Multiple servers with tensor split
./llama-cli \
    --rpc 192.168.1.101:50052,192.168.1.102:50052 \
    --tensor-split 0.5,0.25,0.25 \
    -m model.gguf \
    -t 4 \
    --no-mmap \
    -p "Hello, how are you?"
```

### Build with RPC

```bash
cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --target llama-cli rpc-server -j4
```

### Debug Connectivity

```bash
# On server
netstat -tlnp | grep 50052

# On client
nc -vz 192.168.1.101 50052
```

---

## See Also

- [Android Setup Guide](./ANDROID_SETUP.md) - Basic EXO/Termux installation
- [ARM Optimization Guide](./ARM_OPTIMIZATION.md) - CPU-specific tuning
- [Termux Advanced Topics](./TERMUX_ADVANCED.md) - Background execution, auto-start
- [Android Model Issues](./ANDROID_MODEL_ISSUES.md) - Model-specific troubleshooting

---

## Appendix: Related Documentation

This document is part of a comprehensive guide for llama.cpp on Android/Termux. For additional topics, see:

| Document | Focus Area |
|----------|------------|
| [LLAMA_CPP_RPC_ANDROID.md](./LLAMA_CPP_RPC_ANDROID.md) | **This document** - Limitations, build flags, troubleshooting |
| [LLAMACPP_SHARDING_TERMUX.md](./LLAMACPP_SHARDING_TERMUX.md) | Tensor split, cluster setup, EXO integration |
| [LLAMACPP_RPC_NETWORKING.md](./LLAMACPP_RPC_NETWORKING.md) | TCP tuning, port configuration, diagnostics |
| [ARM_OPTIMIZATION.md](./ARM_OPTIMIZATION.md) | CPU-specific compiler flags, SoC specifications |
| [TERMUX_ADVANCED.md](./TERMUX_ADVANCED.md) | SSH, background execution, auto-start |
| [ANDROID_MODEL_ISSUES.md](./ANDROID_MODEL_ISSUES.md) | Model download and inference issues |

---

*Last updated: December 2024*
*This document is part of the EXO project: https://github.com/exo-explore/exo*

