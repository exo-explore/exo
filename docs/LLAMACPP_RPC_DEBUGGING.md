# llama.cpp RPC Server Debugging & Diagnostics Guide

> **Comprehensive guide for diagnosing hangs, 503 errors, and performance issues in llama.cpp RPC distributed inference**

This document covers verbose logging configuration, tensor offload tracing, health endpoint debugging, and profiling techniques for llama.cpp RPC servers.

---

## Table of Contents

1. [Quick Reference: Environment Variables](#quick-reference-environment-variables)
2. [Verbose RPC Logging](#verbose-rpc-logging)
3. [Command-Line Debugging Flags](#command-line-debugging-flags)
4. [Understanding the /health Endpoint](#understanding-the-health-endpoint)
5. [Diagnosing Persistent 503 Errors](#diagnosing-persistent-503-errors)
6. [Tensor Offload Monitoring](#tensor-offload-monitoring)
7. [Network-Level Debugging](#network-level-debugging)
8. [System-Level Profiling](#system-level-profiling)
9. [CUDA/GPU Debugging](#cudagpu-debugging)
10. [Prometheus Metrics](#prometheus-metrics)
11. [Common Issues & Solutions](#common-issues--solutions)
12. [Android/Termux Specific Issues](#androidtermux-specific-issues)
13. [Advanced Debugging Techniques](#advanced-debugging-techniques)
14. [Debugging Checklist](#debugging-checklist)

---

## Quick Reference: Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `GGML_RPC_DEBUG=1` | Enable RPC server debug messages | `export GGML_RPC_DEBUG=1` |
| `LLAMA_LOG_VERBOSITY=0` | Max verbosity (0 = all messages) | `export LLAMA_LOG_VERBOSITY=0` |
| `LLAMA_LOG_TIMESTAMPS=1` | Add timestamps to logs | `export LLAMA_LOG_TIMESTAMPS=1` |
| `LLAMA_LOG_COLORS=1` | Colorized log output | `export LLAMA_LOG_COLORS=1` |
| `LLAMA_CACHE` | Custom cache directory | `export LLAMA_CACHE=/path/to/cache` |
| `CUDA_LAUNCH_BLOCKING=1` | Sync CUDA for debugging | `export CUDA_LAUNCH_BLOCKING=1` |
| `NCCL_DEBUG=TRACE` | NCCL operation tracing | `export NCCL_DEBUG=TRACE` |

---

## Verbose RPC Logging

### Enable Debug Messages for rpc-server

The `GGML_RPC_DEBUG` environment variable activates detailed debug output from the RPC server:

```bash
# Enable RPC debug logging
export GGML_RPC_DEBUG=1

# Start the RPC server with debug output
./bin/rpc-server -H 0.0.0.0 -p 50052
```

This outputs detailed information about:
- Incoming tensor operations
- Memory allocations
- Data transfers
- Backend operations
- Connection states

### Enable Debug Messages for llama-server

```bash
# Maximum verbosity with all debug features
export LLAMA_LOG_VERBOSITY=0
export LLAMA_LOG_TIMESTAMPS=1
export LLAMA_LOG_COLORS=1

./bin/llama-server \
    --verbose \
    --log-file server.log \
    --log-timestamps \
    --log-colors \
    -m /path/to/model.gguf
```

### Log to File for Analysis

```bash
# Redirect all output to file
./bin/rpc-server -H 0.0.0.0 -p 50052 2>&1 | tee rpc-server.log

# Or with llama-server
./bin/llama-server --log-file server.log --verbose
```

---

## Command-Line Debugging Flags

### llama-server Flags

| Flag | Description |
|------|-------------|
| `--verbose` | Enable verbose logging |
| `--log-verbose` | Set verbosity to maximum (infinity) |
| `--log-verbosity N` | Set verbosity threshold (lower = more verbose) |
| `--log-file FILE` | Write logs to specified file |
| `--log-colors` | Enable colored output |
| `--log-timestamps` | Include timestamps in logs |
| `--log-prefix` | Add prefix to log messages |

### rpc-server Flags

| Flag | Description |
|------|-------------|
| `-H HOST` | Host to bind to (default: 0.0.0.0) |
| `-p PORT` | Port to listen on (default: 50052) |
| `-c` | Enable local tensor cache |
| `-m MEM` | Memory limit for the server |
| `--verbose` | Enable verbose output |

### Example: Full Debug Setup

```bash
# RPC Worker Node (run on each worker)
export GGML_RPC_DEBUG=1
./bin/rpc-server -H 0.0.0.0 -p 50052 -c --verbose 2>&1 | tee worker.log

# Master Node (connects to workers)
export LLAMA_LOG_VERBOSITY=0
export LLAMA_LOG_TIMESTAMPS=1
./bin/llama-server \
    --verbose \
    --log-file master.log \
    -m model.gguf \
    --rpc 192.168.1.10:50052,192.168.1.11:50052 \
    --tensor-split 0.5,0.5
```

---

## Understanding the /health Endpoint

### Health Check States

| HTTP Status | State | Meaning |
|-------------|-------|---------|
| `200 OK` | Healthy | Server ready, model loaded |
| `503 Service Unavailable` | Loading | Model still loading |
| `503 Service Unavailable` | Error | Backend issue or resource exhaustion |

### Health Check Response Format

**Healthy (200):**
```json
{
  "status": "ok"
}
```

**Loading (503):**
```json
{
  "error": {
    "code": 503,
    "message": "Loading model",
    "type": "unavailable_error"
  }
}
```

### Polling the Health Endpoint

```bash
# Simple health check
curl -s http://localhost:8080/health

# With timeout and verbose
curl -v --max-time 5 http://localhost:8080/health

# Watch health status continuously
watch -n 1 'curl -s http://localhost:8080/health | jq .'

# Wait for healthy status (bash script)
until curl -s http://localhost:8080/health | grep -q '"status":"ok"'; do
    echo "Waiting for server..."
    sleep 2
done
echo "Server is ready!"
```

---

## Diagnosing Persistent 503 Errors

### Common Causes

1. **Model Loading Stuck**
   - Model file too large for available memory
   - Slow disk I/O (especially on Android/eMMC)
   - Memory-mapping (mmap) issues

2. **RPC Worker Not Responding**
   - Network connectivity issues
   - Worker crashed or hung
   - Firewall blocking RPC port

3. **Resource Exhaustion**
   - Insufficient RAM
   - GPU memory full
   - CPU overloaded

4. **Tensor Transfer Hanging**
   - Network bottleneck
   - RPC cache corruption
   - Mismatched tensor split configuration

### Diagnostic Steps

```bash
# 1. Check if server process is running
pgrep -a llama-server
pgrep -a rpc-server

# 2. Check server logs
tail -f server.log

# 3. Check system resources
htop
free -h
nvidia-smi  # if using GPU

# 4. Check network connectivity to RPC workers
nc -zv 192.168.1.10 50052
nc -zv 192.168.1.11 50052

# 5. Check for stuck processes
strace -p $(pgrep llama-server) -e read,write,recvfrom,sendto

# 6. Check open connections
netstat -an | grep 50052
ss -tuanp | grep 50052
```

### Timeout Configuration

If the model takes a long time to load, increase your health check timeout:

```python
# Python example
import requests

def wait_for_health(url: str, timeout: int = 300) -> bool:
    """Wait for server to become healthy."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
            print(f"Status: {resp.status_code} - still loading...")
        except requests.exceptions.ConnectionError:
            print("Connection refused - server starting...")
        except requests.exceptions.Timeout:
            print("Request timeout - server busy...")
        time.sleep(2)
    return False
```

---

## Tensor Offload Monitoring

### Configuring Tensor Split

The `--tensor-split` flag controls how tensors are distributed across devices:

```bash
# Equal split between 2 devices
./llama-server --rpc host1:50052,host2:50052 --tensor-split 0.5,0.5

# 3:1 ratio (first device gets 75%)
./llama-server --rpc host1:50052,host2:50052 --tensor-split 0.75,0.25

# Based on memory availability (default behavior)
./llama-server --rpc host1:50052,host2:50052
# Automatically proportions based on each device's available memory
```

### Enable Local Caching

The RPC server can cache tensors locally to avoid repeated network transfers:

```bash
# Enable cache (stored in $HOME/.cache/llama.cpp/rpc by default)
./bin/rpc-server -c

# Custom cache location
export LLAMA_CACHE=/fast/ssd/llama-cache
./bin/rpc-server -c
```

### Monitor Tensor Transfer

```bash
# Watch network traffic on RPC port
sudo tcpdump -i any port 50052 -n

# Count bytes transferred
sudo tcpdump -i any port 50052 -n -q | head -100

# Check interface statistics
ip -s link show eth0

# Monitor bandwidth usage
iftop -i eth0 -f "port 50052"
```

### GPU Layer Offloading

```bash
# Offload 10 layers to GPU
./llama-server --n-gpu-layers 10

# Offload all layers (use with care)
./llama-server --n-gpu-layers 999

# Environment variable alternative
export N_GPU_LAYERS=10
```

---

## Network-Level Debugging

### TCP Connection Analysis

```bash
# Check TCP connection state
ss -tuanp | grep 50052

# Watch connection establishment
sudo tcpdump -i any port 50052 -n -S

# Check for retransmissions (indicates network issues)
netstat -s | grep -i retrans

# Detailed socket statistics
ss -i src :50052
```

### Packet Capture for Analysis

```bash
# Capture RPC traffic to file
sudo tcpdump -i any port 50052 -w rpc_traffic.pcap

# Analyze with Wireshark
wireshark rpc_traffic.pcap

# Quick analysis with tshark
tshark -r rpc_traffic.pcap -Y "tcp.analysis.retransmission"
```

### Network Latency Check

```bash
# Ping test to RPC workers
ping -c 10 192.168.1.10

# TCP latency measurement
hping3 -S -p 50052 -c 10 192.168.1.10

# Traceroute to identify bottlenecks
traceroute 192.168.1.10
```

### Firewall Verification

```bash
# Check if port is open
nc -zv 192.168.1.10 50052

# List firewall rules (Linux)
sudo iptables -L -n | grep 50052

# Check if port is listening
ss -tlnp | grep 50052
```

---

## System-Level Profiling

### strace - System Call Tracing

```bash
# Trace all system calls
strace -f -o trace.log ./bin/llama-server -m model.gguf

# Trace specific calls (read/write/network)
strace -e read,write,recvfrom,sendto -p $(pgrep llama-server)

# Count system calls
strace -c ./bin/llama-server -m model.gguf

# Trace with timestamps
strace -tt -f -o trace.log ./bin/llama-server -m model.gguf
```

### perf - Linux Performance Profiler

```bash
# CPU profiling
sudo perf record -g ./bin/llama-server -m model.gguf
sudo perf report

# Flame graph generation
sudo perf record -F 99 -g -- ./bin/llama-server -m model.gguf
sudo perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg

# Cache miss analysis
sudo perf stat -e cache-misses,cache-references ./bin/llama-server -m model.gguf
```

### ltrace - Library Call Tracing

```bash
# Trace library calls
ltrace -e malloc+free -o alloc.log ./bin/llama-server -m model.gguf
```

### gdb - Debugging Hung Process

```bash
# Attach to running process
sudo gdb -p $(pgrep llama-server)

# In gdb:
(gdb) bt           # Backtrace
(gdb) thread apply all bt  # All thread backtraces
(gdb) info threads # List all threads
(gdb) thread 2     # Switch to thread 2
(gdb) bt           # Backtrace for that thread
```

### Core Dump Analysis

```bash
# Enable core dumps
ulimit -c unlimited
echo "/tmp/core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern

# Run server (will create core dump on crash)
./bin/llama-server -m model.gguf

# Analyze core dump
gdb ./bin/llama-server /tmp/core.llama-server.*
(gdb) bt
(gdb) thread apply all bt
```

---

## CUDA/GPU Debugging

### CUDA Synchronous Execution

Force synchronous CUDA execution to identify the exact failing kernel:

```bash
export CUDA_LAUNCH_BLOCKING=1
./bin/llama-server -m model.gguf
```

### NCCL Debugging (Multi-GPU)

```bash
# Trace level logging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# Info level (less verbose)
export NCCL_DEBUG=INFO
```

### nvidia-smi Monitoring

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Detailed process info
nvidia-smi pmon -s um

# Memory and utilization history
nvidia-smi dmon -s pum
```

### CUDA Memory Check

```bash
# Check available GPU memory before starting
nvidia-smi --query-gpu=memory.free --format=csv

# Enable CUDA memory debugging
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEMORY_DEBUG=1
```

### GPU Profiling with Nsight

```bash
# Nsight Systems profiling
nsys profile --trace=cuda,nvtx ./bin/llama-server -m model.gguf

# Nsight Compute for kernel analysis
ncu --set full ./bin/llama-server -m model.gguf
```

---

## Prometheus Metrics

### Enable Metrics Endpoint

```bash
./bin/llama-server --metrics --host 0.0.0.0 --port 8080
```

### Available Metrics

Access at `http://localhost:8080/metrics`:

| Metric | Description |
|--------|-------------|
| `llamacpp_prompt_tokens_seconds` | Prompt processing speed |
| `llamacpp_predicted_tokens_seconds` | Token generation speed |
| `llamacpp_tokens_predicted_total` | Total tokens generated |
| `llamacpp_tokens_predicted_total` | Total tokens in prompts |
| `llamacpp_kv_cache_usage` | KV cache utilization |
| `llamacpp_kv_cache_tokens` | Tokens in KV cache |
| `llamacpp_requests_processing` | Currently processing requests |
| `http_req_duration` | HTTP request duration histogram |

### Prometheus Scrape Config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llama-server'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 5s
```

### Grafana Dashboard Query Examples

```promql
# Tokens per second
rate(llamacpp_tokens_predicted_total[1m])

# Request latency p99
histogram_quantile(0.99, rate(http_req_duration_bucket[5m]))

# KV cache utilization
llamacpp_kv_cache_usage
```

---

## Common Issues & Solutions

### Issue 1: Server Hangs During Model Loading

**Symptoms:**
- `/health` returns 503 indefinitely
- No log output after "loading model"
- High memory usage, no progress

**Solutions:**

```bash
# 1. Try disabling mmap
./bin/llama-server --no-mmap -m model.gguf

# 2. Reduce context size
./bin/llama-server -c 2048 -m model.gguf

# 3. Use smaller batch size
./bin/llama-server --batch-size 128 -m model.gguf

# 4. Check available memory
free -h
# If low, try a smaller model or close other apps
```

### Issue 2: RPC Workers Not Connecting

**Symptoms:**
- "Connection refused" errors
- Master node waiting for workers
- Tensor transfer failures

**Solutions:**

```bash
# 1. Verify worker is running and listening
netstat -tlnp | grep 50052

# 2. Check firewall rules
sudo iptables -L -n | grep 50052
# Allow if blocked:
sudo iptables -A INPUT -p tcp --dport 50052 -j ACCEPT

# 3. Test connectivity
nc -zv worker_ip 50052

# 4. Check worker logs for errors
tail -f worker.log | grep -i error
```

### Issue 3: Slow Tensor Transfer

**Symptoms:**
- Model loading takes very long
- Network usage is low
- Workers show idle time

**Solutions:**

```bash
# 1. Enable local caching on workers
./bin/rpc-server -c

# 2. Check network bandwidth
iperf3 -c worker_ip

# 3. Use faster network if available
# Switch from WiFi to Ethernet

# 4. Verify tensor split is optimal
# Adjust based on actual device capabilities
./bin/llama-server --tensor-split 0.6,0.4
```

### Issue 4: Memory Exhaustion

**Symptoms:**
- Process killed by OOM killer
- "Failed to allocate memory" errors
- System becomes unresponsive

**Solutions:**

```bash
# 1. Reduce number of layers offloaded
./bin/llama-server --n-gpu-layers 5

# 2. Use smaller context
./bin/llama-server -c 1024

# 3. Reduce batch size
./bin/llama-server --batch-size 64

# 4. Check for memory leaks
valgrind --leak-check=full ./bin/llama-server -m model.gguf

# 5. Set memory limit
ulimit -v 8388608  # 8GB limit
```

### Issue 5: Intermittent 503 During Inference

**Symptoms:**
- Server works, then suddenly returns 503
- Happens during long inference runs
- Correlates with multiple requests

**Solutions:**

```bash
# 1. Check for slot exhaustion
curl http://localhost:8080/slots

# 2. Increase slot count
./bin/llama-server --parallel 4

# 3. Monitor slot states
watch 'curl -s http://localhost:8080/slots | jq .'

# 4. Check for CUDA errors that don't surface
export CUDA_LAUNCH_BLOCKING=1
# Re-run and check for CUDA errors in output
```

---

## Android/Termux Specific Issues

### Disable Memory Mapping

Android's storage can have issues with mmap:

```bash
# Always use --no-mmap on Android
./bin/llama-server --no-mmap -m model.gguf

# Or via environment
export EXO_LLAMA_NO_MMAP=1
```

### Wake Lock for Background Processing

```bash
# Prevent Android from killing process
termux-wake-lock

# Run server
./bin/llama-server -m model.gguf

# When done
termux-wake-unlock
```

### Battery Optimization

Disable battery optimization for Termux:
1. Settings → Apps → Termux → Battery → Unrestricted

### Storage Performance

```bash
# Check I/O speed (models should be on fast storage)
dd if=/dev/zero of=testfile bs=1M count=100 oflag=dsync

# Move model to faster storage if needed
# Internal storage is usually faster than SD card
```

### Thermal Throttling

```bash
# Monitor temperature
cat /sys/class/thermal/thermal_zone*/temp

# Add cooling delays if needed
export LLAMA_DELAY_MS=10
```

### Limited Threads

```bash
# Android may have fewer usable cores
# Explicitly set thread count
./bin/llama-server -t 4 -m model.gguf

# Check available cores
nproc
cat /proc/cpuinfo | grep processor | wc -l
```

---

## Advanced Debugging Techniques

### Function Tracing

For detailed function-level tracing (generates large logs):

```bash
export VLLM_TRACE_FUNCTION=1
./bin/llama-server -m model.gguf

# Warning: This creates extremely verbose output
# Only use for specific debugging sessions
```

### Custom Logging Hooks

Add logging to the code for specific debugging:

```cpp
// In llama.cpp source code
#define DEBUG_LOG(msg) fprintf(stderr, "[DEBUG] %s:%d: %s\n", __FILE__, __LINE__, msg)
```

### Network Protocol Analysis

```bash
# Capture and analyze RPC protocol
sudo tcpdump -i any port 50052 -w rpc.pcap -c 1000

# Extract and analyze with Python
python3 << 'EOF'
from scapy.all import rdpcap, TCP
packets = rdpcap('rpc.pcap')
for pkt in packets:
    if TCP in pkt and pkt[TCP].payload:
        print(f"Len: {len(pkt[TCP].payload)}")
EOF
```

### Memory Profiling

```bash
# Massif (heap profiler)
valgrind --tool=massif ./bin/llama-server -m model.gguf
ms_print massif.out.*

# Memcheck
valgrind --leak-check=full --show-leak-kinds=all \
    ./bin/llama-server -m model.gguf 2>&1 | tee valgrind.log
```

### Timing Analysis

```bash
# Add timing to bash scripts
time_start=$(date +%s.%N)
# ... operation ...
time_end=$(date +%s.%N)
echo "Operation took $(echo "$time_end - $time_start" | bc) seconds"

# Use hyperfine for benchmarking
hyperfine --warmup 1 './bin/llama-cli -m model.gguf -p "Hello" -n 10'
```

---

## Debugging Checklist

Use this checklist when troubleshooting RPC issues:

### Pre-Flight Checks

- [ ] All worker nodes have `rpc-server` running
- [ ] Ports are open (default: 50052)
- [ ] Network connectivity verified between nodes
- [ ] Sufficient memory on all nodes
- [ ] Model file exists and is readable

### Enable Debugging

- [ ] Set `GGML_RPC_DEBUG=1` on workers
- [ ] Set `LLAMA_LOG_VERBOSITY=0` on master
- [ ] Enable `--verbose` flag
- [ ] Log to files for analysis

### Health Check Analysis

- [ ] Check `/health` endpoint response
- [ ] Monitor logs during model loading
- [ ] Verify slots are available
- [ ] Check for error messages

### Network Debugging

- [ ] Test TCP connectivity to workers
- [ ] Check for firewall blocks
- [ ] Monitor network traffic
- [ ] Verify no packet loss

### Resource Monitoring

- [ ] Monitor RAM usage on all nodes
- [ ] Check CPU utilization
- [ ] Monitor GPU memory (if applicable)
- [ ] Watch for thermal throttling

### Advanced Analysis

- [ ] Use strace for system call tracing
- [ ] Attach gdb to hung processes
- [ ] Capture core dumps if crashing
- [ ] Profile with perf if slow

---

## Related Documentation

- [Android Setup](./ANDROID_SETUP.md) - Basic exo installation on Android
- [Termux Advanced](./TERMUX_ADVANCED.md) - Advanced Termux configuration
- [Android Model Issues](./ANDROID_MODEL_ISSUES.md) - Model-specific troubleshooting
- [ARM Optimization](./ARM_OPTIMIZATION.md) - CPU-specific tuning

---

## References

- [llama.cpp RPC README](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [llama.cpp GitHub Issues](https://github.com/ggml-org/llama.cpp/issues)
- [vLLM Debugging Guide](https://docs.vllm.ai/en/latest/getting_started/debugging.html)

---

*Last updated: December 2024*

