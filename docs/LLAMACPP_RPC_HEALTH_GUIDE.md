# llama.cpp RPC Backend: Health Endpoint & Readiness Guide

> **Last Updated:** December 2024  
> **Applies to:** llama.cpp distributed inference with RPC backend

This document provides comprehensive documentation on llama.cpp's `/health` endpoint behavior, RPC backend readiness, known issues, and best practices for production deployments.

---

## Table of Contents

1. [Overview](#overview)
2. [Health Endpoint Behavior](#health-endpoint-behavior)
3. [RPC Server Architecture](#rpc-server-architecture)
4. [Known Issues & Fixes](#known-issues--fixes)
5. [Configuration Options](#configuration-options)
6. [Best Practices](#best-practices)
7. [Debugging & Troubleshooting](#debugging--troubleshooting)
8. [Client-Side Implementation](#client-side-implementation)
9. [Production Deployment](#production-deployment)
10. [References](#references)

---

## Overview

The llama.cpp project provides an RPC (Remote Procedure Call) backend to facilitate **distributed inference** across multiple machines. The `/health` endpoint serves as the primary mechanism for determining server readiness.

### Key Components

| Component | Purpose |
|-----------|---------|
| `llama-server` | Main inference server with HTTP API |
| `rpc-server` | Worker node for distributed tensor operations |
| `/health` endpoint | Server readiness indicator |
| `/slots` endpoint | Processing slot availability |
| `/metrics` endpoint | Prometheus-compatible metrics |

---

## Health Endpoint Behavior

### Response States

The `/health` endpoint has two primary response states:

#### 1. Loading State (HTTP 503)

```json
{
  "error": {
    "code": 503,
    "message": "Loading model",
    "type": "unavailable_error"
  }
}
```

**When returned:**
- Model is being loaded into memory
- Tensors are being allocated
- RPC connections are being established
- Server is initializing

#### 2. Ready State (HTTP 200)

```json
{
  "status": "ok"
}
```

**When returned:**
- Model is fully loaded
- All tensors are allocated and ready
- Server can accept inference requests

### Extended Health Response

When slots are involved, the response may include additional information:

```json
{
  "status": "no slot available",
  "slots_idle": 0,
  "slots_processing": 32
}
```

> **Important:** The `/health` endpoint returning 503 during model loading is **expected behavior**, not a bug.

---

## RPC Server Architecture

### Distributed Inference Flow

```
┌─────────────────┐     RPC      ┌─────────────────┐
│   Master Node   │◄────────────►│   Worker Node   │
│ (llama-server)  │              │  (rpc-server)   │
│   device_rank=0 │              │  device_rank>0  │
└─────────────────┘              └─────────────────┘
        │                                │
        ▼                                ▼
   Model Layers                    Model Layers
   (first N%)                      (remaining %)
```

### Master Node Configuration

```bash
llama-server \
  -m /path/to/model.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --rpc "worker1:50052,worker2:50052" \
  --tensor-split "0.5,0.25,0.25"
```

### Worker Node Configuration

```bash
rpc-server \
  --host 0.0.0.0 \
  --port 50052
```

### Port Assignments

| Device Rank | Role | Default Port |
|-------------|------|--------------|
| 0 | Master (llama-server) | 8080 (HTTP) |
| 1 | Worker 1 (rpc-server) | 50052 |
| 2 | Worker 2 (rpc-server) | 50053 |
| N | Worker N (rpc-server) | 50052 + (N-1) |

---

## Known Issues & Fixes

### Issue 1: Persistent 503 During Loading

**Symptoms:**
- Server returns 503 indefinitely
- Model never finishes loading
- High CPU usage without progress

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Insufficient memory | Reduce model size or use quantization |
| Flash Attention failure in mixed environments | Disable Flash Attention (see below) |
| Automatic parameter fitting failure | Use `-fit off` flag |
| Network timeout to RPC workers | Increase connection timeout |

**Disabling Flash Attention (Mixed Metal/CUDA environments):**

```bash
# For CUDA
export GGML_CUDA_NO_FA=1

# Or use the --no-flash-attn flag
llama-server -m model.gguf --no-flash-attn
```

> **Reference:** [GitHub Issue #12655](https://github.com/ggml-org/llama.cpp/issues/12655)

### Issue 2: RPC Backend Segmentation Faults

**Symptoms:**
- Server crashes during model loading
- Segfault in "fitting params to device memory" phase

**Solution:**

```bash
# Disable automatic memory fitting
llama-server -m model.gguf --rpc "worker:50052" -fit off
```

> **Reference:** [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126)

### Issue 3: GPU Utilization Capped at 50%

**Symptoms:**
- Dual-node RPC setup shows 50% GPU utilization
- Single-node achieves 100% utilization

**Potential Causes:**
- Tensor split not optimized
- Network bottleneck
- Synchronization overhead

**Mitigation:**

```bash
# Experiment with tensor split ratios
llama-server -m model.gguf \
  --rpc "worker:50052" \
  --tensor-split "0.6,0.4"  # Try different ratios
```

> **Reference:** [GitHub Issue #15463](https://github.com/ggml-org/llama.cpp/issues/15463)

### Issue 4: Health Check Timeouts During Large Batch Processing

**Symptoms:**
- `/health` times out when server is busy
- False negative health reports

**Root Cause:**
- `/health` and `/slots` endpoints share task processing
- Large batches block health check responses

**Recommended Solution:**
- Separate health check from slot status checks
- Use dedicated `/health` for liveness
- Use `/slots` for workload status

> **Reference:** [GitHub Discussion #9276](https://github.com/ggml-org/llama.cpp/discussions/9276)

### Issue 5: Server Hangs During Model Loading

**Symptoms:**
- Server stops responding during load
- No error messages
- CPU usage remains high

**Debugging Steps:**

```bash
# Enable debug output
export GGML_RPC_DEBUG=1

# Run with verbose logging
llama-server -m model.gguf --verbose
```

> **Reference:** [GitHub Issue #15128](https://github.com/ggml-org/llama.cpp/issues/15128)

---

## Configuration Options

### Server Flags Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Bind address | 127.0.0.1 |
| `--port` | HTTP port | 8080 |
| `-m, --model` | Model path | (required) |
| `--rpc` | RPC worker addresses | (none) |
| `--tensor-split` | Layer distribution ratios | auto |
| `-c, --ctx-size` | Context size | 2048 |
| `-t, --threads` | CPU threads | auto |
| `-ngl, --n-gpu-layers` | GPU offload layers | 0 |
| `--no-mmap` | Disable memory mapping | false |
| `--no-flash-attn` | Disable Flash Attention | false |
| `-fit` | Memory fitting mode | auto |
| `--no-slots` | Disable /slots endpoint | false |
| `--verbose` | Enable verbose logging | false |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GGML_RPC_DEBUG=1` | Enable RPC debug messages |
| `GGML_CUDA_NO_FA=1` | Disable CUDA Flash Attention |
| `LLAMA_N_THREADS` | Override thread count |
| `LLAMA_N_CTX` | Override context size |
| `LLAMA_N_GPU_LAYERS` | Override GPU layers |
| `LLAMA_N_BATCH` | Override batch size |

### RPC Server Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Bind address | 0.0.0.0 |
| `--port` | RPC port | 50052 |
| `--mem` | Memory limit (bytes) | auto |

---

## Best Practices

### 1. Health Check Implementation

**DO:**
- Poll `/health` endpoint before routing traffic
- Implement exponential backoff on 503 responses
- Separate liveness checks from readiness checks
- Set appropriate timeouts (5-30 seconds)

**DON'T:**
- Treat TCP connection success as readiness
- Assume 503 means server failure
- Use aggressive health check intervals during loading

### 2. Client-Side Retry Logic

```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def wait_for_server_ready(
    url: str,
    timeout: int = 300,
    check_interval: float = 2.0
) -> bool:
    """Wait for llama-server to be ready."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{url}/health",
                timeout=5
            )
            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                # Expected during loading - continue waiting
                pass
            else:
                # Unexpected status
                print(f"Unexpected status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            # Server not yet accepting connections
            pass
        except requests.exceptions.Timeout:
            # Health check timed out
            pass
        
        time.sleep(check_interval)
    
    return False
```

### 3. Distributed Deployment Checklist

- [ ] Build llama.cpp with RPC support: `cmake -B build -DGGML_RPC=ON`
- [ ] Start worker `rpc-server` instances BEFORE master
- [ ] Verify worker ports are accessible from master
- [ ] Configure firewall rules for RPC ports (50052+)
- [ ] Set appropriate tensor split ratios
- [ ] Monitor server logs during initial load
- [ ] Implement health check in orchestration

### 4. Memory and Resource Management

```bash
# Calculate tensor split based on available VRAM
# Example: 2 nodes with 24GB and 16GB VRAM
# Total: 40GB, ratios: 24/40 = 0.6, 16/40 = 0.4

llama-server \
  -m large-model.gguf \
  --rpc "worker:50052" \
  --tensor-split "0.6,0.4" \
  --ctx-size 4096 \
  --threads 8
```

### 5. Startup Order

1. **Start worker RPC servers first:**
   ```bash
   # On each worker node
   rpc-server --host 0.0.0.0 --port 50052
   ```

2. **Wait for workers to be ready:**
   ```bash
   # Verify TCP connectivity
   nc -zv worker1 50052
   nc -zv worker2 50052
   ```

3. **Start master server:**
   ```bash
   llama-server \
     -m model.gguf \
     --rpc "worker1:50052,worker2:50052" \
     --tensor-split "0.5,0.25,0.25"
   ```

4. **Wait for `/health` to return 200:**
   ```bash
   while ! curl -s localhost:8080/health | grep -q '"ok"'; do
     echo "Waiting for server..."
     sleep 2
   done
   echo "Server ready!"
   ```

---

## Debugging & Troubleshooting

### Enable Debug Logging

```bash
# RPC debug messages
export GGML_RPC_DEBUG=1

# Verbose server output
llama-server -m model.gguf --verbose 2>&1 | tee server.log
```

### Check Network Connectivity

```bash
# Test RPC port accessibility
nc -zv worker-ip 50052

# Check if port is listening
ss -tlnp | grep 50052

# Test from master to workers
for worker in worker1 worker2; do
  echo "Testing $worker..."
  timeout 5 bash -c "</dev/tcp/$worker/50052" && echo "OK" || echo "FAILED"
done
```

### Monitor Server State

```bash
# Continuous health monitoring
watch -n 1 'curl -s localhost:8080/health | jq .'

# Check slot status
curl -s localhost:8080/slots | jq .

# Get metrics
curl -s localhost:8080/metrics
```

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Loading model...` (stuck) | Memory exhausted | Reduce model size |
| `Connection refused` to RPC | Worker not started | Start rpc-server first |
| `Segmentation fault` | Memory fitting issue | Use `-fit off` |
| `Flash Attention failure` | Mixed GPU backends | Disable FA |
| `No slot available` | Server busy | Increase parallel slots |

### Log Analysis

Look for these key messages in server output:

```
# Good signs:
"model loaded"
"server listening"
"all slots are idle"

# Warning signs:
"failed to load model"
"out of memory"
"RPC connection failed"
"tensor allocation failed"
```

---

## Client-Side Implementation

### Python Health Check with Retries

```python
import requests
from typing import Optional
import time

class LlamaServerClient:
    def __init__(
        self,
        base_url: str,
        startup_timeout: int = 300,
        health_check_timeout: int = 5
    ):
        self.base_url = base_url.rstrip('/')
        self.startup_timeout = startup_timeout
        self.health_check_timeout = health_check_timeout
    
    def wait_until_ready(self) -> bool:
        """Block until server is ready or timeout."""
        start = time.time()
        last_status = None
        
        while time.time() - start < self.startup_timeout:
            try:
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.health_check_timeout
                )
                
                if resp.status_code == 200:
                    return True
                
                if resp.status_code == 503:
                    data = resp.json()
                    msg = data.get('error', {}).get('message', 'Loading')
                    if msg != last_status:
                        print(f"Server status: {msg}")
                        last_status = msg
                        
            except requests.exceptions.ConnectionError:
                if last_status != "connecting":
                    print("Waiting for server to start...")
                    last_status = "connecting"
            except requests.exceptions.Timeout:
                if last_status != "timeout":
                    print("Health check timeout, retrying...")
                    last_status = "timeout"
            
            time.sleep(2)
        
        return False
    
    def is_healthy(self) -> bool:
        """Check if server is currently healthy."""
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=self.health_check_timeout
            )
            return resp.status_code == 200
        except:
            return False
    
    def get_slot_status(self) -> Optional[dict]:
        """Get current slot status."""
        try:
            resp = requests.get(
                f"{self.base_url}/slots",
                timeout=self.health_check_timeout
            )
            return resp.json() if resp.status_code == 200 else None
        except:
            return None
```

### Bash Health Check Script

```bash
#!/bin/bash

SERVER_URL="${1:-http://localhost:8080}"
TIMEOUT="${2:-300}"
INTERVAL="${3:-2}"

echo "Waiting for $SERVER_URL to be ready (timeout: ${TIMEOUT}s)..."

start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for server"
        exit 1
    fi
    
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 5 \
        --max-time 10 \
        "${SERVER_URL}/health" 2>/dev/null)
    
    case $status in
        200)
            echo "Server is ready! (took ${elapsed}s)"
            exit 0
            ;;
        503)
            echo "[$elapsed/${TIMEOUT}s] Loading model..."
            ;;
        000)
            echo "[$elapsed/${TIMEOUT}s] Connection failed, retrying..."
            ;;
        *)
            echo "[$elapsed/${TIMEOUT}s] Unexpected status: $status"
            ;;
    esac
    
    sleep $INTERVAL
done
```

---

## Production Deployment

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: llama-server
        image: your-llama-image
        ports:
        - containerPort: 8080
        
        # Startup probe - allows long model loading
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 60  # 10 minutes total
        
        # Liveness probe - restart if truly unhealthy
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 0
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        # Readiness probe - don't route traffic until ready
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 0
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 1
```

### Docker Compose with Health Checks

```yaml
version: '3.8'

services:
  rpc-worker:
    image: llama-cpp-rpc
    command: rpc-server --host 0.0.0.0 --port 50052
    ports:
      - "50052:50052"
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "50052"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s

  llama-server:
    image: llama-cpp-server
    command: >
      llama-server
      -m /models/model.gguf
      --host 0.0.0.0
      --port 8080
      --rpc rpc-worker:50052
    ports:
      - "8080:8080"
    depends_on:
      rpc-worker:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 60s
```

### Nginx Load Balancer Configuration

```nginx
upstream llama_servers {
    server llama1:8080;
    server llama2:8080 backup;
}

server {
    listen 80;
    
    location /health {
        proxy_pass http://llama_servers/health;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
    }
    
    location / {
        proxy_pass http://llama_servers;
        proxy_connect_timeout 30s;
        proxy_read_timeout 300s;
        
        # Only route to healthy backends
        proxy_next_upstream error timeout http_503;
        proxy_next_upstream_tries 2;
    }
}
```

### llama-swap Integration

For dynamic model loading with [llama-swap](https://github.com/ggml-org/llama-swap):

```yaml
# config.yaml
models:
  "llama-70b":
    cmd: "llama-server --port 8999 -m /models/llama-70b.gguf --rpc worker:50052"
    proxy: "http://127.0.0.1:8999"
    checkEndpoint: "/health"
    ttl: 300
    
  "qwen-32b":
    cmd: "llama-server --port 8999 -m /models/qwen-32b.gguf"
    proxy: "http://127.0.0.1:8999"
    checkEndpoint: "/health"
    ttl: 300
```

---

## References

### Official Documentation

- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [llama.cpp RPC README](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)

### GitHub Issues & Discussions

| Issue | Topic |
|-------|-------|
| [#9276](https://github.com/ggml-org/llama.cpp/discussions/9276) | Health vs Slots endpoint separation |
| [#15128](https://github.com/ggml-org/llama.cpp/issues/15128) | Server hangs during model loading |
| [#15463](https://github.com/ggml-org/llama.cpp/issues/15463) | RPC dual-node GPU utilization |
| [#12655](https://github.com/ggml-org/llama.cpp/issues/12655) | Flash Attention failure in mixed environments |
| [#9682](https://github.com/ggerganov/llama.cpp/discussions/9682) | RPC backend segmentation faults |
| [#5850](https://github.com/ggml-org/llama.cpp/issues/5850) | Metrics endpoint improvements |
| [#14566](https://github.com/ggml-org/llama.cpp/issues/14566) | HTTP 200 with error in streamed chunk |

### Related Projects

- [llama-swap](https://github.com/ggml-org/llama-swap) - Dynamic model loading
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings

---

## Summary

### Key Takeaways

1. **503 during loading is expected** - The `/health` endpoint correctly returns 503 while the model loads

2. **No built-in delay flags** - There are no specific flags to delay serving; the health check IS the mechanism

3. **Wait for 200 before routing** - Always poll `/health` and wait for 200 before sending requests

4. **Start workers before master** - RPC workers must be running before the master server starts

5. **Use appropriate timeouts** - Model loading can take minutes; set startup timeouts accordingly

6. **Mixed environments need care** - Disable Flash Attention in mixed Metal/CUDA setups

7. **Monitor logs during development** - Use `GGML_RPC_DEBUG=1` and `--verbose` for debugging

### Quick Reference

```bash
# Start worker
rpc-server --host 0.0.0.0 --port 50052

# Start master with RPC
llama-server -m model.gguf --rpc "worker:50052" --tensor-split "0.5,0.5"

# Wait for ready
while ! curl -sf localhost:8080/health; do sleep 2; done

# Check status
curl localhost:8080/health
curl localhost:8080/slots
curl localhost:8080/metrics
```

---

*This document is based on research conducted in December 2024. For the latest information, always consult the official llama.cpp repository.*

