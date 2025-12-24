# RPC Distributed Inference Debugging Findings

## Summary

**Manual llama.cpp RPC works perfectly. EXO orchestration has a timing/startup issue.**

## Test Environment

- **3 Android phones** running Termux
- **llama.cpp version**: `7f459c98e` (tag b7527)
- **EXO branch**: `Multiple-Nodes-Sharding` (commit `bf65fad`)
- **Model**: Qwen2.5-0.5B-Instruct-GGUF (469 MB)

## What Works ✅

### Manual RPC Setup

**Workers** (run on each worker phone):
```bash
export GGML_RPC_DEBUG=1
~/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 60000
```

**Master** (run on master phone):
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/.exo/models/Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --port 8080 --host 127.0.0.1 \
  --rpc 10.99.0.78:60000,10.99.0.152:60000 \
  --tensor-split 0.33,0.33,0.34 \
  --no-mmap -c 1024 --verbose
```

**Result**: After ~1 minute of initial connect/disconnect cycles, the model loads successfully:
- Workers show tensor operations: `[set_tensor]`, `[get_tensor]`, `[graph_compute]`
- Master shows: `main: model loaded`, `server is listening on http://127.0.0.1:8080`

## What Doesn't Work ❌

### EXO Dashboard Launch

When clicking "Launch" in the EXO dashboard:
1. Workers start rpc-server correctly
2. Master starts llama-server with correct `--rpc` flags
3. **But**: The connect/disconnect loop never stabilizes
4. Health checks stay at 503 forever
5. Eventually times out after 600s

## Root Cause Analysis

The llama.cpp RPC protocol has an **initial handshake phase** where:
1. Master connects to each worker
2. Queries device capabilities (`[get_alloc_size]`, `[hello]`)
3. May disconnect and reconnect multiple times
4. Eventually stabilizes and starts tensor transfer

**The key observation**: This handshake takes ~60 seconds in our manual test before becoming stable.

### Likely EXO Issues

1. **Health check interference**: EXO polls `/health` every second while the model is loading. This might interfere with the RPC handshake.

2. **Timeout too aggressive**: Although the timeout is 600s, something else might be killing the process or interfering.

3. **Process spawning order**: EXO might be starting llama-server before all workers are fully initialized and listening.

4. **Environment differences**: The subprocess environment EXO creates might differ from a direct shell.

## Recommended Fixes

### 1. Add Worker Ready Verification

Before starting llama-server, verify each worker is truly ready:
```python
def verify_worker_ready(host: str, port: int) -> bool:
    # Don't just check TCP connection
    # Actually verify rpc-server responds to protocol
    pass
```

### 2. Increase Initial Wait Time

Add a deliberate delay after workers start before launching master:
```python
# Wait for workers to fully initialize
time.sleep(5)  # Give rpc-servers time to stabilize
```

### 3. Reduce Health Check Frequency During Load

During the initial model load phase, reduce health check frequency:
```python
# First 2 minutes: check every 10 seconds
# After that: check every 1 second
```

### 4. Log the Exact Command

Ensure EXO logs the exact command being run so we can compare with manual:
```python
logger.info(f"Full command: {' '.join(command)}")
logger.info(f"Environment: LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH')}")
```

## Next Steps

1. Add a 5-10 second delay between worker startup and master startup
2. Verify workers are responding to RPC protocol (not just TCP)
3. Compare the exact command/environment EXO uses vs manual
4. Consider reducing health check frequency during initial load

## Appendix: Successful Manual Log Sequence

### Worker Logs (during model load)
```
Accepted client connection
[hello] version: 3.6.0
[get_alloc_size] device: 0, buffer: 0x0, data: 0x0
recv returned 0 (peer closed?)
Client connection closed
[~socket_t] closing socket 4
... (repeats for ~60 seconds) ...
[set_tensor] buffer: 0xb400007e1f67afd0, data: 0xb400007c54036800, offset: 0, size: 7168
[graph_compute] device: 0, n_nodes: 443, n_tensors: 630
```

### Master Logs (success)
```
main: model loaded
main: server is listening on http://127.0.0.1:8080
main: starting the main loop...
que  start_loop: processing new tasks
srv  update_slots: all slots are idle
que  start_loop: waiting for new tasks
```

