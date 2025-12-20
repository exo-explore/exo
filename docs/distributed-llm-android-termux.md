# Distributed LLM Inference Across Android Devices Using Termux

This document explores how large language models (LLMs) can be split across multiple Android devices using Termux, and how these devices can be connected together with other devices to form a collaborative inference cluster.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Framework](#theoretical-framework)
3. [Model Partitioning Strategies](#model-partitioning-strategies)
4. [Setting Up Termux on Android](#setting-up-termux-on-android)
5. [Running LLMs on a Single Android Device](#running-llms-on-a-single-android-device)
6. [Distributed Inference Architecture](#distributed-inference-architecture)
7. [Inter-Device Communication](#inter-device-communication)
8. [Load Balancing and Orchestration](#load-balancing-and-orchestration)
9. [Connecting Android Cluster to Other Devices](#connecting-android-cluster-to-other-devices)
10. [Practical Considerations](#practical-considerations)
11. [Existing Projects and Research](#existing-projects-and-research)
12. [Future Directions](#future-directions)

---

## Overview

Large language models require substantial memory and computational resources that typically exceed the capabilities of individual mobile devices. A 70B parameter model, for example, requires approximately 35-140 GB of RAM depending on quantization level—far beyond what any single Android device can provide.

**The Solution**: Distribute the model across multiple devices, where each device:
- Holds a portion of the model weights
- Performs computations on its assigned layers/tensors
- Exchanges intermediate activations with other devices

This approach leverages the **collective resources** of many devices to run models that would otherwise be impossible on mobile hardware.

---

## Theoretical Framework

### Why Distributed Inference Works

LLM inference is fundamentally a sequence of matrix multiplications and non-linear operations flowing through layers. This computational structure allows for natural parallelization:

```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
          ↓          ↓              ↓
       Device A   Device B      Device C
```

Each device processes its assigned portion and passes the result to the next device in the pipeline.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Model Parallelism** | Splitting the model's parameters across devices |
| **Pipeline Parallelism** | Assigning consecutive layers to different devices |
| **Tensor Parallelism** | Splitting individual tensors (weight matrices) across devices |
| **Data Parallelism** | Running the same model on multiple devices with different inputs |

For distributed inference on heterogeneous devices like Android phones, **pipeline parallelism** is most practical due to simpler communication patterns.

---

## Model Partitioning Strategies

### Layer-wise Partitioning

The most straightforward approach splits the model by transformer layers:

```
LLaMA-70B has 80 layers
├── Device 1: Layers 0-19   (embedding + first 20 layers)
├── Device 2: Layers 20-39
├── Device 3: Layers 40-59
└── Device 4: Layers 60-79  (final layers + output head)
```

**Advantages:**
- Simple to implement
- Minimal communication (only activation tensors between devices)
- Works well with GGUF/GGML format models

**Disadvantages:**
- Uneven layer computation times can cause bottlenecks
- Requires sequential processing (no true parallelism)

### Tensor Parallelism

Splits individual weight matrices across devices:

```
Attention Weight Matrix [4096 × 4096]
├── Device 1: [4096 × 1024]
├── Device 2: [4096 × 1024]
├── Device 3: [4096 × 1024]
└── Device 4: [4096 × 1024]
```

**Advantages:**
- True parallel computation
- Better resource utilization
- Up to 1.8x speedup on 2 devices, 3.2x on 4 devices (per EXO benchmarks)

**Disadvantages:**
- Requires frequent communication (every layer)
- High bandwidth requirements
- Complex synchronization

### Optimized Assignment

Using linear optimization to match model segments with device capabilities:

```python
# Pseudocode for optimized assignment
def assign_layers(model_layers, devices):
    device_capabilities = [measure_capability(d) for d in devices]
    layer_costs = [estimate_cost(l) for l in model_layers]
    
    # Solve assignment problem to minimize total inference time
    assignment = linear_optimize(
        layer_costs,
        device_capabilities,
        minimize=max_device_time  # Balance load
    )
    return assignment
```

---

## Setting Up Termux on Android

### Installation

1. **Download Termux** from F-Droid (not Play Store—outdated):
   ```
   https://f-droid.org/packages/com.termux/
   ```

2. **Grant Storage Access**:
   ```bash
   termux-setup-storage
   ```

3. **Update Package Manager**:
   ```bash
   pkg update && pkg upgrade -y
   ```

4. **Install Essential Build Tools**:
   ```bash
   pkg install -y git clang make cmake wget curl openssh python
   ```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Android Version | 7.0+ | 10.0+ |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB free | 50+ GB free |
| CPU | ARM64 | Snapdragon 8 Gen 1+ |

### Optional: GPU Acceleration

Some Android devices support GPU acceleration via:
- **Vulkan** (via Kompute or custom backends)
- **OpenCL** (limited support)
- **NNAPI** (Android's Neural Networks API)

However, CPU inference with NEON optimizations is often more reliable on Termux.

---

## Running LLMs on a Single Android Device

### Using llama.cpp

The most mature solution for running LLMs on Android:

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)

# Download a small quantized model
mkdir -p models && cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Run inference
cd ..
./llama-cli -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --color \
  --ctx-size 2048 \
  -n 256 \
  --temp 0.7 \
  -t 4
```

### Using Ollama

```bash
# Download Ollama for ARM64
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (background)
ollama serve &

# Pull and run a model
ollama run tinyllama
```

### Model Size Guidelines

| Model Size | Quantization | RAM Required | Suitable Devices |
|------------|--------------|--------------|------------------|
| 1-3B | Q4_K_M | 2-4 GB | Most phones |
| 7B | Q4_K_M | 6-8 GB | Flagship phones |
| 13B | Q4_K_M | 10-14 GB | Tablets, high-end |
| 70B | Q4_K_M | 40+ GB | **Requires distribution** |

---

## Distributed Inference Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COORDINATOR NODE                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Task Scheduler│  │ Load Balancer │  │ Result Aggreg │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ TCP/IP or WebSocket
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Android 1  │    │  Android 2  │    │  Android 3  │
    │ Layers 0-26 │───▶│ Layers 27-53│───▶│ Layers 54-79│
    │ (Termux)    │    │ (Termux)    │    │ (Termux)    │
    └─────────────┘    └─────────────┘    └─────────────┘
```

### Data Flow for Inference

1. **Input Tokenization**: Coordinator tokenizes user prompt
2. **Forward Pass Start**: Tokens sent to Device 1
3. **Layer Processing**: Each device processes its layers, forwards activations
4. **Output Generation**: Final device produces next token logits
5. **Token Sampling**: Coordinator samples next token
6. **Iteration**: Repeat until generation complete

### Activation Transfer

Between devices, only **activation tensors** need to be transferred:

```
For a 70B model with 8192 hidden dimension:
- Activation size per token: 8192 × 2 bytes (FP16) = 16 KB
- For batch size 1, sequence length 512: ~8 MB per forward pass
- With 4 devices: 3 transfers × 8 MB = 24 MB total
```

---

## Inter-Device Communication

### Network Options

| Method | Bandwidth | Latency | Setup Complexity |
|--------|-----------|---------|------------------|
| Wi-Fi (Same Network) | 50-150 Mbps | 5-20 ms | Low |
| Wi-Fi Direct | 100-250 Mbps | 3-10 ms | Medium |
| USB Tethering | 300-480 Mbps | 1-3 ms | Medium |
| Bluetooth | 2-3 Mbps | 50-100 ms | Low |

**Recommended**: Wi-Fi with static IPs on same network for simplicity.

### SSH-Based Communication

```bash
# On each device, set up SSH
pkg install openssh
ssh-keygen -t ed25519

# Start SSH daemon
sshd

# Check IP address
ip addr show wlan0 | grep inet

# Exchange keys between devices
ssh-copy-id -p 8022 u0_a123@192.168.1.101
```

### Socket-Based Communication (Python Example)

**Server (receiving device):**
```python
import socket
import pickle
import numpy as np

def start_activation_server(port=5000):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    
    while True:
        conn, addr = server.accept()
        data = b''
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        
        activations = pickle.loads(data)
        # Process activations through local layers
        output = process_layers(activations)
        # Send to next device or return result
        conn.close()
```

**Client (sending device):**
```python
def send_activations(activations, next_device_ip, port=5000):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((next_device_ip, port))
    
    data = pickle.dumps(activations)
    client.sendall(data)
    client.close()
```

### Using ZeroMQ for Robust Communication

```bash
pip install pyzmq
```

```python
import zmq

# Publisher (sending activations)
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://192.168.1.102:5555")
socket.send_pyobj(activations)

# Worker (receiving and processing)
context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5555")
activations = receiver.recv_pyobj()
```

---

## Load Balancing and Orchestration

### Dynamic Load Balancing

Monitor device performance and redistribute work:

```python
import psutil
import time

class DeviceMonitor:
    def __init__(self, device_id):
        self.device_id = device_id
        self.metrics_history = []
    
    def collect_metrics(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_available': psutil.virtual_memory().available,
            'timestamp': time.time()
        }
    
    def should_redistribute(self, threshold=90):
        """Returns True if device is overloaded"""
        recent = self.metrics_history[-5:]
        avg_cpu = sum(m['cpu_percent'] for m in recent) / len(recent)
        return avg_cpu > threshold
```

### Task Queue Architecture

```python
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class InferenceTask:
    priority: int
    sequence_id: str = field(compare=False)
    tokens: list = field(compare=False)
    
class TaskScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.task_queue = PriorityQueue()
        self.device_loads = {d: 0 for d in devices}
    
    def assign_task(self, task):
        # Find least loaded device
        target = min(self.device_loads, key=self.device_loads.get)
        self.device_loads[target] += 1
        return target
```

---

## Connecting Android Cluster to Other Devices

### Hybrid Cluster Architecture

Android devices can join a larger cluster with desktops, laptops, or servers:

```
┌────────────────────────────────────────────────────────────────┐
│                    MASTER COORDINATOR                           │
│                   (Desktop/Server)                              │
└────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────────┐
│   ANDROID CLUSTER   │               │    DESKTOP CLUSTER      │
│  ┌───────┐ ┌───────┐│               │   ┌────────┐ ┌────────┐ │
│  │Phone 1│ │Phone 2││               │   │MacBook │ │Linux PC│ │
│  │Layers │ │Layers ││               │   │ Layers │ │ Layers │ │
│  │0-19   │ │20-39  ││               │   │ 40-59  │ │ 60-79  │ │
│  └───────┘ └───────┘│               │   └────────┘ └────────┘ │
└─────────────────────┘               └─────────────────────────┘
```

### Using EXO for Heterogeneous Clusters

EXO provides automatic device discovery and topology-aware parallelism:

```bash
# On each device (including Android via Termux)
git clone https://github.com/exo-explore/exo
cd exo/dashboard && npm i && npm run build && cd ..
uv run exo
```

Devices automatically discover each other and form a cluster.

### API Gateway Pattern

Expose the Android cluster via a unified API:

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

ANDROID_NODES = [
    "http://192.168.1.101:8080",
    "http://192.168.1.102:8080",
    "http://192.168.1.103:8080",
]

@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
    # Route to first available Android node
    async with httpx.AsyncClient() as client:
        for node in ANDROID_NODES:
            try:
                response = await client.post(
                    f"{node}/inference",
                    json=request,
                    timeout=30.0
                )
                return response.json()
            except:
                continue
    return {"error": "No available nodes"}
```

---

## Practical Considerations

### Performance Expectations

| Metric | Single Phone | 4-Phone Cluster |
|--------|--------------|-----------------|
| Max Model Size | 7B Q4 | 30B Q4 |
| Tokens/Second (7B) | 3-8 tok/s | 8-15 tok/s |
| Tokens/Second (30B) | N/A | 2-5 tok/s |
| Network Overhead | 0% | 15-30% |

### Battery and Thermal Management

```bash
# Monitor battery in Termux
termux-battery-status

# Reduce CPU frequency to prevent throttling
# (requires root)
echo 1500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
```

**Recommendations:**
- Keep devices plugged in during inference
- Use cooling pads or fans
- Implement thermal throttling in software
- Rotate workloads between devices

### Network Reliability

```python
import time
from functools import wraps

def retry_on_network_failure(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator
```

### Security Considerations

1. **Encrypt all inter-device communication** (TLS/SSH)
2. **Use VPN for communication over untrusted networks**
3. **Implement authentication between nodes**
4. **Validate all incoming data**

---

## Existing Projects and Research

### LinguaLinked

Academic research on distributed LLM inference for mobile devices:
- Optimized model assignment using linear programming
- Runtime load balancing
- Efficient data transmission protocols

**Paper**: [LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices](https://arxiv.org/abs/2312.00388)

### Petals

BitTorrent-style distributed inference for large models:
- Each node hosts a portion of the model
- Nodes can join/leave dynamically
- Fault-tolerant design

**GitHub**: https://github.com/bigscience-workshop/petals

### MLC LLM

Machine Learning Compilation for LLMs with Android support:
- GPU acceleration via Vulkan
- Optimized for mobile deployment
- WebGPU support for browser-based inference

**GitHub**: https://github.com/mlc-ai/mlc-llm

### llama.cpp

The de-facto standard for running LLMs on consumer hardware:
- GGUF format for quantized models
- ARM NEON optimizations
- Builds natively on Termux

**GitHub**: https://github.com/ggerganov/llama.cpp

---

## Future Directions

### Emerging Technologies

1. **RDMA over Wi-Fi 7**: Ultra-low latency wireless communication
2. **NPU Integration**: Dedicated AI accelerators in mobile SoCs
3. **Mesh Networking**: Ad-hoc device clusters without infrastructure
4. **Edge TPUs**: Google Coral-style accelerators for mobile

### Research Areas

- **Speculative Decoding**: Parallel token generation for faster inference
- **Dynamic Model Compression**: Adjusting quantization based on device load
- **Federated Learning Integration**: Training while inferencing
- **Checkpoint Migration**: Moving model state between devices

### EXO + Android Vision

Integrating Android devices into EXO's automatic discovery and tensor parallelism:
- Automatic capability detection
- Seamless inclusion in existing clusters
- Mobile-optimized communication protocols

---

## Conclusion

Distributing LLM inference across Android devices using Termux is **technically feasible** and opens possibilities for:

- Running larger models than any single phone could handle
- Creating portable, infrastructure-free AI clusters
- Democratizing access to AI capabilities

**Key Takeaways:**

1. **Start with llama.cpp** for mature, tested mobile inference
2. **Use Wi-Fi on same network** for simplest communication
3. **Layer-wise partitioning** is most practical for mobile
4. **Expect 3-8 tokens/second** on a multi-phone cluster for 7B models
5. **Consider battery and thermal** constraints carefully
6. **Integrate with EXO** for connection to larger clusters

The combination of improving mobile hardware, better quantization techniques, and frameworks like EXO makes distributed mobile LLM inference increasingly practical.

---

## References

1. [LinguaLinked: A Distributed LLM Inference System for Mobile Devices](https://arxiv.org/abs/2312.00388)
2. [Distributed Mixture-of-Agents for Edge Inference](https://arxiv.org/abs/2412.21200)
3. [MLC LLM: Machine Learning Compilation for LLMs](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices)
4. [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
5. [Petals: Collaborative Inference](https://github.com/bigscience-workshop/petals)
6. [EXO: Run Your Own AI Cluster](https://github.com/exo-explore/exo)
7. [Termux Wikipedia](https://en.wikipedia.org/wiki/Termux)

