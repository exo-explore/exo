# Termux for Distributed AI: Complete Guide

> **Build AI Clusters with Android Devices Using Termux**

This comprehensive guide covers everything from Termux fundamentals to advanced distributed LLM inference across Android devices. Whether you're repurposing old smartphones, building edge computing networks, or integrating mobile devices with larger AI clusters, this document provides the complete technical foundation.

---

## Table of Contents

### Part I: Termux Fundamentals
1. [What is Termux?](#1-what-is-termux)
2. [Installation & Initial Setup](#2-installation--initial-setup)
3. [Essential Packages](#3-essential-packages)
4. [Termux Add-ons & APIs](#4-termux-add-ons--apis)
5. [Development Environment](#5-development-environment)
6. [Storage & File System](#6-storage--file-system)

### Part II: Networking & Communication
7. [SSH Server Setup](#7-ssh-server-setup)
8. [Network Discovery & Configuration](#8-network-discovery--configuration)
9. [Inter-Device Communication](#9-inter-device-communication)

### Part III: AI & LLM Inference
10. [Running LLMs on Single Device](#10-running-llms-on-single-device)
11. [Theoretical Framework for Distribution](#11-theoretical-framework-for-distribution)
12. [Model Partitioning Strategies](#12-model-partitioning-strategies)
13. [Distributed Inference Architecture](#13-distributed-inference-architecture)

### Part IV: Cluster Operations
14. [Load Balancing & Orchestration](#14-load-balancing--orchestration)
15. [Connecting to Other Devices (EXO Integration)](#15-connecting-to-other-devices-exo-integration)
16. [Background Execution & Persistence](#16-background-execution--persistence)
17. [Auto-Start on Boot](#17-auto-start-on-boot)

### Part V: Advanced Topics
18. [ADB Automation & Scripting](#18-adb-automation--scripting)
19. [Virtualization with QEMU](#19-virtualization-with-qemu)
20. [Full Linux Distributions (proot-distro)](#20-full-linux-distributions-proot-distro)
21. [Performance Optimization](#21-performance-optimization)
22. [Security Best Practices](#22-security-best-practices)

### Part VI: Reference
23. [Practical Architectures](#23-practical-architectures)
24. [Troubleshooting](#24-troubleshooting)
25. [Existing Projects & Research](#25-existing-projects--research)
26. [Quick Reference](#26-quick-reference)

---

# Part I: Termux Fundamentals

## 1. What is Termux?

**Termux** is an open-source Android terminal emulator and Linux environment that runs directly on Android devices without requiring root access.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Package Manager** | APT-based (`pkg` command) with 1000+ packages |
| **Shell Support** | Bash, Zsh, Fish, and other Unix shells |
| **Development Tools** | Python, Node.js, Ruby, Go, Rust, C/C++, and more |
| **Networking** | Full SSH server/client, wget, curl, networking stack |
| **Architecture** | ARM, ARM64, and x86 Android devices |
| **API Access** | Interact with Android features via Termux:API |
| **No Root Required** | Operates entirely in userspace |

### Why Termux for AI Clusters?

**Advantages:**
- **Hardware Repurposing**: Transform old smartphones/tablets into compute nodes
- **Cost Efficiency**: Leverage existing devices instead of dedicated hardware
- **Portability**: Create truly mobile AI cluster nodes
- **Power Efficiency**: Modern ARM chips offer excellent performance-per-watt
- **Scale**: Easily add devices to expand cluster capacity
- **Edge Computing**: Deploy AI inference with minimal infrastructure

**Limitations:**
- **RAM Constraints**: Most phones have 4-12GB RAM (limiting model size)
- **Thermal Throttling**: Extended compute loads cause performance drops
- **Storage Speed**: eMMC/UFS storage slower than NVMe SSDs
- **GPU Access**: Limited GPU compute access (no CUDA equivalent)
- **Battery Dependency**: Requires power management for persistent operation

### The Distribution Solution

Large language models require substantial memory—a 70B parameter model needs approximately 35-140 GB of RAM depending on quantization. By distributing across multiple devices, we can:
- Hold portions of model weights on each device
- Perform computations on assigned layers/tensors
- Exchange intermediate activations between devices
- Run models impossible on single mobile hardware

---

## 2. Installation & Initial Setup

### Step 1: Download Termux

> ⚠️ **Important**: Do NOT install from Google Play Store—those versions are outdated and unmaintained.

**Recommended Sources:**
1. **F-Droid** (Recommended): https://f-droid.org/packages/com.termux/
2. **GitHub Releases**: https://github.com/termux/termux-app/releases

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Android Version | 7.0+ | 10.0+ |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB free | 50+ GB free |
| CPU | ARM64 | Snapdragon 8 Gen 1+ |

### Step 2: Initial Configuration

```bash
# Update package repository and upgrade all packages
pkg update && pkg upgrade -y

# Grant storage access (important for file operations)
termux-setup-storage
```

### Step 3: Verify Installation

```bash
# Check system information
uname -a

# View available storage
df -h

# Check package manager
pkg --version
```

### Step 4: Configure Terminal Properties

```bash
# Create/edit termux.properties
mkdir -p ~/.termux
nano ~/.termux/termux.properties
```

Add recommended settings:

```properties
# Enable extra keys row
extra-keys = [['ESC','/','-','HOME','UP','END','PGUP'],['TAB','CTRL','ALT','LEFT','DOWN','RIGHT','PGDN']]

# Allow external apps to execute commands
allow-external-apps = true

# Bell settings
bell-character = vibrate
```

Reload settings:

```bash
termux-reload-settings
```

---

## 3. Essential Packages

### Core Development Packages

```bash
# Programming languages
pkg install python python-pip nodejs-lts ruby golang rust

# Version control
pkg install git git-lfs

# Network tools
pkg install openssh curl wget netcat-openbsd nmap

# Text editors
pkg install nano vim neovim

# Build tools
pkg install clang cmake make

# Utilities
pkg install htop tree jq tmux screen zip unzip tar gzip
```

### AI/ML Specific Packages

```bash
# Python scientific stack
pip install numpy pandas scipy scikit-learn

# Deep learning (CPU-based)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

# Network communication
pip install requests aiohttp websockets pyzmq
```

### Networking & Cluster Packages

```bash
# SSH and remote access
pkg install openssh mosh

# File transfer
pkg install rsync rclone

# Message passing (for distributed computing)
pkg install openmpi

# Distributed Python
pip install dask distributed paramiko fabric
```

---

## 4. Termux Add-ons & APIs

### Termux:API

Provides access to Android device features. **Install both the app and package:**

1. Install **Termux:API** app from F-Droid
2. Install the package:

```bash
pkg install termux-api
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `termux-battery-status` | Get battery level and charging status |
| `termux-brightness` | Set screen brightness |
| `termux-camera-photo` | Take photos |
| `termux-clipboard-get/set` | Access clipboard |
| `termux-notification` | Send Android notifications |
| `termux-sms-send` | Send SMS messages |
| `termux-tts-speak` | Text-to-speech |
| `termux-vibrate` | Vibrate device |
| `termux-wake-lock` | Prevent device sleep |
| `termux-wifi-connectioninfo` | Get WiFi connection details |
| `termux-wifi-scaninfo` | Scan for WiFi networks |

**Example: Battery Check Before Compute:**

```bash
#!/bin/bash
BATTERY=$(termux-battery-status | jq -r '.percentage')
if [ "$BATTERY" -lt 20 ]; then
    echo "Low battery ($BATTERY%). Aborting compute task."
    exit 1
fi
echo "Battery at $BATTERY%. Proceeding..."
```

### Termux:Boot

Auto-start scripts when device boots:

1. Install **Termux:Boot** from F-Droid
2. Open Termux:Boot once to initialize
3. Create startup scripts in `~/.termux/boot/`

### Termux:Tasker

Integrate with Tasker automation:

```bash
mkdir -p ~/.termux/tasker
# Scripts here can be triggered by Tasker
```

### Termux:Styling

Customize terminal appearance:

```bash
pkg install termux-styling
```

---

## 5. Development Environment

### Shell Customization

**Option 1: Oh My Zsh**

```bash
pkg install zsh
chsh -s zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Option 2: Starship Prompt**

```bash
pkg install starship
echo 'eval "$(starship init bash)"' >> ~/.bashrc
```

### Code Editor Setup (Neovim + NvChad)

```bash
pkg install neovim ripgrep fd
git clone https://github.com/NvChad/NvChad ~/.config/nvim --depth 1
nvim
```

### Python Environment Management

```bash
pip install virtualenv

mkdir -p ~/projects/ai-cluster
cd ~/projects/ai-cluster
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Storage & File System

### Termux Directory Structure

```
/data/data/com.termux/files/
├── home/              # ~ (home directory)
├── usr/               # System binaries, libraries
│   ├── bin/           # Executables
│   ├── lib/           # Libraries
│   └── share/         # Shared data
└── ...
```

### Accessing Shared Storage

After running `termux-setup-storage`:

```
~/storage/
├── dcim/          # Camera photos
├── downloads/     # Downloads folder
├── movies/        # Movies
├── music/         # Music
├── pictures/      # Pictures
├── shared/        # Internal shared storage root
└── external-1/    # SD card (if present)
```

**Example: Copy models to Termux:**

```bash
cp ~/storage/downloads/model.gguf ~/models/
export MODEL_PATH=~/storage/shared/models/llama-2-7b.gguf
```

---

# Part II: Networking & Communication

## 7. SSH Server Setup

### Starting SSH Server

```bash
pkg install openssh

# Generate host keys (first time)
ssh-keygen -A

# Set a password for your user
passwd

# Start SSH daemon
sshd

# Check if running
pgrep sshd
```

### SSH Configuration

```bash
# Find your username
whoami  # Usually 'u0_a###'

# Find your IP address
ip addr show wlan0 | grep inet

# Default SSH port in Termux is 8022
# Connect from another device:
ssh -p 8022 user@device_ip
```

### Key-Based Authentication

```bash
# On client machine
ssh-keygen -t ed25519

# Copy to Termux device
ssh-copy-id -p 8022 user@device_ip

# Or manually
echo "your-public-key" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Exchange Keys Between Cluster Nodes

```bash
# On each device
ssh-keygen -t ed25519

# Copy keys between all devices
ssh-copy-id -p 8022 u0_a123@192.168.1.101
ssh-copy-id -p 8022 u0_a456@192.168.1.102
ssh-copy-id -p 8022 u0_a789@192.168.1.103
```

---

## 8. Network Discovery & Configuration

### Find Devices on Network

```bash
pkg install nmap

# Scan local network
nmap -sn 192.168.1.0/24

# Find devices with SSH open
nmap -p 8022 192.168.1.0/24
```

### mDNS/Bonjour (Automatic Discovery)

```bash
pkg install avahi

# Publish this device
avahi-publish-service "termux-node-$(hostname)" _ssh._tcp 8022 &

# Discover other services
avahi-browse -a
```

### Network Options Comparison

| Method | Bandwidth | Latency | Setup Complexity |
|--------|-----------|---------|------------------|
| Wi-Fi (Same Network) | 50-150 Mbps | 5-20 ms | Low |
| Wi-Fi Direct | 100-250 Mbps | 3-10 ms | Medium |
| USB Tethering | 300-480 Mbps | 1-3 ms | Medium |
| Bluetooth | 2-3 Mbps | 50-100 ms | Low |

**Recommended**: Wi-Fi with static IPs on same network for simplicity.

### Port Forwarding & Tunneling

```bash
# Forward local port to remote
ssh -L 8080:localhost:80 user@remote_server

# Reverse tunnel (expose Termux to remote server)
ssh -R 9000:localhost:8022 user@remote_server

# Persistent tunnel with autossh
pkg install autossh
autossh -M 0 -f -N -R 9000:localhost:8022 user@remote_server
```

---

## 9. Inter-Device Communication

### Socket-Based Communication (Python)

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
        output = process_layers(activations)
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

### Network Reliability Wrapper

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

---

# Part III: AI & LLM Inference

## 10. Running LLMs on Single Device

### Using llama.cpp

The most mature solution for running LLMs on Android:

```bash
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
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (background)
ollama serve &

# Pull and run a model
ollama run tinyllama
```

### Using Transformers (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"  # ~5GB RAM needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)

prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Model Size Guidelines

| Model Size | Quantization | RAM Required | Suitable Devices |
|------------|--------------|--------------|------------------|
| 1-3B | Q4_K_M | 2-4 GB | Most phones |
| 7B | Q4_K_M | 6-8 GB | Flagship phones |
| 13B | Q4_K_M | 10-14 GB | Tablets, high-end |
| 70B | Q4_K_M | 40+ GB | **Requires distribution** |

### GPU Acceleration Options

Some Android devices support GPU acceleration via:
- **Vulkan** (via Kompute or custom backends)
- **OpenCL** (limited support)
- **NNAPI** (Android's Neural Networks API)

However, CPU inference with NEON optimizations is often more reliable on Termux.

---

## 11. Theoretical Framework for Distribution

### Why Distributed Inference Works

LLM inference is fundamentally a sequence of matrix multiplications and non-linear operations flowing through layers. This computational structure allows for natural parallelization:

```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
          ↓          ↓              ↓
       Device A   Device B      Device C
```

Each device processes its assigned portion and passes the result to the next device.

### Key Parallelization Concepts

| Concept | Description |
|---------|-------------|
| **Model Parallelism** | Splitting the model's parameters across devices |
| **Pipeline Parallelism** | Assigning consecutive layers to different devices |
| **Tensor Parallelism** | Splitting individual tensors (weight matrices) across devices |
| **Data Parallelism** | Running the same model on multiple devices with different inputs |

For distributed inference on heterogeneous devices like Android phones, **pipeline parallelism** is most practical due to simpler communication patterns.

---

## 12. Model Partitioning Strategies

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

### Optimized Assignment with Linear Programming

```python
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

### Activation Transfer Requirements

Between devices, only **activation tensors** need to be transferred:

```
For a 70B model with 8192 hidden dimension:
- Activation size per token: 8192 × 2 bytes (FP16) = 16 KB
- For batch size 1, sequence length 512: ~8 MB per forward pass
- With 4 devices: 3 transfers × 8 MB = 24 MB total
```

---

## 13. Distributed Inference Architecture

### System Architecture Overview

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

### Simple Pipeline Implementation

```python
# node.py - Run on each Android device
import socket
import pickle
import numpy as np
from model import load_layers, forward_layers

class InferenceNode:
    def __init__(self, layer_start, layer_end, next_node_ip=None):
        self.layers = load_layers(layer_start, layer_end)
        self.next_node_ip = next_node_ip
        
    def process(self, activations):
        # Process through local layers
        output = forward_layers(self.layers, activations)
        
        if self.next_node_ip:
            # Send to next node
            self.send_to_next(output)
        else:
            # Final node - return result
            return output
    
    def send_to_next(self, activations):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.next_node_ip, 5000))
        sock.sendall(pickle.dumps(activations))
        sock.close()
    
    def listen(self, port=5000):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('0.0.0.0', port))
        server.listen(1)
        
        while True:
            conn, addr = server.accept()
            data = b''
            while chunk := conn.recv(4096):
                data += chunk
            activations = pickle.loads(data)
            self.process(activations)
            conn.close()
```

---

# Part IV: Cluster Operations

## 14. Load Balancing & Orchestration

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

### SSH-Based Cluster Management

```bash
# Create hosts file with cluster nodes
cat > ~/cluster_hosts << EOF
node1 192.168.1.101 8022
node2 192.168.1.102 8022
node3 192.168.1.103 8022
EOF

# Function to run command on all nodes
run_on_all() {
    while read name ip port; do
        echo "=== Running on $name ($ip) ==="
        ssh -p $port $ip "$@"
    done < ~/cluster_hosts
}

# Example usage
run_on_all "uname -a && free -h"
```

### Using Fabric for Cluster Operations

```python
from fabric import Connection, ThreadingGroup

nodes = [
    Connection('192.168.1.101:8022'),
    Connection('192.168.1.102:8022'),
    Connection('192.168.1.103:8022'),
]

group = ThreadingGroup.from_connections(nodes)
results = group.run('python --version')

for conn, result in results.items():
    print(f"{conn.host}: {result.stdout.strip()}")
```

### Using Dask Distributed

```python
# On scheduler node
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    n_workers=2,
    threads_per_worker=1,
    memory_limit='1GB'
)
client = Client(cluster)
print(f"Dashboard: {client.dashboard_link}")

# On worker nodes
from dask.distributed import Client
client = Client('tcp://scheduler_ip:8786')
```

### MPI for Message Passing

```bash
pkg install openmpi

cat > ~/mpi_hosts << EOF
192.168.1.101 slots=2
192.168.1.102 slots=2
192.168.1.103 slots=2
EOF

mpirun -np 6 --hostfile ~/mpi_hosts python mpi_script.py
```

---

## 15. Connecting to Other Devices (EXO Integration)

### Hybrid Cluster Architecture

Android devices can join larger clusters with desktops, laptops, or servers:

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

## 16. Background Execution & Persistence

### Wake Lock (Prevent Sleep)

```bash
# Acquire wake lock
termux-wake-lock

# Release when done
termux-wake-unlock
```

### Running Processes in Background

```bash
# Using nohup
nohup python cluster_node.py > node.log 2>&1 &

# Using screen
screen -S cluster
python cluster_node.py
# Detach: Ctrl+A, D
# Reattach: screen -r cluster

# Using tmux
tmux new-session -d -s cluster 'python cluster_node.py'
# Attach: tmux attach -t cluster
```

### Process Manager Script

```bash
#!/bin/bash
# cluster_manager.sh

case "$1" in
    start)
        termux-wake-lock
        sshd
        nohup python ~/cluster/node.py > ~/logs/node.log 2>&1 &
        echo $! > ~/cluster/node.pid
        echo "Cluster node started"
        ;;
    stop)
        if [ -f ~/cluster/node.pid ]; then
            kill $(cat ~/cluster/node.pid)
            rm ~/cluster/node.pid
        fi
        pkill sshd
        termux-wake-unlock
        echo "Cluster node stopped"
        ;;
    status)
        if [ -f ~/cluster/node.pid ] && kill -0 $(cat ~/cluster/node.pid) 2>/dev/null; then
            echo "Node is running (PID: $(cat ~/cluster/node.pid))"
        else
            echo "Node is not running"
        fi
        ;;
esac
```

### Disable Battery Optimization

For persistent operation:

1. Go to **Settings** → **Apps** → **Termux**
2. Select **Battery**
3. Choose **Unrestricted** or **Don't optimize**

---

## 17. Auto-Start on Boot

### Using Termux:Boot

1. Install **Termux:Boot** from F-Droid
2. Open Termux:Boot once to initialize
3. Create boot scripts:

```bash
mkdir -p ~/.termux/boot

cat > ~/.termux/boot/01-cluster-init.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash

# Log boot time
echo "$(date): Boot script started" >> ~/boot.log

# Acquire wake lock
termux-wake-lock

# Wait for network
sleep 10

# Start SSH
sshd

# Start cluster node
cd ~/cluster
source venv/bin/activate
nohup python node.py >> ~/logs/node.log 2>&1 &

# Send notification
termux-notification \
    --title "Cluster Node" \
    --content "Node started successfully" \
    --id cluster-boot
    
echo "$(date): Boot script completed" >> ~/boot.log
EOF

chmod +x ~/.termux/boot/01-cluster-init.sh
```

### Boot Script Best Practices

1. **Number scripts for order**: `01-init.sh`, `02-services.sh`
2. **Add delays**: Network may not be immediately available
3. **Log everything**: Debug boot issues easily
4. **Use notifications**: Confirm successful startup
5. **Handle failures gracefully**: Don't let one failure stop everything

---

# Part V: Advanced Topics

## 18. ADB Automation & Scripting

### ADB Basics for Termux Automation

ADB can automate Termux setup from a computer:

```bash
# Connect to Android device
adb devices

# Open Termux
adb shell am start -n com.termux/.HomeActivity

# Wait for Termux to start
sleep 3
```

### Input Commands via ADB

```bash
# Type text into Termux
adb shell input text "pkg update"

# Press Enter key
adb shell input keyevent 66

# Space must be encoded as %s
adb shell input text "pkg%supdate" && adb shell input keyevent 66
```

### Complete Automation Script

```bash
#!/bin/bash
# setup_termux_node.sh - Automated cluster node setup via ADB

wait_for_termux() {
    sleep 2
}

run_cmd() {
    local cmd="$1"
    local escaped_cmd="${cmd// /%s}"
    adb shell input text "$escaped_cmd"
    adb shell input keyevent 66
    wait_for_termux
}

# Start Termux
adb shell am start -n com.termux/.HomeActivity
sleep 5

# Update packages
run_cmd "pkg update -y"
sleep 10

run_cmd "pkg upgrade -y"
sleep 30

# Install essentials
run_cmd "pkg install -y openssh python git"
sleep 60

# Setup SSH
run_cmd "ssh-keygen -A"
run_cmd "sshd"

# Get IP
run_cmd "ip addr show wlan0"

echo "Termux node setup complete!"
```

### Key Event Codes Reference

| Key | Code | Key | Code |
|-----|------|-----|------|
| Enter | 66 | Tab | 61 |
| Backspace | 67 | Escape | 111 |
| Home | 3 | Back | 4 |
| Volume Up | 24 | Volume Down | 25 |
| Ctrl | 113 | Shift | 59 |

---

## 19. Virtualization with QEMU

### Installing QEMU

```bash
pkg install qemu-utils qemu-common qemu-system-x86_64-headless wget
```

### Creating a Virtual Machine

```bash
mkdir -p ~/vms && cd ~/vms

# Download Alpine Linux
wget http://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/alpine-virt-3.19.1-x86_64.iso

# Create disk image
qemu-img create -f qcow2 alpine.qcow2 20G

# Boot and install
qemu-system-x86_64 \
    -m 2048 \
    -smp 2 \
    -hda alpine.qcow2 \
    -cdrom alpine-virt-3.19.1-x86_64.iso \
    -boot d \
    -net nic \
    -net user,hostfwd=tcp::2222-:22 \
    -nographic
```

### Running the VM

```bash
qemu-system-x86_64 \
    -m 2048 \
    -smp 2 \
    -hda alpine.qcow2 \
    -net nic \
    -net user,hostfwd=tcp::2222-:22 \
    -nographic

# Connect via SSH
ssh -p 2222 localhost
```

### Use Cases for VMs
- Running Docker containers
- Full Linux distribution access
- Isolating experimental workloads
- Running x86-only software

---

## 20. Full Linux Distributions (proot-distro)

### Installing proot-distro

```bash
pkg install proot-distro
```

### Available Distributions

```bash
proot-distro list

# Options: alpine, debian, ubuntu, fedora, archlinux
```

### Installing and Using Ubuntu

```bash
# Install
proot-distro install ubuntu

# Login
proot-distro login ubuntu

# You're now in Ubuntu!
apt update && apt upgrade -y
```

### Running Commands Directly

```bash
proot-distro login ubuntu -- apt install python3 -y
proot-distro login ubuntu -- bash -c "cd /project && python train.py"
```

### Shared Storage

```bash
proot-distro login ubuntu --bind ~/shared:/shared
```

---

## 21. Performance Optimization

### Memory Management

```bash
free -h
watch -n 1 free -h
```

### Thermal Management

```bash
#!/bin/bash
while true; do
    TEMP=$(termux-battery-status | jq -r '.temperature')
    if (( $(echo "$TEMP > 45" | bc -l) )); then
        echo "Temperature high ($TEMP°C), pausing..."
        pkill -STOP python
        sleep 60
        pkill -CONT python
    fi
    sleep 30
done
```

### Performance Expectations

| Metric | Single Phone | 4-Phone Cluster |
|--------|--------------|-----------------|
| Max Model Size | 7B Q4 | 30B Q4 |
| Tokens/Second (7B) | 3-8 tok/s | 8-15 tok/s |
| Tokens/Second (30B) | N/A | 2-5 tok/s |
| Network Overhead | 0% | 15-30% |

### Battery and Thermal Recommendations

- Keep devices plugged in during inference
- Use cooling pads or fans
- Implement thermal throttling in software
- Rotate workloads between devices

---

## 22. Security Best Practices

### SSH Security

```bash
cat >> $PREFIX/etc/ssh/sshd_config << EOF
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
EOF

pkill sshd && sshd
```

### API Key Management

```python
import os
from cryptography.fernet import Fernet

key = Fernet.generate_key()

with open(os.path.expanduser('~/.secret_key'), 'wb') as f:
    f.write(key)
os.chmod(os.path.expanduser('~/.secret_key'), 0o600)

cipher = Fernet(key)
encrypted_api_key = cipher.encrypt(b"your-api-key-here")
decrypted_key = cipher.decrypt(encrypted_api_key)
```

### Security Checklist

1. **Encrypt all inter-device communication** (TLS/SSH)
2. **Use VPN for untrusted networks**
3. **Implement authentication between nodes**
4. **Validate all incoming data**
5. **Regular updates**: `pkg update && pkg upgrade -y`

---

# Part VI: Reference

## 23. Practical Architectures

### Architecture 1: Inference Cluster

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │   (Termux Node) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Inference    │   │  Inference    │   │  Inference    │
│  Node 1       │   │  Node 2       │   │  Node 3       │
│  (Termux)     │   │  (Termux)     │   │  (Termux)     │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Architecture 2: Edge Computing Hub

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud/Main Server                        │
│                    (Model Training)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Termux Gateway  │
                    │   (SSH Tunnel)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Edge Sensor 1 │   │ Edge Sensor 2 │   │ Edge Sensor 3 │
│ (Termux+API)  │   │ (Termux+API)  │   │ (Termux+API)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Architecture 3: Federated Learning

```python
# Coordinator (main server)
from flask import Flask, request

app = Flask(__name__)
aggregated_weights = None

@app.route('/submit_weights', methods=['POST'])
def receive_weights():
    global aggregated_weights
    node_weights = request.json
    # Aggregate weights
    return {'status': 'received'}

@app.route('/get_weights', methods=['GET'])
def send_weights():
    return json.dumps(aggregated_weights)

# Node (each Termux device)
import requests

def train_local_model():
    weights = model.get_weights()
    return weights

def submit_to_coordinator(weights):
    requests.post('http://coordinator:5000/submit_weights', json=weights)

def get_global_weights():
    response = requests.get('http://coordinator:5000/get_weights')
    return response.json()
```

---

## 24. Troubleshooting

### Permission Denied

```bash
chmod +x script.sh
termux-setup-storage
```

### SSH Connection Refused

```bash
pgrep sshd || sshd
netstat -tlnp | grep 8022
ip addr show wlan0
```

### Package Installation Fails

```bash
pkg clean
rm -rf $PREFIX/var/cache/apt/
pkg update
termux-change-repo
```

### Out of Memory

```bash
free -h
top -o %MEM
# Close unused Android apps
```

### Network Issues

```bash
ping -c 3 google.com
nslookup google.com
# Toggle airplane mode if needed
```

---

## 25. Existing Projects & Research

### Academic Research

**LinguaLinked**: Distributed LLM inference for mobile devices
- Optimized model assignment using linear programming
- Runtime load balancing
- Paper: [arXiv:2312.00388](https://arxiv.org/abs/2312.00388)

### Open Source Projects

| Project | Description | Link |
|---------|-------------|------|
| **EXO** | Run AI clusters with everyday devices | [GitHub](https://github.com/exo-explore/exo) |
| **Petals** | BitTorrent-style distributed inference | [GitHub](https://github.com/bigscience-workshop/petals) |
| **MLC LLM** | ML Compilation for LLMs with Android support | [GitHub](https://github.com/mlc-ai/mlc-llm) |
| **llama.cpp** | Efficient LLM inference on consumer hardware | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **Termux-Ai** | Multiple AI models in Termux | [GitHub](https://github.com/Anon4You/Termux-Ai) |

### Future Directions

- **RDMA over Wi-Fi 7**: Ultra-low latency wireless
- **NPU Integration**: Dedicated AI accelerators in mobile SoCs
- **Mesh Networking**: Ad-hoc clusters without infrastructure
- **Speculative Decoding**: Parallel token generation
- **Dynamic Compression**: Adjusting quantization based on load

---

## 26. Quick Reference

### Package Management

```bash
pkg update && pkg upgrade -y
pkg install <package>
pkg uninstall <package>
pkg search <query>
pkg list-installed
```

### Termux-Specific

```bash
termux-setup-storage
termux-wake-lock
termux-wake-unlock
termux-reload-settings
termux-battery-status
```

### SSH

```bash
sshd                    # Start server
pkill sshd              # Stop server
ssh -p 8022 user@ip     # Connect
```

### Process Management

```bash
htop
screen -S name
tmux new -s name
nohup cmd &
```

### Network

```bash
ip addr show
ping -c 3 host
nmap -sn 192.168.1.0/24
```

### proot-distro

```bash
proot-distro list
proot-distro install <distro>
proot-distro login <distro>
```

---

## Conclusion

Distributing LLM inference across Android devices using Termux is **technically feasible** and opens possibilities for:

- Running larger models than any single phone could handle
- Creating portable, infrastructure-free AI clusters
- Democratizing access to AI capabilities
- Integrating with larger clusters via EXO

**Key Takeaways:**

1. **Start with llama.cpp** for mature, tested mobile inference
2. **Use Wi-Fi on same network** for simplest communication
3. **Layer-wise partitioning** is most practical for mobile
4. **Expect 3-8 tokens/second** on a multi-phone cluster for 7B models
5. **Consider battery and thermal** constraints carefully
6. **Integrate with EXO** for connection to larger clusters
7. **Use ADB automation** for fleet provisioning
8. **Implement proper security** with SSH keys and encryption

The combination of improving mobile hardware, better quantization techniques, and frameworks like EXO makes distributed mobile LLM inference increasingly practical.

---

## References

1. [LinguaLinked: A Distributed LLM Inference System for Mobile Devices](https://arxiv.org/abs/2312.00388)
2. [Distributed Mixture-of-Agents for Edge Inference](https://arxiv.org/abs/2412.21200)
3. [MLC LLM: Bringing Hardware-Accelerated LLMs to Android](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices)
4. [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
5. [Petals: Collaborative Inference](https://github.com/bigscience-workshop/petals)
6. [EXO: Run Your Own AI Cluster](https://github.com/exo-explore/exo)
7. [Termux Wiki](https://wiki.termux.com/)
8. [Termux:API Documentation](https://wiki.termux.com/wiki/Termux:API)
9. [F-Droid Termux](https://f-droid.org/packages/com.termux/)

---

*This guide is maintained as part of the EXO project documentation. For updates and contributions, see the [main repository](https://github.com/exo-explore/exo).*

