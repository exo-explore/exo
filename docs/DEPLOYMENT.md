# Exo-RKLLM Deployment Guide

Complete guide for deploying exo-rkllama on RK3588-based devices.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Prerequisites](#software-prerequisites)
3. [RKLLM Runtime Setup](#rkllm-runtime-setup)
4. [RKLLAMA Server Installation](#rkllama-server-installation)
5. [Exo-RKLLAMA Installation](#exo-rkllama-installation)
6. [Model Setup](#model-setup)
7. [Systemd Services](#systemd-services)
8. [Network Configuration](#network-configuration)
9. [Monitoring Setup](#monitoring-setup)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Supported Devices

| Device | SoC | RAM | NPU | Status |
|--------|-----|-----|-----|--------|
| Orange Pi 5 Plus | RK3588 | 8-32GB | 6 TOPS | Tested |
| Rock 5B | RK3588 | 8-16GB | 6 TOPS | Tested |
| Turing RK1 | RK3588 | 8-32GB | 6 TOPS | Tested |
| Orange Pi 5 | RK3588S | 4-16GB | 6 TOPS | Compatible |
| Rock 5A | RK3588S | 4-16GB | 6 TOPS | Compatible |
| Radxa CM5 | RK3588S | 4-16GB | 6 TOPS | Compatible |

### Minimum Requirements

- **RAM**: 8GB (16GB+ recommended for larger models)
- **Storage**: 32GB+ (models are 1-4GB each)
- **OS**: Ubuntu 22.04/24.04 or Armbian (aarch64)

### Recommended Setup

- 16GB RAM for Qwen2.5-1.5B/3B models
- NVMe SSD for faster model loading
- Active cooling (heatsink + fan)
- Ethernet connection for stable API access

---

## Software Prerequisites

### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.12+

```bash
# Check current version
python3 --version

# If < 3.12, install from deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Or build from source (Armbian)
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
tar xzf Python-3.12.0.tgz
cd Python-3.12.0
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```

### 3. Install System Dependencies

```bash
sudo apt install -y \
  git \
  curl \
  wget \
  build-essential \
  cmake \
  libffi-dev \
  libssl-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncurses5-dev \
  libgdbm-dev \
  liblzma-dev \
  tk-dev
```

---

## RKLLM Runtime Setup

### 1. Download RKLLM SDK

```bash
# Create installation directory
sudo mkdir -p /opt/rkllm
cd /opt/rkllm

# Download RKLLM runtime (v1.1.4 recommended for stability)
wget https://github.com/airockchip/rknn-llm/releases/download/release-v1.1.4/rkllm-runtime-RK3588-1.1.4.tar.gz

# Extract
tar xzf rkllm-runtime-RK3588-1.1.4.tar.gz
```

### 2. Install Runtime Library

```bash
# Copy library to system path
sudo cp lib/librkllmrt.so /usr/local/lib/
sudo ldconfig

# Verify installation
ldconfig -p | grep rkllm
```

### 3. Set NPU Frequency (Optional but Recommended)

```bash
# Check current frequency
cat /sys/class/devfreq/fdab0000.npu/cur_freq

# Set to maximum performance
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor

# Or set specific frequency (in Hz)
echo 1000000000 | sudo tee /sys/class/devfreq/fdab0000.npu/userspace/set_freq
```

To persist across reboots, add to `/etc/rc.local`:

```bash
#!/bin/bash
echo performance > /sys/class/devfreq/fdab0000.npu/governor
exit 0
```

---

## RKLLAMA Server Installation

### 1. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/jfreed-dev/rkllama.git
sudo chown -R $USER:$USER /opt/rkllama
cd /opt/rkllama
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test server starts
python server.py --target_platform rk3588 --port 8080 &
sleep 5
curl http://localhost:8080/
kill %1
```

---

## Exo-RKLLAMA Installation

### 1. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/jfreed-dev/exo-rkllama.git exo
sudo chown -R $USER:$USER /opt/exo
cd /opt/exo
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3. Verify Installation

```bash
# Check exo is installed
exo --help

# Test import
python -c "from exo.inference.rkllm import RKLLMInferenceEngine; print('OK')"
```

---

## Model Setup

### 1. Create Model Directory

```bash
mkdir -p ~/RKLLAMA/models
```

### 2. Download Pre-converted Models

```bash
cd ~/RKLLAMA/models

# Qwen2.5-1.5B-Instruct (recommended)
wget https://huggingface.co/c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4/resolve/main/Qwen2.5-1.5B-Instruct-w8a8-opt1-hybrid-ratio-1.0.rkllm
mkdir -p Qwen2.5-1.5B-Instruct
mv Qwen2.5-1.5B-Instruct-w8a8-opt1-hybrid-ratio-1.0.rkllm Qwen2.5-1.5B-Instruct/
```

### 3. Create Modelfile

```bash
cat > ~/RKLLAMA/models/Qwen2.5-1.5B-Instruct/Modelfile << 'EOF'
FROM Qwen2.5-1.5B-Instruct-w8a8-opt1-hybrid-ratio-1.0.rkllm
HUGGINGFACE Qwen/Qwen2.5-1.5B-Instruct
MAX_NEW_TOKENS 2048
EOF
```

### 4. Verify Model

```bash
ls -la ~/RKLLAMA/models/Qwen2.5-1.5B-Instruct/
# Should show .rkllm file and Modelfile
```

### Converting Custom Models

To convert your own models, see [rkllm-toolkit](https://github.com/jfreed-dev/rkllm-toolkit).

---

## Systemd Services

### 1. Install Service Files

```bash
cd /opt/exo
sudo cp systemd/rkllama.service /etc/systemd/system/
sudo cp systemd/exo-rkllm.service /etc/systemd/system/
```

### 2. Adjust Paths (if needed)

Edit service files if your installation paths differ:

```bash
sudo nano /etc/systemd/system/rkllama.service
sudo nano /etc/systemd/system/exo-rkllm.service
```

### 3. Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable rkllama.service
sudo systemctl enable exo-rkllm.service

# Start services
sudo systemctl start rkllama.service
sudo systemctl start exo-rkllm.service
```

### 4. Verify Services

```bash
# Check status
sudo systemctl status rkllama.service
sudo systemctl status exo-rkllm.service

# View logs
journalctl -u rkllama.service -f
journalctl -u exo-rkllm.service -f
```

---

## Network Configuration

### Firewall Rules

```bash
# Allow exo API (port 52415)
sudo ufw allow 52415/tcp comment "Exo API"

# Allow rkllama (localhost only by default)
# Only open if needed for remote access:
# sudo ufw allow 8080/tcp comment "RKLLAMA"

# Allow Prometheus metrics (optional)
sudo ufw allow 9090/tcp comment "Prometheus"

# Enable firewall
sudo ufw enable
```

### Remote Access

By default, exo binds to `0.0.0.0:52415`. To restrict to specific interfaces:

```bash
exo --inference-engine rkllm --chatgpt-api-port 52415 --disable-tui
```

### Reverse Proxy (Optional)

For HTTPS/authentication, use nginx:

```nginx
# /etc/nginx/sites-available/exo
server {
    listen 443 ssl;
    server_name exo.example.com;

    ssl_certificate /etc/letsencrypt/live/exo.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/exo.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:52415;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

---

## Monitoring Setup

### 1. Install Prometheus

```bash
# Download
wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-arm64.tar.gz
tar xzf prometheus-2.47.0.linux-arm64.tar.gz
sudo mv prometheus-2.47.0.linux-arm64 /opt/prometheus

# Use exo config
sudo cp /opt/exo/prometheus.yml /opt/prometheus/
```

### 2. Create Prometheus Service

```bash
sudo tee /etc/systemd/system/prometheus.service << 'EOF'
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
ExecStart=/opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

### 3. Install Grafana

```bash
# Add Grafana repository
sudo apt install -y apt-transport-https software-properties-common
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Install
sudo apt update
sudo apt install grafana

# Start
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 4. Import Dashboard

1. Open Grafana at `http://<device-ip>:3000` (default: admin/admin)
2. Add Prometheus data source: `http://localhost:9090`
3. Import dashboard from `/opt/exo/grafana/exo-dashboard.json`

---

## Security Considerations

### 1. API Authentication

Exo doesn't include built-in authentication. Options:

- **Reverse proxy with auth**: Use nginx + basic auth or OAuth
- **Network isolation**: Run on private network only
- **Firewall rules**: Restrict access by IP

### 2. Service Accounts

Create dedicated user for running services:

```bash
sudo useradd -r -s /bin/false exo
sudo chown -R exo:exo /opt/exo /opt/rkllama
```

Update service files to use `User=exo`.

### 3. File Permissions

```bash
# Restrict model directory
chmod 750 ~/RKLLAMA/models

# Restrict config files
chmod 600 /opt/exo/prometheus.yml
```

### 4. Updates

Keep systems updated:

```bash
# System packages
sudo apt update && sudo apt upgrade -y

# Python packages
cd /opt/exo && source venv/bin/activate
pip install --upgrade -e .
```

---

## Troubleshooting

### RKLLAMA Server Won't Start

```bash
# Check if NPU is accessible
ls -la /dev/dri/

# Check RKLLM library
ldconfig -p | grep rkllm

# Check logs
journalctl -u rkllama.service --no-pager -n 50
```

### Exo Can't Connect to RKLLAMA

```bash
# Test RKLLAMA directly
curl http://localhost:8080/

# Check environment variables
echo $RKLLM_SERVER_HOST $RKLLM_SERVER_PORT

# Test from exo
DEBUG=2 exo --inference-engine rkllm --disable-tui
```

### Model Loading Fails

```bash
# Check model files exist
ls -la ~/RKLLAMA/models/

# Check Modelfile format
cat ~/RKLLAMA/models/Qwen2.5-1.5B-Instruct/Modelfile

# Check available models via API
curl http://localhost:8080/models
```

### Slow Inference

```bash
# Check NPU frequency
cat /sys/class/devfreq/fdab0000.npu/cur_freq

# Check thermal throttling
cat /sys/class/thermal/thermal_zone*/temp

# Set performance governor
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

### Out of Memory

```bash
# Check memory usage
free -h

# Reduce model size (use smaller quantization)
# Or add swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Service Logs

```bash
# RKLLAMA logs
journalctl -u rkllama.service -f

# Exo logs
journalctl -u exo-rkllm.service -f

# Verbose mode
DEBUG=2 exo --inference-engine rkllm --disable-tui
```

---

## Quick Reference

### Start/Stop Services

```bash
# Start all
sudo systemctl start rkllama exo-rkllm

# Stop all
sudo systemctl stop exo-rkllm rkllama

# Restart
sudo systemctl restart rkllama exo-rkllm
```

### Test API

```bash
# Health check
curl http://localhost:52415/healthcheck

# List models
curl http://localhost:52415/v1/models

# Chat completion
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct-rkllm", "messages": [{"role": "user", "content": "Hello"}]}'

# Metrics
curl http://localhost:52415/metrics
```

### Useful Paths

| Path | Description |
|------|-------------|
| `/opt/exo` | Exo installation |
| `/opt/rkllama` | RKLLAMA server |
| `~/RKLLAMA/models` | Model files |
| `/var/log/journal` | Service logs |
| `/opt/prometheus` | Prometheus installation |

---

## Support

- **Issues**: [github.com/jfreed-dev/exo-rkllama/issues](https://github.com/jfreed-dev/exo-rkllama/issues)
- **RKLLAMA**: [github.com/jfreed-dev/rkllama](https://github.com/jfreed-dev/rkllama)
- **Upstream exo**: [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
