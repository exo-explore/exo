# Systemd Services for RKLLM

Systemd service files for auto-starting RKLLAMA and Exo on RK3588 devices.

## Services

| Service | Description | Port |
|---------|-------------|------|
| `rkllama.service` | RKLLAMA server (RKLLM runtime) | 8080 |
| `exo-rkllm.service` | Exo with RKLLM inference engine | 52415 |

## Installation

### Quick Install

```bash
# On the RK3588 device
cd /opt/exo/systemd
sudo ./install.sh
```

### Manual Install

```bash
# Copy service files
sudo cp rkllama.service /etc/systemd/system/
sudo cp exo-rkllm.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable rkllama.service
sudo systemctl enable exo-rkllm.service
```

## Usage

### Start Services

```bash
# Start both services (rkllama first, then exo)
sudo systemctl start rkllama
sleep 5  # Wait for rkllama to initialize
sudo systemctl start exo-rkllm
```

### Stop Services

```bash
# Stop in reverse order
sudo systemctl stop exo-rkllm
sudo systemctl stop rkllama
```

### Check Status

```bash
# View status
sudo systemctl status rkllama exo-rkllm

# View logs
sudo journalctl -u rkllama -u exo-rkllm -f

# View only rkllama logs
sudo journalctl -u rkllama -f

# View only exo logs
sudo journalctl -u exo-rkllm -f
```

### Restart Services

```bash
sudo systemctl restart rkllama
sleep 5
sudo systemctl restart exo-rkllm
```

## Service Dependencies

```
                    ┌─────────────────┐
                    │  network.target │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ rkllama.service │
                    │    (port 8080)  │
                    └────────┬────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ exo-rkllm.service│
                   │   (port 52415)   │
                   └──────────────────┘
```

`exo-rkllm.service` waits for `rkllama.service` to be ready before starting.

## Configuration

### Environment Variables

Edit the service files to modify environment variables:

```bash
sudo systemctl edit rkllama.service
```

Add overrides:

```ini
[Service]
Environment="SOME_VAR=value"
```

### Change Default Model

To auto-load a specific model on startup, modify `rkllama.service`:

```ini
ExecStartPost=/bin/bash -c 'sleep 5 && curl -s -X POST http://localhost:8080/load_model -H "Content-Type: application/json" -d "{\"model_name\": \"Qwen2.5-1.5B-Instruct\"}"'
```

## Troubleshooting

### Service won't start

```bash
# Check logs for errors
sudo journalctl -u rkllama -n 50

# Check if port is in use
sudo netstat -tlnp | grep 8080
```

### NPU not detected

```bash
# Run frequency fix manually
sudo /opt/rkllama/lib/fix_freq_rk3588.sh

# Check NPU device
ls -la /dev/dri/
```

### Exo can't connect to rkllama

```bash
# Verify rkllama is running
curl http://localhost:8080/

# Check if rkllama is healthy
curl http://localhost:8080/models
```

## Uninstall

```bash
# Stop and disable services
sudo systemctl stop exo-rkllm rkllama
sudo systemctl disable exo-rkllm rkllama

# Remove service files
sudo rm /etc/systemd/system/rkllama.service
sudo rm /etc/systemd/system/exo-rkllm.service

# Reload systemd
sudo systemctl daemon-reload
```
