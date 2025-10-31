# Exo Cluster Setup - Mini1 & Mini2

## Network Configuration
- **mini1**: 192.168.2.13 (en0)
- **mini2**: 192.168.5.2 (Thunderbolt)
- **Port**: 50051 for gRPC communication

## Setup Instructions for Mini2

### 1. Copy Files to Mini2
Copy the entire exo directory to mini2:
```bash
rsync -av --exclude='.venv' /Users/mini1/Movies/exo/ user@192.168.5.2:~/Movies/exo/
```

### 2. Install on Mini2
SSH into mini2 and run:
```bash
ssh user@192.168.5.2
cd ~/Movies/exo

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.12

# Activate and install
source .venv/bin/activate
uv pip install -e .
```

### 3. Run the Cluster

#### On Mini1:
```bash
cd /Users/mini1/Movies/exo
./run_mini1.sh
```

#### On Mini2:
```bash
cd ~/Movies/exo
./run_mini2.sh
```

## Testing the Cluster

Once both nodes are running, you can access the ChatGPT API:
- **mini1**: http://192.168.2.13:8000
- **mini2**: http://192.168.5.2:8001

Test with curl:
```bash
curl -X POST http://192.168.2.13:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

## Available Models
Run `exo --help` to see available models. Common ones:
- llama-3.2-1b
- llama-3.2-3b
- llama-3.1-8b
- mistral-7b

## Troubleshooting

### Connection Issues
- Verify Thunderbolt connection: `ping 192.168.5.2`
- Check firewall settings on both machines
- Ensure port 50051 is not blocked

### Discovery Issues
- Use `--discovery-module udp` for automatic discovery within same network
- Check discovery_config.json has correct IPs

### Performance
- Thunderbolt provides low latency (~0.2ms)
- Ideal for model partitioning across nodes