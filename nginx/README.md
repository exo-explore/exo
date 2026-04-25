# Nginx Load Balancer for exo-rkllama

Distributes requests across multiple RK3588 nodes for horizontal scaling.

## Why Load Balancing?

RKLLM loads complete models and can't shard layers across nodes. For multiple RK3588 devices, use **request-level parallelism** instead:

```
                    ┌─────────────────┐
                    │  Nginx LB :80   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ RK3588 #1   │   │ RK3588 #2   │   │ RK3588 #3   │
    │ exo :52415  │   │ exo :52415  │   │ exo :52415  │
    └─────────────┘   └─────────────┘   └─────────────┘
```

## Quick Start

### 1. Install nginx

```bash
sudo apt install nginx
```

### 2. Configure

```bash
# Edit node addresses
sudo nano /etc/nginx/sites-available/exo

# Copy config
sudo cp exo-load-balancer.conf /etc/nginx/sites-available/exo

# Enable site
sudo ln -s /etc/nginx/sites-available/exo /etc/nginx/sites-enabled/

# Remove default site (optional)
sudo rm /etc/nginx/sites-enabled/default
```

### 3. Update Node Addresses

Edit the `upstream exo_cluster` block:

```nginx
upstream exo_cluster {
    least_conn;
    server 192.168.1.101:52415;  # Node 1
    server 192.168.1.102:52415;  # Node 2
    server 192.168.1.103:52415;  # Node 3
}
```

### 4. Test and Reload

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Test Load Balancer

```bash
# Health check
curl http://localhost/healthcheck

# API request
curl http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct-rkllm", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Load Balancing Methods

| Method | Directive | Use Case |
|--------|-----------|----------|
| Round-robin | (default) | Equal distribution |
| Least connections | `least_conn;` | Best for varying request times |
| IP hash | `ip_hash;` | Sticky sessions |
| Weighted | `weight=N` | Prefer faster nodes |

## Configuration Options

### Timeouts

```nginx
proxy_connect_timeout 10s;   # Connection timeout
proxy_send_timeout 300s;     # Send timeout (5 min for long prompts)
proxy_read_timeout 300s;     # Read timeout (5 min for slow inference)
```

### Health Checks

Passive health checks are enabled by default:

```nginx
server node1:52415 max_fails=3 fail_timeout=30s;
```

- `max_fails=3`: Mark unhealthy after 3 failures
- `fail_timeout=30s`: Retry after 30 seconds

### Streaming Support

Already configured for SSE streaming:

```nginx
proxy_buffering off;
proxy_cache off;
chunked_transfer_encoding on;
```

## HTTPS Setup

1. Get certificates (Let's Encrypt):
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d exo.example.com
```

2. Uncomment the HTTPS server block in the config.

## Monitoring

### Nginx Status

```bash
curl http://localhost:8080/nginx_status
```

### View Logs

```bash
# Access log with upstream info
tail -f /var/log/nginx/exo-access.log

# Error log
tail -f /var/log/nginx/exo-error.log
```

### Prometheus Metrics

Configure Prometheus to scrape individual nodes:

```yaml
scrape_configs:
  - job_name: 'exo-cluster'
    static_configs:
      - targets:
        - 'rk3588-node1:52415'
        - 'rk3588-node2:52415'
        - 'rk3588-node3:52415'
```

## Troubleshooting

### 502 Bad Gateway

- Check if exo is running on backend nodes
- Verify node addresses are correct
- Check firewall allows port 52415

```bash
# Test direct connection to node
curl http://rk3588-node1:52415/healthcheck
```

### Slow Responses

- Increase timeouts for long inference
- Use `least_conn` for better distribution
- Check individual node performance

### Connection Refused

```bash
# Check nginx is running
sudo systemctl status nginx

# Check listening ports
sudo ss -tlnp | grep nginx
```
