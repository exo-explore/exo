# llama.cpp RPC Network Requirements

> **Comprehensive guide to network configuration, troubleshooting, and best practices for llama.cpp distributed inference via RPC**

This document covers network requirements for llama.cpp Remote Procedure Call (RPC) server operations, including port configuration, TCP tuning, diagnostics, and platform-specific considerations (including Android/Termux).

---

## Table of Contents

1. [Overview](#overview)
2. [Port Configuration](#port-configuration)
3. [TCP/IP Settings](#tcpip-settings)
4. [Keepalive and Timeout Behavior](#keepalive-and-timeout-behavior)
5. [Protocol-Level Expectations](#protocol-level-expectations)
6. [Network Performance Requirements](#network-performance-requirements)
7. [Client Isolation Pitfalls](#client-isolation-pitfalls)
8. [Diagnostic Tools](#diagnostic-tools)
9. [Platform-Specific Considerations](#platform-specific-considerations)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Configuration Scripts](#configuration-scripts)
12. [Best Practices Checklist](#best-practices-checklist)

---

## Overview

### How llama.cpp RPC Works

llama.cpp uses a distributed inference architecture where:

1. **Master Node (device_rank = 0)**: Runs `llama-server` with `--rpc` flag to connect to worker RPC servers
2. **Worker Nodes (device_rank > 0)**: Run `rpc-server` to expose their compute resources via TCP
3. **Tensor Split**: Layers are distributed across devices using `--tensor-split` ratios

```
┌─────────────────┐     TCP/RPC      ┌─────────────────┐
│  Master Node    │◄───────────────► │  Worker Node 1  │
│  (llama-server) │                  │   (rpc-server)  │
│   rank=0        │     TCP/RPC      │    rank=1       │
│                 │◄───────────────► ┌─────────────────┐
│                 │                  │  Worker Node 2  │
│                 │                  │   (rpc-server)  │
└─────────────────┘                  │    rank=2       │
                                     └─────────────────┘
```

### Build Requirements

To enable RPC support, build llama.cpp with the RPC backend:

```bash
cd ~/llama.cpp
cmake -B build -DGGML_RPC=ON
cmake --build build --target rpc-server llama-server
```

Verify the binaries exist:

```bash
ls -la ~/llama.cpp/build/bin/rpc-server
ls -la ~/llama.cpp/build/bin/llama-server
```

---

## Port Configuration

### Default Ports

| Component | Default Port | Port Range | Notes |
|-----------|--------------|------------|-------|
| `rpc-server` | 50052 | 50052-50060 | Worker nodes (sequential per rank) |
| `llama-server` | 8080 | Any available | Master node HTTP API |
| exo RPC base | 60000 | 60000+ | Used by exo framework |

### Port Assignment Strategy

In the exo framework, ports are assigned based on device rank:

```python
# From src/exo/master/placement_utils.py
RPC_BASE_PORT: int = 60000

def get_rpc_ports_for_llamacpp(selected_cycle):
    rpc_ports = {}
    for device_rank, node in enumerate(selected_cycle):
        if device_rank == 0:
            rpc_ports[node.node_id] = 0  # Master doesn't need RPC port
        else:
            rpc_ports[node.node_id] = RPC_BASE_PORT + device_rank - 1
    return rpc_ports
```

### Starting RPC Server Manually

```bash
# On worker node
./rpc-server --host 0.0.0.0 --port 50052

# On master node with single worker
./llama-server -m model.gguf --rpc "192.168.1.100:50052" --tensor-split 0.5,0.5

# With multiple workers
./llama-server -m model.gguf --rpc "192.168.1.100:50052,192.168.1.101:50053" --tensor-split 0.33,0.33,0.34
```

### Firewall Configuration

#### Linux (iptables)

```bash
# Allow RPC ports
sudo iptables -A INPUT -p tcp --dport 50052:50060 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 60000:60010 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

#### Linux (ufw)

```bash
sudo ufw allow 50052:50060/tcp
sudo ufw allow 60000:60010/tcp
sudo ufw allow 8080/tcp
```

#### Windows PowerShell

```powershell
New-NetFirewallRule -DisplayName "llama.cpp RPC" -Direction Inbound -Protocol TCP -LocalPort 50052-50060 -Action Allow
New-NetFirewallRule -DisplayName "exo RPC" -Direction Inbound -Protocol TCP -LocalPort 60000-60010 -Action Allow
```

#### macOS

```bash
# Add to /etc/pf.conf
# pass in proto tcp from any to any port 50052:50060
sudo pfctl -f /etc/pf.conf
```

---

## TCP/IP Settings

### Linux Kernel Parameters

Create `/etc/sysctl.d/99-llama-rpc.conf`:

```bash
# ===== Connection Handling =====
# Increase TCP backlog for burst connections
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192

# Increase ephemeral port range
net.ipv4.ip_local_port_range = 1024 65535

# ===== Keepalive Settings =====
# Time before first keepalive probe (2 hours -> 10 minutes for RPC)
net.ipv4.tcp_keepalive_time = 600
# Interval between keepalive probes
net.ipv4.tcp_keepalive_intvl = 30
# Number of probes before considering connection dead
net.ipv4.tcp_keepalive_probes = 5

# ===== Timeout Settings =====
# Reduce TIME_WAIT state duration
net.ipv4.tcp_fin_timeout = 15
# Number of SYN retries
net.ipv4.tcp_syn_retries = 3
net.ipv4.tcp_synack_retries = 3

# ===== Performance =====
# Enable TIME_WAIT socket reuse
net.ipv4.tcp_tw_reuse = 1

# Increase receive/send buffer sizes
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Increase netdev backlog for high-speed networks
net.core.netdev_max_backlog = 5000
```

Apply immediately:

```bash
sudo sysctl -p /etc/sysctl.d/99-llama-rpc.conf
```

### Quick Application (Temporary)

```bash
# Essential settings for quick testing
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=4096
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"
sudo sysctl -w net.ipv4.tcp_fin_timeout=30
sudo sysctl -w net.ipv4.tcp_keepalive_time=600
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=30
sudo sysctl -w net.ipv4.tcp_keepalive_probes=5
```

---

## Keepalive and Timeout Behavior

### Why Keepalives Matter

- **NAT Devices**: NATs often close idle connections after 5-15 minutes
- **Firewalls**: Stateful firewalls may drop idle connections
- **Model Loading**: Large models can take minutes to load, causing apparent "idle" periods
- **Long Inference**: Extended generation can appear idle between tensor operations

### Recommended Keepalive Configuration

```bash
# Aggressive keepalive for RPC (check every 30s after 5 min idle)
sudo sysctl -w net.ipv4.tcp_keepalive_time=300
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=30
sudo sysctl -w net.ipv4.tcp_keepalive_probes=3

# For very stable local networks (10 min initial, 60s interval)
sudo sysctl -w net.ipv4.tcp_keepalive_time=600
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=60
sudo sysctl -w net.ipv4.tcp_keepalive_probes=5
```

### Application-Level Timeouts

In the exo codebase, connection timeouts are configured:

```python
# From src/exo/worker/engines/llamacpp/utils.py
def wait_for_rpc_workers(rpc_addresses: str, timeout: int = 120) -> bool:
    """Wait for all RPC worker servers to be available."""
    # Polls each worker's RPC port until ready or timeout
    ...

# From src/exo/worker/engines/llamacpp/rpc_server.py  
RPC_SERVER_STARTUP_TIMEOUT: Final[int] = 30  # Seconds to wait for server start
```

---

## Protocol-Level Expectations

### TCP Backlog

The TCP backlog determines how many pending connections can queue before new connections are refused:

```bash
# Check current settings
sysctl net.core.somaxconn
sysctl net.ipv4.tcp_max_syn_backlog

# Recommended for distributed inference
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=4096
```

### Maximum Transmission Unit (MTU)

Ensure consistent MTU across all devices to prevent fragmentation:

```bash
# Check current MTU
ip link show eth0 | grep mtu

# Standard Ethernet MTU (recommended)
sudo ip link set dev eth0 mtu 1500

# For Jumbo frames on 10GbE (if supported by all devices)
sudo ip link set dev eth0 mtu 9000

# Verify path MTU to remote host
tracepath 192.168.1.100
```

**MTU Discovery Issues**: If you see "packet too large" errors or fragmentation issues:

```bash
# Enable Path MTU Discovery
sudo sysctl -w net.ipv4.ip_no_pmtu_disc=0

# Alternative: Set lower MTU if issues persist
sudo ip link set dev eth0 mtu 1400
```

### Socket Buffer Sizes

For high-throughput tensor transfers:

```bash
# Increase socket buffer sizes
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
```

### TCP_NODELAY (Nagle's Algorithm)

llama.cpp RPC benefits from disabling Nagle's algorithm for lower latency. This is typically set at the application level in socket options. The RPC implementation should set:

```c
int flag = 1;
setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
```

---

## Network Performance Requirements

### Bandwidth Requirements

| Cluster Size | Model Size | Minimum Bandwidth | Recommended |
|--------------|------------|-------------------|-------------|
| 2 nodes | 7B | 100 Mbps | 1 Gbps |
| 2-4 nodes | 13B-30B | 500 Mbps | 1 Gbps |
| 4+ nodes | 70B+ | 1 Gbps | 10 Gbps |
| Large clusters | 200B+ | 10 Gbps | 25+ Gbps |

### Latency Requirements

- **Optimal**: < 1ms (same rack, direct connection)
- **Acceptable**: 1-5ms (same building, switched network)
- **Marginal**: 5-20ms (same campus, routed)
- **Not Recommended**: > 20ms (WAN, cloud inter-region)

### Network Performance Benchmark

Use iperf3 to validate network performance:

```bash
# On server (worker node)
iperf3 -s -p 5201

# On client (master node)
# TCP throughput test
iperf3 -c 192.168.1.100 -p 5201 -t 30

# Multiple parallel streams (better for 10GbE+)
iperf3 -c 192.168.1.100 -p 5201 -P 4 -t 30

# Bidirectional test
iperf3 -c 192.168.1.100 -p 5201 -d -t 30

# UDP test with specific bandwidth
iperf3 -c 192.168.1.100 -p 5201 -u -b 1G -t 30
```

**Expected Results**:

| Network Type | Expected Throughput |
|--------------|---------------------|
| WiFi 6 (802.11ax) | 200-800 Mbps |
| Gigabit Ethernet | 900-940 Mbps |
| 2.5 GbE | 2.2-2.4 Gbps |
| 10 GbE | 9.0-9.5 Gbps |
| Thunderbolt 3/4 | 10-20 Gbps |

---

## Client Isolation Pitfalls

### Common Issues

#### 1. NAT Timeout Issues

**Symptom**: Connections drop after idle periods (5-30 minutes)

**Solution**:
```bash
# Reduce keepalive time
sudo sysctl -w net.ipv4.tcp_keepalive_time=300
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=30

# Or configure NAT device timeout (if accessible)
# Most home routers: 300-3600 seconds
```

#### 2. Ephemeral Port Exhaustion

**Symptom**: "Cannot assign requested address" or "Too many open files"

**Solution**:
```bash
# Expand ephemeral port range
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Reduce TIME_WAIT duration
sudo sysctl -w net.ipv4.tcp_fin_timeout=15

# Enable socket reuse
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Increase file descriptor limit
ulimit -n 65536
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

#### 3. VLAN/Subnet Isolation

**Symptom**: Cannot connect between devices on different VLANs

**Solution**:
- Ensure proper routing between VLANs
- Check firewall rules allow inter-VLAN traffic
- Verify all RPC ports are allowed through

#### 4. WiFi Client Isolation

**Symptom**: Devices on same WiFi network cannot reach each other

**Solution**:
- Disable "AP Isolation" or "Client Isolation" on router
- Use Ethernet for RPC traffic
- Create a dedicated network for cluster traffic

#### 5. Double-NAT Scenarios

**Symptom**: Inconsistent connectivity, some nodes unreachable

**Solution**:
- Use static IPs for all cluster nodes
- Configure port forwarding if necessary
- Consider mesh VPN (WireGuard, Tailscale)

---

## Diagnostic Tools

### Essential Connectivity Tests

```bash
#!/bin/bash
# rpc_diagnostics.sh - Run on master node

RPC_HOST="192.168.1.100"
RPC_PORT="50052"

echo "=== RPC Connectivity Diagnostics ==="
echo ""

# 1. Basic connectivity
echo "1. Ping test:"
ping -c 3 $RPC_HOST

# 2. Port availability
echo ""
echo "2. TCP port test:"
nc -zv $RPC_HOST $RPC_PORT 2>&1 || echo "FAILED: Cannot connect to $RPC_HOST:$RPC_PORT"

# 3. Connection timing
echo ""
echo "3. Connection latency:"
time (echo > /dev/tcp/$RPC_HOST/$RPC_PORT) 2>&1

# 4. Route path
echo ""
echo "4. Network path:"
traceroute -n $RPC_HOST 2>/dev/null || tracepath $RPC_HOST

# 5. DNS resolution (if using hostnames)
echo ""
echo "5. DNS resolution:"
nslookup $RPC_HOST 2>/dev/null || echo "N/A (IP address used)"

# 6. MTU check
echo ""
echo "6. MTU path check:"
ping -M do -s 1472 -c 1 $RPC_HOST 2>&1 | head -2
```

### Using netcat for RPC Testing

```bash
# Check if port is listening (from remote)
nc -zv 192.168.1.100 50052

# Check if port is open locally
netstat -tlnp | grep 50052
ss -tlnp | grep 50052

# Listen for connections (testing)
nc -l -p 50052

# Send test data
echo "test" | nc 192.168.1.100 50052
```

### Using tcpdump for Traffic Analysis

```bash
# Capture all RPC traffic on specific port
sudo tcpdump -i eth0 port 50052 -w rpc_traffic.pcap

# Live capture with details
sudo tcpdump -i eth0 port 50052 -nn -vv

# Capture with ASCII output
sudo tcpdump -i eth0 port 50052 -A

# Filter by host
sudo tcpdump -i eth0 host 192.168.1.100 and port 50052
```

### System Resource Monitoring

```bash
# Monitor network connections
watch -n 1 'netstat -an | grep 50052'
watch -n 1 'ss -s'

# Monitor system resources
htop

# I/O monitoring
iotop -o

# Network bandwidth
iftop -i eth0
nload eth0

# Socket statistics
ss -s
cat /proc/net/sockstat
```

### Python RPC Connectivity Test

```python
#!/usr/bin/env python3
"""Test RPC server connectivity."""

import socket
import time
import sys

def test_rpc_connection(host: str, port: int, timeout: float = 5.0) -> bool:
    """Test if RPC server is responding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            start = time.time()
            sock.connect((host, port))
            latency = (time.time() - start) * 1000
            print(f"✓ Connected to {host}:{port} in {latency:.2f}ms")
            return True
    except socket.timeout:
        print(f"✗ Timeout connecting to {host}:{port}")
        return False
    except ConnectionRefusedError:
        print(f"✗ Connection refused at {host}:{port}")
        return False
    except OSError as e:
        print(f"✗ Error connecting to {host}:{port}: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_rpc.py <host> <port>")
        sys.exit(1)
    
    host = sys.argv[1]
    port = int(sys.argv[2])
    
    print(f"Testing RPC connectivity to {host}:{port}...")
    
    # Multiple tests
    successes = 0
    for i in range(5):
        if test_rpc_connection(host, port):
            successes += 1
        time.sleep(0.5)
    
    print(f"\nResults: {successes}/5 successful connections")
    sys.exit(0 if successes == 5 else 1)

if __name__ == "__main__":
    main()
```

---

## Platform-Specific Considerations

### Android/Termux

#### Network Limitations

- **No root access**: Cannot modify kernel TCP parameters
- **Battery optimization**: May kill background network connections
- **WiFi power saving**: May throttle network performance
- **Mobile data**: NAT and carrier restrictions apply

#### Recommended Configuration

```bash
# In Termux - ensure wake lock is held
termux-wake-lock

# Keep SSH/network alive
export TMOUT=0

# Use screen or tmux for persistent sessions
pkg install screen tmux
screen -S rpc
```

#### Port Binding

Termux can only bind to ports > 1024 without root:

```bash
# Valid ports for Termux
./rpc-server --host 0.0.0.0 --port 50052  # OK
./rpc-server --host 0.0.0.0 --port 8080   # OK
./rpc-server --host 0.0.0.0 --port 80     # FAILS (need root)
```

#### WiFi Stability Tips

1. **Disable battery optimization** for Termux
2. **Lock WiFi on** in device settings
3. **Use static IP** (configure in router DHCP)
4. **Disable WiFi scanning** during inference

### macOS

#### Network Settings

```bash
# Check current settings
sysctl net.inet.tcp | grep keep

# Adjust keepalive (requires restart)
sudo sysctl -w net.inet.tcp.keepidle=600000  # milliseconds
sudo sysctl -w net.inet.tcp.keepintvl=30000
sudo sysctl -w net.inet.tcp.keepcnt=5

# Check socket limits
sysctl kern.maxfiles
sysctl kern.maxfilesperproc

# Increase if needed
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=65536
```

#### Firewall

```bash
# Disable macOS firewall for testing
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off

# Or add specific app exceptions
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/rpc-server
```

### Windows

#### PowerShell Network Configuration

```powershell
# Check TCP settings
netsh int tcp show global

# Enable TCP auto-tuning
netsh int tcp set global autotuninglevel=normal

# Set receive window size
netsh int tcp set global chimney=disabled

# Configure keepalive in registry
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" -Name "KeepAliveTime" -Value 600000
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" -Name "KeepAliveInterval" -Value 30000
```

#### Windows Firewall

```powershell
# Add firewall rules
New-NetFirewallRule -DisplayName "RPC Server In" -Direction Inbound -Protocol TCP -LocalPort 50052 -Action Allow
New-NetFirewallRule -DisplayName "RPC Server Out" -Direction Outbound -Protocol TCP -RemotePort 50052 -Action Allow
```

---

## Troubleshooting Guide

### Common Error Messages

#### "Connection refused" (ECONNREFUSED)

**Causes**:
1. RPC server not running
2. Wrong port number
3. Server bound to wrong interface (127.0.0.1 vs 0.0.0.0)
4. Firewall blocking port

**Solutions**:
```bash
# 1. Verify server is running
pgrep -a rpc-server
netstat -tlnp | grep 50052

# 2. Verify binding interface
ss -tlnp | grep 50052
# Should show 0.0.0.0:50052 for external access

# 3. Check firewall
sudo iptables -L -n | grep 50052

# 4. Restart with correct interface
./rpc-server --host 0.0.0.0 --port 50052
```

#### "Connection timed out" (ETIMEDOUT)

**Causes**:
1. Network unreachable
2. Firewall dropping packets (silent drop)
3. Wrong IP address
4. NAT issues

**Solutions**:
```bash
# 1. Verify network connectivity
ping <target_ip>
traceroute <target_ip>

# 2. Test with different port
nc -zv <target_ip> 22  # SSH if available

# 3. Check from target side
sudo tcpdump -i any port 50052
# Should see incoming SYN packets
```

#### "No route to host" (EHOSTUNREACH)

**Causes**:
1. Target device offline
2. Routing misconfiguration
3. Different subnet without gateway

**Solutions**:
```bash
# 1. Verify target is online
arping <target_ip>

# 2. Check routing table
ip route get <target_ip>

# 3. Verify same subnet or route exists
ip addr show
```

#### "Address already in use" (EADDRINUSE)

**Causes**:
1. Previous instance still running
2. Port in TIME_WAIT state
3. Another service using port

**Solutions**:
```bash
# 1. Kill existing process
pkill -f "rpc-server.*--port 50052"

# 2. Check what's using port
lsof -i :50052
fuser -k 50052/tcp

# 3. Wait for TIME_WAIT (or enable reuse)
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# 4. Use different port
./rpc-server --port 50053
```

#### "RPC workers not available" / Timeout waiting

**Causes**:
1. Workers started too late
2. Network latency too high
3. Workers crashed during model load
4. Timeout too short

**Solutions**:
```bash
# 1. Start workers before master
# Worker nodes first:
./rpc-server --host 0.0.0.0 --port 50052

# Then master:
./llama-server --rpc "192.168.1.100:50052" ...

# 2. Increase timeout (in exo)
# Modify wait_for_rpc_workers timeout parameter

# 3. Check worker logs
tail -f worker.log
```

### Performance Issues

#### Slow Inference Speed

**Diagnosis**:
```bash
# 1. Check network bandwidth
iperf3 -c <worker_ip> -t 30

# 2. Check network latency
ping -c 100 <worker_ip> | tail -3

# 3. Monitor network during inference
iftop -i eth0

# 4. Check for packet loss
mtr <worker_ip>
```

**Solutions**:
1. Use wired Ethernet instead of WiFi
2. Reduce number of nodes (fewer network hops)
3. Increase tensor split ratio for faster nodes
4. Check for network congestion

#### High Latency Spikes

**Causes**:
1. WiFi interference
2. Network congestion
3. TCP retransmissions
4. Background traffic

**Solutions**:
```bash
# 1. Check for retransmissions
netstat -s | grep -i retrans

# 2. Use QoS/traffic shaping
tc qdisc add dev eth0 root fq_codel

# 3. Disable TCP slow start
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0
```

---

## Configuration Scripts

### Full Linux Setup Script

```bash
#!/bin/bash
# setup_rpc_network.sh - Configure Linux for llama.cpp RPC

set -e

echo "=== llama.cpp RPC Network Setup ==="

# Create sysctl config
cat << 'EOF' | sudo tee /etc/sysctl.d/99-llama-rpc.conf
# llama.cpp RPC Network Optimization

# Connection handling
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535

# Keepalive (aggressive for RPC)
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5

# Timeouts
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_syn_retries = 3
net.ipv4.tcp_synack_retries = 3

# Performance
net.ipv4.tcp_tw_reuse = 1
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_slow_start_after_idle = 0
EOF

# Apply settings
sudo sysctl -p /etc/sysctl.d/99-llama-rpc.conf

# Configure firewall
if command -v ufw &> /dev/null; then
    sudo ufw allow 50052:50060/tcp comment "llama.cpp RPC"
    sudo ufw allow 60000:60010/tcp comment "exo RPC"
    sudo ufw allow 8080/tcp comment "llama-server HTTP"
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-port=50052-50060/tcp
    sudo firewall-cmd --permanent --add-port=60000-60010/tcp
    sudo firewall-cmd --permanent --add-port=8080/tcp
    sudo firewall-cmd --reload
fi

# Increase file descriptor limits
cat << 'EOF' | sudo tee -a /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
EOF

echo ""
echo "=== Setup Complete ==="
echo "Reboot or re-login for all changes to take effect."
echo ""
echo "Test with: sysctl net.core.somaxconn"
```

### Termux Setup Script

```bash
#!/data/data/com.termux/files/usr/bin/bash
# setup_rpc_termux.sh - Configure Termux for llama.cpp RPC

echo "=== Termux RPC Setup ==="

# Install required packages
pkg update -y
pkg install -y openssh netcat-openbsd iproute2 termux-api

# Configure SSH for remote access
if [ ! -f ~/.ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -A
fi

# Create wake lock script
cat << 'EOF' > ~/rpc_start.sh
#!/data/data/com.termux/files/usr/bin/bash
# Start RPC server with wake lock

termux-wake-lock
echo "Wake lock acquired"

# Start RPC server
cd ~/llama.cpp/build/bin
./rpc-server --host 0.0.0.0 --port 50052

# Release wake lock on exit
termux-wake-unlock
EOF
chmod +x ~/rpc_start.sh

echo ""
echo "=== Termux Setup Complete ==="
echo ""
echo "Start RPC server with: ~/rpc_start.sh"
echo "Or manually: ./rpc-server --host 0.0.0.0 --port 50052"
echo ""
echo "Your IP address:"
ip addr show wlan0 | grep "inet " | awk '{print $2}'
```

---

## Best Practices Checklist

### Before Deployment

- [ ] All nodes can ping each other
- [ ] RPC ports are open on all firewalls
- [ ] Network bandwidth tested with iperf3 (>= 1 Gbps recommended)
- [ ] Network latency tested (< 5ms recommended)
- [ ] MTU is consistent across all nodes
- [ ] Static IPs assigned (or reliable DHCP)
- [ ] Wake locks enabled on mobile devices

### RPC Server Configuration

- [ ] Server bound to `0.0.0.0` (not `127.0.0.1`)
- [ ] Correct port numbers assigned per device rank
- [ ] LD_LIBRARY_PATH set if needed
- [ ] Logs enabled for debugging

### TCP/IP Tuning

- [ ] TCP backlog increased (somaxconn >= 1024)
- [ ] Keepalive configured for NAT environments
- [ ] Socket buffer sizes increased for high throughput
- [ ] TIME_WAIT reuse enabled

### Monitoring

- [ ] Network traffic monitoring in place
- [ ] Connection state monitoring
- [ ] Log aggregation for distributed debugging
- [ ] Performance baseline established

### Security

- [ ] RPC traffic on isolated network/VLAN (if possible)
- [ ] Firewall rules restrict RPC to known IPs
- [ ] No RPC ports exposed to public internet
- [ ] Regular security updates applied

---

## See Also

- [ANDROID_SETUP.md](./ANDROID_SETUP.md) - Basic Android/Termux setup
- [TERMUX_ADVANCED.md](./TERMUX_ADVANCED.md) - Advanced Termux configuration
- [ARM_OPTIMIZATION.md](./ARM_OPTIMIZATION.md) - ARM-specific performance tuning
- [MODELS.md](./MODELS.md) - Model selection for distributed inference

---

*Last updated: December 2024*

