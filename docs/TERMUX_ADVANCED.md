# Termux Advanced Topics

> **Advanced Termux configuration for distributed AI clusters**

This guide covers advanced Termux topics: APIs, networking, background execution, auto-start, and cluster operations. For basic setup, see [ANDROID_SETUP.md](./ANDROID_SETUP.md).

---

## Table of Contents

1. [Termux Add-ons & APIs](#termux-add-ons--apis)
2. [SSH Server Setup](#ssh-server-setup)
3. [Network Discovery](#network-discovery)
4. [Background Execution](#background-execution)
5. [Auto-Start on Boot](#auto-start-on-boot)
6. [ADB Automation](#adb-automation)
7. [proot-distro (Full Linux)](#proot-distro-full-linux)
8. [Security Best Practices](#security-best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Termux Add-ons & APIs

### Termux:API

Access Android device features from the command line.

**Install:**

1. Install **Termux:API** app from F-Droid
2. Install the package:

```bash
pkg install termux-api
```

**Useful Commands:**

| Command | Description |
|---------|-------------|
| `termux-battery-status` | Battery level and charging status |
| `termux-wake-lock` | Prevent device sleep |
| `termux-wake-unlock` | Release wake lock |
| `termux-notification` | Send Android notifications |
| `termux-wifi-connectioninfo` | WiFi connection details |
| `termux-vibrate` | Vibrate device |

**Example: Battery check before compute:**

```bash
#!/bin/bash
BATTERY=$(termux-battery-status | jq -r '.percentage')
if [ "$BATTERY" -lt 20 ]; then
    echo "Low battery ($BATTERY%). Aborting."
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
# Scripts in this directory can be triggered by Tasker
```

---

## SSH Server Setup

### Starting SSH Server

```bash
pkg install openssh

# Generate host keys (first time)
ssh-keygen -A

# Set a password
passwd

# Start SSH daemon
sshd

# Verify running
pgrep sshd
```

### Connection Info

```bash
# Find your username
whoami  # Usually 'u0_a###'

# Find your IP
ip addr show wlan0 | grep inet

# Termux SSH port is 8022 (not 22)
# Connect: ssh -p 8022 user@device_ip
```

### Key-Based Authentication

```bash
# On client machine
ssh-keygen -t ed25519
ssh-copy-id -p 8022 user@device_ip

# Or manually add to Termux
echo "your-public-key" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Secure Configuration

```bash
cat >> $PREFIX/etc/ssh/sshd_config << EOF
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
EOF

pkill sshd && sshd
```

---

## Network Discovery

### Find Devices on Network

```bash
pkg install nmap

# Scan local network
nmap -sn 192.168.1.0/24

# Find devices with SSH open
nmap -p 8022 192.168.1.0/24
```

### mDNS Discovery

```bash
pkg install avahi

# Publish this device
avahi-publish-service "termux-node-$(hostname)" _ssh._tcp 8022 &

# Discover other services
avahi-browse -a
```

### Static IP Configuration

For reliable clusters, assign static IPs via your router's DHCP settings.

---

## Background Execution

### Wake Lock

Prevent Android from killing Termux:

```bash
# Acquire wake lock
termux-wake-lock

# Release when done
termux-wake-unlock
```

### Running in Background

```bash
# Using nohup
nohup python cluster_node.py > node.log 2>&1 &

# Using screen
pkg install screen
screen -S cluster
python cluster_node.py
# Detach: Ctrl+A, D
# Reattach: screen -r cluster

# Using tmux
pkg install tmux
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
            echo "Node running (PID: $(cat ~/cluster/node.pid))"
        else
            echo "Node not running"
        fi
        ;;
esac
```

### Disable Battery Optimization

1. Go to **Settings** → **Apps** → **Termux**
2. Select **Battery**
3. Choose **Unrestricted** or **Don't optimize**

---

## Auto-Start on Boot

Using **Termux:Boot**:

```bash
mkdir -p ~/.termux/boot

cat > ~/.termux/boot/01-cluster-init.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash

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

**Best Practices:**

- Number scripts for order: `01-init.sh`, `02-services.sh`
- Add delays (network may not be immediately available)
- Log everything for debugging
- Use notifications to confirm startup

---

## ADB Automation

### Input Commands via ADB

Automate Termux setup from a PC:

```bash
# Open Termux
adb shell am start -n com.termux/.HomeActivity
sleep 3

# Type commands (space = %s)
adb shell input text "pkg%supdate"
adb shell input keyevent 66  # Enter
```

### Automation Script

```bash
#!/bin/bash
# setup_termux_node.sh

run_cmd() {
    local cmd="$1"
    local escaped="${cmd// /%s}"
    adb shell input text "$escaped"
    adb shell input keyevent 66
    sleep 2
}

# Start Termux
adb shell am start -n com.termux/.HomeActivity
sleep 5

# Run setup
run_cmd "pkg update -y"
sleep 10
run_cmd "pkg install -y openssh python git"
sleep 60
run_cmd "sshd"

echo "Setup complete!"
```

### Key Event Codes

| Key | Code | Key | Code |
|-----|------|-----|------|
| Enter | 66 | Tab | 61 |
| Backspace | 67 | Escape | 111 |
| Home | 3 | Back | 4 |

---

## proot-distro (Full Linux)

Run full Linux distributions inside Termux.

### Install

```bash
pkg install proot-distro

# List available distros
proot-distro list
```

### Install Ubuntu

```bash
proot-distro install ubuntu
proot-distro login ubuntu

# Now in Ubuntu:
apt update && apt upgrade -y
apt install python3 python3-pip
```

### Run Commands Directly

```bash
proot-distro login ubuntu -- apt install python3 -y
proot-distro login ubuntu -- bash -c "cd /project && python train.py"
```

### Shared Storage

```bash
proot-distro login ubuntu --bind ~/shared:/shared
```

---

## Security Best Practices

### SSH Security

1. Use key-based authentication only
2. Disable password authentication
3. Use strong Ed25519 keys

### API Key Management

```python
import os
from cryptography.fernet import Fernet

key = Fernet.generate_key()
with open(os.path.expanduser('~/.secret_key'), 'wb') as f:
    f.write(key)
os.chmod(os.path.expanduser('~/.secret_key'), 0o600)

cipher = Fernet(key)
encrypted = cipher.encrypt(b"your-api-key")
decrypted = cipher.decrypt(encrypted)
```

### Checklist

- [ ] Disable password SSH authentication
- [ ] Use VPN on untrusted networks
- [ ] Implement node authentication
- [ ] Validate all incoming data
- [ ] Regular updates: `pkg update && pkg upgrade -y`

---

## Troubleshooting

### Known Harmless Warnings

**psutil swap memory warning:**
```
RuntimeWarning: 'sin' and 'sout' swap memory stats couldn't be determined
```
Termux can't read `/proc/vmstat` for swap stats. This is expected and doesn't affect functionality.

### Common Issues

**Permission Denied:**
```bash
chmod +x script.sh
termux-setup-storage
```

**SSH Connection Refused:**
```bash
pgrep sshd || sshd
netstat -tlnp | grep 8022
ip addr show wlan0
```

**Package Installation Fails:**
```bash
pkg clean
rm -rf $PREFIX/var/cache/apt/
pkg update
termux-change-repo
```

**Out of Memory:**
```bash
free -h
top -o %MEM
# Close unused Android apps
```

**Network Issues:**
```bash
ping -c 3 google.com
nslookup google.com
# Toggle airplane mode if needed
```

---

## Quick Reference

### Package Management

```bash
pkg update && pkg upgrade -y
pkg install <package>
pkg uninstall <package>
pkg search <query>
pkg list-installed
```

### Termux Commands

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

### Directory Structure

```
/data/data/com.termux/files/
├── home/              # ~ (home directory)
├── usr/               # System binaries, libraries
│   ├── bin/           # Executables
│   ├── lib/           # Libraries
│   └── share/         # Shared data
```

### Shared Storage (after termux-setup-storage)

```
~/storage/
├── dcim/          # Camera photos
├── downloads/     # Downloads folder
├── shared/        # Internal shared storage root
└── external-1/    # SD card (if present)
```

---

## See Also

- [Android Setup](./ANDROID_SETUP.md) - Basic exo installation
- [ARM Optimization](./ARM_OPTIMIZATION.md) - CPU-specific tuning
- [Models](./MODELS.md) - Model selection and download

---

*Last updated: December 2024*

