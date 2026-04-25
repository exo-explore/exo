#!/bin/bash
# Install systemd services for RKLLAMA and Exo on RK3588
# Run this script as root on the RK3588 device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing RKLLM systemd services..."

# Copy service files
cp "$SCRIPT_DIR/rkllama.service" /etc/systemd/system/
cp "$SCRIPT_DIR/exo-rkllm.service" /etc/systemd/system/

# Set permissions
chmod 644 /etc/systemd/system/rkllama.service
chmod 644 /etc/systemd/system/exo-rkllm.service

# Reload systemd
systemctl daemon-reload

# Enable services to start on boot
systemctl enable rkllama.service
systemctl enable exo-rkllm.service

echo ""
echo "Services installed and enabled!"
echo ""
echo "Commands:"
echo "  Start:   systemctl start rkllama exo-rkllm"
echo "  Stop:    systemctl stop exo-rkllm rkllama"
echo "  Status:  systemctl status rkllama exo-rkllm"
echo "  Logs:    journalctl -u rkllama -u exo-rkllm -f"
echo ""
echo "To start now:"
echo "  systemctl start rkllama && sleep 5 && systemctl start exo-rkllm"
