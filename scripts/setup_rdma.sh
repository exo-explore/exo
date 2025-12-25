#!/bin/bash
# Setup RDMA over Thunderbolt (disables bridge0, configures en5 directly)
# Run with sudo on each machine

set -e

HOSTNAME=$(hostname -s)

case "$HOSTNAME" in
    jax|Jax)
        LOCAL_IP="192.168.0.1"
        REMOTE_IP="192.168.0.2"
        ;;
    kee|Kee)
        LOCAL_IP="192.168.0.2"
        REMOTE_IP="192.168.0.1"
        ;;
    *)
        echo "Unknown hostname: $HOSTNAME"
        echo "Usage: Set LOCAL_IP and REMOTE_IP environment variables, or run on jax/kee"
        if [ -z "$LOCAL_IP" ] || [ -z "$REMOTE_IP" ]; then
            exit 1
        fi
        ;;
esac

echo "Setting up RDMA on $HOSTNAME"
echo "  Local IP: $LOCAL_IP"
echo "  Remote IP: $REMOTE_IP"
echo ""

# Disable the Thunderbolt bridge
echo "Disabling bridge0..."
sudo ifconfig bridge0 down

# Configure IP directly on en5
echo "Configuring en5 with IP $LOCAL_IP..."
sudo ifconfig en5 inet $LOCAL_IP netmask 255.255.255.252

# Add route to remote
echo "Adding route to $REMOTE_IP via en5..."
sudo route change $REMOTE_IP -interface en5 2>/dev/null || sudo route add $REMOTE_IP -interface en5

echo ""
echo "RDMA setup complete!"
echo "Verify with: ibv_devinfo -d rdma_en5"
