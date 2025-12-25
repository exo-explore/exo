#!/bin/bash
# Restore Thunderbolt Bridge (undoes setup_rdma.sh)
# Run with sudo on each machine

set -e

HOSTNAME=$(hostname -s)

echo "Restoring Thunderbolt Bridge on $HOSTNAME"
echo ""

# Remove IP from en5
echo "Removing IP from en5..."
sudo ifconfig en5 inet delete 2>/dev/null || true

# Re-enable the Thunderbolt bridge
echo "Re-enabling bridge0..."
sudo ifconfig bridge0 up

# The bridge should automatically pick up a link-local IP or use DHCP
echo "Waiting for bridge0 to get an IP..."
sleep 2

# Show the result
echo ""
echo "Bridge restored! Current bridge0 status:"
ifconfig bridge0 | grep -E "^bridge0:|inet "

echo ""
echo "Note: If bridge0 doesn't get an IP automatically, you may need to:"
echo "  sudo ipconfig set bridge0 DHCP"
echo "  or"
echo "  sudo ifconfig bridge0 inet 169.254.1.X netmask 255.255.255.255"
