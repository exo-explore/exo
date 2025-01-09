#!/usr/bin/env bash

# Get the total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))

# Set WIRED_LIMIT_MB to 80%
WIRED_LIMIT_MB=$(($TOTAL_MEM_MB * 80 / 100))
# Set  WIRED_LWM_MB to 70%
WIRED_LWM_MB=$(($TOTAL_MEM_MB * 70 / 100))

# Display the calculated values
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"

# Apply the values with sysctl
sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB