#!/usr/bin/env bash

# Get the total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))

# Calculate 80% and TOTAL_MEM_GB-5GB in MB
EIGHTY_PERCENT=$(($TOTAL_MEM_MB * 80 / 100))
MINUS_5GB=$((($TOTAL_MEM_MB - 5120)))

# Calculate 70% and TOTAL_MEM_GB-8GB in MB
SEVENTY_PERCENT=$(($TOTAL_MEM_MB * 70 / 100))
MINUS_8GB=$((($TOTAL_MEM_MB - 8192)))

# Set WIRED_LIMIT_MB to higher value
if [ $EIGHTY_PERCENT -gt $MINUS_5GB ]; then
  WIRED_LIMIT_MB=$EIGHTY_PERCENT
else
  WIRED_LIMIT_MB=$MINUS_5GB
fi

# Set WIRED_LWM_MB to higher value
if [ $SEVENTY_PERCENT -gt $MINUS_8GB ]; then
  WIRED_LWM_MB=$SEVENTY_PERCENT
else
  WIRED_LWM_MB=$MINUS_8GB
fi

# Display the calculated values
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"

# Apply the values with sysctl, but check if we're already root
if [ "$EUID" -eq 0 ]; then
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
else
  # Try without sudo first, fall back to sudo if needed
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
fi