#!/bin/bash

# Get the total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))

# Calculate the maximum limit as 80% of the total memory
WIRED_LIMIT_MB=$(($TOTAL_MEM_MB * 80 / 100))

# Calculate the lower bound as 60%-80% of the maximum limit
WIRED_LWM_MIN=$(($WIRED_LIMIT_MB * 60 / 100))
WIRED_LWM_MAX=$(($WIRED_LIMIT_MB * 80 / 100))

# Choose a mid-range value for wired_lwm_mb
WIRED_LWM_MB=$(($WIRED_LWM_MIN + ($WIRED_LWM_MAX - $WIRED_LWM_MIN) / 2))

# Display the calculated values
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"

# Apply the values with sysctl
sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
