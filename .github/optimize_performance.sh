#!/bin/bash
set -e

# Function to log with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Applying comprehensive performance optimizations..."

# System-wide power management
log "Configuring power management..."
sudo pmset -a lessbright 0
sudo pmset -a disablesleep 1
sudo pmset -a sleep 0
sudo pmset -a hibernatemode 0
sudo pmset -a autopoweroff 0
sudo pmset -a standby 0
sudo pmset -a powernap 0
sudo pmset -a proximitywake 0
sudo pmset -a tcpkeepalive 1
sudo pmset -a powermode 2
sudo pmset -a gpuswitch 2
sudo pmset -a displaysleep 0
sudo pmset -a disksleep 0

# Memory and kernel optimizations
log "Configuring memory and kernel settings..."
sudo sysctl -w kern.memorystatus_purge_on_warning=0
sudo sysctl -w kern.memorystatus_purge_on_critical=0
sudo sysctl -w kern.timer.coalescing_enabled=0

# Metal and GPU optimizations
log "Configuring Metal and GPU settings..."
defaults write com.apple.CoreML MPSEnableGPUValidation -bool false
defaults write com.apple.CoreML MPSEnableMetalValidation -bool false
defaults write com.apple.CoreML MPSEnableGPUDebug -bool false
defaults write com.apple.Metal GPUDebug -bool false
defaults write com.apple.Metal GPUValidation -bool false
defaults write com.apple.Metal MetalValidation -bool false
defaults write com.apple.Metal MetalCaptureEnabled -bool false
defaults write com.apple.Metal MTLValidationBehavior -string "Disabled"
defaults write com.apple.Metal EnableMTLDebugLayer -bool false
defaults write com.apple.Metal MTLDebugLevel -int 0
defaults write com.apple.Metal PreferIntegratedGPU -bool false
defaults write com.apple.Metal ForceMaximumPerformance -bool true
defaults write com.apple.Metal MTLPreferredDeviceGPUFrame -bool true

# Create MPS cache directory with proper permissions
sudo mkdir -p /tmp/mps_cache
sudo chmod 777 /tmp/mps_cache

# Process and resource limits
log "Configuring process limits..."
sudo launchctl limit maxfiles 524288 524288
ulimit -n 524288 || log "Warning: Could not set file descriptor limit"
ulimit -c 0
ulimit -l unlimited || log "Warning: Could not set memory lock limit"

# Export performance-related environment variables
cat << 'EOF' > /tmp/performance_env.sh
# Metal optimizations
export MTL_DEBUG_LAYER=0
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=0
export METAL_FORCE_PERFORMANCE_MODE=1
export METAL_DEVICE_PRIORITY=high
export METAL_MAX_COMMAND_QUEUES=1024
export METAL_LOAD_LIMIT=0
export METAL_VALIDATION_ENABLED=0
export METAL_ENABLE_VALIDATION_LAYER=0
export OBJC_DEBUG_MISSING_POOLS=NO
export MPS_CACHEDIR=/tmp/mps_cache

# MLX optimizations
export MLX_USE_GPU=1
export MLX_METAL_COMPILE_ASYNC=1
export MLX_METAL_PREALLOCATE=1
export MLX_METAL_MEMORY_GUARD=0
export MLX_METAL_CACHE_KERNELS=1
export MLX_PLACEMENT_POLICY=metal
export MLX_METAL_VALIDATION=0
export MLX_METAL_DEBUG=0
export MLX_FORCE_P_CORES=1
export MLX_METAL_MEMORY_BUDGET=0
export MLX_METAL_PREWARM=1

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
EOF

log "Performance optimizations completed. Environment variables written to /tmp/performance_env.sh"