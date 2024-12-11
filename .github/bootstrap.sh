#!/bin/bash
set -e

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

if [ "$EUID" -eq 0 ]; then 
    log "Please do not run as root. Run as regular user with sudo access."
    exit 1
fi

# Check for required arguments
if [ -z "$1" ]; then
    log "Error: Runner token is required"
    log "Usage: $0 <runner-token> [tailscale-auth-key]"
    exit 1
fi

RUNNER_TOKEN=$1
TAILSCALE_AUTH_KEY=$2
REPO="exo-explore/exo"

# Add sudoers configuration
log "Configuring sudo access..."
SUDOERS_CONTENT="$(whoami) ALL=(ALL) NOPASSWD: ALL"
echo "$SUDOERS_CONTENT" | sudo tee /etc/sudoers.d/github-runner > /dev/null
sudo chmod 440 /etc/sudoers.d/github-runner

log "Configuring privacy permissions..."
sudo tccutil reset All
sudo tccutil reset SystemPolicyAllFiles
sudo tccutil reset SystemPolicyNetworkVolumes

# For Python specifically
PYTHON_PATH="/opt/homebrew/bin/python3.12"
sudo chmod 755 "$PYTHON_PATH"

# Add to firewall
log "Configuring firewall access..."
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PYTHON_PATH"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock "$PYTHON_PATH"

# Set Homebrew paths based on architecture
if [ "$(uname -p)" = "arm" ]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

# Install Homebrew if not present
if ! command_exists brew; then
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install required packages
log "Installing required packages..."
export HOMEBREW_NO_AUTO_UPDATE=1
brew install python@3.12 coreutils

# Optional Tailscale setup if auth key is provided
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    log "Installing and configuring Tailscale..."
    brew install --quiet tailscale
    sudo brew services stop tailscale 2>/dev/null || true
    sudo rm -f /var/db/tailscale/tailscaled.state 2>/dev/null || true
    sudo brew services start tailscale
    sleep 2
    sudo tailscale up --authkey=$TAILSCALE_AUTH_KEY

    # Enable SSH and Screen Sharing
    log "Enabling remote access services..."
    sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist
    sudo /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart \
        -activate \
        -configure -access -on \
        -configure -allowAccessFor -allUsers \
        -configure -restart -agent -privs -all

    # Create launch daemon for remote access
    sudo bash -c 'cat > /Library/LaunchDaemons/com.remote.access.setup.plist' << 'EOL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.remote.access.setup</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>
            launchctl load -w /System/Library/LaunchDaemons/ssh.plist;
            /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart -activate -configure -access -on
        </string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOL

    sudo chmod 644 /Library/LaunchDaemons/com.remote.access.setup.plist
    sudo launchctl load -w /Library/LaunchDaemons/com.remote.access.setup.plist
fi

# Configure GitHub Actions Runner
log "Gathering system metadata..."
MACHINE_NAME=$(scutil --get ComputerName)
MACHINE_NAME="runner-$(echo -n "$MACHINE_NAME" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-')"

# Enhanced Apple Silicon detection
MACHINE_INFO=$(system_profiler SPHardwareDataType)
CHIP_FULL=$(echo "$MACHINE_INFO" | grep "Chip" | cut -d: -f2 | xargs)
if [[ $CHIP_FULL =~ "Apple" ]]; then
    CHIP_MODEL=$(echo "$CHIP_FULL" | sed 's/^Apple //' | tr -d ' ' | tr '[:lower:]' '[:upper:]')
    GPU_CORES=$(ioreg -l | grep "gpu-core-count" | awk -F'= ' '{print $2}')
    if [ -z "$GPU_CORES" ]; then
        GPU_CORES="N/A"
    fi
else
    CHIP_MODEL="Intel"
    GPU_CORES="N/A"
fi

MEMORY=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))

# Set up GitHub Runner
RUNNER_DIR="$HOME/actions-runner"

# Check if runner is already configured
if [ -f "$RUNNER_DIR/.runner" ]; then
  log "Runner already configured. Stopping existing service..."
  sudo launchctl unload /Library/LaunchDaemons/com.github.runner.plist 2>/dev/null || true
fi

# Create runner directory if it doesn't exist
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

CUSTOM_LABELS="self-hosted,macos,arm64,${CHIP_MODEL}_GPU${GPU_CORES}_${MEMORY}GB"

# Only download and extract if not already present or if forced
if [ ! -f "$RUNNER_DIR/run.sh" ] || [ "${FORCE_SETUP:-false}" = "true" ]; then
  log "Downloading GitHub Actions runner..."
  RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep '"tag_name":' | cut -d'"' -f4)
  curl -o actions-runner.tar.gz -L "https://github.com/actions/runner/releases/download/${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION#v}.tar.gz"
  tar xzf actions-runner.tar.gz
  rm actions-runner.tar.gz
else
  log "Runner already downloaded, skipping download step"
fi

log "Configuring runner with labels: $CUSTOM_LABELS"
./config.sh --unattended \
    --url "https://github.com/${REPO}" \
    --token "${RUNNER_TOKEN}" \
    --name "${MACHINE_NAME}" \
    --labels "${CUSTOM_LABELS}" \
    --work "_work"

# Set optimal performance settings
log "Configuring system for optimal performance..."

# Force high performance mode using native macOS tools
sudo nvram boot-args="gpuswitch=2 amfi_get_out_of_my_way=1 serverperfmode=1"
sudo pmset force -a ac

# Disable power management and force performance mode
sudo pmset -a gpuswitch 2  # Force discrete GPU
sudo pmset -a lowpowermode 0
sudo pmset -a hibernatemode 0
sudo pmset -a standby 0
sudo pmset -a autopoweroff 0
sudo pmset -a powernap 0
sudo pmset -a sleep 0
sudo pmset -a disksleep 0
sudo pmset -a powermode 1
sudo pmset -a lessbright 0
sudo pmset -a halfdim 0
sudo pmset -a acwake 0
sudo pmset -a lidwake 0
sudo pmset -a proximitywake 0

# Create performance mode configuration
sudo tee /Library/Preferences/com.apple.perfmode.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Mode</key>
    <string>Performance</string>
    <key>PowerMode</key>
    <integer>1</integer>
    <key>GPUSwitch</key>
    <integer>2</integer>
</dict>
</plist>
EOF

# Disable Metal validation and debugging
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

# Create MPS cache directory with appropriate permissions
sudo mkdir -p /tmp/mps_cache
sudo chmod 777 /tmp/mps_cache

# Create CPU affinity configuration for performance cores
sudo tee /Library/LaunchDaemons/com.github.runner.cpuaffinity.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.github.runner.cpuaffinity</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/taskpolicy</string>
        <string>-p</string>
        <string>PERFORMANCE</string>
        <string>-b</string>
        <string>PERFORMANCE</string>
        <string>-t</string>
        <string>PERFORMANCE</string>
        <string>--cpu-qos</string>
        <string>USER_INTERACTIVE</string>
        <string>--gpu-qos</string>
        <string>USER_INTERACTIVE</string>
        <string>--io-qos</string>
        <string>USER_INTERACTIVE</string>
        <string>--affinity-tag</string>
        <string>com.github.runner</string>
        <string>${RUNNER_DIR}/run.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Create and load launch daemon
log "Creating LaunchDaemon service..."
sudo tee /Library/LaunchDaemons/com.github.runner.plist > /dev/null << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.github.runner</string>
        <key>UserName</key>
        <string>$(whoami)</string>
        <key>WorkingDirectory</key>
        <string>${RUNNER_DIR}</string>
        <key>ProgramArguments</key>
        <array>
            <string>/usr/bin/taskpolicy</string>
            <string>-p</string>
            <string>PERFORMANCE</string>
            <string>-b</string>
            <string>PERFORMANCE</string>
            <string>-t</string>
            <string>PERFORMANCE</string>
            <string>${RUNNER_DIR}/run.sh</string>
        </array>
        <key>EnvironmentVariables</key>
        <dict>
            <key>PATH</key>
            <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
            <!-- MLX Settings -->
            <key>MLX_USE_GPU</key>
            <string>1</string>
            <key>MLX_METAL_COMPILE_ASYNC</key>
            <string>1</string>
            <key>MLX_METAL_PREALLOCATE</key>
            <string>1</string>
            <key>MLX_METAL_MEMORY_GUARD</key>
            <string>0</string>
            <key>MLX_METAL_CACHE_KERNELS</key>
            <string>1</string>
            <key>MLX_PLACEMENT_POLICY</key>
            <string>metal</string>
            <key>MLX_METAL_VALIDATION</key>
            <string>0</string>
            <key>MLX_METAL_DEBUG</key>
            <string>0</string>
            <key>MLX_FORCE_P_CORES</key>
            <string>1</string>
            <!-- Metal Settings -->
            <key>METAL_DEBUG_ERROR_MODE</key>
            <string>0</string>
            <key>METAL_DEVICE_WRAPPER_TYPE</key>
            <string>1</string>
            <key>METAL_CAPTURE_ENABLED</key>
            <string>0</string>
            <key>METAL_DEBUG_LAYER_ENABLED</key>
            <string>0</string>
            <key>METAL_LOAD_LIMIT</key>
            <string>0</string>
            <key>METAL_MAX_COMMAND_QUEUES</key>
            <string>1024</string>
            <key>METAL_DEVICE_PRIORITY</key>
            <string>high</string>
            <key>METAL_FORCE_PERFORMANCE_MODE</key>
            <string>1</string>
            <key>METAL_VALIDATION_ENABLED</key>
            <string>0</string>
            <key>METAL_ENABLE_VALIDATION_LAYER</key>
            <string>0</string>
            <!-- Process Priority -->
            <key>MPS_PREFETCH_ENABLED</key>
            <string>1</string>
            <key>MPS_CACHE_DIRECTORY</key>
            <string>/tmp/mps_cache</string>
            <key>DISABLE_METAL_VALIDATION</key>
            <string>1</string>
            <!-- CPU Affinity -->
            <key>QOS_CLASS_USER_INTERACTIVE</key>
            <string>1</string>
            <key>PERFORMANCE_MODE</key>
            <string>1</string>
        </dict>
        <key>Nice</key>
        <integer>-20</integer>
        <key>ProcessType</key>
        <string>Interactive</string>
        <key>LowPriorityIO</key>
        <false/>
    </dict>
</plist>
EOF

# Load the LaunchDaemons
sudo launchctl load -w /Library/LaunchDaemons/com.github.runner.cpuaffinity.plist
sudo launchctl load -w /Library/LaunchDaemons/com.github.runner.plist

# Set CPU affinity for the current process
taskpolicy -p PERFORMANCE -b PERFORMANCE -t PERFORMANCE

# Add Runner.Listener permissions (after runner installation)
RUNNER_PATH="$RUNNER_DIR/bin/Runner.Listener"
sudo chmod 755 "$RUNNER_PATH"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$RUNNER_PATH"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock "$RUNNER_PATH"

# Create connection info file if Tailscale is configured
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    TAILSCALE_IP=$(tailscale ip)
    cat > "$HOME/remote_access_info.txt" << EOL
Mac Remote Access Information
============================
Computer Name: $MACHINE_NAME
Username: $USER
Tailscale IP: $TAILSCALE_IP

SSH Command: ssh $USER@$TAILSCALE_IP
Screen Sharing: vnc://$TAILSCALE_IP
EOL
    chmod 600 "$HOME/remote_access_info.txt"
fi

log "Verifying runner service status..."
if sudo launchctl list | grep com.github.runner > /dev/null; then
    log "GitHub Actions runner service is running successfully!"
    log "Runner labels: $CUSTOM_LABELS"
    [ -n "$TAILSCALE_AUTH_KEY" ] && log "Remote access details saved to: $HOME/remote_access_info.txt"
else
    log "Error: Failed to start GitHub Actions runner service"
    exit 1
fi