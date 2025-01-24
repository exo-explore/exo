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

# Configure power management for maximum performance
log "Configuring power management..."
sudo pmset -a powermode 2  # Force highest performance mode
sudo pmset -a gpuswitch 2  # Force discrete/high-performance GPU
sudo pmset -a lowpowermode 0
sudo pmset -a lessbright 0
sudo pmset -a disablesleep 1
sudo pmset -a sleep 0
sudo pmset -a hibernatemode 0
sudo pmset -a autopoweroff 0
sudo pmset -a standby 0
sudo pmset -a powernap 0

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

# Configure CPU performance
log "Setting CPU performance controls..."
# Disable timer coalescing
sudo sysctl -w kern.timer.coalescing_enabled=0
sudo sysctl -w kern.timer_coalesce_bg_scale=-5
sudo sysctl -w kern.timer_resort_threshold_ns=0
# Set minimum timer intervals
sudo sysctl -w kern.wq_max_timer_interval_usecs=1000
sudo sysctl -w kern.timer_coalesce_bg_ns_max=1000
# Set minimum timer coalescing for all tiers
sudo sysctl -w kern.timer_coalesce_tier0_scale=-5
sudo sysctl -w kern.timer_coalesce_tier0_ns_max=1000
sudo sysctl -w kern.timer_coalesce_tier1_scale=-5
sudo sysctl -w kern.timer_coalesce_tier1_ns_max=1000
sudo sysctl -w kern.timer_coalesce_tier2_scale=-5
sudo sysctl -w kern.timer_coalesce_tier2_ns_max=1000
sudo sysctl -w kern.timer_coalesce_tier3_scale=-5
sudo sysctl -w kern.timer_coalesce_tier3_ns_max=1000
sudo sysctl -w kern.timer_coalesce_tier4_scale=-5
sudo sysctl -w kern.timer_coalesce_tier4_ns_max=1000
# Disable QoS restrictions
sudo sysctl -w net.qos.policy.restricted=0
sudo sysctl -w net.qos.policy.restrict_avapps=0
sudo sysctl -w net.qos.policy.wifi_enabled=0
sudo sysctl -w net.qos.policy.capable_enabled=0
# Set scheduler parameters
sudo sysctl -w kern.sched_rt_avoid_cpu0=0
sudo sysctl -w debug.sched=2
sudo sysctl -w net.pktsched.netem.sched_output_ival_ms=1

# Clean up any existing runner services
log "Cleaning up existing runner services..."
for service in com.github.runner com.github.runner.monitor com.github.runner.cpuaffinity com.github.runner.affinity; do
    sudo launchctl bootout system/$service 2>/dev/null || true
    sudo rm -f /Library/LaunchDaemons/$service.plist
done

# Create a simple runner service configuration
sudo tee /Library/LaunchDaemons/com.github.runner.plist > /dev/null << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.github.runner</string>
        <key>UserName</key>
        <string>$(whoami)</string>
        <key>GroupName</key>
        <string>staff</string>
        <key>WorkingDirectory</key>
        <string>$RUNNER_DIR</string>
        <key>ProgramArguments</key>
        <array>
            <string>$RUNNER_DIR/run.sh</string>
        </array>
        <key>RunAtLoad</key>
        <true/>
        <key>KeepAlive</key>
        <dict>
            <key>SuccessfulExit</key>
            <false/>
            <key>Crashed</key>
            <true/>
        </dict>
        <key>ProcessType</key>
        <string>Interactive</string>
        <key>LowPriorityIO</key>
        <false/>
        <key>AbandonProcessGroup</key>
        <false/>
        <key>EnableTransactions</key>
        <true/>
        <key>ThrottleInterval</key>
        <integer>0</integer>
        <key>HardResourceLimits</key>
        <dict>
            <key>NumberOfFiles</key>
            <integer>524288</integer>
            <key>MemoryLock</key>
            <integer>-1</integer>
        </dict>
        <key>SoftResourceLimits</key>
        <dict>
            <key>NumberOfFiles</key>
            <integer>524288</integer>
            <key>MemoryLock</key>
            <integer>-1</integer>
        </dict>
        <key>QOSClass</key>
        <string>User-Interactive</string>
        <key>StandardOutPath</key>
        <string>$RUNNER_DIR/_diag/runner.log</string>
        <key>StandardErrorPath</key>
        <string>$RUNNER_DIR/_diag/runner.err</string>
        <key>EnvironmentVariables</key>
        <dict>
            <key>PATH</key>
            <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        </dict>
        <key>Nice</key>
        <integer>-20</integer>
    </dict>
</plist>
EOF

# Set proper permissions for the LaunchDaemon
sudo chown root:wheel /Library/LaunchDaemons/com.github.runner.plist
sudo chmod 644 /Library/LaunchDaemons/com.github.runner.plist

# Remove any existing service
sudo launchctl bootout system/com.github.runner 2>/dev/null || true

# Load the new service using bootstrap
sudo launchctl bootstrap system /Library/LaunchDaemons/com.github.runner.plist

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