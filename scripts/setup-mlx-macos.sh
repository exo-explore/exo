#!/usr/bin/env bash
# scripts/setup-mlx-macos.sh
#
# Sets up the MLX inference backend for exo on macOS.
#
# MLX requires the full Xcode toolchain (not just Command Line Tools) in order
# to compile Metal GPU kernels. This script verifies each prerequisite and
# provides clear remediation steps when something is missing.
#
# Usage:
#   bash scripts/setup-mlx-macos.sh
#
# What it does:
#   1. Confirms the platform is macOS.
#   2. Confirms Xcode.app is installed.
#   3. Checks that xcode-select points to Xcode.app (not CLT).
#   4. Verifies the Xcode license has been accepted.
#   5. Downloads the Metal Toolchain component if it is absent.
#   6. Runs `uv sync --extra mlx` to install all MLX Python dependencies.
#   7. Imports mlx to confirm the GPU is visible.

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup-mlx]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup-mlx] WARNING:${NC} $*"; }
error()   { echo -e "${RED}[setup-mlx] ERROR:${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Platform check
# ---------------------------------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
  die "MLX requires macOS (Apple Silicon). Detected: $(uname)"
fi

# ---------------------------------------------------------------------------
# 2. Xcode.app check
# ---------------------------------------------------------------------------
XCODE_APP="/Applications/Xcode.app"
XCODE_DEVELOPER_DIR="${XCODE_APP}/Contents/Developer"

if [[ ! -d "$XCODE_APP" ]]; then
  die "Xcode.app not found at ${XCODE_APP}.
       The full Xcode application is required to compile MLX Metal kernels.
       Install Xcode from the App Store: https://apps.apple.com/app/xcode/id497799835
       Command Line Tools alone are not sufficient."
fi

info "Xcode.app found at ${XCODE_APP}"

# ---------------------------------------------------------------------------
# 3. xcode-select check
# ---------------------------------------------------------------------------
CURRENT_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"

if [[ "$CURRENT_DEVELOPER_DIR" != "$XCODE_DEVELOPER_DIR" ]]; then
  warn "xcode-select points to '${CURRENT_DEVELOPER_DIR}' instead of Xcode.app."
  echo "       Fix with: sudo xcode-select -s ${XCODE_DEVELOPER_DIR}"
  echo ""
  echo "       Attempting to proceed using DEVELOPER_DIR environment variable..."
  export DEVELOPER_DIR="$XCODE_DEVELOPER_DIR"
else
  info "xcode-select → ${CURRENT_DEVELOPER_DIR}"
fi

# ---------------------------------------------------------------------------
# 4. Xcode license check
# ---------------------------------------------------------------------------
if ! DEVELOPER_DIR="${DEVELOPER_DIR:-$XCODE_DEVELOPER_DIR}" \
     xcrun --sdk macosx --show-sdk-path &>/dev/null 2>&1; then
  die "The Xcode license has not been accepted.
       Run the following command and follow the prompts:

         sudo xcodebuild -license accept

       Then re-run this script."
fi

info "Xcode license accepted"

# ---------------------------------------------------------------------------
# 5. Metal Toolchain check / download
# ---------------------------------------------------------------------------
METAL_VERSION=""
if ! METAL_VERSION=$(DEVELOPER_DIR="${DEVELOPER_DIR:-$XCODE_DEVELOPER_DIR}" \
                     xcrun --sdk macosx metal --version 2>&1); then
  # metal is unavailable — try to download the component
  warn "Metal Toolchain not found. Downloading via xcodebuild -downloadComponent MetalToolchain..."
  if ! xcodebuild -downloadComponent MetalToolchain; then
    die "Failed to download the Metal Toolchain.
         Try manually: xcodebuild -downloadComponent MetalToolchain
         Or install Xcode from the App Store and open it once to trigger component downloads."
  fi
  # Re-check after download
  METAL_VERSION=$(DEVELOPER_DIR="${DEVELOPER_DIR:-$XCODE_DEVELOPER_DIR}" \
                  xcrun --sdk macosx metal --version 2>&1) \
    || die "Metal Toolchain download succeeded but 'metal' is still unavailable.
            Please open Xcode.app once to finish component installation, then retry."
fi

info "Metal compiler: ${METAL_VERSION%%$'\n'*}"

# ---------------------------------------------------------------------------
# 6. uv sync --extra mlx
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  die "'uv' not found. Install it with:  brew install uv
       Or: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

info "Running: uv sync --extra mlx"
uv sync --extra mlx

# ---------------------------------------------------------------------------
# 7. Verify mlx import and device
# ---------------------------------------------------------------------------
info "Verifying MLX installation..."
MLX_INFO=$(uv run python - <<'EOF'
import mlx.core as mx
print(f"mlx {mx.__version__} | device: {mx.default_device()}")
EOF
)
info "MLX ready — ${MLX_INFO}"
echo ""
echo -e "${GREEN}MLX setup complete. Run exo with:${NC}  uv run exo"
