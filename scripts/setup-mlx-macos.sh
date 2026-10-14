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

if [[ "$(uname -m)" != "arm64" ]]; then
  die "MLX requires Apple Silicon (arm64). Detected architecture: $(uname -m)
       MLX does not support Intel Macs."
fi

# ---------------------------------------------------------------------------
# 2. Xcode.app check
# ---------------------------------------------------------------------------
# The full Xcode application (not just Command Line Tools) is required to
# compile MLX Metal kernels. Xcode may live at a non-default path (e.g. a beta
# install like /Applications/Xcode-beta.app) or already be selected via
# xcode-select / DEVELOPER_DIR, so discover it in order of preference:
#   1. An explicit DEVELOPER_DIR that points inside an Xcode .app.
#   2. The currently selected developer dir (xcode-select -p), if it is an Xcode
#      .app rather than the Command Line Tools.
#   3. Any /Applications/Xcode*.app (covers stable and beta installs).
#   4. The conventional /Applications/Xcode.app fallback.

# Given a Developer dir, return the enclosing .app bundle path (or empty).
xcode_app_from_developer_dir() {
  local dev_dir="$1"
  case "$dev_dir" in
    */Contents/Developer) printf '%s' "${dev_dir%/Contents/Developer}" ;;
    *) printf '' ;;
  esac
}

XCODE_APP=""
XCODE_DEVELOPER_DIR=""

# 1. Honor an explicitly exported DEVELOPER_DIR if it points inside an Xcode.app.
if [[ -n "${DEVELOPER_DIR:-}" ]]; then
  candidate_app="$(xcode_app_from_developer_dir "$DEVELOPER_DIR")"
  if [[ -n "$candidate_app" && -d "$candidate_app" ]]; then
    XCODE_APP="$candidate_app"
  fi
fi

# 2. Use the currently selected developer dir if it is a full Xcode (not CLT).
if [[ -z "$XCODE_APP" ]]; then
  CURRENT_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"
  candidate_app="$(xcode_app_from_developer_dir "$CURRENT_DEVELOPER_DIR")"
  if [[ -n "$candidate_app" && -d "$candidate_app" ]]; then
    XCODE_APP="$candidate_app"
  fi
fi

# 3. Fall back to any Xcode*.app under /Applications (stable or beta).
if [[ -z "$XCODE_APP" ]]; then
  for candidate_app in /Applications/Xcode.app /Applications/Xcode*.app; do
    if [[ -d "$candidate_app" ]]; then
      XCODE_APP="$candidate_app"
      break
    fi
  done
fi

if [[ -z "$XCODE_APP" || ! -d "$XCODE_APP" ]]; then
  die "No Xcode.app found (checked DEVELOPER_DIR, xcode-select, and /Applications/Xcode*.app).
       The full Xcode application is required to compile MLX Metal kernels.
       Install Xcode from the App Store: https://apps.apple.com/app/xcode/id497799835
       Command Line Tools alone are not sufficient."
fi

XCODE_DEVELOPER_DIR="${XCODE_APP}/Contents/Developer"
info "Xcode.app found at ${XCODE_APP}"

# ---------------------------------------------------------------------------
# 3. xcode-select check
# ---------------------------------------------------------------------------
CURRENT_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"

if [[ "$CURRENT_DEVELOPER_DIR" != "$XCODE_DEVELOPER_DIR" ]]; then
  warn "xcode-select points to '${CURRENT_DEVELOPER_DIR}' instead of ${XCODE_DEVELOPER_DIR}."
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
