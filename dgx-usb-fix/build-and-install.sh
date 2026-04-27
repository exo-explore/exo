#!/usr/bin/env bash
set -Eeuo pipefail

FIX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

KERNEL_VERSION="${KERNEL_VERSION:-$(uname -r)}"
WORKDIR="${WORKDIR:-${HOME}/apple1905-cdcncm}"
KERNEL_BUILD_DIR="/lib/modules/${KERNEL_VERSION}/build"
INSTALL_DIR="${INSTALL_DIR:-/lib/modules/${KERNEL_VERSION}/updates/dgx-usb-fix}"
MOK_KEY="${MOK_KEY:-/root/MOK.priv}"
MOK_CERT="${MOK_CERT:-/root/MOK.der}"
SKIP_LOAD=0

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1. Run from: nix develop .#dgx-usb-fix"
}

usage() {
  cat <<'EOF'
Usage: sudo ./dgx-usb-fix/build-and-install.sh [options]

Options:
  --skip-load      Build, sign, install, and depmod, but do not unload/reload cdc_ncm.
  -h, --help       Show this help.

Environment:
  KERNEL_VERSION   Kernel version to build for. Defaults to uname -r.
  WORKDIR          Temporary source/build workspace. Defaults to $HOME/apple1905-cdcncm.
  INSTALL_DIR      Module install directory. Defaults to /lib/modules/$KERNEL_VERSION/updates/dgx-usb-fix.
  MOK_KEY          Secure Boot signing key. Defaults to /root/MOK.priv.
  MOK_CERT         Secure Boot signing cert. Defaults to /root/MOK.der.

Build tools are provided by the repo dev shell:

  nix develop .#dgx-usb-fix
  sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh
EOF
}

while (($#)); do
  case "$1" in
    --skip-load)
      SKIP_LOAD=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ $EUID -eq 0 ]] || die "Run with sudo"
[[ -d $KERNEL_BUILD_DIR ]] || die "Missing kernel headers: $KERNEL_BUILD_DIR"
need dpkg-query

if [[ -z ${DGX_USB_FIX_NIX_ENV:-} ]]; then
  cat >&2 <<'EOF'
WARNING: DGX_USB_FIX_NIX_ENV is not set.
Expected flow:
  nix develop .#dgx-usb-fix
  sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh
Continuing because required tools may already be available.
EOF
fi

for cmd in curl dpkg-source find make modinfo depmod python3 mokutil; do
  need "$cmd"
done

resolve_source() {
  local source_info source_package source_version orig_version

  source_info="$(
    dpkg-query -W -f='${source:Package} ${source:Version}\n' \
      "linux-image-${KERNEL_VERSION}" 2>/dev/null || true
  )"
  source_package="${source_info%% *}"
  source_version="${source_info##* }"

  [[ -n $source_package && -n $source_version && $source_package != "$source_version" ]] \
    || die "Could not resolve source package for linux-image-${KERNEL_VERSION}"

  source_package="${source_package/linux-signed-/linux-}"
  orig_version="${source_version%-*}"

  SRC_PACKAGE="$source_package"
  SRC_VERSION="$source_version"
  ORIG_VERSION="$orig_version"
  SRC_URL="https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/${SRC_PACKAGE}/${SRC_VERSION}"
}

download_source() {
  local orig_tarball debian_diff dsc_file

  log "Preparing workspace at ${WORKDIR}"
  rm -rf "$WORKDIR"
  mkdir -p "$WORKDIR"
  cd "$WORKDIR"

  orig_tarball="${SRC_PACKAGE}_${ORIG_VERSION}.orig.tar.gz"
  debian_diff="${SRC_PACKAGE}_${SRC_VERSION}.diff.gz"
  dsc_file="${SRC_PACKAGE}_${SRC_VERSION}.dsc"

  log "Downloading ${SRC_PACKAGE} ${SRC_VERSION}"
  curl -fL "${SRC_URL}/${orig_tarball}" -o "${orig_tarball}"
  curl -fL "${SRC_URL}/${debian_diff}" -o "${debian_diff}"
  curl -fL "${SRC_URL}/${dsc_file}" -o "${dsc_file}"

  log "Extracting source"
  dpkg-source --no-check -x "${dsc_file}"
  SRCDIR="$(find . -maxdepth 1 -mindepth 1 -type d -name "${SRC_PACKAGE}-*" | head -n1)"
  [[ -n $SRCDIR ]] || die "Could not find extracted source directory"
  SRCDIR="${WORKDIR}/${SRCDIR#./}"
}

patch_source() {
  local source_file="${SRCDIR}/drivers/net/usb/cdc_ncm.c"

  [[ -f $source_file ]] || die "cdc_ncm.c not found at $source_file"
  log "Patching cdc_ncm.c"
  python3 "${FIX_ROOT}/scripts/patch_cdc_ncm.py" "$source_file"
}

build_module() {
  BUILD_DIR="${WORKDIR}/buildmod"
  log "Building patched module in ${BUILD_DIR}"
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  install -m 0644 "${SRCDIR}/drivers/net/usb/cdc_ncm.c" "${BUILD_DIR}/cdc_ncm.c"
  install -m 0644 "${FIX_ROOT}/templates/Makefile" "${BUILD_DIR}/Makefile"

  make -C "$KERNEL_BUILD_DIR" M="$BUILD_DIR" modules

  PATCHED_MODULE="${BUILD_DIR}/cdc_ncm.ko"
  [[ -f $PATCHED_MODULE ]] || die "Build failed: cdc_ncm.ko not produced"
}

secure_boot_enabled() {
  mokutil --sb-state 2>/dev/null | grep -q "SecureBoot enabled"
}

sign_module_if_needed() {
  local sign_file="${KERNEL_BUILD_DIR}/scripts/sign-file"

  if secure_boot_enabled; then
    log "Secure Boot is enabled; signing module"
    [[ -f $MOK_KEY ]] || die "Missing MOK private key: $MOK_KEY. Run: sudo ${FIX_ROOT}/create-mok-key.sh"
    [[ -f $MOK_CERT ]] || die "Missing MOK cert: $MOK_CERT. Run: sudo ${FIX_ROOT}/create-mok-key.sh"
    [[ -x $sign_file ]] || die "Missing kernel sign-file tool: $sign_file"
    "$sign_file" sha256 "$MOK_KEY" "$MOK_CERT" "$PATCHED_MODULE"
  else
    log "Secure Boot is not enabled; leaving module unsigned"
  fi
}

verify_patched_module() {
  log "Verifying patched module metadata"
  modinfo "$PATCHED_MODULE" | sed -n '1,32p'
  modinfo "$PATCHED_MODULE" | grep -q 'dgx_usb_fix:[[:space:]]*apple-05ac-1905-cdc-ncm' \
    || die "Patched module is missing dgx_usb_fix modinfo marker"
  modinfo "$PATCHED_MODULE" | grep -qi 'alias:.*v05ACp1905.*in00' \
    || die "Patched module is missing 05ac:1905 interface 0 alias"
  modinfo "$PATCHED_MODULE" | grep -qi 'alias:.*v05ACp1905.*in02' \
    || die "Patched module is missing 05ac:1905 interface 2 alias"
}

install_module() {
  log "Installing patched module into ${INSTALL_DIR}"
  mkdir -p "$INSTALL_DIR"
  install -m 0644 "$PATCHED_MODULE" "${INSTALL_DIR}/cdc_ncm.ko"
  depmod "$KERNEL_VERSION"

  local selected
  selected="$(modinfo -n cdc_ncm)"
  [[ $selected == "${INSTALL_DIR}/cdc_ncm.ko" ]] \
    || die "depmod did not select installed module. modinfo -n cdc_ncm returned: $selected"
}

reload_module() {
  if ((SKIP_LOAD != 0)); then
    log "Skipping module reload by request"
    return
  fi

  need modprobe
  log "Reloading cdc_ncm from installed updates module"
  modprobe -r cdc_mbim 2>/dev/null || true
  modprobe -r cdc_ncm 2>/dev/null || true
  modprobe cdc_ncm

  log "Loaded module path"
  modinfo -n cdc_ncm
}

resolve_source
log "Detected kernel source: ${SRC_PACKAGE} ${SRC_VERSION} (orig ${ORIG_VERSION})"
download_source
patch_source
build_module
sign_module_if_needed
verify_patched_module
install_module
reload_module

cat <<EOF

Done.

Next:
  1. Unplug and replug the USB-C cable to the Mac.
  2. Run:
       ${FIX_ROOT}/diagnose.sh
       ip -br link
       lsusb -t

Expected:
  - modinfo -n cdc_ncm resolves to ${INSTALL_DIR}/cdc_ncm.ko
  - modinfo cdc_ncm contains v05ACp1905 aliases
  - no new cdc_ncm bind() failure for 05ac:1905 after replug
  - a Linux network interface appears for the Mac USB-C link

Workspace:
  ${WORKDIR}
EOF
