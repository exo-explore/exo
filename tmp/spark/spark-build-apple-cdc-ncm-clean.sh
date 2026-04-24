#!/usr/bin/env bash
# Patch the cdc_ncm kernel module to recognize Apple USB-C networking (05ac:1905).

set -Eeuo pipefail

# ---- Config ---------------------------------------------------------------

KERNEL_VERSION="$(uname -r)"
WORKDIR="${HOME}/apple1905-cdcncm"

# Ubuntu source package matching the running kernel.
SRC_PACKAGE="linux-nvidia-6.14"
SRC_VERSION="6.14.0-1015.15"
SRC_URL="https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/${SRC_PACKAGE}/${SRC_VERSION}"

KERNEL_BUILD_DIR="/lib/modules/${KERNEL_VERSION}/build"

# ---- Helpers --------------------------------------------------------------

log() { printf '\n==> %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

# ---- Preflight ------------------------------------------------------------

[[ $EUID -eq 0 ]]          || die "Run with sudo"
[[ -d $KERNEL_BUILD_DIR ]] || die "Missing kernel headers: $KERNEL_BUILD_DIR"

for cmd in curl dpkg-source make python3 modprobe insmod modinfo ip dmesg find; do
  require "$cmd"
done

# ---- Install build dependencies -------------------------------------------

log "Installing build dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  linux-headers-"${KERNEL_VERSION}" \
  dpkg-dev \
  ca-certificates \
  curl \
  kmod \
  zstd \
  python3

# ---- Download and extract kernel source -----------------------------------

log "Preparing workspace at $WORKDIR"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

log "Downloading $SRC_PACKAGE $SRC_VERSION source"
curl -fL "${SRC_URL}/${SRC_PACKAGE}_6.14.0.orig.tar.gz"     -o orig.tar.gz
curl -fL "${SRC_URL}/${SRC_PACKAGE}_${SRC_VERSION}.diff.gz" -o debian.diff.gz
curl -fL "${SRC_URL}/${SRC_PACKAGE}_${SRC_VERSION}.dsc"     -o source.dsc

log "Extracting source"
dpkg-source -x source.dsc
SRCDIR="$(find . -maxdepth 1 -mindepth 1 -type d -name "${SRC_PACKAGE}-*" | head -n1)"
[[ -n $SRCDIR ]] || die "Could not find extracted source directory"
cd "$SRCDIR"

[[ -f drivers/net/usb/cdc_ncm.c ]] || die "cdc_ncm.c not found in source tree"

# ---- Patch cdc_ncm.c ------------------------------------------------------
#
# Three edits:
#   1. Add explicit device match for Apple 05ac:1905 (interfaces 0 and 2).
#   2. Allow a missing status endpoint specifically for that device.
#   3. Log a marker line on module init so we can confirm the patched module
#      is what actually loaded.

log "Patching cdc_ncm.c"
python3 <<'PY'
import re
from pathlib import Path

path = Path("drivers/net/usb/cdc_ncm.c")
source = path.read_text()

# 1. Add Apple device-specific match entries, just above the generic
#    USB_CLASS_COMM / CDC_SUBCLASS_NCM class-match entry.
if "0x05ac, 0x1905, 0" not in source:
    apple_entries = (
        '\t/* Apple Mac direct USB-C networking quirk */\n'
        '\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),\n'
        '\t  .driver_info = (unsigned long)&apple_private_interface_info,\n'
        '\t},\n'
        '\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),\n'
        '\t  .driver_info = (unsigned long)&apple_private_interface_info,\n'
        '\t},\n'
    )
    class_match = re.compile(
        r'(\{\s*USB_INTERFACE_INFO\s*\(\s*USB_CLASS_COMM\s*,\s*'
        r'USB_CDC_SUBCLASS_NCM\s*,\s*USB_CDC_PROTO_NONE\s*\)\s*,\s*'
        r'\.driver_info\s*=\s*\(unsigned long\)&cdc_ncm_info\s*,\s*\},)',
        re.S,
    )
    source, n = class_match.subn(apple_entries + r'\1', source, count=1)
    if n != 1:
        raise SystemExit("Could not find generic CDC NCM class-match entry")

# 2. Relax the "must have status endpoint" check in bind_common() for
#    Apple 05ac:1905. Newer trees use FLAG_LINK_INTR instead; leave those alone.
old_check = re.compile(
    r'if\s*\(\s*!dev->in\s*\|\|\s*!dev->out\s*\|\|\s*!dev->status\s*\)',
    re.S,
)
new_check = (
    'if (!dev->in || !dev->out || '
    '(!dev->status && '
    '!(le16_to_cpu(dev->udev->descriptor.idVendor)  == 0x05ac && '
    '  le16_to_cpu(dev->udev->descriptor.idProduct) == 0x1905)))'
)
source, n = old_check.subn(new_check, source, count=1)
if n == 0 and "FLAG_LINK_INTR" not in source:
    raise SystemExit("Could not patch bind_common() missing-status check")

# 3. Add a marker pr_info on module init, so dmesg confirms the patched
#    module is loaded (and not the stock one).
marker = 'cdc_ncm: Apple 05ac:1905 quirk test module loaded'
if marker not in source:
    init_fn = re.search(
        r'static\s+int\s+__init\s+cdc_ncm_init\s*\(\s*void\s*\)\s*\{',
        source,
    )
    if init_fn:
        i = init_fn.end()
        source = source[:i] + f'\n\tpr_info("{marker}\\n");' + source[i:]

path.write_text(source)
print("Patched", path)
PY

# ---- Build the patched module out-of-tree ---------------------------------

log "Building patched module"
BUILD_DIR="buildmod"
mkdir -p "$BUILD_DIR"
cp drivers/net/usb/cdc_ncm.c "$BUILD_DIR/"

# Makefile recipe lines must start with a real TAB. Using printf with explicit
# \t keeps this robust against editor auto-indentation.
{
  printf 'obj-m += cdc_ncm.o\n'
  printf 'ccflags-y += -Wno-error\n'
  printf '\n'
  printf 'all:\n'
  printf '\t$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules\n'
  printf '\n'
  printf 'clean:\n'
  printf '\t$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean\n'
} >"$BUILD_DIR/Makefile"

make -C "$KERNEL_BUILD_DIR" M="$PWD/$BUILD_DIR" modules

PATCHED_MODULE="$PWD/$BUILD_DIR/cdc_ncm.ko"
[[ -f $PATCHED_MODULE ]] || die "Build failed: cdc_ncm.ko not produced"

log "Module info (first 20 lines)"
modinfo "$PATCHED_MODULE" | sed -n '1,20p'

# ---- Swap in the patched module -------------------------------------------

log "Unloading stock cdc_ncm and loading patched version"
modprobe -r cdc_mbim 2>/dev/null || true   # depends on cdc_ncm
modprobe -r cdc_ncm  2>/dev/null || true
insmod "$PATCHED_MODULE"

log "Recent dmesg"
dmesg | tail -n 60

cat <<EOF

Done. Next steps:

  1. In another terminal, start watching the kernel log:
       sudo dmesg -w

  2. Unplug and replug the USB-C cable to the Mac.

  3. Check that the interface appeared:
       ip -br link
       lsusb | grep 05ac:1905

  You should see:
    - "cdc_ncm: Apple 05ac:1905 quirk test module loaded" in dmesg
    - no bind() failure for 05ac:1905
    - a new network interface in \`ip -br link\`

  Workspace: $WORKDIR
EOF
