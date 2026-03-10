#!/usr/bin/env bash
set -Eeuo pipefail

KVER="$(uname -r)"
WORKDIR="${HOME}/apple1905-cdcncm"
SRCVER="6.14.0-1015.15"
SRCNAME="linux-nvidia-6.14"
BASE_URL="https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/${SRCNAME}/${SRCVER}"

ORIG="${SRCNAME}_6.14.0.orig.tar.gz"
DIFF="${SRCNAME}_${SRCVER}.diff.gz"
DSC="${SRCNAME}_${SRCVER}.dsc"

KBUILD="/lib/modules/${KVER}/build"

msg() { printf '\n==> %s\n' "$*"; }
die() {
  echo "ERROR: $*" >&2
  exit 1
}
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

[[ $EUID -eq 0 ]] || die "Run with sudo"
[[ -d $KBUILD ]] || die "Missing kernel headers/build dir: $KBUILD"

need_cmd curl
need_cmd dpkg-source
need_cmd make
need_cmd python3
need_cmd modprobe
need_cmd insmod
need_cmd modinfo
need_cmd ip
need_cmd dmesg
need_cmd find

msg "Installing build deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  linux-headers-"${KVER}" \
  dpkg-dev \
  ca-certificates \
  curl \
  kmod \
  zstd \
  python3

msg "Preparing workspace"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

msg "Downloading exact source package for ${SRCNAME} ${SRCVER}"
curl -fL "${BASE_URL}/${ORIG}" -o "${ORIG}"
curl -fL "${BASE_URL}/${DIFF}" -o "${DIFF}"
curl -fL "${BASE_URL}/${DSC}" -o "${DSC}"

msg "Extracting source"
dpkg-source -x "${DSC}"

SRCDIR="$(find . -maxdepth 1 -mindepth 1 -type d -name "${SRCNAME}-*" | head -n1)"
[[ -n ${SRCDIR} ]] || die "Could not find extracted source directory"
cd "${SRCDIR}"

[[ -f drivers/net/usb/cdc_ncm.c ]] || die "cdc_ncm.c not found in source tree"

msg "Patching cdc_ncm.c"
python3 <<'PY'
from pathlib import Path
import re

p = Path("drivers/net/usb/cdc_ncm.c")
s = p.read_text()

if "0x05ac, 0x1905, 0" not in s:
    block = (
        '\t/* Apple Mac direct USB-C networking quirk */\n'
        '\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 0),\n'
        '\t  .driver_info = (unsigned long)&apple_private_interface_info,\n'
        '\t},\n'
        '\t{ USB_DEVICE_INTERFACE_NUMBER(0x05ac, 0x1905, 2),\n'
        '\t  .driver_info = (unsigned long)&apple_private_interface_info,\n'
        '\t},\n'
    )

    pat = re.compile(
        r'(\{\s*USB_INTERFACE_INFO\s*\(\s*USB_CLASS_COMM\s*,\s*USB_CDC_SUBCLASS_NCM\s*,\s*USB_CDC_PROTO_NONE\s*\)\s*,\s*\.driver_info\s*=\s*\(unsigned long\)&cdc_ncm_info\s*,\s*\},)',
        re.S,
    )
    s2, n = pat.subn(block + r'\1', s, count=1)
    if n != 1:
        raise SystemExit("Could not find generic CDC NCM class-match entry to patch")
    s = s2

# Patch old bind check if needed.
# Old style:
#   if (!dev->in || !dev->out || !dev->status)
# New behavior for this Mac:
#   allow missing status endpoint for 05ac:1905
old = re.compile(
    r'if\s*\(\s*!dev->in\s*\|\|\s*!dev->out\s*\|\|\s*!dev->status\s*\)',
    re.S,
)
repl = (
    'if (!dev->in || !dev->out || '
    '(!dev->status && '
    '!(le16_to_cpu(dev->udev->descriptor.idVendor) == 0x05ac && '
    'le16_to_cpu(dev->udev->descriptor.idProduct) == 0x1905)))'
)
s, n = old.subn(repl, s, count=1)

# If the tree already has the newer FLAG_LINK_INTR logic, leave it alone.
if n == 0 and "FLAG_LINK_INTR" not in s:
    raise SystemExit("Could not patch bind_common() missing-status check")

# Add a loud marker so we know the patched module actually loaded.
if 'Apple 05ac:1905 quirk test module loaded' not in s:
    m = re.search(r'static\s+int\s+__init\s+cdc_ncm_init\s*\(\s*void\s*\)\s*\{', s)
    if m:
        insert_at = m.end()
        s = s[:insert_at] + '\n\tpr_info("cdc_ncm: Apple 05ac:1905 quirk test module loaded\\n");' + s[insert_at:]

p.write_text(s)
print("Patched", p)
PY

msg "Preparing out-of-tree build dir"
mkdir -p buildmod
cp drivers/net/usb/cdc_ncm.c buildmod/

cat >buildmod/Makefile <<'EOF'
obj-m += cdc_ncm.o
ccflags-y += -Wno-error

all:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
EOF

msg "Building patched module"
cd buildmod
make -C "$KBUILD" M="$PWD" modules
PATCHED_KO="$PWD/cdc_ncm.ko"
[[ -f $PATCHED_KO ]] || die "Patched cdc_ncm.ko was not built"

msg "Patched module info"
modinfo "$PATCHED_KO" | sed -n '1,20p'

msg "Reloading module stack"
modprobe -r cdc_mbim 2>/dev/null || true
modprobe -r cdc_ncm 2>/dev/null || true
insmod "$PATCHED_KO"

msg "Recent dmesg"
dmesg | tail -n 60

cat <<EOF

Done.

Now:
  1. run: sudo dmesg -w
  2. unplug and replug the USB-C cable to the Mac
  3. then run:
       ip -br link
       lsusb | grep 05ac:1905

You want to see:
  - cdc_ncm: Apple 05ac:1905 quirk test module loaded
  - no bind() failure for 05ac:1905
  - a new interface appearing

Workspace:
  $WORKDIR

EOF
