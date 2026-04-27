#!/usr/bin/env bash
set -Eeuo pipefail

MOK_KEY="${MOK_KEY:-/root/MOK.priv}"
MOK_CERT="${MOK_CERT:-/root/MOK.der}"
MOK_SUBJECT="${MOK_SUBJECT:-/CN=DGX USB Fix Module Signing/}"
FORCE=0

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

usage() {
  cat <<'EOF'
Usage: sudo ./dgx-usb-fix/create-mok-key.sh [--force]

Creates only:
  /root/MOK.priv
  /root/MOK.der

It does not import the key into MOK and does not change Secure Boot trust.

Use the repo dev shell for openssl:

  nix develop .#dgx-usb-fix
  sudo env "PATH=$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/create-mok-key.sh
EOF
}

while (($#)); do
  case "$1" in
    --force)
      FORCE=1
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
need openssl

if ((FORCE == 0)); then
  [[ ! -e $MOK_KEY ]] || die "$MOK_KEY already exists; pass --force to replace it"
  [[ ! -e $MOK_CERT ]] || die "$MOK_CERT already exists; pass --force to replace it"
fi

openssl req \
  -new \
  -x509 \
  -newkey rsa:2048 \
  -keyout "$MOK_KEY" \
  -outform DER \
  -out "$MOK_CERT" \
  -nodes \
  -days 36500 \
  -subj "$MOK_SUBJECT"

chmod 0600 "$MOK_KEY"
chmod 0644 "$MOK_CERT"

cat <<EOF

Created:
  $MOK_KEY
  $MOK_CERT

To enroll this certificate for Secure Boot module loading:
  sudo mokutil --import "$MOK_CERT"
  sudo reboot

Complete the MOK enrollment in the boot UI, then rerun:
  sudo env "PATH=\$PATH" DGX_USB_FIX_NIX_ENV=1 ./dgx-usb-fix/build-and-install.sh

EOF
