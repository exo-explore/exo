# DGX USB-C Mac CDC-NCM Kernel Fix

This directory contains the route A workaround for direct USB-C networking from
DGX Spark / GX10 Linux to an Apple Mac exposing USB device `05ac:1905`.

The problem is small: the Mac exposes CDC-NCM functions whose control
interfaces have no interrupt/status endpoint. Linux's generic `cdc_ncm` match
therefore binds and then fails. The workaround builds a replacement
`cdc_ncm.ko` for the running kernel with explicit `05ac:1905` matches mapped to
Linux's existing Apple private CDC-NCM driver info.

## Files

- `build-and-install.sh`: download matching Ubuntu kernel source, patch, build,
  optionally sign, install, and optionally reload `cdc_ncm`.
- `diagnose.sh`: read-only Spark diagnostics before or after installing.
- `create-mok-key.sh`: explicitly create `/root/MOK.priv` and `/root/MOK.der`
  for local module signing. It does not import the key or change firmware trust.
- `configure-link-local.sh`: set Apple `05ac:1905` CDC-NCM NetworkManager
  profiles to IPv6 link-local-only so `fe80::` addresses persist without DHCP.
- `scripts/patch_cdc_ncm.py`: source-aware patcher for `drivers/net/usb/cdc_ncm.c`.
- `patches/apple-05ac-1905-cdc-ncm.patch`: human-readable reference patch for
  Linux 6.17-style trees.
- `templates/Makefile`: external-module build template.

## Recommended Flow On Spark

From the repo root on Spark:

```sh
nix develop .#dgx-usb-fix
dgx-usb-fix-diagnose
dgx-usb-fix-install --skip-load
sudo modprobe -r cdc_mbim cdc_ncm
sudo modprobe cdc_ncm
dgx-usb-fix-configure-link-local
dgx-usb-fix-diagnose
```

Then unplug and replug the USB-C cable to the Mac, and check:

```sh
lsusb -t
ip -br link
ip -6 addr
journalctl -k --no-pager | grep -E 'cdc_ncm|05ac|1905|bind'
```

If Secure Boot is enabled, `build-and-install.sh` signs the module with
`/root/MOK.priv` and `/root/MOK.der` by default. If those files do not exist,
the build stops with instructions. Creating and enrolling a MOK is a boot-trust
change, so it is intentionally not hidden inside the build script.

To create the key files only:

```sh
nix develop .#dgx-usb-fix
dgx-usb-fix-create-mok-key
```

To trust the generated cert, import it and complete the reboot-time enrollment:

```sh
sudo mokutil --import /root/MOK.der
sudo reboot
```

After enrollment, rerun `dgx-usb-fix-install`.

## Notes

The installed module is written to:

```text
/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko
```

After `depmod`, `modinfo -n cdc_ncm` must resolve to that path. The patched
module also carries:

```text
dgx_usb_fix: apple-05ac-1905-cdc-ncm
```

and should expose `v05ACp1905` aliases for interfaces `00` and `02`.
