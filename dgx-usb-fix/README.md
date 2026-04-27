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
- `scripts/patch_cdc_ncm.py`: source-aware patcher for `drivers/net/usb/cdc_ncm.c`.
- `patches/apple-05ac-1905-cdc-ncm.patch`: human-readable reference patch for
  Linux 6.17-style trees.
- `templates/Makefile`: external-module build template.

## Recommended Flow On Spark

From the repo root on Spark:

```sh
./dgx-usb-fix/diagnose.sh
sudo ./dgx-usb-fix/build-and-install.sh --skip-load
sudo modprobe -r cdc_mbim cdc_ncm
sudo modprobe cdc_ncm
./dgx-usb-fix/diagnose.sh
```

Then unplug and replug the USB-C cable to the Mac, and check:

```sh
lsusb -t
ip -br link
journalctl -k --no-pager | grep -E 'cdc_ncm|05ac|1905|bind'
```

If Secure Boot is enabled, `build-and-install.sh` signs the module with
`/root/MOK.priv` and `/root/MOK.der` by default. If those files do not exist,
the build stops with instructions. Creating and enrolling a MOK is a boot-trust
change, so it is intentionally not hidden inside the build script.

To create the key files only:

```sh
sudo ./dgx-usb-fix/create-mok-key.sh
```

To trust the generated cert, import it and complete the reboot-time enrollment:

```sh
sudo mokutil --import /root/MOK.der
sudo reboot
```

After enrollment, rerun `build-and-install.sh`.

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
