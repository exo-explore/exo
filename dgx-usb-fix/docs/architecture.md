# Architecture Direction

Last updated: 2026-04-27.

## Goal

Patch and load Linux's `cdc_ncm` host driver so the operating system owns the
Mac USB-C network link as a normal kernel netdev:

```text
Apple Mac USB CDC-NCM function
  <-> Linux cdc_ncm / usbnet
  <-> Linux network interface
  <-> Linux IP stack
```

This is route A. It avoids maintaining a userspace USB/TAP data plane and should
inherit Linux's existing CDC-NCM batching, short-packet, ZLP, and control-request
behavior.

## File Layout

```text
dgx-usb-fix/
  README.md
  build-and-install.sh
  configure-link-local.sh
  create-mok-key.sh
  diagnose.sh
  parts.nix
  docs/
  patches/
    apple-05ac-1905-cdc-ncm.patch
  scripts/
    patch_cdc_ncm.py
  templates/
    Makefile
```

## Responsibilities

`parts.nix`:

- Defines the Nix dev shell `.#dgx-usb-fix`.
- Supplies compilers, kernel-module build tooling, `dpkg-source`, `kmod`,
  `mokutil`, `openssl`, `python3`, and diagnostics tools.

`build-and-install.sh`:

- Orchestrates the route-A flow.
- Does not install packages with apt.
- Expects the repo Nix shell or equivalent host tools.
- Keeps source download, patching, build, signing, install, and module reload in
  separate shell functions.

`scripts/patch_cdc_ncm.py`:

- Applies source-aware edits to `drivers/net/usb/cdc_ncm.c`.
- Adds `05ac:1905` explicit device matches for control interfaces `0` and `2`.
- Uses `apple_private_interface_info`.
- Adds a stable `MODULE_INFO` marker for verification.
- Contains a fallback for older trees with a direct `!dev->status` check.

`patches/apple-05ac-1905-cdc-ncm.patch`:

- Human-readable reference patch for Linux 6.17-style source.
- Not the primary application mechanism; the Python patcher is more tolerant of
  small source layout differences.

`create-mok-key.sh`:

- Creates signing key files only.
- Does not run `mokutil --import`.
- Does not enroll a key or modify firmware/boot trust by itself.

`diagnose.sh`:

- Read-only state collection for route A.

`configure-link-local.sh`:

- Finds Apple `05ac:1905` interfaces driven by `cdc_ncm`.
- Configures their NetworkManager profiles for IPv6 link-local-only activation.
- Avoids NetworkManager tearing down `fe80::` addresses after DHCP/RA failure.

## Install Location

The patched module is installed to:

```text
/lib/modules/$(uname -r)/updates/dgx-usb-fix/cdc_ncm.ko
```

This location is intentionally outside the stock module path. After `depmod`,
`modinfo -n cdc_ncm` should resolve to the `updates/dgx-usb-fix` module.

## Trust Boundary

With Secure Boot enabled, the kernel rejects unsigned or untrusted modules. The
tooling signs with `/root/MOK.priv` and `/root/MOK.der`, but the user must
explicitly import and enroll the public cert with `mokutil` and the MOK Manager
pre-boot UI.
