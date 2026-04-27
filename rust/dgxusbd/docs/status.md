# Current Status

Last updated: 2026-04-27.

## Repositories

Local repo:

- Path: `/home/royalguard/Desktop/exo-all/networking-related/exo-dgx-mac-usb-fix`
- Branch: `andrei/dgx-mac-usb-fix`
- HEAD before docs were added: `189d88ea`
- Initial worktree state: clean

Spark repo:

- Host: `jensen@gx10-a174`
- Path: `~/Desktop/exo`
- Branch: `andrei/dgx-mac-usb-fix`
- HEAD when checked: `189d88ea`

## Rust Stub

Current files before this documentation pass:

- `rust/dgxusbd/Cargo.toml`
- `rust/dgxusbd/src/main.rs`
- `rust/dgxusbd/src/lib.rs`

Current binary behavior: initializes `color-eyre` and tracing, then exits successfully. Current library contains only a placeholder `foo()`.

`dgxusbd` is already registered as a workspace member and flake app/package.

## Build Checks Run

Local:

```sh
nix develop -c cargo check -p dgxusbd
```

Result: success.

Spark:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix develop -c cargo check -p dgxusbd"
```

Result: success.

Spark flake app smoke run:

```sh
ssh jensen@gx10-a174 "cd ~/Desktop/exo && nix run .#dgxusbd"
```

Result: success; no output, as expected for the current no-op binary.

## Hardware Findings

Spark:

- Hostname: `gx10-a174`
- Kernel: `Linux 6.17.0-1014-nvidia aarch64`
- Secure Boot: enabled
- No `/sys/class/udc` was present, so Spark does not currently look able to act as a Linux USB gadget device.
- No `/sys/bus/thunderbolt/devices` was present, and no Thunderbolt bus was observed on Spark.
- `enP7s7` is Realtek PCI Ethernet via `r8127`, not the USB-C Mac link.

Mac:

- SSH target: `e2@e2`
- OS when checked: Darwin 25.4.0
- macOS reported no Thunderbolt device connected on its Thunderbolt/USB4 buses during the investigation.

USB-C link:

- Spark sees the Mac as `05ac:1905 Apple, Inc. Mac`.
- `lsusb -t` showed the device on a 10000M USB path with all four interfaces unbound after `cdc_ncm` bind failure.
- Kernel log showed:

```text
cdc_ncm 4-1:1.0: bind() failure
cdc_ncm 4-1:1.2: bind() failure
```

The kernel-driver workaround script in `tmp/spark` patches `cdc_ncm` to add a device-specific `05ac:1905` quirk. That remains the cleaner long-term fix if shipped by NVIDIA/Ubuntu, but `dgxusbd` is exploring a no-kernel-module userspace path.

