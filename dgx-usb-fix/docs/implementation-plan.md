# Implementation Plan

Last updated: 2026-04-27.

Each route-A iteration should be small enough to commit, push, pull to Spark,
and test independently.

## Iteration 1: Split Tooling

Status: complete.

Changes:

- Moved route-A implementation into `dgx-usb-fix/`.
- Replaced monolithic shell scripts with:
  - top-level orchestration shell script
  - Python source patcher
  - reference `.patch`
  - Makefile template
  - diagnostics script
  - explicit MOK key creation script
- Converted old `tmp/spark/spark-build-apple-cdc-ncm*.sh` scripts into wrappers.

Validation:

- Shell syntax checks passed.
- Python compile check passed.
- Patcher tested against upstream Linux 6.17 `cdc_ncm.c`.
- Reference patch tested with `patch --dry-run` against upstream Linux 6.17.

## Iteration 2: Nix Build Environment

Status: complete.

Changes:

- Added `dgx-usb-fix/parts.nix`.
- Imported it from the root flake.
- Exposed dev shell `nix develop .#dgx-usb-fix`.
- Removed apt-based dependency installation from `build-and-install.sh`.

Acceptance:

- `nix develop .#dgx-usb-fix` provides compiler/build dependencies and route-A
  utilities.
- `build-and-install.sh` no longer mutates host package state.
- Root commands preserve the dev-shell `PATH` through flake-provided wrappers.

## Iteration 3: MOK Enrollment And Signed Build

Status: complete on Spark.

Purpose: establish a trusted local module-signing key, then build and load the
patched module.

Steps:

- Create `/root/MOK.priv` and `/root/MOK.der`.
- Import `/root/MOK.der` with `mokutil --import`.
- Reboot and enroll the cert in MOK Manager.
- Confirm `mokutil --test-key /root/MOK.der`.
- Run `dgx-usb-fix-install`.

Acceptance:

- Signed `cdc_ncm.ko` loads under Secure Boot.
- `modinfo -n cdc_ncm` points at `updates/dgx-usb-fix/cdc_ncm.ko`.
- `modinfo cdc_ncm` shows `dgx_usb_fix` marker and `v05ACp1905` aliases.

Observed:

- `modinfo -n cdc_ncm` resolves to
  `/lib/modules/6.17.0-1014-nvidia/updates/dgx-usb-fix/cdc_ncm.ko`.
- Module marker: `dgx_usb_fix: apple-05ac-1905-cdc-ncm`.
- Signer: `DGX USB Fix Module Signing`.

## Iteration 4: Bind And Netdev Validation

Status: complete on current Spark/Mac link.

Purpose: prove route A fixes the original kernel bind failure.

Steps:

- Unplug/replug the USB-C cable after loading the patched module.
- Watch kernel logs.
- Confirm no new `bind() failure` for `05ac:1905`.
- Confirm a Linux netdev appears.
- Record the interface name and addresses.

Acceptance:

- Interfaces 0/1 and/or 2/3 bind to `cdc_ncm` instead of remaining unbound.
- Linux creates a network interface for the Mac USB-C link.

Observed:

- Spark created `enxd2dcb4ccf72d` for USB interface `00`.
- Spark created `enxd2dcb4ccf70d` for USB interface `02`.
- Mac pairing observed:
  - `enxd2dcb4ccf72d` <-> `en5`
  - `enxd2dcb4ccf70d` <-> `anpi0`
- `lsusb -t` showed interfaces 0, 1, 2, and 3 using `Driver=cdc_ncm`.
- Fresh route-A binding no longer shows the old `bind() failure` for the
  current replugged device.

## Iteration 5: Link-Local NetworkManager Profiles

Status: complete on Spark.

Purpose: make Spark behave like macOS for direct links: while the cable is
connected, each USB network interface keeps an IPv6 link-local address even
without DHCP, RA, ULA, or global address assignment.

Steps:

- Run `dgx-usb-fix-configure-link-local`.
- Confirm the Apple `05ac:1905` `cdc_ncm` profiles are set to IPv6 link-local.
- Confirm `ip -6 addr` retains `fe80::` addresses after NetworkManager's normal
  DHCP timeout window.

Acceptance:

- NetworkManager no longer withdraws `fe80::` addresses after DHCP/RA failure.
- `ping -6 -I <spark-usb-iface> ff02::1` remains usable.

Observed current pairs:

- `spark@fe80::5786:d998:b0b6:9041%enxd2dcb4ccf72d`
  <-> `mac@fe80::8a2:83dc:50cd:d9a%en5`
- `spark@fe80::dc3d:13fd:b127:97ac%enxd2dcb4ccf70d`
  <-> `mac@fe80::d0dc:b4ff:fecc:f7f2%anpi0`

## Iteration 6: Connectivity And Throughput

Status: pending.

Purpose: compare in-kernel route A against route B.

Initial tests:

- IPv6 link-local ping both directions.
- Optional static IPv4 ping.
- TCP and UDP `iperf3` both directions.

Acceptance:

- Bidirectional ping is stable.
- Spark-to-Mac TCP/UDP does not reproduce the route-B userspace transmit-format
  failure.
- Throughput is recorded in `status.md`.

## Later: Upstreamable Patch

Status: pending.

Purpose: turn the lab quirk into a clean patch suitable for NVIDIA/Ubuntu or
upstream Linux.

Steps:

- Reduce local marker/debug additions.
- Keep only the `05ac:1905` device IDs mapped to `apple_private_interface_info`.
- Write a concise commit message with observed descriptors and bind failure.
- Include `lsusb -v` descriptor evidence.
