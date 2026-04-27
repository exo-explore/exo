# Architecture Direction

Last updated: 2026-04-27.

## Goal

Implement a userspace bridge:

```text
Apple Mac USB CDC-NCM function
  <-> userspace USB bulk/control handling
  <-> CDC-NCM NTB encoder/decoder
  <-> Linux TAP device
  <-> Linux IP stack
```

This avoids loading or patching Spark's `cdc_ncm` kernel module. It should not require BIOS changes, MOK enrollment, or custom kernel modules.

The motivation is not that userspace is cleaner than the kernel fix. The kernel fix is the smallest and cleanest technical solution, but it currently runs into Secure Boot/module-signing constraints on the lab Spark. See [motivation.md](motivation.md).

## Binary Boundary

Keep `src/main.rs` thin:

- parse CLI
- initialize diagnostics
- call a library-level `run(config)` entry point
- map top-level errors to process exit

Most logic should live in library modules.

## Proposed Library Modules

Potential module layout:

```text
src/
  lib.rs
  cli.rs
  config.rs
  error.rs
  usb/
    mod.rs
    discovery.rs
    descriptors.rs
    transfers.rs
  ncm/
    mod.rs
    constants.rs
    ntb.rs
    rx.rs
    tx.rs
  tap.rs
  bridge.rs
  stats.rs
```

The exact names can change, but keep these responsibilities separate.

## Preferred Crates

Already present and useful:

- `tokio`: async runtime and task management
- `tun-rs`: TAP device creation and I/O
- `zerocopy`: fixed-layout CDC-NCM headers
- `thiserror` / `color-eyre`: errors
- `clap`: CLI
- `tracing`: runtime diagnostics
- `nix` / `libc`: Linux-specific calls when `tun-rs` does not cover something
- `macaddr`: MAC address values

Likely missing dependency to add during implementation:

- Preferred: `nusb`, for a pure-Rust userspace USB driver API.
- Fallback: `rusb`, if `nusb` lacks a needed operation or behaves poorly on this hardware.

Need to verify that the chosen USB crate supports:

- opening device by VID/PID
- reading descriptors/string descriptors
- claiming multiple interfaces
- setting data interface alternate setting
- control transfers
- bulk IN/OUT transfers

## Runtime Shape

Minimal runtime:

1. Discover `05ac:1905`.
2. Select one NCM pair, initially control interface 0 and data interface 1.
3. Claim interfaces.
4. Set data interface altsetting 1.
5. Create TAP interface, initially `dgxusb0` or configurable.
6. Start two loops:
   - USB bulk IN -> NCM RX decode -> TAP write
   - TAP read -> NCM TX encode -> USB bulk OUT
7. Log counters and errors.

MVP can use one pair only. Pair 2/3 can be added after the first pair is proven.

Current bridge status: forwarding now lives in `src/dataplane.rs` rather than the CLI call stack. It uses separate TAP-to-USB and USB-to-TAP OS workers, TAP readiness through `mio`, queued USB bulk-IN reads, queued USB bulk-OUT writes, and multi-frame CDC-NCM TX batching. This is much closer to the desired hot-path shape than the original single-loop MVP.

Current dataplane constraints:

- Mac-to-Spark traffic is healthy and reaches multi-gigabit rates over IPv6 link-local.
- Spark-to-Mac TCP/UDP data traffic is still unstable and low-throughput even after scheduler, TAP multi-queue, offload, and USB OUT queueing experiments.
- Next work should focus on Apple CDC-NCM transmit-format compatibility rather than generic runtime shape.
- Keep IPv6 link-local addressing as the primary test path. IPv4 static addressing should wait until final validation.

## Non-Goals For The First Iteration

- No kernel module patching.
- No persistent udev/systemd service.
- No DHCP server/client integration inside `dgxusbd`.
- No automatic routing.
- No multi-link aggregation.
- No Mac-side automation beyond observation commands.

Configure IP addresses manually or with external tools after the TAP interface exists.
