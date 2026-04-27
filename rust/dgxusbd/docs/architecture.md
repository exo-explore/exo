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

Current bridge status: the MVP still uses one synchronous loop with bounded drains and USB read/write timeouts. That is acceptable for proof-of-viability, but it is not the right hot-path shape for throughput because one direction can insert idle waits into the other direction.

Target dataplane shape for the next iteration:

- Keep CLI/control setup outside the hot path.
- Move forwarding into a dataplane object with explicit start, stop, and counters.
- Use readiness for TAP, and either readiness or independent workers for USB endpoints. If `nusb` bulk endpoints cannot be polled like file descriptors, prefer one TAP-to-USB worker and one USB-to-TAP worker over one alternating timeout loop.
- Precompute selected pair parameters, endpoint addresses, NTB parse/build configs, max sizes, alignment, and reusable buffers before entering the forwarding path.
- Batch multiple Ethernet frames into each CDC-NCM NTB. Babblerd's dataplane is a good model for runtime structure, but its one-packet-per-datagram forwarding shape should not be copied for CDC-NCM.

## Non-Goals For The First Iteration

- No kernel module patching.
- No persistent udev/systemd service.
- No DHCP server/client integration inside `dgxusbd`.
- No automatic routing.
- No multi-link aggregation.
- No Mac-side automation beyond observation commands.

Configure IP addresses manually or with external tools after the TAP interface exists.
