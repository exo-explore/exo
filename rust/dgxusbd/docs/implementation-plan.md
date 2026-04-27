# Proposed Implementation Plan

Last updated: 2026-04-27.

This is a plan for review. Do not start code changes until the plan is accepted or revised.

## Iteration 1: Probe And Dry-Run

Purpose: confirm a Rust USB library can see and claim the Apple NCM interfaces without involving TAP or packet forwarding.

Proposed changes:

- Add USB crate dependency, likely `nusb`.
- Add CLI with modes:
  - `list`: print candidate USB devices and interfaces
  - `probe`: open `05ac:1905`, inspect descriptors, print detected NCM pairs/endpoints
- Keep binary thin and call library logic.
- No TAP device, no claims unless `probe --claim` or equivalent is explicit.

Acceptance:

- `nix develop -c cargo check -p dgxusbd` passes locally and on Spark.
- `nix run .#dgxusbd -- list` runs on Spark.
- `sudo -E nix run .#dgxusbd -- probe --claim` can claim the unbound interfaces or reports a precise permission/kernel-driver error.

## Iteration 2: NCM Codec Unit Tests

Purpose: implement CDC-NCM framing without needing the hardware loop to be correct yet.

Proposed changes:

- Add `ncm` module.
- Define NTH16/NDP16/DPE16 layouts with `zerocopy`.
- Implement RX parser for NTB16 without CRC.
- Implement TX builder for one or more Ethernet frames per NTB.
- Add synthetic unit tests:
  - single-frame RX
  - multi-frame RX
  - malformed header rejection
  - TX round-trip through RX parser
  - alignment and bounds checks

Acceptance:

- Unit tests pass locally and on Spark.
- Codec is independent of USB/TAP runtime.

## Iteration 3: TAP Skeleton

Purpose: create and exercise the Linux TAP path independently.

Proposed changes:

- Add TAP module around `tun-rs`.
- Add CLI mode to create TAP and optionally echo/log frames.
- Do not yet connect USB to TAP.

Acceptance:

- On Spark, root invocation creates a TAP interface.
- The process can read/write frames without panics.

## Iteration 4: One-Pair Bridge MVP

Purpose: move Ethernet frames between one Apple NCM pair and one TAP interface.

Proposed changes:

- Hardcode or default to pair:
  - control interface 0
  - data interface 1
  - IN endpoint `0x81`
  - OUT endpoint `0x01`
- Run two forwarding tasks.
- Add counters for USB reads/writes, TAP reads/writes, decoded frames, malformed NTBs.
- Treat USB presence as carrier; no interrupt/status endpoint is expected.

Acceptance:

- Spark creates TAP interface.
- Mac-generated traffic appears as TAP RX counters.
- Spark-generated traffic is sent to USB OUT endpoint without NCM formatting errors.
- Manual static IP ping is attempted if basic frame movement is visible.

## Iteration 5: Robustness

Purpose: make it useful enough for repeated lab testing.

Proposed changes:

- Reconnect handling.
- Select pair by CLI flag.
- Optional pair 2/3 support.
- Better error messages for permission, detach/claim, endpoint stall, malformed NTBs.
- Optional pcap/debug dump for NTBs or Ethernet frames.

Acceptance:

- Cable replug does not require killing stuck processes.
- Logs are enough to diagnose failure without attaching a debugger.

