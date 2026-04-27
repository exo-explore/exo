# Proposed Implementation Plan

Last updated: 2026-04-27.

This plan tracks implementation iterations. Each iteration should be small enough to commit, push, pull to the Spark, and test independently.

## Iteration 1: Probe And Dry-Run

Purpose: confirm a Rust USB library can see and claim the Apple NCM interfaces without involving TAP or packet forwarding.

Status: complete in `df1fa775`.

Implemented changes:

- Added USB crate dependency `nusb`.
- Add CLI with modes:
  - `list`: print candidate USB devices and interfaces
  - `probe`: open `05ac:1905`, inspect descriptors, print detected NCM pairs/endpoints
- Keep binary thin and call library logic.
- No TAP device, no claims unless `probe --claim` or equivalent is explicit.

Acceptance result:

- `nix develop -c cargo check -p dgxusbd` passes locally and on Spark.
- `nix run .#dgxusbd -- list` runs on Spark and reports the Mac as `05ac:1905`.
- `sudo -E nix run .#dgxusbd -- probe --claim` claimed interfaces 0, 1, 2, and 3.

Important result:

- The observed descriptors confirm the expected Apple-specific shape: valid CDC-NCM function and data interfaces, but no interrupt/status endpoint on the control interfaces.

## Iteration 2: NCM Codec Unit Tests

Purpose: implement CDC-NCM framing without needing the hardware loop to be correct yet.

Status: complete in `ed1cfe49` and adjusted in `d2a87e6b`.

Implemented changes:

- Add `ncm` module.
- Define NTH16/NDP16/DPE16 layouts with `zerocopy`.
- Implement RX parser for NTB16 without CRC.
- Implement TX builder for one or more Ethernet frames per NTB.
- Add synthetic unit tests:
  - single-frame RX
  - multi-frame RX
  - malformed header rejection
  - TX round-trip through RX parser
  - bounds checks and conservative malformed-input rejection

Acceptance result:

- Unit tests pass locally and on Spark.
- Codec is independent of USB/TAP runtime.

Important correction:

- RX parsing originally rejected Mac NTBs whose DPE datagram index was 30 because the code required 4-byte alignment.
- Live hardware showed those NTBs are valid enough for macOS traffic; the RX parser now avoids enforcing datagram-index alignment and continues to enforce header, bounds, terminator, and minimum-frame checks.

## Iteration 3: TAP Skeleton

Purpose: create and exercise the Linux TAP path independently.

Status: complete in `ed1cfe49`.

Implemented changes:

- Add TAP module around `tun-rs`.
- Add `tap-smoke` CLI mode to create a TAP interface.

Acceptance result:

- On Spark, root invocation created `dgxusb0` with MTU 1500.
- The TAP disappears when the process exits.

## Iteration 4: One-Pair Bridge MVP

Purpose: move Ethernet frames between one Apple NCM pair and one TAP interface.

Status: complete in `ed1cfe49` and adjusted in `d2a87e6b`.

Implemented changes:

- Default to pair:
  - control interface 0
  - data interface 1
  - IN endpoint `0x81`
  - OUT endpoint `0x01`
- Run a conservative single-loop forwarder:
  - drain TAP frames to USB OUT as NTB16
  - poll USB IN, parse NTB16, and write Ethernet frames to TAP
- Add counters for USB reads/writes, TAP reads/writes, decoded frames, malformed NTBs.
- Treat USB presence as carrier; no interrupt/status endpoint is expected.

Acceptance result:

- Spark creates TAP interface.
- Mac-generated traffic appears as decoded USB RX frames and TAP TX counters.
- Spark-generated traffic is sent to USB OUT endpoint without NCM formatting errors.
- Manual static IP ping succeeded:
  - Mac `en5`: temporary `192.168.254.1/30`
  - Spark `dgxusb0`: temporary `192.168.254.2/30`
  - Result: 3/3 ICMP replies, 0% loss.

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
