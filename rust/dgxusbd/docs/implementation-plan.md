# Proposed Implementation Plan

Last updated: 2026-04-27.

This plan tracks implementation iterations. Each iteration should be small enough to commit, push, pull to the Spark, and test independently.

## Numbering Note

The first docs split probe, codec, TAP, bridge, and scheduler work into Iterations 1-5. During live review, the user-facing numbering collapsed the scheduler and cleanup work into Iteration 3b. From here, "Iteration 4" means the next active user-facing iteration: dataplane runtime and throughput correctness.

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

## Iteration 5: Scheduler, Validation, And Throughput Baseline

Purpose: turn the proof-of-packet-movement bridge into a measurable bridge that cannot starve one direction under sustained traffic.

Completed changes:

- Add per-loop frame or byte budgets for TAP-to-USB and USB-to-TAP work.
- Make bridge mode fail on required NCM setup failures, especially selected NTB format, NTB input size, and max datagram size.
- Fix NTB builder sizing so it uses checked length accumulation and returns `BuiltNtbTooLarge` before allocating oversized output.
- Add repeatable iperf3 TCP and UDP test commands to the lab workflow and capture a baseline in `status.md`.
- Refactor repeated CLI options with `clap(flatten)` structs.
- Introduce small domain newtypes where they reduce primitive misuse, starting with USB IDs if parsing grows.

Implementation notes:

- In this conversation, the CLI cleanup was called Iteration 3a and the scheduler/validation/baseline pass was called Iteration 3b. In this document's original numbering, those map to Iteration 5.
- Simple budgets were enough to avoid the original unlimited TAP drain, but the loop is still timeout-driven: after a full TAP budget it can wait for USB IN traffic even during one-way Spark-to-Mac traffic. That likely contributes to the asymmetric iperf baseline.

Acceptance result:

- Under sustained one-way TAP traffic, the opposite USB-to-TAP direction still gets scheduled.
- Bridge exits early when a required NCM setup step fails.
- TCP and UDP iperf3 tests can be run in both directions with documented commands.
- Measured throughput is no longer treated as an unknown side effect of the smoke test.

## Iteration 4: Dataplane Runtime And Throughput Correctness

Purpose: replace the timeout-shaped MVP loop with a real forwarding dataplane before spending more time on operational polish.

Status: implemented, with a remaining Spark-to-Mac transmit blocker.

Completed changes:

- First, fix the review's small correctness issues:
  - make `--max-events` exact by passing remaining event budgets into the drain functions
  - rename or alias `--usb-timeout-ms` as the USB write timeout and update help text
  - add NTB builder tests for multi-frame alignment padding and aggregate-size rejection
- Moved forwarding out of the CLI call path into a dataplane object with explicit start, stop, and counters, similar in spirit to `rust/babblerd/src/dataplane.rs`.
- Replaced timeout fairness with readiness plus real concurrency:
  - register TAP readiness with `mio` where possible
  - use independent TAP-to-USB and USB-to-TAP workers because `nusb` endpoints are not file-descriptor readiness sources in this code
- Keep hot-path state precomputed before forwarding starts:
  - selected NCM pair, endpoint addresses, max packet sizes
  - NTB parse/build configs, max NTB sizes, datagram alignment, max datagram count
  - reusable TAP and NTB buffers
  - counters and stop/cancellation channels
- Batch multiple TAP Ethernet frames into each CDC-NCM NTB up to negotiated size/count limits. Do not copy babblerd's one-packet-per-datagram shape; for CDC-NCM, batching is the main throughput lever.
- Keep multiple USB bulk-IN transfers queued.
- Keep multiple USB bulk-OUT transfers queued.
- Rerun the TCP/UDP iperf baseline over IPv6 link-local addresses and compare with the Iteration 3b numbers in `status.md`.

Acceptance result:

- One-way Spark-to-Mac traffic is no longer forced to wait on idle USB-IN polling after every TAP budget.
- `--max-events` counts submitted/received bridge events exactly in counters.
- Bridge counters remain accurate under the concurrent forwarding shape.
- Mac-to-Spark throughput improved materially, reaching multi-gigabit over IPv6 link-local.
- Spark-to-Mac data traffic remains the active blocker. It is not fixed by the scheduler, TAP multi-queue, Linux TAP offload settings, or USB OUT queueing. See `status.md`.

## Active Iteration 5: Apple CDC-NCM Transmit Correctness

Purpose: make Spark-generated CDC-NCM bulk-OUT traffic match what Apple's `05ac:1905` function expects well enough for stable TCP/UDP from Spark to Mac.

Proposed changes:

- Compare `dgxusbd` NTB16 transmit layout against Linux `cdc_ncm` with Apple quirks, especially:
  - NDP-at-end versus NDP-near-front placement
  - short packet or zero-length packet behavior after NTBs whose length is a multiple of the endpoint packet size
  - sequence-number expectations
  - `wNdpOutDivisor`, `wNdpOutPayloadRemainder`, and `wNdpOutAlignment` interpretation
  - whether macOS expects one NTB per USB transfer or tolerates multiple queued OUT transfers
- Add optional debug capture for transmitted NTBs and parsed received NTBs so hardware observations can be compared byte-for-byte.
- Add targeted transmit-shape flags for lab testing before committing to one layout.
- Keep the main lab tests on IPv6 link-local addresses. IPv4 static addressing is deferred until final validation.

Acceptance:

- Spark-to-Mac link-local ping stays healthy after sustained Spark-originated TCP/UDP.
- Spark-to-Mac TCP reaches a stable, materially higher baseline and iperf receives final results cleanly.
- Spark-to-Mac paced UDP has low loss at a documented bitrate.

## Later Iteration: Operational Robustness

Purpose: make the bridge useful enough for repeated lab testing once the hot dataplane shape is sound.

Proposed changes:

- Reconnect handling.
- Select pair by CLI flag.
- Optional pair 2/3 support.
- Better error messages for permission, detach/claim, endpoint stall, malformed NTBs.
- Optional pcap/debug dump for NTBs or Ethernet frames.
- Optional IP assignment helper for the lab static-IP flow.

Acceptance:

- Cable replug does not require killing stuck processes.
- Logs are enough to diagnose failure without attaching a debugger.
