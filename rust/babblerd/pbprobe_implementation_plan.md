# PBProbe Implementation Plan

This file tracks the staged implementation and validation of a paper-faithful
PBProbe profiler for link-local lab links.

## Stage 1: Local Implementation

- Add `src/profiling/pbprobe/` as a separate module from the simple packet
  train profiler.
- Implement the paper protocol:
  - START initiates one direction.
  - RTS requests each sample.
  - the sender replies with a packet bulk of length `k`, meaning `k + 1`
    packets.
  - the receiver measures first and last packet arrival time, delay sum, and
    dispersion.
  - END reports the selected sample and estimate.
- Implement Algorithm 1:
  - start with `k = 1`.
  - if measured minimum dispersion is below `D_thresh`, multiply `k` by 10 and
    restart.
  - otherwise pace samples with `G = 2D / U`.
  - stop after fixed `n` accepted samples.
- Keep the C implementation as a reference, but use the paper's units for `G`.

## Stage 2: Local Verification

- Unit-test packet encoding/decoding.
- Unit-test estimator selection by minimum delay sum.
- Unit-test bulk-length adaptation and pacing calculations.
- Compile the standalone example.

## Stage 3: Lab Validation

- Discover the current link-local addresses and interface names on the Mac mini
  ring via SSH.
- Build or run the PBProbe example on the relevant remotes.
- Run `iperf3` over the same link-local scoped addresses as the baseline.
- Compare PBProbe estimates against `iperf3` with a reasonable tolerance.
- If estimates are outside tolerance, adjust only algorithm parameters or
  implementation bugs, not the scoring target.

## Current Notes

- The repo license is Apache-2.0. The dropped PBProbe source has a permissive
  MIT-like license header with notice retention and academic citation language.
- The C code appears to implement the core estimator, but its `G` sleep units
  look inconsistent with the paper. This implementation should follow the paper.
