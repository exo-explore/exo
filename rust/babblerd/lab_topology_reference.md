# Lab Topology Reference

This file records the current, still-relevant lab topology and bring-up
context for `babblerd`.

## Hosts

- The lab consists of four Mac minis.
- SSH targets:
  - `e4@e4`
  - `e2@e2`
  - `e11@e11`
  - `e16@e16`
- You can SSH into these machines directly to inspect or run commands.

## Physical Topology

- The machines are connected in a Thunderbolt ring:
  - `e4 -> e2 -> e11 -> e16 -> e4`
- The Thunderbolt-facing interface names are not fixed to `en2` and `en3`.
  macOS can expose additional Thunderbolt links as other `en*` interfaces such
  as `en5`, `en6`, or host-specific names after reconfiguration.
- Treat `en2,en3` as an old bring-up heuristic only. For normal lab testing,
  run without `BABBLER_INTERFACE_ALLOWLIST` and let `babblerd`/Babel discover
  the live interfaces.

## Repository Location On The Macs

- Each machine has a checkout of the Exo repository at:
  - `~/babeld-exo`
- That checkout is expected to already be on the correct branch for this work.

## Current Node Addresses

- `e2`: `fde0:20c6:1fa7:ffff:aeb:e53a:cb17:aa42`
- `e11`: `fde0:20c6:1fa7:ffff:34a:26dd:46ff:1a3f`
- `e16`: `fde0:20c6:1fa7:ffff:7c5d:5e2d:54df:e665`
- `e4`: `fde0:20c6:1fa7:ffff:cc78:aec2:d64e:f125`

## Current Route Policy

`babblerd` now uses the forked `babeld` `neighbour-cost` command to bias route
selection after neighbours appear:

- `en0` and `en1` get maximum finite cost with `bias-256 16776704` and
  `coef-256 0`, yielding cost `65534`.
- Most other `enN` interfaces get `bias-256 N * 100 * 256` and `coef-256 0`,
  so `en2` costs `200`, `en3` costs `300`, `en5` costs `500`, and so on.
- This is a temporary Mac lab heuristic, not measured link scoring.
- After convergence it keeps steady-state routes away from `en0`/`en1` and
  toward lower-numbered direct Thunderbolt-style links. Immediately after
  restart, Babel can still transiently install worse high-cost routes until
  better neighbour state arrives.

## Running `babblerd`

From `~/babeld-exo`, pull first so the machine is not testing stale commits,
then start `babblerd`:

```sh
cd ~/babeld-exo
git pull
RUST_LOG=info sudo -E nix run .#babblerd --impure
```

The default dataplane transport is UDP. To force the experimental TCP
neighbour transport for Mac Thunderbolt throughput tests, start every node with
one of:

```sh
BABBLER_ROUTER_TRANSPORT=tcp RUST_LOG=info sudo -E nix run .#babblerd --impure
RUST_LOG=info sudo -E nix run .#babblerd --impure -- --force-tcp
```

TCP mode still uses Babel for route selection and still sends Babel packets on
the direct link-local `en*` interfaces. Only node-ULA overlay traffic uses the
TCP streams.

On macOS, forced-TCP listener sockets are wildcard listeners rather than
per-interface-bound listeners. Per-interface `IPV6_BOUND_IF` on TCP listeners
left e4/e16 handshakes stuck in `SYN_RCVD` during lab testing, while a plain
link-local TCP listener on the same cable completed. Outbound TCP streams remain
scoped to the Babel-selected interface. Accepted TCP streams are rejected unless
the peer is link-local and matches a live Babel neighbour on the accepted
scope/interface.

Forced TCP defaults the TUN MTU to `65535` to reduce per-packet TUN syscalls on
the Mac Thunderbolt fast path. Override it during sweeps with either command
form, but keep the value identical on every forced-TCP peer in the run:

```sh
BABBLER_TUN_MTU=16384 BABBLER_ROUTER_TRANSPORT=tcp RUST_LOG=info sudo -E nix run .#babblerd --impure
RUST_LOG=info sudo -E nix run .#babblerd --impure -- --force-tcp --tun-mtu 32768
```

TCP mode uses `256 KiB` TCP read buffers and `256 KiB` opportunistic write batch
targets by default. Sweep write targets and TCP socket buffers dynamically with
the same values on every node:

```sh
BABBLER_TCP_BATCH_TARGET_BYTES=1048576 BABBLER_TCP_SOCKET_BUFFER_BYTES=16777216 BABBLER_ROUTER_TRANSPORT=tcp RUST_LOG=info sudo -E nix run .#babblerd --impure
```

Good first matrix values are `262144`, `524288`, `1048576`, and `2097152` for
`BABBLER_TCP_BATCH_TARGET_BYTES`, plus `4194304`, `8388608`, `16777216`, and
`33554432` for `BABBLER_TCP_SOCKET_BUFFER_BYTES`. TCP receive reads directly
into the frame decoder buffer, and stream readiness is reregistered only when
write interest changes. UDP mode still defaults to `1452`, derived from a
`1500` byte physical MTU minus outer IPv6 and UDP headers.

If broad interface discovery causes unrelated links to interfere with a
specific debug run, `BABBLER_INTERFACE_ALLOWLIST` is still available as a
temporary escape hatch. Do not use it as the default lab topology description.

`babblerd` no longer needs to wait for an initial interface before starting
`babeld`. The forked `babeld` can start with no managed interfaces, and
`babblerd` adds interfaces later through the read-write local control socket.

## `iperf3`

Use the flake-provided forked `iperf3` when testing this branch:

```sh
nix run .#iperf3 -- -s
nix run .#iperf3 -- -c <addr>
```

The current fork lives at
`/home/royalguard/Desktop/exo-all/networking-related/iperf3`; commit `962e05b`
adds `%scopeID` rendering for link-local IPv6 output.

Latest useful test commands:

```sh
nix run .#iperf3 -- -s -1
nix run .#iperf3 -- -6 -b 100M -t 5 -c <node-ula>
nix run .#iperf3 -- -6 -u -b 100M -t 5 -c <node-ula>
nix run .#iperf3 -- -6 -u -b 0 -t 10 -c <node-ula>
nix run .#iperf3 -- -6 -b 0 -t 10 -c <node-ula>
```

For the forced-TCP dataplane experiment, capture both correctness and
performance:

1. Start all four nodes with TCP mode enabled.
2. Wait for convergence and record route/FIB state.
3. Run a directed `ping6` matrix over node ULAs.
4. Run adjacent TCP and UDP iperf smoke tests at `100M`.
5. Run direct overlay TCP/UDP `-b 0` on an adjacent pair such as `e4 -> e16`.
6. Run single-hop overlay TCP/UDP `-b 0` such as `e2 -> e16`.
7. Capture dataplane counter deltas, especially TCP batches/frames/errors, and
   verify bidirectional `ping6` still works after each full-rate run.

Latest findings:

- Direct physical UDP without the software router is about `11 Gbit/s`.
- Direct overlay `e4 -> e16`, UDP `-b 0`: about `1.46 Gbit/s` received with
  negligible loss.
- Direct overlay `e4 -> e16`, TCP `-b 0`: about `1.24 Gbit/s` received.
- Single-hop overlay `e2 -> e16`, UDP `-b 100M`: `100 Mbit/s` with no loss.
- Single-hop overlay `e2 -> e16`, UDP `-b 0`: earlier server intervals were
  around `1.11-1.16 Gbit/s` with `12-14%` loss. A later run sent about
  `1.32 Gbit/s`; dataplane counters showed roughly `1.16M` packets delivered
  at `e16`, but the `iperf3` control connection broke before a receiver
  summary was produced. Treat that as an overload/control-path failure, not as
  a zero-throughput receiver result. The overlay path then needed a
  `babblerd` restart to recover.
- Single-hop overlay `e2 -> e16`, TCP `-b 0`: about `1.07-1.08 Gbit/s`
  received, with route/path recovery sometimes lagging briefly after the run.

For `1452` byte packets, `11 Gbit/s` is roughly a one-microsecond packet
budget: about `947 kpps`, or `1.06 us/packet`. The current direct overlay
result is roughly `126 kpps`, or `8 us/packet`. Forced TCP now defaults the TUN
MTU to `65535`, so compare packet counters before attributing any result to the
outer TCP socket alone.

Babel protocol packets should not traverse the software router. `babblerd`
starts `babeld` without startup interfaces and later adds only eligible
physical `en*` interfaces through the local control socket; the TUN/overlay
interface is not added to Babel. What does traverse the overlay is traffic
addressed to node ULAs, including `iperf3` data, the `iperf3` TCP control
connection, and `ping6` to a peer ULA. Full overlay load can still indirectly
perturb Babel by consuming shared NIC queues, kernel buffers, and CPU time on
the same physical interfaces, but it is not because Babel's link-local packets
are being encapsulated by `babblerd`.

## Important Current Note

- With the current codebase, `babblerd` has an internal dummy keepalive client.
- That means you do **not** need to connect an external client socket just to
  make the daemon stay active during testing.
