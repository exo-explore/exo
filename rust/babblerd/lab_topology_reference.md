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
nix run .#iperf3 -- -6 -u -b 100M -t 5 -c <node-ula>
nix run .#iperf3 -- -6 -u -b 0 -t 10 -c <node-ula>
nix run .#iperf3 -- -6 -b 0 -t 10 -c <node-ula>
```

Latest findings:

- Direct physical UDP without the software router is about `11 Gbit/s`.
- Direct overlay `e4 -> e16`, UDP `-b 0`: about `1.46 Gbit/s` received with
  negligible loss.
- Direct overlay `e4 -> e16`, TCP `-b 0`: about `1.24 Gbit/s` received.
- Single-hop overlay `e2 -> e16`, UDP `-b 100M`: `100 Mbit/s` with no loss.
- Single-hop overlay `e2 -> e16`, UDP `-b 0`: server intervals around
  `1.11-1.16 Gbit/s`, but with `12-14%` loss and a post-test path wedge until
  `babblerd` was restarted.
- Single-hop overlay `e2 -> e16`, TCP `-b 0`: about `1.07 Gbit/s` received.

For MTU-sized traffic, `11 Gbit/s` is roughly a one-microsecond packet budget:
with the current `1452` byte TUN MTU it is about `947 kpps`, or
`1.06 us/packet`. The current direct overlay result is roughly `126 kpps`, or
`8 us/packet`. That means the next performance work should focus on dataplane
packet cost, syscalls/copies, batching or aggregation, jumbo MTU opportunities,
and overload recovery rather than assuming the remaining gap is Babel route
selection.

## Important Current Note

- With the current codebase, `babblerd` has an internal dummy keepalive client.
- That means you do **not** need to connect an external client socket just to
  make the daemon stay active during testing.
