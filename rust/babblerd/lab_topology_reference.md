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

## Important Current Note

- With the current codebase, `babblerd` has an internal dummy keepalive client.
- That means you do **not** need to connect an external client socket just to
  make the daemon stay active during testing.
