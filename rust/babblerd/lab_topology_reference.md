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
- On each machine, the Thunderbolt-facing interfaces are:
  - `en2`
  - `en3`

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

## Important Current Note

- With the current codebase, `babblerd` has an internal dummy keepalive client.
- That means you do **not** need to connect an external client socket just to
  make the daemon stay active during testing.
