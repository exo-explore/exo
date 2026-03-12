# Thunderbolt Bridge Ops

This repo currently has two incompatible networking models on macOS:

- EXO's `exo` network location, which destroys `bridge0`
- a fixed `Thunderbolt Bridge` model, which keeps `bridge0` and static `10.0.0.x` addresses

For the recurring MacBook Pro <-> Mac mini setup, use the second model.

## Target state

- MacBook Pro:
  - location: `Automatic`
  - bridge: `bridge0`
  - IP: `10.0.0.2/24`
- Mac mini:
  - bridge: `bridge0`
  - IP: `10.0.0.1/24`
- EXO:
  - namespace: `leo-m2`
  - startup command: `EXO_LIBP2P_NAMESPACE=leo-m2 /opt/homebrew/bin/uv run exo -v --no-downloads`

Do not run `tmp/set_rdma_network_config.sh` for this setup. It destroys `bridge0`.

## Scripts

- `scripts/exo-thunderbolt-check`
  - checks local location, `bridge0`, IP, route to peer, EXO API, and optionally cluster visibility
- `scripts/exo-thunderbolt-recover`
  - restores `bridge0`, assigns `10.0.0.2/24`, restarts local EXO, and can restart the peer with a `ssh -tt` flow that matches the known-good recovery path
- `scripts/exo-thunderbolt-watchdog`
  - loop that checks health every 15s and recovers automatically when possible
- `scripts/install-exo-thunderbolt-launchagent`
  - installs the watchdog as a user `launchd` agent

The defaults are MacBook-oriented:

- local IP: `10.0.0.2`
- peer IP: `10.0.0.1`
- peer host: `zealous`
- members: `en1 en2 en3`

Override them with environment variables if needed:

```bash
EXO_LOCAL_BRIDGE_IP=10.0.0.9 EXO_PEER_BRIDGE_IP=10.0.0.10 ./scripts/exo-thunderbolt-check
```

## Manual recovery

Run:

```bash
cd ~/exo
./scripts/exo-thunderbolt-recover --restart-peer
```

That script will:

- switch back to `Automatic` if needed
- re-enable the `bridge0` service if macOS still knows about it
- recreate `bridge0` if missing
- attach `en1 en2 en3`
- assign `10.0.0.2/24`
- restart local EXO
- optionally restart EXO on `zealous`
- wait for cluster convergence and retry the remote restart once if needed

It uses `sudo` for bridge repair.

## Watchdog

The watchdog has two modes:

- if `bridge0` is healthy but the cluster is missing, it restarts local EXO
- if `bridge0` is broken and `sudo -n` works, it runs the full bridge recovery path

To run it directly:

```bash
cd ~/exo
EXO_WATCHDOG_RESTART_PEER=1 ./scripts/exo-thunderbolt-watchdog
```

To install it with `launchd`:

```bash
cd ~/exo
./scripts/install-exo-thunderbolt-launchagent
```

Template plist:

- `tmp/config_examples/io.exo.thunderbolt-watchdog.plist`

## Non-interactive recovery

Automatic bridge repair requires non-interactive `sudo`.

Without that, the watchdog can still:

- detect failure
- restart EXO if the bridge is already healthy
- log that manual bridge recovery is required

If you want fully automatic repair on cable replug, grant tightly-scoped passwordless `sudo` for the commands used by `scripts/exo-thunderbolt-recover`.

## Healthy checks

These should succeed on the MacBook:

```bash
./scripts/exo-thunderbolt-check
./scripts/exo-thunderbolt-check --require-cluster
route -n get 10.0.0.1
curl -sS http://localhost:52415/state | jq '.topology.nodes, .topology.connections'
```

Expected route:

- `10.0.0.1` via `bridge0`

Expected cluster:

- both MacBook and Mac mini node IDs visible in `.topology.nodes`
