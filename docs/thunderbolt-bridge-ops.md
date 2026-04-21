# Thunderbolt Bridge Ops

This repo currently has two incompatible networking models on macOS:

- EXO's `exo` network location, which destroys `bridge0`
- a fixed `Thunderbolt Bridge` model, which keeps `bridge0` and static `10.0.0.x` addresses

For the recurring bridge-preserving two-node setup, use the second model.

## Target state

Apr 11 2026 verified pair:

- Gandalf (Mac mini M4):
  - location: `Automatic`
  - bridge: `bridge0`
  - IP: `10.0.0.2/24`
  - exo launchd label: `io.exo.zealous-peer`
- Zealous (Mac Studio M4 Max):
  - bridge: `bridge0`
  - IP: `10.0.0.1/24`
  - exo launchd label: `com.avantgaera.exo-thunderbolt`
  - API port: `52416`
  - libp2p port: `52415`
- EXO:
  - namespace: `zealous-cluster-apr11`
  - bootstrap preference on each node: Thunderbolt `/ip4/10.0.0.x/tcp/52415` first, Tailscale fallback second

Current verified note: the Thunderbolt datapath is healthy and exo topology shows the peer connection over `rdma_en2`, and after the final 2026-04-11 recovery-mode enable step both hosts now report RDMA enabled in live exo state. Zealous is confirmed locally with `rdma_ctl status = enabled`, and Zealous `/state` shows `nodeRdmaCtl.enabled = true` plus `nodeThunderboltBridge.enabled = true` for both Zealous and Gandalf. On 2026-04-11 after the Gandalf update, Zealous briefly showed a stale single-node topology view until `launchctl kickstart -k gui/$(id -u)/com.avantgaera.exo-thunderbolt` was run locally; after that, both nodes reappeared correctly in `/state`.

Additional Apr 11 note: a later Saruman bootstrap experiment briefly left the live Zealous process with a stale third bootstrap peer in its environment. The safe fix was a conservative unload, process stop, and reload of `com.avantgaera.exo-thunderbolt` using the clean launchd plist. After that restart, Zealous rejoined Gandalf normally, `/state` again showed the two-node RDMA topology over `rdma_en2`, and the production bootstrap set was confirmed back to Gandalf only (Thunderbolt first, Tailscale second).

Later on Apr 11, after RDMA was enabled on Saruman, the safer join pattern was used: Saruman alone was updated to dial into Zealous via its own launchd config, while Zealous's launchd plist was left unchanged. This successfully expanded the live Zealous topology view to three RDMA-enabled nodes. Saruman's Thunderbolt Bridge was then corrected to static `10.0.0.3/24` to avoid accidental link-local bridge routing. Observed shape at verification time: Zealous saw Saruman present and RDMA-capable, while Saruman's own local `/state` had not yet reflected the full 3-node view, so treat this as a live but still-settling 3-node cluster.

Important caveat: the EXO dashboard still raises a Thunderbolt Bridge cycle warning for the Zealous-Gandalf pair, but live checks show the pair remains healthy, routable over `bridge0`, and RDMA-connected. Do not disable Thunderbolt Bridge on Zealous or Gandalf in response to this warning unless a separate live connectivity failure is verified. At this point the warning is best treated as a detector/UI bug for the healthy 2-node Thunderbolt core.

Root cause and prepared fix: `src/exo/shared/topology.py` originally reported Thunderbolt bridge cycles for any directed cycle with length >=2, which incorrectly included a healthy bidirectional 2-node RDMA pair. A local fix was prepared to require cycle length >=3 for Thunderbolt bridge loop reporting, and regression coverage was added in `src/exo/shared/tests/test_thunderbolt_cycles.py` (`2 passed`). That fix correctly removed the 2-node false positive.

Deeper finding after deployment to both Zealous and Saruman: the remaining warning is a false 3-node cycle, not a real bridge storm. The current detector still treats any socket edge whose sink IP belongs to the Thunderbolt Bridge subnet (`10.0.0.x`) as a Thunderbolt edge. In the 3-node setup, that causes indirect routed reachability over the shared bridge subnet to be mistaken for direct Thunderbolt adjacency, producing a false triangle. This is now a pure exo detection-logic bug, not a live networking failure.

Current live state: the 3-node cluster is visible again from both Zealous and Saruman, RDMA remains enabled, and Thunderbolt Bridge should not be disabled on any node in response to this warning. The next fix should refine detector semantics so only direct Thunderbolt peer links contribute to bridge-cycle detection.

Update after code inspection and focused tests: the minimal safe backend fix is narrower than first assumed. In `src/exo/shared/topology.py`, Thunderbolt Bridge cycle detection should only use RDMA-backed topology edges as evidence of direct Thunderbolt adjacency. Generic socket edges, even when their sink multiaddr is on the Thunderbolt subnet (`10.0.0.x`), are not sufficient evidence of a direct Thunderbolt link and must be excluded from cycle construction. A focused regression test now covers this exact false-positive case in `src/exo/shared/tests/test_thunderbolt_cycles.py`.

Important caveat: if the live process already has a stale `thunderboltBridgeCycles` value in state, code fixes alone do not clear that field until a recomputation path is triggered by a new relevant topology or node-info event, or the process is minimally restarted and state is rebuilt. On Zealous, direct recomputation against the current live topology returns `[]` with the patched logic even while `/state` still reports the stale old cycle. That means detector logic is now correct in code, but live warning disappearance still depends on conservative runtime refresh.

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

## Logs + rotation

- Watchdog logs: `$HOME/Library/Logs/exo-thunderbolt/watchdog.log`
- EXO run logs: `$HOME/Library/Logs/exo-thunderbolt/exo-run.log`
- Script wrapper logs: `/tmp/exo-thunderbolt-*.log`
- The watchdog auto-rotates these logs when they exceed ~5MB, keeping the last ~4000 lines.
- Tunables: `EXO_LOG_MAX_BYTES`, `EXO_LOG_TAIL_LINES`

## Non-interactive recovery

Automatic bridge repair requires non-interactive `sudo`.

Without that, the watchdog can still:

- detect failure
- restart EXO if the bridge is already healthy
- log that manual bridge recovery is required

If you want fully automatic repair on cable replug, grant tightly-scoped passwordless `sudo` for the commands used by `scripts/exo-thunderbolt-recover`.

## Healthy checks

Current live checks for the Apr 11 Zealous + Gandalf pair:

On Studio:

```bash
ping -c 3 10.0.0.2
route -n get 10.0.0.2
curl -sS http://localhost:52416/state | jq '.topology.nodes, .topology.connections, .nodeRdmaCtl, .nodeThunderboltBridge'
```

On Gandalf:

```bash
ping -c 3 10.0.0.1
route -n get 10.0.0.1
```

Expected route:

- peer `10.0.0.x` via `bridge0`

Expected cluster:

- both Studio and Gandalf node IDs visible in `.topology.nodes`
- `.topology.connections` populated with the peer edge over `rdma_en2`
- `.nodeThunderboltBridge.*.enabled == true`
- `.nodeRdmaCtl.*.enabled == true`

Important:

- do not rely on `scripts/exo-thunderbolt-check` defaults without overrides on this pair, because the helper script defaults are still MacBook-oriented and assume local IP `10.0.0.2` plus API port `52415`
- exact macOS parity on both hosts is now confirmed for the Apr 11 pair: Zealous and Gandalf are both on `26.4.1` (`25E253`)
- `rdma_ctl enable` must still be done from macOS Recovery on each machine when onboarding new nodes, but for the current Apr 11 Zealous + Gandalf pair that step is now complete and verified in live cluster state
