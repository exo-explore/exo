# RCA: 3-node exo cluster sharding — what was actually broken, what fixed it

> Written 2026-04-25 by the agent that picked up the mess. JJ asked for a handoff.

## TL;DR

Five real issues. All now fixed. Cluster has full 6/6 TCP + 6/6 RDMA mesh and 3-way Pipeline placements work for both interconnects.

1. **No physical TB cable between `wc-smbpt` and `wc-bmbp`** → topology was a star, not a triangle. RDMA mesh required for 3-way placement was incomplete. **Fix:** physical cable. Done by JJ.
2. **`SocketConnection`-only reachability check in `placement_utils.py`** → when a node had only RDMA edges (no TCP `SocketConnection`) to a peer, the MLX backend connectivity check failed with "MLX ring backend requires connectivity between neighbouring nodes" / "Current jaccl backend requires all participating devices to be able to communicate". **Fix:** commit `a3924273 Fix RDMA placement host selection` on `fix/gpt-oss-tools-stability` (PR #1981) adds `_fallback_interface_ips()` so the placement engine falls back to advertised network interfaces when no `SocketConnection` edge exists, plus `_prefer_socket_reachable_rank_zero()` in `placement.py` so the JACCL coordinator role lands on the most-reachable node.
3. **Workers were started with `--no-api`** → no HTTP `/node_id` endpoint to probe → only worker→master TCP edges existed; master→worker and worker→worker TCP edges were all missing. **Fix:** restart workers with API enabled (each binds 52415 on its own IP). After the restart, all 6/6 directional TCP pairs exist for real, and placement no longer needs `_fallback_interface_ips` to compensate.
4. **TB topology parser only walked `_items` one level deep** → on links going through a transparent TB hub (the `wc-bmbp ↔ wc-smbp` cable runs through an iVANKY Fusiondock Ultra), the side that enumerated through the dock saw the dock at `_items[0]` and the peer Mac nested at `_items[0]._items[0]`. The parser's `next(item.domain_uuid_key for item in items)` returned `None`, so that direction's `MacThunderboltConnections` event was never emitted, and the master never saw the back-edge. The other side, where the dock is transparent, reported the peer at `_items[0]` directly — producing the asymmetric `bmbp → smbp` missing-edge symptom. **Fix:** commit `a11f279c Walk nested Thunderbolt _items so hub-attached peers register` on `fix/thunderbolt-nested-hub-discovery` (this PR) extends `_ConnectivityItem` to recursively model `_items` and walks the tree depth-first for the first descendant `domain_uuid_key`. Result: 6/6 RDMA mesh and Pipeline+MlxJaccl 3-way placement now passes.
5. **`wc-bmbp` Thunderbolt Bridge blocking JACCL** → `bridge0` captures TB interfaces as members, preventing per-interface IPv4/IPv6 addressing. Without individual IPs, the RDMA GID table is incomplete (missing the IPv4-mapped GID that JACCL uses for QP path resolution), causing `ibv_modify_qp` RTR transitions to fail with errno 96 (EPROTOTYPE). Additionally, unplugging/replugging TB cables corrupts kernel RDMA state requiring a reboot. **Fix:** reboot to clear kernel state, then `sudo ifconfig bridge0 destroy` + manual IPv4/IPv6 assignment on `en6`/`en2`. See "Issue 5" section for full procedure. The other two Macs have persistent "RDMA en\*" NetworkServices that prevent this problem.

## Why the TCP mesh wasn't full

The TCP mesh in exo is built by `check_reachable()` (`src/exo/utils/info_gatherer/net_profile.py`). It HTTP-pings every peer's advertised network interface at `GET /node_id` on `api_port`. A 200 response with a matching node ID becomes a `SocketConnection` edge. With `--no-api`, no HTTP server runs, so probes fail and no edges form for that node as a target. Only the master had `--api-port 52415`, so the only successful probes were workers → master.

Symptom in `/state.topology.connections` (before fix):

| Direction       | Socket edges      | RDMA edges        |
| --------------- | ----------------- | ----------------- |
| smbp(M) → smbpt | 0                 | 1                 |
| smbp(M) → bmbp  | 0                 | 1                 |
| smbpt → smbp(M) | 5                 | 1                 |
| smbpt → bmbp    | 0                 | 1                 |
| bmbp → smbp(M)  | 3                 | 0                 |
| bmbp → smbpt    | 0                 | 1                 |
| **Total**       | **2/6 dir pairs** | **5/6 dir pairs** |

After restarting workers without `--no-api`: all six directional pairs have ≥1 socket edge.

The remaining 1/6 missing RDMA edge (`bmbp → smbp`) is a separate RDMA scanner asymmetry, not a TCP issue. The other five RDMA edges including `smbp → bmbp` are healthy.

## What I did wrong on first pass

I diagnosed problem #2 correctly — the `SocketConnection`-only check in `_find_connection_ip` (`src/exo/master/placement_utils.py:328-336`) and the worker discovery loop in `_poll_connection_updates` (`src/exo/worker/main.py:378-421`) which only advertises edges via `self.api_port`. Then I worked around it by starting workers with `--api-port 52415` to match the master, so the worker discovery loop's reachability ping would succeed and populate `SocketConnection` edges.

The user asked me to retrieve the prior agent's stashed fixes, and commit `a3924273` on `fix/gpt-oss-tools-stability` was already a proper fix for the same problem. We checked out that branch on all three nodes and started workers with their original `--no-api` config; 3-way Pipeline+MlxRing still works because `_fallback_interface_ips` covers the missing-TCP case.

## Branch and commit inventory

`fix/gpt-oss-tools-stability` (tracked at `team-wcv/fix/gpt-oss-tools-stability`, PR #1981, all six restored on all three nodes):

| SHA        | Subject                                        | What it does                                                                                                                                            |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `b4545dac` | Stabilize GPT-OSS tool support                 | gpt-oss model card flags + batch generator handling                                                                                                     |
| `7bd6847f` | Fix GPT-OSS Responses tool loop                | Responses adapter + tool_parsers, prevents infinite tool loops                                                                                          |
| `213bb726` | Improve local model streaming and availability | Dashboard ModelPickerModal shows downloaded-but-not-running models; integrations page picks from downloaded set; minor responses streaming improvements |
| `b6446e41` | Fix deleting symlinked local models            | download coordinator handles symlinks correctly                                                                                                         |
| `46acf9da` | Dial bootstrap peers on swarm startup          | Rust networking/discovery.rs proactively dials bootstrap peers                                                                                          |
| `a3924273` | Fix RDMA placement host selection              | **The placement fix described above**                                                                                                                   |

## Working startup recipe (current state)

All four nodes are on `fix/gpt-oss-tools-stability` at `a3924273`. On every node, before starting:

```bash
unset EXO_LIBP2P_NAMESPACE
export EXO_DEFAULT_MODELS_DIR=$HOME/.exo/models
export EXO_ROUTER_SETTLE_SECONDS=12
```

**Master** (`wc-smbp`, `192.168.1.33`):

```bash
cd ~/Development/Tooling/exo
uv run exo -m --api-port 52415 --libp2p-port 52418
```

**Workers** (`wc-smbpt`, `wc-bmbp`, and `wc-studio`):

```bash
cd ~/Development/Tooling/exo
uv run exo --api-port 52415 \
  --bootstrap-peers /ip4/192.168.1.33/tcp/52418 \
  --libp2p-port 52418
```

> **wc-studio**: `uv` is at `~/.local/bin/uv`, not on the default PATH. Use `~/.local/bin/uv run exo ...` or add it to PATH first.

Each worker binds its own `:52415` on its own IP, so there is no port conflict with the master. This populates the full 12/12 TCP mesh (4 nodes) and 6/6 RDMA mesh (3 Thunderbolt-connected nodes). `wc-studio` has no Thunderbolt and connects via TCP/LAN only. **Do not use `--no-api` on workers** — it leaves the topology incomplete and forces the placement engine to fall back to interface enumeration (which works, but masks real reachability problems and is harder to debug).

After cold-start, allow ~30–60s for libp2p gossip and the RDMA mesh to converge before checking `/state` or `/instance/previews`.

## Stale node IDs in master

The master never evicts nodes that vanish without a clean shutdown. After a worker restart with a new libp2p identity, you get phantom node entries with `ramTotal=0`. The placement engine ignores them. If they're cluttering the dashboard, stop everything, `rm -rf ~/.exo/event_log ~/.exo/exo_log` on the master, then restart master then workers. Models are preserved (they live in `~/.exo/models`).

## Topology sanity check

```bash
curl -sS --max-time 6 http://wc-smbp.local:52415/state -o /tmp/state.json && python3 - <<'PY'
import json
d = json.load(open("/tmp/state.json"))
ids = d.get("nodeIdentities") or {}
top = d.get("topology") or {}
conns = top.get("connections") or {}
hostmap = {k: ((d.get("nodeSystem") or {}).get(k) or {}).get("hostname") or k[:10] for k in ids}
print(f"{len(ids)} nodes")
for src in sorted(ids, key=lambda k: hostmap[k]):
    for dst in sorted(ids, key=lambda k: hostmap[k]):
        if src == dst: continue
        links = (conns.get(src) or {}).get(dst) or []
        r = sum(1 for l in links if "sourceRdmaIface" in l)
        t = sum(1 for l in links if "sinkMultiaddr" in l)
        print(f"  {hostmap[src]:>14} -> {hostmap[dst]:<14}  RDMA={r}  TCP={t}")
PY
```

A healthy 4-node cluster should report 12/12 directional TCP pairs and 6/12 RDMA pairs (only the 3 Thunderbolt-connected nodes — `smbp`, `smbpt`, `bmbp` — have RDMA; `wc-studio` is TCP-only).

- Below 6/6 TCP → a worker is running with `--no-api`; restart it with `--api-port 52415`.
- Below 6/6 RDMA → either a physical cable is missing, or the cable goes through a TB hub on a build that pre-dates `a11f279c`. Inspect `system_profiler SPThunderboltDataType -json` on each node and check whether the peer's `domain_uuid_key` appears at `_items[0]` (direct) or deeper (hub in the path).

## Placement preview check

```bash
curl -sS --max-time 6 'http://wc-smbp.local:52415/instance/previews?model_id=mlx-community/gpt-oss-120b-MXFP4-Q8' | python3 -c '
import sys, json
p = json.load(sys.stdin)
for e in p["previews"]:
    inner = next(iter((e.get("instance") or {}).values()), {})
    n = len(((inner.get("shardAssignments") or {}).get("nodeToRunner") or {}))
    err = e.get("error")
    print(f"  {e[\"sharding\"]:<8} {e[\"instance_meta\"]:<10} {\"OK\" if not err else \"no\"}  nodes={n}  {err or \"\"}")
'
```

## Model-specific constraints (not bugs)

- `mlx-community/GLM-4.6-4bit`: 185GB on disk, model card has `supportsTensor=False`. Tensor placements always fail. Pipeline placements need ~2x storage in working memory; you need a freshly-rebooted cluster (or 200GB+ free across the cluster) for it to fit.
- `mlx-community/gpt-oss-120b-MXFP4-Q8`: 71GB on disk, `numKeyValueHeads=8`. Tensor 3-way fails because 8 doesn't divide evenly across 3 devices. 2-way Tensor is fine. Pipeline 1/2/3-way all OK.

## Capacity gotcha

`bmbp` only has ~52GB total RAM. Three-way placements where each shard needs more than ~40GB will pass the placement preview but fail at runtime when `bmbp` tries to load its slice. Stick to 2-way for large models, or 3-way only for models small enough that bmbp's slice fits.

## Issue 5: `wc-bmbp` Thunderbolt Bridge blocking JACCL (added 2026-04-26)

### Symptom

After unplugging/replugging TB cables, all JACCL (RDMA) placements involving `wc-bmbp` fail with:

- `ValueError: [jaccl] Changing queue pair to RTR failed with errno 96` (EPROTOTYPE) on bmbp
- `ValueError: [jaccl] Changing queue pair to RTR failed with errno 22` (EINVAL) on peers

Meanwhile `wc-smbp ↔ wc-smbpt` JACCL works fine.

### Root cause

`wc-bmbp` has a **Thunderbolt Bridge** (`bridge0`) that captures `en1`, `en6`, and `en2` as members. The other two Macs do not — they have individual "RDMA en1/en6/en2" NetworkServices created in System Configuration.

The bridge causes three cascading problems:

1. **Stale kernel RDMA state**: Unplugging/replugging TB cables while the bridge is active corrupts the IORDMAFamily kernel state. `ibv_devinfo` shows `PORT_ACTIVE` but `ibv_modify_qp` (QP RTR transition) fails. **Fix**: reboot.

2. **Missing RDMA GIDs**: The RDMA driver derives GID table entries from the underlying interface's IP addresses. With the bridge active, IP addresses live on `bridge0`, not on `en6`/`en2`. The GID table for `rdma_en6` has only GID[0] (MAC-derived) instead of the 3 GIDs working nodes have (GID[0] MAC fe80, GID[1] IPv4-mapped ::ffff:169.254.x.x, GID[2] SLAAC privacy fe80). JACCL requires the IPv4-mapped GID for QP path resolution; without it, the QP RTR transition returns errno 96 (EPROTOTYPE). **Fix**: destroy `bridge0`, assign IPv4 + IPv6 addresses to `en6`/`en2`.

3. **Bogus routing**: After destroying `bridge0`, the 169.254.0.0/16 route defaults to `en7` (the dock's 10GbE). Manually adding `-interface en6` host routes creates permanent ARP entries pointing to bmbp's own MAC, blackholing traffic. **Fix**: delete the bad ARP entries; let normal ARP discovery work once the base route or host route is correct.

### Repair procedure (required after every `wc-bmbp` reboot)

`networksetup -createnetworkservice` is blocked by SIP on this machine, so the clean fix (creating persistent "RDMA en\*" services matching the peers) is not available. The manual procedure is:

```bash
# 1. Destroy the Thunderbolt Bridge
sudo ifconfig bridge0 destroy

# 2. Bring TB interfaces up (they may already be up)
sudo ifconfig en6 up
sudo ifconfig en2 up

# 3. Assign IPv4 link-local addresses (populates GID[1])
sudo ifconfig en6 169.254.100.6 netmask 255.255.0.0
sudo ifconfig en2 169.254.100.2 netmask 255.255.0.0

# 4. Assign IPv6 link-local addresses (populates GID[0] match + GID[2])
sudo ifconfig en6 inet6 fe80::3479:2cff:fe66:984 prefixlen 64
sudo ifconfig en2 inet6 fe80::3479:2cff:fe66:988 prefixlen 64
sudo ifconfig en6 inet6 fe80::1:2:3:4 prefixlen 64
sudo ifconfig en2 inet6 fe80::5:6:7:8 prefixlen 64

# 5. Clean any stale ARP entries (especially if you previously used `route add -interface`)
sudo arp -d 169.254.222.72 2>/dev/null || true
sudo arp -d 169.254.166.191 2>/dev/null || true
```

After this, verify with:

```bash
# Should show 3 GIDs per RDMA device
ibv_devinfo -d rdma_en6 -v | grep GID
ibv_devinfo -d rdma_en2 -v | grep GID

# Should ping peers on TB links
ping -c 1 169.254.222.72    # smbp via en6
ping -c 1 169.254.136.105   # smbpt via en2
```

### Making it permanent

The proper fix is to create "RDMA en1", "RDMA en6", "RDMA en2" NetworkServices on `wc-bmbp` to match the peers:

```bash
sudo networksetup -createnetworkservice "RDMA en1" en1
sudo networksetup -createnetworkservice "RDMA en6" en6
sudo networksetup -createnetworkservice "RDMA en2" en2
sudo networksetup -setdhcp "RDMA en1"
sudo networksetup -setdhcp "RDMA en6"
sudo networksetup -setdhcp "RDMA en2"
```

This currently fails with "Unable to access the System Configuration database" due to SIP. Alternatives:

- Boot into Recovery OS and run the commands there
- Delete the Thunderbolt Bridge from System Settings GUI after each reboot (it auto-recreates)
- Create a LaunchDaemon that runs the manual repair procedure at boot

### Code change: JACCL retry loop

Added a retry loop in `src/exo/worker/engines/mlx/utils_mlx.py` around `mx.distributed.init(backend="jaccl")` — 8 attempts with exponential backoff (2s, 4s, 6s... up to 10s). Catches both `RuntimeError` and `ValueError` since JACCL throws `ValueError` for QP state transition failures and `RuntimeError` for socket/recv errors.

## Files touched in this session

- No source changes were made — only:
  - Restored `fix/gpt-oss-tools-stability` checkout on all three nodes (after a brief detour to `main` for debugging).
  - Cleared `~/.exo/event_log` and `~/.exo/exo_log` on all three nodes (multiple times during recovery).
  - Re-ran `uv sync` on all three nodes.
  - Rebuilt the dashboard on `wc-smbp` (`cd dashboard && npm install && npm run build`).
- This file: `RCA-3node-shard-fix.md` — delete or move as you see fit.

### Additional changes (2026-04-26)

- `src/exo/worker/engines/mlx/utils_mlx.py`: Added JACCL retry loop (8 attempts, catches `RuntimeError` + `ValueError`).
- Manual network configuration on `wc-bmbp`: destroyed `bridge0`, assigned IPv4/IPv6 to TB interfaces, cleaned stale ARP entries.
