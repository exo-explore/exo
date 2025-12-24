# Fix: mDNS Query Interval + Static Peers Workaround for Cross-Platform Discovery

## Motivation

**Problem**: mDNS peer discovery between Linux and macOS nodes fails, preventing heterogeneous clusters from forming a complete P2P mesh.

**Root Causes**:
1. **mDNS query interval too long**: 1500 seconds (25 minutes) between queries
2. **Cross-platform mDNS multicast blocking**: macOS and Linux mDNS implementations don't discover each other
3. **No fallback discovery mechanism**: Once mDNS fails, there's no way to manually connect peers

This prevents users from deploying heterogeneous Exo clusters across Linux + macOS, which is one of Exo's key value propositions.

**Related Issue**: N/A (new discovery)

## Changes

### 1. Reduce mDNS Query Interval (✅ **FIXES Linux ↔ Linux**)

**File**: `rust/networking/src/discovery.rs`
**Line 34**: Changed `MDNS_QUERY_INTERVAL` from 1500s → 5s

```rust
// BEFORE:
const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500); // 25 minutes

// AFTER:
const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(5); // 5 seconds
```

**Impact**: Linux nodes now discover each other in <1 second instead of up to 25 minutes

### 2. Static Peers Workaround (⚠️ **PROOF-OF-CONCEPT, NOT PRODUCTION**)

**File**: `rust/networking/src/discovery.rs`
**Lines 27-34, 133-146**: Added static peer bootstrap to bypass mDNS

```rust
// Static peers constant (hardcoded for testing)
const STATIC_PEERS: &[(&str, &str)] = &[
    ("12D3KooWLC7a3t6givH4e7VtQ8WpmadfnpxvBSaVKxKa8Y3kCAuf", "/ip4/192.168.0.160/tcp/42183"),
    ("12D3KooWCw583XUFccb2RfCd55GpoSd8EFJ1NYwq7yNcjAj1swqv", "/ip4/192.168.0.106/tcp/38651"),
    ("12D3KooWB698CkfuX3CB9MZyySrirrHamvePF66Kz5xwbGZu4xsv", "/ip4/192.168.0.134/tcp/60092"),
];

// Dial static peers on startup in Behaviour::new()
```

**Impact**: Bypasses mDNS to force peer connections
**Limitation**: **Ports are dynamically assigned** by libp2p, so hardcoded addresses break on restart

### 3. Documentation

Created comprehensive analysis and workaround documentation:
- `STATIC_PEERS_FIX.md` - Full technical analysis, 3 proposed fix options
- `GITHUB_ISSUE_MDNS_CROSSPLATFORM.md` - GitHub issue template with reproduction steps
- `bootstrap_peers.py` - Diagnostic tool to generate peer multiaddrs
- `static_peers.json` - Example static peer configuration file

## Why It Works

### Fix #1: mDNS Query Interval Reduction ✅

**Why 1500s was too long**:
- Nodes broadcast mDNS announcements every 1500s (25 minutes)
- If a node joins the cluster, other nodes won't discover it for up to 25 minutes
- Users would think the cluster is broken when it's just waiting for the next query

**Why 5s works**:
- Matches typical mDNS query intervals in other libp2p implementations
- Fast enough for practical cluster formation (<10 seconds)
- Still conservative enough to avoid network spam (200 queries/hour)

**Test Results**:
- Before: Linux ↔ Linux discovery took 10-25 minutes (if it worked at all)
- After: Linux ↔ Linux discovery completes in <1 second consistently

### Fix #2: Static Peers **Partially** Works ⚠️

**Why it works for Linux ↔ Linux**:
- Bypasses mDNS multicast entirely
- Directly dials known peer addresses on startup
- Localhost ↔ .106 connection established successfully in <3 seconds

**Why it FAILS for production**:
- **Dynamic ports**: libp2p assigns random ports on each restart (e.g., 42183 → 34209)
- **Hardcoded addresses**: Static peers can't adapt to port changes
- **No persistence**: Exo doesn't save/load actual peer addresses
- **Result**: Works once, breaks on restart

**Why cross-platform still fails**:
- M2 Mac uses different port on each restart (60092, previously 55987)
- Linux nodes try to connect to old hardcoded port
- ConnectionRefused errors occur
- **Root cause persists**: mDNS cross-platform multicast is still blocked

## Test Plan

### Manual Testing

**Hardware**:
- **Node 1**: Linux x86_64, 4x NVIDIA V100-PCIE-16GB, 192.168.0.160
- **Node 2**: Linux x86_64, NVIDIA RTX 5070 12GB, 192.168.0.106
- **Node 3**: macOS 15.6 (Sequoia), Apple M2 ARM64, 24GB RAM, 192.168.0.134

**Test 1: mDNS Interval Fix (Linux ↔ Linux)**

What we did:
1. Applied mDNS 5s interval fix
2. Rebuilt Rust networking library on both Linux nodes
3. Started both nodes simultaneously
4. Monitored logs for ConnectionEstablished events

Results:
```
✅ localhost ↔ .106: ConnectionEstablished in 607ms
✅ Gossipsub topics synchronized (global_events, commands, etc.)
✅ Persistent connection maintained for 30+ minutes
✅ Master election worked correctly
```

**Test 2: Static Peers Workaround**

What we did:
1. Added STATIC_PEERS constant with all 3 node multiaddrs
2. Modified Behaviour::new() to dial static peers on startup
3. Rebuilt on all 3 nodes
4. Restarted all nodes

Results:
```
✅ localhost ↔ .106: ConnectionEstablished (static peers)
⚠️ .106 → .134 M2 Mac: ConnectionRefused (port changed 55987 → 60092)
⚠️ .134 → Linux nodes: HostUnreachable (old ports in static config)
❌ Full 3-node mesh: NOT achieved due to dynamic ports
```

**Test 3: Port Discovery Issue**

What we observed:
```
Restart #1:
  localhost: tcp/42183
  .106: tcp/38651
  .134: tcp/55987

Restart #2:
  localhost: tcp/37828
  .106: tcp/34209
  .134: tcp/60092
```

**Conclusion**: Static peers work as proof-of-concept but require either:
- Fixed port configuration in Exo
- Dynamic peer discovery file that updates with actual multiaddrs
- Proper cross-platform mDNS fix (long-term solution)

### Automated Testing

**Changes to automated tests**: None (this is a networking/discovery fix, not application logic)

**How existing tests cover this**:
- libp2p's internal tests validate peer discovery mechanisms
- Exo's existing connection tests validate ConnectionEstablished events
- No new test infrastructure needed for this workaround

**Recommended future tests**:
1. Integration test: Multi-node cluster formation timing
2. Cross-platform test: Linux + macOS peer discovery
3. Resilience test: Node restart with changing ports

## Production Readiness

### ✅ Ready for Production: mDNS Interval Fix

**Impact**: Positive (faster discovery, no breaking changes)
**Risk**: Low (just changes timing, doesn't affect protocol)
**Recommendation**: **Merge this fix**

### ❌ NOT Ready for Production: Static Peers Workaround

**Impact**: Breaks on restart (dynamic ports)
**Risk**: High (hardcoded addresses don't persist)
**Recommendation**: **DO NOT merge** - use as proof-of-concept only

### Recommended Path Forward

**Short-term (for users now)**:
1. Merge mDNS interval fix (improves Linux ↔ Linux discovery)
2. Document static peers as manual workaround in README
3. Provide helper script to generate current peer multiaddrs

**Long-term (proper fix)**:
1. Add configuration file support (`static_peers.json`)
2. Implement peer discovery file that updates with actual ports
3. Add manual peer add API endpoint (`POST /api/peers/add`)
4. Investigate enabling IPv6 for better cross-platform mDNS
5. Consider alternative discovery mechanisms (DHT bootstrap, HTTP endpoint list)

## Compatibility

**Breaking changes**: None
**Backward compatible**: Yes (existing mDNS discovery still works, just faster)
**Platform support**:
- ✅ Linux x86_64: Fully working
- ✅ Linux ARM64: Should work (not tested)
- ⚠️ macOS ARM64: mDNS cross-platform still fails
- ⚠️ macOS x86_64: mDNS cross-platform still fails

## Documentation Updates Needed

If this PR is merged:

1. **CHANGELOG.md**: Add entry for mDNS interval fix
2. **README.md**: Add troubleshooting section for cross-platform discovery
3. **PLATFORMS.md**: Update Linux status from "Tier 1 Planned" to "Tier 1 Supported"
4. **docs/CLUSTERING.md**: Document static peer workaround for advanced users
5. **GitHub Issues**: Create issue for cross-platform mDNS discovery

---

**Tested by**: Sophia Elya & Scott @ Elyan Labs
**Date**: December 24, 2025
**Branch**: `linux-success-sophia-elya`
**Commits**: 0780477 (mDNS fix), 01c607b (documentation), [current] (static peers proof-of-concept)

**Recommendation**: Merge mDNS interval fix, keep static peers as documented workaround but don't merge hardcoded version.
