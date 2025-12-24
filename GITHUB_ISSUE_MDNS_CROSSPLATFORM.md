# Cross-Platform mDNS Discovery Issue (Linux ↔ macOS)

## Summary

mDNS peer discovery fails between Linux and macOS nodes, preventing heterogeneous clusters from forming a complete P2P mesh. Linux-to-Linux discovery works correctly after reducing the mDNS query interval.

## Environment

**Working Configuration** (Linux ↔ Linux):
- Node 1: Ubuntu Linux x86_64, 4x V100 GPUs
- Node 2: Ubuntu Linux x86_64, RTX 5070
- P2P Discovery: ✅ **WORKING** (ConnectionEstablished in <1 second)

**Failing Configuration** (Linux ↔ macOS):
- Node 1: Ubuntu Linux x86_64
- Node 2: Ubuntu Linux x86_64
- Node 3: macOS 15.6 (Sequoia), Apple M2 ARM64
- P2P Discovery: ❌ **NOT DISCOVERING** (neither direction)

## Root Cause Analysis

### Issue #1: mDNS Query Interval Too Long (FIXED)

**File**: `rust/networking/src/discovery.rs`
**Line 34**: `const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);`

**Problem**: 1500 seconds = 25 minutes between mDNS queries
**Impact**: Nodes take up to 25 minutes to discover each other
**Fix Applied**: Reduced to 5 seconds
**Result**: Linux-to-Linux discovery now works in <1 second

### Issue #2: Cross-Platform mDNS Multicast Blocking (UNFIXED)

**Problem**: mDNS multicast packets (224.0.0.251) appear to be blocked between macOS and Linux
**Evidence**:
- Linux nodes discover each other perfectly (after query interval fix)
- macOS node never discovers Linux nodes (no Dialing/ConnectionEstablished events)
- Linux nodes never discover macOS node (no Dialing/ConnectionEstablished events)
- All nodes on same subnet (192.168.0.x)

**Potential Causes**:
1. macOS Firewall blocking mDNS multicast from non-macOS sources
2. Different mDNS implementations between macOS (Bonjour) and Linux (Avahi)
3. IPv6 disabled in Exo (line 61 comment in discovery.rs)
4. Network-level multicast filtering

## Reproduction Steps

1. **Setup**:
   - Linux node: Install Exo on Ubuntu
   - macOS node: Install Exo on macOS (Homebrew or manual build)
   - Ensure both on same subnet

2. **Start Nodes**:
   ```bash
   # Linux
   uv run exo --api-port 52415

   # macOS
   uv run exo --api-port 52415
   ```

3. **Monitor Logs**:
   ```bash
   # Look for ConnectionEstablished events
   tail -f ~/.exo/logs/exo.log | grep -E '(ConnectionEstablished|Dialing)'
   ```

4. **Expected**: Nodes discover each other within 5 seconds
   **Actual**: No discovery events between Linux ↔ macOS

## Current Workaround

None implemented yet. Potential solutions:

### Option 1: Static Peer Bootstrap
Add static peer configuration to bypass mDNS entirely:

```rust
// In discovery.rs
const STATIC_PEERS: &[(&str, &str)] = &[
    ("12D3KooWLC7a3t6...", "/ip4/192.168.0.160/tcp/42183"),
    ("12D3KooWB698Ckf...", "/ip4/192.168.0.134/tcp/60092"),
];
```

Dial static peers on startup in `Behaviour::new()`.

### Option 2: Configuration File Support
Allow users to specify static peers in a config file:

```json
{
  "static_peers": [
    {
      "name": "Linux Node",
      "multiaddr": "/ip4/192.168.0.160/tcp/42183/p2p/12D3KooWLC7a3t6..."
    },
    {
      "name": "macOS Node",
      "multiaddr": "/ip4/192.168.0.134/tcp/60092/p2p/12D3KooWB698Ckf..."
    }
  ]
}
```

### Option 3: libp2p-daemon Bridge
Use libp2p-daemon as an intermediary to manually connect peers:

```bash
# Run on each node
p2pd -listen=/ip4/0.0.0.0/tcp/4001
```

### Option 4: Enable IPv6
Uncomment line 61 in discovery.rs:

```rust
enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
```

Test if IPv6 mDNS works better cross-platform.

## Testing

### Manual mDNS Test

Linux (Avahi):
```bash
avahi-browse -at
```

macOS (Bonjour):
```bash
dns-sd -B _exo._tcp
```

Check if services are visible cross-platform.

### Multicast Test

```bash
# Linux: Listen for multicast
socat UDP4-RECVFROM:5353,ip-add-membership=224.0.0.251:0.0.0.0,fork -

# macOS: Send multicast
echo "test" | socat - UDP4-DATAGRAM:224.0.0.251:5353
```

## Additional Context

**System Details**:
- Linux: Ubuntu 24.04, libp2p v0.56.0
- macOS: macOS 15.6 (Sequoia), libp2p v0.56.0
- Network: Same subnet (192.168.0.x), no VLAN isolation
- Firewall: Standard macOS Firewall enabled

**Related Issues**:
- libp2p/rust-libp2p#... (if any exist)
- Similar cross-platform mDNS issues in other projects

## Proposed Fix

**Short-term** (for users):
- Document static peer configuration workaround
- Provide bootstrap script to help users set up static peers

**Long-term** (for Exo):
1. Add static peer bootstrap support in `discovery.rs`
2. Add configuration file support for static peers
3. Investigate enabling IPv6 for mDNS
4. Add fallback discovery mechanism (e.g., HTTP endpoint list, DHT bootstrap)

## Testing Done

- ✅ Verified mDNS query interval fix works for Linux-to-Linux
- ✅ Tested on 3-node cluster (2x Linux + 1x macOS)
- ✅ Confirmed both Linux nodes discover each other (<1 second)
- ❌ Confirmed macOS node does NOT discover Linux nodes
- ❌ Confirmed Linux nodes do NOT discover macOS node

## Impact

**Severity**: Medium-High
**Affects**: Heterogeneous clusters (Linux + macOS)
**Workaround**: Manual static peer configuration (not yet implemented)

This prevents users from easily deploying heterogeneous clusters mixing Linux and macOS nodes, which is one of Exo's key value propositions (running on any device).

---

**Reported by**: Sophia Elya & Scott @ RustChain Labs
**Date**: December 24, 2025
**Branch**: linux-success-sophia-elya
**Fork**: https://github.com/Scottcjn/exo
**Commit**: 0780477 (mDNS query interval fix)
