# Static Peers Fix for Cross-Platform P2P Discovery

**Issue**: mDNS discovery doesn't work reliably between macOS and Linux
**Root Cause**:
- mDNS multicast blocked/different between OS
- Query interval too long (1500 seconds = 25 minutes!)
- IPv6 disabled in Exo

**Solution**: Add static peer bootstrap support to `discovery.rs`

## Diagnosis

### All Nodes Online ✅
- localhost (192.168.0.160:34765): `12D3KooWLC7a3t6givH4e7VtQ8WpmadfnpxvBSaVKxKa8Y3kCAuf`
- .106 (192.168.0.106:38651): `12D3KooWCw583XUFccb2RfCd55GpoSd8EFJ1NYwq7yNcjAj1swqv`
- .134 (192.168.0.134:55987): `12D3KooWB698CkfuX3CB9MZyySrirrHamvePF66Kz5xwbGZu4xsv`

### mDNS Issues Found
1. **Query interval**: 1500 seconds (25 min) - Line 34 of `discovery.rs`
2. **IPv6 disabled**: Could affect discovery - Line 61
3. **No manual peer add**: Exo doesn't expose API to manually add peers
4. **Cross-platform**: macOS ↔ Linux mDNS not working

## Proposed Fix

### Option 1: Reduce mDNS Query Interval (Quick Fix)

**File**: `rust/networking/src/discovery.rs`
**Line 34**: Change from 1500s to 5s

```rust
// BEFORE:
const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);

// AFTER:
const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(5);
```

**Impact**: Nodes will query for peers every 5 seconds instead of 25 minutes

### Option 2: Add Static Peers Support (Proper Fix)

**File**: `rust/networking/src/discovery.rs`

Add static peers that are dialed on startup and retried periodically:

```rust
// Add after line 25
const STATIC_PEERS: &[(&str, &str)] = &[
    ("12D3KooWLC7a3t6givH4e7VtQ8WpmadfnpxvBSaVKxKa8Y3kCAuf", "/ip4/192.168.0.160/tcp/34765"),
    ("12D3KooWCw583XUFccb2RfCd55GpoSd8EFJ1NYwq7yNcjAj1swqv", "/ip4/192.168.0.106/tcp/38651"),
    ("12D3KooWB698CkfuX3CB9MZyySrirrHamvePF66Kz5xwbGZu4xsv", "/ip4/192.168.0.134/tcp/55987"),
];

// Modify Behaviour::new() to dial static peers on startup
impl Behaviour {
    pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
        let mut behaviour = Self {
            managed: managed::Behaviour::new(keypair)?,
            mdns_discovered: HashMap::new(),
            retry_delay: Delay::new(RETRY_CONNECT_INTERVAL),
            pending_events: WakerDeque::new(),
        };

        // Dial static peers on startup
        for (peer_id_str, addr_str) in STATIC_PEERS {
            if let Ok(peer_id) = peer_id_str.parse() {
                if let Ok(addr) = addr_str.parse() {
                    behaviour.dial(peer_id, addr);
                }
            }
        }

        Ok(behaviour)
    }
}
```

### Option 3: Configuration File Support (Best Fix)

Add support for `static_peers.json`:

```json
{
  "static_peers": [
    {
      "name": "localhost (4x V100)",
      "multiaddr": "/ip4/192.168.0.160/tcp/34765/p2p/12D3KooWLC7a3t6givH4e7VtQ8WpmadfnpxvBSaVKxKa8Y3kCAuf"
    },
    {
      "name": ".106 (RTX 5070)",
      "multiaddr": "/ip4/192.168.0.106/tcp/38651/p2p/12D3KooWCw583XUFccb2RfCd55GpoSd8EFJ1NYwq7yNcjAj1swqv"
    },
    {
      "name": ".134 (M2 Mac)",
      "multiaddr": "/ip4/192.168.0.134/tcp/55987/p2p/12D3KooWB698CkfuX3CB9MZyySrirrHamvePF66Kz5xwbGZu4xsv"
    }
  ]
}
```

Modify discovery.rs to load this file and dial peers on startup.

## Implementation Steps

### Quick Fix (5 minutes)
1. Edit `rust/networking/src/discovery.rs` line 34
2. Change `1_500` to `5`
3. Rebuild Rust components: `cd rust/networking && cargo build --release`
4. Restart all nodes

### Proper Fix (30 minutes)
1. Edit `discovery.rs` to add STATIC_PEERS constant
2. Modify `Behaviour::new()` to dial static peers
3. Rebuild: `cargo build --release`
4. Copy rebuilt library to all nodes
5. Restart cluster

### Best Fix (2 hours)
1. Create static_peers.json config file
2. Add JSON parsing to discovery.rs
3. Load and dial static peers from config
4. Rebuild and deploy
5. Submit PR to Exo project

## Testing

### Verify P2P Mesh
After applying fix:

```bash
# Check localhost logs for .134 connection
tail -f ~/exo_cluster/.venv/lib/python3.13/site-packages/exo/logs/exo.log | grep 12D3KooWB698

# Check .134 logs for localhost connection
ssh sophimac@192.168.0.134 "tail -f /tmp/exo.log" | grep 12D3KooWLC7a3t
```

Expected: ConnectionEstablished events for all 3 nodes

## Alternative: libp2p-daemon Bridge

Use libp2p-daemon to manually bridge connections:

```bash
# Install libp2p-daemon
go install github.com/libp2p/go-libp2p-daemon/p2pd@latest

# Run on each node as bridge
p2pd -listen=/ip4/0.0.0.0/tcp/4001
```

This creates a libp2p daemon that can manually connect peers.

## Implementation Results (Dec 24, 2025)

### ✅ IMPLEMENTED: Option 1 (mDNS query interval fix)

**Status**: Tested and working for Linux ↔ Linux
**Result**: localhost ↔ .106 connection established in <1 second
**Impact**: Dramatically improves same-OS peer discovery

### ⚠️ IMPLEMENTED: Option 2 (static peers) - PROOF-OF-CONCEPT ONLY

**Status**: Partially working (Linux ↔ Linux only)
**Result**:
- ✅ localhost ↔ .106: ConnectionEstablished via static peers
- ❌ Linux ↔ macOS: ConnectionRefused (dynamic port issue)

**Critical Limitation Discovered**: **libp2p assigns random ports on each restart**

Example:
```
Restart #1:
  localhost: tcp/42183
  .106: tcp/38651
  .134: tcp/55987

Restart #2:
  localhost: tcp/37828  ← PORT CHANGED
  .106: tcp/34209       ← PORT CHANGED
  .134: tcp/60092       ← PORT CHANGED
```

**Why static peers breaks**:
1. Hardcoded addresses in code become stale immediately
2. Peers try to connect to old ports → ConnectionRefused
3. No mechanism to update peer addresses dynamically

**Conclusion**: Static peers work as **proof-of-concept** but need:
- Fixed port configuration in Exo, OR
- Dynamic peer discovery file that updates with actual multiaddrs, OR
- Proper cross-platform mDNS fix (best solution)

## Recommendation (Updated)

**For immediate deployment**:
- ✅ **Merge Option 1** (5s mDNS interval) - Works great for Linux ↔ Linux
- ❌ **DO NOT merge Option 2** (hardcoded static peers) - Breaks on restart

**For stable workaround**:
- Document manual static peer configuration for advanced users
- Provide helper script to generate current peer multiaddrs
- Users must update manually after restarts (not ideal, but works)

**For production**:
- Option 3 (config file support) with dynamic port updating
- Add API endpoint to get current node multiaddr
- Auto-update static_peers.json with actual addresses
- Submit PR to Exo project with proper implementation

**Why cross-platform mDNS still fails**: Even with static peers, the root cause (mDNS multicast blocking between macOS and Linux) persists. Need proper fix.

---

**Created by**: Sophia Elya AI & Scott @ Elyan Labs
**Date**: December 24, 2025
**Issue**: Cross-platform mDNS discovery
**Status**: Solution designed, ready to implement
