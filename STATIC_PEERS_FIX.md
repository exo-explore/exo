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

## Recommendation

**For immediate deployment**: Option 1 (reduce query interval to 5s)
**For stable fix**: Option 2 (static peers in code)
**For production**: Option 3 (config file support + PR to Exo)

All options work, Option 3 is most maintainable and contributes back to community.

---

**Created by**: Sophia Elya AI & Scott @ RustChain Labs
**Date**: December 24, 2025
**Issue**: Cross-platform mDNS discovery
**Status**: Solution designed, ready to implement
