# rust/networking

The `networking` crate is exo's peer-to-peer transport layer: a thin, opinionated wrapper around [libp2p](https://github.com/libp2p/rust-libp2p) that provides mDNS-driven peer discovery, a gossipsub pub/sub overlay, a Noise-encrypted TCP transport hardened by a Pre-Shared-Key private-network upgrade, and a channel-based "swarm" API suitable for embedding in an async Python runtime via pyo3 bindings.

This page documents what the crate actually does today. For how the Rust swarm is surfaced into Python, see [`rust-pyo3-bindings.md`](./rust-pyo3-bindings.md). For how exo routes typed messages on top of gossipsub topics, see [`routing.md`](./routing.md). For the full master/worker message flow, see [`../architecture/data-flow.md`](../architecture/data-flow.md).

## Overview

The crate has two responsibilities and nothing else:

1. Build a libp2p `Swarm` with a specific set of behaviours (mDNS + ping for discovery, gossipsub for messaging) (`rust/networking/src/swarm.rs:245-286`).
2. Expose that swarm as an `async_stream::stream!` of `FromSwarm` events, plus a tokio mpsc channel of `ToSwarm` commands, so that clients never need to touch the libp2p types directly (`rust/networking/src/swarm.rs:22-80`).

Everything else — master election, event routing, inference task dispatch, RDMA path selection, Thunderbolt bridge orchestration — lives in the Python side of exo. This crate is intentionally narrow.

A placeholder crate docstring (`rust/networking/src/lib.rs:1-5`) flags that proper crate-level documentation is still pending.

## Cargo workspace layout

The crate is one of three members of the top-level Cargo workspace declared in `/Users/leozealous/exo/Cargo.toml:1-3`, alongside `rust/exo_pyo3_bindings` (the Python-facing shim) and `rust/util` (shared helpers such as `WakerDeque`).

`rust/networking/Cargo.toml:1-43` pins the crate to workspace-level `version`, `edition`, and lints. The dependency surface is small and deliberate:

- `libp2p` with the `full` feature flag (`rust/networking/Cargo.toml:41`) — everything needed for TCP, noise, yamux, gossipsub, mDNS, ping, and pnet.
- `tokio` with `full` (`rust/networking/Cargo.toml:27`) — async runtime. `create_swarm` must be called inside a tokio context or the `.build()` call panics; the pyo3 shim guards this with a runtime guard at `rust/exo_pyo3_bindings/src/networking.rs:196`.
- `futures-lite`, `futures-timer`, `async-stream` — stream plumbing (`rust/networking/Cargo.toml:24-26`).
- `either`, `extend`, `delegate`, `pin-project` — small macro/datastructure helpers (`rust/networking/Cargo.toml:17-21, 42`).
- `keccak-const` (`rust/networking/Cargo.toml:35`) — compile-time SHA3-256 used to derive the private-network preshared key.
- `util` (`rust/networking/Cargo.toml:30`) — provides `WakerDeque` used for the discovery behaviour's pending-event queue (`rust/networking/src/discovery.rs:22`).

Source layout:

- `src/lib.rs` — module declarations plus a `MultiaddrExt::try_to_tcp_addr` extension trait that extracts `(IpAddr, u16)` from a libp2p `Multiaddr` (`rust/networking/src/lib.rs:17-44`).
- `src/swarm.rs` — public API: `Swarm`, `ToSwarm`, `FromSwarm`, `create_swarm`, and the private `transport` and `behaviour` submodules.
- `src/discovery.rs` — a custom `NetworkBehaviour` that wraps mDNS + ping and promotes "discovered" peers into real durable connections.
- `src/RESEARCH_NOTES.txt` — exploratory notes on low-level Thunderbolt/NDRV socket work; **not** compiled code (`rust/networking/src/RESEARCH_NOTES.txt:1-44`).
- `examples/chatroom.rs` — minimal stdin-to-gossipsub demo (`rust/networking/examples/chatroom.rs:1-82`).
- `tests/dummy.rs` — placeholder test with a single `does_nothing` function (`rust/networking/tests/dummy.rs:1-8`).

## libp2p integration

### Swarm composition

The top-level `Behaviour` struct (`rust/networking/src/swarm.rs:252-256`) is a derived `NetworkBehaviour` with two children:

- `discovery: discovery::Behaviour` — the custom mDNS+ping wrapper described below.
- `gossipsub: gossipsub::Behaviour` — standard libp2p gossipsub.

`create_swarm` (`rust/networking/src/swarm.rs:149-169`) wires these together via `SwarmBuilder::with_existing_identity(keypair).with_tokio().with_other_transport(tcp_transport).with_behaviour(...).build()`, then binds a listener to `/ip4/0.0.0.0/tcp/{listen_port}` (`rust/networking/src/swarm.rs:167`). Passing `listen_port = 0` lets the OS assign one (`rust/networking/src/swarm.rs:147`).

### Transport stack

`tcp_transport` (`rust/networking/src/swarm.rs:213-242`) builds the byte pipe:

1. `libp2p::tcp::tokio::Transport` with `TCP_NODELAY` on (`rust/networking/src/swarm.rs:222, 234`) — disables Nagle to keep latency low.
2. A custom `pnet_upgrade` stage wedged in via `.and_then(...)` (`rust/networking/src/swarm.rs:199-210, 235`) that runs a `PnetConfig` handshake using a 32-byte preshared key.
3. `Version::V1Lazy` multistream upgrade (`rust/networking/src/swarm.rs:225, 236`) for 0-RTT negotiation.
4. Noise authentication (`rust/networking/src/swarm.rs:228, 237`) — the source comment notes this was chosen over TLS for speed and that "we don't care much for security", which is true because the pnet layer below already restricts who can even start a handshake.
5. Yamux multiplexing with defaults (`rust/networking/src/swarm.rs:231, 238`).

### Private-network isolation

The `PNET_PRESHARED_KEY` (`rust/networking/src/swarm.rs:184-194`) is derived at runtime as `Sha3_256("exo_discovery_network" ‖ version)`, where `version` is either:

- the byte string in `NETWORK_VERSION` (currently hardcoded to `b"v0.0.1"`, `rust/networking/src/swarm.rs:17`), or
- the value of the `EXO_LIBP2P_NAMESPACE` environment variable if set (`rust/networking/src/swarm.rs:18, 187-189`).

This is how operators partition overlapping clusters on the same LAN — the Thunderbolt ops runbook documents the live pair using `namespace: zealous-cluster-apr11` (`/Users/leozealous/exo/docs/thunderbolt-bridge-ops.md:26`), which becomes `EXO_LIBP2P_NAMESPACE` at the Python layer.

### Gossipsub configuration

`gossipsub_behaviour` (`rust/networking/src/swarm.rs:270-285`) uses:

- `MessageAuthenticity::Signed(keypair)` (`rust/networking/src/swarm.rs:277`) — every message is signed with the peer's identity keypair, so gossipsub can compute deterministic message IDs itself.
- `ValidationMode::Strict` (`rust/networking/src/swarm.rs:280`) — drops unsigned or malformed messages.
- `max_transmit_size = 8 * 1024 * 1024` (`rust/networking/src/swarm.rs:279`) — 8 MiB per message. Exceeding this surfaces on the Python side as `MessageTooLargeError` (`rust/exo_pyo3_bindings/src/networking.rs:100-129`).

### Control-plane API

Client code never holds the `libp2p::Swarm` directly. Instead it gets an mpsc `Sender<ToSwarm>` and a `Stream<FromSwarm>`:

- `ToSwarm` (`rust/networking/src/swarm.rs:22-36`) carries `Subscribe`, `Unsubscribe`, and `Publish` requests, each with a `tokio::sync::oneshot::Sender` for the reply. `on_message` (`rust/networking/src/swarm.rs:82-116`) dispatches these into the gossipsub behaviour.
- `FromSwarm` (`rust/networking/src/swarm.rs:37-49`) emits `Message { from, topic, data }`, `Discovered { peer_id }`, and `Expired { peer_id }`. `filter_swarm_event` (`rust/networking/src/swarm.rs:118-143`) is the single place where raw libp2p `SwarmEvent`s are projected into this minimal vocabulary; everything else is dropped.

`Swarm::into_stream` (`rust/networking/src/swarm.rs:57-80`) drives a `tokio::select!` loop inside `async_stream::stream!`, alternating between polling `from_client.recv()` (operator commands) and `swarm.next()` (libp2p events), yielding a boxed `Stream<Item = FromSwarm> + Send`.

### Discovery behaviour

`discovery::Behaviour` (`rust/networking/src/discovery.rs:103-208`) is a custom `NetworkBehaviour` that exists because libp2p's mDNS "discovered" event does not mean "connected". The wrapper converts mDNS hints into durable TCP connections.

The inner `managed::Behaviour` (`rust/networking/src/discovery.rs:37-50`) bundles:

- `mdns::tokio::Behaviour` tuned with `ttl = 2500s` and `query_interval = 1500s` (`rust/networking/src/discovery.rs:32-33, 52-66`). An inline comment (`rust/networking/src/discovery.rs:60`) flags that `enable_ipv6` is currently disabled because TCP+mDNS over IPv6 was unreliable.
- `ping::Behaviour` with a 2500 ms timeout and 2500 ms interval (`rust/networking/src/discovery.rs:34-35, 68-74`) — used as a liveness probe; a failed ping closes the connection.

The main state machine (`rust/networking/src/discovery.rs:115-208`) does four things:

1. Track mDNS-discovered peers in a `HashMap<PeerId, BTreeSet<Multiaddr>>` and dial each newly-seen `(peer, addr)` pair immediately (`rust/networking/src/discovery.rs:140-154`). The code asserts that the same multiaddress cannot be "discovered" twice (`rust/networking/src/discovery.rs:152`).
2. Remove expired peers and their addresses, asserting they were previously present (`rust/networking/src/discovery.rs:156-173`).
3. On every retry tick (`RETRY_CONNECT_INTERVAL = 5s`, `rust/networking/src/discovery.rs:24, 366-380`), re-dial every known mDNS peer and every configured bootstrap peer. Redialing an already-connected peer is a safe no-op.
4. Emit only truly-durable connection events (`Event::ConnectionEstablished` / `Event::ConnectionClosed`, `rust/networking/src/discovery.rs:79-92`) and only for remotes whose multiaddr has a TCP leaf (`rust/networking/src/discovery.rs:280-316`, using `MultiaddrExt::try_to_tcp_addr`).

Ping errors immediately close the offending connection (`rust/networking/src/discovery.rs:343-347`), and any mid-session `AddressChange` event is treated as unreachable because the TCP transport forbids it (`rust/networking/src/discovery.rs:319-322`).

## Thunderbolt discovery and RDMA

The Rust crate does **not** implement RDMA and does **not** speak Thunderbolt-specific protocols. Its role on the Thunderbolt path is indirect but load-bearing:

1. The swarm listens on `0.0.0.0:${listen_port}` (`rust/networking/src/swarm.rs:167`). When macOS assigns a `10.0.0.x/24` address to `bridge0` (the Thunderbolt Bridge interface), the libp2p TCP listener naturally accepts connections arriving over that interface — no special casing needed.
2. mDNS multicasts across every non-loopback interface, including `bridge0`, so peers on the Thunderbolt bridge discover each other as ordinary mDNS neighbours (`rust/networking/src/discovery.rs:52-66`).
3. The `bootstrap_peers` argument to `create_swarm` (`rust/networking/src/swarm.rs:152-159`) lets Python hand in explicit Thunderbolt multiaddrs like `/ip4/10.0.0.1/tcp/52415`. In the live Apr 11 2026 pair these are used with "Thunderbolt first, Tailscale fallback second" preference (`/Users/leozealous/exo/docs/thunderbolt-bridge-ops.md:27`). The discovery behaviour redials every bootstrap peer on each `RETRY_CONNECT_INTERVAL` tick (`rust/networking/src/discovery.rs:373-378`), which is how mDNS-less environments (e.g. Tailscale) still converge.

RDMA path selection (`rdma_en2`, `nodeRdmaCtl.enabled`, `nodeThunderboltBridge.enabled`) happens entirely in the Python layer and is observable in `/state` at runtime (`/Users/leozealous/exo/docs/thunderbolt-bridge-ops.md:29`). The Rust swarm is unaware of it — it simply delivers a libp2p gossipsub session over whichever interface TCP reaches first, and Python decides whether to overlay RDMA on top.

The file `rust/networking/src/RESEARCH_NOTES.txt` (`rust/networking/src/RESEARCH_NOTES.txt:1-44`) captures exploratory notes on PF_NDRV sockets, `thunderbolt-net` on Linux, and ZeroTier's kext-free fake-ethernet approach. None of this is wired into the crate yet.

## Public API surface

The crate's `pub` surface, re-exported to `exo_pyo3_bindings`:

- Module `networking::swarm` (`rust/networking/src/lib.rs:7`) with:
  - `pub const NETWORK_VERSION: &[u8]` (`rust/networking/src/swarm.rs:17`)
  - `pub const OVERRIDE_VERSION_ENV_VAR: &str` (`rust/networking/src/swarm.rs:18`)
  - `pub enum ToSwarm { Subscribe, Unsubscribe, Publish }` (`rust/networking/src/swarm.rs:22-36`)
  - `pub enum FromSwarm { Message, Discovered, Expired }` (`rust/networking/src/swarm.rs:37-49`)
  - `pub struct Swarm` with `pub fn into_stream(self)` (`rust/networking/src/swarm.rs:51-80`)
  - `pub fn create_swarm(keypair, from_client, bootstrap_peers, listen_port) -> AnyResult<Swarm>` (`rust/networking/src/swarm.rs:149-169`)
  - Re-exports `Behaviour`, `BehaviourEvent` from the private `behaviour` submodule (`rust/networking/src/swarm.rs:5`).
- Module `networking::discovery` (`rust/networking/src/lib.rs:6`) exposes `pub enum Event` (`rust/networking/src/discovery.rs:79-92`) and `pub struct Behaviour` (`rust/networking/src/discovery.rs:103-113`). These are referenced by `swarm.rs` but not typically used directly by Python.

The pyo3 shim consumes exactly three items from the public surface — `create_swarm`, `ToSwarm`, `FromSwarm` (`rust/exo_pyo3_bindings/src/networking.rs:14`) — and wraps them in a `PyNetworkingHandle` with `gossipsub_subscribe`, `gossipsub_unsubscribe`, `gossipsub_publish`, and `recv` (`rust/exo_pyo3_bindings/src/networking.rs:132-299`). Three specific gossipsub `PublishError` variants are lifted into Python-visible exceptions: `NoPeersSubscribedToTopicError`, `AllQueuesFullError`, `MessageTooLargeError` (`rust/exo_pyo3_bindings/src/networking.rs:24-130, 287-295`). See [`rust-pyo3-bindings.md`](./rust-pyo3-bindings.md) for the full binding contract.

## Testing

Testing is essentially absent at the crate level. `rust/networking/tests/dummy.rs:1-8` is an empty placeholder:

```rust
// maybe this will hold test in the future...??

#[cfg(test)]
mod tests {
    #[test]
    fn does_nothing() {}
}
```

The only runnable manual verification is the `chatroom` example (`rust/networking/examples/chatroom.rs:1-82`), which spawns a swarm, subscribes to topic `"test-net"`, and pipes stdin lines into `gossipsub_publish` while printing incoming `Discovered` / `Expired` / `Message` events. Run two copies on the same LAN to confirm mDNS discovery, pnet pairing, and gossipsub delivery all work end-to-end.

Functional coverage lives in the Python test suite on top of the pyo3 bindings — see [`rust-pyo3-bindings.md`](./rust-pyo3-bindings.md) and the routing tests referenced from [`routing.md`](./routing.md).

## Gotchas

- **`create_swarm` must run inside a tokio runtime.** `SwarmBuilder::with_tokio().build()` (`rust/networking/src/swarm.rs:162-165`) resolves the current tokio handle at construction time. The pyo3 shim wraps the call in `pyo3_async_runtimes::tokio::get_runtime().enter()` (`rust/exo_pyo3_bindings/src/networking.rs:196-197`) — replicate that pattern if you ever build a swarm outside `PyNetworkingHandle::py_new`.
- **The `chatroom` example's signature is stale.** `rust/networking/examples/chatroom.rs:19` calls `swarm::create_swarm(identity, from_client)` with two arguments, but the real signature (`rust/networking/src/swarm.rs:149-154`) takes four: keypair, receiver, bootstrap peers, listen port. The example will not compile as written; before running it you need to pass `vec![]` and a port (e.g. `0`).
- **`NETWORK_VERSION` is hardcoded.** Any two peers whose `NETWORK_VERSION` differs — or whose `EXO_LIBP2P_NAMESPACE` env vars differ — cannot complete the pnet handshake and will appear as silent peers. This is the intended isolation mechanism but means version bumps are a hard cluster-wide break. The source itself flags this as open design territory (`rust/networking/src/swarm.rs:12-16`).
- **mDNS IPv6 is off.** `rust/networking/src/discovery.rs:60` deliberately keeps `enable_ipv6` at its default (`false`) because of an unresolved TCP+mDNS interaction. IPv6-only networks will not discover peers.
- **Discovery asserts on the mDNS contract.** `handle_mdns_discovered` and `handle_mdns_expired` (`rust/networking/src/discovery.rs:140-173`) use `assert!` / `expect` to enforce that mDNS cannot report a peer discovered twice, or expire a peer it never discovered. A buggy upstream mDNS could panic the whole swarm task. This is deliberate: the invariants must hold for the wake-up scheduling in `poll` to be correct.
- **Ping failure drops the connection with no backoff.** `rust/networking/src/discovery.rs:343-347` closes the connection on any ping error; the next retry tick (up to 5 s later, `rust/networking/src/discovery.rs:24`) redials. Flappy links will see repeated reconnects.
- **`AddressChange` is `unreachable!()`.** `rust/networking/src/discovery.rs:320-322` deliberately panics if a connection's remote address changes, because the TCP transport in use should never produce one. Introducing QUIC or another roaming-capable transport will hit this.
- **The preshared-key comment says "we don't care much for security".** `rust/networking/src/swarm.rs:227`. Anyone on the same LAN who can observe the `EXO_LIBP2P_NAMESPACE` value can join the network. The pnet layer is designed to partition, not authenticate — do not rely on it for secrecy.
- **Empty or malformed bootstrap multiaddrs are silently dropped.** `rust/networking/src/swarm.rs:155-159` filters empties and `filter_map(|s| s.parse().ok())` swallows parse errors. A typo in a bootstrap multiaddr becomes a missing peer, not an error at startup.

---

**Sources:**

- `/Users/leozealous/exo/rust/networking/Cargo.toml`
- `/Users/leozealous/exo/rust/networking/src/lib.rs`
- `/Users/leozealous/exo/rust/networking/src/swarm.rs`
- `/Users/leozealous/exo/rust/networking/src/discovery.rs`
- `/Users/leozealous/exo/rust/networking/src/RESEARCH_NOTES.txt`
- `/Users/leozealous/exo/rust/networking/examples/chatroom.rs`
- `/Users/leozealous/exo/rust/networking/tests/dummy.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/networking.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/Cargo.toml`
- `/Users/leozealous/exo/Cargo.toml`
- `/Users/leozealous/exo/docs/thunderbolt-bridge-ops.md`

**See also:**

- [`rust-pyo3-bindings.md`](./rust-pyo3-bindings.md) — Python-facing wrapper around this crate.
- [`routing.md`](./routing.md) — typed pub/sub topics layered on top of gossipsub.
- [`../architecture/data-flow.md`](../architecture/data-flow.md) — end-to-end cluster message flow.

*Last indexed: c0d5bf92, 2026-04-21*
