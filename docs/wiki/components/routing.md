# Routing Layer

## Overview

The `src/exo/routing/` module is exo's in-process message bus and the Python-side wrapper over the Rust libp2p swarm. It owns three concerns:

1. **Typed topic definitions** — six pub/sub topics, each bound to a pydantic model type and a publish policy that decides whether traffic stays local or crosses the network. [`src/exo/routing/topics.py:14-51`](../../../src/exo/routing/topics.py)
2. **The `Router`** — the runtime that hosts per-topic fanout, forwards inbound gossipsub messages into typed channels, and publishes outbound messages to peers via the `NetworkingHandle` PyO3 binding. [`src/exo/routing/router.py:106-270`](../../../src/exo/routing/router.py)
3. **Event sequencing helpers** — `EventRouter` layers per-session ingestion, ordered buffering, and NACK-driven catch-up on top of the two global event topics. [`src/exo/routing/event_router.py:23-167`](../../../src/exo/routing/event_router.py)

Every node in an exo cluster spins up exactly one `Router`. Commands, events, election messages, and connection updates all flow through it, and every other subsystem (Master, Worker, Election, API) obtains `Sender`/`Receiver` handles from it rather than touching libp2p directly. See [../architecture/event-sourcing-message-passing.md](../architecture/event-sourcing-message-passing.md) for how this fits into the event-sourcing model and [master.md](master.md) / [worker.md](worker.md) for consumers.

The `__init__.py` is intentionally empty — nothing is re-exported at the package level. [`src/exo/routing/__init__.py:1`](../../../src/exo/routing/__init__.py)

## Topics

A topic is a `TypedTopic[T]` dataclass binding three things: a string name used as the gossipsub topic identifier, a `PublishPolicy` (`Never` / `Minimal` / `Always`), and the pydantic `CamelCaseModel` subclass that payloads on this topic must serialize to. [`src/exo/routing/topics.py:23-37`](../../../src/exo/routing/topics.py)

### PublishPolicy

```
Never   — local-only; never goes on the wire
Minimal — publish only if no local receiver is registered for this topic
Always  — always publish, even if there is a local receiver
```

Definitions at [`src/exo/routing/topics.py:14-20`](../../../src/exo/routing/topics.py). The policy is enforced by `TopicRouter.run` before fanning out to local subscribers — see [`src/exo/routing/router.py:56-69`](../../../src/exo/routing/router.py).

### The six built-in topics

All topics are declared as module-level constants in `topics.py`. They are the complete set of named channels the system uses. [`src/exo/routing/topics.py:40-51`](../../../src/exo/routing/topics.py)

| Topic constant | String name | Payload type | Policy | Purpose |
|---|---|---|---|---|
| `GLOBAL_EVENTS` | `global_events` | `GlobalForwarderEvent` | Always | Master broadcasts indexed events to all workers |
| `LOCAL_EVENTS` | `local_events` | `LocalForwarderEvent` | Always | Workers send session-tagged events to master |
| `COMMANDS` | `commands` | `ForwarderCommand` | Always | Workers / API submit commands to master |
| `ELECTION_MESSAGES` | `election_messages` | `ElectionMessage` | Always | Bully-algorithm traffic between nodes |
| `CONNECTION_MESSAGES` | `connection_messages` | `ConnectionMessage` | Never | Local-only peer up/down notifications |
| `DOWNLOAD_COMMANDS` | `download_commands` | `ForwarderDownloadCommand` | Always | Download-related commands split from the main command channel |

`CONNECTION_MESSAGES` is the only `Never` topic — the payloads originate inside the Rust swarm (peer discovered / expired) and are only ever relevant to this node, so they short-circuit the outbound path. [`src/exo/routing/topics.py:46-48`](../../../src/exo/routing/topics.py)

### Serialization helpers

`TypedTopic` carries its own serializer pair, and it is the only place pydantic-to-bytes conversion happens for routed payloads:

- `serialize(t)` → `t.model_dump_json().encode("utf-8")` [`src/exo/routing/topics.py:32-34`](../../../src/exo/routing/topics.py)
- `deserialize(b)` → `model_type.model_validate_json(b.decode("utf-8"))` [`src/exo/routing/topics.py:36-37`](../../../src/exo/routing/topics.py)

## Router

`Router` is the long-lived singleton per node. It stores a map of `topic.topic → TopicRouter` and a single outbound channel into the networking layer. [`src/exo/routing/router.py:106-138`](../../../src/exo/routing/router.py)

### Construction

The `create` classmethod builds the Rust `NetworkingHandle` and wires it into a fresh `Router`:

- Takes a `Keypair` (obtained via `get_node_id_keypair`), optional bootstrap peer multiaddrs, optional listen port. [`src/exo/routing/router.py:107-118`](../../../src/exo/routing/router.py)
- The constructor allocates the outbound `(topic, bytes)` channel, stashes the sender as `_tmp_networking_sender` so it can be handed to the first registered `TopicRouter`, and captures a `TaskGroup` for the run loop. [`src/exo/routing/router.py:120-138`](../../../src/exo/routing/router.py)
- Bootstrap peers come either from the constructor argument or from the `EXO_BOOTSTRAP_PEERS` environment variable (comma-separated). [`src/exo/routing/router.py:259-270`](../../../src/exo/routing/router.py)

### Registering topics

`register_topic(typed_topic)` creates a `TopicRouter[T]` for the topic and inserts it into `topic_routers`. The first registration consumes `_tmp_networking_sender`; subsequent registrations clone a new sender from the receiver. If the router is already running (i.e. topics are registered dynamically after startup), it immediately subscribes via gossipsub. [`src/exo/routing/router.py:140-149`](../../../src/exo/routing/router.py)

### Obtaining senders and receivers

- `sender(topic)` returns a cloned `Sender[T]` from the topic's `TopicRouter`. Clones share the same underlying channel — all produced items flow through the single `TopicRouter.run` loop. [`src/exo/routing/router.py:151-157`](../../../src/exo/routing/router.py)
- `receiver(topic)` allocates a fresh `(send, recv)` pair, registers the sender in `router.senders`, and returns the receiver. Each caller gets its own dedicated inbound queue. [`src/exo/routing/router.py:159-170`](../../../src/exo/routing/router.py)

### The run loop

`Router.run` starts one task per `TopicRouter`, plus the networking receive loop and the networking publish loop, then issues gossipsub subscribe calls for every registered topic. It sleeps forever; shutdown is driven by cancellation of the task group, which triggers an unsubscribe pass inside the `finally` block (shielded for 1 second). [`src/exo/routing/router.py:172-193`](../../../src/exo/routing/router.py)

### Inbound: `_networking_recv`

Pulls `PyFromSwarm` values off the Rust handle and dispatches by variant: [`src/exo/routing/router.py:203-238`](../../../src/exo/routing/router.py)

- `PyFromSwarm.Message(origin, topic, data)` — looks up the `TopicRouter` by topic string, calls `publish_bytes(data)` on it, which deserializes and fans out to local subscribers. Messages on unregistered topics are logged as a warning and dropped. [`src/exo/routing/router.py:209-219`](../../../src/exo/routing/router.py)
- `PyFromSwarm.Connection(peer_id, connected)` — synthesizes a `ConnectionMessage` via `ConnectionMessage.from_update` and fans it out on the `CONNECTION_MESSAGES` topic (the one topic whose payloads come from the swarm itself, not from user code). [`src/exo/routing/router.py:220-229`](../../../src/exo/routing/router.py)

### Outbound: `_networking_publish`

Reads `(topic, bytes)` tuples from the shared outbound channel (fed by `TopicRouter._send_out`) and calls `self._net.gossipsub_publish(topic, data)`. It warns on payloads larger than 1 MiB and catches three specific Rust exceptions: [`src/exo/routing/router.py:240-257`](../../../src/exo/routing/router.py)

- `NoPeersSubscribedToTopicError` — silently swallowed; common on a fresh cluster
- `AllQueuesFullError` — logged, message dropped
- `MessageTooLargeError` — logged with byte count, message dropped

These three exception classes are defined in Rust at [`rust/exo_pyo3_bindings/src/networking.rs:29-130`](../../../rust/exo_pyo3_bindings/src/networking.rs).

### Keypair persistence

`get_node_id_keypair(path)` is a helper (not a `Router` method) that loads or generates an Ed25519 keypair at `EXO_NODE_ID_KEYPAIR`, guarded by a cross-process `FileLock`. On first run it generates, writes atomically via a temp file + `os.replace`, and returns; on subsequent runs it reads the protobuf-encoded bytes. Invalid files are logged and a fresh keypair is regenerated over the top. [`src/exo/routing/router.py:273-305`](../../../src/exo/routing/router.py)

## TopicRouter

A `TopicRouter[T]` is the per-topic fanout actor. One instance exists per registered `TypedTopic`. [`src/exo/routing/router.py:41-103`](../../../src/exo/routing/router.py)

### State

- `self.topic` — the `TypedTopic[T]` it serves
- `self.senders: set[Sender[T]]` — every subscriber's inbound sender (one per `Router.receiver()` call)
- `self._sender` / `self.receiver` — the inbound channel pair; producers call `new_sender()` to clone `_sender`
- `self.networking_sender` — the shared outbound `(topic, bytes)` channel into the Rust side

Constructor at [`src/exo/routing/router.py:42-53`](../../../src/exo/routing/router.py).

### The fanout loop

`TopicRouter.run` reads items off the internal receiver and decides whether to send to the network, to local subscribers, or both, according to `publish_policy`: [`src/exo/routing/router.py:55-69`](../../../src/exo/routing/router.py)

- **Minimal + no subscribers** → send out to network, skip local fanout (`continue`). This is the "only publish when nobody local wants it" path.
- **Always** → send out to network, then fall through to local fanout. Local subscribers still get a copy.
- **Never** (or Minimal with at least one local subscriber) → skip the outbound send, fanout locally only.

### Local fanout

`publish(item)` iterates a snapshot of `self.senders` (via `copy`), calls `sender.send(item)` on each, and garbage-collects any senders that raise `ClosedResourceError` or `BrokenResourceError`. [`src/exo/routing/router.py:79-91`](../../../src/exo/routing/router.py)

`publish_bytes(data)` is the inbound entry point used by `Router._networking_recv`: it deserializes the bytes through `self.topic.deserialize` and delegates to `publish`. [`src/exo/routing/router.py:93-94`](../../../src/exo/routing/router.py)

### Producer API

`new_sender()` returns `self._sender.clone()` — every producer gets its own clone, but all clones funnel into the same `self.receiver` that drives `run`. [`src/exo/routing/router.py:96-97`](../../../src/exo/routing/router.py)

### Outbound side

`_send_out(item)` serializes via `self.topic.serialize(item)` and pushes `(topic_name, bytes)` onto the shared outbound channel. It is only called from `run` when policy allows. [`src/exo/routing/router.py:99-103`](../../../src/exo/routing/router.py)

### Shutdown

`TopicRouter.shutdown` closes every subscriber sender, the internal sender, and the receiver. [`src/exo/routing/router.py:71-77`](../../../src/exo/routing/router.py)

## EventRouter

`EventRouter` is a higher-level helper layered on top of the event topics — it is not itself a topic router. It mediates between the outbound `LOCAL_EVENTS` channel and the inbound `GLOBAL_EVENTS` channel for a given session. [`src/exo/routing/event_router.py:23-77`](../../../src/exo/routing/event_router.py)

Key responsibilities:

- **Ingest** — workers produce raw `Event`s via `sender()`; each producer gets its own `SystemId` and per-producer `origin_idx` counter. Each event is wrapped in a `LocalForwarderEvent(origin_idx, origin, session, event)` and pushed out on `external_outbound` plus tracked in `out_for_delivery` with a timestamp. [`src/exo/routing/event_router.py:66-97`](../../../src/exo/routing/event_router.py)
- **Deliver** — indexed events from the master arrive on `external_inbound`; they are filtered to enforce that `event.origin == event.session.master_node_id`, deduplicated against `out_for_delivery`, ordered via an `OrderedBuffer`, and fanned out to every registered `internal_outbound` sender. [`src/exo/routing/event_router.py:99-137`](../../../src/exo/routing/event_router.py)
- **NACK on gaps** — when the buffer doesn't drain (i.e. we received an out-of-order event), it starts `_nack_request` which sleeps `0.5 * 2^n` seconds (capped at 10s) and then sends a `RequestEventLog(since_idx=...)` command to the master. The attempt counter and cancel scope reset once a drain succeeds. [`src/exo/routing/event_router.py:117-166`](../../../src/exo/routing/event_router.py)
- **Retry outbound** — `_simple_retry` loops every 1-2 seconds and re-sends any `out_for_delivery` entry older than 5 seconds. This is how workers recover if the master hasn't acknowledged an event by indexing it back. [`src/exo/routing/event_router.py:57-64`](../../../src/exo/routing/event_router.py)

See [../architecture/event-sourcing-message-passing.md](../architecture/event-sourcing-message-passing.md) for the end-to-end event-sourcing picture this plugs into.

## Network transport

The Python `Router` is thin; the heavy lifting lives behind the `NetworkingHandle` PyO3 class. The Rust side is the libp2p swarm.

### libp2p stack

Configured in [`rust/networking/src/swarm.rs:149-169`](../../../rust/networking/src/swarm.rs):

- **Transport** — TCP with `nodelay(true)`, upgraded through a **pnet preshared-key layer** (SHA-3-256 of `"exo_discovery_network"` XOR the version string, so nodes on different `NETWORK_VERSION`s can't interconnect), then Noise authentication, then Yamux multiplexing. [`rust/networking/src/swarm.rs:171-242`](../../../rust/networking/src/swarm.rs)
- **Network version** — hardcoded `b"v0.0.1"`, overridable via `EXO_LIBP2P_NAMESPACE`. [`rust/networking/src/swarm.rs:17-18`](../../../rust/networking/src/swarm.rs)
- **Listen address** — `/ip4/0.0.0.0/tcp/{listen_port}`; `0` lets the OS pick. [`rust/networking/src/swarm.rs:167`](../../../rust/networking/src/swarm.rs)
- **Behaviour** — a `NetworkBehaviour`-derived composite of `discovery::Behaviour` and `gossipsub::Behaviour`. [`rust/networking/src/swarm.rs:245-286`](../../../rust/networking/src/swarm.rs)

### Gossipsub configuration

- `MessageAuthenticity::Signed(keypair)` — every message is signed with the node's Ed25519 key
- `max_transmit_size = 8 * 1024 * 1024` (8 MiB)
- `ValidationMode::Strict` — message IDs are computed by gossipsub automatically

See [`rust/networking/src/swarm.rs:270-285`](../../../rust/networking/src/swarm.rs). Note the Python side warns at 1 MiB and the Rust limit is 8 MiB — a payload between those sizes triggers a warning but still goes through. Beyond 8 MiB, publish fails with `MessageTooLarge`.

### Discovery

`discovery::Behaviour` wraps mDNS + libp2p ping plus a manual retry loop: [`rust/networking/src/discovery.rs:103-208`](../../../rust/networking/src/discovery.rs)

- **mDNS** — `MDNS_QUERY_INTERVAL = 1500 s`, `MDNS_RECORD_TTL = 2500 s`, IPv4 only (IPv6 is commented out because of a TCP+mDNS interaction). [`rust/networking/src/discovery.rs:32-66`](../../../rust/networking/src/discovery.rs)
- **Ping** — 2.5s timeout, 2.5s interval; ping failure immediately closes the connection via `close_connection`. [`rust/networking/src/discovery.rs:34-35`, `68-74`, `343-348`](../../../rust/networking/src/discovery.rs)
- **Retry loop** — every `RETRY_CONNECT_INTERVAL = 5 s` the behaviour re-dials every known mDNS peer (safe — already-connected dials fail cheaply) and every bootstrap peer. This is the fallback for environments without mDNS. [`rust/networking/src/discovery.rs:24`, `367-380`](../../../rust/networking/src/discovery.rs)
- **Connection tracking** — only true listening connections/disconnections surface as `Event::ConnectionEstablished` / `Event::ConnectionClosed`, with peer ID and remote IP/port. [`rust/networking/src/discovery.rs:78-92`, `175-207`, `283-316`](../../../rust/networking/src/discovery.rs)

See [rust-networking.md](rust-networking.md) for a deeper dive.

### Swarm command protocol

The Rust swarm is driven from Python via two mpsc channels: `ToSwarm` (Python → swarm) and the stream yielded by `Swarm::into_stream()` (swarm → Python). [`rust/networking/src/swarm.rs:22-80`](../../../rust/networking/src/swarm.rs)

`ToSwarm` has three variants, each carrying a `oneshot::Sender` for the result: [`rust/networking/src/swarm.rs:22-36`](../../../rust/networking/src/swarm.rs)

- `Subscribe { topic, result_sender }`
- `Unsubscribe { topic, result_sender }`
- `Publish { topic, data, result_sender }`

`FromSwarm` has three variants (note these are the Rust variants — the Python-facing `PyFromSwarm` collapses `Discovered` + `Expired` into one `Connection` variant): [`rust/networking/src/swarm.rs:37-49`](../../../rust/networking/src/swarm.rs)

- `Message { from, topic, data }`
- `Discovered { peer_id }`
- `Expired { peer_id }`

## FFI layer

The PyO3 bindings live in `rust/exo_pyo3_bindings/`. See [rust-pyo3-bindings.md](rust-pyo3-bindings.md) for full coverage.

### `NetworkingHandle`

Defined at [`rust/exo_pyo3_bindings/src/networking.rs:132-205`](../../../rust/exo_pyo3_bindings/src/networking.rs). Constructor:

- Creates an `mpsc::channel(1024)` (`MPSC_CHANNEL_SIZE` is hardcoded at [`rust/exo_pyo3_bindings/src/lib.rs:20`](../../../rust/exo_pyo3_bindings/src/lib.rs))
- Clones the keypair out of the `PyKeypair`, enters the tokio runtime, calls `create_swarm`, and boxes the resulting stream under an `Arc<Mutex<...>>`. [`rust/exo_pyo3_bindings/src/networking.rs:182-205`](../../../rust/exo_pyo3_bindings/src/networking.rs)

Methods exposed to Python:

- `recv() -> PyFromSwarm` — awaitable; errors with `PyConnectionError` if the channel closes or if called concurrently. [`rust/exo_pyo3_bindings/src/networking.rs:207-219`](../../../rust/exo_pyo3_bindings/src/networking.rs)
- `gossipsub_subscribe(topic) -> bool` — returns `False` if we were already subscribed. [`rust/exo_pyo3_bindings/src/networking.rs:223-243`](../../../rust/exo_pyo3_bindings/src/networking.rs)
- `gossipsub_unsubscribe(topic) -> bool` — returns `True` if we were subscribed. [`rust/exo_pyo3_bindings/src/networking.rs:245-264`](../../../rust/exo_pyo3_bindings/src/networking.rs)
- `gossipsub_publish(topic, data) -> None` — maps libp2p `PublishError` variants to the three typed Python exceptions. [`rust/exo_pyo3_bindings/src/networking.rs:266-297`](../../../rust/exo_pyo3_bindings/src/networking.rs)

All `.await`-ing methods wrap their futures in `.allow_threads_py()` so the GIL is released while blocked. [`rust/exo_pyo3_bindings/src/networking.rs:176-178`](../../../rust/exo_pyo3_bindings/src/networking.rs)

### `Keypair`

Ed25519-only wrapper with `generate`, `from_bytes`, `to_bytes`, and `to_node_id` (base58 PeerId string). [`rust/exo_pyo3_bindings/src/ident.rs:1-47`](../../../rust/exo_pyo3_bindings/src/ident.rs)

### `PyFromSwarm` shape

The Python-facing enum collapses the Rust `FromSwarm::Discovered` + `Expired` variants into a single `Connection { peer_id, connected }` variant. `Message` keeps origin (base58 PeerId), topic string, and payload as `PyBytes`. [`rust/exo_pyo3_bindings/src/networking.rs:140-171`](../../../rust/exo_pyo3_bindings/src/networking.rs)

### `ConnectionMessage`

The Python wrapper over `PyFromSwarm.Connection` — a `CamelCaseModel` with `node_id: NodeId` and `connected: bool`, constructed via `ConnectionMessage.from_update`. [`src/exo/routing/connection_message.py:1-15`](../../../src/exo/routing/connection_message.py)

## Serialization

There is only one serialization format in the routing layer: **pydantic's JSON via `model_dump_json()` + UTF-8 encoding**. Defined once on `TypedTopic` at [`src/exo/routing/topics.py:32-37`](../../../src/exo/routing/topics.py) and reused on every topic.

This differs from the msgpack format used by the master's on-disk event log (see [master.md](master.md)); JSON is the wire format, msgpack is the durable format.

`CamelCaseModel` is defined in `src/exo/utils/pydantic_ext.py` — it's a strict, frozen pydantic `BaseModel` with camelCase field aliasing for wire compatibility. Every topic payload type must extend it. [`src/exo/routing/topics.py:11`](../../../src/exo/routing/topics.py)

## Gotchas

1. **No self-loopback suppression.** `TopicRouter.publish` has no notion of which subscriber originated a message, so a component holding both a sender and a receiver for the same topic will receive its own publishes. The file has an explicit comment flagging this as a current limitation, noted as relevant to election. [`src/exo/routing/router.py:37-40`](../../../src/exo/routing/router.py)

2. **`Minimal` fanout is binary, not selective.** `Minimal` means "publish to network only if `len(self.senders) == 0`". A single local subscriber suppresses network fanout entirely — partial subscription sets are not supported. [`src/exo/routing/router.py:60-65`](../../../src/exo/routing/router.py)

3. **The first registered topic claims the outbound sender.** `_tmp_networking_sender` is set to `None` after the first `register_topic` call; every subsequent topic clones a fresh sender from the receiver. Registering zero topics before `run()` means the outbound publish loop will still run but have nothing feeding it. [`src/exo/routing/router.py:140-147`](../../../src/exo/routing/router.py)

4. **Dropped messages are warned, not retried.** `_networking_publish` silently swallows `NoPeersSubscribedToTopicError` and logs (but does not retry) `AllQueuesFullError` and `MessageTooLargeError`. Retry semantics, where they exist, live one layer up in `EventRouter._simple_retry`. [`src/exo/routing/router.py:250-257`](../../../src/exo/routing/router.py), [`src/exo/routing/event_router.py:57-64`](../../../src/exo/routing/event_router.py)

5. **Two byte-size limits, not one.** The Python layer warns at 1 MiB (`1024 * 1024`) but the Rust `max_transmit_size` is 8 MiB. Payloads between those sizes publish successfully but produce a warning. [`src/exo/routing/router.py:245-248`](../../../src/exo/routing/router.py), [`rust/networking/src/swarm.rs:279`](../../../rust/networking/src/swarm.rs)

6. **`CONNECTION_MESSAGES` bypasses serialization.** Its payload comes from the Rust swarm and is constructed directly in `_networking_recv`; its `publish_policy = Never` ensures it is never pushed to the outbound channel where `serialize` would run. If someone accidentally flipped the policy, the message would still round-trip through `model_dump_json` cleanly, but every node would then rebroadcast its own peer-discovery events. [`src/exo/routing/topics.py:46-48`](../../../src/exo/routing/topics.py), [`src/exo/routing/router.py:220-229`](../../../src/exo/routing/router.py)

7. **Topic strings, not the `TypedTopic` object, key the router map.** `register_topic` stores under `topic.topic` (the string), and `sender`/`receiver` assert both `router.topic == topic` and `router.topic.model_type == topic.model_type`. Two `TypedTopic` instances with the same string name but different model types would collide at registration time rather than at use time. [`src/exo/routing/router.py:147`, `151-170`](../../../src/exo/routing/router.py)

8. **Session filtering tolerates cluster-convergence drift.** `EventRouter._run_ext_in` accepts events where `event.origin == event.session.master_node_id`, not where the session matches the node's own current session. This is deliberate — during master election the cluster may have transient sessions — and is called out in a comment. [`src/exo/routing/event_router.py:99-110`](../../../src/exo/routing/event_router.py)

9. **NACK backoff is attempt-counter based but can re-fire before the previous attempt lands.** The cancel scope is cleared only in the `finally` block after the sleep+send; a gap that persists across multiple inbound events produces one NACK per gap-spanning event until the first succeeds. Protection is the `self._nack_cancel_scope is None or cancel_called` guard. [`src/exo/routing/event_router.py:117-128`, `139-166`](../../../src/exo/routing/event_router.py)

10. **`get_node_id_keypair` regenerates silently on parse errors.** An invalid `node_id.keypair` file is logged at `warning` level and then overwritten with a fresh keypair — the old `NodeId` is lost. Cross-process safety is via `FileLock` on `<path>.lock`. [`src/exo/routing/router.py:286-305`](../../../src/exo/routing/router.py)

## Sources

- `/Users/leozealous/exo/src/exo/routing/__init__.py`
- `/Users/leozealous/exo/src/exo/routing/topics.py`
- `/Users/leozealous/exo/src/exo/routing/router.py`
- `/Users/leozealous/exo/src/exo/routing/event_router.py`
- `/Users/leozealous/exo/src/exo/routing/connection_message.py`
- `/Users/leozealous/exo/rust/networking/src/lib.rs`
- `/Users/leozealous/exo/rust/networking/src/swarm.rs`
- `/Users/leozealous/exo/rust/networking/src/discovery.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/lib.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/networking.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/ident.rs`
- `/Users/leozealous/exo/src/exo/shared/constants.py` (for `EXO_NODE_ID_KEYPAIR`)

Cross-references: [master.md](master.md) · [worker.md](worker.md) · [rust-networking.md](rust-networking.md) · [rust-pyo3-bindings.md](rust-pyo3-bindings.md) · [../architecture/event-sourcing-message-passing.md](../architecture/event-sourcing-message-passing.md)

Last indexed: 2026-04-21
