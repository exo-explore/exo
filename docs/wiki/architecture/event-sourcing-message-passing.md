# Event Sourcing & Message Passing

> exo uses an *Event Sourcing* architecture, and Erlang-style *message passing*. To facilitate this, we've written a channel library extending anyio channels with inspiration from `tokio::sync::mpsc`. Each logical module — designed to be functional independently of the others — communicates with the rest of the system by sending messages on topics.
>
> — `docs/architecture.md:1-5`

This page extends that seed with concrete citations into the code that realises it. It is the single most important architectural concept in the codebase: every other component (Master, Worker, Runner, API, Election) is defined by which topics it reads and writes, and by which events it produces or folds into `State`.

See also: [`../components/master.md`](../components/master.md), [`../components/worker.md`](../components/worker.md), [`../components/shared.md`](../components/shared.md), [`module-boundaries.md`](./module-boundaries.md), [`data-flow.md`](./data-flow.md).

---

## Why event sourcing

The rule of thumb: **`Event`s are past tense, `Command`s are imperative** (`docs/architecture.md:80`).

- An **Event** is an immutable record of something that already happened. "This node is using 300GB of RAM" is an event. Events are folded into a `State` object via a pure `apply()` function (`src/exo/shared/apply.py:58-104`). Events SHOULD NOT cause side effects on their own — they're facts, not instructions (`docs/architecture.md:80`).
- A **Command** is an imperative request to perform an action: "place this model", "give me a copy of the event log", "cancel this task". The Worker's `Task`s are also commands (`docs/architecture.md:80`). Commands are consumed by the Master, combined with current `State`, and produce new events.

Event sourcing buys exo several properties that matter for a distributed inference cluster on unreliable home networks:

1. **Fault tolerance through replay.** The Master persists the ordered event log to disk (`src/exo/master/main.py:92`, via `DiskEventLog(EXO_EVENT_LOG_DIR / "master")`). Any node that rejoins can request missing events via `RequestEventLog` (`src/exo/master/main.py:350-358`) and reconstruct the same `State` everyone else has.
2. **Single-writer ordering.** Only the Master writes to `GLOBAL_EVENTS`. Workers write candidate events to `LOCAL_EVENTS`; the Master orders them and re-broadcasts as `IndexedEvent`s. This is enforced structurally — see [§ Single-writer discipline](#single-writer-discipline).
3. **Auditability.** Every state mutation is traceable to an indexed event. `apply()` asserts sequential delivery (`src/exo/shared/apply.py:98-102`: `assert state.last_event_applied_idx == event.idx - 1`) so gaps are detected immediately.
4. **Centralized ACKing.** Quoting the seed: "event sourcing […] lets us centralize faulty connections and message ACKing" (`docs/architecture.md:78`). A Worker knows its event was accepted when it sees the `IndexedEvent` come back on `GLOBAL_EVENTS` with an index.
5. **Commands vs. events as a design forcing function.** When the team finds themselves reaching for a side effect inside an event handler, the seed acknowledges this as a "crack" — `_plan` on the Master (`src/exo/master/main.py:364-384`) is annotated as such. The distinction keeps pressure on authors to name their imperative vs. observational intents.

---

## Message passing primitive

The channel library lives at `src/exo/utils/channels.py`. It wraps `anyio.streams.memory` with MPSC-inspired ergonomics.

### `channel[T]` — in-process async

`channel[T](max_buffer_size)` returns a `(Sender[T], Receiver[T])` pair backed by `anyio`'s memory streams (`src/exo/utils/channels.py:285-292`). Both halves are clonable:

- `Sender.clone()` / `Sender.clone_receiver()` (`src/exo/utils/channels.py:29-38`)
- `Receiver.clone()` / `Receiver.clone_sender()` (`src/exo/utils/channels.py:42-51`)

This gives you multi-producer/single-consumer or the reverse — the router uses it for fan-out by cloning a `Sender` per subscriber (`src/exo/routing/router.py:96-97`, `_sender.clone()`), and for fan-in by cloning the networking `Sender` back into the receiver's state (`src/exo/routing/router.py:141-147`).

`Receiver` also has convenience methods for batching:

- `Receiver.collect()` drains all currently-available items non-blockingly (`src/exo/utils/channels.py:53-62`).
- `Receiver.receive_at_least(n)` blocks for at least one, then opportunistically batches up to `n` (`src/exo/utils/channels.py:64-71`).

### `mp_channel[T]` — cross-process sync

`mp_channel[T]` is the inter-process variant, backed by `multiprocessing.Queue` (`src/exo/utils/channels.py:295-309`). It mimics the anyio shape but is **synchronous by default**; an async bridge exists via `send_async` / `receive_async` which offloads to a thread:

```python
async def send_async(self, item: T) -> None:
    await to_thread.run_sync(
        self.send, item, limiter=CapacityLimiter(1), abandon_on_cancel=True
    )
```
(`src/exo/utils/channels.py:128-131`, mirrored in `MpReceiver.receive_async` at `:207-210`.)

`mp_channel` trades clonability for simplicity: "none of the clone methods are implemented for simplicity, for now" (`src/exo/utils/channels.py:102-104`). It also adds a `join()` method that blocks until the queue's background thread drains — required for clean shutdown across the process boundary (`src/exo/utils/channels.py:141-146`, `:220-225`).

### Closing semantics

Both channel flavours raise `ClosedResourceError` on send-after-close and `EndOfStream` on receive-after-close. The `mp_channel` variant uses a sentinel `_MpEndOfStream` object pushed on close (`src/exo/utils/channels.py:77-78`, `:133-138`) so the receiver-side blocking `get()` wakes up.

### Backpressure

`channel[T](max_buffer_size=...)` accepts either `math.inf` (unbounded) or an integer (bounded). The anyio memory stream blocks senders when the buffer is full — this is the natural backpressure signal. `mp_channel` also accepts `math.inf` or a positive integer; it explicitly rejects `0` because "0-sized buffers are not supported by multiprocessing" (`src/exo/utils/channels.py:300-307`).

Most current call sites default to `inf` (e.g. `TopicRouter` at `src/exo/routing/router.py:47`, `channel[T]()` at `:50`), which means **the system is optimistic about consumers keeping up**. When a consumer stalls, messages accumulate in memory. See [§ Gotchas](#gotchas) for the implication.

---

## Topics

A **topic** is a typed pub/sub channel with a publish policy. Topics are defined as `TypedTopic[T]` instances in `src/exo/routing/topics.py`:

```python
@dataclass
class TypedTopic[T: CamelCaseModel]:
    topic: str
    publish_policy: PublishPolicy
    model_type: type[T]
```
(`src/exo/routing/topics.py:23-37`)

There are six in-use topics. Note: `docs/architecture.md:53` says "5 topics" — that predates `DOWNLOAD_COMMANDS`, which was added as a seventh-style channel (though technically six if you exclude the seed's older count). `CONNECTION_MESSAGES` has `PublishPolicy.Never` (it's a local-only firehose from the Rust networking layer), so if you count wire-visible topics you're back to five.

| Name | Model type | Publish policy | Flow | Purpose |
|---|---|---|---|---|
| `GLOBAL_EVENTS` | `GlobalForwarderEvent` | `Always` | Master → All nodes | Master-ordered `IndexedEvent`s that every node folds into `State` (`src/exo/routing/topics.py:40`, seed `docs/architecture.md:63-65`). |
| `LOCAL_EVENTS` | `LocalForwarderEvent` | `Always` | Any node → Master | Candidate events from workers (resource info, task status, download progress) awaiting indexing (`src/exo/routing/topics.py:41`, seed `docs/architecture.md:59-61`). |
| `COMMANDS` | `ForwarderCommand` | `Always` | API/Worker → Master | Imperative requests: placement, catchup, task creation, cancellation (`src/exo/routing/topics.py:42`, seed `docs/architecture.md:55-57`). |
| `ELECTION_MESSAGES` | `ElectionMessage` | `Always` | Any node ↔ Any node | Bully-style master election negotiation before the cluster is established (`src/exo/routing/topics.py:43-45`, seed `docs/architecture.md:67-69`). |
| `CONNECTION_MESSAGES` | `ConnectionMessage` | `Never` | Rust networking → Local | mDNS-discovered hardware connections surfaced from libp2p (`src/exo/routing/topics.py:46-48`, seed `docs/architecture.md:71-73`). Never gossipped; strictly local fan-out. |
| `DOWNLOAD_COMMANDS` | `ForwarderDownloadCommand` | `Always` | Worker/Master → Worker | Download lifecycle commands (`StartDownload`, cancellation) routed separately from generic `COMMANDS` (`src/exo/routing/topics.py:49-51`). |

`PublishPolicy` itself has three variants (`src/exo/routing/topics.py:14-21`):

- `Never` — local-only, never hits libp2p.
- `Minimal` — only publish to the network if there's no local receiver for this type of message.
- `Always` — always publish, even if a local subscriber also exists.

---

## Single-writer discipline

The Master is the **sole writer** to `GLOBAL_EVENTS`. This is the load-bearing invariant of the whole event-sourcing design.

The Master's ingest path, `_event_processor`, reads `LOCAL_EVENTS` and threads every event through a deduplicating-and-ordering `MultiSourceBuffer` before indexing (`src/exo/master/main.py:386-411`):

```python
async def _event_processor(self) -> None:
    with self.local_event_receiver as local_events:
        async for local_event in local_events:
            if local_event.session != self.session_id:
                continue
            self._multi_buffer.ingest(
                local_event.origin_idx,
                local_event.event,
                local_event.origin,
            )
            for event in self._multi_buffer.drain():
                ...
                indexed = IndexedEvent(event=event, idx=len(self._event_log))
                self.state = apply(self.state, indexed)
                ...
                self._event_log.append(event)
                await self._send_event(indexed)
```
(`src/exo/master/main.py:386-411`)

Three things happen in strict sequence per event:

1. **Index assignment.** `idx = len(self._event_log)` — monotonic, gap-free. `apply()` on the consuming side asserts this holds (`src/exo/shared/apply.py:98-102`).
2. **Durable append.** `self._event_log.append(event)` — disk persistence, for replay and catchup.
3. **Broadcast.** `_send_event(indexed)` publishes `GlobalForwarderEvent(origin=node_id, origin_idx=idx, session=session_id, event=event)` to `GLOBAL_EVENTS` (`src/exo/master/main.py:413-423`).

Parallel to the event processor, the Master's `_command_processor` converts commands into events (`src/exo/master/main.py:117-363`). For instance, a `TextGeneration` command walks the existing `state.instances`, picks the least-loaded instance, and appends a `TaskCreated` event (`src/exo/master/main.py:129-170`). The resulting events go through the same `event_sender` channel that `_event_processor` drains — meaning **commands produce events that are ordered alongside worker-generated events** and pass through the same indexing bottleneck.

Workers never skip this bottleneck. Every worker state mutation is an event sent upstream (`src/exo/worker/main.py:108-114` sends `NodeGatheredInfo`, `:162` sends `TaskCreated`, etc.), then received back indexed via `event_receiver` and applied locally (`src/exo/worker/main.py:119-123`):

```python
async def _event_applier(self):
    with self.event_receiver as events:
        async for event in events:
            self.state = apply(self.state, event=event)
            event = event.event
            ...
```

Because only the Master indexes, there is exactly one total order of events — the one etched in its `DiskEventLog`. Split-brain is handled out-of-band via the election system and `session_id` filtering (`src/exo/master/main.py:389-391`: events from the wrong session are discarded).

---

## Fault tolerance through isolation

Event sourcing solves "what's the true state of the cluster"; process isolation solves "a bad inference job doesn't kill the cluster". The Runner is the primary fault boundary.

`RunnerSupervisor.create` (`src/exo/worker/runner/runner_supervisor.py:72-109`) wires three `mp_channel` pairs — events upward, tasks downward, cancellations downward — and spawns a daemon subprocess:

```python
ev_send, ev_recv = mp_channel[Event]()
task_sender, task_recv = mp_channel[Task]()
cancel_sender, cancel_recv = mp_channel[TaskId]()

runner_process = mp.Process(
    target=entrypoint,
    args=(bound_instance, ev_send, task_recv, cancel_recv, logger),
    daemon=True,
)
```
(`src/exo/worker/runner/runner_supervisor.py:80-94`)

If the runner dies — OOM, CUDA panic, segfault — the parent Worker sees it via:

- `_watch_runner` polling `runner_process.is_alive()` every 5s (`src/exo/worker/runner/runner_supervisor.py:227-232`).
- `_forward_events` catching `ClosedResourceError`/`BrokenResourceError` on the event receiver (`src/exo/worker/runner/runner_supervisor.py:215-216`).

Either path funnels into `_check_runner`, which synthesizes `RunnerStatusUpdated(RunnerFailed(...))` and `ChunkGenerated(ErrorChunk(...))` events for in-progress tasks (`src/exo/worker/runner/runner_supervisor.py:234-288`). **The failure is converted into events** and flows through the same `event_sender` as everything else. The cluster observes the failure through `State` transitions.

Supervision also uses anyio's structured concurrency. The Worker (`src/exo/worker/main.py:88-102`) runs all its loops inside `async with self._tg as tg:` — if any fail, the others are cancelled. The Master does the same (`src/exo/master/main.py:99-108`). The `TaskGroup` abstraction is exo's `utils/task_group.py` wrapper around anyio's; the Runner supervisor uses it too (`src/exo/worker/runner/runner_supervisor.py:111-115`).

Shutdown ordering matters. The Master closes its senders in the `finally` block so downstream consumers see `EndOfStream` and drain cleanly (`src/exo/master/main.py:104-108`); the Runner supervisor suppresses `ClosedResourceError` during its own shutdown to tolerate already-closed pipes (`src/exo/worker/runner/runner_supervisor.py:117-131`).

---

## Message routing and publishing

The `Router` (`src/exo/routing/router.py:106-270`) bridges in-process `channel[T]` pairs with libp2p gossipsub running in Rust. Its moving parts:

### `TopicRouter[T]`

One per registered topic (`src/exo/routing/router.py:41-103`). Structurally:

- Holds a set of `senders: set[Sender[T]]` — one per subscriber obtained via `Router.receiver(topic)` (`src/exo/routing/router.py:159-170`).
- Owns one internal `(_sender, receiver)` pair. `new_sender()` clones `_sender` for publishers (`src/exo/routing/router.py:96-97`).
- In its `run()` loop (`src/exo/routing/router.py:55-69`): pulls an item from the receiver, maybe forwards to the network based on `PublishPolicy`, then fans out to every local subscriber sender via `publish()`.

The fan-out logic handles dead subscribers: if `sender.send()` raises `ClosedResourceError`/`BrokenResourceError`, that sender is removed from the set (`src/exo/routing/router.py:85-91`). No garbage collection of stale subscribers is needed — they clear themselves on first failed delivery.

### `Router`

Owns all topic routers (`src/exo/routing/router.py:120-138`), a single networking `(send, recv)` pair (`:127`), and a `TaskGroup`. `register_topic` wires a new `TopicRouter` to share the networking sender (`src/exo/routing/router.py:140-149`) — either claiming the original sender once, or cloning from the receiver's state thereafter (`:141-145`).

Its `run()` method (`src/exo/routing/router.py:172-189`) starts every topic router, plus two networking coroutines:

- `_networking_recv` (`src/exo/routing/router.py:203-238`) — pulls messages from libp2p via Rust bindings, matches on `PyFromSwarm` variants (message vs. connection update), and calls `router.publish_bytes(data)` which deserializes and fans out locally.
- `_networking_publish` (`src/exo/routing/router.py:240-257`) — pulls from the shared networking receiver and calls `self._net.gossipsub_publish(topic, data)`. It handles three expected failure modes inline: no peers subscribed (silently skip), all peer queues full (warn and drop), message too large (warn and drop). These are not fatal — they're the shape of an unreliable network.

### `PublishPolicy` in action

Inside `TopicRouter.run` (`src/exo/routing/router.py:55-69`):

```python
async for item in items:
    if (
        len(self.senders) == 0
        and self.topic.publish_policy is PublishPolicy.Minimal
    ):
        await self._send_out(item)
        continue
    if self.topic.publish_policy is PublishPolicy.Always:
        await self._send_out(item)
    await self.publish(item)
```

`Always` fans out to both the network and local subscribers — this is the common case for `GLOBAL_EVENTS`, `LOCAL_EVENTS`, `COMMANDS`, `ELECTION_MESSAGES`, `DOWNLOAD_COMMANDS`. `Minimal` is conceptually a "fallback to network only if no one local is listening" optimization — not currently used by any in-repo topic. `Never` (used by `CONNECTION_MESSAGES`, `src/exo/routing/topics.py:46-48`) short-circuits the network entirely.

---

## Gotchas

### Don't block in handlers

Every `async for` consumer in this codebase shares the same event loop with its peers. Blocking work — CPU-heavy computation, synchronous I/O, pre-anyio blocking sockets — stalls every other task in the same TaskGroup. When you must call blocking code, wrap it in `to_thread.run_sync` (that's exactly what `mp_channel.send_async`/`receive_async` do at `src/exo/utils/channels.py:128-131` and `:207-210`).

Consequence for authors: if you add a new event handler, do your work in small `await`able slices or delegate to a thread/process. The `_event_applier` (`src/exo/worker/main.py:119-135`) keeps per-event work minimal: apply to state, maybe buffer an input chunk. That's it.

### Channel lifecycle — always `close()`

`Sender.close()` is what wakes up a blocked `Receiver` with `EndOfStream`. Skip it and consumers hang forever. This is why:

- `Master.run()` explicitly closes three senders and one receiver in `finally` (`src/exo/master/main.py:104-108`).
- `Worker.run()` closes its three senders in `finally` (`src/exo/worker/main.py:95-102`).
- `TopicRouter.shutdown()` closes every subscriber sender plus its own internal pair (`src/exo/routing/router.py:71-77`).

Pattern to copy: use the `with receiver as stream:` context-manager form (`src/exo/utils/channels.py:73-74`, `:248-257`) — it closes on exit.

### `mp_channel` is sync by default

Calling `MpSender.send(item)` from an async function **blocks the event loop** if the queue is full. Use `send_async` / `receive_async` for the async path; they offload to a worker thread (`src/exo/utils/channels.py:128-131`, `:207-210`). The `RunnerSupervisor` is careful about this — it uses `send_async` for task dispatch (`src/exo/worker/runner/runner_supervisor.py:163`) and cancellation (`:178`).

Also: `mp_channel`'s clone methods are deliberately unimplemented (`src/exo/utils/channels.py:102-104`). If you need multiple producers/consumers across the process boundary, you currently need multiple channels.

### Broadcast feedback

From `src/exo/routing/router.py:37-40`:

> A significant current limitation of the `TopicRouter` is that it is not capable of preventing feedback, as it does not ask for a system id so cannot tell which message is coming/going to which system. This is currently only relevant for Election.

`TopicRouter.publish` also notes:

> NB: this sends to ALL receivers, potentially including receivers held by the object doing the sending. You should handle your own output if you hold a sender + receiver pair.

(`src/exo/routing/router.py:80-84`)

Translation: if a system publishes on a topic it also subscribes to, it will receive its own message back. Handle self-echoes explicitly — the Master filters events by `session_id` (`src/exo/master/main.py:389-391`) and the `MultiSourceBuffer` tags events by `origin` to deduplicate.

### Event ordering is asserted, not merely expected

`apply()` does not just hope for monotonic indices — it asserts (`src/exo/shared/apply.py:98-102`):

```python
assert state.last_event_applied_idx == event.idx - 1
```

A gap crashes the worker. This is intentional: silent divergence is worse than a loud crash. If you see this assert fire, the catchup path (`RequestEventLog` command, `src/exo/master/main.py:350-358`) is how a lagging node rebuilds — but the current call sites for that are sparse; adding more is discussed in the seed as "more things could be commands" (`src/exo/master/main.py:364`).

### "Events SHOULD never cause side effects on their own"

Quoted from the seed (`docs/architecture.md:80`). The `_plan` loop on the Master (`src/exo/master/main.py:364-384`) periodically emits `InstanceDeleted` and `NodeTimedOut` events based on wall-clock time — which is arguably a side effect from the perspective of any node's apply pipeline. The code comments this as "cracks showing in our event sourcing architecture". Authors extending the system should prefer turning such polling into explicit commands rather than synthetic events.

### Purity and data flow

A significant goal of the current design is to make data flow explicit: classes are either pure data (`CamelCaseModel`s, `TaggedModel`s for unions) or active `System`s (Erlang-style actors), with transformations being referentially transparent — destructure and construct, don't mutate (`docs/architecture.md:82-84`). `apply()` exemplifies this: every helper returns a new `State` via `state.model_copy(update=...)` (e.g. `src/exo/shared/apply.py:130`, `:135`, `:142`, `:159`, `:170`, `:180`, `:187`, `:195`, `:200`, `:248-262`) — never an in-place mutation.

---

## Cross-references

- [`../components/master.md`](../components/master.md) — single-writer details, `_command_processor`, `_event_processor`, event log.
- [`../components/worker.md`](../components/worker.md) — `_event_applier`, `plan_step`, runner supervision.
- [`../components/shared.md`](../components/shared.md) — `State`, `Event`, `Command`, `apply()` type surface.
- [`module-boundaries.md`](./module-boundaries.md) — which system owns which topic.
- [`data-flow.md`](./data-flow.md) — end-to-end traces of a request through the topics.

---
Sources: docs/architecture.md:1-84, src/exo/utils/channels.py:28-309, src/exo/routing/topics.py:14-51, src/exo/routing/router.py:37-270, src/exo/master/main.py:68-449, src/exo/worker/main.py:52-339, src/exo/shared/apply.py:58-104, src/exo/worker/runner/runner_supervisor.py:52-288
Last indexed: 2026-04-21 (commit c0d5bf92)
