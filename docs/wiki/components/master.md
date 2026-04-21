# Master Component

## Overview

The Master is the single-writer coordinator of the exo cluster. It executes **placement decisions** (which workers run which models) and **orders all events** through a centralized log, ensuring cluster consistency. Exactly one Master runs per cluster election cycle, elected via the distributed bully algorithm. [`src/exo/master/main.py:68-96`][master-class]

The Master runs on the elected master node alongside the Worker, API, and Election systems. It receives commands from the API and workers via pub/sub, processes them synchronously, indexes events from workers, and broadcasts the ordered event log back to all nodes so they can apply the same state transformations.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Core Master class; processes commands and events; maintains single-writer discipline [`main.py:68-95`][master-class] |
| `placement.py` | Decides which cycle (node group) runs a model; validates topology/memory constraints [`placement.py:63-140`][place-instance] |
| `placement_utils.py` | Layer allocation, shard assignment math, RDMA/ring topology discovery [`placement_utils.py:47-273`][placement-utils-main] |
| `event_log.py` | Append-only disk log (msgpack + length-prefixed records) for durability [`event_log.py:62-150`][event-log] |
| `api.py` | FastAPI server; adapts OpenAI / Claude / Ollama / Responses API to internal types [`api.py:197-202`][api-class] |
| `image_store.py` | In-memory cache + disk store for generated images with TTL expiry [`image_store.py:16-65`][image-store] |

## Placement Algorithm

Placement is the process of choosing which workers will run a model for a new inference job.

**Input:** A `PlaceInstance` command with model requirements, sharding strategy, and optional node constraints.

**Process:**

1. **Find candidate cycles** (connected node groups) from the topology graph [`placement.py:71-85`][placement-cycles]
2. **Filter by memory** — reject cycles where total RAM cannot hold the model [`placement_utils.py:21-37`][memory-filter]
3. **Filter by sharding constraints:**
   - **Tensor parallel** — cycle size must divide hidden_size evenly [`placement.py:87-106`][tensor-shard]
   - **Pipeline parallel** — no constraint (each node gets 1+ layers proportionally) [`placement_utils.py:203-240`][pipeline-shard]
   - **CFG parallel** — special case for image generation; requires even world size and both last-stage nodes as neighbors [`placement_utils.py:140-200`][cfg-shard]
4. **Select smallest cycle** with sufficient memory (minimizes latency) [`placement_utils.py:40-44`][smallest-cycles]
5. **Prefer leaf nodes** (devices at network periphery, e.g., single Mac vs. data center) [`placement.py:127-139`][leaf-preference]
6. **Allocate layers proportionally** to each node's available RAM [`placement_utils.py:96-122`][layer-allocation]
7. **Build instance configuration** — instance metadata (MlxJaccl for RDMA, MlxRing for socket), host lists, coordinator IPs [`placement.py:155-200`][instance-meta]

**Output:** A new `Instance` (shard assignments + network config) added to cluster state.

Throws `ValueError` if no valid placement exists (e.g., model too large, topology disconnected, sharding incompatible).

## Single-Writer Discipline

The Master enforces event ordering via a **strict single-writer pattern** across three concurrent subsystems:

**1. Command Processor** [`main.py:117-363`][command-processor]
- Receives commands (TextGeneration, ImageGeneration, PlaceInstance, etc.) from one channel
- Processes each command **serially** (no async concurrency within this loop)
- Generates zero or more `Event`s (TaskCreated, InstanceCreated, etc.)
- Sends events to the outbound channel

**2. Event Processor** [`main.py:386-411`][event-processor]
- Receives local events from workers (NodeGatheredInfo, ChunkGenerated, etc.)
- Buffers out-of-order events from multiple workers using `MultiSourceBuffer`
- Drains ordered events, applies them to state, **appends to disk log**, broadcasts globally
- **Single-threaded loop**; processes one event at a time

**3. Plan Loop** [`main.py:365-384`][plan-loop]
- Runs every 10 seconds
- Watches for dead nodes or unconnected instance shards
- Emits cleanup events (InstanceDeleted, NodeTimedOut)

**Ordering guarantee:**
- All events written to the disk log are **indexed sequentially** [`main.py:403-410`][indexing]
- Every event is wrapped in an `IndexedEvent(idx, event)` before broadcast [`main.py:414-422`][send-event]
- Workers apply events by index, reconstructing identical state everywhere
- Commands are **not idempotent** by design (intentional: "place this model" generates a new instance each time)

**Why this works:**
- The single event_sender ensures no concurrent writes to `_event_log` [`main.py:89`][event-sender]
- `MultiSourceBuffer` guarantees order even when workers send events out-of-order [`main.py:91`][multi-buffer]
- Disk log is **append-only** with 4-byte length prefix + msgpack payload [`event_log.py:18-24`][disk-format]
- On master restart, state is replayed from the log [`main.py:92`][event-log-init]

## State It Owns

The Master maintains an immutable `State` object [`main.py:82`][state]:

- **Topology** — live node list + network connections (UDP, RDMA, etc.) [`State.topology`](../../../src/exo/shared/types/state.py)
- **Instances** — all live model deployments mapped by `InstanceId` [`State.instances`](../../../src/exo/shared/types/state.py)
  - Each Instance specifies which model + which nodes run it, shard assignments, backend config (ring/jaccl)
- **Tasks** — pending/running inference jobs mapped by `TaskId` [`State.tasks`](../../../src/exo/shared/types/state.py)
  - Each Task is bound to an Instance (model) and a Command (user request)
- **Downloads** — model weight download progress per node [`State.downloads`](../../../src/exo/shared/types/state.py)
- **Node Memory** — available RAM on each node (updated by workers) [`State.node_memory`](../../../src/exo/shared/types/state.py)
- **Node Network** — interface types (ethernet/wifi/thunderbolt) for topology-aware scheduling [`State.node_network`](../../../src/exo/shared/types/state.py)
- **Last Seen** — timestamp of most recent health check per node; triggers timeout cleanup [`main.py:378-382`][timeout-cleanup]

All updates flow through `apply(state, indexed_event)` — a pure function that returns new state [`apply` in shared/apply.py](../../../src/exo/shared/apply.py).

## Public API

### Command Topics (Master receives)

**Source:** Workers, API, clients via `/commands` pub/sub topic [`topics.py:42`][commands-topic]

| Command | Handler | Generates |
|---------|---------|-----------|
| `TextGeneration` | `_command_processor` [`main.py:129-168`][text-gen-cmd] | `TaskCreated` |
| `ImageGeneration` | `_command_processor` [`main.py:171-213`][image-gen-cmd] | `TaskCreated` |
| `PlaceInstance` | `place_instance()` [`main.py:293-303`][place-cmd] | `InstanceCreated` or `ValueError` |
| `DeleteInstance` | `delete_instance()` [`main.py:279-292`][delete-cmd] | `InstanceDeleted`, `CancelDownload` |
| `RequestEventLog` | `_command_processor` [`main.py:350-358`][request-log] | `IndexedEvent` (replay) |
| `TaskCancelled` / `TaskFinished` | `_command_processor` | `TaskStatusUpdated` |

### Event Topics (Master broadcasts)

**Destination:** All nodes via `/global_events` pub/sub [`topics.py:40`][global-events-topic]

- `IndexedEvent` — wraps any event with index + session ID so all nodes apply in order
- `TaskCreated` — new inference job spawned
- `InstanceCreated` — new model instance placed
- `InstanceDeleted` — instance removed or timed out
- `NodeTimedOut` — worker hasn't checked in for 30s
- `TracesMerged` — collected and merged distributed traces from task

### Inbound Channels

| Channel | Source | Type |
|---------|--------|------|
| `command_receiver` | Router (COMMANDS topic) | `ForwarderCommand` |
| `local_event_receiver` | Router (LOCAL_EVENTS topic) | `LocalForwarderEvent` |

### Outbound Channels

| Channel | Destination | Type |
|---------|-------------|------|
| `event_sender` | Local event_router → API/workers | `Event` |
| `global_event_sender` | Router (GLOBAL_EVENTS topic) | `GlobalForwarderEvent` |
| `download_command_sender` | Router (DOWNLOAD_COMMANDS topic) | `ForwarderDownloadCommand` |

## Gotchas

1. **Placement is not reversible.** Once a `PlaceInstance` command generates a `TaskCreated`, there's no compensation. If placement fails mid-way (e.g., a node is unreachable), the instance may be half-created. Use explicit `DeleteInstance` to clean up.

2. **Commands are serialized but not transaction-safe.** The command processor runs async loops to send events, but reads `self.state` without locking. If state is modified by another subsystem (race condition), you may see inconsistent placements. Mitigation: all writes go through the single event log, so re-apply events on restart.

3. **Event log is never pruned.** The disk log grows indefinitely. In production, archive old logs monthly or implement log rotation [`event_log.py:19-20`][log-rotation-todo].

4. **Single-node instances are forced to Pipeline + MlxRing.** If you want Tensor parallel or Jaccl on a single node, placement will reject it [`placement.py:141-144`][single-node-force].

5. **Topology cycles are recomputed on every placement.** No caching; if topology changes frequently, O(V²) cycle-finding may bottleneck. Consider memoizing `topology.get_cycles()` if topology is stable.

6. **Layer allocation assumes proportional memory distribution.** If one node has vastly more RAM than another, the allocation heuristic (largest remainder method) may give one node most layers, leaving others idle. Works well in practice; tweak `allocate_layers_proportionally()` if asymmetric clusters are common [`placement_utils.py:47-75`][largest-remainder].

7. **No preemption.** Once a task is placed on an instance, you cannot migrate it to another. Killing the node kills the task; use explicit `DeleteInstance` + `PlaceInstance` to rebalance.

## See Also

- [`worker.md`](./worker.md) — handles task execution, health checks, downloads
- [`routing.md`](./routing.md) — pub/sub message delivery, topics
- [`shared.md`](./shared.md) — `State`, `Event`, `Command` types
- [`architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) — event sourcing design rationale
- [`architecture/module-boundaries.md`](../architecture/module-boundaries.md) — Master's role in cluster architecture

---

**Sources**
- `src/exo/master/main.py` — Master class, command processor, event processor
- `src/exo/master/placement.py` — Placement algorithm and instance creation
- `src/exo/master/placement_utils.py` — Cycle filtering, shard allocation, topology utilities
- `src/exo/master/event_log.py` — Disk event log implementation
- `src/exo/master/api.py` — API server integration
- `src/exo/routing/topics.py` — Pub/sub topics and message types
- `docs/architecture.md` — System overview

**Last Indexed:** 2025-04-21 | exo commit: current
