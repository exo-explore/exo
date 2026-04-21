# Shared Types

## Overview

`src/exo/shared/` is the **central type vocabulary** of the exo cluster. Every other module — Master, Worker, Routing, API, Election — imports from here. The module has no upward dependencies: it defines pure data types, the pure `event_apply` reducer, and the `Topology` graph primitive, and that is all. This is what makes event sourcing tractable: one canonical definition of what an `Event` is, what a `Command` is, what the `State` looks like, and one single place where events become new state.

The key files:

| File | Purpose |
|------|---------|
| `types/events.py` | Discriminated union `Event` — every mutation the cluster can record [`types/events.py:140-161`](../../../src/exo/shared/types/events.py#L140-L161) |
| `types/commands.py` | Discriminated union `Command` — every intent the API/workers can issue [`types/commands.py:87-99`](../../../src/exo/shared/types/commands.py#L87-L99) |
| `types/state.py` | The global `State` model — what every node tries to converge on [`types/state.py:27-85`](../../../src/exo/shared/types/state.py#L27-L85) |
| `types/tasks.py` | `Task` union — units of work a Worker plans to run [`types/tasks.py:92-103`](../../../src/exo/shared/types/tasks.py#L92-L103) |
| `types/topology.py` | `Connection`, `RDMAConnection`, `SocketConnection` edge types [`types/topology.py:20-36`](../../../src/exo/shared/types/topology.py#L20-L36) |
| `topology.py` | Mutable `Topology` graph + `TopologySnapshot` (rustworkx-backed) [`topology.py:23-282`](../../../src/exo/shared/topology.py#L23-L282) |
| `apply.py` | `event_apply(event, state) -> state` — the single pattern-match site [`apply.py:58-94`](../../../src/exo/shared/apply.py#L58-L94) |
| `types/common.py` | Newtype IDs (`NodeId`, `SystemId`, `ModelId`, `CommandId`, `SessionId`) [`types/common.py:10-47`](../../../src/exo/shared/types/common.py#L10-L47) |
| `constants.py` | XDG paths, libp2p topic names, feature flags [`constants.py:25-104`](../../../src/exo/shared/constants.py#L25-L104) |

Every other major module depends on this surface:

- `src/exo/master/main.py` imports `apply`, `Event`, `Command`, `State`, `Task` [`master/main.py:14-62`](../../../src/exo/master/main.py#L14-L62)
- `src/exo/worker/main.py` imports `apply`, `Event`, `Command`, `State`, `Task` [`worker/main.py:9-42`](../../../src/exo/worker/main.py#L9-L42)
- `src/exo/routing/topics.py` imports `ForwarderCommand`, `GlobalForwarderEvent`, `LocalForwarderEvent` [`routing/topics.py:5-7`](../../../src/exo/routing/topics.py#L5-L7)
- `src/exo/routing/event_router.py` imports events + `SessionId`/`SystemId` [`routing/event_router.py:9-11`](../../../src/exo/routing/event_router.py#L9-L11)

See [`../architecture/module-boundaries.md`](../architecture/module-boundaries.md) for how these consumers fit together, and [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) for why this module is shaped the way it is.

## Event types

An `Event` is an **immutable, indexed, serializable fact** about something that happened. Events are the only mechanism that mutates cluster state — anywhere.

Every concrete event extends `BaseEvent`, which carries an auto-generated `EventId` and a (debug-only) master timestamp [`types/events.py:24-28`](../../../src/exo/shared/types/events.py#L24-L28). `BaseEvent` in turn extends `TaggedModel`, whose serializer wraps the payload under the class name (`{"TaskCreated": {...}}`) so the discriminated union round-trips cleanly through JSON/msgpack [`utils/pydantic_ext.py:36-51`](../../../src/exo/utils/pydantic_ext.py#L36-L51).

The full union is declared as `Event` [`types/events.py:140-161`](../../../src/exo/shared/types/events.py#L140-L161):

| Event | Fields | Effect |
|-------|--------|--------|
| `TestEvent` | (none) | Pass-through used by tests [`types/events.py:30-31`](../../../src/exo/shared/types/events.py#L30-L31) |
| `TaskCreated` | `task_id`, `task: Task` | Inserts a task into `state.tasks` [`types/events.py:34-36`](../../../src/exo/shared/types/events.py#L34-L36) |
| `TaskStatusUpdated` | `task_id`, `task_status: TaskStatus` | Transitions task status; clears error fields on non-Failed [`types/events.py:47-49`](../../../src/exo/shared/types/events.py#L47-L49) |
| `TaskFailed` | `task_id`, `error_type`, `error_message` | Records failure reason on a task [`types/events.py:52-55`](../../../src/exo/shared/types/events.py#L52-L55) |
| `TaskDeleted` | `task_id` | Removes task from `state.tasks` [`types/events.py:43-44`](../../../src/exo/shared/types/events.py#L43-L44) |
| `TaskAcknowledged` | `task_id` | Pass-through — worker tells master "I saw it" [`types/events.py:39-40`](../../../src/exo/shared/types/events.py#L39-L40) |
| `InstanceCreated` | `instance: Instance` | Adds a running model instance to `state.instances` [`types/events.py:58-65`](../../../src/exo/shared/types/events.py#L58-L65) |
| `InstanceDeleted` | `instance_id` | Removes from `state.instances` [`types/events.py:68-69`](../../../src/exo/shared/types/events.py#L68-L69) |
| `RunnerStatusUpdated` | `runner_id`, `runner_status: RunnerStatus` | Updates `state.runners`; `RunnerShutdown` removes the entry [`types/events.py:72-74`](../../../src/exo/shared/types/events.py#L72-L74) |
| `NodeTimedOut` | `node_id` | Purges all per-node state + topology edges for a dead node [`types/events.py:77-78`](../../../src/exo/shared/types/events.py#L77-L78) |
| `NodeGatheredInfo` | `node_id`, `when: str`, `info: GatheredInfo` | Updates one of 8 node info mappings depending on `info` subtype [`types/events.py:82-85`](../../../src/exo/shared/types/events.py#L82-L85) |
| `NodeDownloadProgress` | `download_progress: DownloadProgress` | Upserts a download record keyed by `(node_id, shard_metadata)` [`types/events.py:88-89`](../../../src/exo/shared/types/events.py#L88-L89) |
| `ChunkGenerated` | `command_id`, `chunk: GenerationChunk` | Pass-through — streams tokens/images to API [`types/events.py:92-94`](../../../src/exo/shared/types/events.py#L92-L94) |
| `InputChunkReceived` | `command_id`, `chunk: InputImageChunk` | Pass-through — image upload chunk for edits [`types/events.py:97-99`](../../../src/exo/shared/types/events.py#L97-L99) |
| `TopologyEdgeCreated` | `conn: Connection` | Adds an edge to `state.topology` [`types/events.py:102-103`](../../../src/exo/shared/types/events.py#L102-L103) |
| `TopologyEdgeDeleted` | `conn: Connection` | Removes an edge from `state.topology` [`types/events.py:106-107`](../../../src/exo/shared/types/events.py#L106-L107) |
| `CustomModelCardAdded` | `model_card: ModelCard` | Registers a user-added model card [`types/events.py:110-111`](../../../src/exo/shared/types/events.py#L110-L111) |
| `CustomModelCardDeleted` | `model_id` | Removes a custom model card [`types/events.py:114-115`](../../../src/exo/shared/types/events.py#L114-L115) |
| `TracesCollected` | `task_id`, `rank`, `traces` | Pass-through — per-rank trace upload [`types/events.py:128-131`](../../../src/exo/shared/types/events.py#L128-L131) |
| `TracesMerged` | `task_id`, `traces` | Pass-through — master-merged traces [`types/events.py:135-137`](../../../src/exo/shared/types/events.py#L135-L137) |

Three wrappers put events on the wire [`types/events.py:164-186`](../../../src/exo/shared/types/events.py#L164-L186):

- `IndexedEvent(idx, event)` — master-assigned monotonic index, used in-process
- `GlobalForwarderEvent(origin_idx, origin: NodeId, session: SessionId, event)` — master → workers over libp2p `GLOBAL_EVENTS`
- `LocalForwarderEvent(origin_idx, origin: SystemId, session: SessionId, event)` — worker → master over libp2p `LOCAL_EVENTS`

See [`master.md`](./master.md) for how these are ordered and broadcast, and [`routing.md`](./routing.md) for topic wiring.

## Command types

A `Command` is an **intent** — something the API or a worker wants the master to decide. Unlike events, commands are not persisted and are not idempotent; they are processed serially by the master and may generate zero or more events [`master.md`](./master.md).

Every command extends `BaseCommand` with a `CommandId` [`types/commands.py:16-17`](../../../src/exo/shared/types/commands.py#L16-L17). The full union is `Command` [`types/commands.py:87-99`](../../../src/exo/shared/types/commands.py#L87-L99):

**Inference commands:**
- `TextGeneration(task_params: TextGenerationTaskParams)` — canonical internal text-gen request [`types/commands.py:24-25`](../../../src/exo/shared/types/commands.py#L24-L25)
- `ImageGeneration(task_params: ImageGenerationTaskParams)` — text-to-image [`types/commands.py:28-29`](../../../src/exo/shared/types/commands.py#L28-L29)
- `ImageEdits(task_params: ImageEditsTaskParams)` — image-to-image [`types/commands.py:32-33`](../../../src/exo/shared/types/commands.py#L32-L33)
- `SendInputChunk(chunk: InputImageChunk)` — streams a chunk of an uploaded image; master converts to an `InputChunkReceived` event [`types/commands.py:59-62`](../../../src/exo/shared/types/commands.py#L59-L62)

**Instance management:**
- `PlaceInstance(model_card, sharding, instance_meta, min_nodes)` — "run this model somewhere"; triggers the placement algorithm [`types/commands.py:36-40`](../../../src/exo/shared/types/commands.py#L36-L40)
- `CreateInstance(instance)` — direct placement with pre-computed assignments [`types/commands.py:43-44`](../../../src/exo/shared/types/commands.py#L43-L44)
- `DeleteInstance(instance_id)` — tear down a running instance [`types/commands.py:47-48`](../../../src/exo/shared/types/commands.py#L47-L48)

**Task lifecycle:**
- `TaskCancelled(cancelled_command_id)` — abort an in-flight inference [`types/commands.py:51-52`](../../../src/exo/shared/types/commands.py#L51-L52)
- `TaskFinished(finished_command_id)` — worker signals completion [`types/commands.py:55-56`](../../../src/exo/shared/types/commands.py#L55-L56)

**Event log:**
- `RequestEventLog(since_idx)` — a reconnecting worker asks for events it missed [`types/commands.py:65-66`](../../../src/exo/shared/types/commands.py#L65-L66)

**Download commands** live in a separate union `DownloadCommand` [`types/commands.py:84`](../../../src/exo/shared/types/commands.py#L84) routed on their own topic:
- `StartDownload(target_node_id, shard_metadata)` [`types/commands.py:69-71`](../../../src/exo/shared/types/commands.py#L69-L71)
- `DeleteDownload(target_node_id, model_id)` [`types/commands.py:74-76`](../../../src/exo/shared/types/commands.py#L74-L76)
- `CancelDownload(target_node_id, model_id)` [`types/commands.py:79-81`](../../../src/exo/shared/types/commands.py#L79-L81)

Commands crossing the network are wrapped with their origin `SystemId` [`types/commands.py:102-109`](../../../src/exo/shared/types/commands.py#L102-L109):

```python
class ForwarderCommand(CamelCaseModel):
    origin: SystemId
    command: Command

class ForwarderDownloadCommand(CamelCaseModel):
    origin: SystemId
    command: DownloadCommand
```

### Task types (Worker-side)

`Task` is a separate union from `Command` — it is the **unit of work a Worker plans to execute**, carried inside `TaskCreated` events. The full union [`types/tasks.py:92-103`](../../../src/exo/shared/types/tasks.py#L92-L103):

| Task | Emitted by | Purpose |
|------|------------|---------|
| `CreateRunner(bound_instance)` | Worker | Spawn a runner subprocess [`types/tasks.py:39-41`](../../../src/exo/shared/types/tasks.py#L39-L41) |
| `DownloadModel(shard_metadata)` | Worker | Fetch shard weights [`types/tasks.py:43-45`](../../../src/exo/shared/types/tasks.py#L43-L45) |
| `LoadModel` | Worker | Load weights into runner memory [`types/tasks.py:47-48`](../../../src/exo/shared/types/tasks.py#L47-L48) |
| `ConnectToGroup` | Worker | Establish inter-runner networking [`types/tasks.py:51-52`](../../../src/exo/shared/types/tasks.py#L51-L52) |
| `StartWarmup` | Worker | Prefill dummy tokens to warm caches [`types/tasks.py:55-56`](../../../src/exo/shared/types/tasks.py#L55-L56) |
| `TextGeneration(command_id, task_params, error_type?, error_message?)` | Master | Run a chat completion [`types/tasks.py:59-65`](../../../src/exo/shared/types/tasks.py#L59-L65) |
| `ImageGeneration(...)` | Master | Run text-to-image [`types/tasks.py:72-77`](../../../src/exo/shared/types/tasks.py#L72-L77) |
| `ImageEdits(...)` | Master | Run image-to-image [`types/tasks.py:80-85`](../../../src/exo/shared/types/tasks.py#L80-L85) |
| `CancelTask(cancelled_task_id, runner_id)` | (either) | Abort an in-flight task [`types/tasks.py:67-69`](../../../src/exo/shared/types/tasks.py#L67-L69) |
| `Shutdown(runner_id)` | Worker | Tear down a runner [`types/tasks.py:88-89`](../../../src/exo/shared/types/tasks.py#L88-L89) |

Every task carries a `TaskStatus` — the state machine `Pending → Running → Complete | Failed | TimedOut | Cancelled` [`types/tasks.py:24-30`](../../../src/exo/shared/types/tasks.py#L24-L30). The sentinel `CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")` is used in cancel propagation [`types/tasks.py:21`](../../../src/exo/shared/types/tasks.py#L21).

## State

`State` is the **single global data structure every node tries to converge on** via event sourcing. It is a pydantic `CamelCaseModel` — immutable in spirit, copied on every event application via `model_copy(update={...})`. Defined at [`types/state.py:27-85`](../../../src/exo/shared/types/state.py#L27-L85):

| Field | Type | Populated by |
|-------|------|--------------|
| `instances` | `Mapping[InstanceId, Instance]` | `InstanceCreated` / `InstanceDeleted` |
| `runners` | `Mapping[RunnerId, RunnerStatus]` | `RunnerStatusUpdated` (shutdown removes) |
| `downloads` | `Mapping[NodeId, Sequence[DownloadProgress]]` | `NodeDownloadProgress` |
| `tasks` | `Mapping[TaskId, Task]` | `TaskCreated` / `TaskStatusUpdated` / `TaskFailed` / `TaskDeleted` |
| `last_seen` | `Mapping[NodeId, datetime]` | `NodeGatheredInfo` |
| `topology` | `Topology` | `NodeGatheredInfo`, `TopologyEdgeCreated/Deleted`, `NodeTimedOut` |
| `last_event_applied_idx` | `int` (≥ -1) | `apply()` itself [`apply.py:96-104`](../../../src/exo/shared/apply.py#L96-L104) |
| `node_identities` | `Mapping[NodeId, NodeIdentity]` | `NodeGatheredInfo` (MiscData / StaticNodeInformation) |
| `node_memory` | `Mapping[NodeId, MemoryUsage]` | `NodeGatheredInfo` (MacmonMetrics / MemoryUsage) |
| `node_disk` | `Mapping[NodeId, DiskUsage]` | `NodeGatheredInfo` (NodeDiskUsage) |
| `node_system` | `Mapping[NodeId, SystemPerformanceProfile]` | `NodeGatheredInfo` (MacmonMetrics) |
| `node_network` | `Mapping[NodeId, NodeNetworkInfo]` | `NodeGatheredInfo` (NodeNetworkInterfaces) |
| `node_thunderbolt` | `Mapping[NodeId, NodeThunderboltInfo]` | `NodeGatheredInfo` (MacThunderboltIdentifiers) |
| `node_thunderbolt_bridge` | `Mapping[NodeId, ThunderboltBridgeStatus]` | `NodeGatheredInfo` (ThunderboltBridgeInfo) |
| `node_rdma_ctl` | `Mapping[NodeId, NodeRdmaCtlStatus]` | `NodeGatheredInfo` (RdmaCtlStatus) |
| `thunderbolt_bridge_cycles` | `Sequence[Sequence[NodeId]]` | recomputed when TB state changes |

The per-node state is **granular on purpose** — each mapping updates at a different frequency (memory every few seconds, disk every minute, identity once at connection). Splitting them means a memory-only event doesn't invalidate disk caches [`types/state.py:51-59`](../../../src/exo/shared/types/state.py#L51-L59).

The `model_config` pins strict validation semantics [`types/state.py:35-42`](../../../src/exo/shared/types/state.py#L35-L42):

```python
model_config = ConfigDict(
    alias_generator=to_camel,
    validate_by_name=True,
    extra="forbid",
    strict=True,
    arbitrary_types_allowed=True,
)
```

`Topology` is not a pydantic model (it wraps a `rustworkx.PyDiGraph`), so `State` defines custom serializers that round-trip through the immutable `TopologySnapshot` [`types/state.py:64-84`](../../../src/exo/shared/types/state.py#L64-L84).

## Topology types

Edge types are minimal and frozen — every edge is either RDMA (direct Thunderbolt) or a plain socket. Defined at [`types/topology.py:20-36`](../../../src/exo/shared/types/topology.py#L20-L36):

```python
class RDMAConnection(FrozenModel):
    source_rdma_iface: str
    sink_rdma_iface: str

class SocketConnection(FrozenModel):
    sink_multiaddr: Multiaddr

class Connection(FrozenModel):
    source: NodeId
    sink: NodeId
    edge: RDMAConnection | SocketConnection
```

`Cycle` is a `@dataclass(frozen=True)` wrapping a list of `NodeId` — used for placement candidates [`types/topology.py:9-17`](../../../src/exo/shared/types/topology.py#L9-L17).

The `Topology` class itself lives in [`shared/topology.py:33-282`](../../../src/exo/shared/topology.py#L33-L282) and wraps `rustworkx.PyDiGraph[NodeId, SocketConnection | RDMAConnection]`. Key operations [`shared/topology.py:39-282`](../../../src/exo/shared/topology.py#L39-L282):

- `add_node` / `remove_node` / `contains_node`
- `add_connection` / `remove_connection` / `get_all_connections_between`
- `replace_all_out_rdma_connections(source, new_connections)` — used by `NodeGatheredInfo(MacThunderboltConnections)` to refresh an entire node's RDMA edges atomically [`topology.py:163-170`](../../../src/exo/shared/topology.py#L163-L170)
- `get_cycles` / `get_rdma_cycles` — simple-cycle enumeration (including singletons) for placement [`topology.py:184-217`](../../../src/exo/shared/topology.py#L184-L217)
- `get_thunderbolt_bridge_cycles(node_tb_bridge_status, node_network)` — cycles of ≥3 nodes that are all TB-bridge-enabled and directly connected via RDMA [`topology.py:244-282`](../../../src/exo/shared/topology.py#L244-L282)
- `to_snapshot` / `from_snapshot` — the JSON bridge via `TopologySnapshot` [`topology.py:23-59`](../../../src/exo/shared/topology.py#L23-L59)

Note: "Device" is not a first-class type in `shared/`. Per-node device info lives in `types/profiling.py` (`NodeIdentity`, `NodeThunderboltInfo`, `ThunderboltBridgeStatus`, etc.) and is flattened into the granular mappings on `State`.

## The apply function

`event_apply(event, state) -> state` in [`apply.py:58-94`](../../../src/exo/shared/apply.py#L58-L94) is the **single pattern-match site** for event → state transformation. There is no other place in the codebase where an event becomes a state change. That is the whole point.

```python
def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent()
            | ChunkGenerated()
            | TaskAcknowledged()
            | InputChunkReceived()
            | TracesCollected()
            | TracesMerged()
        ):  # Pass-through events that don't modify state
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeGatheredInfo():
            return apply_node_gathered_info(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)
```

Every branch is a **pure function** — it takes `(event, state)`, returns a new `State` via `state.model_copy(update={...})`, and never mutates the input. Example [`apply.py:133-135`](../../../src/exo/shared/apply.py#L133-L135):

```python
def apply_task_created(event: TaskCreated, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})
```

The wrapper `apply(state, indexed_event)` enforces **strict sequential ordering** [`apply.py:96-104`](../../../src/exo/shared/apply.py#L96-L104):

```python
def apply(state: State, event: IndexedEvent) -> State:
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(...)
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})
```

The `assert` is the correctness spine: if the master broadcasts events `[0, 1, 2, 3]` and a worker tries to apply `3` with `last_event_applied_idx == 1`, the process crashes rather than silently diverge. Out-of-order delivery must be buffered upstream (see `MultiSourceBuffer` in the Master and `EventBuffer` in Routing).

Two branches deserve a closer look because they are non-trivial:

- `apply_node_timed_out` [`apply.py:203-262`](../../../src/exo/shared/apply.py#L203-L262) purges the node from every granular mapping and removes it from the topology, then **conditionally** recomputes `thunderbolt_bridge_cycles` only if the departing node had TB bridge enabled.
- `apply_node_gathered_info` [`apply.py:265-372`](../../../src/exo/shared/apply.py#L265-L372) is itself a nested pattern match on `info: GatheredInfo` — eight cases, one per kind of profiler payload. `MacThunderboltConnections` uses the current `node_thunderbolt` mapping to translate TB domain UUIDs into RDMA edges and install them via `replace_all_out_rdma_connections` [`apply.py:325-349`](../../../src/exo/shared/apply.py#L325-L349).

Both Master and Worker call `apply()` on every incoming `IndexedEvent` — this is **why all nodes converge**. See [`master/main.py:386-411`](../../../src/exo/master/main.py) and [`worker/main.py`](../../../src/exo/worker/main.py), and [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) for the full design rationale.

## Serialization discipline

Every type in `shared/types/` is a pydantic model with consistent rules, enforced by three small base classes in [`utils/pydantic_ext.py:13-51`](../../../src/exo/utils/pydantic_ext.py#L13-L51):

```python
class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        strict=True,
    )

class FrozenModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        strict=True,
        frozen=True,
    )

class TaggedModel(CamelCaseModel):
    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        inner = handler(self)
        return {self.__class__.__name__: inner}

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v, handler):
        if isinstance(v, dict) and len(v) == 1 and cls.__name__ in v:
            return handler(v[cls.__name__])
        return handler(v)
```

This gives the codebase four non-negotiable invariants:

1. **Snake-case in Python, camelCase on the wire.** `alias_generator=to_camel` + `validate_by_name=True` means `last_event_applied_idx` serializes to `lastEventAppliedIdx` but still accepts Python-native names during construction.
2. **`extra="forbid"`.** Unknown fields are validation errors, not silently dropped. Schema drift surfaces immediately.
3. **`strict=True`.** No implicit `int → str` coercion. Type mismatches are errors.
4. **Tagged unions round-trip.** `TaggedModel` (extended by `BaseEvent`, `BaseCommand`, `BaseTask`, `BaseChunk`, `BaseDownloadProgress`, etc.) wraps payloads as `{"ClassName": {...}}`. That one trick is how `Event | Command | Task` unions survive msgpack on disk [`master/event_log.py`](../../../src/exo/master/event_log.py) and JSON on the wire without needing an explicit discriminator field.

Newtype IDs in `types/common.py` wrap `str` with a uuid4 default and a pydantic core-schema hook so they validate as strings but carry type identity [`types/common.py:10-47`](../../../src/exo/shared/types/common.py#L10-L47):

```python
class Id(str):
    def __new__(cls, value: str | None = None) -> Self:
        return super().__new__(cls, value or str(uuid4()))

    @classmethod
    def __get_pydantic_core_schema__(cls, _source, handler):
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )

class NodeId(Id): pass
class SystemId(Id): pass
class ModelId(Id): ...
class CommandId(Id): pass
```

`SessionId` is structured (`master_node_id: NodeId`, `election_clock: int`) so the `(master, clock)` pair uniquely identifies an election epoch [`types/common.py:40-42`](../../../src/exo/shared/types/common.py#L40-L42). Every `GlobalForwarderEvent` / `LocalForwarderEvent` on the wire carries its `SessionId` — events from stale sessions are rejected in routing.

## See Also

- [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) — why `event_apply` is the only mutation site
- [`../architecture/module-boundaries.md`](../architecture/module-boundaries.md) — which systems depend on this module
- [`master.md`](./master.md) — the indexer; only caller allowed to generate new event indices
- [`worker.md`](./worker.md) — the other `apply()` caller; reconstructs state from broadcast events
- [`routing.md`](./routing.md) — how `ForwarderCommand` / `GlobalForwarderEvent` / `LocalForwarderEvent` traverse libp2p

---

**Sources**
- `src/exo/shared/types/events.py` — `Event` union, `BaseEvent`, `IndexedEvent`, forwarder wrappers
- `src/exo/shared/types/commands.py` — `Command` union, `DownloadCommand`, forwarder wrappers
- `src/exo/shared/types/state.py` — `State` model with custom `Topology` (de)serializers
- `src/exo/shared/types/tasks.py` — `Task` union, `TaskStatus` enum, `CANCEL_ALL_TASKS`
- `src/exo/shared/types/topology.py` — `Connection`, `RDMAConnection`, `SocketConnection`, `Cycle`
- `src/exo/shared/types/common.py` — `Id`, `NodeId`, `SystemId`, `ModelId`, `CommandId`, `SessionId`
- `src/exo/shared/types/chunks.py` — `GenerationChunk` union used in streaming events
- `src/exo/shared/apply.py` — `event_apply` pattern match; per-event apply helpers
- `src/exo/shared/topology.py` — `Topology` class + `TopologySnapshot` serialization
- `src/exo/shared/constants.py` — XDG paths, libp2p topic names, feature flags
- `src/exo/utils/pydantic_ext.py` — `CamelCaseModel`, `FrozenModel`, `TaggedModel`
- `src/exo/master/main.py` — confirms consumer imports
- `src/exo/worker/main.py` — confirms consumer imports
- `src/exo/routing/topics.py`, `src/exo/routing/event_router.py` — confirms consumer imports

**Last Indexed:** 2026-04-21 | exo commit: current
