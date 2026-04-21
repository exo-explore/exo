# Architecture Diagrams

Visual reference for EXO's runtime structure. Start here to orient yourself, then drill into [module-boundaries.md](./module-boundaries.md), [data-flow.md](./data-flow.md), [event-sourcing-message-passing.md](./event-sourcing-message-passing.md), or the per-system deep dives in [../components/](../components/).

Every diagram below is anchored to a code citation. If a diagram ever drifts from the source, the citation is the ground-truth.

---

## System topology

The 5 systems named in [docs/architecture.md:9-29](../../architecture.md) (Master, Worker, Runner, API, Election) plus the Router that glues them together. Each arrow is a topic send/receive pair; the Router is the in-process fan-out layer over libp2p gossipsub.

```mermaid
flowchart LR
    Client([HTTP client])
    subgraph Node["Node (src/exo/main.py)"]
        direction LR
        API[API]
        Master[Master]
        Worker[Worker]
        Election[Election]
        DL[DownloadCoordinator]
        Router(((Router<br/>+ EventRouter)))
        Runner[[Runner<br/>mp.Process]]
    end
    Client -- "HTTP/SSE" --> API
    API -- "COMMANDS" --> Router
    Router -- "COMMANDS" --> Master
    Worker -- "COMMANDS" --> Router
    Master -- "GLOBAL_EVENTS" --> Router
    Router -- "GLOBAL_EVENTS" --> API
    Router -- "GLOBAL_EVENTS" --> Worker
    Worker -- "LOCAL_EVENTS" --> Router
    Router -- "LOCAL_EVENTS" --> Master
    Election <-- "ELECTION_MESSAGES" --> Router
    Router -- "CONNECTION_MESSAGES" --> Election
    Worker <-- "mp.Queue (Task / Event)" --> Runner
    DL <-- "DOWNLOAD_COMMANDS" --> Router
    Router <== "libp2p gossipsub" ==> Peers[(Other nodes)]
```

Caption: Every node wires Router + EventRouter + Election unconditionally; Master/Worker/API/DownloadCoordinator are conditional on flags (`src/exo/main.py:46-142`, `src/exo/main.py:144-159`). The 6 topics (GLOBAL_EVENTS, LOCAL_EVENTS, COMMANDS, ELECTION_MESSAGES, CONNECTION_MESSAGES, DOWNLOAD_COMMANDS) are defined in `src/exo/routing/topics.py:40-51`. Master reads LOCAL_EVENTS, writes GLOBAL_EVENTS (`src/exo/master/main.py:386-411`); Workers/API are the inverse (`src/exo/master/api.py:1676-1699`, `src/exo/worker/main.py:119-135`).

---

## Process model

A single `exo` invocation is one Python process hosting the systems above. The Runner is the sole exception: it's a `multiprocessing.Process` (spawn method) so a crashing inference job can't take the Worker down. The Router calls into Rust through pyo3 bindings, and Rust owns its own tokio runtime on a separate thread pool.

```mermaid
flowchart TB
    subgraph NodeProc["exo Python process (single, anyio event loop)"]
        direction TB
        subgraph PyRing["Python systems sharing event loop"]
            MasterP[Master]
            WorkerP[Worker]
            APIP[API]
            ElectionP[Election]
            DLP[DownloadCoordinator]
            RouterP[Router / EventRouter]
        end
        subgraph FFI["exo_pyo3_bindings (Rust, in-process)"]
            NetHandle[NetworkingHandle]
            TokioRT[tokio multi-thread runtime]
            NetHandle --> TokioRT
            TokioRT --> Libp2p[libp2p swarm + gossipsub]
        end
        RouterP -. "pyo3 FFI" .- NetHandle
    end
    subgraph RunnerProc1["Runner process 1 (mp.Process, spawn)"]
        Bootstrap1[runner.bootstrap.entrypoint]
        MLX1[MLX engine]
    end
    subgraph RunnerProcN["Runner process N (mp.Process, spawn)"]
        BootstrapN[runner.bootstrap.entrypoint]
        MLXN[MLX engine]
    end
    WorkerP == "mp.Queue: Task / Event / cancel" ==> RunnerProc1
    WorkerP == "mp.Queue: Task / Event / cancel" ==> RunnerProcN
```

Caption: `mp.set_start_method("spawn", force=True)` is set globally at process entry (`src/exo/main.py:279`), guaranteeing Runner processes start fresh (no fork-inherited GPU/MLX state). `RunnerSupervisor.create` constructs three `mp_channel` pairs and a `daemon=True` `mp.Process` targeting `runner.bootstrap.entrypoint` (`src/exo/worker/runner/runner_supervisor.py:72-109`). Python systems all share one anyio event loop, started by the `TaskGroup` in `Node.run` (`src/exo/main.py:144-159`). The Rust tokio runtime is initialized once per process in `main_module` (`rust/exo_pyo3_bindings/src/lib.rs:155-172`).

---

## Request lifecycle

End-to-end sequence for a streaming chat completion: HTTP client → API adapter → Master (via COMMANDS) → Worker (via GLOBAL_EVENTS) → Runner (via `mp.Queue`) → token events back up the stack. This is the hottest path in the system.

```mermaid
sequenceDiagram
    autonumber
    participant Client as HTTP client
    participant API as API (FastAPI)
    participant Router as Router / EventRouter
    participant Master as Master
    participant Worker as Worker
    participant Sup as RunnerSupervisor
    participant Runner as Runner (mp.Process)
    Client->>API: POST /v1/chat/completions (stream=true)
    API->>API: chat_request_to_text_generation()
    API->>Router: ForwarderCommand(TextGeneration) on COMMANDS
    Router->>Master: COMMANDS receiver
    Master->>Master: match command -> pick instance, build TaskCreated
    Master->>Router: TaskCreated on LOCAL_EVENTS
    Router->>Master: LOCAL_EVENTS -> MultiSourceBuffer -> IndexedEvent
    Master->>Router: IndexedEvent on GLOBAL_EVENTS
    Router->>Worker: GLOBAL_EVENTS -> apply(state, event)
    Worker->>Worker: plan_step() selects Task for runner
    Worker->>Sup: start_task(Task)
    Sup->>Runner: mp.Queue.send(Task)
    loop per token
        Runner-->>Sup: mp.Queue event: ChunkGenerated(TokenChunk)
        Sup-->>Worker: forward Event
        Worker->>Router: LOCAL_EVENTS (ChunkGenerated)
        Router->>Master: index -> GLOBAL_EVENTS
        Router->>API: GLOBAL_EVENTS -> _apply_state puts chunk in queue
        API-->>Client: SSE data: {choices: [{delta: ...}]}
    end
    Runner-->>Sup: TaskStatusUpdated(Complete)
    Sup-->>Worker: forward
    Worker->>Router: LOCAL_EVENTS -> indexed -> GLOBAL_EVENTS
    API-->>Client: SSE data: [DONE]
```

Caption: API's `chat_completions` converts the OpenAI request via `chat_request_to_text_generation`, sends a `TextGeneration` command, and returns `StreamingResponse` over `_token_chunk_stream` (`src/exo/master/api.py:700-733`). Master's `_command_processor` picks the least-loaded instance and emits `TaskCreated` (`src/exo/master/main.py:117-170`). Master indexes local events via `MultiSourceBuffer` and republishes as `GlobalForwarderEvent` (`src/exo/master/main.py:386-423`). Worker's `_event_applier` folds events into `State` then `plan_step` hands a `Task` to the supervisor (`src/exo/worker/main.py:119-163`). `RunnerSupervisor._forward_events` streams events back out of the runner process (`src/exo/worker/runner/runner_supervisor.py:188-214`). API's `_apply_state` pushes `ChunkGenerated` chunks into the per-command queue backing the SSE stream (`src/exo/master/api.py:1676-1699`).

---

## Module dependency graph

Directory-level Python import graph within `src/exo/`. `shared` (types + apply + election + constants) is the foundation; `routing` depends only on `shared` + `utils`; `master`, `worker`, `download` sit on top; `main.py` wires everything.

```mermaid
flowchart TB
    main[src/exo/main.py]
    master[exo.master]
    worker[exo.worker]
    download[exo.download]
    routing[exo.routing]
    shared[exo.shared<br/>+ shared.types]
    utils[exo.utils]
    ffi[exo_pyo3_bindings<br/>Rust FFI]
    main --> master
    main --> worker
    main --> download
    main --> routing
    main --> shared
    main --> utils
    master --> shared
    master --> utils
    master --> download
    worker --> shared
    worker --> utils
    worker --> download
    download --> shared
    download --> utils
    routing --> shared
    routing --> utils
    routing --> ffi
    shared --> utils
```

Caption: Verified by grepping top-level imports at each subpackage. `src/exo/main.py:13-27` imports from every sibling except `exo.shared.types` internals. `routing/router.py:17-31` is the only module in pure-Python land that imports `exo_pyo3_bindings`. `master/main.py:1-65` and `worker/main.py:1-49` depend on `shared.types.*` + `utils` + `download.*` (via `worker` downloading models and `master` reading downloads into state). `shared` never imports from `master`/`worker`/`routing` — enforced by code review, since a backward dep would create a cycle.

---

## Network topology

How nodes actually wire up at the physical layer. Thunderbolt-4 links between directly connected Apple Silicon devices become `RDMAConnection` edges (used by MLX ring/jaccl for model sharding); everything else (Wi-Fi, Ethernet, TB hops over TCP) becomes a `SocketConnection` edge (used for gossipsub traffic). A 4-node cluster with two Thunderbolt pairs and fallback TCP in between:

```mermaid
flowchart LR
    subgraph TBPair1["TB-4 pair"]
        A[Node A<br/>M3 Max]
        B[Node B<br/>M3 Max]
        A ==RDMA==> B
        B ==RDMA==> A
    end
    subgraph TBPair2["TB-4 pair"]
        C[Node C<br/>M2 Ultra]
        D[Node D<br/>M2 Ultra]
        C ==RDMA==> D
        D ==RDMA==> C
    end
    A <-. TCP/UDP gossipsub .-> C
    A <-. TCP/UDP gossipsub .-> D
    B <-. TCP/UDP gossipsub .-> C
    B <-. TCP/UDP gossipsub .-> D
    mdns{{mDNS discovery}}
    mdns -.-> A
    mdns -.-> B
    mdns -.-> C
    mdns -.-> D
```

Caption: `Connection` is `source: NodeId, sink: NodeId, edge: RDMAConnection | SocketConnection` (`src/exo/shared/types/topology.py:32-35`). `RDMAConnection` carries the source/sink RDMA interface names (`topology.py:20-22`); `SocketConnection` only carries the sink's `Multiaddr` (`topology.py:25-29`). mDNS-discovered links are written to `CONNECTION_MESSAGES` (policy `Never` — node-local only) by the networking subsystem (`src/exo/routing/topics.py:46-48`, `docs/architecture.md:72-73`). Placement decisions read `self.state.topology` to prefer RDMA-connected node groups when sharding an instance (`src/exo/master/main.py:293-304`).

---

## Python ↔ Rust FFI boundary

Everything crossing the language boundary goes through `exo_pyo3_bindings`, which is the only pyo3 crate and the only Python import target. The `networking` Rust crate is the libp2p swarm; `util` and `system_custodian` are pure-Rust helpers consumed only by the bindings crate.

```mermaid
flowchart LR
    subgraph Py["Python"]
        Router[exo.routing.router.Router]
        Topics[exo.routing.topics]
        KPgen[get_node_id_keypair]
    end
    subgraph Bindings["rust/exo_pyo3_bindings (pyo3)"]
        PyKeypair[PyKeypair / Keypair]
        NetHandle[NetworkingHandle]
        PyFromSwarm[PyFromSwarm enum]
        ExcTypes[AllQueuesFullError<br/>MessageTooLargeError<br/>NoPeersSubscribedToTopicError]
    end
    subgraph Crates["Rust crates (non-FFI)"]
        NetCrate[rust/networking<br/>swarm + discovery]
        UtilCrate[rust/util]
    end
    Router -- "NetworkingHandle(identity, bootstrap, port)" --> NetHandle
    Router -- "gossipsub_subscribe / publish / recv" --> NetHandle
    Router -- "match PyFromSwarm::Message | Connection" --> PyFromSwarm
    KPgen -- "Keypair.generate / from_bytes / to_bytes" --> PyKeypair
    Topics -. "no Rust dep" .- Bindings
    NetHandle --> NetCrate
    NetCrate --> UtilCrate
    PyKeypair --> NetCrate
```

Caption: The Rust entry point is `#[pymodule] fn main_module` in `rust/exo_pyo3_bindings/src/lib.rs:151-172`, which registers `PyKeypair` and the networking submodule. Python imports are concentrated in `src/exo/routing/router.py:17-24`: `AllQueuesFullError`, `Keypair`, `MessageTooLargeError`, `NetworkingHandle`, `NoPeersSubscribedToTopicError`, `PyFromSwarm`. `NetworkingHandle` methods `gossipsub_subscribe`, `gossipsub_unsubscribe`, `gossipsub_publish`, and `recv` are the entire hot-path API used by `Router._networking_subscribe/_networking_publish/_networking_recv` (`src/exo/routing/router.py:195-237`). The `networking` crate's swarm (`rust/networking/src/lib.rs:1-8` and `swarm.rs`/`discovery.rs`) is wrapped but not directly imported from Python.

---

Sources: src/exo/main.py:1-310, src/exo/routing/topics.py:14-51, src/exo/routing/router.py:17-238, src/exo/routing/event_router.py:1-55, src/exo/master/main.py:68-423, src/exo/master/api.py:197-735, src/exo/master/api.py:1641-1700, src/exo/worker/main.py:52-163, src/exo/worker/runner/runner_supervisor.py:52-214, src/exo/shared/types/topology.py:1-35, rust/exo_pyo3_bindings/src/lib.rs:150-172, rust/networking/src/lib.rs:1-44, docs/architecture.md:1-85

Last indexed: 2026-04-21 (commit c0d5bf92)
