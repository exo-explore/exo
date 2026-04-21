# Worker Component

## Overview

The Worker is the **per-node executor** of the exo cluster. Every node runs a Worker that consumes `IndexedEvent`s from `GLOBAL_EVENTS` (broadcast by the elected master), applies them to a local copy of the cluster `State`, and drives that state forward by emitting `Task`s on `LOCAL_EVENTS`. It owns the node's inference subprocesses (Runners), streams system info (memory, thunderbolt, disk, macmon metrics), and probes network reachability to keep the topology live. [`src/exo/worker/main.py:52-102`](../../../src/exo/worker/main.py)

Unlike the Master, the Worker is **not** a single-writer — it is a reactive planner that compares local `State` against its own runner supervisors and proposes one action at a time through a pure `plan()` function. [`src/exo/worker/main.py:137-162`](../../../src/exo/worker/main.py), [`src/exo/worker/plan.py:44-65`](../../../src/exo/worker/plan.py)

A Worker runs on every node that isn't started with `--no-worker`. It's constructed alongside the Router, EventRouter, Master, Election, and API inside `Node`. [`src/exo/main.py:95-102`](../../../src/exo/main.py)

> Architecture summary from [`docs/architecture.md:15-21`](../../architecture.md): "Worker — Schedules work on a node, gathers system information, etc. Runner — Executes inference jobs (for now) in an isolated process from the worker for fault-tolerance."

## Key files

Files directly under `src/exo/worker/`:

| File | Purpose |
|------|---------|
| `main.py` | `Worker` class — orchestrates info gathering, event application, planning, runner supervision, and reachability polling. [`main.py:52-339`](../../../src/exo/worker/main.py) |
| `plan.py` | Pure `plan()` function — given fresh runners + state, returns the next `Task` the worker should emit. Short-circuits through seven priority-ordered stages. [`plan.py:44-65`](../../../src/exo/worker/plan.py) |
| `__init__.py` | Empty package marker. [`__init__.py`](../../../src/exo/worker/__init__.py) |
| `runner/runner_supervisor.py` | `RunnerSupervisor` — spawns the inference subprocess, forwards its events, detects crashes, and reports `RunnerFailed`. [`runner_supervisor.py:52-289`](../../../src/exo/worker/runner/runner_supervisor.py) |
| `runner/bootstrap.py` | Subprocess entrypoint — sets MLX env vars, chooses LLM vs. image runner, catches crashes and emits `RunnerFailed`. [`bootstrap.py:15-73`](../../../src/exo/worker/runner/bootstrap.py) |
| `runner/llm_inference/` | MLX-based text-generation runner implementation. [`runner.py:76-100`](../../../src/exo/worker/runner/llm_inference/runner.py) |
| `runner/image_models/` | Image generation / image edits runner implementation. [`bootstrap.py:38-44`](../../../src/exo/worker/runner/bootstrap.py) |
| `engines/mlx/`, `engines/image/` | Inference engines (KV cache, generate loop, model adapters) used **inside** the Runner subprocess — not by Worker proper. |
| `tests/` | `test_plan/*` unit tests for the `plan()` state machine; `test_runner/*` for the supervisor; `test_mlx/*` for engines. [`tests/`](../../../src/exo/worker/tests/) |

## Event application

The Worker subscribes to `GLOBAL_EVENTS` via `event_receiver: Receiver[IndexedEvent]` and applies every indexed event to its local `State` through the shared pure function `apply()`. [`main.py:57-70, 119-135`](../../../src/exo/worker/main.py)

```python
async def _event_applier(self):
    with self.event_receiver as events:
        async for event in events:
            # 2. for each event, apply it to the state
            self.state = apply(self.state, event=event)
            event = event.event
            ...
```
[`main.py:119-135`](../../../src/exo/worker/main.py)

Key properties:

- **State is immutable** — `apply(state, event)` returns a fresh `State`. The Worker simply reassigns `self.state`. [`main.py:123`](../../../src/exo/worker/main.py), [`src/exo/shared/apply.py:1-50`](../../../src/exo/shared/apply.py)
- **Same reducer on every node** — both Master and Worker call the same `apply()`, so every node reconstructs an identical `State` by replaying the same ordered `IndexedEvent` stream. [`src/exo/shared/apply.py`](../../../src/exo/shared/apply.py)
- **Side-effect: input-chunk buffering.** The one piece of Worker-local state that isn't in `State` is the image-edit input buffer — a map from `CommandId` to base64 chunks, assembled when `InputChunkReceived` events arrive. [`main.py:127-135`](../../../src/exo/worker/main.py)
- **No writes** happen inside `_event_applier`. All mutations to the cluster go through `event_sender` (→ `LOCAL_EVENTS`) and `command_sender` (→ `COMMANDS`), which are **only** written from the planner and info-forwarder loops. [`main.py:58-68`](../../../src/exo/worker/main.py)

See [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) for the event-sourcing rationale.

## Task scheduling

The Worker runs a **plan loop** that wakes every 100 ms, calls the pure `plan()` function, and — if it returns a `Task` — emits a `TaskCreated` event and drives the task to its next status. [`main.py:137-162`](../../../src/exo/worker/main.py)

```python
async def plan_step(self):
    while True:
        await anyio.sleep(0.1)
        task: Task | None = plan(
            self.node_id,
            self.runners,
            self.state.downloads,
            self.state.instances,
            self.state.runners,
            self.state.tasks,
            self.input_chunk_buffer,
            self.input_chunk_counts,
        )
        if task is None:
            continue
        ...
```
[`main.py:137-162`](../../../src/exo/worker/main.py)

### The `plan()` priority ladder

`plan()` is a Python short-circuit `or` chain. The **first** stage that returns a task wins; this enforces a strict per-tick priority. [`plan.py:55-65`](../../../src/exo/worker/plan.py)

| # | Stage | Emits | Trigger |
|---|-------|-------|---------|
| 1 | `_cancel_tasks` | `CancelTask` | A task in state has `TaskStatus.Cancelled` and hasn't been cancelled on its owning runner yet. [`plan.py:310-326`](../../../src/exo/worker/plan.py) |
| 2 | `_kill_runner` | `Shutdown` | Runner's instance was deleted from state, or any peer runner in the instance is `RunnerFailed`. [`plan.py:68-88`](../../../src/exo/worker/plan.py) |
| 3 | `_create_runner` | `CreateRunner` | An instance has a runner assignment for this node but no local `RunnerSupervisor` exists. [`plan.py:91-112`](../../../src/exo/worker/plan.py) |
| 4 | `_model_needs_download` | `DownloadModel` | Runner is idle and the shard's model isn't recorded as ongoing/completed/failed for this node. [`plan.py:115-138`](../../../src/exo/worker/plan.py) |
| 5 | `_init_distributed_backend` | `ConnectToGroup` | Multi-node instance: all peer runners are `RunnerConnecting`/`RunnerIdle`, and either we're an accepting rank or the last rank with everyone else connecting. [`plan.py:141-188`](../../../src/exo/worker/plan.py) |
| 6 | `_load_model` | `LoadModel` | All node-local downloads complete; single-node + idle, or connected + all peers connected/loading/loaded. [`plan.py:191-229`](../../../src/exo/worker/plan.py) |
| 7 | `_ready_to_warmup` | `StartWarmup` | Runner is `RunnerLoaded` and all peers are loaded (or warming up). Rank-0 waits for all peers; rank > 0 proceeds when everyone is loaded. [`plan.py:232-268`](../../../src/exo/worker/plan.py) |
| 8 | `_pending_tasks` | `TextGeneration`/`ImageGeneration`/`ImageEdits` | A pending or running task targets this worker's instance, all peer runners are `RunnerReady`/`RunnerRunning`, and the task isn't already in progress or completed locally. [`plan.py:271-307`](../../../src/exo/worker/plan.py) |

### Dispatch of the chosen task

After `plan()` returns, the Worker:

1. **Rate-limits `DownloadModel`** via `KeyedBackoff` (base 0.5s, cap 10s per model) to avoid flooding the event log when downloads keep failing. [`main.py:80, 155-158, 173-175`](../../../src/exo/worker/main.py)
2. **Emits `TaskCreated`** on `LOCAL_EVENTS`. [`main.py:162`](../../../src/exo/worker/main.py)
3. **Dispatches by task type** with a `match`/`case`:
   - `CreateRunner` → `_create_supervisor()` spawns a new `RunnerSupervisor` and marks the task `Complete`. [`main.py:166-172, 287-295`](../../../src/exo/worker/main.py)
   - `DownloadModel` → if the model is already in `EXO_MODELS_PATH`, synthesize `NodeDownloadProgress(DownloadCompleted, read_only=True)` and `TaskStatus.Complete`; otherwise send `StartDownload` on the download-command topic and mark `Running`. [`main.py:173-214`](../../../src/exo/worker/main.py)
   - `Shutdown` → pop the supervisor, call `start_task(Shutdown)` with a 3-second `fail_after` (marks `TimedOut` on expiry), then `shutdown()` the supervisor. [`main.py:215-227`](../../../src/exo/worker/main.py)
   - `CancelTask` → forward to the runner's `cancel_task()` and mark the outer task `Complete`. [`main.py:228-236`](../../../src/exo/worker/main.py)
   - `ImageEdits` with pending input chunks → assemble base64 chunks into a single image, build a modified `ImageEdits`, clear the buffer, then forward. [`main.py:237-274`](../../../src/exo/worker/main.py)
   - Otherwise → `_start_runner_task()` looks up the runner via `instance.shard_assignments.node_to_runner[self.node_id]` and calls `start_task()`. [`main.py:275-276, 281-285`](../../../src/exo/worker/main.py)

## Runner supervision

Each `CreateRunner` task creates a `RunnerSupervisor` that owns a real `multiprocessing.Process`. Subprocess isolation is the **fault-tolerance boundary** for inference — a segfault, OOM, or Python exception in MLX code kills the subprocess without taking down the Worker. [`runner_supervisor.py:52-109`](../../../src/exo/worker/runner/runner_supervisor.py), [`docs/architecture.md:19-21`](../../architecture.md)

### Spawn

`RunnerSupervisor.create()` allocates three mp channels (events, tasks, cancels) and starts a daemon process targeting `entrypoint` in `bootstrap.py`. [`runner_supervisor.py:72-109`](../../../src/exo/worker/runner/runner_supervisor.py)

```python
runner_process = mp.Process(
    target=entrypoint,
    args=(
        bound_instance,
        ev_send,
        task_recv,
        cancel_recv,
        logger,
    ),
    daemon=True,
)
```
[`runner_supervisor.py:84-94`](../../../src/exo/worker/runner/runner_supervisor.py)

The entrypoint bumps `RLIMIT_NOFILE` (min 2048), sets `MLX_METAL_FAST_SYNCH` (overridable via `EXO_FAST_SYNCH`), then imports and runs either `exo.worker.runner.image_models.runner.Runner` or `exo.worker.runner.llm_inference.runner.Runner` based on `bound_instance.is_image_model`. Uncaught exceptions are logged and converted into `RunnerStatusUpdated(RunnerFailed)`. [`bootstrap.py:15-64`](../../../src/exo/worker/runner/bootstrap.py)

### Task / cancel pipes

- `start_task(task)` sends the task over the mp channel and awaits a `TaskAcknowledged` event from the subprocess; duplicates (already-pending or already-completed ids) are skipped. [`runner_supervisor.py:147-168`](../../../src/exo/worker/runner/runner_supervisor.py)
- `cancel_task(task_id)` adds the id to a local `cancelled` set and pushes it onto the cancel channel with a 500 ms send timeout. If the send blocks, the supervisor runs a crash check. [`runner_supervisor.py:170-186`](../../../src/exo/worker/runner/runner_supervisor.py)

### Event forwarding

`_forward_events()` drains the subprocess event channel and republishes every event on the Worker's outbound `event_sender`. Along the way it maintains supervisor-local bookkeeping: [`runner_supervisor.py:188-219`](../../../src/exo/worker/runner/runner_supervisor.py)

- `RunnerStatusUpdated` → updates `self.status` (the supervisor's cached `RunnerStatus`).
- `TaskAcknowledged` → signals the `anyio.Event` registered in `start_task()`.
- `TaskStatusUpdated(Complete)` → asserts the runner was in an executing state and moves the task from `in_progress` to `completed`.

Stream errors (`ClosedResourceError`, `BrokenResourceError`) trigger `_check_runner()` to diagnose a crash. [`runner_supervisor.py:215-216`](../../../src/exo/worker/runner/runner_supervisor.py)

### Crash detection

`_watch_runner()` wakes every 5 seconds and checks `runner_process.is_alive()`. On a dead process it calls `_check_runner(RuntimeError("Runner found to be dead"))`. [`runner_supervisor.py:227-232`](../../../src/exo/worker/runner/runner_supervisor.py)

`_check_runner()` joins the process, reads `exitcode`, formats the cause (negative exitcode → signal name via `signal.strsignal`), and: [`runner_supervisor.py:234-288`](../../../src/exo/worker/runner/runner_supervisor.py)

1. For every in-progress `TextGeneration`/`ImageGeneration`/`ImageEdits`, emits a `ChunkGenerated(ErrorChunk(...))` so the API client sees a terminal error chunk instead of hanging.
2. Sets `self.status = RunnerFailed(...)` and broadcasts `RunnerStatusUpdated(RunnerFailed)` so Master, API, and peer workers learn of the failure via the normal event loop.
3. Calls `self.shutdown()` to tear down channels, `join(5)` the process, and — if it's still alive — `terminate()` then `kill()` escalation. [`runner_supervisor.py:117-145`](../../../src/exo/worker/runner/runner_supervisor.py)

A subsequent `plan()` tick hits `_kill_runner` (any peer `RunnerFailed` kills the local runner too), ensuring the whole multi-node instance tears down cleanly. [`plan.py:78-88`](../../../src/exo/worker/plan.py)

A `__del__` guard `kill()`s the process if the supervisor was GCed without a clean shutdown. [`runner_supervisor.py:221-225`](../../../src/exo/worker/runner/runner_supervisor.py)

## Info gathering

The Worker composes an `InfoGatherer` which publishes `GatheredInfo` values to a local channel; `_forward_info` wraps each one in a `NodeGatheredInfo` event and sends it on `LOCAL_EVENTS` so the Master can index it into `State`. [`main.py:85-117`](../../../src/exo/worker/main.py)

```python
info_send, info_recv = channel[GatheredInfo]()
info_gatherer: InfoGatherer = InfoGatherer(info_send)
...
tg.start_soon(info_gatherer.run)
tg.start_soon(self._forward_info, info_recv)
```
[`main.py:85-91`](../../../src/exo/worker/main.py)

The `GatheredInfo` union is defined in `utils/info_gatherer/info_gatherer.py`. [`info_gatherer.py:357-369`](../../../src/exo/utils/info_gatherer/info_gatherer.py)

| Info type | Source | Default interval |
|-----------|--------|------------------|
| `StaticNodeInformation` (model, chip, os version/build) | `system_info` helpers, gathered once but refreshed periodically | 60 s [`info_gatherer.py:381`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `MiscData` (friendly name) | `get_friendly_name()` | 60 s [`info_gatherer.py:376`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `NodeNetworkInterfaces` | `get_network_interfaces()` | 10 s [`info_gatherer.py:375, 480-490`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `MemoryUsage` | `macmon` on Darwin, else `psutil` (`memory_poll_rate=1s` fallback) | 1 s [`info_gatherer.py:378, 462-478`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `MacmonMetrics` (CPU/GPU power + usage) | `macmon pipe --interval` subprocess on Darwin only | ~1 s [`info_gatherer.py:379, 528-563`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `MacThunderboltIdentifiers`, `MacThunderboltConnections` | `system_profiler` via `ThunderboltConnectivity.gather()` | 5 s (Darwin) [`info_gatherer.py:377, 435-460`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `ThunderboltBridgeInfo` | `networksetup` + `ifconfig` bridge-member inspection | 10 s (Darwin) [`info_gatherer.py:380, 492-503`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `RdmaCtlStatus` | `rdma_ctl status` if the binary exists | 10 s (Darwin) [`info_gatherer.py:382, 505-515`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `NodeDiskUsage` | `DiskUsage.from_path(EXO_MODELS_DIR)` | 30 s [`info_gatherer.py:383, 517-526`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |
| `NodeConfig` | `EXO_CONFIG_FILE` (TOML) | Once at startup [`info_gatherer.py:406-408`](../../../src/exo/utils/info_gatherer/info_gatherer.py) |

### Connectivity probing

In parallel with info gathering, `_poll_connection_updates()` runs every 10 seconds, pinging every reachable node via `check_reachable()` and emitting `TopologyEdgeCreated` / `TopologyEdgeDeleted` events so the Master can maintain a live topology graph. Multiaddrs are synthesized from discovered IPs (`/ip4/.../tcp/52415` or `/ip6/.../tcp/52415`). [`main.py:297-339`](../../../src/exo/worker/main.py)

## Gotchas

1. **`plan()` expects FRESH runners.** The `runners` argument must come from the Worker's live `self.runners: dict[RunnerId, RunnerSupervisor]`, **not** from `state.runners` (which lags). The docstring hammers this: *"Runners is expected to be FRESH and so should not come from state"*. [`plan.py:46-47`](../../../src/exo/worker/plan.py), [`main.py:140-149`](../../../src/exo/worker/main.py)

2. **Plan loop tick is 100 ms.** Steady-state CPU cost of the Worker is dominated by repeated traversal of `state.instances` and `state.runners`. Fine for clusters of tens of nodes; large clusters may want adaptive backoff. [`main.py:138-139`](../../../src/exo/worker/main.py)

3. **Download backoff only gates `DownloadModel` — not actual download work.** `KeyedBackoff` prevents spamming `TaskCreated` events, but the real download runs in the `download.coordinator`. If downloads keep failing, expect `NodeDownloadProgress(DownloadFailed)` to keep flowing. [`main.py:80, 155-158`](../../../src/exo/worker/main.py)

4. **Event channel closes are swallowed on shutdown.** `_forward_info` exits quietly on `ClosedResourceError`/`BrokenResourceError`; the planner does not retry. [`main.py:104-117`](../../../src/exo/worker/main.py)

5. **`RunnerSupervisor.shutdown()` kills aggressively.** After `join(5)` fails it escalates to `terminate()`, then `kill()`. Inference subprocesses get up to 6 seconds to exit cleanly. If you need longer (e.g., saving KV cache on exit), bump those timeouts. [`runner_supervisor.py:132-145`](../../../src/exo/worker/runner/runner_supervisor.py)

6. **One runner failure cascades through the instance.** `_kill_runner` tears down any local runner whose peer is `RunnerFailed`, so multi-node instances are all-or-nothing. There is no partial recovery. [`plan.py:78-88`](../../../src/exo/worker/plan.py)

7. **Master node changes rebuild the Worker.** On master re-election, `Node._elect_loop` shuts down and re-creates both the `EventRouter` and `Worker` to drop stale per-session state. Any in-flight runner supervisors are destroyed. [`src/exo/main.py:180-199`](../../../src/exo/main.py)

8. **Image-edit chunk buffer is unbounded.** `input_chunk_buffer` / `input_chunk_counts` only free entries when the task is dispatched (or, for `ImageEdits`, after assembly). A client that streams `InputChunkReceived` events but never completes the request leaks memory. [`main.py:77-78, 127-135, 237-273`](../../../src/exo/worker/main.py)

9. **Fast-synch is on by default on Metal.** Set `EXO_FAST_SYNCH=off` to disable `MLX_METAL_FAST_SYNCH=1` in the Runner subprocess — useful when diagnosing MLX stream sync bugs. [`bootstrap.py:28-34`](../../../src/exo/worker/runner/bootstrap.py)

10. **Prefill / decode timeouts are supervisor constants.** `PREFILL_TIMEOUT_SECONDS = 60`, `DECODE_TIMEOUT_SECONDS = 5` live in `runner_supervisor.py` as module-level constants, not config. [`runner_supervisor.py:48-49`](../../../src/exo/worker/runner/runner_supervisor.py)

## See also

- [`master.md`](./master.md) — single-writer counterpart; indexes the events Worker emits.
- [`shared.md`](./shared.md) — `State`, `Event`, `Task`, `RunnerStatus`, `apply()`.
- [`../architecture/module-boundaries.md`](../architecture/module-boundaries.md) — where Worker fits in the 5-module architecture.
- [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) — why `apply(state, event)` is pure and shared across nodes.

---

**Sources**

- `src/exo/worker/main.py` — `Worker` class, plan loop, event applier, info forwarder, reachability poller.
- `src/exo/worker/plan.py` — pure `plan()` and its eight priority stages.
- `src/exo/worker/runner/runner_supervisor.py` — `RunnerSupervisor`, crash detection, event forwarding.
- `src/exo/worker/runner/bootstrap.py` — subprocess entrypoint + MLX env setup.
- `src/exo/utils/info_gatherer/info_gatherer.py` — system info polling tasks and `GatheredInfo` union.
- `src/exo/shared/apply.py` — shared event reducer used by Worker and Master.
- `src/exo/main.py` — Node-level wiring of Worker alongside Master, API, Election.
- `docs/architecture.md` — system overview.

**Last indexed:** 2026-04-21 | exo commit: `c0d5bf92`
