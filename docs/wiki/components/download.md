# Download System

## Overview

The download system is responsible for fetching model weights (safetensors, tokenizers, configs) from the HuggingFace hub onto each node's local disk so that Runners can memory-map them for inference. It is intentionally a **per-node** concern: every worker that is assigned a shard of a model downloads its own copy — there is no cross-node weight streaming. Coordination across nodes happens only through `StartDownload` / `CancelDownload` / `DeleteDownload` commands issued by the Master and consumed by each target node's `DownloadCoordinator`.

The system is split into three layers: a lifecycle orchestrator (`DownloadCoordinator`) that listens on the `DOWNLOAD_COMMANDS` pub/sub topic and emits `NodeDownloadProgress` events; an abstract `ShardDownloader` + concrete `ResumableShardDownloader` that encapsulates the "get this shard onto disk" operation, wrapped by a `SingletonShardDownloader` to deduplicate concurrent calls; and the low-level `download_utils.py` module which talks HTTP to HuggingFace, supports range-resumable partial files, and verifies sha1/sha256 ETags. Because the system is wired over event-sourced pub/sub, cross-link to [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md) for the broader model.

## Key files

| File | Purpose |
|------|---------|
| `coordinator.py` | Lifecycle orchestrator; consumes `ForwarderDownloadCommand` from pub/sub, emits `NodeDownloadProgress` events [`coordinator.py:40-58`](/Users/leozealous/exo/src/exo/download/coordinator.py) |
| `shard_downloader.py` | Abstract `ShardDownloader` interface + `NoopShardDownloader` for tests [`shard_downloader.py:18-54`](/Users/leozealous/exo/src/exo/download/shard_downloader.py) |
| `impl_shard_downloader.py` | `SingletonShardDownloader` (dedupes concurrent requests for the same shard) + `ResumableShardDownloader` [`impl_shard_downloader.py:50-118`](/Users/leozealous/exo/src/exo/download/impl_shard_downloader.py) |
| `download_utils.py` | HTTP layer: file-list fetch, range-resume downloads, hash verification, progress accounting [`download_utils.py:481-608`](/Users/leozealous/exo/src/exo/download/download_utils.py) |
| `huggingface_utils.py` | HF endpoint / token discovery, auth headers, allow-pattern filtering [`huggingface_utils.py:61-88`](/Users/leozealous/exo/src/exo/download/huggingface_utils.py) |
| `tests/test_re_download.py` | End-to-end test of Start → Delete → Start through the coordinator [`test_re_download.py:128-192`](/Users/leozealous/exo/src/exo/download/tests/test_re_download.py) |

## Download lifecycle

A model download proceeds through the following stages, end-to-end:

1. **Trigger.** The Worker's planner decides a shard is needed and publishes a `ForwarderDownloadCommand{command: StartDownload{target_node_id, shard_metadata}}` on the `DOWNLOAD_COMMANDS` topic [`worker/main.py:200-208`](/Users/leozealous/exo/src/exo/worker/main.py). The Master's HTTP API (`POST /downloads/start`) also issues `StartDownload` commands directly [`master/api.py:1745-1753`](/Users/leozealous/exo/src/exo/master/api.py).

2. **Dispatch.** Every node runs a `DownloadCoordinator` (unless started with `--no-downloads`). Each coordinator's `_command_processor` receives all download commands but ignores any whose `target_node_id` does not match its own `node_id` [`coordinator.py:123-136`](/Users/leozealous/exo/src/exo/download/coordinator.py). This is how fan-out works — see [Multi-node coordination](#multi-node-coordination).

3. **Early-exit checks.** `_start_download` (a) skips if a download for this `model_id` is already ongoing, completed, or failed [`coordinator.py:156-163`](/Users/leozealous/exo/src/exo/download/coordinator.py); (b) checks `EXO_MODELS_PATH` (a set of read-only directories configured via `EXO_MODELS_READ_ONLY_DIRS`) for a pre-existing complete copy and emits `DownloadCompleted{read_only=True}` if found [`coordinator.py:165-182`](/Users/leozealous/exo/src/exo/download/coordinator.py); (c) calls `get_shard_download_status_for_shard` to detect a resume-from-complete situation (all files already on disk in the writable models dir) and short-circuits with `DownloadCompleted` [`coordinator.py:193-209`](/Users/leozealous/exo/src/exo/download/coordinator.py).

4. **Offline guard.** If `offline=True` and the model is not already present locally, the coordinator emits `DownloadFailed` rather than attempting network I/O [`coordinator.py:211-223`](/Users/leozealous/exo/src/exo/download/coordinator.py).

5. **Task launch.** `_start_download_task` creates an `asyncio.Task` that calls `shard_downloader.ensure_shard(shard)`. The task reference is stored in `active_downloads[model_id]` so it can be cancelled later [`coordinator.py:228-265`](/Users/leozealous/exo/src/exo/download/coordinator.py).

6. **Actual download.** Inside `ResumableShardDownloader.ensure_shard`, the call flows into `download_shard` [`impl_shard_downloader.py:106-118`](/Users/leozealous/exo/src/exo/download/impl_shard_downloader.py). `download_shard` fetches a file list (cached on disk), filters by allow-patterns, and launches `download_file_with_retry` for each file under an `asyncio.Semaphore(max_parallel_downloads)` (default 8) [`download_utils.py:846-872`](/Users/leozealous/exo/src/exo/download/download_utils.py).

7. **Progress throttling.** Each file-level progress callback rolls up into a `RepoDownloadProgress` via `calculate_repo_progress` [`download_utils.py:770-826`](/Users/leozealous/exo/src/exo/download/download_utils.py), which is delivered to `DownloadCoordinator._download_progress_callback`. The coordinator throttles per-model `DownloadOngoing` events to one per second using `_last_progress_time` [`coordinator.py:86-106`](/Users/leozealous/exo/src/exo/download/coordinator.py).

8. **Completion.** When `status == "complete"` fires in the callback, the coordinator sends a final `DownloadCompleted{total, model_directory}` event, removes the task from `active_downloads`, and clears the throttle timestamp [`coordinator.py:69-85`](/Users/leozealous/exo/src/exo/download/coordinator.py).

9. **Failure.** If `ensure_shard` raises, the wrapper catches it and emits `DownloadFailed{error_message}` [`coordinator.py:245-262`](/Users/leozealous/exo/src/exo/download/coordinator.py). `HuggingFaceAuthenticationError` (401/403) produces a user-actionable message referencing `HF_TOKEN` [`download_utils.py:55-69`](/Users/leozealous/exo/src/exo/download/download_utils.py).

## Resume and verification

Downloads are resumable at two levels:

**Per-file partial resume.** `_download_file` always writes to `{target}.partial`. If the partial exists from a previous run, it stats it to get `resume_byte_pos` and sends an HTTP `Range: bytes={resume_byte_pos}-` header [`download_utils.py:563-578`](/Users/leozealous/exo/src/exo/download/download_utils.py). If the partial size already equals the expected content length (the branch at `download_utils.py:569`), no new request is made and verification runs on the existing partial.

**Hash verification.** Every downloaded file is hashed after the body finishes. The hash type is chosen from the ETag length: 64-char hex → sha256, otherwise sha1 (git-blob style with `blob {size}\0` prefix) [`download_utils.py:433-441`](/Users/leozealous/exo/src/exo/download/download_utils.py). If the computed hash mismatches the remote ETag, the `.partial` file is deleted and the function raises — forcing a fresh re-download on the next retry [`download_utils.py:594-605`](/Users/leozealous/exo/src/exo/download/download_utils.py). Only on match is the `.partial` atomically renamed to the final name.

**Existing-file verification.** For files that already exist at the final path (not `.partial`), `_download_file` does a `HEAD` against HuggingFace to compare `content-length` vs local `st_size`. A size mismatch triggers re-download; a network error during the HEAD is swallowed and the local file is trusted (explicit "offline-tolerant" branch) [`download_utils.py:532-553`](/Users/leozealous/exo/src/exo/download/download_utils.py).

**Retry policy.** File-level downloads retry up to 3 times with exponential backoff `2.0 ** attempt` seconds [`download_utils.py:490-519`](/Users/leozealous/exo/src/exo/download/download_utils.py). `HuggingFaceAuthenticationError` and `FileNotFoundError` are raised immediately without retry. `HuggingFaceRateLimitError` (HTTP 429) retries but with the same backoff. The same retry shape is used for the repo-tree listing call [`download_utils.py:338-358`](/Users/leozealous/exo/src/exo/download/download_utils.py).

**Etag redirect handling.** On HTTP 307 redirects (the common HF CDN pattern), `file_meta` prefers `x-linked-size` / `x-linked-etag` headers as the authoritative values; only if those are missing does it follow the redirect [`download_utils.py:444-467`](/Users/leozealous/exo/src/exo/download/download_utils.py). A `-gzip` suffix on the etag is stripped before comparison [`download_utils.py:562`](/Users/leozealous/exo/src/exo/download/download_utils.py).

## Multi-node coordination

**There is no fan-out transfer.** Each node that needs a shard downloads it independently from HuggingFace. The distributed part is which nodes start downloading, and when — not the byte stream itself.

The mechanism is the `DOWNLOAD_COMMANDS` pub/sub topic. It's registered at node startup in `main.py` and published with `PublishPolicy.Always`, meaning every node receives every download command [`main.py:61`](/Users/leozealous/exo/src/exo/main.py), [`routing/topics.py:49-51`](/Users/leozealous/exo/src/exo/routing/topics.py). Targeting happens in the receiver: `DownloadCoordinator._command_processor` filters on `cmd.command.target_node_id != self.node_id` and drops non-matching commands [`coordinator.py:127-128`](/Users/leozealous/exo/src/exo/download/coordinator.py).

Fan-out therefore comes from the Master issuing N commands, one per target node. When a `PlaceInstance` decision assigns a model to a cluster, the Master-side placement code produces per-node `StartDownload` commands (this flow lives in the Master placement pipeline, not in the download component itself). Conversely, when an instance is deleted or re-placed, `cancel_unnecessary_downloads` computes which `(node_id, model_id)` pairs are downloading but no longer assigned and emits `CancelDownload` commands for each [`master/placement.py:257-280`](/Users/leozealous/exo/src/exo/master/placement.py), called from the Master's `DeleteInstance` handler [`master/main.py:284-291`](/Users/leozealous/exo/src/exo/master/main.py).

Progress reporting is also per-node: each coordinator emits `NodeDownloadProgress{download_progress: ...}` events which flow through the normal event-sourcing pipeline (see [`../architecture/event-sourcing-message-passing.md`](../architecture/event-sourcing-message-passing.md)) and end up on `State.downloads[node_id]`. The Master reads from this same aggregated state when deciding which downloads to cancel [`master/placement.py:258-267`](/Users/leozealous/exo/src/exo/master/placement.py).

## Integration with Master

The master-download integration is two-way:

**Commands flow Master → nodes (`DOWNLOAD_COMMANDS` topic):**

- `StartDownload{target_node_id, shard_metadata}` — begin fetching a shard on a specific node [`shared/types/commands.py:69-71`](/Users/leozealous/exo/src/exo/shared/types/commands.py)
- `DeleteDownload{target_node_id, model_id}` — remove a model from local disk (guarded against `read_only` models) [`shared/types/commands.py:74-76`](/Users/leozealous/exo/src/exo/shared/types/commands.py), [`coordinator.py:267-303`](/Users/leozealous/exo/src/exo/download/coordinator.py)
- `CancelDownload{target_node_id, model_id}` — stop an in-flight download, reverting status to `DownloadPending` [`shared/types/commands.py:79-81`](/Users/leozealous/exo/src/exo/shared/types/commands.py), [`coordinator.py:138-151`](/Users/leozealous/exo/src/exo/download/coordinator.py)

All three are wrapped in a `ForwarderDownloadCommand{origin, command}` envelope before publishing [`shared/types/commands.py:107-109`](/Users/leozealous/exo/src/exo/shared/types/commands.py). `DOWNLOAD_COMMANDS` is kept separate from the general `COMMANDS` topic so that the election/pause logic in the Master API (`self.paused` gate around `_send`) does not block downloads — `_send_download` bypasses the pause entirely [`master/api.py:1733-1743`](/Users/leozealous/exo/src/exo/master/api.py).

**Events flow nodes → Master (`LOCAL_EVENTS` topic, via `event_sender`):**

- `NodeDownloadProgress{download_progress}` wraps one of `DownloadPending`, `DownloadOngoing`, `DownloadFailed`, `DownloadCompleted` [`coordinator.py:78-106`](/Users/leozealous/exo/src/exo/download/coordinator.py).

The Master indexes these events, updates `State.downloads`, and rebroadcasts to all workers so that placement decisions on any node see the same download state. See [`master.md`](./master.md) for the event-indexing half of this loop and [`worker.md`](./worker.md) for how a worker's planner decides to request a download in the first place.

**Periodic reconciliation.** On startup and then every 60 seconds, `_emit_existing_download_progress` scans both the writable models dir (via `get_shard_download_status` → one tuple per known model card) and the read-only `EXO_MODELS_PATH` dirs, emitting `DownloadCompleted` / `DownloadPending` / `DownloadOngoing` events for each as appropriate [`coordinator.py:305-399`](/Users/leozealous/exo/src/exo/download/coordinator.py). This is how a freshly-started node tells the cluster "I already have model X on disk" without needing an explicit `StartDownload`.

## Gotchas

- **`resolve_allow_patterns` is currently hardcoded to `["*"]`.** The intended "download only the files this shard needs" path (using the safetensors weight map) is disabled via an early `return ["*"]` with a TODO explaining that smart downloads break sticky sessions and tensor parallel [`download_utils.py:692-704`](/Users/leozealous/exo/src/exo/download/download_utils.py). Every node currently downloads the full repo even when it only runs a subset of layers.

- **`resolve_model_in_path` requires a safetensors index.** The completeness check at `is_model_directory_complete` only validates directories that contain at least one `*.safetensors.index.json` [`download_utils.py:188-190`](/Users/leozealous/exo/src/exo/download/download_utils.py). Models without an index file will never resolve from `EXO_MODELS_PATH` even if all their weight files are present.

- **Read-only models cannot be deleted.** `_delete_download` refuses to operate on `DownloadCompleted` entries with `read_only=True`, logs a warning, and returns without side effects [`coordinator.py:267-275`](/Users/leozealous/exo/src/exo/download/coordinator.py).

- **Image models skip root-level safetensors.** `download_shard` filters out top-level `*.safetensors` for image models because weights live in component subdirs like `transformer/` and `vae/` — a root-level tensor would be a stale artifact [`download_utils.py:762-767`](/Users/leozealous/exo/src/exo/download/download_utils.py).

- **File-list cache is session-pinned.** `fetch_file_list_with_cache` uses a module-level `_fetched_file_lists_this_session: set[str]` to ensure at most one network fetch per `(model_id, revision)` per process [`download_utils.py:266-315`](/Users/leozealous/exo/src/exo/download/download_utils.py). If the HF repo changes mid-session, exo will not notice until restart.

- **SSL cert resolution has a Nix-friendly escape hatch.** `create_http_session` reads `SSL_CERT_FILE` before falling back to `certifi.where()` [`download_utils.py:415-417`](/Users/leozealous/exo/src/exo/download/download_utils.py). `HTTPS_PROXY` / `HTTP_PROXY` env vars are also honoured [`download_utils.py:423`](/Users/leozealous/exo/src/exo/download/download_utils.py).

- **`SingletonShardDownloader` is the dedup gate, not the coordinator.** Two concurrent calls to `ensure_shard` with the same `ShardMetadata` (must compare equal, not just same model_id) share one task [`impl_shard_downloader.py:61-72`](/Users/leozealous/exo/src/exo/download/impl_shard_downloader.py). The coordinator itself guards on `model_id` in `_start_download` [`coordinator.py:157-163`](/Users/leozealous/exo/src/exo/download/coordinator.py) — two different shards of the same model submitted simultaneously would race at the coordinator level but merge at the singleton level.

- **`DOWNLOAD_COMMANDS` topic bypasses the master-pause gate.** While the API's general `_send` blocks on `self.paused` during election churn, `_send_download` does not [`master/api.py:1733-1743`](/Users/leozealous/exo/src/exo/master/api.py) — downloads are allowed to continue across elections.

- **Worker has a separate download backoff layer.** Before emitting the `StartDownload` command, `Worker._download_backoff.should_proceed(model_id)` gates against flooding the event log with repeated failed downloads [`worker/main.py:153-175`](/Users/leozealous/exo/src/exo/worker/main.py). This is independent from the retry logic in `download_file_with_retry`.

- **`EXO_MODELS_PATH` is legacy alias.** The current setting is `EXO_MODELS_READ_ONLY_DIRS` (colon-separated list of paths). `EXO_MODELS_PATH` is kept as a back-compat alias [`shared/constants.py:57-60`](/Users/leozealous/exo/src/exo/shared/constants.py).

## Sources

- `/Users/leozealous/exo/src/exo/download/coordinator.py`
- `/Users/leozealous/exo/src/exo/download/shard_downloader.py`
- `/Users/leozealous/exo/src/exo/download/impl_shard_downloader.py`
- `/Users/leozealous/exo/src/exo/download/download_utils.py`
- `/Users/leozealous/exo/src/exo/download/huggingface_utils.py`
- `/Users/leozealous/exo/src/exo/download/tests/test_re_download.py`
- `/Users/leozealous/exo/src/exo/main.py`
- `/Users/leozealous/exo/src/exo/master/main.py`
- `/Users/leozealous/exo/src/exo/master/api.py`
- `/Users/leozealous/exo/src/exo/master/placement.py`
- `/Users/leozealous/exo/src/exo/worker/main.py`
- `/Users/leozealous/exo/src/exo/routing/topics.py`
- `/Users/leozealous/exo/src/exo/shared/types/commands.py`
- `/Users/leozealous/exo/src/exo/shared/constants.py`
- `/Users/leozealous/exo/README.md`

---

_Last indexed: commit c0d5bf92, 2026-04-21_
