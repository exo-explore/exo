# Plan: native Windows devices in an exo cluster (Tinygrad / OpenCL)

Status: **parked**
Date: 2026-09-27
Branch: `cursor/tinygrad-engine-skeleton-2dfd` (base `main`; branch started from the
Tinygrad engine skeleton commits through `6e8a8168`)

This document records the plan, the work in progress, and the alternatives
research so the effort can be resumed later.

## Objective

Let a native Windows machine (AMD, CUDA, or CPU) join an existing exo cluster and
contribute inference, with the same operational quality of life an existing Mac
node has: discovery, device reporting, model selection and download, placement,
runner lifecycle, cancellation, and failure recovery.

Acceptance criterion: clean Windows install → cluster join → model selection
through the normal UI/API → actual CPU/GPU contribution → reliable lifecycle and
recovery.

## Scope decisions

- Every device participating in a Windows-shared model runs Tinygrad. An
  MLX–Tinygrad cross-engine pipeline is out of scope.
- Broad MLX feature parity was never in scope; this follows the branch's existing
  feature subset (dense Llama / Qwen2 / Qwen3, pipeline parallelism only).
- Quality-of-life features exo already provides and that should be reused:
  cluster discovery and election, replicated state, placement commands, download
  coordination (progress, cancel, retry), runner lifecycle
  (`create → connect → load → warmup → ready`), task cancellation, runner failure
  events, and instance restart backoff. See `src/exo/worker/plan.py`,
  `src/exo/worker/main.py`, `src/exo/download/coordinator.py`.
- Current exo deletes instances whose nodes vanish. Seamless live migration or
  re-sharding is a new feature, not a parity requirement. See
  `src/exo/master/main.py` (`_plan`).

## Work completed in this effort (uncommitted unless noted)

Streaming correctness (verified by isolated execution):
- `src/exo/backends/tinygrad_generate.py`: `advance_stop_buffer` /
  `stop_prefix_length` replace the stateless trim, so a stop string split across
  tokens no longer leaks or duplicates.
- `src/exo/backends/tinygrad_tokenizer.py`: stateful incremental UTF-8 decoding,
  so a character split across byte-level tokens is emitted once.
- `src/exo/backends/tinygrad_engine.py`: withheld-prefix buffer on
  `_ActiveGeneration`, flushed at finish.

Pipeline transport (verified by isolated execution):
- `src/exo/backends/tinygrad_pipeline.py`: nonblocking sockets with `select` for
  both directions; `_send_all` waits for writability and bounds the send with a
  timeout. Reading no longer leaves the socket nonblocking.

GGUF correctness:
- `src/exo/backends/tinygrad_checkpoint.py`: Q/K rotary permutation applies to
  Llama GGUF only (Qwen2/Qwen3 converters do not interleave); GGUF rope scaling is
  read (linear applied, yarn/unknown rejected).
- GGUF checkpoint management: `gguf_split_groups`, `gguf_checkpoint_paths`,
  `_split_group_is_complete`, variant-aware `_gguf_paths`,
  `read_gguf_checkpoint(directory, preferred)`. Ambiguous directories raise
  instead of merging quantizations.
- `src/exo/shared/models/model_cards.py`: new optional `gguf_filename`.
- `src/exo/download/download_utils.py`: `gguf_allow_patterns` downloads only the
  selected variant; GGUF-aware local completeness
  (`_scan_gguf_directory`), `_scan_local_gguf_files` for offline discovery,
  `build_model_path(model_id, card)`.
- `src/exo/backends/tinygrad_weights.py` and `tinygrad_engine.py` pass the card's
  `gguf_filename` through load paths.

Windows enablement (structurally complete, not platform-validated):
- `WinAMD` maps to Tinygrad `CL` (OpenCL) instead of `AMD`, whose runtime asserts
  against `win32`; `tinygrad_memory.py` gained a `CL` path using the VRAM override.
- `pyproject.toml` adds a `win32` environment and pins the Tinygrad extra to
  `>=0.14.0`; `uv.lock` re-locked with Windows wheels/markers.
- `rust/exo_rs`: `pidfile` module and `pidfile-rs` dependency gated to `cfg(unix)`.
- `src/exo/shared/pidfile.py`: cross-platform PID file (Unix uses `exo_rs`,
  Windows uses an `msvcrt` lock). `src/exo/main.py` imports from it.
- `src/exo/shared/constants.py`: model-directory env lists split on `os.pathsep`
  so `C:\...` is not split.
- `PLATFORMS.md`: Windows tiers and the `gguf_filename` contract.

Placement / API:
- `src/exo/master/placement.py`: backend compatibility is filtered before choosing
  the smallest cycle.
- `src/exo/api/main.py`: placement previews enumerate `InstanceMeta.Tinygrad`;
  per-node preview memory follows the layer allocation, not an even split.

Tests added: `src/exo/backends/tests/test_tinygrad_{checkpoint,engine,generate,memory,pipeline}.py`,
`src/exo/download/tests/test_gguf_downloads.py`.

Verification done: syntax parse of all changed Python files; all 123 built-in
model-card TOMLs parse; isolated harnesses for streaming, transport, and GGUF
selection pass; `git diff --check` clean.

Verification **not** done: `pytest`, `basedpyright`, `ruff`, `nix fmt`; no Windows
execution; no AMD/CUDA hardware. `cargo check` failed on a pre-existing
`pyo3-stub-gen`/`pyo3` error unrelated to the Rust change (`Cargo.lock` unchanged).

## Remaining plan

### 1. Runtime feasibility (P0, needs hardware)

- Tinygrad 0.14 opens devices from the **first OpenCL platform only**, so a
  mixed-GPU Windows machine may not select AMD even with `CL:1`
  (<https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/runtime/ops_cl.py>).
- The OpenCL renderer enables FP16 only when the device advertises `cl_khr_fp16`;
  the branch realizes GGUF as float16
  (<https://github.com/tinygrad/tinygrad/blob/v0.14.0/tinygrad/renderer/cstyle.py#L308-L342>).
- Required: enumerate all platforms/devices; persist an explicit adapter choice;
  run a separate bounded probe (load driver, open context, compile and execute
  representative ops, verify numerics); advertise the accelerator as usable only
  after the probe succeeds.

### 2. Windows lifecycle (P0)

- Hard blocker: `src/exo/utils/async_process.py` imports Unix-only
  `multiprocessing.resource_sharer.DupFd` and drains pipe fds with
  `anyio.wait_readable`. Needs a socketpair/handle-based capture that works on
  Windows, preserving runner stdout/stderr diagnostics.
- Validate spawn-channel behavior, startup failures, cancellation, and cleanup.
- Ensure close/restart releases GPU allocations and pipeline sockets.
- Fix and test `src/exo/shared/pidfile.py`: the Windows lock is taken at the
  file's current position and released after a write, rather than a fixed range.

### 3. Resource accounting (P0)

- The `CL` path reports the manual override as both total and available, and it
  never changes as models load (`src/exo/backends/tinygrad_memory.py`).
- Placement budgets `model_card.storage_size` (checkpoint bytes), not the realized
  float16 shard plus KV cache, temporaries, and transfer buffers
  (`src/exo/master/placement_utils.py`).
- Candidate sources: OpenCL `CL_DEVICE_GLOBAL_MEM_SIZE` /
  `CL_DEVICE_MAX_MEM_ALLOC_SIZE`
  (<https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html>);
  DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` for process budget/usage
  (<https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo>),
  noting the info-gatherer process cannot measure a separate runner's allocations.
- Report capacity, ceiling, current reservations, estimate and source, max single
  allocation, and host RAM needs separately.

### 4. Normal user workflow (P0/P1)

- GGUF is now variant-aware (this effort), but a GGUF-only repository still cannot
  be registered through the normal flow: `ModelCard.fetch_from_hf` requires a
  safetensors index. Add a GGUF registration path that sets `gguf_filename` and a
  storage/realized size.
- Publish curated cards with verified Tinygrad/Windows backends; today no built-in
  card advertises them.

### 5. Dashboard (P1)

- `dashboard/src/routes/+page.svelte` and `lib/stores/app.svelte.ts` are MLX-only
  (`MlxRing`/`MlxJaccl`); single-node onboarding forces `MlxRing`. Needs Tinygrad
  runtime selection, participating/excluded node display, device+memory+driver
  reporting, and accurate lifecycle states.

### 6. Recovery and diagnostics (P1)

- Add progress deadlines (compile, prefill, decode, transport) and classify
  OpenCL compile/allocation/device-loss/pipeline errors. Today
  `src/exo/worker/runner/diagnostics.py` classifies only Metal and MLX-ring
  failures; Tinygrad errors surface as `RunnerUnknown`.
- Add handshake identity (instance/rank/model/protocol/version) and request
  sequencing; exercise cancel-immediately-followed-by-request, sleep/resume, and
  disconnect/rejoin.

### 7. Launcher and packaging (P1/P2)

- `src/exo/windows/launcher.py` has only Start and process-liveness. Add saved
  settings, Stop/Restart/Close, distinct process/connection/accelerator status,
  log access, Windows hardware/driver identification, and validated
  firewall/discovery behavior.

### Acceptance gates

| Milestone | Gate |
|---|---|
| Runtime feasibility | A real Windows AMD GPU runs a supported model correctly through Tinygrad OpenCL on the intended adapter |
| Windows lifecycle | Repeated start/stop/cancel/restart leaves no orphaned resources |
| Resource correctness | Placement predicts actual shard memory; multiple instances and unload are accounted for |
| Normal workflow | Dashboard → correct GGUF download → distributed load → ready → generation without manual env editing |
| Operational reliability | Offline restart, mixed-GPU selection, disconnect/rejoin, sleep/resume, driver failure all predictable |

CI currently has no Windows job (`nix` matrix is macOS/Linux; `pytest` runs on
macOS only). Windows CPU CI is needed for platform plumbing; native AMD needs a
real Windows machine.

## Research context: alternatives to Tinygrad for Windows AMD (including Vulkan)

Read-only research; no hardware validation was performed. Conclusion: none of the
candidates is a drop-in distributed layer engine.

### llama.cpp (Vulkan) + RPC — recommended first experiment

- Provides real native Windows AMD execution via Vulkan and its own RPC
  (`ggml-rpc-server`) for distributed execution. MIT.
- Suggested shape: a supervised `llama-server` coordinator plus supervised
  `ggml-rpc-server` workers from one pinned revision. Keep one tokenizer/sampler
  at the coordinator; adapt coordinator output into exo's `Chunk`, cancellation,
  completion, and failure events; make endpoint reservation, startup, readiness,
  shutdown, and resource release part of the builder/engine lifecycle.
- Shortest route to proving distributed execution and memory behavior before
  investing in native bindings.

### PyTorch HIP as a layer engine — plausible now

- ROCm 10.0 lists native Windows PyTorch 2.13.0 on Python 3.11–3.14, and the
  package index contains a `cp313-cp313-win_amd64` wheel, so exo's Python 3.13 is
  no longer an automatic blocker. Eligibility still depends on GPU/Windows/driver.
  (<https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html>,
  <https://stable.repo.amd.com/rocm/whl-next/torch/>)
- Design: port the dense Llama/Qwen layer ops from
  `src/exo/backends/tinygrad_llama.py` to PyTorch, reuse the existing
  `HiddenStateBuffer` TCP protocol and rank sequencing, one sampling owner at the
  last rank. Eager ops first; no Windows Triton/FlashAttention dependency.
- Caveat: GGUF through a PyTorch loader does not establish packed quantized GPU
  execution; a deliberate quantized-kernel path would be required
  (<https://huggingface.co/docs/transformers/en/gguf>).
- `pyproject.toml` pins PyTorch 2.10.0 for Darwin/Linux only; a Windows HIP extra
  and index policy would be needed.

### Other candidates

- **Distributed Llama**: genuinely partitions computation, but as *tensor*
  parallelism with its own protocol and `.m`/`.t` model/tokenizer formats, not
  GGUF or exo's layer pipeline
  (<https://github.com/b4rtaz/distributed-llama>).
- **ONNX Runtime / DirectML / Windows ML**: credible Windows execution, not a
  distributed LLM engine. Import each shard as a session and move activations over
  exo's TCP; work is export, partitioning, cache updates, bucketed shapes, and EP
  coverage. DirectML is sustained engineering; Windows ML is the active ORT-based
  path (<https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>,
  <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview>).
- **MLC/TVM Vulkan**: real Windows Vulkan and real pipeline-parallel compiler code,
  but their combination does not work — the distributed engine selects NCCL/RCCL
  and rejects other devices, and MLC rejects GGUF as a direct weight input
  (<https://github.com/mlc-ai/mlc-llm/blob/9fa644f54b04983adea4d0168f49fc6af4a893ba/cpp/serve/engine.cc#L827-L877>,
  <https://github.com/mlc-ai/mlc-llm/blob/9fa644f54b04983adea4d0168f49fc6af4a893ba/python/mlc_llm/support/auto_weight.py#L172-L178>).
- **IREE Vulkan**: covers Windows/AMD deployment, but its Vulkan driver returns
  `UNIMPLEMENTED` for collective channels, and GGUF parameter archive support is
  not automatic quantized-kernel support
  (<https://iree.dev/guides/deployment-configurations/gpu-vulkan/>,
  <https://github.com/iree-org/iree/blob/5cf0a123b0c53ea8cb6a02ae923f46f7b3224eb8/runtime/src/iree/hal/drivers/vulkan/logical_device.c#L957-L965>).
- **PyTorch DirectML**: published `torch-directml` remains `0.2.5.dev`, pins
  PyTorch 2.4.1, and ships wheels only through Python 3.12 — conflicts with exo's
  3.13 (<https://pypi.org/pypi/torch-directml/json>).

### Proof-of-concept gates (proposed)

Native Windows execution; real remote ownership (terminating a worker interrupts
the request); teacher-forced logits within a pre-set tolerance; peak memory agrees
with assigned weights/KV/workspace; packed-quantized execution verified from device
allocations (not file size); ≥10 load/generate/cancel/unload/rejoin cycles with no
leaks; disconnect during prefill and decode fails within a chosen timeout and
recovers; p50/p95 first-token and inter-token latency, network bytes, coordinator
utilization; and one model that exceeds either worker alone but fits the combined
budget.

Recommended highest-value next experiment: **one exo-managed llama.cpp instance
spanning a native Windows AMD Vulkan worker and a CUDA or CPU peer**, with logged
ownership and measured memory.

## How to resume

1. Confirm the working tree state on `cursor/tinygrad-engine-skeleton-2dfd`.
2. Run the required checks (`uv run basedpyright && uv run ruff check && nix fmt && uv run pytest`)
   and fix anything the unverified changes broke.
3. Start at Remaining plan item 1 (runtime feasibility) with real Windows AMD
   hardware, or item 2 (Windows lifecycle) if only a Windows CPU/CUDA box is
   available.
4. Do not declare native Windows support complete until runner startup,
   stdout/stderr capture, cancellation, and shutdown work through a
   Windows-compatible process implementation.
