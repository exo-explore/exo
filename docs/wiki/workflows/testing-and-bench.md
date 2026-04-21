# Testing and benchmarking exo

exo is tested at three layers — Python unit/integration tests colocated with the code they cover, a handful of multi-host integration drivers under the repo-root `tests/` tree, and Rust tests per crate. Benchmarks and quality evals live in a separate `bench/` workspace that drives a running exo cluster over its HTTP API.

## Test categories

### Python unit / integration tests (colocated)

Most Python tests live in `tests/` subdirectories next to the code they exercise — this is the pattern called out in `CLAUDE.md:114-116`. The current tree:

- `src/exo/shared/tests/` — event application, election, state serialization, Thunderbolt cycle detection, XDG paths, node-id persistence (e.g. `test_apply/test_apply_node_download.py`, `test_election.py`, `test_thunderbolt_cycles.py`).
- `src/exo/master/tests/` — placement, master loop, API error handling, Claude/Responses adapters, cancellation, topology (`test_master.py`, `test_placement.py`, `test_claude_api.py`, `test_openai_responses_api.py`, `test_topology.py`).
- `src/exo/routing/tests/` — pub/sub event buffer (`test_event_buffer.py`).
- `src/exo/worker/tests/unittests/` — plan/runner/MLX/download subtrees:
  - `test_plan/` — download + load, runner lifecycle, task forwarding, warmup.
  - `test_runner/` — runner supervisor, GLM/GPT-OSS/tool-call parsing, event ordering, DSML e2e.
  - `test_mlx/` — batch vs generate, KV prefix cache, tokenizers, pipeline prefill callbacks, auto-parallel.
- `src/exo/worker/engines/mlx/tests/test_batch_generate.py` — engine-level MLX test.
- `src/exo/download/tests/` — verification, offline mode, re-download (`test_download_verification.py`, `test_offline_mode.py`, `test_re_download.py`).
- `src/exo/utils/tests/` — channels, power sampler, tagged unions.

Shared pytest fixtures live in `conftest.py` files at `src/exo/shared/tests/conftest.py`, `src/exo/master/tests/conftest.py`, `src/exo/worker/tests/unittests/conftest.py`, and `src/exo/worker/tests/unittests/test_mlx/conftest.py`.

### Multi-host integration drivers (repo-root `tests/`)

The top-level `tests/` directory is not a pytest tree — it's a set of scripts for driving exo across real hardware. pytest explicitly ignores one of them: `pyproject.toml:139` sets `addopts = "-m 'not slow' --ignore=tests/start_distributed_test.py"`.

| File | Purpose |
|---|---|
| `tests/start_distributed_test.py` | Launches a distributed `jaccl`/`ring` test across SSH hosts using tailscale IP discovery and each node's `/run_test` endpoint on port 52414 (`tests/start_distributed_test.py:10-85`). |
| `tests/headless_runner.py` | In-process FastAPI harness that boots a runner without the full master/worker stack — used by `start_distributed_test.py` via `/run_test` (`tests/headless_runner.py:1-50`). |
| `tests/auto_bench.sh` | SSH-builds the current commit on each host via `nix build github:exo-explore/exo/<commit>`, waits for `http://host:52415/models`, then runs `exo-bench` for every model on the cluster (`tests/auto_bench.sh:13-55`). |
| `tests/eval_tool_calls.sh` | Same SSH bootstrap pattern, runs `exo-eval-tool-calls` per model (`tests/eval_tool_calls.sh:13-55`). |
| `tests/run_exo_on.sh` | SSH helper for launching exo on remote hosts. |
| `tests/get_all_models_on_cluster.py` | Queries one cluster node for the shared model set; wrapped by the Nix script `exo-get-all-models-on-cluster` (`python/parts.nix:187`). |

### Rust tests

Rust crates each have a `tests/` directory. Today these are mostly placeholders plus one Python-driven integration:

- `rust/networking/tests/dummy.rs` — placeholder (`rust/networking/tests/dummy.rs:1-7`).
- `rust/exo_pyo3_bindings/tests/dummy.rs` — placeholder.
- `rust/exo_pyo3_bindings/tests/test_python.py` — async pytest that exercises the PyO3 `NetworkingHandle` + gossipsub publish from Python (`rust/exo_pyo3_bindings/tests/test_python.py:1-30`).

## Running Python tests

Test runner config is in `pyproject.toml:134-140`: `asyncio_mode = "auto"`, `EXO_TESTS=1` is injected as an env var, and the `slow` marker is deselected by default.

```bash
# Default run (excludes slow tests, ignores tests/start_distributed_test.py)
uv run pytest

# Include slow tests
uv run pytest -m ""

# Run a specific file
uv run pytest src/exo/shared/tests/test_election.py

# Run a specific function
uv run pytest src/exo/shared/tests/test_election.py::test_function_name
```

These commands are the canonical set from `CLAUDE.md:20-28`. `EXO_TESTS=1` is set automatically via `pyproject.toml:138` so tests that branch on it don't need manual setup.

Static checks that gate CI live alongside tests (`CLAUDE.md:38-60`):

```bash
uv run basedpyright   # strict mode, 0 errors required (pyproject.toml:78-94)
uv run ruff check     # pyproject.toml:123-132
nix fmt               # formatter
```

## Running Rust tests

The full workspace test command runs every crate:

```bash
cargo test --workspace
```

In CI this is wrapped by `nix flake check` (`.github/workflows/pipeline.yml:106-107`), which runs Rust tests as part of the flake checks. The PyO3 integration test (`rust/exo_pyo3_bindings/tests/test_python.py`) is collected by the top-level `uv run pytest` run because `pyproject.toml:135` sets `pythonpath = "."`.

## Running benchmarks

The benchmark suite is a separate `uv` workspace member. `pyproject.toml:59-60` declares `members = ["rust/exo_pyo3_bindings", "bench"]`, and `bench/pyproject.toml:1-19` pins its own deps (`httpx`, `transformers`, `lm-eval[api,math]`, `human-eval`, `math-verify`, `datasets`). All bench tools target a **running exo cluster** on `localhost:52415` (or `$EXO_HOST:$EXO_PORT` — see `bench/harness.py:454-457`). See [running-exo.md](./running-exo.md) for how to start one.

### Bench layout

```
bench/
  bench.toml              # manifest: list of suite files (bench/bench.toml:5)
  scenarios.toml          # tool-calling scenarios for eval_tool_calls
  single-m3-ultra.toml    # single-node M3 Ultra benchmark suite
  harness.py              # shared ExoClient + placement/download/ready helpers
  exo_bench.py            # perf bench driver (prefill + token gen sweeps)
  exo_eval.py             # AA-style quality evals (gpqa, mmlu_pro, aime, humaneval, livecodebench)
  eval_tool_calls.py      # tool-calling scenario eval
  parallel_requests.py    # parallel request stress
  eval_configs/
    models.toml           # per-model temperature/top_p/max_tokens/reasoning (bench/eval_configs/models.toml:1-20)
  src/exo_bench/          # shared bench package (bench/src/exo_bench/)
  vendor/
    lcb_testing_util.py   # vendored livecodebench harness (bench/vendor/lcb_testing_util.py)
```

### Core bench drivers

**`bench/exo_bench.py`** — performance benchmarking. Driver description and flags in `bench/exo_bench.py:1-16`; it uses the shared harness in `bench/harness.py`. Typical invocation (from the Nix wrapper exposed at `python/parts.nix:184`):

```bash
uv run exo-bench --model mlx-community/Qwen3-30B-A3B-4bit --pp 512 2048 --tg 128
# or via nix:
nix run .#exo-bench -- --model <short-id-or-hf-id> --pp 128 4096 --tg 128 --stdout
```

**`bench/exo_eval.py`** — Artificial-Analysis-style quality evals. Supported tasks are documented in `bench/exo_eval.py:7-14`: `gpqa_diamond`, `mmlu_pro`, `aime_2024`, `aime_2025`, `humaneval`, `livecodebench`. Per-model reasoning/temperature configs are read from `bench/eval_configs/models.toml` (see its header `bench/eval_configs/models.toml:1-20`).

```bash
uv run exo-eval --model <model-id> --tasks gpqa_diamond
uv run exo-eval --model <model-id> --tasks humaneval,livecodebench --limit 50
uv run exo-eval --model <model-id> --tasks gpqa_diamond --compare-concurrency 1,4
```

**`bench/eval_tool_calls.py`** — tool-calling eval driven by `bench/scenarios.toml`. The scenarios cover single tool calls, multi-turn, chained calls, nested-schema regressions, tool-name integrity (harmony tokens), and negative cases that should *not* call a tool (`bench/scenarios.toml:1-307`). Usage comment is in `bench/eval_tool_calls.py:11-16`:

```bash
uv run exo-eval-tool-calls --model <model-id>
uv run exo-eval-tool-calls --model <model-id> --scenarios weather_simple calculator_multi_turn
```

### Benchmark manifests

`bench/bench.toml:1-6` is the canary manifest that `include`s one or more suite files. Today it points at `bench/single-m3-ultra.toml`, which defines:

- Cluster constraints: `All(Chip(m3_ultra))`, `Hosts(=1)`, `All(GpuCores(=80))`, `All(MacOsBuild(=25D125))` (`bench/single-m3-ultra.toml:4-9`).
- Topology: `type = "none"` (`bench/single-m3-ultra.toml:11-12`).
- Default sweeps: `pp = [512, 2048, 8192, 16384]`, `tg = 128` (`bench/single-m3-ultra.toml:14-17`).
- ~40 model entries, each with memory constraints (e.g. `All(Memory(>=96GiB))` through `All(Memory(>=512GiB))`).

### Bench harness

`bench/harness.py` provides `ExoClient`, placement filters, download/ready waiters, and the `add_common_instance_args` argparser shared by all drivers (`bench/harness.py:19-67`, `bench/harness.py:454-507`). Key capabilities:

- Resolve short id vs HuggingFace id (`bench/harness.py:168-196`).
- Filter placements by sharding (`pipeline`/`tensor`/`both`) and instance meta (`ring`/`jaccl`/`both`) with `--skip-pipeline-jaccl` / `--skip-tensor-ring` (`bench/harness.py:199-270`).
- `run_planning_phase` checks disk, optionally frees space via `--danger-delete-downloads`, and waits for downloads to finish (`bench/harness.py:297-451`).
- `wait_for_instance_ready` and `wait_for_instance_gone` for lifecycle assertions (`bench/harness.py:114-166`).

### Driving benches across a real cluster

`tests/auto_bench.sh:13-55` is the reference "bench a commit across N hosts" script: it SSHes to each host, runs `nix build github:exo-explore/exo/<commit>`, starts exo, waits for `/models`, enumerates the shared model set via `exo-get-all-models-on-cluster`, and calls `exo-bench` per model with `--pp 128 4096 --tg 128 --stdout --skip-tensor-ring`. `tests/eval_tool_calls.sh:13-55` mirrors it for `exo-eval-tool-calls`.

The Nix-exposed CLIs are declared in `python/parts.nix:184-187`: `exo-bench`, `exo-eval`, `exo-eval-tool-calls`, `exo-get-all-models-on-cluster`.

## CI

CI configuration lives in `.github/workflows/`:

- **`.github/workflows/pipeline.yml`** — `ci-pipeline`, runs on every push and on PRs into `staging` and `main` (`.github/workflows/pipeline.yml:3-8`). Matrix across `aarch64-darwin` (macos-26), `x86_64-linux` (ubuntu-latest), and `aarch64-linux` (ubuntu-24.04-arm) (`.github/workflows/pipeline.yml:14-23`). Steps:
  1. Install Nix + Cachix cache (`.github/workflows/pipeline.yml:30-38`).
  2. Build the Metal toolchain on macOS — from cachix if available, otherwise extract it from the installed Xcode via `xcodebuild -downloadComponent MetalToolchain`, pack into a NAR, and add it to the Nix store (`.github/workflows/pipeline.yml:40-95`).
  3. `nix flake show` → `nix build` every package and devshell output for the current system (`.github/workflows/pipeline.yml:97-104`).
  4. `nix flake check` — runs formatting checks, linting, and Rust tests (`.github/workflows/pipeline.yml:106-107`).
  5. **macOS only:** build the test env (`nix build '.#exo-test-env' --option sandbox relaxed`), then run pytest *outside* the Nix sandbox because MLX needs GPU access: `$TEST_ENV/bin/python -m pytest src -m "not slow" --import-mode=importlib` with `EXO_TESTS=1`, `EXO_DASHBOARD_DIR`, `EXO_RESOURCES_DIR` exported (`.github/workflows/pipeline.yml:109-120`). The `exo-test-env` venv is defined in `python/parts.nix:128` and includes everything except the `exo-bench` workspace member.

- **`.github/workflows/build-app.yml`** — `Build EXO macOS DMG`, release workflow for `v*` tags and pushes to `test-app`. Builds + signs + notarizes the DMG, generates Sparkle appcast, uploads to S3, publishes the draft GitHub Release (`.github/workflows/build-app.yml:14-22`, `.github/workflows/build-app.yml:294-448`). It does not run tests — it assumes the ci-pipeline already passed.

## Pre-commit hooks

The `.githooks/` directory holds **Git-LFS hooks only** — `post-checkout`, `post-commit`, `post-merge`, `pre-push` each simply delegate to `git lfs <name> "$@"` and abort with a message if `git-lfs` isn't on PATH (`.githooks/pre-push:1-3`, `.githooks/post-checkout:1-3`, `.githooks/post-commit:1-3`, `.githooks/post-merge:1-3`). To enable them point `core.hookspath` at `.githooks/` (the error message in each hook references this).

There is no automated pre-commit lint/test runner. The required local gate is the manual sequence from `CLAUDE.md:38-62`:

```bash
# All four must pass before committing — CI will fail otherwise
uv run basedpyright   # 0 errors (strict)
uv run ruff check
nix fmt               # stage any files it rewrites
uv run pytest
```

Chained:

```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

`CONTRIBUTING.md:156-158` explicitly notes that exo relies heavily on manual testing today and that the automated story is being built out — add automated tests where possible.

## Adding a new test

Conventions distilled from the existing tree:

1. **Colocate with code.** Put Python tests in a `tests/` subdirectory of the package you're testing (e.g. `src/exo/master/tests/test_my_feature.py`). This is the pattern in `CLAUDE.md:114-116` and mirrors every existing subpackage — `src/exo/shared/tests/`, `src/exo/master/tests/`, `src/exo/routing/tests/`, `src/exo/worker/tests/unittests/`, `src/exo/download/tests/`, `src/exo/utils/tests/`.
2. **File naming.** `test_*.py` — pytest's default discovery plus the `pythonpath = "."` in `pyproject.toml:135` is all you need; no extra registration.
3. **Async is the default.** `pyproject.toml:136` sets `asyncio_mode = "auto"`, so `async def test_...` works without `@pytest.mark.asyncio`.
4. **Slow tests.** Mark with `@pytest.mark.slow` — they're skipped by default via `pyproject.toml:137,139`. Run them with `uv run pytest -m ""`.
5. **Fixtures go in `conftest.py`.** Follow the style in `src/exo/shared/tests/conftest.py` (session-scoped `event_loop`, autouse reset fixtures, typed helpers).
6. **Types are strict.** basedpyright runs in strict mode with `failOnWarnings = true` (`pyproject.toml:80-81`) and `include` covers `src` and `bench` (`pyproject.toml:79`) — your test file must type-check.
7. **For new components,** see [components/master.md](../components/master.md) and [architecture/module-boundaries.md](../architecture/module-boundaries.md) for where new test subtrees should live relative to the component layout.
8. **Don't add to repo-root `tests/`** unless you're writing a cluster-level harness. That directory is for multi-host drivers and is not part of the pytest run (`pyproject.toml:139` ignores `start_distributed_test.py`; the `.sh` files aren't pytest targets).
9. **For benches,** add a model entry to the appropriate suite under `bench/` (e.g. `bench/single-m3-ultra.toml`) or a new scenario to `bench/scenarios.toml` following the `[[scenarios]]` pattern in `bench/scenarios.toml:68-98`. Per-model reasoning/sampling config goes in `bench/eval_configs/models.toml`.

## Sources

- `/Users/leozealous/exo/pyproject.toml`
- `/Users/leozealous/exo/bench/pyproject.toml`
- `/Users/leozealous/exo/bench/bench.toml`
- `/Users/leozealous/exo/bench/single-m3-ultra.toml`
- `/Users/leozealous/exo/bench/scenarios.toml`
- `/Users/leozealous/exo/bench/eval_configs/models.toml`
- `/Users/leozealous/exo/bench/harness.py`
- `/Users/leozealous/exo/bench/exo_bench.py`
- `/Users/leozealous/exo/bench/exo_eval.py`
- `/Users/leozealous/exo/bench/eval_tool_calls.py`
- `/Users/leozealous/exo/bench/src/exo_bench/`
- `/Users/leozealous/exo/bench/vendor/lcb_testing_util.py`
- `/Users/leozealous/exo/tests/auto_bench.sh`
- `/Users/leozealous/exo/tests/eval_tool_calls.sh`
- `/Users/leozealous/exo/tests/headless_runner.py`
- `/Users/leozealous/exo/tests/start_distributed_test.py`
- `/Users/leozealous/exo/tests/get_all_models_on_cluster.py`
- `/Users/leozealous/exo/.github/workflows/pipeline.yml`
- `/Users/leozealous/exo/.github/workflows/build-app.yml`
- `/Users/leozealous/exo/.githooks/pre-push`
- `/Users/leozealous/exo/.githooks/post-checkout`
- `/Users/leozealous/exo/.githooks/post-commit`
- `/Users/leozealous/exo/.githooks/post-merge`
- `/Users/leozealous/exo/rust/networking/tests/dummy.rs`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/tests/test_python.py`
- `/Users/leozealous/exo/src/exo/shared/tests/conftest.py`
- `/Users/leozealous/exo/src/exo/master/tests/`, `src/exo/worker/tests/unittests/`, `src/exo/routing/tests/`, `src/exo/download/tests/`, `src/exo/utils/tests/`, `src/exo/worker/engines/mlx/tests/`
- `/Users/leozealous/exo/python/parts.nix`
- `/Users/leozealous/exo/CONTRIBUTING.md`
- `/Users/leozealous/exo/CLAUDE.md`

Last indexed: c0d5bf92, 2026-04-21
