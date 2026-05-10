# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

exo is a distributed AI inference system that connects multiple devices into a cluster. It enables running large language models across multiple machines using MLX as the inference backend and libp2p for peer-to-peer networking.

## Build & Run Commands

```bash
# Build the dashboard (required before running exo)
cd dashboard && npm install && npm run build && cd ..

# Run exo (starts both master and worker with API at http://localhost:52415)
uv run exo

# Run with verbose logging
uv run exo -v   # or -vv for more verbose

# Run tests (excludes slow tests by default)
uv run pytest

# Run all tests including slow tests
uv run pytest -m ""

# Run a specific test file
uv run pytest src/exo/shared/tests/test_election.py

# Run a specific test function
uv run pytest src/exo/shared/tests/test_election.py::test_function_name

# Type checking (strict mode)
uv run basedpyright

# Linting
uv run ruff check

# Format code (using nix)
nix fmt
```

## Pre-Commit Checks (REQUIRED)

**IMPORTANT: Always run these checks before committing code. CI will fail if these don't pass.**

```bash
# 1. Type checking - MUST pass with 0 errors
uv run basedpyright

# 2. Linting - MUST pass
uv run ruff check

# 3. Formatting - MUST be applied
nix fmt

# 4. Tests - MUST pass
uv run pytest
```

Run all checks in sequence:
```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

If `nix fmt` changes any files, stage them before committing. The CI runs `nix flake check` which verifies formatting, linting, and runs Rust tests.

## Architecture

### Node Composition
A single exo `Node` (src/exo/main.py) runs multiple components:
- **Router**: libp2p-based pub/sub messaging via Rust bindings (exo_pyo3_bindings)
- **Worker**: Handles inference tasks, downloads models, manages runner processes
- **Master**: Coordinates cluster state, places model instances across nodes
- **Election**: Bully algorithm for master election
- **API**: FastAPI server for OpenAI-compatible chat completions

### Message Flow
Components communicate via typed pub/sub topics (src/exo/routing/topics.py):
- `GLOBAL_EVENTS`: Master broadcasts indexed events to all workers
- `LOCAL_EVENTS`: Workers send events to master for indexing
- `COMMANDS`: Workers/API send commands to master
- `ELECTION_MESSAGES`: Election protocol messages
- `CONNECTION_MESSAGES`: libp2p connection updates

### Event Sourcing
The system uses event sourcing for state management:
- `State` (src/exo/shared/types/state.py): Immutable state object
- `apply()` (src/exo/shared/apply.py): Pure function that applies events to state
- Master indexes events and broadcasts; workers apply indexed events

### Key Type Hierarchy
- `src/exo/shared/types/`: Pydantic models for all shared types
  - `events.py`: Event types (discriminated union)
  - `commands.py`: Command types
  - `tasks.py`: Task types for worker execution
  - `state.py`: Cluster state model

### Rust Components
Rust code in `rust/` provides:
- `networking`: libp2p networking (gossipsub, peer discovery)
- `exo_pyo3_bindings`: PyO3 bindings exposing Rust to Python
- `system_custodian`: System-level operations

### Dashboard
Svelte 5 + TypeScript frontend in `dashboard/`. Build output goes to `dashboard/build/` and is served by the API.

## Code Style Requirements

From .cursorrules:
- Strict, exhaustive typing - never bypass the type-checker
- Use `Literal[...]` for enum-like sets, `typing.NewType` for primitives
- Pydantic models with `frozen=True` and `strict=True`
- Pure functions with injectable effect handlers for side-effects
- Descriptive names - no abbreviations or 3-letter acronyms
- Catch exceptions only where you can handle them meaningfully
- Use `@final` and immutability wherever applicable

## Testing

Tests use pytest-asyncio with `asyncio_mode = "auto"`. Tests are in `tests/` subdirectories alongside the code they test. The `EXO_TESTS=1` env var is set during tests.

Integration tests live in `tests/` (root) and are opt-in via `--ignore=tests` in the default pytest addopts. They require an `eco`-managed cluster:

```bash
uv run pytest tests/ -v                    # constraint-driven host pick
uv run pytest tests/ -v --hosts s4         # explicit host override
```

## Benchmarking

Benchmarks live in `bench/`. The framework is a CLI with subcommands; each benchmark is a small library module under `bench/lib/<name>.py` plus a CLI front-end under `bench/cli/<name>.py`.

```
bench/
├── lib/                       # composable, typed building blocks
│   ├── prompt.py              # PromptSizer, load_tokenizer_for_bench
│   ├── completion.py          # run_one_completion + typed payloads
│   ├── session.py             # BenchSession (cluster + client + instance)
│   ├── results.py             # RunMetadata, ResultsBundle, JSON schema
│   ├── model_meta.py          # HF API: total weights size, max context, layers
│   ├── cluster.py             # managed_cluster + managed_instance ctx-managers
│   └── context_scaling.py     # prompt-TPS / decode-TPS vs context-size sweep
├── cli/                       # CLI subcommands
│   ├── _common.py             # shared argparse args + SharedOptions
│   ├── context_scaling.py     # `python -m bench.cli context-scaling …`
│   └── __main__.py            # subcommand dispatcher
└── exo_bench.py, prefill_decode_bench.py
                                # legacy CLI scripts; PromptSizer / run_one_completion
                                # / load_tokenizer_for_bench are re-exports of bench.lib.
```

Run a benchmark:

```bash
# Defaults assume a multi-node, Thunderbolt-connected cluster with tensor
# parallelism + JACCL: --sharding Tensor --comm MlxJaccl --thunderbolt a2a.
# Memory + disk minimums are auto-derived from HF metadata.
uv run python -m bench.cli context-scaling \
  --model mlx-community/Qwen3-30B-A3B-4bit --nodes 2 --num-steps 32

# Single-node smoke: opt out of TB / tensor / jaccl
uv run python -m bench.cli context-scaling --hosts s4 \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit --num-steps 4 \
  --sharding Pipeline --comm MlxRing --thunderbolt none

# From a TOML config (CLI flags override config values)
uv run python -m bench.cli context-scaling \
  --config bench/configs/context_scaling.example.toml --hosts s4,s9
```

Shared CLI flags (every subcommand inherits these via `bench/cli/_common.py`):
- `--config <path>.toml` — load run parameters from a TOML file
- `--model`, `--sharding {Pipeline,Tensor}` (default Tensor), `--comm {MlxRing,MlxJaccl}` (default MlxJaccl), `--min-nodes` — placement
- `--hosts`, `--nodes` (number of cluster hosts; distinct from `--min-nodes`), `--thunderbolt {a2a,ring,none}` (default a2a), `--chip` — host pool
- `--min-memory-gb`, `--max-memory-gb`, `--min-disk-gb`, `--max-disk-gb` (minimums auto-derived from HF model size when not supplied)
- `--evict-downloads` (default on; auto-evicts smallest-first when disk is short)
- `--cleanup-instance` (default on; deletes the instance on exit)
- `--output-dir`, `--tag key=value` (repeatable)

Run a multi-run campaign from a single TOML file (each `[[runs]]` = its own cluster deploy + bench + teardown; `[defaults]` is shared, per-run keys override; `[plot]` triggers a comparison PNG):

```bash
uv run python -m bench.cli campaign bench/configs/llama-family-smoke.toml
```

Plot any results JSON to a PNG (auto-detects benchmark type from `metadata.benchmark`):

```bash
uv run python -m bench.cli plot bench/results/context_scaling/latest.json
uv run python -m bench.cli plot a.json b.json --label-tag operator   # multi-run comparison
```

Adding a new benchmark = (1) write a `bench/lib/<name>.py` exposing a typed `run(session, params, bundle)` callable; (2) add a `bench/cli/<name>.py` with `add_subparser(...)` + `run(args) -> Path`; (3) register the imports in `bench/cli/__main__.py`. To enable plotting for the new benchmark, add a `render_<name>(inputs)` function in `bench/lib/plotting.py` and a dispatch entry in `bench/cli/plot.py::run`.

Results land at `bench/results/<benchmark>/<run_id>.json` (with a `latest.json` symlink alongside) containing metadata (exo SHA, hostname, platform, ISO timestamps, methodology version, user tags), full cluster snapshot, the resolved + derived params, per-step rows, optional cold-control rows, and any derived summaries (e.g. `t_cum_seconds[]` for context-scaling).

## Dashboard UI Testing & Screenshots

### Building and Running the Dashboard
```bash
# Build the dashboard (must be done before running exo)
cd dashboard && npm install && npm run build && cd ..

# Start exo (serves the dashboard at http://localhost:52415)
uv run exo &
sleep 8  # Wait for server to start
```

### Taking Headless Screenshots with Playwright
Use Playwright with headless Chromium for programmatic screenshots — no manual browser interaction needed.

**Setup (one-time):**
```bash
npx --yes playwright install chromium
cd /tmp && npm init -y && npm install playwright
```

**Taking screenshots:**
```javascript
// Run from /tmp where playwright is installed: cd /tmp && node -e "..."
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto('http://localhost:52415', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Inject test data into localStorage if needed (e.g., recent models)
  await page.evaluate(() => {
    localStorage.setItem('exo-recent-models', JSON.stringify([
      { modelId: 'mlx-community/Qwen3-30B-A3B-4bit', launchedAt: Date.now() },
    ]));
  });
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Interact with UI elements
  await page.locator('text=SELECT MODEL').click();
  await page.waitForTimeout(1000);

  // Take screenshot
  await page.screenshot({ path: '/tmp/screenshot.png', fullPage: false });
  await browser.close();
})();
```

### Uploading Images to GitHub PRs
GitHub's API doesn't support direct image upload for PR comments. Workaround:

1. **Commit images to the branch** (temporarily):
   ```bash
   cp /tmp/screenshot.png .
   git add screenshot.png
   git commit -m "temp: add screenshots for PR"
   git push origin <branch>
   COMMIT_SHA=$(git rev-parse HEAD)
   ```

2. **Post PR comment** referencing the raw image URL (uses permanent commit SHA so images survive deletion):
   ```bash
   gh pr comment <PR_NUMBER> --body "![Screenshot](https://raw.githubusercontent.com/exo-explore/exo/${COMMIT_SHA}/screenshot.png)"
   ```

3. **Remove the images** from the branch:
   ```bash
   git rm screenshot.png
   git commit -m "chore: remove temporary screenshot files"
   git push origin <branch>
   ```
   The images still render in the PR comment because they reference the permanent commit SHA.
