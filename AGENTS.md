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
