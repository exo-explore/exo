# exo Overview

## What exo is

exo is a distributed AI inference system that lets you run frontier language models locally across your own cluster of devices. It automatically discovers devices on your network, places models across them using topology-aware sharding, and exposes compatible APIs (OpenAI, Claude, Ollama) so existing tools just work. Built on MLX for inference and libp2p for low-latency networking—with day-0 RDMA over Thunderbolt 5 support for 99% latency reduction between macOS devices.

## Core concepts

### Event sourcing + message passing

exo uses an [event sourcing architecture](architecture/event-sourcing-message-passing.md) where every state change is captured as an immutable event. Components communicate via typed pub/sub topics: `GLOBAL_EVENTS` (master broadcasts state to workers), `LOCAL_EVENTS` (workers send events to master), `COMMANDS` (API/worker instructions), `ELECTION_MESSAGES`, and `CONNECTION_MESSAGES`. All data transformations are referentially transparent—functions construct new data rather than mutating in place (README.md:20, docs/architecture.md:1-85).

### Five systems

exo's architecture consists of five coordinating systems (docs/architecture.md:9-29):

- **Master**: Executes placement decisions and orders events through a single writer
- **Worker**: Schedules inference jobs, gathers system info, manages runner processes
- **Runner**: Executes inference in isolated processes for fault tolerance
- **API**: FastAPI webserver exposing OpenAI/Claude/Ollama-compatible endpoints
- **Election**: Distributed consensus for master election in unstable networks

## Stack

- **Python 3.13+** primary language (174 files in `src/exo/`), strict static typing with Pydantic v2 and basedpyright (pyproject.toml:6)
- **Rust** for networking and PyO3 bindings: 3 crates in `rust/exo_pyo3_bindings/`, `rust/networking/`, `rust/util/` (README.md:20, CLAUDE.md:100-103)
- **MLX** inference backend on macOS/Linux (with CPU fallback on Linux) and MLX distributed for sharding (pyproject.toml:20-22, README.md:28)
- **SvelteKit 5** dashboard (TypeScript, served at `http://localhost:52415`) built and embedded in the API (CLAUDE.md:106)
- **Nix flakes** reproducible dev environment with `nix run .#exo` entry point (README.md:80)
- **FastAPI** for REST API with WebSocket streaming (pyproject.toml:12)

## Entry points

### CLI

Run exo with `uv run exo` (defined in `pyproject.toml:36` as `exo = "exo.main:main"`). This starts a `Node` (src/exo/main.py) with Router, Worker, Master, Election, and API components all running in one process. Nodes automatically discover each other via libp2p mDNS.

### Dashboard

Built-in web UI at `http://localhost:52415` for cluster monitoring, model selection, and chat (README.md:34, AGENTS.md:106).

### API endpoints

- **OpenAI Chat Completions**: `POST /v1/chat/completions`
- **Claude Messages**: `POST /v1/messages`
- **OpenAI Responses**: `POST /v1/responses`
- **Ollama API**: `GET /ollama/api/tags`, `POST /ollama/api/chat`
- **Instance management**: `POST /instance/previews`, `POST /instance`, `DELETE /instance/{id}`

All endpoints are compatible with existing tools and clients (README.md:29, docs/architecture.md:31-48, CONTRIBUTING.md:100-103).

## Where to go next

- [**Architecture**](architecture/event-sourcing-message-passing.md): Event sourcing model, topic structure, and the five systems in detail
- [**Components**](components/) *(planned)*: Deep dives on Worker, Master, Runner, API, Election
- [**Workflows**](workflows/) *(planned)*: Model loading, inference scheduling, placement algorithm, distributed tracing
- [**Contributing**](../../CONTRIBUTING.md): Code style, testing, adding API adapters, model cards

---

Sources: README.md:1-85, docs/architecture.md:1-85, pyproject.toml:1-40, CLAUDE.md:68-106, CONTRIBUTING.md:1-100, AGENTS.md:5-77
Last indexed: 2026-04-21
