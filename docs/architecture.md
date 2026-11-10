# Architecture

This document gives a high-level overview of how exo is structured: the main
components that make up a cluster, and the event-sourcing model they share. It is
meant as an entry point for new contributors — see [`docs/api.md`](api.md) for the
REST API reference.

> The conceptual model below follows how the maintainers describe exo's design
> (see issue #930). File paths point at the current `src/exo/` layout.

## Components

A running exo cluster is made up of a few cooperating components. Each node runs the
same binary; the roles below describe responsibilities rather than separate programs.

| Component | Lives in | Responsibility |
|-----------|----------|----------------|
| **master** | `src/exo/master/` | Orders events to give the cluster *strong eventual consistency*, and decides model placement across devices (`placement.py`). |
| **api** | `src/exo/api/` | Prepares cluster state for client applications and exposes the REST surface (OpenAI Chat Completions, Claude Messages, OpenAI Responses, and Ollama-compatible). |
| **worker** | `src/exo/worker/` | Schedules distributed work and plans how a model is run across the devices it owns (`plan.py`). |
| **runner** | `src/exo/worker/runner/` | The ML-specific code that actually loads a shard and runs inference (e.g. via the MLX engine). |
| **election** | `src/exo/shared/election.py` | The entry point into clustering: nodes discover one another and elect a single, unique master. |
| **routing** | `src/exo/routing/` | Moves messages between nodes so events and commands reach the right component. |

A typical flow: a node comes up and joins the cluster through **election**, which
settles on one **master**. Clients talk to the **api**, which reflects cluster
**state**. The **master** orders the resulting events and the **worker** on each node
schedules the actual inference work, handing ML tasks to its **runner**.

## The event-sourcing model

exo's networking is built around event sourcing. Three core ideas, with their types in
[`src/exo/shared/types/`](../src/exo/shared/types/):

- **Events** (`events.py`) record that a side effect *happened* — for example, a chunk
  of output was generated, or a runner changed status.
- **State** (`state.py`) is the **summation of all events**. There is no separate
  source of truth; replaying the events reconstructs the current state.
- **Commands** (`commands.py`) combine with the current state to trigger the creation
  of *new* side effects (and therefore new events).

So the loop is: `command + state → side effect → event → new state`.

### Tasks

**Tasks** (`tasks.py`) are a special kind of command. They flow **worker → runner**
only, so — unlike other commands — they are **not networked** across the cluster. This
keeps the hot path of ML execution local to the node that owns the work.

## Where to look next

- **Add a model**: model cards live in `resources/inference_model_cards/` — see
  [`CONTRIBUTING.md`](../CONTRIBUTING.md#model-cards).
- **The REST API**: [`docs/api.md`](api.md).
- **Build & run from source**: [`CONTRIBUTING.md`](../CONTRIBUTING.md).

> This is an initial overview and intentionally high-level. Corrections and additions
> from maintainers are very welcome — particularly around the routing/RPC layer, which
> is still evolving.
