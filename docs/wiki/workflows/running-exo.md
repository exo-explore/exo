# Running exo

A workflow-oriented walk-through: clone, install, launch one node, join a second
node, load a model, and hit the API. For what the components do under the
covers, see [../architecture/data-flow.md](../architecture/data-flow.md) and
[../components/dashboard.md](../components/dashboard.md). For benchmarking
placements, see [./testing-and-bench.md](./testing-and-bench.md).

---

## Prerequisites

exo's tier-1 target is Apple Silicon macOS; Linux runs on CPU only
(`PLATFORMS.md:L1-L21`).

- **Hardware**: Mac Studio M3 Ultra, Mac Mini M4 Pro, or MacBook Pro M5 / M4 Max
  are the explicitly tested configurations (`PLATFORMS.md:L4-L8`). Linux CUDA,
  Vulkan, and Windows are on the roadmap, not tier-1 yet (`PLATFORMS.md:L16-L37`).
- **Python**: `>=3.13` — the project declares this in `pyproject.toml:L6` and
  `basedpyright` is pinned to 3.13 (`pyproject.toml:L92`).
- **Toolchain** (`README.md:L90-L109`): Xcode (Metal ToolChain for MLX),
  Homebrew, `uv` (Python deps), `macmon` (Apple Silicon telemetry, macOS-only
  per `README.md:L167`), `node` (dashboard build, `README.md:L99`), `rustup`
  + `nightly` for the `exo_pyo3_bindings` workspace crate
  (`README.md:L104-L109`, `pyproject.toml:L60`).
- **Optional Nix**: `flake.nix:L131-L165` defines a devShell pinning
  `python313`, `uv`, `ruff`, `basedpyright`, Rust, and (Darwin only)
  `macmon`. `nix run .#exo` builds and launches exo without any manual tool
  installs (`README.md:L77-L88`).
- **Optional Thunderbolt 5 / RDMA**: drops inter-node latency and powers the
  fastest tensor-parallel configs (`README.md:L20-L28`, `README.md:L251-L280`).
  Supported Macs: M4 Pro Mac Mini, M4 Max Mac Studio, M4 Max MacBook Pro, M3
  Ultra Mac Studio (`README.md:L253`). Skip for first run — regular networking
  still works.

Nix users wanting the Cachix binary cache (to skip rebuilding Metal ToolChain)
must add themselves to `trusted-users` and enable flakes in `/etc/nix/nix.conf`
(`README.md:L83-L88`).

---

## Install

The canonical flow is: clone, build the dashboard, let `uv` materialize the
Python env on first `uv run` (`README.md:L113-L122`).

```bash
# Clone
git clone https://github.com/exo-explore/exo
cd exo

# Build the Svelte dashboard (served by the API at /, required for the UI)
cd dashboard && npm install && npm run build && cd ..
```

The dashboard build output is written to `dashboard/build/` and is served by
the API process (`CLAUDE.md:L105-L106`).

If you skip the dashboard build, `uv run exo` still starts and the API still
answers — you just won't have a UI.

**Nix alternative.** `nix run .#exo` (`README.md:L79-L81`) builds the Darwin
`exo` package at `flake.nix:L115-L129` and launches it — no `uv`, `brew`, or
`rustup` needed.

**Linux.** Same steps minus `macmon`; use `apt` or Homebrew-on-Linux for `uv`
and `node`; inference runs on CPU (`README.md:L130-L184`). XDG paths are
honoured for models, logs, config (`README.md:L196-L204`).

---

## First run

Launch a single node to confirm your install works before bringing up a second
device.

```bash
uv run exo
```

The `exo` entrypoint is declared in `pyproject.toml:L35-L36` as
`exo.main:main`, which parses CLI args, constructs a `Node` with router,
worker, master, election, and API, and runs it under `anyio`
(`src/exo/main.py:L30-L115`, `src/exo/main.py:L300-L310`).

What a healthy first run looks like:

- Logs print `Starting node <node_id>` (`src/exo/main.py:L69`).
- The API binds `http://localhost:52415/` — the default port is hard-coded in
  the parser at `src/exo/main.py:L357-L362` and confirmed at
  `README.md:L124`.
- The dashboard is served from the same origin when you visit the root URL
  (`README.md:L71`, `CLAUDE.md:L105-L106`).

Quick sanity check from another terminal:

```bash
curl http://localhost:52415/models | head
```

`/models` lists every model card exo knows about (`README.md:L485`). If the
call returns JSON, the API is healthy.

**Useful first-run flags** (argparse block at `src/exo/main.py:L329-L410`):

- `-v`, `-vv` — verbosity (`CLAUDE.md:L19`).
- `--no-worker` — coordinator-only node (`README.md:L188-L192`,
  `src/exo/main.py:L377-L380`).
- `--no-downloads` — skip the download coordinator if models are pre-staged
  (`src/exo/main.py:L381-L385`).
- `--offline` — same as `EXO_OFFLINE=true`; uses only local models
  (`src/exo/main.py:L386-L391`, `README.md:L293`).
- `--api-port <port>` / `--libp2p-port <port>` — override defaults
  (`src/exo/main.py:L357-L369`).

---

## Joining a cluster

exo's discovery is automatic — `README.md:L24` and `README.md:L71`: "Devices
running exo automatically discover each other, without needing any manual
configuration." Each device gets its own `Node` with the same pub/sub topics
(`src/exo/main.py:L56-L61`), and the master is elected via the bully algorithm
(`CLAUDE.md:L75`).

**Minimum steps to bring up two devices:**

1. Make sure both machines are on the same L2 network. If you plan to use
   Thunderbolt 5 RDMA, connect every RDMA-participating device to every other
   one (`README.md:L274`) with TB5-rated cables (`README.md:L275`), and
   remember that on a Mac Studio the TB5 port next to Ethernet is unusable for
   RDMA (`README.md:L276`).
2. On **device A**, launch exo normally:
   ```bash
   uv run exo
   ```
3. On **device B**, launch exo the same way:
   ```bash
   uv run exo
   ```
4. Open `http://localhost:52415/` on either device. The dashboard shows the
   cluster view with both nodes (`README.md:L32-L39`).

**Cluster isolation.** If you have more than one exo cluster on the same
network (dev vs prod, two people on the same Wi-Fi), set
`EXO_LIBP2P_NAMESPACE` to a shared string on all devices that should form
one cluster (`README.md:L218-L229`, `README.md:L310-L311`). Devices with
different namespaces will not discover each other.

**Explicit bootstrap peers.** If multicast discovery is blocked, pass libp2p
multiaddrs via `--bootstrap-peers` or `EXO_BOOTSTRAP_PEERS`
(`src/exo/main.py:L370-L376`).

**RDMA enablement** (one-time, per machine, macOS 26.2+,
`README.md:L259-L280`):

1. Shut down, hold power 10s, enter Recovery, open Terminal.
2. Run `rdma_ctl enable`.
3. Reboot. exo picks it up automatically.
4. If running from source, apply the network config via
   `tmp/set_rdma_network_config.sh` — or `docs/thunderbolt-bridge-ops.md` for
   fixed-address bridge setups.

OS versions must match exactly (betas included, `README.md:L278`) or RDMA
ports won't discover each other.

---

## Loading a model

Instances (running model placements) are created through the master's API. The
flow has three steps — the README walks it explicitly at `README.md:L327-L412`.

**Step 1 — preview valid placements.** Given a `model_id`, ask the master what
ways it can shard the model across the cluster:

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b"
```

Each preview includes the sharding kind (`Pipeline` or tensor), the instance
metadata (`MlxRing`, `JACCL`, etc.), and the per-node memory delta
(`README.md:L336-L353`). exo chooses from these based on topology, device
resources, and link latency/bandwidth (`README.md:L26`): this is what
"Topology-Aware Auto Parallel" means.

Grab the first valid one with `jq`:

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b" \
  | jq -c '.previews[] | select(.error == null) | .instance' | head -n1
```

**Step 2 — create the instance.** POST the chosen placement back to
`/instance`:

```bash
curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d '{"instance": { /* paste preview.instance here */ }}'
```

The master persists the placement, workers download shards, and the instance
moves toward ready (`README.md:L364-L384`). You can poll `/state` to watch the
deployment progress (`README.md:L488`).

**Step 3 — list and delete instances.**

```bash
curl http://localhost:52415/state
curl -X DELETE http://localhost:52415/instance/<INSTANCE_ID>
```

(`README.md:L404-L412`.)

### HuggingFace custom models

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/my-custom-model"}'
```

(`README.md:L467-L477`.) Custom cards land in `~/.exo/custom_model_cards/`
(macOS) or `~/.local/share/exo/custom_model_cards/` (Linux,
`CONTRIBUTING.md:L44-L47`, `README.md:L196-L204`). Models needing
`trust_remote_code` must be opted in explicitly (default `false`,
`README.md:L479-L481`, `CONTRIBUTING.md:L82-L96`).

### Pre-staging + sharding

`EXO_MODELS_PATH` is a colon-separated search path for pre-downloaded models
(`README.md:L288-L302`); `EXO_MODELS_DIR` controls where downloads land
(`README.md:L291`). Combined with `--offline` you get an air-gapped node.

Sharding kinds: `Pipeline` runs layers as a pipeline; tensor parallel splits
each layer across devices (the 1.8x / 3.2x numbers at `README.md:L27-L28`).
Tensor parallel requires `supports_tensor = true` in the model card
(`CONTRIBUTING.md:L73`).

---

## Running inference via API

exo speaks four API dialects from the same port
(`README.md:L29`, `README.md:L320-L323`, `CONTRIBUTING.md:L112-L116`).
All examples assume you've created an instance of
`mlx-community/Llama-3.2-1B-Instruct-4bit`.

**OpenAI Chat Completions** — `/v1/chat/completions`
(`README.md:L388-L402`):

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is Llama 3.2 1B?"}],
    "stream": true
  }'
```

**Claude Messages** — `/v1/messages` (`README.md:L414-L429`):

```bash
curl -N -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
    "stream": true
  }'
```

**OpenAI Responses** — `/v1/responses` (`README.md:L431-L445`):

```bash
curl -N -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

**Ollama** — `/ollama/api/chat` and `/ollama/api/tags`
(`README.md:L447-L465`):

```bash
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

curl http://localhost:52415/ollama/api/tags
```

The adapter pattern lives at `src/exo/master/adapters/` — one file per dialect
(`CONTRIBUTING.md:L104-L117`). Nothing API-specific leaks into worker or
runner code.

---

## Using the dashboard

Point a browser at `http://localhost:52415/` on any node
(`README.md:L71`, `README.md:L124`). The dashboard is a Svelte 5 + TypeScript
app served by the API — its build output lives at `dashboard/build/`
(`CLAUDE.md:L105-L106`).

From the dashboard you can:

- See the cluster view — every discovered node with live `macmon` telemetry
  on Apple Silicon (`README.md:L32-L39`, `README.md:L98`).
- Browse downloaded and HuggingFace-searchable models (`README.md:L483-L488`).
- Create instances interactively (same API as `/instance/previews` + `/instance`).
- Chat with any loaded model.

See [../components/dashboard.md](../components/dashboard.md) for a deeper
walk-through of the views and their data sources.

---

## Troubleshooting

**`uv run exo` fails importing MLX on macOS.** MLX needs the Metal ToolChain
from Xcode (`README.md:L91`) — install Xcode proper, not just CLT. Under Nix
it comes from the unfree `metal-toolchain` derivation
(`flake.nix:L82-L88`, `flake.nix:L121-L125`).

**Rust bindings won't build.** Install the nightly toolchain
(`README.md:L104-L109`). The `exo_pyo3_bindings` crate is a uv workspace
member (`pyproject.toml:L59-L63`); nuking stale `target/` and `.venv` often
fixes it.

**Dashboard 404 at `/`.** You skipped `npm run build` in `dashboard/`. Run
the build (`README.md:L117-L118`) — no exo restart needed.

**Nodes don't see each other.** Check `EXO_LIBP2P_NAMESPACE` matches (or is
unset) on both (`README.md:L218-L229`); confirm same L2 network, else pass
`--bootstrap-peers` (`src/exo/main.py:L370-L376`); for RDMA, verify macOS
versions match (`README.md:L278`) and every device is connected to every
other (`README.md:L274`).

**Port 52415 already in use.** Re-launch with `--api-port <free-port>`
(`src/exo/main.py:L357-L362`).

**Tensor-parallel placement doesn't appear.** The model card may lack
`supports_tensor = true` (`CONTRIBUTING.md:L73`), or the cluster is too
memory-tight — `/instance/previews` will only return `Pipeline` placements
then.

**Offline box can't find a model.** Stage under `EXO_MODELS_PATH`
(`README.md:L290`) and launch with `--offline`
(`src/exo/main.py:L386-L391`, `README.md:L293-L305`).

**macOS app vs source install conflict.** The
[EXO macOS app](https://assets.exolabs.net/EXO-latest.dmg)
(`README.md:L206-L216`) and a source install fight over network config.
Uninstall the app via menu bar → Advanced → Uninstall (`README.md:L233`) or
`sudo ./app/EXO/uninstall-exo.sh` (`README.md:L237-L239`).

**Pre-commit checks.** CI runs basedpyright, ruff, `nix fmt`, pytest
(`CLAUDE.md:L43-L64`). Local equivalent:

```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

---

## Related

- [./testing-and-bench.md](./testing-and-bench.md) — running tests, measuring
  prefill and generation throughput via `bench/exo_bench.py`.
- [../components/dashboard.md](../components/dashboard.md) — dashboard
  internals and views.
- [../architecture/data-flow.md](../architecture/data-flow.md) — how
  `COMMANDS`, `LOCAL_EVENTS`, and `GLOBAL_EVENTS` move between API, master,
  and workers.

---

**Sources**

- `/Users/leozealous/exo/README.md` (full read)
- `/Users/leozealous/exo/CLAUDE.md`
- `/Users/leozealous/exo/CONTRIBUTING.md`
- `/Users/leozealous/exo/PLATFORMS.md`
- `/Users/leozealous/exo/RULES.md`
- `/Users/leozealous/exo/pyproject.toml`
- `/Users/leozealous/exo/flake.nix`
- `/Users/leozealous/exo/src/exo/main.py`
- `/Users/leozealous/exo/packaging/` (dmg, pyinstaller)
- `/Users/leozealous/exo/scripts/` (Thunderbolt watchdog + install-launchagent)

Last indexed: `c0d5bf92` — 2026-04-21
