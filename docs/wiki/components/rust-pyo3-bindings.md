# rust/exo_pyo3_bindings — FFI Boundary Crate

## Overview

`rust/exo_pyo3_bindings` is the single FFI boundary between exo's Python application layer and its Rust networking/utility crates. It is a `cdylib + rlib` crate (`rust/exo_pyo3_bindings/Cargo.toml:14`) that wraps types from the `networking` workspace crate (`rust/exo_pyo3_bindings/Cargo.toml:25`) and re-exports them as pyo3 classes under the Python module name `exo_pyo3_bindings` (`rust/exo_pyo3_bindings/src/lib.rs:154`).

The crate owns three concerns:
1. Wrapping `libp2p::identity::Keypair` as a Python-usable identity type (`rust/exo_pyo3_bindings/src/ident.rs:7-11`).
2. Wrapping the `networking::swarm` swarm handle — `ToSwarm` / `FromSwarm` channels produced by `create_swarm(...)` — as a Python-usable `NetworkingHandle` (`rust/exo_pyo3_bindings/src/networking.rs:132-138`, `rust/exo_pyo3_bindings/src/networking.rs:182-205`).
3. Translating libp2p's typed errors into Python-native exception types (`rust/exo_pyo3_bindings/src/networking.rs:24-130`).

This is the only Rust→Python surface in the repo; every Python consumer that needs libp2p goes through here. The primary consumer is `src/exo/routing/router.py` (see below).

## Exposed classes/functions

The Rust → Python name mapping is driven by `#[pyclass(name = "...")]` attributes and registered via `networking_submodule` (`rust/exo_pyo3_bindings/src/networking.rs:310-319`) and the top-level `main_module` (`rust/exo_pyo3_bindings/src/lib.rs:154-172`). Stub signatures mirror these in `rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi`.

| Rust type | Python name | Kind | Source |
|---|---|---|---|
| `PyKeypair(libp2p::identity::Keypair)` | `Keypair` | class (frozen) | `src/ident.rs:7-11` |
| `PyKeypair::generate` | `Keypair.generate()` | staticmethod | `src/ident.rs:18-21` |
| `PyKeypair::from_bytes` | `Keypair.from_bytes(bytes)` | staticmethod | `src/ident.rs:23-28` |
| `PyKeypair::to_bytes` | `Keypair.to_bytes()` | method | `src/ident.rs:30-41` |
| `PyKeypair::to_node_id` | `Keypair.to_node_id()` | method | `src/ident.rs:43-46` |
| `PyNetworkingHandle` | `NetworkingHandle` | class | `src/networking.rs:132-138` |
| `PyNetworkingHandle::py_new` | `NetworkingHandle(identity, bootstrap_peers, listen_port)` | constructor | `src/networking.rs:182-205` |
| `PyNetworkingHandle::recv` | `NetworkingHandle.recv()` | async method | `src/networking.rs:207-219` |
| `PyNetworkingHandle::gossipsub_subscribe` | `NetworkingHandle.gossipsub_subscribe(topic)` | async method | `src/networking.rs:226-243` |
| `PyNetworkingHandle::gossipsub_unsubscribe` | `NetworkingHandle.gossipsub_unsubscribe(topic)` | async method | `src/networking.rs:248-264` |
| `PyNetworkingHandle::gossipsub_publish` | `NetworkingHandle.gossipsub_publish(topic, data)` | async method | `src/networking.rs:269-297` |
| `PyFromSwarm::Connection { peer_id, connected }` | `PyFromSwarm.Connection` | complex enum variant | `src/networking.rs:142-146` |
| `PyFromSwarm::Message { origin, topic, data }` | `PyFromSwarm.Message` | complex enum variant | `src/networking.rs:147-151` |
| `PyNoPeersSubscribedToTopicError` | `NoPeersSubscribedToTopicError` | exception | `src/networking.rs:29-63` |
| `PyAllQueuesFullError` | `AllQueuesFullError` | exception | `src/networking.rs:65-98` |
| `PyMessageTooLargeError` | `MessageTooLargeError` | exception | `src/networking.rs:100-129` |

The `FromSwarm` → `PyFromSwarm` conversion at `rust/exo_pyo3_bindings/src/networking.rs:153-171` collapses libp2p's `Discovered`/`Expired` variants into a single `Connection { connected: bool }` variant and base58-encodes `PeerId` into strings at the FFI boundary — Python never sees a raw `PeerId`.

The `recv` method is declared as `#[gen_stub(skip)]` (`rust/exo_pyo3_bindings/src/networking.rs:207`) and its async signature is hand-written via `gen_methods_from_python!` (`rust/exo_pyo3_bindings/src/networking.rs:301-308`) because it uses `future_into_py` instead of pyo3's `experimental-async` — stub-gen can't introspect it automatically.

## Async bridge

pyo3's `experimental-async` feature (`rust/exo_pyo3_bindings/Cargo.toml:31`) is enabled, plus `pyo3-async-runtimes` 0.27.0 with `tokio-runtime` (`rust/exo_pyo3_bindings/Cargo.toml:42-46`). The tokio runtime is initialized inside `main_module` at module-import time:

```rust
let mut builder = tokio::runtime::Builder::new_multi_thread();
builder.enable_all();
pyo3_async_runtimes::tokio::init(builder);
```
(`rust/exo_pyo3_bindings/src/lib.rs:158-160`)

Every `await` in an `#[pymethods] async fn` is wrapped with `.allow_threads_py()` (e.g. `rust/exo_pyo3_bindings/src/networking.rs:235-236`, `239-240`, `257-258`, `261-262`, `280-281`, `285-286`). This custom adapter lives at `rust/exo_pyo3_bindings/src/allow_threading.rs:15-37` and is installed as a blanket `FutureExt` trait on every `Future` via `rust/exo_pyo3_bindings/src/lib.rs:52-62`.

The adapter is a pin-projected wrapper whose `poll` implementation attaches to the GIL, calls `py.detach(...)` around the inner `poll`, and re-attaches on return (`rust/exo_pyo3_bindings/src/allow_threading.rs:33-36`). This matches the pattern documented at https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await and referenced directly in comments at `rust/exo_pyo3_bindings/src/lib.rs:53` and `rust/exo_pyo3_bindings/src/networking.rs:176-178`. Without it, pyo3 would hold the GIL across every `.await`, stalling all other Python threads.

The `recv` method takes the non-`experimental-async` path: it manually constructs a Python future with `pyo3_async_runtimes::tokio::future_into_py` (`rust/exo_pyo3_bindings/src/networking.rs:210-218`) because `recv` needs to borrow `Arc<Mutex<Stream>>` across the await, which is easier to express outside the `async fn` macro.

`TokioRuntimeExt::spawn_with_scope` (`rust/exo_pyo3_bindings/src/lib.rs:89-99`) uses `pyo3_async_runtimes::tokio::scope` to propagate the current asyncio loop locals into spawned tasks — this is available to the crate but not currently used from the bindings themselves.

## Error handling

Rust errors cross the FFI boundary via three mechanisms.

**Generic Rust errors** use the `ResultExt::pyerr` extension at `rust/exo_pyo3_bindings/src/lib.rs:42-50`: any `Result<T, E>` where `E: ToString` becomes `PyResult<T>` by mapping errors to `PyRuntimeError::new_err(e.to_string())`. Used in `PyKeypair::from_bytes` (`rust/exo_pyo3_bindings/src/ident.rs:26`), `PyKeypair::to_bytes` (`rust/exo_pyo3_bindings/src/ident.rs:36`), the swarm constructor (`rust/exo_pyo3_bindings/src/networking.rs:198`), and the subscribe callback (`rust/exo_pyo3_bindings/src/networking.rs:242`).

**Channel closure** is surfaced via `PyErrExt::receiver_channel_closed` (`rust/exo_pyo3_bindings/src/lib.rs:64-69`), which returns `PyConnectionError::new_err("Receiver channel closed unexpectedly")`. The oneshot-reply channels in every gossipsub RPC raise this on sender drop (`rust/exo_pyo3_bindings/src/networking.rs:241`, `263`, `287`). `recv()` raises it when the swarm stream ends (`rust/exo_pyo3_bindings/src/networking.rs:216`).

**Gossipsub-specific errors** from `libp2p::gossipsub::PublishError` are pattern-matched into custom Python exception types at `rust/exo_pyo3_bindings/src/networking.rs:288-295`:

| Rust variant | Python exception |
|---|---|
| `PublishError::NoPeersSubscribedToTopic` | `NoPeersSubscribedToTopicError` |
| `PublishError::AllQueuesFull(_)` | `AllQueuesFullError` |
| `PublishError::MessageTooLarge` | `MessageTooLargeError` |
| any other variant | `PyRuntimeError` with the Display string |

Each custom exception is a `#[pyclass(frozen, extends=PyException, name="...")]` with a fixed `__str__` message (`rust/exo_pyo3_bindings/src/networking.rs:34-36`, `70-71`, `104-105`). `Router._networking_publish` at `src/exo/routing/router.py:240-257` catches all three by name and translates them into log-and-drop behavior.

## Build integration

The crate has **two** build-system files because it is both a maturin-buildable Python wheel and a member of exo's Rust workspace:

1. `rust/exo_pyo3_bindings/pyproject.toml` — maturin project definition. `build-backend = "maturin"` (`rust/exo_pyo3_bindings/pyproject.toml:1-3`), `module-name = "exo_pyo3_bindings"`, `features = ["pyo3/extension-module", "pyo3/experimental-async"]` (`rust/exo_pyo3_bindings/pyproject.toml:20-24`), `requires-python = ">=3.13"` (`rust/exo_pyo3_bindings/pyproject.toml:14`). Maturin compiles the `cdylib` crate-type output into a Python extension module.
2. `rust/exo_pyo3_bindings/Cargo.toml` — Rust workspace member; `crate-type = ["cdylib", "rlib"]` (`Cargo.toml:14`). The `rlib` half exists so `src/bin/stub_gen.rs` can link against `exo_pyo3_bindings::stub_info()` (`rust/exo_pyo3_bindings/src/bin/stub_gen.rs:5`).

The top-level exo `pyproject.toml` wires the bindings into the Python package via uv workspaces:

- Workspace members include `rust/exo_pyo3_bindings` (`pyproject.toml:59-60`).
- `exo_pyo3_bindings` is declared as a workspace source (`pyproject.toml:62-63`) and listed as a dependency of `exo` itself (`pyproject.toml:18`).
- Ruff is configured to skip the Rust crate directory (`pyproject.toml:124-129`).

Running `uv sync` or `uv run exo` triggers maturin to build the wheel and install it alongside the Python package.

Stub generation: the `stub_gen` binary (`rust/exo_pyo3_bindings/Cargo.toml:16-19`, `rust/exo_pyo3_bindings/src/bin/stub_gen.rs`) walks the `define_stub_info_gatherer!` inventory registered at `rust/exo_pyo3_bindings/src/lib.rs:174` and emits the `exo_pyo3_bindings.pyi` file checked into the repo. Each `#[gen_stub_pyclass]` and `#[gen_stub_pymethods]` macro (e.g. `rust/exo_pyo3_bindings/src/ident.rs:8,13`, `rust/exo_pyo3_bindings/src/networking.rs:132,173`) contributes to that inventory. The `recv` override uses `gen_methods_from_python!` at `rust/exo_pyo3_bindings/src/networking.rs:301-308` to paste a hand-written async signature into the generated stubs.

## Testing

Two test surfaces exist:

- **Rust tests** at `rust/exo_pyo3_bindings/tests/dummy.rs:1-54`: currently only a `test_drop_channel` integration test verifying tokio mpsc semantics — no FFI assertions.
- **Python tests** at `rust/exo_pyo3_bindings/tests/test_python.py:1-37`: imports the compiled module, constructs `NetworkingHandle(Keypair.generate())`, runs a publish-recv loop, and catches `NoPeersSubscribedToTopicError`. `pyproject.toml:26-29` sets `asyncio_mode = "auto"` and `log_cli = true`.

Note: `test_python.py:15` calls `NetworkingHandle(Keypair.generate())` with a single argument, but the live Rust constructor now requires three (`rust/exo_pyo3_bindings/src/networking.rs:183-188`). The test predates the `bootstrap_peers` / `listen_port` additions.

pytest is a dev-dependency of the bindings crate itself (`rust/exo_pyo3_bindings/pyproject.toml:17-18`), alongside a circular `exo_pyo3_bindings` dev-dep so tests can resolve the installed wheel.

## Gotchas

1. **GIL across await is the default, and wrong.** Every `.await` inside `#[pymethods] async fn` must be prefixed with `.allow_threads_py()` or the interpreter will stall. This is an easy footgun — see the explicit reminder comment at `rust/exo_pyo3_bindings/src/networking.rs:176-178`. Missing the wrap compiles and runs but causes silent contention.

2. **The swarm must be constructed inside a tokio context.** `PyNetworkingHandle::py_new` calls `pyo3_async_runtimes::tokio::get_runtime().enter()` and holds a `_guard` across `create_swarm(...)` (`rust/exo_pyo3_bindings/src/networking.rs:196-199`) — without this guard `libp2p` panics at spawn time. The comment "within tokio context!! or it crashes" flags it.

3. **Submodules don't work cleanly with maturin.** Everything is flat under `exo_pyo3_bindings` despite the TODO at `rust/exo_pyo3_bindings/src/lib.rs:162-164` wishing for a `exo_pyo3_bindings.networking` submodule. Consumers import `from exo_pyo3_bindings import NetworkingHandle, Keypair, PyFromSwarm, ...` (see `src/exo/routing/router.py:17-24`).

4. **`recv()` is not re-entrant.** The swarm stream is wrapped in `Arc<Mutex<...>>` (`rust/exo_pyo3_bindings/src/networking.rs:137`) and `recv` uses `try_lock` — a second concurrent `recv` returns `PyRuntimeError("called recv twice concurrently")` (`rust/exo_pyo3_bindings/src/networking.rs:212-213`). Callers must serialize reads; `Router._networking_recv` (`src/exo/routing/router.py:203-238`) relies on this by running a single receive task.

5. **`PyFromSwarm::Connection` flattens `Discovered` and `Expired`.** Python sees `connected: bool` but cannot distinguish "newly discovered" from "re-added after expiry" — the Rust side collapses them at `rust/exo_pyo3_bindings/src/networking.rs:156-163`.

6. **`PyKeypair::to_bytes` only returns bytes for Ed25519.** The implementation calls `try_into_ed25519()` and propagates any failure via `.pyerr()` (`rust/exo_pyo3_bindings/src/ident.rs:35-36`). If a non-Ed25519 keypair is ever introduced, roundtripping breaks. `get_node_id_keypair` in `src/exo/routing/router.py:273-305` catches `RuntimeError` and `ValueError` and regenerates on failure.

7. **Custom exceptions don't preserve Rust error context.** `NoPeersSubscribedToTopicError`, `AllQueuesFullError`, and `MessageTooLargeError` each carry a fixed static `MSG` string and ignore constructor args (`rust/exo_pyo3_bindings/src/networking.rs:50-54`, `86-90`, `116-120`). You can't attach topic name or message size to the exception — those have to be logged separately, as `Router._networking_publish` does at `src/exo/routing/router.py:252-257`.

8. **`py-clone` feature is intentionally disabled.** See the comment at `rust/exo_pyo3_bindings/Cargo.toml:33`: "may cause panics — remove if panics happen". Do not enable it to work around `Py<T>` move issues; use explicit `Py::clone_ref(py)` instead.

9. **Stub sync is a build-time concern.** `exo_pyo3_bindings.pyi` is generated by the `stub_gen` binary but checked into the repo. Drift between the `.pyi` and the Rust source is possible if `stub_gen` isn't re-run after changing `#[pymethods]`. The existing `.pyi` at `rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi:45` shows `NetworkingHandle.__new__(cls, identity: Keypair)` with a single arg — stale vs. `rust/exo_pyo3_bindings/src/networking.rs:183-188` which takes three.

10. **Logging initialization happens exactly once at module import.** `pyo3_log::init()` at `rust/exo_pyo3_bindings/src/lib.rs:157` bridges Rust `log::` calls into Python's `logging` module. If Python reconfigures logging later, Rust log events may be lost — this is a known pyo3-log behavior, not a bug in this crate.

## Cross-references

- [rust-networking.md](rust-networking.md) — the `networking` crate whose `create_swarm`, `ToSwarm`, and `FromSwarm` types this crate wraps.
- [routing.md](routing.md) — the Python `Router` consumer that owns a `NetworkingHandle` and dispatches messages to typed topic routers.
- [../architecture/module-boundaries.md](../architecture/module-boundaries.md) — how the FFI boundary fits into the overall Rust/Python split in exo.

## Sources

- `/Users/leozealous/exo/rust/exo_pyo3_bindings/Cargo.toml:1-67`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/pyproject.toml:1-29`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/README.md:1`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi:1-94`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/lib.rs:1-175`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/ident.rs:1-47`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/networking.rs:1-319`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/allow_threading.rs:1-37`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/src/bin/stub_gen.rs:1-8`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/tests/dummy.rs:1-54`
- `/Users/leozealous/exo/rust/exo_pyo3_bindings/tests/test_python.py:1-37`
- `/Users/leozealous/exo/src/exo/routing/router.py:1-305`
- `/Users/leozealous/exo/pyproject.toml:1-141`

---

Last indexed: 2026-04-21
