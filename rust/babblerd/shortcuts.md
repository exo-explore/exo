# `babblerd` Shortcuts

This file tracks architectural and implementation shortcuts that were taken
deliberately during the refactors. They are acceptable for now, but they are
not meant to be the final design.

This is not a dump of every `TODO` comment in the crate. It is the curated list
of shortcuts that should be revisited later.

## Architecture / IPC

- The public control socket still uses an ad-hoc line protocol instead of the
  intended `zbus`/D-Bus-style IPC surface.
  Files:
  - `src/daemon.rs`
  - `src/main.rs`
  Follow-up:
  - Replace `keepalive <ttl_ms>` / `get-state` string commands with a typed IPC
    API.

- The daemon core currently tracks a single global keepalive deadline, not
  per-client leases.
  Files:
  - `src/daemon.rs`
  - `src/main.rs`
  Why this is a shortcut:
  - It does not model multiple clients independently.
  - It cannot distinguish which client is keeping the service alive.
  - The current tree also includes a temporary internal self-client in
    `main.rs` that periodically issues keepalive commands just to keep the
    daemon/routing stack alive during bring-up.
  Follow-up:
  - Introduce real lease ownership/tracking in the daemon core.
  - Remove the temporary internal keepalive client once a real frontend or test
    harness is driving the daemon.

- Raw Babel debug output currently only goes to tracing logs.
  Files:
  - `src/babel/runtime.rs`
  - `src/daemon.rs`
  Why this is a shortcut:
  - There is no configurable or structured diagnostics stream anymore.
  - That is fine for now, but eventually debugging should not require tailing
    daemon logs.
  Follow-up:
  - Add configurable debug output or a separate structured diagnostics stream
    once the real IPC surface exists.

- The daemon core exposes state only through `get-state` polling and inline
  command responses.
  Files:
  - `src/daemon.rs`
  Follow-up:
  - Add real state publication/signals once the IPC surface is upgraded.

## Service Lifecycle

- The daemon now has explicit `Off/Starting/On/Stopping`, but the control model
  is still minimal.
  Files:
  - `src/daemon.rs`
  Why this is a shortcut:
  - There is no richer lifecycle API yet.
  - There is no explicit enable/disable policy beyond keepalive-driven on/off.
  Follow-up:
  - Revisit the final lifecycle API once IPC is made real.

- `ServiceState::On` currently means “the routing tasks were started”, not a
  stronger readiness guarantee such as “babeld is healthy, has admitted
  interfaces, and is actually usable for mesh forwarding”.
  Files:
  - `src/daemon.rs`
  - `src/routing_stack.rs`
  - `src/babel/runtime.rs`
  Why this is a shortcut:
  - The frontend may eventually want to distinguish process/task liveness from
    actual routing readiness.
  Follow-up:
  - Add a separate readiness field or richer public state model instead of
    overloading `ServiceState::On`.

- The resident `utun` vs heavy routing-stack split is now in place, but the
  naming and abstractions are still transitional.
  Files:
  - `src/daemon.rs`
  - `src/routing_stack.rs`
  - `src/tun.rs`
  Follow-up:
  - Revisit names and boundaries after the daemon core / IPC architecture settles.

- `RoutingStack::stop` still uses abort-driven shutdown for the interface
  watcher and logger task.
  Files:
  - `src/routing_stack.rs`
  Why this is a shortcut:
  - It is pragmatic, but not a carefully coordinated shutdown protocol.
  Follow-up:
  - Replace task abortion with explicit shutdown signaling where it matters.

## Babel Integration

- `babeld` runtime startup config is still assembled partly as raw strings.
  Files:
  - `src/babel/runtime.rs`
  - `src/babel/command.rs`
  Why this is a shortcut:
  - The local-socket command side is typed, but spawn-time `-C` config is not.
  Follow-up:
  - Add a typed Babel config/config-statement layer.

- The runtime still depends on fork-specific `babeld` behavior
  (`kernel-install false`) while spawning `"babeld"` from `PATH`.
  Files:
  - `src/babel/runtime.rs`
  - `../nix/babeld.nix`
  Why this is a shortcut:
  - It assumes the right binary is on `PATH`.
  - The Nix packaging is still not pinned to a specific revision.
  Follow-up:
  - Pin the fork revision and make the runtime use that exact binary.

## Networking / Interface Admission

- Interface admission is still heuristic and too broad on macOS.
  Files:
  - `src/lib.rs` (`if_watcher`)
  Why this is a shortcut:
  - Any `en*` interface with link-local IPv6 and `is_up()` can still get pulled
    into Babel.
  - This can include unrelated Wi‑Fi, built-in Ethernet, USB Ethernet, etc.
  Follow-up:
  - Replace the current heuristic with a stronger admission policy
    (neighbor proof, richer metadata, or both).

- `router_udp_port` exists in config, but there is still no UDP dataplane using
  it.
  Files:
  - `src/config.rs`
  Follow-up:
  - Either wire it into the real dataplane soon, or remove it until the
    dataplane exists.

- `UtunDevice` is still a thin platform-specific wrapper with some rough edges.
  Files:
  - `src/tun.rs`
  Why this is a shortcut:
  - It still stores the address as `Ipv6Net` even though usage is `/128`-only.
  - It still has hard-coded MTU and other tun-rs builder assumptions.
  Follow-up:
  - Tighten the type and revisit the platform-specific tuning once the dataplane
    is implemented.

## Identity / Security / Filesystem

- The node-id file is created with `0600`, but existing files are only
  owner-checked, not mode-checked.
  Files:
  - `src/identity.rs`
  Why this is a shortcut:
  - A root-owned but group/world-writable file would still be accepted.
  Follow-up:
  - Enforce safe permissions on reload, not just on initial creation.

- The public IPC socket is intentionally world-accessible for now.
  Files:
  - `src/main.rs`
  Why this is a shortcut:
  - Any local user can connect, issue keepalives, and drive the daemon's public
    control surface.
  Follow-up:
  - Revisit permissions/authz once the IPC surface is finalized.

## Error Modeling

- Several orchestration-layer errors are flattened to `String`/`Arc<str>` too
  early.
  Files:
  - `src/daemon.rs`
  - `src/babel/runtime.rs`
  - `src/lib.rs` (`BabbleError::Other(String)`)
  Why this is a shortcut:
  - It loses structure and source-chain information.
  Follow-up:
  - Prefer typed errors or `eyre::Report` internally, and stringify only at the
    IPC/UI boundary.

## Constants / Magic Values

- A few important constants are still effectively magic values:
  - EXO ULA prefix details
  - default router UDP port
  - various timeout/sleep durations in the Babel runtime
  - tun MTU
  Files:
  - `src/config.rs`
  - `src/babel/runtime.rs`
  - `src/tun.rs`
  Follow-up:
  - Either justify them clearly as real protocol/runtime constants or move them
    into better configuration/abstraction layers.

## Testing

- The typed Babel parser/state layers are tested, but the newer daemon-core and
  routing-stack lifecycle behavior is still lightly tested.
  Files:
  - `src/daemon.rs`
  - `src/routing_stack.rs`
  - `src/main.rs`
  Follow-up:
  - Add focused tests for:
    - keepalive-driven transitions,
    - stack start/stop behavior,
    - public socket command behavior,
    - failure propagation from the routing stack.
