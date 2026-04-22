# `babblerd` Future Architectural Directions

This file is not a debt list.

Use [shortcuts.md](./shortcuts.md) for concrete shortcuts, footguns, and
implementation compromises that should be cleaned up later.

This file is for directional reasoning:

- what the current architecture is trying to become,
- which major steps are worth doing next,
- and why those steps are ordered the way they are.

It should evolve as the architecture evolves.

## Current Position

`babblerd` is no longer just a thin wrapper around `babeld`.

It now has the beginnings of a real daemon architecture:

- a resident daemon process,
- a resident TUN interface,
- a keepalive-driven daemon core,
- a heavy routing stack that can turn on and off,
- a typed Babel control/runtime layer,
- a derived FIB layer,
- and a dedicated dataplane thread module scaffold,
- a persisted node identity,
- and a central config module.

That is enough structure to stop treating the whole project as “just Babel
plumbing”.

For bring-up, the current tree also contains a temporary internal self-client
that connects to the public socket and periodically sends keepalive commands.
That is only a testing scaffold so the routing stack stays on without a real
frontend process yet. It should be removed once a real controller exists.

It is also enough structure to begin building the actual dataplane without
needing to perfect every IPC and control-plane detail first.

## The Most Important Architectural Decision

The current codebase is already good enough to serve as the shell around a
first real dataplane.

That means the next major effort should **not** automatically be:

- replacing the line protocol with `zbus`,
- perfecting lease ownership,
- or fully polishing lifecycle semantics.

Those are still desirable, but they are not the blocking step for getting to a
working end-to-end system.

The next big milestone should be:

- a real UDP dataplane,
- driven by the current daemon core and current Babel-derived state,
- with a basic but coherent forwarding model.

In other words: move from “control plane with architecture” to “working router
with acceptable architecture”.

## Near-Term Goal

The near-term target is:

> a basic end-to-end MVC where:
>
> - the daemon has a stable node identity,
> - the daemon can be kept alive by the frontend,
> - the daemon maintains Babel-derived routing state,
> - the daemon can forward packets through a UDP overlay between nodes,
> - and the frontend can inspect enough daemon state to be useful.

This does **not** require the final IPC architecture first.

## Recommended Next Phase

### 1. Wire the UDP dataplane into the daemon

This should be the next major feature.

The basic pieces now exist:

- `fib.rs` derives immutable forwarding snapshots from `BabelState`,
- `dataplane.rs` provides a dedicated-thread hot-path module using `mio`,
  `socket2`, `crossbeam-channel`, `hashbrown`, `ahash`, `slab`, and
  `arrayvec`.

So the next step is no longer “invent those modules”.

It is:

- duplicate/expose the resident TUN fd for the dataplane thread,
- derive `FibSnapshot`s from `watch<Arc<BabelState>>`,
- start the dataplane thread from the routing stack,
- and feed whole snapshots into it.

The current tree now owns the kernel route that steers overlay traffic into the
resident TUN interface:

- the local node `/128` address is installed on the TUN device,
- `EXO_ULA_PREFIX -> tunX` is added when the routing stack turns on,
- that prefix route is removed when the routing stack turns off,
- and `babeld` kernel installs remain disabled.

That means local application traffic can now be steered into the overlay once
the UDP dataplane exists.

The first version can stay simple:

- one UDP datagram carries exactly one inner IPv6 packet,
- no custom framing,
- no batching,
- no crypto,
- no relays,
- no multiplexed control/data protocol.

The dataplane should:

- read packets from TUN,
- classify local-delivery vs forwarding,
- look up next-hop information from a derived forwarding view,
- send encapsulated packets to direct neighbors over UDP,
- receive UDP packets from neighbors,
- decapsulate them,
- either inject them locally into TUN or forward them onward.

This gives the project a real “V” and “M” to go with the current daemon/control
shell.

The current tree now hardcodes:

- physical link MTU assumption: `1500`
- outer overhead assumption: `40 bytes IPv6 + 8 bytes UDP`
- derived TUN MTU: `1452`

That is acceptable for bring-up, but it is still only a temporary model.

The future direction should be:

- route-aware MTU derivation,
- PMTUD-aware behavior,
- and better support for environments where hop-to-hop links can use jumbo
  frames without exposing that complexity to user traffic.

### 2. Keep the derived forwarding table separate from `BabelState`

`BabelState` should remain a mirror of what `babeld` says.

The current code now reflects that direction:

- `BabelState` is still the protocol mirror,
- `FibSnapshot` is the dataplane view.

The next layer should be a derived forwarding table/FIB that:

- is keyed by destination prefix or node address,
- only keeps the routes the dataplane should actually use,
- captures next hop / outgoing interface / any other forwarding metadata,
- and is cheap for the dataplane to consult.

This avoids mixing:

- “what Babel currently knows”
- with
- “what the UDP router should do with packets”.

### 3. Add a stronger public state/readiness model

The current `ServiceState` is useful, but it is only lifecycle state:

- `Off`
- `Starting`
- `On`
- `Stopping`

That is not the same thing as routing readiness.

Once the dataplane exists, a separate readiness/status view should exist too.
For example, the frontend may want to distinguish:

- daemon is idle,
- daemon is starting,
- Babel is running but no eligible interfaces exist,
- interfaces exist but no neighbors are usable,
- forwarding is nominal,
- forwarding is degraded.

That should be modeled separately from `ServiceState`, not by making
`ServiceState::On` carry too much meaning.

## What Can Wait Until After the Dataplane Exists

These are still desirable, but they do not need to block the first end-to-end
router:

### `zbus` / D-Bus-style IPC

This is still the likely long-term direction.

But the current line protocol is good enough for:

- `keepalive <ttl_ms>`
- `get-state`

while the dataplane is being built.

So `zbus` should remain a planned improvement, not the immediate blocker.

### Per-client leases

The daemon should eventually track leases per client/connection rather than via
a single global keepalive deadline.

That is a real architectural improvement, but it is control-plane polish rather
than dataplane unblocker.

It can happen after the first router path works.

### Structured diagnostics/debug output

Right now diagnostics are tracing-only.

That is acceptable for development while the dataplane is first being brought
up.

A configurable debug stream or structured diagnostics feed should be added
later, preferably once the public IPC shape is stabilized.

## The First Dataplane Should Stay Intentionally Small

The first version should avoid solving every future overlay concern.

It should **not** attempt to solve:

- encryption,
- authentication,
- path quality metrics beyond what Babel already provides,
- batching,
- relay protocols,
- or multi-transport negotiation.

The first version should prove the simplest useful thing:

- stable node addresses,
- UDP transport between neighbors,
- Babel-driven next-hop selection,
- TUN injection/extraction,
- packet forwarding that actually works end-to-end.

If that works, the rest can be improved incrementally.

## Architectural Path After the First Dataplane Works

Once the basic dataplane exists and works, the likely next path is:

1. Improve the public state/readiness model.
2. Replace the ad-hoc control socket with `zbus`.
3. Replace the single keepalive deadline with per-client leases.
4. Tighten interface admission beyond the current broad heuristic.
5. Pin and explicitly invoke the exact forked `babeld`.
6. Harden node-id file mode checks and other local security edges.
7. Revisit diagnostics streaming.
8. Revisit platform abstractions around TUN / transport / forwarding.

That ordering is intentional:

- prove the router first,
- then harden and refine the daemon architecture around it.

## Guiding Principle

The project should prefer:

- a coherent working router with a few acknowledged shortcuts

over:

- a beautifully abstract control plane that still does not move packets.

That does **not** mean ignoring architecture.

It means using the current architecture as a platform for the next real
capability, rather than repeatedly polishing the control shell before the
dataplane exists.
