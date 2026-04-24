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
- a dedicated dataplane thread wired into the routing stack,
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

Stable one-hop forwarding is now proven on the four-Mac Thunderbolt lab for an
adjacent pair after switching dataplane TUN I/O on macOS to `tun-rs`
`SyncDevice::recv`/`send` instead of raw fd `read`/`write`.

Steady-state ICMPv6 reachability is also now green across the full four-node
lab ring, and small generic TCP application payloads now work too once the
mesh has converged. The current remaining gap is no longer basic correctness of
non-ICMP transport traffic, but sustained throughput under load: current
`iperf3` testing transfers an initial burst and then collapses into heavy
retransmits and near-zero receive-side throughput.

So the current architecture is good enough for continued correctness and
reliability bring-up, but serious performance work should wait until that
throughput-collapse behavior is understood.

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

### 1. Validate and harden the first UDP dataplane

This should be the next major feature.

The basic pieces are now in place:

- `fib.rs` derives immutable forwarding snapshots from `BabelState`,
- those snapshots now carry the admitted interface set as well as routes,
- `dataplane.rs` provides a dedicated-thread hot-path module using `mio`,
  `socket2`, `crossbeam-channel`, `hashbrown`, `ahash`, `slab`, and
  `arrayvec`.
- `routing_stack.rs` now starts the dataplane and publishes coalesced
  `FibSnapshot` updates into it.
- dataplane socket ownership is now driven by interfaces that currently have
  live Babel neighbours rather than only by currently selected routes, and
  retained sockets are refreshed when an `ifname` resolves to a new ifindex.
- socket reconcile is now best-effort under interface churn: transient
  resolution/open failures are logged and retried without killing the
  dataplane during reconcile.
- unchanged FIB snapshots are now deduplicated in the control plane, so the
  dataplane also carries its own lightweight timer-driven reconcile retry for
  admitted interfaces that still do not have usable sockets.
- the stable-link packet path is now working on the lab ring for adjacent
  one-hop traffic.

So the next step is no longer “invent or wire the modules”.

It is:

- extend live validation from adjacent one-hop traffic to multi-hop and churn,
- harden the remaining dataplane failure/reporting behavior under interface
  churn,
- confirm the interface-bound UDP socket model behaves correctly on the target
  machines,
- and then fill the first obvious protocol gaps such as ICMPv6 error handling.

One caveat that is now proven on the lab Macs: the current macOS receive path
cannot treat "which UDP socket got the packet" as trustworthy interface
attribution. In live tests, packets sent directly over one Thunderbolt
interface are still being received by a different reuseport socket while the
peer scope-id reflects the real physical ingress interface. That means the
current multi-socket receive model is acceptable for basic forwarding bring-up,
but it is not yet a reliable source of receive-side interface truth on macOS.

The likely long-term fix is to move receive-side interface attribution onto
ancillary packet metadata (`IPV6_PKTINFO` / receive-interface data) rather than
inferring it from which socket woke up.

For the current four-Mac Thunderbolt lab, the broad macOS `en*` watcher
heuristic has proven too permissive in practice. The dataplane now corrects
that somewhat by only owning sockets on interfaces that Babel has actually
formed neighbour adjacencies on, but the watcher/bootstrap side is still broad
and may still need a per-host allowlist during bring-up while the long-term
admission policy is refined.

The current broad-admission behavior is acceptable for v1 as long as point to
point and multihop forwarding remain reliable, but it does mean that multiple
wired interfaces can become equally admissible at once.

The desired longer-term policy is:

- admit any interface that Babel can actually form a live neighbour adjacency
  on, regardless of naming convention,
- keep that broad admissibility for reachability,
- but rank competing links by measured quality rather than treating all wired
  links as equivalent.

That future link-scoring direction likely requires:

- computing local link metrics such as latency, loss, and possibly sustainable
  throughput without generating excessive probe traffic,
- sharing or projecting those metrics into the distributed routing view in a
  way Babel can actually consume,
- and then teaching the Babel path-selection logic to prefer the better direct
  link when multiple usable adjacencies exist.

That is explicitly post-v1 work. For the first version, correctness and
reliability of the multihop mesh matter more than optimal link preference.

The current sustained-throughput investigation is therefore focused on two
nearer-term issues before any serious performance tuning:

- understanding whether load-induced loss is primarily backpressure on UDP
  socket send / TUN reinjection,
- and separating that from the now-proven macOS receive-side interface
  attribution oddities.

The live restart-sensitive failures are now narrowed more precisely than that.
On the four-Mac lab, a restarted node can receive an encapsulated packet, push
it through local TUN delivery successfully, and still blackhole the exchange
because the generated return packet is resolved onto a worse broad-admission
path (for example `en1`) instead of the direct Thunderbolt neighbour that just
delivered the request.

So the current main blocker for reliable restart/churn behavior is not "the
dataplane cannot receive or decapsulate packets". It is "the control plane can
still choose an asymmetric installed route that is valid enough for Babel to
advertise, but poor enough to break or destabilize the actual return path".

That strengthens the case for the planned future link-quality policy:

- broad admissibility is still the right v1 reachability rule,
- but equally admissible wired links need a better ranking signal than today's
  flat wired costs,
- otherwise restart-time route selection can still land on a functionally worse
  path even when a direct high-quality neighbour exists.

The current tree now owns the kernel route that steers overlay traffic into the
resident TUN interface:

- the local node `/128` address is installed on the TUN device,
- `EXO_ULA_PREFIX -> tunX` is added when the routing stack turns on,
- that prefix route is removed when the routing stack turns off,
- and `babeld` kernel installs remain disabled.

That means local application traffic can now be steered into the overlay once
the UDP dataplane is active.

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
