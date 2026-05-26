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

It is also enough structure to improve the actual dataplane without needing to
perfect every IPC and control-plane detail first.

Stable forwarding is now proven on the four-Mac Thunderbolt lab:

- adjacent overlay traffic works,
- single-hop forwarded UDP works at modest rates,
- steady-state ICMPv6 reachability is green across the full ring,
- small generic TCP application payloads work after convergence,
- and the temporary `enN` link-cost policy successfully keeps steady-state
  routes away from `en0`/`en1` and toward the intended lower-numbered direct
  Thunderbolt-style links.

The current remaining gap is no longer basic dataplane correctness or forcing
Babel onto the intended route. The route-selection heuristic is now good enough
to expose the next bottleneck: raw dataplane throughput and overload behavior.

The latest lab numbers put the userspace overlay far below the direct physical
UDP baseline:

- direct UDP without the software router: about `11 Gbit/s` observed outside
  the overlay,
- direct overlay UDP, `e4 -> e16`, `iperf3 -6 -u -b 0 -t 10`: about
  `1.46 Gbit/s` received with negligible loss,
- direct overlay TCP, `e4 -> e16`, `iperf3 -6 -b 0 -t 10`: about
  `1.24 Gbit/s` received,
- single-hop overlay UDP, `e2 -> e16`, `iperf3 -6 -u -b 0 -t 10`: server
  intervals around `1.11-1.16 Gbit/s` with `12-14%` loss in one run; a later
  run sent about `1.32 Gbit/s` and dataplane counters showed about `1.16M`
  packets delivered, but the `iperf3` control connection broke before a valid
  receiver summary was produced and the overlay path needed a `babblerd`
  restart to recover,
- single-hop overlay TCP, `e2 -> e16`, `iperf3 -6 -b 0 -t 10`: about
  `1.07 Gbit/s` received.

So the architecture is good enough for continued correctness bring-up, but
serious performance work should now treat packet processing cost, syscalls,
copies, batching, and backpressure as the main suspects.

## The Most Important Architectural Decision

The current codebase is already good enough to serve as the shell around the
first real dataplane.

That means the next major effort should **not** automatically be:

- replacing the line protocol with `zbus`,
- perfecting lease ownership,
- or fully polishing lifecycle semantics.

Those are still desirable, but they are not the blocking step for getting to a
working end-to-end system.

The next big milestone should be:

- make the existing UDP dataplane observable enough to explain overload,
- make the forwarding path robust when the sender exceeds what the router can
  currently drain,
- and then reduce per-packet cost enough to move beyond the current
  `1-1.5 Gbit/s` overlay ceiling.

In other words: the project has moved from “build the router” to “make the
router fast and predictable under load”.

## Near-Term Goal

The near-term target is now:

> a measurable end-to-end router where:
>
> - the daemon has a stable node identity,
> - the daemon can be kept alive by the frontend,
> - the daemon maintains Babel-derived routing state,
> - the daemon forwards through the UDP overlay between nodes,
> - the route heuristic selects the intended fast links after convergence,
> - and overload is visible through counters rather than guessed from `iperf3`
>   alone.

This still does **not** require the final IPC architecture first.

## Recommended Next Phase

### 1. Harden and measure the UDP dataplane

This is the current major feature.

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
- the dataplane now drains ready TUN and UDP fds up to fairness budgets instead
  of handling only one packet per readiness event.
- the dataplane logs useful packet/drop counters every few seconds when active:
  TUN RX/TX, UDP RX/TX, TUN-to-UDP, forwarded, local-delivered, no-route,
  invalid, hop-limit, and UDP/TUN `WouldBlock` drops.
- the UDP receive path now mutates the stack buffer slice directly instead of
  allocating a `Vec` per received packet.
- the dataplane compiles each `FibSnapshot` into a local fast route table that
  stores direct socket slots, avoiding the old per-packet `FibEntry` clone and
  route-ifname-to-socket lookup.

So the next step is no longer “invent or wire the modules”.

It is:

- explain the current `1-1.5 Gbit/s` ceiling against packet counters and host
  CPU/syscall behavior,
- make full-blast UDP overload recover cleanly instead of destabilizing the
  path,
- compare direct, single-hop, and multi-hop runs with the same counter set,
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

The forked `babeld` used by the Nix build can now start with no managed
interfaces as long as `babblerd` gives it a read-write local control socket.
That means `babblerd` no longer waits for a first interface before spawning
`babeld`; it starts `babeld` immediately and sends `interface <ifname>` commands
later as the watcher discovers eligible links.

For the current four-Mac Thunderbolt lab, the broad macOS `en*` watcher
heuristic has proven too permissive in practice. The dataplane now corrects
that somewhat by only owning sockets on interfaces that Babel has actually
formed neighbour adjacencies on, but the watcher/bootstrap side is still broad.
The env allowlist should remain an escape hatch, not the default topology
description.

The current broad-admission behavior is acceptable for v1 as long as point to
point and multihop forwarding remain reliable, but it does mean that multiple
wired interfaces can become equally admissible at once.

The desired longer-term policy is:

- admit any interface that Babel can actually form a live neighbour adjacency
  on, regardless of naming convention,
- keep that broad admissibility for reachability,
- but rank competing links by measured quality rather than treating all wired
  links as equivalent.

The forked `babeld` now has the primitive needed for this: the read-write local
socket accepts `neighbour-cost` commands, and neighbour monitor lines report the
active `external-bias-256` and `external-coef-256` fields. Those values are
fixed-point controls in units of `1/256`: the bias is additive, and the
coefficient multiplies the native base cost before the RTT penalty is added.

The measured scoring system is still post-MVP work. It likely requires:

- computing local link metrics such as latency, loss, and possibly sustainable
  throughput without generating excessive probe traffic,
- feeding those metrics into `neighbour-cost`,
- and then validating that Babel path selection consistently prefers the better
  direct link when multiple usable adjacencies exist.

For MVP, `babblerd` now applies a deliberately simpler policy: for each live
neighbour on an `enN` interface, it sends `neighbour-cost` with
`coef-256 0`. Most `enN` links get `bias-256 N * 100 * 256`, making Babel
treat the native base cost as an absolute synthetic interface-index cost of
roughly `N * 100`, so lower-numbered links such as `en2`, `en3`, and `en4`
win over high-numbered links such as `en18`. `en0` and `en1` are temporarily
deprioritized with the maximum finite `bias-256` value so shared low-index
networks do not dominate Thunderbolt-style links during throughput smoke tests.
This is not a robust scoring model; it is a temporary selection heuristic so
raw throughput work can proceed on the intended fast links.

The current sustained-throughput investigation is therefore focused on the
dataplane hot path and overload behavior:

- direct overlay UDP tops out around `1.46 Gbit/s` in the latest test, far
  below the `11 Gbit/s` direct physical UDP baseline,
- single-hop overlay UDP can receive around `1.1 Gbit/s` during full-blast
  `-b 0` tests; a later run sent about `1.32 Gbit/s` and delivered roughly
  `1.16M` packets at the receiver according to dataplane counters, but the
  `iperf3` control connection broke before a receiver summary was produced and
  the overlay path needed a restart to recover,
- full-blast UDP should be treated as a stress/failure test until the
  backpressure story is better,
- route selection is still important during convergence, but after the mesh
  settles the `enN` policy is no longer the main explanation for the throughput
  gap.

Be precise about "control traffic" during these tests. Babel's own protocol
packets should remain link-local traffic on the direct physical `en*`
interfaces that `babblerd` explicitly gives to `babeld`; the TUN/overlay
interface is not a Babel interface. The overlay does carry traffic addressed to
node ULAs, including `iperf3` payload, the `iperf3` TCP control/session
connection, and `ping6` to peer ULAs. Saturating the overlay can still disturb
Babel indirectly through shared physical NIC queues, socket buffers, and CPU
scheduling, but not because Babel packets are routed through the userspace
overlay.

That distinction matters for the next diagnosis step. Protecting or separating
control traffic may make tests less fragile and may avoid `iperf3` control
connection failures, but it does not by itself close the `7-8x` dataplane
throughput gap. When a full-load run wedges the path, capture raw Babel state,
the derived `BabelState`, the dataplane FIB/socket map, dataplane counters, and
host route state before concluding whether the failure is route churn, overlay
queue exhaustion, or application-control failure.

At small inner MTUs, approaching `11 Gbit/s` is a packet-rate problem. Ignoring
Ethernet/IP/UDP overhead, the per-packet budget is:

```text
packet_budget_seconds = dataplane_packet_bytes * 8 / target_bits_per_second
```

For `11 Gbit/s`:

- `1452` byte packets, the UDP default TUN MTU: about `947 kpps`, or
  `1.06 us/packet`.
- `9000` byte packets, the forced-TCP default TUN MTU: about `153 kpps`, or
  `6.55 us/packet`.
- `1500` byte packets: about `917 kpps`, or `1.09 us/packet`.
- `1200` byte packets: about `1.15 Mpps`, or `873 ns/packet`.
- `9000` byte jumbo packets: about `153 kpps`, or `6.55 us/packet`.
- `64` byte minimum-size packets: about `21.5 Mpps`, or `46.5 ns/packet`.

So for MTU-sized `iperf3` traffic this is not a "few nanoseconds per packet"
target, but it is roughly a one-microsecond total budget per packet. That
budget has to cover all user/kernel crossings, copies, route lookup, hop-limit
mutation on forwarded packets, UDP send/receive, TUN read/write, and scheduler
overhead. The latest direct overlay result of `1.46 Gbit/s` at `1452` bytes
corresponds to roughly `126 kpps`, or about `8 us/packet`, so getting near
`11 Gbit/s` means shrinking per-packet cost by around `7-8x` or reducing packet
rate with larger packets/aggregation.

Likely optimization directions, in priority order:

- keep improving counters and expose them over the public state surface, so
  tests can distinguish no-route, UDP send backpressure, TUN reinjection
  backpressure, invalid packets, forwarding, and local delivery without log
  scraping;
- add recovery/backpressure policy for overload rather than just
  drop-on-`WouldBlock`;
- keep expanding OS packet batching where the target platform allows it. The
  dataplane now receives through `iroh-quinn-udp`, which maps to
  `recvmsg_x`/`sendmsg_x` on Apple fast builds and `recvmmsg` on Linux-like
  Unix. Transmit still sends one packet at a time from the forwarding loop, so
  real output batching remains future work;
- consider overlay aggregation, where one outer UDP datagram carries several
  inner packets, to amortize syscall and UDP/IP overhead;
- explore jumbo MTUs on the Thunderbolt links, because `9000` byte packets
  reduce the `11 Gbit/s` packet rate from about `947 kpps` to about `153 kpps`;
- consider multi-core dataplane sharding once single-thread costs are
  measured, because one dedicated thread is a likely ceiling for this design;
- treat kernel-bypass or moving more forwarding into the kernel as a separate
  architecture track if `11 Gbit/s` at standard MTU is a hard requirement on
  macOS.

Before jumbo frames or overlay aggregation, the current per-packet cost audit
points at these near-term bottlenecks:

- A transit packet still implies one UDP receive syscall and one UDP send
  syscall in the forwarding process. At `11 Gbit/s` and `1452` byte packets,
  that is roughly `947 kpps`, or nearly `1.9M` UDP syscalls/sec on the transit
  node before counting TUN work on endpoints. That alone makes a full `7-8x`
  improvement unlikely from ordinary Rust-level cleanup.
- The UDP ingress path was still cloning `socket.ifname` for every received
  packet just to support logging/error context. Because `Box<str>::clone()`
  allocates, that is an avoidable heap allocation per overlay packet.
- UDP ingress previously used `recv_from` even though the peer address is not
  needed for forwarding. Decoding the source address is useful for debugging
  but should not be mandatory hot-path work.
- TUN and UDP packet buffers were stack-created as zeroed arrays for each
  packet. Reusing worker-owned buffers avoids repeated stack initialization and
  keeps the packet loop closer to "syscall, parse, lookup, syscall".
- The send path still builds a `SocketAddrV6` and emits one `iroh-quinn-udp`
  transmit per packet. A future connected per-neighbour output-socket model
  could remove that address construction and let the kernel cache more route
  state.
- The dataplane now uses `iroh-quinn-udp` with the Apple fast datapath, which
  exposes `sendmsg_x`/`recvmsg_x` batching. This is not a QUIC routing change;
  the useful part is Quinn's UDP socket layer. The current patch batches
  receive calls and routes transmit through the same abstraction, but still
  emits one transmit call per forwarded packet. The next useful version should
  group same-peer/same-size packets into a single `Transmit` with
  `segment_size` set.
- Any output batching must be opportunistic, not latency-gating. A single ready
  packet must still be sent immediately; batching should flush at the end of a
  poll/drain slice or when the next packet targets a different peer/size, never
  wait for a full batch.
- The tree now also has an opt-in TCP neighbour transport for Mac Thunderbolt
  experiments: `BABBLER_ROUTER_TRANSPORT=tcp`, `--router-transport tcp`, or
  `--force-tcp`. UDP remains the default. TCP mode opens scoped link-local TCP
  streams to next-hop neighbours, frames inner IPv6 packets with a `u16`
  big-endian length, batches framed packets in bounded per-peer write buffers,
  and flushes partial batches at drain/poll boundaries. This is intended to
  test whether macOS Thunderbolt TCP can expose the higher native TCP path while
  avoiding one syscall per inner packet.
- macOS TCP mode keeps one wildcard IPv6 listener per daemon, not one
  `IPV6_BOUND_IF` listener per admitted interface. The lab showed
  per-interface-bound TCP listeners can stall Thunderbolt handshakes in
  `SYN_RCVD`; outbound streams are still scoped to the Babel-selected
  interface. Inbound streams are accepted only from link-local peers that match
  a live Babel neighbour on the accepted interface/scope.
- TCP mode now defaults the TUN MTU to `9000` to cut packet rate through
  macOS `utun`; `BABBLER_TUN_MTU=<mtu>` or `--tun-mtu <mtu>` can be used for
  jumbo sweeps. All forced-TCP peers in one test mesh must use the same value,
  because receivers reject TCP frames larger than their local TUN MTU. UDP mode
  still defaults to the physical-MTU-derived `1452`.
- TCP mode changes overload behavior. Instead of UDP drops on send backpressure,
  it can accumulate bounded per-stream pending bytes and then drop once that
  bound is reached. Its counters must be watched separately:
  `tcp_tx_batches`, `tcp_queued_packets`, `tcp_written_frames`,
  `tcp_rx_frames`, `tcp_blocked_writes`, `tcp_queue_drops`,
  `tcp_frame_errors`, and `tcp_stream_errors`.

The first low-risk cleanup pass has now landed: UDP ingress no longer clones
`ifname`, packet buffers are worker-owned instead of stack-created for every
packet, and UDP I/O goes through `iroh-quinn-udp` so the OS-specific fast path
is selected by the crate. These are worth doing, but they should be expected to
remove avoidable overhead rather than close the full `7-8x` gap by themselves.

The current tree now owns the kernel route that steers overlay traffic into the
resident TUN interface:

- the local node `/128` address is installed on the TUN device,
- `EXO_ULA_PREFIX -> tunX` is added when the routing stack turns on,
- that prefix route is removed when the routing stack turns off,
- and `babeld` kernel installs remain disabled.

That means local application traffic can now be steered into the overlay once
the UDP dataplane is active.

The current MVP can stay simple:

- one UDP datagram carries exactly one inner IPv6 packet,
- no custom framing,
- no batching,
- no crypto,
- no relays,
- no multiplexed control/data protocol.

The dataplane currently does this basic loop:

- reads packets from TUN,
- classifies local-delivery vs forwarding,
- looks up next-hop information from a derived forwarding view,
- sends encapsulated packets to direct neighbors over UDP,
- receives UDP packets from neighbors,
- decapsulates them,
- either injects them locally into TUN or forwards them onward.

This gives the project a real “V” and “M” to go with the current daemon/control
shell.

The current tree now hardcodes:

- physical link MTU assumption: `1500`
- outer overhead assumption: `40 bytes IPv6 + 8 bytes UDP`
- UDP TUN MTU default: `1452`
- forced-TCP TUN MTU default: `9000`, with `BABBLER_TUN_MTU`/`--tun-mtu`
  override

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

The derived forwarding layer should keep moving toward a table that:

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

Now that the dataplane exists, a separate readiness/status view should exist too.
For example, the frontend may want to distinguish:

- daemon is idle,
- daemon is starting,
- Babel is running but no eligible interfaces exist,
- interfaces exist but no neighbors are usable,
- forwarding is nominal,
- forwarding is degraded.

That should be modeled separately from `ServiceState`, not by making
`ServiceState::On` carry too much meaning.

## What Can Wait Until After Throughput Bring-Up

These are still desirable, but they do not need to block the current
throughput/backpressure work:

### `zbus` / D-Bus-style IPC

This is still the likely long-term direction.

But the current line protocol is good enough for:

- `keepalive <ttl_ms>`
- `get-state`

while the dataplane is being measured and tuned.

So `zbus` should remain a planned improvement, not the immediate blocker.

### Per-client leases

The daemon should eventually track leases per client/connection rather than via
a single global keepalive deadline.

That is a real architectural improvement, but it is control-plane polish rather
than dataplane unblocker.

It can happen after the current router path is faster and better characterized.

### Structured diagnostics/debug output

Right now diagnostics are tracing-only.

That is acceptable for development while the dataplane is being characterized.

A configurable debug stream or structured diagnostics feed should be added
later, preferably once the public IPC shape is stabilized.

## The MVP Dataplane Should Stay Intentionally Small

The MVP version should avoid solving every future overlay concern.

It should **not** attempt to solve:

- encryption,
- authentication,
- path quality metrics beyond what Babel already provides,
- relay protocols,
- or multi-transport negotiation.

Batching and packet aggregation are now valid throughput experiments, but they
should be evaluated as dataplane optimizations rather than bundled with
unrelated control-plane redesign.

The current version has proven the simplest useful thing:

- stable node addresses,
- UDP transport between neighbors,
- Babel-driven next-hop selection,
- TUN injection/extraction,
- packet forwarding that actually works end-to-end.

The rest can be improved incrementally.

## Architectural Path After the MVP Dataplane Works

Now that the basic dataplane exists and works, the likely next path is:

1. Expose dataplane counters and route/FIB state through a better public status
   surface.
2. Make overload/backpressure behavior recoverable and measurable.
3. Benchmark batching, aggregation, jumbo MTUs, and eventually multi-core
   dataplane options.
4. Replace the temporary `enN` link policy with measured link-quality scoring.
5. Replace the ad-hoc control socket with `zbus`.
6. Replace the single keepalive deadline with per-client leases.
7. Tighten interface admission beyond the current broad heuristic.
8. Pin and explicitly invoke the exact forked `babeld`.
9. Harden node-id file mode checks and other local security edges.
10. Revisit diagnostics streaming.
11. Revisit platform abstractions around TUN / transport / forwarding.

That ordering is intentional:

- keep the router measurable while improving throughput,
- then harden and refine the daemon architecture around it.

## Guiding Principle

The project should prefer:

- a coherent working router with a few acknowledged shortcuts

over:

- a beautifully abstract control plane that still does not move packets.

That does **not** mean ignoring architecture.

It means using the current architecture as a platform for the next real
capability, rather than polishing the control shell ahead of the dataplane's
current throughput and overload problems.
