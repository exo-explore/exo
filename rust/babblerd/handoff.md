# babblerd Handoff

This is the current handoff for a new session picking up `babblerd` work.

## Repo / Branch / State

- Repo: `/home/royalguard/Desktop/exo-all/networking-related/exo-babbler`
- Branch: `babbler`
- Current HEAD when this handoff was refreshed: `0b7a3ad3` (`receiver side batching (different socket type)`)
- Recent relevant commits:
  - `0b7a3ad3` wires dataplane UDP receive/send through `iroh-quinn-udp`
    while keeping `mio` readiness; receive-side batching is in, true transmit
    batching is not
  - `3cbb5758` removes several avoidable hot-path costs and adds dataplane
    counter coverage
  - `2f038588` deprioritizes `en0` and `en1` with maximum finite neighbour cost
  - `b02cf2cb` adds temporary `enN -> N * 100` link scoring
  - `5a158a51` adds Babel neighbour-cost parsing/command support
  - `5bf8f62f` added iperf3
  - `af0b6e17` remove first interface requirement
  - `82adc5d9` it builds
  - `59747529` no longer need optional build flags
  - earlier dataplane bring-up commits remain relevant, but the local repo has
    since moved to `networking-related/exo-babbler`
- Do not trust this file for working-tree cleanliness; run `git status --short`.

## Handoff To Next Agent

The committed baseline before the forced-TCP work was `0b7a3ad3`. Do not infer
working-tree cleanliness from this file; run `git status --short`.

What is implemented:

- `src/dataplane.rs` keeps `mio::net::UdpSocket` for readiness polling.
- Each dataplane interface socket also owns an `iroh_quinn_udp::UdpSocketState`.
- UDP receive now uses `UdpSocketState::recv`, so Apple fast builds can use
  `recvmsg_x` and Linux-like Unix can use `recvmmsg` through the crate.
- UDP send now goes through `UdpSocketState::try_send`, but still one transmit
  call per forwarded packet.
- Receive batching handles `RecvMeta::stride`, so GRO-style buffers containing
  multiple datagrams are split back into inner packets.
- Regression test:
  `dataplane::tests::udp_batch_recv_returns_single_datagram_without_full_batch`
  proves a single datagram returns immediately as a one-packet batch.
- The current working tree adds an opt-in TCP neighbour transport:
  `--force-tcp`, `--router-transport tcp`, or `BABBLER_ROUTER_TRANSPORT=tcp`.
  UDP remains the default transport.
- TCP mode opens scoped link-local TCP streams to next-hop neighbours, frames
  inner IPv6 packets with a `u16` big-endian length, batches framed packets into
  bounded per-peer write buffers, and flushes partial batches at drain/poll
  boundaries or when the batch reaches the target size.
- On macOS, TCP mode keeps one wildcard IPv6 listener per daemon and separate
  no-fd slots for admitted interfaces. Lab testing found per-interface-bound
  TCP listeners could leave e4/e16 Thunderbolt handshakes stuck in `SYN_RCVD`;
  outbound TCP streams are still scoped to the Babel-selected interface.
  Accepted streams are admitted only when the peer is a link-local Babel
  neighbour on the accepted scope/interface.
- TCP mode is intended as an experimental Mac Thunderbolt fast path to reduce
  one-syscall-per-packet overhead. It is not the default mesh transport.
- TCP mode uses a jumbo `65535` byte TUN MTU by default to reduce userspace TUN
  packet rate on the Mac Thunderbolt fast path. Override with `--tun-mtu <mtu>`
  or `BABBLER_TUN_MTU=<mtu>` when sweeping Mac `utun` limits. All forced-TCP
  peers in a test mesh must use the same TUN MTU; a receiver rejects TCP frames
  larger than its local MTU. UDP mode keeps the old `1452` byte default.
- TCP mode uses `256 KiB` TCP read buffers and `256 KiB` opportunistic write
  batch targets. This avoids splitting max-size framed packets across multiple
  TCP read calls and lets several max-size inner packets share one TCP write
  when the TUN queue is busy. Partial batches still flush at drain/poll
  boundaries.

What is not implemented:

- No true UDP output batching yet.
- No UDP output queue, aggregation, or waiting-to-fill behavior.
- No connected per-neighbour output sockets yet.
- Full-load remote tests have not yet been recorded for TCP mode in this
  handoff.

If implementing actual UDP transmit batching next:

1. Add a worker-owned `TxBatch` scratch buffer.
2. Append only packets with the same output socket, next-hop peer, and packet
   length.
3. Flush on peer change, packet-size change, full batch, end of TUN/UDP drain
   slice, poll-loop boundary, snapshot/reconcile, stop, or send error.
4. Flush via one `Transmit { contents: batch_bytes, segment_size: Some(packet_len), ... }`.
5. Never wait for a full batch. Batching is syscall amortization inside an
   already-ready drain turn, not a latency queue.
6. Add tests for single-packet flush, peer-change flush, size-change flush, and
   full-batch flush.

Docs are part of the fix. Any future code change should update this handoff and
the relevant architecture/lab notes in the same patch, especially when it
changes what is implemented versus future work.

## Core Conclusion

The project is past the “is the overlay architecture wrong?” phase.

The current architecture is the right one:

- `babeld` is control plane only
- Babel kernel installs are disabled
- local mesh traffic is steered into a resident TUN
- userspace dataplane forwards one inner IPv6 packet per UDP datagram hop-by-hop over neighbour link-locals

So the main remaining work is now:

- dataplane throughput and overload behavior
- making full-blast UDP recover cleanly
- restart/convergence robustness around transient route choices
- eventually replacing the temporary `enN` link scoring with measured link
  quality

## Why The Old Approach Failed

The original macOS idea was effectively:

- let `babeld` install routes
- try to make kernel source selection behave

That did not work cleanly for this use case:

- no usable IPv6 pref-src route install path on macOS/BSD for this design
- no native source-specific IPv6 routing model that solves the app behavior wanted here
- putting ULAs on `lo0` or `utun` did not reliably fix source selection
- app-aware binding alone was not enough in practice

That is why the design pivoted to the userspace overlay dataplane.

## Files To Read First

- `src/daemon.rs`
- `src/routing_stack.rs`
- `src/dataplane.rs`
- `src/fib.rs`
- `src/tun.rs`
- `src/babel/runtime.rs`
- `src/route_ctl.rs`
- `lab_topology_reference.md`
- `shortcuts.md`
- `future_architectural_directions.md`

## Current Intended Architecture

Model:

- one stable node `/128` on TUN
- `EXO_ULA_PREFIX -> tunX` installed by `babblerd`
- `babeld` kernel installs are disabled / ignored
- `BabelState` mirrors `babeld`
- `FibSnapshot` is a reduced immutable dataplane view
- control plane stays on Tokio
- dataplane is a dedicated thread
- one UDP datagram carries exactly one inner IPv6 packet
- no custom framing yet
- outer IPv6 destination is neighbour link-local
- outer UDP port is `router_udp_port`

This is the v1 forwarding model:

- exact-match `/128` host routes only
- interface identity in FIB is `ifname`
- dataplane owns sockets from admitted interface set
- admitted dataplane interfaces come from live Babel neighbours

## Important Design Decisions Already Landed

- typed Babel parsing/state model, not raw string handling
- monitor-driven Babel runtime, not periodic dump polling
- persistent node identity across restarts
- explicit daemon lifecycle: `Off | Starting | On | Stopping`
- resident TUN lifetime, separate heavy routing stack
- dataplane thread + immutable FIB snapshot swaps
- socket ownership from admitted interface set, not only current route set
- same-name/new-ifindex socket refresh handled
- timer-driven socket reconcile retry in dataplane, so deduped unchanged FIB snapshots do not suppress retries forever
- dataplane exit supervision back into routing stack / daemon
- macOS dataplane uses `tun-rs` packet I/O (`SyncDevice::recv/send`), not raw fd reads/writes

## Important Forked `babeld` Changes

The Nix build now uses a forked `babeld` from
`/home/royalguard/Desktop/exo-all/networking-related/babeld`, packaged as
`1.13.1+local`.

Recent fork behavior that matters to `babblerd`:

- `babeld` can start with no managed interfaces when a read-write local control
  socket exists.
- `babblerd` now spawns `babeld` immediately and adds interfaces later with
  local-socket `interface <ifname>` commands from the watcher.
- `kernel-install false` is still used so `babeld` performs route selection and
  reports installed routes without touching kernel routes.
- `neighbour-cost <ifname> <link-local-neighbour> bias-256 <bias> coef-256 <coef>`
  is available for external link-cost steering.
- `bias-256` is a signed fixed-point additive value in units of `1/256`.
  `256` adds one Babel cost unit and `-256` subtracts one.
- `coef-256` is an unsigned fixed-point multiplier in units of `1/256`.
  `256` is neutral, `128` halves the native base cost, and `0` ignores the
  native base cost while still adding the RTT penalty.
- `dump`/`monitor` neighbour lines now include `external-bias-256` and
  `external-coef-256` before `cost`.

Automatic measured link scoring is not part of the MVP. The current temporary
policy is a simple Mac heuristic: for most `enN` links, set an absolute
synthetic base cost around `N * 100` with `coef-256 0` and
`bias-256 N * 100 * 256`, so lower-numbered Thunderbolt-style interfaces are
preferred over high-numbered interfaces such as `en18`. `en0` and `en1` are
temporarily assigned the maximum finite `bias-256` value (`16776704`, yielding
cost `65534` with `coef-256 0`) so shared low-index networks do not dominate
the direct-link smoke tests. This is intentionally a temporary selection aid so
raw throughput work can assume the good direct links are chosen after
convergence. Immediately after a restart, Babel may still transiently install a
bad high-cost route until better neighbour state arrives.

## Very Important Fix After Earlier Handovers

The previously-deployed node addresses were wrong.

There was a real bug in `EXO_ULA_PREFIX` construction:

- intended prefix: `fde0:20c6:1fa7:ffff::/64`
- broken runtime prefix had become: `20c6:1fa7:ffff:0::/64`

Cause:

- `config.rs` used a `u128` left-shift construction that dropped the high `fde0` bits

Fix:

- commit `b0f508ac` changed the prefix constant to explicit hextets and added a regression test

Live verification after redeploy:

- `e4 utun5`: `fde0:20c6:1fa7:ffff:cc78:aec2:d64e:f125/128`
- `e2 utun5`: `fde0:20c6:1fa7:ffff:aeb:e53a:cb17:aa42/128`
- `e11 utun5`: `fde0:20c6:1fa7:ffff:34a:26dd:46ff:1a3f/128`
- `e16 utun5`: `fde0:20c6:1fa7:ffff:7c5d:5e2d:54df:e665/128`

So any older notes mentioning the truncated non-ULA prefix are stale.

## Current Dataplane Behavior

In `src/dataplane.rs`:

- TUN ingress:
  - read inner IPv6 packet
  - parse destination
  - drop self-directed
  - FIB lookup
  - send raw inner packet as UDP payload to neighbour
- UDP ingress:
  - receive UDP payload
  - payload is raw inner IPv6 packet
  - if destination local, inject into TUN
  - else decrement inner hop limit and forward

Fast-path traits:

- dedicated OS thread
- `mio` polling
- `socket2` UDP sockets
- immutable FIB snapshot swaps over `crossbeam-channel`
- no lock on packet lookup path
- ready TUN/UDP fds are drained up to fairness budgets
- UDP receive uses a stack buffer slice directly, not `to_vec()`
- each `FibSnapshot` is compiled into dataplane-local fast routes that include
  direct socket slots, so packets no longer clone `FibEntry` or do an ifname
  lookup to find the output socket
- dataplane counters are logged periodically when active: TUN RX/TX, UDP RX/TX,
  TUN-to-UDP, forwarded, local-delivered, no-route, invalid, hop-limit, and
  UDP/TUN `WouldBlock` drops

## What Works

These things are now real:

- one-hop two-node `ping6`
- adjacent dataplane path
- small low-rate UDP matrix
- single-hop forwarded UDP at `100M` with no loss in the latest smoke test
- encapsulation / decapsulation itself
- basic generic TCP correctness after convergence
- steady-state route choice avoiding `en0`/`en1` after the temporary cost policy
  has converged
- full-bandwidth direct overlay tests that show a repeatable `1-1.5 Gbit/s`
  dataplane ceiling rather than a basic correctness failure

Latest route examples after convergence:

- `e2 -> e11`: direct on `en3`, metric `300`
- `e2 -> e16`: single-hop via `e4` on `en2`, metric `400`
- `e4 -> e16`: direct on `en2`, metric `200`
- `e16 -> e2`: single-hop via `e11` on `en2`, metric `400`

## What Is Still Broken

The main remaining live problem is dataplane throughput and overload behavior.

Observed latest performance shape:

- direct physical UDP without the software router is about `11 Gbit/s`
  according to the latest external baseline,
- direct overlay UDP is about `1.46 Gbit/s` received,
- direct overlay TCP is about `1.24 Gbit/s` received,
- single-hop overlay TCP is about `1.07-1.08 Gbit/s` received,
- single-hop overlay UDP at `-b 0` receives around `1.11-1.16 Gbit/s` during
  one run and loses `12-14%`; a later run sent about `1.32 Gbit/s` and
  delivered about `1.16M` packets according to dataplane counters, but the
  `iperf3` control connection broke before a valid receiver summary was
  produced and the overlay path needed a `babblerd` restart to recover.

So the current blocker is:

- not “UDP overlay cannot carry packets”,
- not primarily “Babel selected the wrong steady-state route”,
- but packet processing cost, syscall/copy overhead, and drop/recovery behavior
  when the dataplane is overdriven.

Restart/convergence route quality is still worth watching. Immediately after a
restart, Babel can transiently install high-cost `en0`/`en1` routes before the
better neighbours converge. But after convergence, the current `enN` policy is
good enough for throughput work.

At `11 Gbit/s` with `1452` byte inner packets, the budget is about `947 kpps`,
or `1.06 us/packet`. The latest direct overlay result at `1.46 Gbit/s` is about
`126 kpps`, or `8 us/packet`. Closing the gap means cutting per-packet cost by
roughly `7-8x`, increasing effective packet size with jumbo/aggregation, or
both.

Near-term cost audit before jumbo/aggregation:

- A transit node needs one UDP receive syscall and one UDP send syscall per
  forwarded packet, so standard-MTU `11 Gbit/s` implies nearly `1.9M` UDP
  syscalls/sec on that node.
- Rust-level cleanup alone is unlikely to recover a full `7-8x`, but avoidable
  hot-path work should still be removed before blaming the architecture.
- First targets now landed: remove the per-packet `socket.ifname` clone on UDP
  ingress, avoid source-address decoding when the peer address is not needed,
  and reuse packet buffers instead of stack-zeroing a new array per packet.
- Initial `iroh-quinn-udp` wiring has landed. The dataplane still keeps `mio`
  sockets for readiness, but each socket also has a `UdpSocketState`; receives
  can batch through the crate's Apple `recvmsg_x` path or Linux `recvmmsg`
  path, and sends go through the same abstraction. This is not QUIC.
- Next candidates are connected per-neighbour output sockets and true output
  batching: collect same-peer/same-size packets and send them as one
  `Transmit` with `segment_size` set, instead of one transmit call per forwarded
  packet.
- Batching invariant: never wait for a full batch. Receive batching must return
  whatever is already queued on the nonblocking fd, and future output batching
  must flush partial batches at drain boundaries or peer/size changes.

Control traffic terminology:

- Babel protocol packets should stay on the direct link-local `en*`
  interfaces that `babblerd` explicitly adds to `babeld`; the TUN/overlay
  interface is not added to Babel.
- `iperf3` data, the `iperf3` TCP control/session connection, and `ping6` to a
  peer ULA do traverse the overlay because they are addressed to node ULAs.
- Full overlay load can still perturb Babel indirectly through shared physical
  NIC queues, kernel buffers, and CPU scheduling, but Babel packets are not
  being encapsulated by the software router in the normal design.
- Therefore "protect control traffic" means making measurements and recovery
  less fragile; it is not a direct explanation for the order-of-magnitude
  throughput gap.

## Very Important macOS Receive-Side Finding

On macOS, receive-side socket attribution is not trustworthy in the current one-socket-per-interface model.

Observed live behavior:

- traffic sent directly over one Thunderbolt link can be delivered to a different UDP socket than expected
- the peer scope-id still reflects the real ingress interface

Implication:

- do not trust “which socket woke up” as authoritative ingress truth on macOS
- if receive-side interface attribution matters, use peer scope-id and likely ancillary packet metadata later

This is a real quirk, but it is not the primary explanation for the current
order-of-magnitude throughput gap.

## Key Local FIB Caveat

Do not assume route-choice issues are only Babel’s fault.

In `src/fib.rs`, `FibBuilder` collapses multiple installed host routes by choosing the lowest:

- `metric`
- then `refmetric`
- then `handle`

So if restart churn leaves multiple `installed=yes` candidates, babblerd’s
derived `FibSnapshot` can still be part of why traffic transiently goes via
`en0`/`en1`.

If route-choice anomalies reappear, compare all three:

1. raw Babel route events / dump
2. current `BabelState`
3. derived `FibSnapshot`

Not Babel in isolation.

## Lab Topology / Operations

Source of truth file:

- `lab_topology_reference.md`

Key facts:

- four Mac minis
- hostnames:
  - `e4@e4`
  - `e2@e2`
  - `e11@e11`
  - `e16@e16`
- ring topology:
  - `e4 -> e2 -> e11 -> e16 -> e4`
- remote repo path:
  - `~/babeld-exo`
- each remote must `git pull` before running
- current start command:
  - `cd ~/babeld-exo && git pull && RUST_LOG=info sudo -E nix run .#babblerd --impure`
- temporary internal keepalive client exists, so no external `nc -U ...` client is needed just to keep daemon alive
- `iperf3` is provided by the flake:
  - `nix run .#iperf3 -- -s`
  - `nix run .#iperf3 -- -c <addr>`
- Force the experimental TCP dataplane transport with either:
  - `BABBLER_ROUTER_TRANSPORT=tcp RUST_LOG=info sudo -E nix run .#babblerd --impure`
  - `RUST_LOG=info sudo -E nix run .#babblerd --impure -- --force-tcp`
- In forced TCP mode on macOS, each daemon has one wildcard listener shared by
  all admitted interfaces; per-peer outbound streams remain interface-scoped.
  Accepted TCP streams are rejected unless the peer is link-local and matches a
  live Babel neighbour on the accepted scope/interface.
- Forced TCP defaults the TUN MTU to `65535`. Use `--tun-mtu <mtu>` or
  `BABBLER_TUN_MTU=<mtu>` to test smaller values such as `16384` or `32768`. Keep the
  value identical on every forced-TCP peer in a run.
- The current `iperf3` source is the fork at
  `/home/royalguard/Desktop/exo-all/networking-related/iperf3`.
  Commit `962e05b` adds `%scopeID` rendering for link-local IPv6 output.

## Current Docs Status

Read:

- `shortcuts.md`
- `future_architectural_directions.md`
- `lab_topology_reference.md`

They correctly capture:

- broad admissibility is acceptable for v1 reachability
- flat wired costs are not enough for good best-path choice, hence the
  temporary `enN` policy
- forked `babeld` now has the `neighbour-cost` primitive needed for temporary
  external cost steering
- the route-selection heuristic is now active and good enough after convergence
  to expose dataplane throughput limits
- the latest direct/single-hop `iperf3` results and the `11 Gbit/s` direct UDP
  baseline
- the remaining debt around backpressure, batching/aggregation, jumbo MTU,
  interface identity, macOS receive attribution, IPC/authz, and incomplete
  ICMP/PMTUD behavior

## Important Remaining Technical Debt

Still unresolved:

- public IPC socket is too open
- `ServiceState::On` is not the same as “fully ready/routable”
- broad interface admission is still heuristic
- route ownership of `EXO_ULA_PREFIX` is aggressive
- no ICMPv6 Time Exceeded
- no Packet Too Big handling
- no real backpressure/queueing; `WouldBlock` is still drop-on-backpressure
- counters are logged but not yet exposed as a structured public status surface
- direct overlay throughput is still about `1.46 Gbit/s`, far below the
  `11 Gbit/s` direct physical UDP baseline
- full-blast single-hop UDP can destabilize the overlay path after the run
- macOS receive-side interface attribution needs a better long-term path
- multi-link path selection still uses a temporary `enN -> N * 100`
  absolute-cost heuristic, not measured scoring
- automatic measured link-scoring policy is not implemented yet

## What Not To Revisit Right Now

These are settled enough for now:

- overlay architecture itself
- TUN + userspace UDP forwarding model
- one-packet-per-datagram framing as the MVP correctness model; batching or
  aggregation can now be evaluated as a performance extension
- control plane on Tokio, dataplane on dedicated thread
- exact-match `/128` FIB for v1
- `tun-rs` packet I/O on macOS instead of raw fd reads/writes
- disabling Babel kernel installs and owning `EXO_ULA_PREFIX -> tunX` locally

## Best Next Performance Step

Use the current route heuristic and focus on dataplane cost.

The most concrete current experiment is forced TCP transport on the Mac
Thunderbolt lab. It should be compared against the UDP default with the same
routes, same iperf pairs, same dataplane counter deltas, and same recovery
checks. TCP mode batches framed inner packets before kernel writes; UDP mode
still emits one send operation per forwarded packet.

Capture each run with:

1. `iperf3` sender/receiver summaries
2. `babblerd` dataplane counter deltas
3. CPU usage on sender, transit node, and receiver
4. route/FIB snapshots before and after the run
5. whether bidirectional `ping6` still works after the run
6. for TCP mode, `tcp_tx_batches`, `tcp_queued_packets`,
   `tcp_written_frames`, `tcp_rx_frames`, `tcp_rejected_peers`,
   `tcp_queue_drops`, `tcp_frame_errors`, and `tcp_stream_errors` deltas

Goal:

- separate CPU/syscall ceiling from UDP/TUN backpressure,
- explain the single-hop UDP wedge and distinguish overlay application-control
  failure from Babel route churn,
- and measure whether receive-side batching changed direct and single-hop
  throughput before implementing true transmit batching, aggregation, jumbo MTU
  support, or multi-core sharding.

The temporary `neighbour-cost` policy now lives in `babel/link_policy.rs`.
For each live neighbour on an `enN` interface, `babblerd` sets `coef-256 0`.
Most `enN` links use `bias-256 N * 100 * 256`, so Babel sees lower-numbered
interfaces as cheaper while keeping the distributed Babel view and dataplane
view aligned. `en0` and `en1` are the temporary exceptions: they get the
largest finite `bias-256` value so they lose to the explicit direct-link
interfaces during smoke tests.

If route-choice bugs reappear, then inspect all three together for the
problematic `/128` pair: raw Babel route events over time, current `BabelState`,
and derived `FibSnapshot`.

## Best Next Live Tests

1. full directed `ping6` matrix on node `/128`s
2. small directed UDP matrix
3. direct overlay TCP/UDP `-b 0` with counter capture
4. single-hop overlay TCP/UDP `-b 0` with counter capture
5. short soak tests on adjacent and two-hop pairs
6. restart/convergence tests
7. physical churn tests
8. for failures, always capture:
   - symptom
   - raw Babel route state / dump
   - current `BabelState`
   - derived FIB state if relevant
   - dataplane logs
   - relevant `ifconfig`

## Short Version

The project is now in the:

- throughput / backpressure robustness
- overload recovery
- restart convergence sanity-checking

phase.

The dataplane is basically real.

The current main question is not “can the overlay forward packets at all?”

It is:

- why the userspace router tops out around `1-1.5 Gbit/s` when direct physical
  UDP can reach about `11 Gbit/s`
- and how much of that gap comes from one-packet-per-datagram syscalls/copies,
  single-thread processing, TUN/UDP backpressure, or recoverability bugs under
  overload
