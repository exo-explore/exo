# babblerd Handoff

This is the current handoff for a new session picking up `babblerd` work.

## Repo / Branch / State

- Repo: `/home/royalguard/Desktop/exo-all/networking-related/exo-babbler`
- Branch: `babbler`
- Current HEAD when this handoff was refreshed: `5bf8f62f` (`added iperf3`)
- Recent relevant commits:
  - `5bf8f62f` added iperf3
  - `af0b6e17` remove first interface requirement
  - `82adc5d9` it builds
  - `59747529` no longer need optional build flags
  - earlier dataplane bring-up commits remain relevant, but the local repo has
    since moved to `networking-related/exo-babbler`
- Do not trust this file for working-tree cleanliness; run `git status --short`.

## Core Conclusion

The project is past the “is the overlay architecture wrong?” phase.

The current architecture is the right one:

- `babeld` is control plane only
- Babel kernel installs are disabled
- local mesh traffic is steered into a resident TUN
- userspace dataplane forwards one inner IPv6 packet per UDP datagram hop-by-hop over neighbour link-locals

So the main remaining work is now:

- route-selection debugging
- restart/convergence robustness
- load/backpressure/throughput behavior
- eventually link scoring across multiple admissible wired links

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
preferred over high-numbered interfaces such as `en18`. `en0` is temporarily
preferred over high-numbered interfaces such as `en18`. `en0` and `en1` are
temporarily assigned the maximum finite `bias-256` value so shared low-index
networks do not dominate the direct-link smoke tests. This is intentionally a
temporary selection aid so raw throughput work can assume the good direct links
are chosen.

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

## What Works

These things are now real:

- one-hop two-node `ping6`
- adjacent dataplane path
- small low-rate UDP matrix
- encapsulation / decapsulation itself
- basic generic TCP correctness after convergence

A very important live proof point:

On a failing restart-sensitive `e11 -> e16` case, the dataplane itself still did the right work:

- `e11` emitted the encapsulated packet on `en3`
- `e16` received that UDP packet on `en18`
- `e16` dataplane delivered the inner packet into TUN
- the local stack generated a reply packet

So the dataplane is not fundamentally broken anymore.

## What Is Still Broken

The main remaining live problem is restart/convergence behavior and path selection quality.

Observed failure shape:

- after restart, a node can receive and decapsulate correctly
- but the return packet gets resolved onto a worse path, often `en1`, instead of the direct Thunderbolt link
- this creates blackholes or severe instability

So the current blocker is:

- control-plane / derived-FIB route choice under broad multi-link admission
- not “UDP overlay cannot carry packets”

## Very Important macOS Receive-Side Finding

On macOS, receive-side socket attribution is not trustworthy in the current one-socket-per-interface model.

Observed live behavior:

- traffic sent directly over one Thunderbolt link can be delivered to a different UDP socket than expected
- the peer scope-id still reflects the real ingress interface

Implication:

- do not trust “which socket woke up” as authoritative ingress truth on macOS
- if receive-side interface attribution matters, use peer scope-id and likely ancillary packet metadata later

This is a real quirk, but it is not the primary blocker for the current restart blackhole.

## Key Local FIB Caveat

Do not assume route-choice issues are only Babel’s fault.

In `src/fib.rs`, `FibBuilder` collapses multiple installed host routes by choosing the lowest:

- `metric`
- then `refmetric`
- then `handle`

So if restart churn leaves multiple `installed=yes` candidates, babblerd’s derived `FibSnapshot` can still be part of why traffic goes via `en1`.

That means the next debug pass must compare all three:

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
- The current `iperf3` source is the fork at
  `/home/royalguard/Desktop/exo-all/networking-related/iperf3`.
  Commit `962e05b` adds `%scopeID` rendering for link-local IPv6 output.

## Current Docs Are Mostly Accurate

Read:

- `shortcuts.md`
- `future_architectural_directions.md`

They correctly capture:

- broad admissibility is acceptable for v1 reachability
- flat wired costs are not enough for good best-path choice
- forked `babeld` now has the `neighbour-cost` primitive needed for temporary
  external cost steering
- throughput testing must be repeated once route selection is biased toward the
  intended fast direct links
- restart-sensitive failures are now dominated by route selection, not dataplane decode
- there is still debt around interface identity, macOS receive attribution, IPC/authz, and incomplete ICMP/PMTUD behavior

## Important Remaining Technical Debt

Still unresolved:

- public IPC socket is too open
- `ServiceState::On` is not the same as “fully ready/routable”
- broad interface admission is still heuristic
- route ownership of `EXO_ULA_PREFIX` is aggressive
- no ICMPv6 Time Exceeded
- no Packet Too Big handling
- no real backpressure/queueing; `WouldBlock` is still drop-on-backpressure
- sustained throughput still needs a clean measurement pass on biased fast links
- macOS receive-side interface attribution needs a better long-term path
- multi-link path selection still uses a temporary `enN -> N * 100`
  absolute-cost heuristic, not measured scoring
- automatic measured link-scoring policy is not implemented yet

## What Not To Revisit Right Now

These are settled enough for now:

- overlay architecture itself
- TUN + userspace UDP forwarding model
- one-packet-per-datagram framing
- control plane on Tokio, dataplane on dedicated thread
- exact-match `/128` FIB for v1
- `tun-rs` packet I/O on macOS instead of raw fd reads/writes
- disabling Babel kernel installs and owning `EXO_ULA_PREFIX -> tunX` locally

## Best Next Debugging Step

For the restart-sensitive `en1` misroute, inspect all three together for a single problematic `/128` pair:

1. raw Babel route events over time
2. current `BabelState`
3. derived `FibSnapshot`

Goal:

- determine whether the bad path is already in Babel’s installed route set
- or introduced when `FibBuilder` collapses multiple `installed=yes` routes

The temporary `neighbour-cost` policy now lives in `babel/link_policy.rs`.
For each live neighbour on an `enN` interface, `babblerd` sets `coef-256 0`.
Most `enN` links use `bias-256 N * 100 * 256`, so Babel sees lower-numbered
interfaces as cheaper while keeping the distributed Babel view and dataplane
view aligned. `en0` and `en1` are the temporary exceptions: they get the
largest finite `bias-256` value so they lose to the explicit direct-link
interfaces during smoke tests.

## Best Next Live Tests

1. full directed `ping6` matrix on node `/128`s
2. small directed UDP matrix
3. short soak tests on adjacent and two-hop pairs
4. restart/convergence tests
5. physical churn tests
6. for failures, always capture:
   - symptom
   - raw Babel route state / dump
   - current `BabelState`
   - derived FIB state if relevant
   - dataplane logs
   - relevant `ifconfig`

## Short Version

The project is now in the:

- route-selection debugging
- restart convergence
- throughput / backpressure robustness

phase.

The dataplane is basically real.

The current main question is not “can the overlay forward packets at all?”

It is:

- why do restart-time and multi-link route choices still select worse return paths
- and whether that bad choice originates in Babel’s installed set or in local FIB collapse
