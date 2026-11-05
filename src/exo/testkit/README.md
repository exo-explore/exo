# exo.testkit: cluster tests on one machine

`exo.testkit` starts N real exo nodes as separate processes on one machine, with no MLX and
no GPU, and lets a test break them: kill -9, SIGSTOP/SIGCONT, quit, restart, and network
cuts. Everything is exo's production code except the model, which a stub engine replaces
with deterministic tokens, so a test can check every token a client receives.

## Run the tests

```bash
uv run pytest -m "cluster and not slow"        # fast set, as CI runs on Linux: ~5 min
uv run pytest src/exo/testkit -m ""            # every scenario, as the nightly runs: ~15 min
uv run pytest                                  # the default run skips cluster tests
```

The `cluster` marker keeps them out of the default run: they start real nodes and take
minutes. CI runs the fast set on Linux for every push and pull request (the "Cluster
scenarios" jobs in pipeline.yml), and `cluster-nightly.yml` runs every scenario on macOS
and Linux each night.

When a cluster doesn't form, the timeline says why it can: each node's ports (noted at
spawn), and `diagnose()`'s lines, prefixed "diagnosis:": what the simulated network passed
and dropped (Hellos per node, WhatsUps per pair, zenoh connections, bytes and connect
errors per pair), which processes hold each node's discovery and zenoh port, and the master
each node follows.

A failing test's report includes each cluster's timeline and the end of every node's log.
The files stay in pytest's `tmp_path` (`<cluster>/n0/exo_log/exo.log`, `n0/output.log`,
`timeline.log`).

## Write a scenario

```python
import time

import pytest

from exo.testkit.tests.conftest import MakeCluster

pytestmark = pytest.mark.cluster


def test_killing_a_follower_ends_its_chat(make_cluster: MakeCluster) -> None:
    cluster = make_cluster(3)                    # started, one master, edges: ready
    master = cluster.wait_for_master(timeout=30) # never assume which node it is
    a, b = (i for i in range(3) if i != master)
    instance = cluster.launch(on=(a, b))         # a 2-node pipeline ring instance, ready
    chat = cluster.chat_in_background("story [stub tokens=3000 rate=50]", via=master)
    time.sleep(3)
    cluster.kill(b)                              # kill -9 of that node and its runners
    result = chat.result(timeout=60)
    assert not result.ok and not result.problems()   # clean error, no corrupted tokens
    cluster.wait_until_consistent(timeout=35)    # one master, no ghosts, no leaked tasks
```

Write the expectation as what main does today. For a failure main still has, write the
behaviour you want and mark the test
`@pytest.mark.xfail(strict=True, raises=EXPECTED_FAILURES, reason=...)` (from
`exo.testkit.cluster`), saying what goes wrong and what was measured. The PR that fixes it deletes the marker; strict means an unexpected pass fails
the run. Give each timing bound the value you measured and the margin you added, in the
docstring. Mark tests that take more than about 30 s `@pytest.mark.slow`; they run nightly.

### Timing on a busy machine

Several clusters often share one machine (CI runners, or several test runs at once), and under
load macOS stretches short sleeps and timers: a 20 ms sleep has been seen to take 127 ms.
The stub keeps its token rate anyway, by catching up on late steps (up to 1 s of backlog,
so a stopped runner still resumes at its rate instead of in a burst). exo's own asyncio
timers in the nodes stretch the same way, so give sub-second bounds a generous margin, and
derive bounds from exo's constants (the 15 s node timeout, the 30 s master silence)
rather than from measured sub-second times.

xfail markers name what the test may fail with, `raises=EXPECTED_FAILURES` (an assertion or
a `HarnessTimeoutError`): a node that dies on its own (`NodeExitedError`) or a cluster that
never forms (`ClusterStartError`) then fails the run instead of passing as the expected
failure. Put a scenario's setup (the steps before its fault) in `with _setup("..."):`
(test_scenarios.py): anything that goes wrong there raises `ScenarioSetupError`, which an
xfail never expects. Only signal processes the harness started: `kill`, `pause` and `quit` act on the
node's own process group, recorded when it was spawned. Never `pkill` by pattern; other
clusters on the machine run the same command line. And never `killpg` an exo node with
SIGKILL yourself: its group includes multiprocessing's resource tracker, which unlinks the
node's named POSIX semaphores (about 40 per node with a runner) once the node is gone.
Killed along with it, they leak until the machine restarts, and macOS allows 10,000 in all.
`Cluster.kill` spares it (`test_killing_a_node_leaves_its_semaphores_to_the_resource_tracker`).

## The `Cluster` API (exo/testkit/cluster.py)

Nodes are addressed by index. `n0` has the largest id, so on main it usually becomes the
first master, but that depends on the election and on which link opens first: look the master
up (`wait_for_master()`) and pick followers from the rest. A scenario that stops the master
should check, just before, that the node it stops is still the master.

| what | call |
|---|---|
| start and stop | `Cluster(size, root)`, `.start()`, `.close()` (or the `make_cluster` fixture) |
| faults | `.kill(i)` (SIGKILL to the node and its runners, sparing the resource tracker), `.kill_node_only(i)` (SIGKILL to exo's own process alone; returns what it left, which must exit by itself), `.pause(i)` / `.resume(i)` (SIGSTOP / SIGCONT), `.quit(i)` (SIGINT; returns seconds to exit), `.restart(i, new_identity=True)` |
| network | `.cut(i, j, mode)`, `.isolate(i, mode)`, `.heal()`. `mode="blackhole"` holds every byte, as when a cable is pulled; `"reset"` tears connections down at once |
| models | `.launch(on=(1, 2))` (placed on exactly those nodes, then waits until ready), `.launch()` (exo places it), `.place(...)` (doesn't wait for ready), `.preview(on)` (raises `NoPlacementError` when exo offers none), `.wait_ready(id)`, `.wait_serving(via=i)` (after a fault: a chat that succeeds, since the state can still show the runners from before it as ready), `.delete(id)` |
| chats | `.chat(prompt, via=i, stream=True, max_tokens=None)` returns a `ChatResult`; `.chat_in_background(...)` returns a handle with `.result(timeout)` |
| results | `ChatResult.ok`, `.error`, `.content`, `.finish_reason`, `.problems()` (empty when every token is right; a reply cut short by an error must be an exact prefix), `.longest_gap`, `.summary()` |
| state | `.state(i)` (the node's `/state` as exo's `State`), `.master()`, `.masters()`, `.wait_for_master(timeout, other_than=)`, `.violations()`, `.wait_until_consistent(timeout)` |
| helpers | `.wait_for(predicate, timeout=, what=)` polls every 0.25 s until the predicate returns `True` (it must return a bool; anything else raises `TypeError`); `.wait_for_value(get, timeout=, what=)` returns the first value that isn't `None` (0, `False`, `[]` are values). Both raise `HarnessTimeoutError` on timeout, and `NodeExitedError` at once if a node died by itself. `.note(message)` adds a line to the timeline. `.fallbacks` lists what the harness had to force (a node that didn't exit within 10 s of SIGINT at `close()`, a runner that outlived its node's quit); the fixture fails the test at teardown unless it set `cluster.expect_fallbacks = True` |

`violations()` checks the invariants: one master that every running node follows; every
running node sees exactly the running nodes (no ghosts, none missing) and the same
instances; no instance on a node outside the cluster; no task left Pending or Running
(`allow_tasks=True` while requests are in flight).

## The stub engine's prompts

The reply to a prompt is a function of the prompt: token i is `"<key>.<i> "`, where key is the
first 8 hex digits of the prompt's sha256. Give each request its own prompt. A `[stub ...]` tag
in the prompt sets the reply and its faults:

| directive | meaning (default) |
|---|---|
| `tokens=N` | reply length (16); `max_tokens` below it ends early with `length` |
| `rate=R` | tokens per second per request (200) |
| `prefill=S` | seconds before the first token (0) |
| `fault=crash\|raise\|hang\|stall at=K rank=R stall=S` | rank R's runner, just before token K: dies by SIGKILL, raises, stops forever (the supervisor's 90 s stall check ends it), or stops S seconds (10) |

Delays and faults on the way to serving go in the nodes' stub config:
`cluster.configure_stub(StubConfig(connect_seconds=, load_seconds=, warmup_seconds=,
fault_phase="connect"|"load"|"warmup", fault=..., fault_rank=R, fault_limit=N),
nodes=(i, ...))`. A runner reads it at the start of each phase. A fault hits only rank R if
`fault_rank` is given, and only the first N times on each node if `fault_limit` is given, so
the replacement runner gets through.

## What is real and what isn't

Real, and unchanged: discovery's Hello/WhatsUp protocol and dial rule, zenoh, the election,
the master, workers, the API, downloads (models are pre-installed and marked complete), the
runner supervisor, and the runner process with its state machine and its error handling.

Replaced or simulated:

- **The model.** `StubBuilder`/`StubEngine` (stub/engine.py) are run by exo's own bootstrap
  in place of MLX (stub/entrypoint.py). The ranks of a multi-node instance agree on every
  step's requests and cancellations over a TCP ring built from the instance's placement
  (stub/ring.py), as the real generators agree through MLX collectives. The ring copies
  what MLX 0.32.4's ring backend did on real hardware: the higher rank of each neighbour pair
  connects, trying 5 times (1, 2, 4, 8, 16 s) before "Couldn't connect"; a rank that only
  accepts waits forever; listeners use SO_REUSEPORT and close once the ring is up; any
  process at the address pairs; and a lost neighbour raises "[ring] connection to a peer was
  lost". MLX gives every rank the placement's port; here every node is on one machine, so
  each rank gets a port of its own from the harness, the same for every runner of that
  rank.
- **Ports.** Every port the harness hands out (each node's API, zenoh and discovery port,
  and the stub ranks' ports) comes from below the OS's ephemeral range and is leased: a lock
  file per port in the temp directory, held until the cluster closes (exo/testkit/ports.py).
  A port found by binding port 0 can be handed to another socket before the node binds it,
  and for a discovery port exo doesn't notice: on macOS its [::] bind with SO_REUSEPORT
  succeeds next to a socket on [::1], which then gets every Hello meant for the node.
- **Discovery's transport.** Each node sends its Hellos to the harness's relay on `[::1]`
  (`EXO_TESTKIT_DISCOVERY_ANNOUNCE_TO`, the one hook in production code,
  rust/networking/src/discovery.rs). On one machine multicast can't work: nodes share the
  discovery port and a unicast reply reaches only one of them, and on macOS a process
  without Local Network access can't send multicast. The relay passes Hellos between nodes
  and rewrites each reply's zenoh port to a per-pair proxy.
- **The network.** Every zenoh link and every stub ring link passes through a proxy in the
  harness (netsim.py), which is what lets a test cut a pair without root.
- **Three things in each node process** (node.py): its node id is chosen by the harness;
  runners use the stub; and the worker's reachability probe asks each peer at that peer's own
  API port, only on 127.0.0.1, and respects cuts (exo itself probes every peer at its own API
  port, see `test_nodes_with_different_api_ports_can_run_a_model_together`). The probe reports
  each peer under a label of its own, 127.0.0.k, because exo's worker tells topology edges
  apart only by address. The labels are only labels: Linux routes all of 127/8 to the
  loopback but macOS doesn't, so nothing may dial them. The stub ring's connections go
  through the harness, which maps them to 127.0.0.1; any new exo code that dials an edge
  address would work on Linux and hang on macOS here.

What this can't show: anything MLX or Metal does (GPU timeouts, real load and warm-up times,
memory), Thunderbolt and RDMA, real sleep (a SIGSTOP keeps sockets open and the clock
running), and real Wi-Fi loss, latency or packet loss. Use real hosts for those.
