# fix(discovery): multicast discovery never forms a cluster on macOS

## Summary

On macOS, nodes never discover each other and every node reports a topology of
one. **Three** independent defects combine to cause it: two restrict which
interfaces are announced on, and a third stops announcing altogether after the
first discovery. This PR fixes all three.

Verified on 4× Mac Studio (M3 Ultra, macOS 26.6.2) connected by both wifi and a
full Thunderbolt 5 mesh.

## Defect 1 — `AddrInUse` silently drops interfaces

`rust/networking/src/discovery.rs`, in the `netwatcher` callback:

```rust
match sock.join_multicast_v6(&GROUP, *iface_idx) {
    Ok(()) => ifaces.lock().push(/* ... */),
    Err(e) if e.kind() != io::ErrorKind::AddrInUse => { warn!(/* ... */) }
    _ => {}          // AddrInUse lands here — interface never registered
}
```

Multicast membership is held per-socket, so on macOS the **second and every
subsequent** `join_multicast_v6` for the same group returns `AddrInUse`. The
existing comment ("skip AddrInUse - just means we've already joined the mv6")
shows the intent was to tolerate it, but because the guard excludes `AddrInUse`
the value falls through to `_ => {}` and the interface is never pushed onto
`ifaces`.

Since `announce()` iterates `ifaces`, only the first interface to join
successfully ever receives a Hello.

## Defect 2 — the multicast egress interface is never set

`announce()` sends to `SocketAddrV6::new(GROUP, port, 0, iface_idx)`, relying on
the scope id to select the outgoing interface. That is not sufficient for IPv6
multicast: `IPV6_MULTICAST_IF` must be set on the socket before each send.
Without it every datagram leaves via the default multicast interface no matter
which scoped address it was sent to. The socket only ever calls
`set_multicast_loop_v6`.

## Defect 3 — a blocking `connect_peer` silences discovery permanently

`rust/networking/src/lib.rs`. `Discovery::next()` is the **only** driver of
`announce()` — it announces on each 1s tick and returns as soon as a peer is
discovered. The task that consumes it awaits `connect_peer` inline:

```rust
let discovered = discovery.next().await;   // announces while polling
// ...
runtime.connect_peer(&discovered.zid.into(), &[locator]).await;   // blocks here
```

`connect_peer` can block for a long time — indefinitely, when a peer is
unreachable or its handshake stalls. While it is blocked the task never returns
to `next()`, so the node **stops announcing entirely**, and never resumes.

This is the defect that actually prevents the cluster forming. Fixing defects 1
and 2 alone changes which interfaces get the first two announcements; it does
not stop the loop from wedging immediately afterwards.

## The fix

1. Register the interface when the join returns either `Ok` or `AddrInUse`, and
   warn only on genuine errors.
2. Set `IPV6_MULTICAST_IF` (via `socket2::SockRef`) for the target interface
   immediately before each `send_to`.
3. Spawn `connect_peer` on its own task so the discovery loop returns to
   `next()` immediately and keeps announcing while connections are attempted.

## Verification

With `RUST_LOG=networking=debug` on the `heartbeat` example:

**Before**
```
announcing Hello([...]) to [[ff12::e0a1:de89%25]]
```
`%25` is `awdl0`.

**After**
```
announces on 14 interfaces: lo0 anpi3 en3 en4 en5 en12 en1 awdl0 llw0 utun0 ...
```
Now including `en1` (wifi) and `en3`/`en4`/`en5` (all three Thunderbolt links).

## On the evidence for defect 3

Defect 3 is reported on **code inspection**, not measurement, and I want to be
explicit about why.

I originally supported it with packet captures: a node emitting only two
`announcing Hello` lines and then going silent, `tcpdump` showing zero packets
on the discovery port, and a Python sender/listener pair on the same group and
interfaces exchanging packets every time.

That comparison turned out to be invalid. The test host runs per-process network
filters (Little Snitch and a SentinelOne network extension) which block egress
for freshly built, non-notarised binaries. A three-line Rust program that only
opens a TCP connection reproduces it:

```
fresh Rust binary -> tcp 10.10.1.x:22 : timed out after 7.0s
python3 (same shell, same second) -> same host:port : connected in 0.01s
```

So the "no packets on the wire" observations say nothing about this code, and
the Python comparison was measuring the filter's allowlist rather than a
difference in socket handling. I am withdrawing that evidence.

What stands on its own is the shape of the code: `Discovery::next()` is the only
driver of both `announce()` and `recv_from`, and the consuming task awaits
`connect_peer` inline. Any long block there stops announcements and reception
for the whole node until it returns. That is true regardless of the host, and it
is what defect 3 describes.

Defects 1 and 2 are unaffected - the interface-count measurement below is taken
from the announce list before any packet is sent, so the filter cannot influence
it.

## Related, not included

- `--bootstrap-peers` is accepted by the CLI and documented as taking libp2p
  multiaddrs, but `main.py` raises `ValueError("Bootstrap peers has been
  temporarily removed")`. With multicast discovery failing there is no manual
  fallback.
- `uv.lock` pins `mlx-vlm` to 0.4.4 while `pyproject.toml` declares
  `>=0.3.11`. 0.6.17 carries 222 model architectures versus 60, including
  `glm5_next` and `deepseek_v4`.
- The `networking` logger is hardcoded to `INFO` in `logger_setup`, and the
  Rust side reaches Python via `pyo3_log`, so the `trace!` drop-reasons in
  `discovery.rs` cannot be enabled by any CLI flag. Raising that to `DEBUG`
  under `-v` would have made this a five-minute diagnosis.

## Follow-up: GPU-budget-blind placement, and a pipeline deadlock on glm5_next

Two further findings from bringing up GLM-5.3-Flash (glm5_next, 102.6 GiB
2-bit MoE) on the same 4x Mac Studio cluster.

### Placement ignores the GPU working-set ceiling

`placement` sizes shards from *system* RAM. macOS caps a process's GPU
working set at `iogpu.wired_limit_mb` (default ~75% of RAM, ~72 GiB on a
96 GiB machine), and memory already wired counts against it. When a shard
exceeds what the GPU can hold the runner dies with

    [METAL] Command buffer execution failed: Insufficient Memory
    (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)

Critically the runner aborts *mid-collective*, so its peers stay blocked in
`recv` indefinitely. From the outside that is indistinguishable from a
deadlock, and the real cause is only visible in the runner's stderr. On this
cluster EXO placed 32 of 45 layers (~73 GiB) on one node and every attempt
looked like a hang.

Suggested fix: clamp per-node shard size by the GPU limit (and by already
wired memory), not just by `ramAvailable`.

### Pipeline + MlxJaccl deadlocks on main for ANY real model

With a GPU-safe placement the model loads completely - all 45 layers, no OOM,
full 6/6 RDMA mesh - and then rank 0 hangs forever in the `mx.eval(output)`
inside `PipelineLastLayer`, with MLX's stream thread idle (nothing queued)
and no Metal error. Peers sit correctly in `first.recv`.

Isolation performed (all on the real checkpoint, all passing, so none of
these is the cause):

| test | result |
| --- | --- |
| rank 0's shard (layers 0-11) sliced via `get_inner_model`/`get_layers`, evaluated standalone | 1.1 s |
| same, with the model's own `make_cache()` (11 entries, correct types) | 0.2 s |
| purely local `mx.eval` while a jaccl peer is blocked in `recv` | 0.20 s |
| `PipelineFirstLayer`/`PipelineLastLayer` across 2 jaccl ranks, 4-layer toy model | 0.00 s |
| same wrappers, 22 layers x 4096 with a 4-D hyper-connection shaped input | 0.00 s |

So the model, the layer slicing, the cache, RDMA, and the pipeline wrappers
are each fine in isolation; the hang needs the real glm5_next graph *and*
multi-rank pipeline execution together. It reproduces with 2, 3 and 4 ranks.

### Also required to get this far (not in this PR - they belong upstream)

- `Glm5NextModel` caches `ssm_idx`/`fa_idx` at `__init__` over the full layer
  list. Pipeline sharding replaces `self.layers` with a slice, so those
  indices then address the wrong entry of the per-shard cache, and the
  `next(..., 0)` fallback silently returns 0 for a shard containing no layer
  of that type - feeding a full-attention cache to `create_ssm_mask`. On the
  real 4-way split 2 of 4 shards were mis-indexed. Making them properties
  resolved against the current layer list fixes it.
- mlx-vlm's `sanitize` re-nests only `f_a_proj.weight`/`f_b_proj.weight`
  under `forget_gate.`, leaving `.scales`/`.biases` behind on a quantized
  checkpoint.
- mlx-lm's `load_model` tests `config["quantization"][p]` with the *nested*
  module path while the checkpoint keys it un-nested, so those layers stay
  `nn.Linear` while their siblings are quantized and the fused
  linear-attention matmul dies on `m.scales`.

#### Not model specific

The same hang reproduces with `mlx-community/GLM-4.7-Flash-4bit` - a
different architecture (`glm4_moe_lite`), 16 GiB, 14/33-layer shards, and a
model that serves correctly in the released 1.0.71 app. Identical signature:

    rank 0   cpu frozen 0:04.81   [PIPE r=0] last.layer DONE   stream IDLE
    rank 1   cpu 8:36 -> 8:41     [PIPE r=1] first.recv ENTER  stream IDLE

So this is not about glm5_next, quantization, MoE, or model size. Pipeline
sharding over MlxJaccl appears broken on `main` for real models in general.
Note the released app runs the same cluster fine with **Tensor** sharding, so
the fault is specific to the Pipeline path.

Reduced repro attempts that all PASS (so the fault is none of these):
`PipelineFirstLayer`/`PipelineLastLayer` across 2 jaccl ranks with a toy
model, and the same at realistic scale (22 layers x 4096, 4-D input) - both
complete in 0.00 s. A real model's shard evaluated standalone completes in
0.2-1.1 s. Only real-model-plus-pipeline hangs.

#### Tensor sharding on main stalls too

For completeness, `Tensor` + `MlxJaccl` x2 on the same cluster and the same
known-good model does not stall at warmup but at *load*:

    runner A  layersLoaded 46/47   cpu advancing
    runner B  layersLoaded  0/47   cpu frozen at 0:02.92

One rank never begins loading. So on `main` both distributed backends fail on
this hardware - Pipeline (either MlxRing or MlxJaccl) deadlocks at warmup, and
Tensor stalls during load - while the released 1.0.71 app drives the same four
machines correctly with Tensor + MlxJaccl.

All of the above was re-verified with `auto_parallel.py` restored to pristine
(`git checkout`), so none of it is an artifact of the tracing used to diagnose
it.

## Update: GLM-5.3-Flash does run, on v1.0.71 built from source

Everything above was found while trying to make `main` work. It turned out not
to be the right path. **v1.0.71 built from source** federates over libp2p with a
working `--bootstrap-peers`, forms a full RDMA mesh in ~20s, and reaches
`Pipeline/MlxJaccl ready=3/3` with GLM-5.3-Flash (102.6 GiB, `glm5_next`) split
15/15/15 across three Mac Studios. The same placement on `main` never got past
`mx.eval` in `PipelineLastLayer`.

Worth stating plainly: the shipped `.app` cannot accept a new architecture,
because its Python lives in a PyInstaller archive. That is not true of a source
checkout, and mistaking the two is what made this look impossible for a while.

The out-of-tree changes this needs are in `patches/`, as diffs against the
pristine upstream wheels. Two of them are arguably exo's problem rather than
mlx's:

- exo detects recurrent (SSM) layers with `isinstance` against **mlx-lm's**
  `ArraysCache`. A model whose `make_cache` returns mlx-vlm's API-identical
  duplicate fails `has_non_kv_caches()`, the post-prefill rollback and the trim
  path, and exo then calls `.trim()` on recurrent state. Matching on a
  structural property (e.g. `is_trimmable`, or absence of `.keys`) would be
  robust to which package a model's caches come from.
- `pipeline_auto_parallel` replaces `inner.layers` with a slice after the model
  is constructed. Any model that caches layer indices in `__init__` is silently
  wrong afterwards - masks get built from the wrong cache entry and generation
  degrades without raising. exo already special-cases `GptOssMoeModel` for
  `layer_types` here; a general "re-derive per-shard state after slicing" hook
  would cover the rest.

Also confirmed: placement sizes shards from system RAM with no notion of the GPU
working-set ceiling (`iogpu.wired_limit_mb`, default ~75% of RAM) or of memory
already wired. Exceeding it aborts the runner *mid-collective* with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`, and the peers then block in
`recv` forever - externally indistinguishable from a deadlock.

Open: output quality on the `oQ2` 2-bit checkpoint is wrong (valid tokens, no
semantics). A control on the identical 3-node MlxJaccl pipeline is coherent
(`GLM-4.7-Flash-4bit` returns `391` and `Paris`), so the pipeline and RDMA path
are not implicated. mlx-vlm's own loader rejects this checkpoint outright (483
unplaceable parameters), so there is no native reference to compare against.
