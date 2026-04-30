"""RDMA probe subprocess entry point.

Invoked as `python -m exo.utils.profilers.rdma_probe_main <rank> <params_json>`.

Initialises an MLX `jaccl` distributed group with two ranks (rank 0 is the
node that initiated the probe and the jaccl coordinator). Runs three
back-to-back micro-benchmarks and prints the result on rank 0:

  1. **Upload bandwidth (rank 0 -> rank 1)**: `mx.distributed.send` on
     rank 0, `recv` on rank 1, large payload. Time on rank 0 → bytes/sec.
  2. **Download bandwidth (rank 1 -> rank 0)**: same pattern reversed.
  3. **Latency**: tiny-payload `all_sum` ping-pong (one round-trip per
     iteration on a 2-rank group). Time/iter = RTT.

Runs in a *child process* because `mx.distributed.init` is process-global —
calling it inside the worker would conflict with active inference groups.
"""

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from exo.utils.profilers.rdma_probe import RdmaProbeParams

WARMUP_ITERATIONS = 1
LATENCY_PAYLOAD_BYTES = 64
LATENCY_ITERATIONS = 50


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <rank> <params_json>",
            file=sys.stderr,
        )
        return 2

    rank_arg, params_json = sys.argv[1], sys.argv[2]
    try:
        rank = int(rank_arg)
        params = RdmaProbeParams.model_validate_json(params_json)
    except (ValueError, ValidationError) as e:
        print(f"invalid arguments: {e}", file=sys.stderr)
        return 2

    if rank not in (0, 1):
        print(f"rank must be 0 or 1, got {rank}", file=sys.stderr)
        return 2

    source_iface = params.source_rdma_iface
    sink_iface = params.sink_rdma_iface
    coordinator_ip = params.coordinator_ip
    coordinator_port = params.coordinator_port
    payload_bytes = params.payload_bytes
    iterations = params.iterations

    # `ibv_devs[i][j]` is the RDMA interface on node i used to reach node j.
    # Diagonals are None. The matrix is symmetric in shape but the interface
    # names differ per node.
    ibv_devs = [[None, sink_iface], [source_iface, None]]

    with tempfile.NamedTemporaryFile(
        prefix="exo_rdma_probe_ibv_devs_", suffix=".json", mode="w", delete=False
    ) as f:
        json.dump(ibv_devs, f)
        ibv_devs_path = f.name

    try:
        os.environ["MLX_IBV_DEVICES"] = ibv_devs_path
        os.environ["MLX_RANK"] = str(rank)
        os.environ["MLX_JACCL_COORDINATOR"] = f"{coordinator_ip}:{coordinator_port}"

        import mlx.core as mx  # imported here so failures surface as a clean exit

        try:
            group = mx.distributed.init(backend="jaccl", strict=True)
        except Exception as e:  # noqa: BLE001
            print(f"jaccl init failed: {e!r}", file=sys.stderr)
            return 1

        if group.size() != 2:
            print(
                f"expected jaccl group size 2, got {group.size()}",
                file=sys.stderr,
            )
            return 1

        try:
            bytes_per_element = 2  # float16
            n_elements = max(1, payload_bytes // bytes_per_element)
            shape = (n_elements,)
            dtype = mx.float16
            tensor = mx.zeros(shape=shape, dtype=dtype)
            mx.eval(tensor)

            # Warm-up the distributed transport with one round-trip in each
            # direction so kernel-launch / connection-setup overhead doesn't
            # contaminate the first timed iteration.
            if rank == 0:
                mx.eval(mx.distributed.send(tensor, dst=1, group=group))
                mx.eval(
                    mx.distributed.recv(shape=shape, dtype=dtype, src=1, group=group)
                )
            else:
                mx.eval(
                    mx.distributed.recv(shape=shape, dtype=dtype, src=0, group=group)
                )
                mx.eval(mx.distributed.send(tensor, dst=0, group=group))

            # ---- Upload bandwidth: rank 0 -> rank 1 ----
            # `mx.eval` inside the loop because MLX is lazy: without it only
            # the final op actually executes, overestimating throughput by Nx.
            start = time.perf_counter()
            if rank == 0:
                for _ in range(iterations):
                    mx.eval(mx.distributed.send(tensor, dst=1, group=group))
            else:
                for _ in range(iterations):
                    mx.eval(
                        mx.distributed.recv(
                            shape=shape, dtype=dtype, src=0, group=group
                        )
                    )
            upload_elapsed = time.perf_counter() - start

            # ---- Download bandwidth: rank 1 -> rank 0 ----
            start = time.perf_counter()
            if rank == 0:
                for _ in range(iterations):
                    mx.eval(
                        mx.distributed.recv(
                            shape=shape, dtype=dtype, src=1, group=group
                        )
                    )
            else:
                for _ in range(iterations):
                    mx.eval(mx.distributed.send(tensor, dst=0, group=group))
            download_elapsed = time.perf_counter() - start

            total_bits = payload_bytes * 8 * iterations
            upload_mbps = (
                total_bits / upload_elapsed / 1e6 if upload_elapsed > 0 else 0.0
            )
            download_mbps = (
                total_bits / download_elapsed / 1e6
                if download_elapsed > 0
                else 0.0
            )

            # ---- Latency: tiny-payload all_sum is a ping-pong on a 2-rank
            # group. Time each iter individually so we can report a median
            # (resistant to a stray scheduler hiccup at loop start).
            n_lat_elements = max(1, LATENCY_PAYLOAD_BYTES // bytes_per_element)
            lat_tensor = mx.zeros(shape=(n_lat_elements,), dtype=dtype)
            mx.eval(lat_tensor)
            for _ in range(WARMUP_ITERATIONS):
                mx.eval(mx.distributed.all_sum(lat_tensor, group=group))

            per_iter_ms: list[float] = []
            for _ in range(LATENCY_ITERATIONS):
                t0 = time.perf_counter()
                mx.eval(mx.distributed.all_sum(lat_tensor, group=group))
                per_iter_ms.append((time.perf_counter() - t0) * 1000.0)
            latency_ms = (
                statistics.median(per_iter_ms) if per_iter_ms else None
            )
        except Exception as e:  # noqa: BLE001
            # Without this catch the rank-0 process can exit with returncode 0
            # (because the parent `try/finally` only protects file cleanup) but
            # an empty stdout — the caller would see "stdout unparseable" with
            # no clue what actually went wrong. Surface the exception on stderr.
            print(
                f"rdma probe op failed (rank={rank}): {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 1

        # Both ranks print the result. The sink-side HTTP handler also calls
        # `_spawn_probe_subprocess` and parses stdout to confirm the probe
        # completed cleanly — without rank-1 emitting JSON the handler 500s
        # back to the source even on success. Both ranks measured the same
        # loops, so the numbers should be near-identical.
        print(
            json.dumps(
                {
                    "upload_mbps": upload_mbps,
                    "download_mbps": download_mbps,
                    "payload_bytes": payload_bytes,
                    "iterations": iterations,
                    "latency_ms": latency_ms,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        Path(ibv_devs_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
