"""RDMA probe orchestration shared between source and sink.

The actual `mx.distributed.init(backend="jaccl")` runs in a child process
(`rdma_probe_main`) so it does not contaminate the worker process — `init`
is a one‑shot global that conflicts with active inference groups.

This module owns:

- `RdmaProbeParams`  — request body for both ends
- `RdmaProbeResult`  — what the source side parses out of the subprocess stdout
- `_rdma_probe_lock` — process‑global lock; ensures we never have two probes
                       in flight on the same node, regardless of whether the
                       reconciler kicked one off or a peer's HTTP request did.
- `run_rdma_probe_source_side`  — issued by `RDMALinkProfile.measure()`
- `handle_rdma_probe_sink_request` — invoked by the `/profile/rdma_probe` API
"""

import sys
from typing import final

import anyio
import httpx
from anyio import fail_after
from loguru import logger

from exo.utils.ports import random_ephemeral_port
from exo.utils.pydantic_ext import FrozenModel

PAYLOAD_BYTES = 64 * 1024 * 1024
ITERATIONS = 4
SUBPROCESS_TIMEOUT_SECONDS = 60.0
SINK_HTTP_CONNECT_TIMEOUT_SECONDS = 10.0


# Single-flight lock for any RDMA probe activity on this node, sourced or
# sinked. Module-global because both the reconciler (Worker) and the FastAPI
# handler (API) must respect the same exclusion zone, and they live in the same
# process.
_rdma_probe_lock: anyio.Lock = anyio.Lock()


@final
class RdmaProbeParams(FrozenModel):
    """Body sent over the wire from source to sink before the probe runs."""

    source_rdma_iface: str
    sink_rdma_iface: str
    coordinator_ip: str
    coordinator_port: int = 0  # 0 = source picks an ephemeral port
    payload_bytes: int = PAYLOAD_BYTES
    iterations: int = ITERATIONS


@final
class RdmaProbeResult(FrozenModel):
    # Per-direction bandwidth from rank 0's (source) perspective. Upload =
    # source -> sink, download = sink -> source. Apple Silicon TB5 is
    # symmetric in spec (~80 Gb/s each way) but the controller's tx/rx
    # pipelines can drift, so we measure each independently with send/recv.
    upload_mbps: float
    download_mbps: float
    payload_bytes: int
    iterations: int
    # Round-trip latency over the same edge, measured with a tiny-payload
    # all_sum ping-pong. None when the latency loop was skipped or failed.
    latency_ms: float | None = None


class RdmaProbeError(Exception):
    """Raised when an RDMA probe could not be initiated cleanly.

    This is distinct from "the probe ran and failed" — for that we just return
    None. RdmaProbeError signals a precondition violation (busy, lock held,
    invalid params) that the caller surfaces as 409.
    """


class RdmaProbeBusyError(RdmaProbeError):
    pass


def is_node_busy(state_runners: object) -> bool:
    """True if there are active runners on this node.

    Accepts the runners mapping by abstract type to avoid an import cycle with
    `exo.shared.types.state`.
    """
    try:
        return bool(len(state_runners))  # pyright: ignore[reportArgumentType]
    except TypeError:
        return False


async def run_rdma_probe_source_side(
    *,
    client: httpx.AsyncClient,
    params: RdmaProbeParams,
    sink_ip: str,
    api_port: int,
) -> RdmaProbeResult | None:
    """Coordinate an RDMA probe with the peer at `sink_ip` and return the result.

    Both ranks must run *concurrently* — jaccl init blocks each side until the
    other rendezvous over the coordinator socket. So we kick off our own
    rank‑0 subprocess and the peer's rank‑1 subprocess in parallel, with the
    coordinator bound on the source. If the peer refuses (busy / error), we
    cancel the local subprocess to avoid waiting on a rendezvous that will
    never happen.

    Returns None when the probe was skipped (peer busy) or failed (timeout,
    subprocess crash, parse error). Raises RdmaProbeBusyError if the local lock is
    already held — the reconciler treats that as "try again next tick".
    """
    if _rdma_probe_lock.locked():
        raise RdmaProbeBusyError("rdma probe already in flight on this node")

    coordinator_port = params.coordinator_port or random_ephemeral_port()
    sink_params = params.model_copy(update={"coordinator_port": coordinator_port})
    result_holder: list[RdmaProbeResult | None] = [None]

    async def _run_source() -> None:
        result_holder[0] = await _spawn_probe_subprocess(rank=0, params=sink_params)

    async def _ask_sink(cancel_scope: anyio.CancelScope) -> None:
        try:
            response = await client.post(
                _rdma_probe_url(sink_ip, api_port),
                content=sink_params.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as e:
            logger.debug(f"RDMA probe sink request failed: {e}")
            cancel_scope.cancel()
            return
        if response.status_code != 200:
            logger.debug(
                f"RDMA probe sink returned {response.status_code}: "
                f"{response.text[:200]}"
            )
            cancel_scope.cancel()

    async with _rdma_probe_lock, anyio.create_task_group() as tg:
        tg.start_soon(_run_source)
        tg.start_soon(_ask_sink, tg.cancel_scope)

    return result_holder[0]


async def handle_rdma_probe_sink_request(
    *, params: RdmaProbeParams, runners: object
) -> RdmaProbeResult:
    """Run the sink side of an RDMA probe in response to an HTTP request.

    Raises RdmaProbeBusyError if a probe is already in flight or if runners are
    active. The caller (FastAPI handler) translates that to 409.
    """
    if is_node_busy(runners):
        raise RdmaProbeBusyError("node has active runners")
    if _rdma_probe_lock.locked():
        raise RdmaProbeBusyError("rdma probe already in flight on this node")

    async with _rdma_probe_lock:
        result = await _spawn_probe_subprocess(rank=1, params=params)
        if result is None:
            raise RdmaProbeError("rdma probe subprocess produced no result")
        return result


async def _spawn_probe_subprocess(
    *, rank: int, params: RdmaProbeParams
) -> RdmaProbeResult | None:
    """Run rdma_probe_main with the given rank and parameters.

    Returns the parsed result on a clean exit. Returns None on any subprocess
    failure (timeout, non-zero exit, malformed output) — callers treat None
    as "try again on the next reconciler tick".
    """
    cmd = [
        sys.executable,
        "-m",
        "exo.utils.profilers.rdma_probe_main",
        str(rank),
        params.model_dump_json(),
    ]
    try:
        with fail_after(SUBPROCESS_TIMEOUT_SECONDS):
            completed = await anyio.run_process(cmd, check=False)
    except TimeoutError:
        logger.warning(f"RDMA probe subprocess (rank={rank}) timed out")
        return None
    except Exception as e:
        logger.opt(exception=e).warning(
            f"RDMA probe subprocess (rank={rank}) failed to launch"
        )
        return None

    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="replace")[:512]
        logger.warning(
            f"RDMA probe subprocess (rank={rank}) exited "
            f"{completed.returncode}: {stderr_text}"
        )
        return None

    stdout_text = completed.stdout.decode("utf-8", errors="replace").strip()
    last_line = stdout_text.splitlines()[-1] if stdout_text else ""
    try:
        return RdmaProbeResult.model_validate_json(last_line)
    except ValueError as e:
        # If stdout was empty there's almost always something illuminating in
        # stderr (jaccl init failure, bad iface name, etc.). Log it so the
        # operator can debug without re-running the subprocess by hand.
        stderr_text = completed.stderr.decode("utf-8", errors="replace")[:1024]
        logger.warning(
            f"RDMA probe (rank={rank}) stdout unparseable: {e}; "
            f"stdout={last_line!r}; stderr={stderr_text!r}"
        )
        return None


def _rdma_probe_url(sink_ip: str, api_port: int) -> str:
    bracketed = f"[{sink_ip}]" if ":" in sink_ip else sink_ip
    return f"http://{bracketed}:{api_port}/profile/rdma_probe"
