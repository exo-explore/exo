import asyncio
import contextlib
import sys

import psutil
from loguru import logger

from exo.shared.constants import LB_DISK_GBPS, LB_MEMBW_GBPS, LB_TFLOPS
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata


async def kill_process_tree(runner_process: asyncio.subprocess.Process) -> None:
    """Kill the process and all its children forcefully."""
    if runner_process.returncode is not None:
        return  # Process already dead

    try:
        # Get the main process
        pid = runner_process.pid

        # Find all child processes
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Kill all children first (bottom-up)
            for child in reversed(children):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.kill()  # SIGKILL

            # Kill the parent
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                parent.kill()  # SIGKILL

        except psutil.NoSuchProcess:
            # Process already gone, try subprocess kill anyway
            runner_process.kill()

        # Wait for the subprocess to exit
        try:
            await asyncio.wait_for(runner_process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.error(f"Process {pid} did not exit after kill signal")

    except Exception as e:
        logger.error(f"Error killing process tree: {e}")


def get_runner_command() -> list[str]:
    python = sys.executable
    return [python, "-m", "exo.worker.runner.runner"]


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
    )


def get_init_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size = get_weights_size(model_shard_meta)

    kbps_read = 1024 * 1024 * LB_DISK_GBPS / 3

    return weights_size.in_kb / kbps_read + 2.0


def _prefill_flops_for_shard(model_shard_meta: ShardMetadata, s: int) -> float:
    p = get_weights_size(model_shard_meta).in_bytes
    flops = 2.0 * p * s  # parameter-dependent GEMMs
    # flops += _attention_flops(meta, S)  # optional S^2 term
    return flops


def get_prefil_timeout(
    model_shard_meta: ShardMetadata,
    prompt_tokens: int,
    *,
    effective_tflops: float = LB_TFLOPS,
    safety_mult: float = 1.6,
    base_pad_s: float = 5.0,
) -> float:
    """
    Returns a conservative timeout (seconds) for the prefill stage.
    """
    total_flops = _prefill_flops_for_shard(model_shard_meta, prompt_tokens)

    # Convert to seconds using sustained throughput
    time_seconds = total_flops / (effective_tflops * 1e12)

    # Prefill across pipeline stages is largely sequential; summing FLOPs already accounts for it.
    # Add a base pad (launch/IO) and a safety multiplier for variance.
    return base_pad_s + safety_mult * time_seconds


def get_token_generate_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size = get_weights_size(model_shard_meta)

    kbps_read = 1024 * 1024 * LB_MEMBW_GBPS / 3

    return weights_size.in_kb / kbps_read + 2.0
