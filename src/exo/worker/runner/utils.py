import asyncio
import contextlib
import sys

import psutil
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


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
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )
