import asyncio
import contextlib
import sys
from logging import Logger

import psutil

from exo.shared.constants import LB_DISK_GBPS, LB_MEMBW_GBPS, LB_TFLOPS
from exo.shared.types.worker.shards import ShardMetadata


async def kill_process_tree(runner_process: asyncio.subprocess.Process, logger: Logger) -> None:
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

def get_weights_size_kb(model_shard_meta: ShardMetadata) -> float:
    return (model_shard_meta.end_layer - model_shard_meta.start_layer) / model_shard_meta.n_layers * model_shard_meta.model_meta.storage_size_kilobytes

def get_init_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size_kb = get_weights_size_kb(model_shard_meta)

    kbps_read = 1024 * 1024 * LB_DISK_GBPS / 3

    return weights_size_kb / kbps_read + 2.0

def get_prefil_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size_gb = get_weights_size_kb(model_shard_meta) / (1024 * 1024)
    
    tokens = 1000 # constant for now - the prompt is only tokenized in the device...
    prompt_gflops = tokens * weights_size_gb * 2

    return LB_TFLOPS / (1024 * prompt_gflops) * 3 + 10.0

def get_token_generate_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size_kb = get_weights_size_kb(model_shard_meta)

    kbps_read = 1024 * 1024 * LB_MEMBW_GBPS / 3

    return weights_size_kb / kbps_read + 2.0