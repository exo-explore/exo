import sys

from shared.constants import LB_DISK_GBPS, LB_MEMBW_GBPS, LB_TFLOPS
from shared.types.worker.shards import ShardMetadata


def get_runner_command() -> list[str]:
    python = sys.executable
    return [python, "-m", "worker.runner.runner"]

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