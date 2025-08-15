import sys

from shared.constants import LB_DISK_GBPS, LB_MEMBW_GBPS
from shared.types.tasks import Task
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

def get_prefil_timeout(task: Task, model_shard_meta: ShardMetadata) -> float:    
    def get_prompt_str(task: Task) -> str:
        messages = [x.content for x in task.task_params.messages if x.content]
        return ''.join(messages)

    # TODO: made this timeout very long
    tokens = len(get_prompt_str(task)) // 3 + 3000 # constant for now - the prompt is only tokenized in the device...

    # TODO: For now we just hack and assume we prefil at 10tok/s
    return tokens * 0.1

    # prompt_gflops = tokens * weights_size_gb * 2

    # return LB_TFLOPS / (1024 * prompt_gflops) * 3 + 10.0

def get_token_generate_timeout(model_shard_meta: ShardMetadata) -> float:
    weights_size_kb = get_weights_size_kb(model_shard_meta)

    kbps_read = 1024 * 1024 * LB_MEMBW_GBPS / 3

    return weights_size_kb / kbps_read + 2.0