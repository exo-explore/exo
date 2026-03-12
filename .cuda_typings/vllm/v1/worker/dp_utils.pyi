import numpy as np
import torch
from _typeshed import Incomplete
from vllm.config import ParallelConfig as ParallelConfig
from vllm.distributed.parallel_state import get_dp_group as get_dp_group
from vllm.logger import init_logger as init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds as check_ubatch_thresholds,
    is_last_ubatch_empty as is_last_ubatch_empty,
)

logger: Incomplete

def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    num_scheduled_tokens_per_request: np.ndarray | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]: ...
