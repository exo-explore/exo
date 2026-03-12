import torch
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group as get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor as BatchExecutionDescriptor,
    CudaGraphManager as CudaGraphManager,
)

def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None: ...
def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]: ...
