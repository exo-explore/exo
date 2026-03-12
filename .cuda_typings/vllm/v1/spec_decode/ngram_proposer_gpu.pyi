import torch
from _typeshed import Incomplete
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    CompilationConfig as CompilationConfig,
    CompilationMode as CompilationMode,
    VllmConfig as VllmConfig,
)
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.utils import (
    record_function_or_nullcontext as record_function_or_nullcontext,
)
from vllm.v1.worker.gpu_input_batch import (
    CachedRequestState as CachedRequestState,
    InputBatch as InputBatch,
)

class NgramGPUKernel(nn.Module):
    min_n: Incomplete
    max_n: Incomplete
    k: Incomplete
    max_model_len: Incomplete
    max_num_seqs: Incomplete
    device: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", device: torch.device = "cuda"
    ) -> None: ...
    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def load_model(self, *args, **kwargs) -> None: ...

class NgramProposerGPU:
    vllm_config: Incomplete
    min_n: Incomplete
    max_n: Incomplete
    k: Incomplete
    max_model_len: Incomplete
    max_num_seqs: Incomplete
    device: Incomplete
    kernel: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner=None
    ) -> None: ...
    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        valid_sampled_token_ids_gpu: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def load_model(self, *args, **kwargs) -> None: ...

def update_scheduler_for_invalid_drafts(
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens_cpu: torch.Tensor,
    scheduler_output: SchedulerOutput,
    req_id_to_index: dict[str, int],
) -> None: ...
def update_ngram_gpu_tensors_incremental(
    input_batch: InputBatch,
    token_ids_gpu_tensor: torch.Tensor,
    num_tokens_no_spec_gpu: torch.Tensor,
    new_reqs: list[CachedRequestState],
    device: torch.device,
    _pinned_idx_buf: torch.Tensor,
    _pinned_val_buf: torch.Tensor,
) -> None: ...
def copy_num_valid_draft_tokens(
    num_valid_draft_tokens_cpu: torch.Tensor,
    num_valid_draft_tokens_copy_stream: torch.cuda.Stream,
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens: torch.Tensor | None,
    batch_size: int,
) -> None: ...
