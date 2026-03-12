import torch
from .utils import tensor_cache as tensor_cache
from vllm.triton_utils import triton as triton

@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor: ...
@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor: ...
@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor: ...
