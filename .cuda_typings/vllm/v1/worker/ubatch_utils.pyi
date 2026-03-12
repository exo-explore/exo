import numpy as np
import torch
from dataclasses import dataclass
from typing import TypeAlias
from vllm.config import ParallelConfig as ParallelConfig
from vllm.v1.attention.backend import CommonAttentionMetadata as CommonAttentionMetadata

@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice
    def is_empty(self) -> bool: ...
    @property
    def num_tokens(self) -> int: ...

UBatchSlices: TypeAlias = list[UBatchSlice]

def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool: ...
def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool: ...
def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]: ...
def slice_query_start_locs(
    query_start_loc: torch.Tensor, request_slice: slice
) -> torch.Tensor: ...
def split_attn_metadata(
    ubatch_slices: list[UBatchSlice], common_attn_metadata: CommonAttentionMetadata
) -> list[CommonAttentionMetadata]: ...
