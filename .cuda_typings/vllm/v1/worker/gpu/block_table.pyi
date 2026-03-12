import torch
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID as PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as StagedWriteTensor,
    UvaBackedTensor as UvaBackedTensor,
)

class BlockTables:
    block_sizes: Incomplete
    max_num_reqs: Incomplete
    max_num_batched_tokens: Incomplete
    max_model_len: Incomplete
    device: Incomplete
    cp_size: Incomplete
    cp_rank: Incomplete
    cp_interleave: Incomplete
    num_kv_cache_groups: Incomplete
    block_tables: list[StagedWriteTensor]
    block_table_ptrs: Incomplete
    block_table_strides: Incomplete
    block_sizes_tensor: Incomplete
    num_blocks: Incomplete
    input_block_tables: list[torch.Tensor]
    input_block_table_ptrs: Incomplete
    slot_mappings: Incomplete
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None: ...
    def append_block_ids(
        self, req_index: int, new_block_ids: tuple[list[int], ...], overwrite: bool
    ) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def gather_block_tables(
        self, idx_mapping: torch.Tensor, num_reqs_padded: int
    ) -> tuple[torch.Tensor, ...]: ...
    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]: ...
    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor: ...
    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor: ...
