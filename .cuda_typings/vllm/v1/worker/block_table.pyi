import numpy as np
import torch
from _typeshed import Incomplete
from vllm.distributed import (
    get_dcp_group as get_dcp_group,
    get_pcp_group as get_pcp_group,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.utils import CpuGpuBuffer as CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size as get_total_cp_world_size

logger: Incomplete

class BlockTable:
    max_num_reqs: Incomplete
    max_num_batched_tokens: Incomplete
    pin_memory: Incomplete
    device: Incomplete
    block_size: Incomplete
    blocks_per_kv_block: int
    use_hybrid_blocks: bool
    max_num_blocks_per_req: Incomplete
    block_table: Incomplete
    num_blocks_per_row: Incomplete
    slot_mapping: Incomplete
    pcp_world_size: Incomplete
    pcp_rank: Incomplete
    dcp_world_size: Incomplete
    dcp_rank: Incomplete
    cp_kv_cache_interleave_size: Incomplete
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ) -> None: ...
    def append_row(self, block_ids: list[int], row_idx: int) -> None: ...
    def add_row(self, block_ids: list[int], row_idx: int) -> None: ...
    def move_row(self, src: int, tgt: int) -> None: ...
    def swap_row(self, src: int, tgt: int) -> None: ...
    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None: ...
    def commit_block_table(self, num_reqs: int) -> None: ...
    def commit_slot_mapping(self, num_tokens: int) -> None: ...
    def clear(self) -> None: ...
    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray: ...
    def get_device_tensor(self, num_reqs: int) -> torch.Tensor: ...
    def get_cpu_tensor(self) -> torch.Tensor: ...
    def get_numpy_array(self) -> np.ndarray: ...

class MultiGroupBlockTable:
    block_tables: Incomplete
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None: ...
    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None: ...
    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None: ...
    def move_row(self, src: int, tgt: int) -> None: ...
    def swap_row(self, src: int, tgt: int) -> None: ...
    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None: ...
    def commit_block_table(self, num_reqs: int) -> None: ...
    def commit_slot_mapping(self, num_tokens: int) -> None: ...
    def clear(self) -> None: ...
    def __getitem__(self, idx: int) -> BlockTable: ...
