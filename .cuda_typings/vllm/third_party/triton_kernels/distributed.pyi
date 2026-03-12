import torch
import torch.distributed as dist
from _typeshed import Incomplete
from dataclasses import dataclass

@dataclass
class ExptAssignment:
    expt_bitmask: torch.Tensor
    expt_boolmask: torch.Tensor
    expt_map: torch.Tensor
    n_expts_per_shard: list[int]

@dataclass
class _MemoryRegion:
    base: int
    size: int
    alignment: int

class SymmetricMemoryPool:
    size: int
    buf: Incomplete
    bufs: Incomplete
    hdl: Incomplete
    regions: Incomplete
    def __init__(self) -> None: ...
    @staticmethod
    def align_up(value: int, alignment: int) -> int: ...
    def make_empty(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        region: str,
        region_offset: int = 0,
        clear: bool = False,
    ) -> tuple[torch.Tensor, ...]: ...
    def initialize_matmul_ogs(
        self,
        n_tokens_global: int,
        d_input: int,
        d_model: int,
        n_expts_act: int,
        n_expts_tot: int,
        dtype: torch.dtype,
        n_ranks: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None: ...

symm_mem_pool: Incomplete

def make_expt_dict_uniform(n_expt_shard, n_expt_tot): ...
def make_expt_dict_random(n_expt_shard, n_expt_tot): ...
def make_expt_assignment(
    n_expt_shard, n_expt_tot, expt_dict: dict[int, list[int]], device
) -> ExptAssignment: ...
def convert_dp_to_ep(src, expt_assignment, expt_indx, gate_indx): ...
def convert_ep_to_dp(src, expt_assignment, expt_indx, topk_indx): ...
