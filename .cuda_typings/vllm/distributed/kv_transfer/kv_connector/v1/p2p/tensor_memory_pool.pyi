import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.logger import init_logger as init_logger

logger: Incomplete

@dataclass
class MemoryBlock:
    size: int
    addr: int

class TensorMemoryPool:
    max_block_size: Incomplete
    min_block_size: Incomplete
    free_lists: dict[int, dict[int, MemoryBlock]]
    allocated_blocks: dict[int, MemoryBlock]
    def __init__(self, max_block_size: int, min_block_size: int = 512) -> None: ...
    def allocate(self, size: int) -> int: ...
    def free(self, addr: int): ...
    def store_tensor(self, tensor: torch.Tensor) -> int: ...
    def load_tensor(
        self,
        addr: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor: ...
    def cleanup(self) -> None: ...
    def __del__(self) -> None: ...
