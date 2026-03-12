import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.platform_utils import is_uva_available as is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d as async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor as get_accelerator_view_from_cpu_tensor,
)

def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor: ...

class UvaBuffer:
    cpu: Incomplete
    np: Incomplete
    uva: Incomplete
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype) -> None: ...

class UvaBufferPool:
    size: Incomplete
    dtype: Incomplete
    max_concurrency: Incomplete
    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ) -> None: ...
    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor: ...
    def copy_to_gpu(
        self, x: torch.Tensor | np.ndarray, out: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class UvaBackedTensor:
    dtype: Incomplete
    cpu: Incomplete
    np: Incomplete
    pool: Incomplete
    gpu: Incomplete
    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ) -> None: ...
    def copy_to_uva(self, n: int | None = None) -> torch.Tensor: ...

class StagedWriteTensor:
    num_rows: Incomplete
    dtype: Incomplete
    device: Incomplete
    max_concurrency: Incomplete
    gpu: Incomplete
    write_indices: Incomplete
    write_starts: Incomplete
    write_cu_lens: Incomplete
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ) -> None: ...
    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None: ...
    def stage_write_elem(self, index: int, x: int) -> None: ...
    def apply_write(self) -> None: ...
    def clear_staged_writes(self) -> None: ...
