import contextlib
import numpy.typing as npt
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Collection, Generator
from torch._opaque_base import OpaqueBase
from torch.library import Library
from typing import Any, TypeVar
from vllm.config import ModelConfig as ModelConfig
from vllm.logger import init_logger as init_logger
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete
STR_DTYPE_TO_TORCH_DTYPE: Incomplete
TORCH_DTYPE_TO_NUMPY_DTYPE: Incomplete
MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP: Incomplete
T = TypeVar("T")

def is_strictly_contiguous(t: torch.Tensor) -> bool: ...
@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype): ...
@contextlib.contextmanager
def set_default_torch_num_threads(num_threads: int | None = None): ...
@contextlib.contextmanager
def guard_cuda_initialization() -> Generator[None]: ...
def get_dtype_size(dtype: torch.dtype) -> int: ...
def is_lossless_cast(src_dtype: torch.dtype, tgt_dtype: torch.dtype): ...
def common_broadcastable_dtype(dtypes: Collection[torch.dtype]): ...
def get_kv_cache_torch_dtype(
    cache_dtype: str | torch.dtype | None, model_dtype: str | torch.dtype | None = None
) -> torch.dtype: ...
def get_kv_cache_quant_algo_string(quant_cfg: dict[str, Any]) -> str | None: ...
def get_kv_cache_quant_algo_dtype(quant_cfg: dict[str, Any]) -> torch.dtype | None: ...
def resolve_kv_cache_dtype_string(
    kv_cache_dtype: str, model_config: ModelConfig
) -> str: ...
def kv_cache_dtype_str_to_dtype(
    kv_cache_dtype: str, model_config: ModelConfig
) -> torch.dtype: ...
def set_random_seed(seed: int | None) -> None: ...
def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
    cache_layout: str | None = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...
def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...
def async_tensor_h2d(
    data: list, dtype: torch.dtype, target_device: str | torch.device, pin_memory: bool
) -> torch.Tensor: ...
def make_ndarray_with_pad(
    x: list[list[T]], pad: T, dtype: npt.DTypeLike, *, max_len: int | None = None
) -> npt.NDArray: ...
def make_tensor_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor: ...

prev_set_stream: Incomplete

class _StreamPlaceholder:
    synchronize: Incomplete
    def __init__(self) -> None: ...

def current_stream() -> torch.cuda.Stream: ...
def aux_stream() -> torch.cuda.Stream | None: ...
def cuda_device_count_stateless() -> int: ...
def weak_ref_tensor(tensor: Any) -> Any: ...
def weak_ref_tensors(
    tensors: torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
    | IntermediateTensors,
) -> torch.Tensor | list[Any] | tuple[Any] | Any: ...
def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor: ...
def is_torch_equal_or_newer(target: str) -> bool: ...
def is_torch_equal(target: str) -> bool: ...

HAS_OPAQUE_TYPE: Incomplete
OpaqueBase = object

class ModuleName(OpaqueBase):
    value: Incomplete
    def __init__(self, value: str) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __fx_repr__(self): ...

def supports_xccl() -> bool: ...
def supports_xpu_graph() -> bool: ...

vllm_lib: Incomplete

def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
): ...
