import torch
import types
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass
from typing import Any
from vllm.compilation.cuda_graph import CUDAGraphWrapper as CUDAGraphWrapper
from vllm.config import CUDAGraphMode as CUDAGraphMode, VllmConfig as VllmConfig
from vllm.distributed import get_ep_group as get_ep_group
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id as set_graph_pool_id,
)
from vllm.forward_context import (
    DPMetadata as DPMetadata,
    create_forward_context as create_forward_context,
    get_forward_context as get_forward_context,
    override_forward_context as override_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.offloader.base import get_offloader as get_offloader
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.import_utils import has_deep_gemm as has_deep_gemm
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.v1.worker.ubatching import (
    UBatchContext as UBatchContext,
    make_ubatch_contexts as make_ubatch_contexts,
)

logger: Incomplete

@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int

@dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = ...

class SMControlContextManager:
    total_sms: Incomplete
    compute_sms: Incomplete
    comm_sms: Incomplete
    set_comm_sms: Incomplete
    set_compute_sms: Incomplete
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

class UBatchWrapper:
    runnable: Incomplete
    vllm_config: Incomplete
    compilation_config: Incomplete
    comm_stream: Incomplete
    ready_barrier: Incomplete
    cudagraphs: dict[int, CUDAGraphMetaData]
    cudagraph_wrapper: Incomplete
    sm_control: Incomplete
    device: Incomplete
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ) -> None: ...
    @property
    def graph_pool(self): ...
    def clear_graphs(self) -> None: ...
    def __getattr__(self, key: str): ...
    def unwrap(self) -> Callable: ...
    def __call__(self, *args, **kwargs): ...
