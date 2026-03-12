from _typeshed import Incomplete
from collections.abc import Set as AbstractSet
from vllm.config import CUDAGraphMode as CUDAGraphMode, VllmConfig as VllmConfig
from vllm.forward_context import BatchDescriptor as BatchDescriptor
from vllm.logger import init_logger as init_logger
from vllm.lora.utils import get_captured_lora_counts as get_captured_lora_counts

logger: Incomplete

class CudagraphDispatcher:
    vllm_config: Incomplete
    compilation_config: Incomplete
    uniform_decode_query_len: Incomplete
    cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]]
    keys_initialized: bool
    specialize_lora_count: Incomplete
    cudagraph_mode: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ): ...
    captured_lora_counts: Incomplete
    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ): ...
    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]: ...
    def get_capture_descs(
        self,
    ) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]: ...
