import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine as P2pNcclEngine,
)
from vllm.distributed.parallel_state import get_world_group as get_world_group
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata as MLACommonMetadata,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.request import Request as Request

logger: Incomplete

@dataclass
class ReqMeta:
    request_id: str
    block_ids: torch.Tensor
    num_tokens: int
    @staticmethod
    def make_meta(
        request_id: str, token_ids: list[int], block_ids: list[int], block_size: int
    ) -> ReqMeta: ...

@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]
    def __init__(self) -> None: ...
    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None: ...

class P2pNcclConnector(KVConnectorBase_V1):
    is_producer: Incomplete
    chunked_prefill: dict[str, tuple[list[int], list[int] | None]]
    p2p_nccl_engine: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None: ...
    def wait_for_layer_load(self, layer_name: str) -> None: ...
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None: ...
    def wait_for_save(self) -> None: ...
    def get_finished(
        self, finished_req_ids: set[str], **kwargs: Any
    ) -> tuple[set[str] | None, set[str] | None]: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    @staticmethod
    def parse_request_id(
        request_id: str, is_prefill: bool = True
    ) -> tuple[str, int]: ...
    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim) -> None: ...
