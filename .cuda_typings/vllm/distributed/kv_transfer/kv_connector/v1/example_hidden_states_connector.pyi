import torch
from _typeshed import Incomplete
from dataclasses import dataclass, field
from typing import Any
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import (
    NewRequestData as NewRequestData,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.request import Request as Request

logger: Incomplete

def extract_from_kv_cache(
    kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_tokens: int
) -> torch.Tensor: ...
@dataclass
class ReqMeta:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    new_req: bool
    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool,
    ) -> ReqMeta: ...

@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool = True,
    ) -> None: ...

class ExampleHiddenStatesConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool: ...
    cache_layers: list[str]
    num_hidden_states: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    def start_load_kv(self, *args, **kwargs: Any) -> None: ...
    def wait_for_layer_load(self, layer_name: str) -> None: ...
    def wait_for_save(self) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None: ...
