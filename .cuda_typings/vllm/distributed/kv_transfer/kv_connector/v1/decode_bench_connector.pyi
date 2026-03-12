import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata as KVConnectorMetadata,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.request import Request as Request

logger: Incomplete

@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]

class DecodeBenchConnector(KVConnectorBase_V1):
    connector_scheduler: DecodeBenchConnectorScheduler | None
    connector_worker: DecodeBenchConnectorWorker | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
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

class DecodeBenchConnectorScheduler:
    vllm_config: Incomplete
    block_size: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def request_finished(self, request: Request): ...

class DecodeBenchConnectorWorker:
    vllm_config: Incomplete
    block_size: Incomplete
    fill_mean: Incomplete
    fill_std: Incomplete
    kv_caches: dict[str, torch.Tensor] | None
    group_to_layers: dict[int, list[str]] | None
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata): ...
