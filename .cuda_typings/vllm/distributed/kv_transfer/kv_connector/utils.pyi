import torch
from _typeshed import Incomplete
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase as KVConnectorBase,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory as KVConnectorFactory,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.outputs import (
    KVConnectorOutput as KVConnectorOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)

logger: Incomplete
EngineId = str
BlockIds = tuple[list[int], ...] | list[list[int]]

def get_kv_connector_cache_layout(): ...

class KVOutputAggregator:
    def __init__(self, expected_finished_count: int) -> None: ...
    @classmethod
    def from_connector(cls, connector: KVConnectorBase, world_size: int): ...
    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None: ...

def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None: ...
def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio) -> None: ...
def kv_postprocess_layout_on_receive(cache, indices) -> None: ...
def kv_postprocess_blksize_and_layout_on_receive(
    cache, indices, block_size_ratio
) -> None: ...
def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...], bool]]: ...
@dataclass
class TpKVTopology:
    tp_rank: int
    remote_tp_size: dict[EngineId, int]
    is_mla: bool
    total_num_kv_heads: int
    attn_backend: type[AttentionBackend]
    engine_id: EngineId
    remote_block_size: dict[EngineId, int]
    tensor_shape: torch.Size | None = ...
    def __post_init__(self) -> None: ...
    @property
    def is_kv_layout_blocks_first(self) -> bool: ...
    @property
    def split_k_and_v(self) -> bool: ...
    @property
    def tp_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def cross_layers_blocks(self) -> bool: ...
    def tp_ratio(self, remote_tp_size: int) -> int: ...
    def block_size_ratio(self, remote_block_size: int) -> int: ...
    def tp_ratio_from_engine_id(self, remote_engine_id: EngineId) -> int: ...
    def block_size_ratio_from_engine_id(self, remote_engine_id: EngineId) -> int: ...
    def is_kv_replicated(self, engine_id: EngineId) -> bool: ...
    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool: ...
    def get_target_remote_ranks(self, remote_tp_size: int) -> list[int]: ...
    def get_target_remote_ranks_from_engine_id(
        self, remote_engine_id: EngineId
    ) -> list[int]: ...

def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]: ...
def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]: ...
