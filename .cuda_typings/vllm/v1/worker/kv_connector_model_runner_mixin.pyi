import torch
from _typeshed import Incomplete
from contextlib import AbstractContextManager
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown as ensure_kv_transfer_shutdown,
    get_kv_transfer_group as get_kv_transfer_group,
    has_kv_transfer_group as has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase as KVConnectorBase,
)
from vllm.forward_context import (
    get_forward_context as get_forward_context,
    set_forward_context as set_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    KVCacheConfig as KVCacheConfig,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT as EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput as KVConnectorOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

logger: Incomplete

class KVConnectorModelRunnerMixin:
    @staticmethod
    def ensure_kv_transfer_shutdown() -> None: ...
    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: SchedulerOutput, vllm_config: VllmConfig
    ) -> ModelRunnerOutput: ...
    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: SchedulerOutput, defer_finalize: bool = False
    ) -> AbstractContextManager[KVConnectorOutput | None]: ...
    @staticmethod
    def finalize_kv_connector() -> None: ...
    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]], cache_dtype: CacheDType
    ) -> bool: ...
    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]: ...
