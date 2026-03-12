from _typeshed import Incomplete
from collections.abc import Mapping
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.multimodal.processing import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
)
from vllm.multimodal.registry import MultiModalRegistry as MultiModalRegistry
from vllm.utils.torch_utils import (
    set_default_torch_num_threads as set_default_torch_num_threads,
)
from vllm.v1.core.encoder_cache_manager import (
    compute_mm_encoder_budget as compute_mm_encoder_budget,
)

logger: Incomplete

def get_mm_max_toks_per_item(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    processor: BaseMultiModalProcessor,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]: ...

class MultiModalBudget:
    model_config: Incomplete
    scheduler_config: Incomplete
    max_model_len: Incomplete
    max_num_reqs: Incomplete
    cache: Incomplete
    processor: Incomplete
    mm_limits: Incomplete
    encoder_compute_budget: Incomplete
    encoder_cache_size: Incomplete
    mm_max_toks_per_item: Incomplete
    mm_max_items_per_prompt: Mapping[str, int]
    mm_max_items_per_batch: Mapping[str, int]
    def __init__(
        self, vllm_config: VllmConfig, mm_registry: MultiModalRegistry
    ) -> None: ...
    def get_modality_with_max_tokens(self) -> str: ...
    def get_encoder_budget(self) -> int: ...
    def reset_cache(self) -> None: ...
