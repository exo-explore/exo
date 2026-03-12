from _typeshed import Incomplete
from collections.abc import Mapping
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonInputs as SingletonInputs,
)
from vllm.inputs.parse import split_enc_dec_inputs as split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor as InputPreprocessor
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalRegistry as MultiModalRegistry,
)
from vllm.multimodal.encoder_budget import MultiModalBudget as MultiModalBudget
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions as argsort_mm_positions
from vllm.platforms import current_platform as current_platform
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import (
    BaseRenderer as BaseRenderer,
    renderer_from_config as renderer_from_config,
)
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.tasks import (
    GENERATION_TASKS as GENERATION_TASKS,
    POOLING_TASKS as POOLING_TASKS,
    SupportedTask as SupportedTask,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
    random_uuid as random_uuid,
)
from vllm.utils.jsontree import json_iter_leaves as json_iter_leaves
from vllm.v1.engine import EngineCoreRequest as EngineCoreRequest

logger: Incomplete

class InputProcessor:
    vllm_config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    lora_config: Incomplete
    scheduler_config: Incomplete
    speculative_config: Incomplete
    structured_outputs_config: Incomplete
    observability_config: Incomplete
    generation_config_fields: Incomplete
    renderer: Incomplete
    supports_mm_inputs: Incomplete
    mm_encoder_cache_size: int
    skip_prompt_length_check: bool
    input_preprocessor: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = ...,
    ) -> None: ...
    @property
    def tokenizer(self) -> TokenizerLike | None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    @staticmethod
    def assign_request_id(request: EngineCoreRequest): ...
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest: ...
