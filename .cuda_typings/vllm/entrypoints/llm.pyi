import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any
from typing_extensions import overload
from vllm.beam_search import (
    BeamSearchInstance as BeamSearchInstance,
    BeamSearchOutput as BeamSearchOutput,
    BeamSearchSequence as BeamSearchSequence,
    create_sort_beams_key_function as create_sort_beams_key_function,
)
from vllm.config import (
    AttentionConfig as AttentionConfig,
    CompilationConfig as CompilationConfig,
    PoolerConfig as PoolerConfig,
    ProfilerConfig as ProfilerConfig,
    StructuredOutputsConfig as StructuredOutputsConfig,
    is_init_field as is_init_field,
)
from vllm.config.compilation import CompilationMode as CompilationMode
from vllm.config.model import (
    ConvertOption as ConvertOption,
    HfOverrides as HfOverrides,
    ModelDType as ModelDType,
    RunnerOption as RunnerOption,
    TokenizerMode as TokenizerMode,
)
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest as WeightTransferInitRequest,
    WeightTransferUpdateRequest as WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateConfig as ChatTemplateConfig,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    load_chat_template as load_chat_template,
)
from vllm.entrypoints.pooling.io_processor_factories import (
    init_pooling_io_processors as init_pooling_io_processors,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreData as ScoreData,
    ScoreMultiModalParam as ScoreMultiModalParam,
    compress_token_type_ids as compress_token_type_ids,
    compute_maxsim_score as compute_maxsim_score,
    get_score_prompt as get_score_prompt,
    score_data_to_prompts as score_data_to_prompts,
    validate_score_input as validate_score_input,
)
from vllm.entrypoints.utils import log_non_default_args as log_non_default_args
from vllm.inputs.data import (
    DataPrompt as DataPrompt,
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonPrompt as SingletonPrompt,
    TextPrompt as TextPrompt,
    TokensPrompt as TokensPrompt,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.outputs import (
    ClassificationRequestOutput as ClassificationRequestOutput,
    EmbeddingRequestOutput as EmbeddingRequestOutput,
    PoolingRequestOutput as PoolingRequestOutput,
    RequestOutput as RequestOutput,
    ScoringRequestOutput as ScoringRequestOutput,
)
from vllm.platforms import current_platform as current_platform
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import ChatParams as ChatParams, merge_kwargs as merge_kwargs
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq as conversation_to_seq,
    parse_model_prompt as parse_model_prompt,
    prompt_to_seq as prompt_to_seq,
)
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
)
from vllm.tasks import PoolingTask as PoolingTask
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils.counter import Counter as Counter
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer
from vllm.utils.tqdm_utils import maybe_tqdm as maybe_tqdm
from vllm.v1.engine import PauseMode as PauseMode
from vllm.v1.engine.llm_engine import LLMEngine as LLMEngine
from vllm.v1.metrics.reader import Metric as Metric
from vllm.v1.sample.logits_processor import LogitsProcessor as LogitsProcessor

logger: Incomplete

class LLM:
    llm_engine: Incomplete
    engine_class: Incomplete
    request_counter: Incomplete
    default_sampling_params: dict[str, Any] | None
    supported_tasks: Incomplete
    model_config: Incomplete
    renderer: Incomplete
    chat_template: Incomplete
    io_processor: Incomplete
    input_processor: Incomplete
    chat_template_config: Incomplete
    pooling_io_processors: Incomplete
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    def get_world_size(self, include_dp: bool = True) -> int: ...
    def reset_mm_cache(self) -> None: ...
    def get_default_sampling_params(self) -> SamplingParams: ...
    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]: ...
    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]: ...
    @overload
    def wait_for_completion(
        self, *, use_tqdm: bool | Callable[..., tqdm] = True
    ) -> list[RequestOutput | PoolingRequestOutput]: ...
    @overload
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...
    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]: ...
    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]: ...
    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list[BeamSearchOutput]: ...
    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]: ...
    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        pooling_task: PoolingTask | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]: ...
    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[EmbeddingRequestOutput]: ...
    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[ClassificationRequestOutput]: ...
    def reward(
        self,
        prompts: PromptType | Sequence[PromptType],
        /,
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]: ...
    def score(
        self,
        data_1: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        data_2: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        chat_template: str | None = None,
    ) -> list[ScoringRequestOutput]: ...
    def start_profile(self, profile_prefix: str | None = None) -> None: ...
    def stop_profile(self) -> None: ...
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    def sleep(self, level: int = 1, mode: PauseMode = "abort"): ...
    def wake_up(self, tags: list[str] | None = None): ...
    def get_metrics(self) -> list["Metric"]: ...
    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None: ...
    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None: ...
