import abc
import torch
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    MixtureOfExperts as MixtureOfExperts,
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .qwen3_next import (
    Qwen3NextAttention as Qwen3NextAttention,
    Qwen3NextDecoderLayer as Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet as Qwen3NextGatedDeltaNet,
    Qwen3NextModel as Qwen3NextModel,
    Qwen3NextSparseMoeBlock as Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts as QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder as Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration as Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor as Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo as Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer as Qwen3_VisionTransformer,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config as Qwen3_5Config,
    Qwen3_5TextConfig as Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig as Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig as Qwen3_5MoeTextConfig,
)

logger: Incomplete

class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self): ...

class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self): ...

class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ): ...
    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear: ...
    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear: ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor): ...

class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    layer_type: Incomplete
    layer_idx: Incomplete
    linear_attn: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    layer_scale: Incomplete
    attn_layer_scale: Incomplete
    ffn_layer_scale: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, layer_type: str, prefix: str = ""
    ) -> None: ...

class Qwen3_5Model(Qwen3NextModel):
    num_redundant_experts: Incomplete
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    make_empty_intermediate_tensors: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3_5ForCausalLMBase(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    vllm_config: Incomplete
    model_config: Incomplete
    quant_config: Incomplete
    config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase, metaclass=abc.ABCMeta): ...

class Qwen3_5MoeForCausalLM(
    Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts, metaclass=abc.ABCMeta
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

class Qwen3_5ForConditionalGeneration(
    Qwen3VLForConditionalGeneration, IsHybrid, metaclass=abc.ABCMeta
):
    supports_multimodal_pruning: bool
    packed_modules_mapping: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    is_multimodal_pruning_enabled: bool
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def recompute_mrope_positions(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]: ...

class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_redundant_experts: Incomplete
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    expert_weights: Incomplete
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: int
    num_shared_experts: int
    num_logical_experts: Incomplete
    num_routed_experts: Incomplete
    def set_moe_parameters(self) -> None: ...

class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts, metaclass=abc.ABCMeta
):
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    is_multimodal_pruning_enabled: bool
    visual: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
