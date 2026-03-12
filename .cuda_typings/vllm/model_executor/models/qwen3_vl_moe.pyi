import abc
import torch
from .interfaces import MixtureOfExperts as MixtureOfExperts
from .qwen3_moe import (
    Qwen3MoeForCausalLM as Qwen3MoeForCausalLM,
    Qwen3MoeModel as Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock as Qwen3MoeSparseMoeBlock,
)
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder as Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration as Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor as Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo as Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer as Qwen3_VisionTransformer,
)
from .utils import (
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tokenizers.registry import (
    cached_tokenizer_from_config as cached_tokenizer_from_config,
)

logger: Incomplete

class Qwen3VLMoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self): ...

class Qwen3MoeLLMModel(Qwen3MoeModel):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class Qwen3VLMoeMixtureOfExperts(MixtureOfExperts):
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

class Qwen3VLMoeForConditionalGeneration(
    Qwen3VLForConditionalGeneration, Qwen3VLMoeMixtureOfExperts, metaclass=abc.ABCMeta
):
    is_3d_moe_weight: bool
    packed_modules_mapping: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    video_pruning_rate: Incomplete
    is_multimodal_pruning_enabled: Incomplete
    use_deepstack: Incomplete
    deepstack_num_level: Incomplete
    visual_dim: Incomplete
    multiscale_dim: Incomplete
    visual: Incomplete
    deepstack_input_embeds: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
