import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    ZeroExpertFusedMoE as ZeroExpertFusedMoE,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    block_dequant as block_dequant,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention as DeepseekV2MLAAttention,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class FlashConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    ep_size: Incomplete
    kv_lora_rank: Incomplete
    q_lora_rank: Incomplete
    qk_rope_head_dim: Incomplete
    v_head_dim: Incomplete
    qk_nope_head_dim: Incomplete
    num_experts_per_tok: Incomplete
    norm_topk_prob: Incomplete
    num_key_value_heads: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    pretraining_tp: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    mla_scale_q_lora: Incomplete
    mla_scale_kv_lora: Incomplete
    zero_expert_num: Incomplete
    zero_expert_type: Incomplete
    routed_scaling_factor: Incomplete
    hidden_act: str
    intermediate_size: Incomplete
    moe_intermediate_size: Incomplete
    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 8192,
        num_layers: int = 28,
        num_hidden_layers=None,
        num_attention_heads: int = 96,
        num_key_value_heads: int = 128,
        ep_size: int = 1,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        num_experts_per_tok=None,
        norm_topk_prob: bool = False,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        pad_token_id=None,
        bos_token_id: int = 100000,
        eos_token_id: int = 100001,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_parameters=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mla_scale_q_lora: bool = False,
        mla_scale_kv_lora: bool = False,
        dtype: str = "bfloat16",
        params_dtype: str = "bfloat16",
        router_dtype: str = "float32",
        router_bias: bool = False,
        topk_method=None,
        routed_scaling_factor: float = 1.0,
        zero_expert_num: int = 0,
        zero_expert_type=None,
        nextn_use_scmoe: bool = False,
        **kwargs,
    ) -> None: ...

class FlashMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class LongcatRouter(nn.Module):
    n_routed_experts: Incomplete
    classifier: Incomplete
    e_score_correction_bias: Incomplete
    def __init__(
        self,
        config: FlashConfig,
        zero_expert_num: int,
        router_params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states): ...

class LongcatMoe(nn.Module):
    hidden_size: Incomplete
    router_params_dtype: Incomplete
    router: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: FlashConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class FlashDecoderLayer(nn.Module):
    layer_idx: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    mlps: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: FlashConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class FlashModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class LongcatFlashForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
