from _typeshed import Incomplete
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from typing import Any

logger: Incomplete
ARCTIC_PRETRAINED_CONFIG_ARCHIVE_MAP: Incomplete

@dataclass
class ArcticLoRAConfig:
    lora_r: int = ...
    lora_alpha: float = ...
    shard_base_weights: bool = ...

@dataclass
class ArcticQuantizationConfig:
    q_bits: int = ...
    rounding: str = ...
    mantissa_bits: int = ...
    group_size: int = ...

class ArcticConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    sliding_window: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    attention_dropout: Incomplete
    num_experts_per_tok: Incomplete
    num_local_experts: Incomplete
    router_aux_loss_coef: Incomplete
    moe_layer_frequency: Incomplete
    moe_train_capacity_factor: Incomplete
    moe_eval_capacity_factor: Incomplete
    enable_expert_tensor_parallelism: Incomplete
    moe_min_capacity: Incomplete
    moe_token_dropping: Incomplete
    parallel_attn_mlp_res: Incomplete
    quantization: Incomplete
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads=None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        pad_token_id=None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        sliding_window=None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 8,
        router_aux_loss_coef: float = 0.001,
        moe_layer_frequency: int = 2,
        parallel_attn_mlp_res: bool = False,
        moe_train_capacity_factor: int = 1,
        moe_eval_capacity_factor: int = 1,
        enable_expert_tensor_parallelism: bool = False,
        moe_min_capacity: int = 0,
        moe_token_dropping: bool = True,
        quantization=None,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> ArcticConfig: ...
    def to_dict(self) -> dict[str, Any]: ...
