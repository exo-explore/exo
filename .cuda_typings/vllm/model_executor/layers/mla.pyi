import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.config import CacheConfig as CacheConfig
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention as MLAAttention
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)

@dataclass
class MLAModules:
    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = ...

class MultiHeadLatentAttentionWrapper(PluggableLayer):
    hidden_size: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    num_heads: Incomplete
    fused_qkv_a_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    rotary_emb: Incomplete
    o_proj: Incomplete
    indexer: Incomplete
    indexer_rope_emb: Incomplete
    is_sparse: Incomplete
    topk_tokens: Incomplete
    topk_indices_buffer: Incomplete
    mla_attn: Incomplete
    prefix: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
