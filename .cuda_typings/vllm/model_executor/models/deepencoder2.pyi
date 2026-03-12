import torch
import torch.nn as nn
from _typeshed import Incomplete

class CustomQwen2Decoder(nn.Module):
    model: Incomplete
    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
    ) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ): ...

class Qwen2Decoder2Encoder(nn.Module):
    model: Incomplete
    query_768: Incomplete
    query_1024: Incomplete
    def __init__(
        self,
        decoder_layer: int,
        hidden_dimension: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def build_qwen2_decoder_as_encoder(
    decoder_layer: int = 24,
    hidden_dimension: int = 896,
    num_attention_heads: int = 14,
    num_key_value_heads: int = 2,
    intermediate_size: int = 4864,
): ...
