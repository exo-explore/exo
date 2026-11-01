"""Recursive Flint (``flint``) model classes: mlx-lm's, with every RMSNorm computed the way
Flint's reference implementation does.

Hugging Face transformers' ``FlintRMSNorm`` (``GptOssRMSNorm``) normalises in float32 and
multiplies by the weight in float32, rounding to bf16 once:
``(weight * normalized).to(input_dtype)``. MLX's ``nn.RMSNorm`` rounds the normalised
activations to bf16 first and multiplies by the weight in bf16. Flint has four RMSNorms per
layer, two of them the per-head query and key norms in front of attention logits that reach
tens, so the extra rounding shows.

Measured in bf16 against transformers in float32 on 30 prompts (5,343 teacher-forced
positions, up to 7.4k tokens), mean KL goes from 0.0482 to 0.0454 and top-1 agreement where
the reference is at least 1 nat confident from 0.9897 to 0.9921. transformers' own bf16 run
scores 0.0452 and 0.9916 on the same positions. Everything else is mlx-lm's model unchanged.

It costs speed: on one M3 Ultra, decode 8-9 % slower (int4 115.6 to 105.4 tok/s, bf16 76.3
to 70.3) and prefill 2-5 %, from the float32 activations around ~200 norms per token. A fused
kernel that applies the weight in float32 before rounding would remove that.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.flint import Model as MlxLmModel
from mlx_lm.models.flint import ModelArgs

__all__ = ["Float32RMSNorm", "Model", "ModelArgs"]


class Float32RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` that normalises and applies its weight in float32, then rounds once."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32), self.eps
        ).astype(x.dtype)


def _float32_norm(norm: nn.RMSNorm) -> Float32RMSNorm:
    return Float32RMSNorm(norm.weight.shape[0], eps=norm.eps)


class Model(MlxLmModel):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        for layer in self.model.layers:
            layer.input_layernorm = _float32_norm(layer.input_layernorm)
            layer.post_attention_layernorm = _float32_norm(
                layer.post_attention_layernorm
            )
            layer.self_attn.q_norm = _float32_norm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = _float32_norm(layer.self_attn.k_norm)
        self.model.norm = _float32_norm(self.model.norm)
