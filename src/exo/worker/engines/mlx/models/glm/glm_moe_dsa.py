"""GLM-5.x (``glm_moe_dsa``) model classes: mlx-lm's, computing two things the way the
reference implementations do.

1. The DSA indexer's key LayerNorm uses epsilon 1e-6, as in the official reference
   implementation and Hugging Face transformers (``GlmMoeDsaIndexer``). mlx-lm's
   ``deepseek_v32.Indexer`` uses MLX's default, 1e-5. The indexer picks the 2048 keys each
   query attends to, and the epsilon moves keys across that boundary.
2. The MoE router logits are computed in float32, as transformers does and as GLM-5.2+
   configs ask (``moe_router_dtype: "float32"``). mlx-lm computes ``x @ W.T`` in bf16 and
   only then the sigmoid in float32, which flips expert choices at near-ties.

Measured against transformers in float32 on the first 8 layers of mlx-community/GLM-5.3-4bit
(same weights, dequantised), the float32 router moves next-token top-1 agreement from 0.956
to 0.967 and halves the last layer's error (1.4e-2 to 7.2e-3) on a 1.5k-token prompt, and
past 2048 tokens of a 20k-token prompt takes top-1 from 0.959 to 0.968 (mean KL 1.0e-4 to
6.2e-5). The epsilon alone is below bf16 noise on those weights; in float32 on
tiny-random/glm-5.3 it cuts the last layer's error from 2.6e-3 to 6.0e-5. Everything else
is mlx-lm's model unchanged.
"""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v32 import ModelArgs as DeepseekV32ModelArgs
from mlx_lm.models.deepseek_v32 import (
    MoEGate,
    group_expert_select,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.glm_moe_dsa import Model as MlxLmModel
from mlx_lm.models.glm_moe_dsa import ModelArgs

__all__ = ["INDEXER_KEY_NORM_EPS", "Float32RouterGate", "Model", "ModelArgs"]

INDEXER_KEY_NORM_EPS = 1e-6


class Float32RouterGate(MoEGate):
    """mlx-lm's MoE gate with the router logits computed in float32."""

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        return cast(
            tuple[mx.array, mx.array],
            group_expert_select(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.n_group,
                self.topk_group,
                self.routed_scaling_factor,
                self.norm_topk_prob,
            ),
        )


class Model(MlxLmModel):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        layers = cast(list[nn.Module], self.model.layers)  # pyright: ignore[reportUnknownMemberType]
        for layer in layers:
            attention = cast(nn.Module, layer.self_attn)
            # Layers that reuse the previous layer's top-k ("shared") have no indexer.
            indexer = cast(nn.Module | None, getattr(attention, "indexer", None))
            if indexer is not None:
                key_norm = cast(nn.LayerNorm, indexer.k_norm)
                key_norm.eps = INDEXER_KEY_NORM_EPS
            mlp = cast(nn.Module, layer.mlp)
            if isinstance(getattr(mlp, "gate", None), MoEGate):
                # Same parameter names (gate.weight, gate.e_score_correction_bias).
                # MoEGate reads only the routing fields, which GLM's args share.
                mlp.gate = Float32RouterGate(
                    cast(DeepseekV32ModelArgs, cast(object, config))
                )
