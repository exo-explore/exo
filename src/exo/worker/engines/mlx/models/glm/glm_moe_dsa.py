"""GLM-5.x (``glm_moe_dsa``) model classes: mlx-lm's, computing some things the way the
reference implementations do, and attention restructured for speed.

Precision, as the reference implementations compute it:

1. The DSA indexer's key LayerNorm uses epsilon 1e-6, as in the official reference
   implementation and Hugging Face transformers (``GlmMoeDsaIndexer``). mlx-lm's
   ``deepseek_v32.Indexer`` uses MLX's default, 1e-5. The indexer picks the 2048 keys each
   query attends to, and the epsilon moves keys across that boundary.
2. The MoE router logits are computed in float32, as transformers does and as GLM-5.2+
   configs ask (``moe_router_dtype: "float32"``). mlx-lm computes ``x @ W.T`` in bf16 and
   only then the sigmoid in float32, which flips expert choices at near-ties. The router
   weight is converted to float32 once, at load.
3. The indexer's per-key score (relu(q.k) times the head weight, summed over its 32 heads)
   is summed in float32 (kernels.indexer_scores). mlx-lm adds the heads one at a time in
   bf16, which rounds 31 times and leaves many keys tied at the top-2048 boundary.

Measured against transformers in float32 on the first 8 layers of mlx-community/GLM-5.3-4bit
(same weights, dequantised), the float32 router moves next-token top-1 agreement from 0.956
to 0.967 and halves the last layer's error (1.4e-2 to 7.2e-3) on a 1.5k-token prompt, and
past 2048 tokens of a 20k-token prompt takes top-1 from 0.959 to 0.968 (mean KL 1.0e-4 to
6.2e-5). The epsilon alone is below bf16 noise on those weights; in float32 on
tiny-random/glm-5.3 it cuts the last layer's error from 2.6e-3 to 6.0e-5.

Speed: the attention is dsa_attention.GlmAttention (same math, restructured; see its
docstring), and the indexer's head reduction there is summed in float32 (3. above).
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

from exo.worker.engines.mlx.models.glm.dsa_attention import (
    SPARSE_PREFILL_MIN_CONTEXT,
    GlmAttention,
)

__all__ = [
    "INDEXER_KEY_NORM_EPS",
    "SPARSE_PREFILL_MIN_CONTEXT",
    "Float32RouterGate",
    "GlmAttention",
    "Model",
    "ModelArgs",
]

INDEXER_KEY_NORM_EPS = 1e-6


class Float32RouterGate(MoEGate):
    """mlx-lm's MoE gate with the router logits computed in float32 (Model.sanitize
    stores the router weight in float32)."""

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        weight = self.weight
        if weight.dtype != mx.float32:
            weight = weight.astype(mx.float32)
        logits = x.astype(mx.float32) @ weight.T
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
        for layer_idx, layer in enumerate(layers):
            attention = GlmAttention(config, layer_idx)
            layer.self_attn = attention
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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        weights = cast(
            dict[str, mx.array],
            super().sanitize(weights),  # pyright: ignore[reportUnknownMemberType]
        )
        # The router runs in float32: store its weight in float32 once.
        return {
            name: (
                value.astype(mx.float32) if name.endswith(".mlp.gate.weight") else value
            )
            for name, value in weights.items()
        }
