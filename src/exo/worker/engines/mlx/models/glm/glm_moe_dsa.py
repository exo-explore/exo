"""GLM-5.x (``glm_moe_dsa``) model classes: mlx-lm's, with the DSA indexer's key-norm
epsilon of the reference implementations.

mlx-lm's ``deepseek_v32.Indexer`` builds its key LayerNorm with MLX's default epsilon,
1e-5. The official reference implementation and Hugging Face transformers
(``GlmMoeDsaIndexer``) use 1e-6. The indexer picks which 2048 keys every query attends to, so the small
difference in the normalised keys moves keys across the top-k boundary: on
tiny-random/glm-5.3 in float32 against transformers, the last layer's median per-token
error falls from 2.6e-3 to 6.0e-5 with 1e-6, and the indexer's selections agree 0.991
instead of 0.810 at the last layer. Everything else is mlx-lm's model unchanged.
"""

from typing import cast

import mlx.nn as nn
from mlx_lm.models.glm_moe_dsa import Model as MlxLmModel
from mlx_lm.models.glm_moe_dsa import ModelArgs

__all__ = ["INDEXER_KEY_NORM_EPS", "Model", "ModelArgs"]

INDEXER_KEY_NORM_EPS = 1e-6


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
