from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.quantization.layers import (
    QuantizedEmbedding,
    QuantizedLinear,
)

EmbedWeight = Tensor | QuantizedEmbedding
LinearWeight = Tensor | QuantizedLinear

def apply_embedding(
    embed: EmbedWeight,
    input_ids: Tensor,
) -> Tensor:
    if isinstance(embed, QuantizedEmbedding):
        return embed(input_ids)
    return embed[input_ids]

def apply_lm_head(x: Tensor, lm_head_weight: LinearWeight) -> Tensor:
    if isinstance(lm_head_weight, QuantizedLinear):
        return lm_head_weight(x)
    return x @ lm_head_weight.T
