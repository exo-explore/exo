from tinygrad import Tensor

def apply_embedding(
    embed: Tensor,
    input_ids: Tensor,
) -> Tensor:
    return embed[input_ids]

def apply_lm_head(x: Tensor, lm_head_weight: Tensor) -> Tensor:
    return x @ lm_head_weight.T
