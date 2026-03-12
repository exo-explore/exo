import torch
from _typeshed import Incomplete

MASK_64_BITS: Incomplete

def random_uuid() -> str: ...
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int: ...
