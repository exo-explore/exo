import torch

def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    permuted_hidden_states: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
]: ...
def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None: ...
def moe_permute_unpermute_supported(): ...
