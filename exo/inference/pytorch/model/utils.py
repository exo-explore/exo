import torch
from torch.nn import functional as F

def top_p_sampling(scaled_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        scaled_logits (torch.Tensor): The scaled logits from the model's output.
        top_p (float): The cumulative probability threshold for top-p filtering.
        temp (float): Temperature parameter for softmax distribution reshaping.
    
    Returns:
        torch.Tensor: Token selected based on the top-p criterion.

    Ref:
        https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py#L67C1-L97C17
    """
    scaled_logits = torch.where(torch.isnan(scaled_logits), torch.zeros_like(scaled_logits), scaled_logits)
    scaled_logits = torch.where(torch.isinf(scaled_logits), torch.full_like(scaled_logits, 1e6), scaled_logits)

    probs = torch.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p

    top_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
    sum_probs = top_probs.sum(dim=-1, keepdim=True)
    top_probs = torch.where(sum_probs > 0, top_probs / sum_probs, torch.ones_like(top_probs) / top_probs.size(-1))

    if torch.isnan(top_probs).any() or torch.isinf(top_probs).any():
        print("Warning: Top probabilities contain NaN or Inf values after normalization")
        top_probs = torch.where(torch.isnan(top_probs) | torch.isinf(top_probs), 
                                1.0 / top_probs.size(-1), 
                                top_probs)

    sorted_token = torch.multinomial(top_probs, num_samples=1)

    token = sorted_indices.gather(-1, sorted_token)

    return token.squeeze(-1)

def sample_logits(logits, temp, top_p, top_k):
    """
    Sample tokens from logits using temperature, top-k, and top-p (nucleus) sampling.

    Args:
        logits (torch.Tensor): The logits distribution to sample from.
        temp (float): temp for scaling logits.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: The selected token index.
    """

     # Ensure logits are float
    logits = logits.float()

    # If temp is very low, just use argmax
    if temp == 0:
        return logits.argmax(dim=-1)
    
    scaled_logits = logits/temp

    # top k
    if top_k > 0:
        top_values, top_indices = torch.topk(scaled_logits, top_k, dim=-1)
        scaled_logits = torch.zeros_like(logits).scatter_(-1, top_indices, top_values)
    
    # Top-p sampling
    if 0 < top_p < 1.0:
        return top_p_sampling(scaled_logits, top_p)
    else:
        # random distribution selection
        probs = torch.softmax(scaled_logits, dim=-1)
        rand_sample = torch.distributions.Categorical(probs)
        return rand_sample.sample().squeeze()