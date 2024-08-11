import torch
from torch.nn import functional as F

def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, alpha_f=0.0, alpha_p=0.0):
    """
    Sample tokens from logits using temperature, top-k, and top-p (nucleus) sampling.

    Args:
        logits (torch.Tensor): The logits distribution to sample from.
        temperature (float): Temperature for scaling logits.
        top_k (int): The number of top tokens to consider for sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        alpha_f (float): Penalty factor for repetition frequency.
        alpha_p (float): Penalty for repeated selection.

    Returns:
        torch.Tensor: The selected token index.
    """

     # Ensure logits are float
    logits = logits.float()

    # If temperature is very low, just use argmax
    if temperature < 1e-6:
        return logits.argmax(dim=-1)

    # Alpha sampling (adjusting logits based on past selections)
    if alpha_f > 0.0 or alpha_p > 0.0:
        logits -= (sample_logits.alpha_counter * alpha_f + (sample_logits.alpha_counter > 0) * alpha_p)

    # Replace NaNs with -inf to prevent softmax issues
    logits = torch.where(torch.isnan(logits), torch.full_like(logits, -float('inf')), logits)

    # Apply temperature scaling
    logits = logits / temperature

    # Top-k sampling
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, -float('inf'))
        logits.scatter_(-1, top_k_indices, top_k_values)

    # Top-p sampling
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits[sorted_indices_to_remove] = -float('inf')
        logits = sorted_logits

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Sample from the probabilities
    sampled_token = torch.multinomial(probabilities, 1)

    return sampled_token.squeeze()