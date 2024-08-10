import torch
from torch.nn import functional as F

def top_p_sampling(logits, top_p: float, temperature: float = 1.0):
    """
    Perform top-p sampling (nucleus sampling) on logits.
    
    Args:
        logits (torch.Tensor): The logits distribution to sample from.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        temperature (float): Sampling temperature.

    Returns:
        torch.Tensor: The selected token indices.
    """
    # Apply temperature scaling
    logits = logits/temperature
    
    # Sort the logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Create a mask to remove logits with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Mask the logits
    sorted_logits[sorted_indices_to_remove] = -float('Inf')
    
    # Sample from the filtered distribution
    probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, 1)
    
    # Convert to original index order
    return sorted_indices.gather(-1, sampled_token)

def sample_logits(logits, temp=0.8, top_k=32, top_p=1.0, alpha_f=0.1, alpha_p=0.0):
    """
    Sample tokens from logits using temperature, top-k, top-p, and alpha sampling.

    Args:
        logits (torch.Tensor): The logits distribution to sample from.
        temp (float): Temperature for scaling logits.
        top_k (int): The number of top tokens to consider for sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        alpha_f (float): Penalty factor for repetition.
        alpha_p (float): Penalty for selecting already selected tokens.

    Returns:
        torch.Tensor: The selected token indices.
    """
    # Return argmax for deterministic output at low temperature
    if temp < 1e-6: 
        return logits.argmax(dim=-1)

    # Apply Top-k sampling if specified
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_values)

    # Apply Top-p (nucleus) sampling if specified
    if 0 < top_p < 1.0:
        logits = top_p_sampling(logits, top_p, temp)

    # Apply alpha sampling to discourage repetition
    if alpha_f or alpha_p:
        if not hasattr(sample_logits, "alpha_counter"):
            sample_logits.alpha_counter = torch.zeros_like(logits, dtype=torch.int32)
        logits = logits - (sample_logits.alpha_counter * alpha_f + (sample_logits.alpha_counter > 0) * alpha_p)

    # Sample from the logits
    probabilities = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, 1)

    # Update alpha counter
    if alpha_f or alpha_p:
        sample_logits.alpha_counter.scatter_(-1, sampled_token, sample_logits.alpha_counter.gather(-1, sampled_token) + 1)

    return sampled_token.squeeze()

