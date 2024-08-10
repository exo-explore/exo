import torch
from torch.nn import functional as F

def sample_logits(logits, temp=0.0, top_k=15, top_p=0.9, alpha_f=0.1, alpha_p=0.0):
    # Ensure logits is a 2D tensor
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Apply temperature scaling
    if temp > 0:
        logits = logits / temp

    # Top-k sampling
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_values)

    # Top-p (nucleus) sampling
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[torch.arange(logits.size(0)).unsqueeze(1), indices_to_remove] = float('-inf')

    # Alpha sampling (to discourage repetition)
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