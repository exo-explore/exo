import torch
from torch.nn import functional as F

def top_k_sampling(logits, thres):
    num_logits = logits.shape[-1]
    val, ind = torch.topk(logits, thres, dim=-1, largest=True, sorted=True)
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, ind, 1)
    logits = logits * mask

    return logits

def top_p_sampling(logits, thres):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    print(f"top_p_sampling sorted_logits\n{sorted_logits}\nsorted_indices {sorted_indices}")
    softmax_logits = F.softmax(sorted_logits, dim=-1)
    print(f"top_p_sampling\nsoftmax_logits {softmax_logits}")
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    print(f"top_p_sampling\n{cumulative_probs}")


    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > thres
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    print(f"top_p_sampling\nindicies_to_remove: {indices_to_remove}")
    logits[indices_to_remove] = float('-inf')
    return logits

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
    # If temp is very low, just use argmax
    if temp == 0:
        return logits.argmax(dim=-1)
    
    print(f"logits {logits}")

    scaled_logits = logits/temp

    print(f"scaled_logits: {scaled_logits}")

    if 0 < top_p < 1.0:
        top_p_logits = top_p_sampling(scaled_logits, top_p)
        print(f"top_p logits {top_p_logits}")
        if top_k > 0:
            top_k_logits = top_k_sampling(top_p_logits, top_k)
            return top_k_logits.argmax(dim=-1)
    elif top_k > 0:
        top_k_logits = top_k_sampling(logits, top_k)
        print(f"top_k logits {top_k_logits}")
        return top_k_logits.argmax(dim=-1)

    return scaled_logits.argmax(dim=-1)


# from tinygrad llama model sample
def sample(logits: torch.Tensor, temp: float, k: int, p: float, af: float, ap: float):
    assert logits.ndim == 1, "only works on 1D tensors"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

    # If temperature is very low, just use argmax
    if temp < 1e-6:
        return logits.argmax().reshape(1)

    # Alpha sampling
    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            sample.alpha_counter = torch.zeros_like(logits, dtype=torch.int32).contiguous()
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0).float() * ap)

    # Replace NaNs with -inf
    logits = torch.where(logits != logits, torch.tensor(-float("inf"), device=logits.device), logits)

    # Apply softmax after temperature scaling
    t = F.softmax(logits / temp, dim=-1)

    counter = torch.arange(t.numel(), device=logits.device).contiguous()
    counter2 = torch.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()

    # Top-k sampling
    if k:
        output = torch.zeros(k, device=logits.device).contiguous()
        output_indices = torch.zeros(k, device=logits.device, dtype=torch.int32).contiguous()

        for i in range(k):
            t_max = t.max()
            t_argmax = (t.numel() - ((t == t_max) * counter2).max() - 1).to(torch.int)
            output[i] = t_max
            output_indices[i] = t_argmax
            t = torch.where(counter == t_argmax, torch.tensor(0.0, device=logits.device), t)

        # Approximate top-p sampling
        output_cumsum = output.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,)) + t.sum()
        mask = output_cumsum >= (1 - p)
        output = output * mask.float()
        output_indices = output_indices * mask.int()

        # Sample from the distribution
        output_idx = output.multinomial(num_samples=1)
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial(num_samples=1)

    # Increase alpha counter
    if af or ap:
        sample.alpha_counter = torch.where(counter == output_token, sample.alpha_counter + 1, sample.alpha_counter)

    return output_token


def sample_3d(logits: torch.Tensor, temp: float, k: int, p: float, af: float, ap: float):
    assert logits.ndim == 3, "only works on 3D tensors"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert 0 <= k <= logits.shape[-1], "k must be between 0 and the last dimension size"

    batch_size, seq_len, vocab_size = logits.shape

    # If temperature is very low, just use argmax
    if temp < 1e-6:
        return logits.argmax(dim=-1)

    # Alpha sampling
    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            sample.alpha_counter = torch.zeros_like(logits, dtype=torch.int32).contiguous()
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0).float() * ap)

    # Replace NaNs with -inf
    logits = torch.where(logits != logits, torch.tensor(-float("inf"), device=logits.device), logits)

    # Apply softmax after temperature scaling
    t = F.softmax(logits / temp, dim=-1)

    counter = torch.arange(vocab_size, device=logits.device).unsqueeze(0).unsqueeze(0).expand_as(t).contiguous()
    counter2 = torch.arange(vocab_size - 1, -1, -1, device=logits.device).unsqueeze(0).unsqueeze(0).expand_as(t).contiguous()

    # Top-k sampling
    if k:
        output = torch.zeros((batch_size, seq_len, k), device=logits.device).contiguous()
        output_indices = torch.zeros((batch_size, seq_len, k), device=logits.device, dtype=torch.int32).contiguous()

        for i in range(k):
            t_max, _ = t.max(dim=-1, keepdim=True)
            t_argmax = (vocab_size - ((t == t_max) * counter2).max(dim=-1, keepdim=True)[0] - 1).to(torch.int)
            output[:, :, i] = t_max.squeeze(-1)
            output_indices[:, :, i] = t_argmax.squeeze(-1)
            t = torch.where(counter == t_argmax, torch.tensor(0.0, device=logits.device), t)

        # Approximate top-p sampling
        output_cumsum = output.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,)) + t.sum(dim=-1, keepdim=True)
        mask = output_cumsum >= (1 - p)
        output = output * mask.float()
        output_indices = output_indices * mask.int()

        # Sample from the distribution
        output_flat = output.view(batch_size * seq_len, -1)
        output_idx = output_flat.multinomial(num_samples=1).squeeze(-1)
        output_indices_flat = output_indices.view(batch_size * seq_len, -1)
        output_token = output_indices_flat.gather(dim=-1, index=output_idx.unsqueeze(-1)).view(batch_size, seq_len)
    else:
        output_flat = t.view(batch_size * seq_len, -1)
        output_token = output_flat.multinomial(num_samples=1).view(batch_size, seq_len)

    # Increase alpha counter
    if af or ap:
        sample.alpha_counter = torch.where(counter == output_token.unsqueeze(-1), sample.alpha_counter + 1, sample.alpha_counter)

    return output_token

