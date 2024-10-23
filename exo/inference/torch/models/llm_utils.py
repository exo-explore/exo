"""
Utility methods used by LLMs
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from exo.helpers import DEBUG

def load_model_config(model_config_path: Path) -> dict:
  """
  Loads the config.json of the model

  Args:
    model_path (Path): local path to model config json

  Returns:
    dict: The config as a dictionary
  """
  model_config = {}
  with open(model_config_path, "r") as f:
    model_config = json.load(f)
  return model_config

def select_next_token(
    logits,
    top_k=0,
    top_p=0.0,
    temperature=1.0,
    use_max=False,
):
  """
  Selects the next token from logits using top-k, top-p, and temperature scaling.

  Args:
    logits (torch.Tensor): Logits tensor of shape (batch_size, vocab_size).
    top_k (int): Number of top logits to consider for sampling.
    top_p (float): Cumulative probability threshold for nucleus sampling.
    temperature (float): Scaling factor for temperature.
    use_max (bool): Whether to use argmax for next token selection.
    debug (bool): If True, prints debugging information.

  Returns:
    next_token (torch.Tensor): The next token selected (batch_size,).
  """
  # Get logits for the last token in the sequence
  logits = logits[:, -1, :].clone().float()

  # Apply temperature scaling
  if temperature != 1.0:
    logits = logits / temperature

  # Apply top-k filtering
  if top_k > 0:
    # Get the top-k logits and set the rest to -inf
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    min_top_k_value = top_k_values[:, -1, None]
    logits = torch.where(logits < min_top_k_value, torch.tensor(float('-inf'), device=logits.device), logits)

  # Apply top-p (nucleus) filtering
  if top_p > 0.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens exceeding the top-p threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()  # Shift right
    sorted_indices_to_remove[:, 0] = 0  # Ensure at least one token is selected

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

  # Calculate probabilities
  probs = F.softmax(logits, dim=-1)

  # Select next token
  if not use_max:
    next_token = torch.multinomial(probs, num_samples=1)
  else:
    next_token = torch.argmax(logits, dim=-1, keepdim=True)

  # Debugging output
  if DEBUG >= 4:
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Next token: {next_token}")

  return next_token.squeeze(-1)


