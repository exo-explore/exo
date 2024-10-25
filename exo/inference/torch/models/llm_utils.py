"""
Utility methods used by LLMs
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
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

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim, activation='gelu', dropout=0.0, use_batchnorm=False):
    """
    General MLP (Multi-Layer Perceptron) module.

    Args:
      input_dim (int): Dimensionality of the input.
      hidden_dims (list of int): List of hidden layer dimensions.
      output_dim (int): Dimensionality of the output.
      activation (str): Activation function ('relu', 'gelu', 'tanh', 'sigmoid', etc.).
      dropout (float): Dropout probability.
      use_batchnorm (bool): Whether to use batch normalization.
    """
    super(MLP, self).__init__()

    self.layers = nn.ModuleList()
    self.use_batchnorm = use_batchnorm

    # Activation function mapping
    activations = {
      'relu': nn.ReLU(),
      'gelu': nn.GELU(),
      'tanh': nn.Tanh(),
      'sigmoid': nn.Sigmoid(),
      'leaky_relu': nn.LeakyReLU(0.2)
    }

    # Ensure valid activation
    if activation not in activations:
      raise ValueError(f"Invalid activation: {activation}. Choose from {list(activations.keys())}")

    self.activation = activations[activation]

    # Construct MLP layers
    prev_dim = input_dim
    for h_dim in hidden_dims:
      self.layers.append(nn.Linear(prev_dim, h_dim))
      if use_batchnorm:
        self.layers.append(nn.BatchNorm1d(h_dim))
      self.layers.append(self.activation)
      if dropout > 0:
        self.layers.append(nn.Dropout(dropout))
      prev_dim = h_dim

    # Output layer
    self.output_layer = nn.Linear(prev_dim, output_dim)

  def forward(self, x):
    """
    Forward pass for the MLP module.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after the MLP transformations.
    """
    for layer in self.layers:
      x = layer(x)
    return self.output_layer(x)

def create_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    seq_len: int,
    target_len: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_pos: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Creates a 4D causal attention mask from a 2D mask, with adjustments for static caching.

    Args:
      attention_mask (torch.Tensor):
          A 2D tensor of shape (batch_size, key_value_length) or a 4D tensor of shape
          (batch_size, 1, query_length, key_value_length).
      seq_len (int):
          Sequence length of the input being processed.
      target_len (int):
          Target length to generate the causal mask.
      dtype (torch.dtype):
          Data type for the causal mask.
      device (torch.device):
          Device to place the causal mask on.
      cache_pos (torch.Tensor):
          Cache position indices indicating the position of the input tokens in the sequence.
      batch_size (int):
          Number of samples in the batch.

    Returns:
      torch.Tensor:
          A 4D causal mask of shape (batch_size, 1, query_length, key_value_length).
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # If the mask is already 4D, return it directly
        return attention_mask

    min_value = torch.finfo(dtype).min

    # Create a 2D causal mask of shape (seq_len, target_len)
    causal_mask = torch.full(
        (seq_len, target_len), fill_value=min_value, dtype=dtype, device=device
    )

    if seq_len != 1:
        # Mask positions after the current position
        causal_mask = torch.triu(causal_mask, diagonal=1)

    # Adjust causal mask for cache position
    causal_mask *= (torch.arange(target_len, device=device) > cache_pos.view(-1, 1))

    # Expand to 4D and batch size
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    if attention_mask is not None:
        # Create a padding mask based on the input attention_mask
        mask_len = attention_mask.shape[-1]
        causal_mask = causal_mask.clone()  # Ensure contiguous memory for in-place operations
        padding_mask = causal_mask[:, :, :, :mask_len] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0

        # Apply padding to the causal mask
        causal_mask[:, :, :, :mask_len] = causal_mask[:, :, :, :mask_len].masked_fill(
            padding_mask, min_value
        )

    return causal_mask

