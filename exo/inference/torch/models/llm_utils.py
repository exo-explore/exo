"""
Utility methods used by LLMs
"""
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune.modules as ttm
import math

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

class MultiLayerPreceptron(nn.Module):
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
    super(MultiLayerPreceptron, self).__init__()

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

def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

class MultiHeadAttention(nn.Module):
  """Multi-headed attention mechanism."""

  def __init__(
    self,
    hidden_size,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_emb,
    kv_cache: Optional[ttm.KVCache] = None,
    attention_dropout=0.0,
    is_causal=True
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.attention_dropout = attention_dropout
    self.is_causal = is_causal
    self.rotary_emb = rotary_emb

    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    self.kv_cache = kv_cache

  def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.size()

    if self.kv_cache is None or self.kv_cache.batch_size != batch_size:
      self.kv_cache = ttm.KVCache(
        batch_size=batch_size,
        max_seq_len=seq_len,
        num_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        dtype=hidden_states.dtype
      )

    # Project to queries, keys, and values
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to [batch_size, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Apply rotary positional embeddings if position_ids are provided
    # or use position_embeddings
    if position_embeddings is not None:
      cos, sin = position_embeddings
    else:
      cos, sin = self.rotary_emb(query_states, position_ids)

    # Expand cos and sin to match the shape of query_states
    cos = cos[:, :, None, :self.head_dim].expand_as(query_states)
    sin = sin[:, :, None, :self.head_dim].expand_as(query_states)
    print(f"cos: {cos.shape} | sin: {sin.shape}")

    # Apply rotary embeddings to queries and keys
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    # Repeat keys and values if needed
    if self.num_heads > self.num_kv_heads:
      n_rep = self.num_heads // self.num_kv_heads
      key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
      value_states = torch.repeat_interleave(value_states, n_rep, dim=1)

    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Forcing caching always enabled
    key_states, value_states = self.kv_cache.update(key_states, value_states)

    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # Apply causal mask, if applicable
    if self.is_causal:
      causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device))
      attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

    # Apply attention mask, if provided
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    # Softmax normalization
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape to [batch_size, seq_len, hidden_size]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    # Project back to hidden size
    attn_output = self.o_proj(attn_output)

    return attn_output

class RotaryEmbedding(nn.Module):
  """Rotary Position Embedding."""

  def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, rope_type="default", device=None):
    super().__init__()
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    self.scaling_factor = scaling_factor
    self.rope_type = rope_type

    # Initialize the inverse frequency for RoPE
    inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(self, x, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotary position embeddings (cos, sin) for the given input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_heads, head_dim).
        position_ids (torch.Tensor): The position indices for the sequence.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cos and sin embeddings.
    """
    # Expand inv_freq to match the batch size and sequence length
    batch_size, seq_len = position_ids.size(0), position_ids.size(1)
    inv_freq_expanded = self.inv_freq[None, :, None].expand(batch_size, -1, seq_len)

    # Expand position_ids to match the frequency tensor
    position_ids_expanded = position_ids[:, None, :].float()

    # Compute cos and sin embeddings
    freqs = torch.einsum("bnd,bnl->bnd", inv_freq_expanded, position_ids_expanded)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Apply the scaling factor to cos and sin embeddings
    cos = cos * self.scaling_factor
    sin = sin * self.scaling_factor

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

