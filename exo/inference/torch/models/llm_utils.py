"""
Utility methods used by LLMs
"""
import re
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune.modules as ttm
import math

from safetensors.torch import load_file as load_safetensors

from transformers import (
  LogitsProcessorList,
  TopKLogitsWarper,
  TopPLogitsWarper,
  TemperatureLogitsWarper
)
from transformers.cache_utils import Cache, DynamicCache

from exo.helpers import DEBUG
from exo.inference.shard import Shard
from exo.inference.torch.tests.test_llama3_model import MAX_SEQ_LEN

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

def check_weights(model, state_dict):
  """
  Verifies that the weights from the state dictionary are properly loaded into the model.
  """
  model_state_dict = model.state_dict()
  for name, param in model_state_dict.items():
    if name in state_dict:
      print(f"\nchecking {name}\n")
      loaded_param = state_dict[name]
      if param.shape != loaded_param.shape:
        print(f"Shape mismatch for {name}: expected {param.shape}, got {loaded_param.shape}")
      else:
        print(f"{name}: loaded correctly")
  
  for name in state_dict:
    if name not in model_state_dict:
      print(f"Unexpected weight {name} found in state_dict")

def load_model_weights_torchtune(cache_dir: Path, shard: Shard, model: Any):
  """
  Loads weights from huggingface and changes it to match torchtune naming structure
  """
  # Load weights from safetensors files in the cache directory
  safetensors_files = list(cache_dir.glob("*.safetensors"))
  if not safetensors_files:
    raise FileNotFoundError("No safetensors files found in the cache directory.")

  # Load weights from each found safetensors file
  paried_lmhead = True
  for safetensor_file in safetensors_files:
    state_dict = load_safetensors(safetensor_file)

    # remap to work with our model
    remapped_state_dict = {}
    paried_embed_weight = None
    for key, value in state_dict.items():
      # load layer by shard
      lnrgx = re.findall(r'model\.layers\.(\d+).*', key)
      layer_num = int(lnrgx[0]) if len(lnrgx) > 0 else None
      shard_layer_range = list(range(shard.start_layer, shard.end_layer))
      if layer_num in shard_layer_range:
        # change input layer norm to sa_norm for torchtune
        re_iln = re.findall(
          rf'model.layers\.{layer_num}\.(input_layernorm)\.weight', key)
        if len(re_iln) != 0:
          key = f"model.layers.{layer_num}.sa_norm.weight"

        # change post attention layernorm to mlp_norm for torchtune
        re_pal = re.findall(
          rf'model.layers\.{layer_num}\.(post_attention_layernorm)\.weight', key)
        if len(re_pal) != 0:
          key = f"model.layers.{layer_num}.mlp_norm.weight"

        # change o_proj to output_proj
        re_o_proj = re.findall(rf'model\.layers\.{layer_num}.(\w+)\.o_proj\.weight', key)
        if len(re_o_proj) != 0:
          key = f"model.layers.{layer_num}.{re_o_proj[0]}.output_proj.weight"

        # change self_attn to attn
        re_attn = re.findall(rf'model\.layers\.{layer_num}.(\w+)\.(\w+)\.(\w+)', key)
        if len(re_attn) != 0 and re_attn[0][0] == "self_attn":
          key = f"model.layers.{layer_num}.attn.{re_attn[0][1]}.{re_attn[0][2]}"

      # saving embed for paired weights
      elif key == 'model.embed_tokens.weight':
        paried_embed_weight = value
        # change name for torchtune
        key = 'model.tok_embeddings.weight'

      elif key == 'lm_head.weight':
        paried_lmhead = False
        # change key for torchtune
        key = 'model.output.weight'

      elif key == 'model.norm.weight':
        key = 'model.norm.weight'

      remapped_state_dict[key] = value

    if paried_lmhead:
      remapped_state_dict['model.output.weight'] = paried_embed_weight

    model.load_state_dict(remapped_state_dict, strict=False)

    #if DEBUG >= 7:
    print("\n--- checking weights ----\n")
    check_weights(model, remapped_state_dict)

def hf_logit_sample(
  logits,
  input_ids,
  use_max: bool=False,
  top_k: int=0,
  top_p: float=0.9,
  temp: float=1.0,
) -> torch.Tensor:
  """
  Logit sampling using transformers
  """
  logits_processor = LogitsProcessorList([
    TopKLogitsWarper(top_k),
    TemperatureLogitsWarper(temp),
    TopPLogitsWarper(top_p)
  ])

  # get a single cloned logit
  logits = logits[:, -1, :].clone().float()

  next_token_scores = logits_processor(input_ids, logits)

  if not use_max:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
  else:
    next_token = torch.argmax(next_token_scores, dim=-1)

  if DEBUG >= 4:
    print(f"input_ids: {input_ids}")
    print(f"next_token: {next_token}")

  return next_token[:, None].squeeze(-1)

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
  Creates a 4D causal attention mask from a 2D mask

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

class RotaryEmbedding(nn.Module):
  """
  Rotary Position Embedding.

  This computes the inverse frequencies according to the original RoPE implementation.
  There are other implementations that will be added.
  Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
  """

  def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
    super().__init__()
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    self.scaling_factor = scaling_factor

    # Initialize the inverse frequency for RoPE
    inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  @torch.no_grad()
  def forward(self, x, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotary position embeddings (cos, sin) for the given input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_heads, head_dim).
        position_ids (torch.Tensor): The position indices for the sequence.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cos and sin embeddings.
    """
    # Expand inv_freq to match the batch size
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.size(0), -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    # Compute cos and sin embeddings
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()

    # Apply the scaling factor to cos and sin embeddings
    cos = cos * self.scaling_factor
    sin = sin * self.scaling_factor

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MultiLayerPreceptron(nn.Module):
  def __init__(
    self,
    input_dim,
    hidden_dim,
    activation='silu',
    use_bias=False
  ):
    """
    General MLP (Multi-Layer Perceptron) module.

    Args:
      input_dim (int): Dimensionality of the input.
      hidden_dims (int): Hidden layer/intermediate dimensions.
      output_dim (int): Dimensionality of the output.
      activation (str): Activation function ('relu', 'gelu', 'tanh', 'sigmoid', etc.).
      dropout (float): Dropout probability.
      use_batchnorm (bool): Whether to use batch normalization.
    """
    super(MultiLayerPreceptron, self).__init__()

    # Activation function mapping
    activations = {
      'relu': nn.ReLU(),
      'gelu': nn.GELU(),
      'tanh': nn.Tanh(),
      'sigmoid': nn.Sigmoid(),
      'leaky_relu': nn.LeakyReLU(0.2),
      'silu': nn.SiLU()
    }

    # Ensure valid activation
    if activation not in activations:
      raise ValueError(f"Invalid activation: {activation}. Choose from {list(activations.keys())}")

    # Construct MLP layers
    self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.up_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.down_proj = nn.Linear(hidden_dim, input_dim, bias=use_bias)
    self.act_fn = activations[activation]

  def forward(self, x) -> torch.Tensor:
    """
    Forward pass for the MLP module.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after the MLP transformations.
    """

    return self.down_proj(
      self.act_fn(
        self.gate_proj(x)
      ) * self.up_proj(x)
    )

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

# ------------------
# Attention Methods
# ------------------

class MultiHeadAttention(nn.Module):
  """
  Multi-headed attention mechanism.

  Using the "attention is all you need" implementation. Other implementations will follow.
  Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L277
  Ref: https://pytorch.org/torchtune/0.3/_modules/torchtune/modules/attention.html
  """

  def __init__(
    self,
    hidden_size,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_emb,
    attention_dropout=0.0,
    is_causal=True,
    attention_bias=False
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.attention_dropout = attention_dropout
    self.is_causal = is_causal

    # nn layers
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
    self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
    self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)
    self.rotary_emb = rotary_emb

  def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[ttm.KVCache] = None,
    cos_sin_unsqueeze: int=1
  ) -> Tuple[torch.Tensor, Optional[ttm.KVCache]]:
    batch_size, seq_len, _ = hidden_states.size()

    # Project to queries, keys, and values
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Reshape to [batch_size, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Apply rotary positional embeddings if position_ids are provided
    # or use position_embeddings
    if position_embeddings is not None:
      cos, sin = position_embeddings
    else:
      cos, sin = self.rotary_emb(value_states, position_ids)

    print(f"cos: {cos.shape} | sin: {sin.shape}")
    # Expand cos and sin to match hidden_states' shape
    cos = cos.unsqueeze(cos_sin_unsqueeze)
    sin = sin.unsqueeze(cos_sin_unsqueeze)
    print(f"cos: {cos.shape} | sin: {sin.shape}")

    # Apply rotary embeddings to queries and keys
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Forcing caching always enabled
    if kv_cache is not None:
      #print(f"kv_cache.size {kv_cache.size}")

      #print(f"key_states.size(2) {key_states.size(2)}")

      #if kv_cache.size != key_states.size(2):
      #  print(f"\n MAKE NEW KVCACHE batch_size={key_states.size(0)} max_seq_len={key_states.size(2)}")
      #  kv_cache = ttm.KVCache(
      #    batch_size=key_states.size(0),
      #    max_seq_len=key_states.size(2),
      #    num_heads=self.num_kv_heads,
      #    head_dim=self.head_dim,
      #    dtype=hidden_states.dtype
      #  )

      key_states, value_states = kv_cache.update(key_states, value_states)
      print(f"kv_cache: {kv_cache.size}")
      print(f"key_states: {key_states.shape}")
      print(f"value_states: {value_states.shape}")

    # Repeat keys and values if needed
    #if self.num_heads > self.num_kv_heads:
    n_rep = self.num_heads // self.num_kv_heads
    key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
    value_states = torch.repeat_interleave(value_states, n_rep, dim=1)

    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    print(f"attn_weights: {attn_weights.shape}")

    # Apply attention mask, if provided
    if attention_mask is not None:
      print(f"attention_mask: {attention_mask.shape}")
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
      print(f"causal_mask: {causal_mask.shape}")
      attn_weights = attn_weights + causal_mask
      print(f"attn_weights: {attn_weights.shape}")

    # Softmax normalization
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    print(f"attn_weights: {attn_weights.shape}")

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    print(f"attn_output: {attn_output.shape}")

    # Transpose attention output
    attn_output = attn_output.transpose(1,2).contiguous()
    print(f"attn_output: {attn_output.shape}")

    # Reshape [batch_size, seq_len, -1]
    attn_output = attn_output.reshape(batch_size, seq_len, -1)
    print(f"attn_output after transpose: {attn_output.shape}")

    # Project back to hidden size
    attn_output = self.o_proj(attn_output)
    print(f"attn_output: {attn_output.shape}")

    return attn_output, kv_cache

class SDPAttention(nn.Module):
  """
  Scaled dot product attention mechanism.

  Using the scaled dot product attention method from pytorch
  Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L524
  """

  def __init__(
    self,
    hidden_size,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_emb,
    attention_dropout=0.0,
    is_causal=True,
    attention_bias=False,
    kv_max_seq_len=2048
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.attention_dropout = attention_dropout
    self.is_causal = is_causal
    self.kv_max_seq_len = kv_max_seq_len

    # nn layers
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
    self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
    self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)
    self.rotary_emb = rotary_emb

  def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[ttm.KVCache] = None,
    cos_sin_unsqueeze: int=1
  ) -> Tuple[torch.Tensor, Optional[ttm.KVCache]]:
    batch_size, seq_len, _ = hidden_states.size()

    # Project to queries, keys, and values
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Reshape to [batch_size, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Apply rotary positional embeddings if position_ids are provided
    # or use position_embeddings
    if position_embeddings is not None:
      cos, sin = position_embeddings
    else:
      cos, sin = self.rotary_emb(value_states, position_ids)

    print(f"cos: {cos.shape} | sin: {sin.shape}")
    # Expand cos and sin to match hidden_states' shape
    cos = cos.unsqueeze(cos_sin_unsqueeze)
    sin = sin.unsqueeze(cos_sin_unsqueeze)
    print(f"cos: {cos.shape} | sin: {sin.shape}")

    # Apply rotary embeddings to queries and keys
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    # Caching
    if kv_cache is not None:
      if kv_cache.size >= self.max_seq_len:
        # double the cache each time space is ran out
        self.kv_max_seq_len = self.kv_max_seq_len + self.kv_max_seq_len

        kv_cache = ttm.KVCache(
          batch_size=key_states.size(0),
          max_seq_len=self.kv_max_seq_len,
          num_heads=self.num_kv_heads,
          head_dim=self.head_dim,
          dtype=hidden_states.dtype
        )

      key_states, value_states = kv_cache.update(key_states, value_states)

      # **Slice KVCache to match `query_states` length**
      key_states = key_states[:, :, :seq_len, :]
      value_states = value_states[:, :, :seq_len, :]

      # kv_cache.update(key_states, value_states)
      print(f"kv_cache: {kv_cache.size}")
      print(f"from kv_cache / key_states: {key_states.shape}")
      print(f"from kv_cache / value_states: {value_states.shape}")

    # Repeat keys and values if needed
    #if self.num_heads > self.num_kv_heads:
    n_rep = self.num_heads // self.num_kv_heads
    key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
    value_states = torch.repeat_interleave(value_states, n_rep, dim=1)

    print(f"query_states: {query_states.shape}")
    print(f"key_states: {key_states.shape}")
    print(f"value_states: {value_states.shape}")

    causal_mask = attention_mask
    if causal_mask is not None:
      causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
      print(f"causal_mask: {causal_mask.shape}")

    if query_states.device.type == "cuda" and causal_mask is not None:
      query_states = query_states.contiguous()
      key_states = key_states.contiguous()
      value_states = value_states.contiguous()

      print(f"query_states: {query_states.shape}")
      print(f"key_states: {key_states.shape}")
      print(f"value_states: {value_states.shape}")

    is_causal = True if causal_mask is None and seq_len > 1 else False

    attn_output = F.scaled_dot_product_attention(
      query_states,
      key_states,
      value_states,
      attn_mask=causal_mask,
      dropout_p=0.0,
      is_causal=is_causal,
    )

    print(f"attn_output: {attn_output.shape}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)

    attn_output = self.o_proj(attn_output)

    print(f"attn_output: {attn_output.shape}")

    return attn_output, kv_cache

