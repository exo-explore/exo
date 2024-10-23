"""
llama3 model

Written with pytorch using torchtune and other methods
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune.modules import (
  MultiHeadAttention,
  RotaryPositionalEmbeddings,
  KVCache,
  RMSNorm
)

from exo.inference.shard import Shard

class LlamaBlock(nn.Module):
  """
  Encoder block class for the LLaMA model
  """
  def __init__(
    self,
    dim,
    heads,
    num_kv_heads,
    head_dim,
    ff_dim,
    rms_norm_eps,
    rotary_pos_emb,
    attention_dropout=0.0,
    use_bias=False,
    max_seq_len=4096,
    pos_embeddings=None
  ):
    super(LlamaBlock, self).__init__()
    # Class vars
    self.dim = dim
    self.heads = heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.ff_dim = ff_dim
    self.rms_norm_eps = rms_norm_eps
    self.attention_dropout = attention_dropout
    self.use_bias = use_bias
    self.max_seq_len = max_seq_len
    self.pos_embeddings = pos_embeddings
    self.rotary_pos_emb = rotary_pos_emb

    # Define linear projections for Q, K, V, and Output
    self.q_proj = nn.Linear(dim, heads * head_dim, bias=use_bias)
    self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.output_proj = nn.Linear(heads * head_dim, dim, bias=use_bias)

    # Define optional query normalization
    self.q_norm = nn.LayerNorm(head_dim, eps=rms_norm_eps)

    # MultiHeadAttention from torchtune
    self.attn = MultiHeadAttention(
      embed_dim=dim,
      num_heads=heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      q_proj=self.q_proj,
      k_proj=self.k_proj,
      v_proj=self.v_proj,
      output_proj=self.output_proj,
      pos_embeddings=pos_embeddings,
      q_norm=self.q_norm,
      k_norm=self.q_norm,
      kv_cache=None,
      max_seq_len=max_seq_len,
      is_causal=True,
      attn_dropout=attention_dropout
    )

    # RMSNorm layers before and after attention and feed-forward layers
    self.norm1 = nn.LayerNorm(dim, eps=rms_norm_eps)
    self.norm2 = nn.LayerNorm(dim, eps=rms_norm_eps)

    # Feed-forward layer with SwiGLU activation
    self.feed_forward = nn.Sequential(
      nn.Linear(dim, ff_dim),
      nn.GLU(),  # SwiGLU approximation
      nn.Linear(ff_dim // 2, dim)
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    position_embeddings: Optional[torch.FloatTensor] = None
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Forward pass with integrated attention and key-value caching.

    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
      kv_cache (Optional[KVCache]): KVCache object for managing past key-value states.
      attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len).
      position_ids (Optional[torch.Tensor]): Position IDs tensor of shape (batch_size, seq_len).

    Returns:
      Tuple[torch.Tensor, KVCache]:
        - x (torch.Tensor): Output tensor of shape (batch_size, seq_len, dim).
        - kv_cache (KVCache): Updated KVCache object.
    """
    batch_size, seq_len, _ = hidden_states.shape
    attn_output, attn_weights = None

    # Do kvq projection
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape
    query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Initialize or update KVCache
    if kv_cache is None:
      kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=self.attn.max_seq_len,
        num_heads=self.heads,
        head_dim=self.attn.head_dim,
        dtype=hidden_states.dtype
      )

    # cache
    value_states = kv_cache.update(key_states, value_states)

    # Attention weights and causal mask
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
      attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    

    return attn_output, attn_weights, kv_cache

class LlamaModel(nn.Module):
  """
  LlamaModel is a pure PyTorch implementation of the LLaMA architecture
  """

  def __init__(self, config: dict, shard: Shard):
    """
    Initialize the LlamaModel.

    Args:
      config (dict): Configuration dictionary containing model parameters.
        - hidden_size (int): Size of the hidden layers.
        - num_hidden_layers (int): Number of transformer layers.
        - num_attention_heads (int): Number of attention heads.
        - intermediate_size (int): Size of the intermediate (feed-forward) layers.
        - vocab_size (int): Vocabulary size for the embedding layer.
        - max_position_embeddings (int): Maximum number of positional embeddings.
        - rms_norm_eps (float): Epsilon for RMS normalization.
        - head_dim (int): Dimension of each attention head.
        - attention_dropout (float): Dropout rate for attention layers.
    """
    super(LlamaModel, self).__init__()

    self.shard = shard

    # Load configurations from config
    self.config = config
    self.hidden_size = config['hidden_size']
    self.num_layers = config['num_hidden_layers']
    self.num_heads = config['num_attention_heads']
    self.num_kv_heads = config['num_key_value_heads']
    self.intermediate_size = config['intermediate_size']
    self.vocab_size = config['vocab_size']
    self.max_position_embeddings = config['max_position_embeddings']
    self.rms_norm_eps = config['rms_norm_eps']
    self.head_dim = config['head_dim']
    self.attention_dropout = config.get('attention_dropout', 0.0)
    self.padding_idx = config["pad_token_id"]

    # Model layers
    self.embed = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
    self.rotary_pos_emb = RotaryPositionalEmbeddings(
      self.hidden_size // self.num_heads,
      config['rope_scaling']['original_max_position_embeddings'],
      config['rope_theta']
    )
    self.layers = nn.ModuleList([
      LlamaBlock(
        dim=self.hidden_size,
        heads=self.hidden_size // self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        ff_dim=self.intermediate_size,
        rms_norm_eps=self.rms_norm_eps,
        attention_dropout=self.attention_dropout,
        use_bias=config.get('attention_bias', False),
        rotary_pos_emb=self.rotary_pos_emb
      ) for _ in range(self.num_layers)
    ])
    self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
    self.to_logits = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_kv_cache: Optional[KVCache] = None,
  ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], KVCache]:
    """
    Forward pass with integrated position ID handling, attention mask, and optional KVCache.

    Args:
      input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
      attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).
      position_ids (Optional[torch.Tensor]): Position IDs. If None, they are calculated automatically.
      cache_position (Optional[torch.LongTensor]): the positions of inputs in the sequence
      past_kv_cache (Optional[KVCache]): Optional KVCache for efficient generation.
        If provided, it stores past key-value states for faster autoregressive inference.

    Returns:
      Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], KVCache]:
        - logits (Optional[torch.Tensor]): Output logits of shape (batch_size, seq_len, vocab_size).
        - hidden_states (Optional[torch.Tensor]): Hidden states from each layer
        - past_kv_cache (KVCache): Updated KVCache object.
    """
    batch_size, seq_len = input_ids.shape

    # Create initial embeddings
    input_embeds = self.embed(input_ids)

    # Initialize or use the provided KVCache
    if past_kv_cache is None:
      past_kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=self.max_position_embeddings,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dtype=x.dtype
      )

    # Initialize position IDs if not provided
    if cache_position is None:
      past_seen_tokens = past_kv_cache.size if past_kv_cache is not None else 0
      cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + seq_len,
        device=input_ids.device
      ).unsqueeze(0).expand(batch_size, -1)

    if position_ids is None:
      position_ids = cache_position.unsqueeze(0)

    hidden_states = input_embeds

    # Apply rotary positional embeddings
    position_embeddings = self.rotary_pos_emb(
      hidden_states,
      input_pos=position_ids
    )

    # Apply attention mask if provided (convert to appropriate format)
    if attention_mask is not None:
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
      attention_mask = (1.0 - attention_mask) * -1e4  # Convert to large negative values

    # Forward pass through layers with KVCache
    for layer_idx in range(self.shard.end_layer, self.shard.start_layer):
      layer_hidden_state, layer_kv_cache = layer(
        hidden_states=hidden_states,
        kv_cache=past_kv_cache,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings
      )

      hidden_states = layer_hidden_state

    # Apply final layer normalization
    hidden_states = self.norm(hidden_states)

    # Compute logits if at end layer
    if self.shard.is_last_layer():
      logits = self.to_logits(hidden_states)
    else:
      logits = None

    # Prepare the return values
    if return_hidden_states:
      return logits, hidden_states, past_kv_cache
    else:
      return logits, None, past_kv_cache
