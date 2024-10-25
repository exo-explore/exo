"""
llama3 model

Written with pytorch using torchtune and other methods
"""
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
from exo.inference.torch.models.llm_utils import MLP, create_4d_causal_attention_mask

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
    self.q_proj = nn.Linear(dim, heads * head_dim, bias=use_bias)
    self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
    self.output_proj = nn.Linear(heads * head_dim, dim, bias=use_bias)
    self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
    self.mlp = MLP(
      input_dim=dim,
      hidden_dims=[ff_dim],  # Single hidden layer with ff_dim as the hidden size
      output_dim=dim,
      activation='gelu',
      dropout=attention_dropout
    )
    self.post_norm = RMSNorm(dim, eps=rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """
    Forward pass with integrated attention, resnet and key-value caching.

    Args:
      hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
      kv_cache (Optional[KVCache]): KVCache object for managing past key-value states.
      attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_len).
      position_ids (Optional[torch.Tensor]): Position IDs tensor of shape (batch_size, seq_len).

    Returns:
      Tuple[torch.Tensor, KVCache]:
        - Output tensor of shape (batch_size, seq_len, dim).
        - Updated KVCache object.
    """
    # setting up resnet
    residual = hidden_states

    # Apply RMSNorm to input
    hidden_states = self.input_norm(hidden_states)

    # Apply MultiHeadAttention with KVCache
    hidden_states = MultiHeadAttention(
      embed_dim=self.dim,
      num_heads=self.heads,
      num_kv_heads=self.num_kv_heads,
      head_dim=self.head_dim,
      q_proj=self.q_proj,
      k_proj=self.k_proj,
      v_proj=self.v_proj,
      output_proj=self.output_proj,
      pos_embeddings=self.rotary_pos_emb,
      q_norm=self.q_norm,
      k_norm=self.q_norm,
      kv_cache=kv_cache,  # Passed during the forward call
      max_seq_len=self.max_seq_len,
      is_causal=True,
      attn_dropout=self.attention_dropout
    )(
      x=hidden_states,
      mask=attention_mask,
      input_pos=position_ids
    )

    # Residual connection
    hidden_states = residual + hidden_states
    residual = hidden_states
    # Post attention normalization
    hidden_states = self.post_norm(hidden_states)
    # Feed-forward network with MLP and residual connection
    hidden_states = self.mlp(hidden_states)
    hidden_states = hidden_states + residual

    return hidden_states, kv_cache

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
    self.padding_idx = config.get("pad_token_id")
    self.device_map="any"

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
  ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[KVCache]]:
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
        dtype=input_embeds.dtype
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

    # Reshape hidden_states to (batch_size, seq_len, num_heads, head_dim)
    batch_size, seq_len, _ = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

    # Reshape position_ids to match (batch_size, seq_len)
    if position_ids.dim() != 2:
      position_ids = position_ids.squeeze(0)

    print(f"hidden_states: {hidden_states.shape}")
    print(f"position_ids: {position_ids.shape}")

    # Apply rotary positional embeddings
    position_embeddings = self.rotary_pos_emb(
      hidden_states,
      input_pos=position_ids
    )

    print(f"position_embeddings: {position_embeddings.shape}")

    # Reshape back to (batch_size, seq_len, hidden_size)
    hidden_states = hidden_states.view(batch_size, seq_len, self.hidden_size)
    print(f"hidden_states: {hidden_states.shape}")

    # create 4d causal mask
    causal_mask = None
    if attention_mask is not None:
      causal_mask = create_4d_causal_attention_mask(
        attention_mask=attention_mask,
        seq_len=hidden_states.shape[1],
        target_len=attention_mask.shape[-1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        cache_pos=cache_position,
        batch_size=hidden_states.size(0)
      )

      print(f"attention_mask: {attention_mask.shape}")
      print(f"causal_mask: {causal_mask.shape}")

    # Forward pass through layers with KVCache
    for layer_idx in range(self.shard.end_layer, self.shard.start_layer):
      print(f"forward layer #{layer_idx}")
      encoder_layer = self.layers[layer_idx]
      print(f"encoder_layer\n{encoder_layer}")
      layer_hidden_state, layer_kv_cache = self.layers[layer_idx](
        hidden_states=hidden_states,
        kv_cache=past_kv_cache,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings
      )

      hidden_states = layer_hidden_state
      past_kv_cache = layer_kv_cache

    # Apply final layer normalization
    hidden_states = self.norm(hidden_states)

    # Compute logits if at end layer
    if self.shard.is_last_layer():
      logits = self.to_logits(hidden_states)
    else:
      logits = None

    if logits is None:
      return logits, hidden_states, past_kv_cache
    else:
      return logits, None, past_kv_cache
