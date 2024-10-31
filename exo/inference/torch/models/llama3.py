"""
llama3 model

Written with pytorch using torchtune and other methods
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune.modules import (
  KVCache,
  RMSNorm
)

from exo.inference.shard import Shard
from exo.inference.torch.models.llm_utils import (
  MultiLayerPreceptron,
  #MultiHeadAttention,
  SDPAttention,
  RotaryEmbedding,
  create_4d_causal_attention_mask
)

class LlamaBlock(nn.Module):
  """
  Encoder block class for the LLaMA model
  """
  def __init__(
    self,
    dim,
    mlp,
    self_attn,
    rms_norm_eps=1e-6
  ):
    super(LlamaBlock, self).__init__()
    self.self_attn = self_attn
    self.mlp = mlp
    self.input_layernorm = RMSNorm(dim, eps=rms_norm_eps)
    self.post_attention_norm = RMSNorm(dim, eps=rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    kv_cache: Optional[KVCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None
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
    hidden_states = self.input_layernorm(hidden_states)
    print(f"self.input_layernorm(hidden_states) {hidden_states.shape}")

    #batch_size, seq_len, _ = hidden_states.shape
    #hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).squeeze()
    #print(f"hidden_states: {hidden_states.shape}")

    # Apply MultiHeadAttention with KVCache
    hidden_states, kv_cache = self.self_attn(
      hidden_states=hidden_states,
      position_ids=position_ids,
      attention_mask=attention_mask,
      position_embeddings=position_embeddings
    )

    # Residual connection
    hidden_states = residual + hidden_states
    residual = hidden_states
    print(f"hidden_states: {hidden_states.shape}")
    print(f"residual: {residual.shape}")
    # Post attention normalization
    hidden_states = self.post_attention_norm(hidden_states)
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

    # Model layers and methods, order matters
    self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([
      LlamaBlock(
        dim=self.hidden_size,
        rms_norm_eps=self.rms_norm_eps,
        self_attn=SDPAttention(
          hidden_size=self.hidden_size,
          num_heads=self.num_heads,
          num_kv_heads=self.num_kv_heads,
          head_dim=self.hidden_size // self.num_heads,
          is_causal=True,
          attention_dropout=self.attention_dropout,
          rotary_emb=RotaryEmbedding(
            self.head_dim
          ),
          attention_bias=config.get('attention_bias', False)
        ),
        mlp=MultiLayerPreceptron(
          input_dim=self.hidden_size,
          hidden_dim=self.intermediate_size,
          activation=self.config.get("hidden_act", "silu"),
          use_bias=self.config.get("mlp_bias", False)
        ),
      ) for _ in range(self.num_layers)
    ])
    self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
    self.rotary_emb = RotaryEmbedding(
      self.head_dim
    )
    self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
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
        - pred_score (Optional[torch.Tensor]): Prediction scores from lm_head of model.
        - hidden_states (Optional[torch.Tensor]): Hidden states from each layer
        - past_kv_cache (KVCache): Updated KVCache object.
    """
    batch_size, seq_len = input_ids.shape

    # Create initial embeddings
    input_embeds = self.embed_tokens(input_ids)

    ## Initialize or use the provided KVCache
    #if past_kv_cache is None:
    #  past_kv_cache = KVCache(
    #    batch_size=batch_size,
    #    max_seq_len=self.max_position_embeddings,
    #    num_heads=self.num_heads,
    #    head_dim=self.head_dim,
    #    dtype=input_embeds.dtype
    #  )

    # Initialize position IDs if not provided
    if cache_position is None:
      past_seen_tokens = past_kv_cache.size if past_kv_cache is not None else 0
      cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + input_embeds.shape[1],
        device=input_ids.device
      )
      #.unsqueeze(0).expand(batch_size, -1)

      print(f"cache_position: {cache_position.shape}")

    if position_ids is None:
      #position_ids = cache_position.unsqueeze(0)
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)

    # cache based input generation
    if past_kv_cache is not None:
      hidden_states = input_embeds[:, -cache_position.shape[0]:]
    else:
      hidden_states = input_embeds

    print(f"LM hidden_states: {hidden_states.shape}")

    # Reshape hidden_states to (batch_size, seq_len, num_heads, head_dim)
    batch_size, seq_len, _ = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

    # Reshape position_ids to match (batch_size, seq_len)
    if position_ids.dim() != 2:
      position_ids = position_ids.squeeze(0)

    print(f"hidden_states: {hidden_states.shape}")
    print(f"position_ids: {position_ids.shape}")

    # Apply rotary positional embeddings
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Reshape back to (batch_size, seq_len, hidden_size)
    print(f"hidden_size: {self.hidden_size}")
    hidden_states = hidden_states.view(batch_size, seq_len, self.hidden_size)
    print(f"hidden_states: {hidden_states.shape}")

    # create/update 4d causal mask
    seq_len = input_embeds.shape[1]

    if past_kv_cache is not None:
      target_len = past_kv_cache.size + seq_len + 1
    else:
      target_len = seq_len + 1
    attention_mask = create_4d_causal_attention_mask(
      attention_mask=attention_mask,
      seq_len=seq_len,
      target_len=target_len,
      dtype=input_embeds.dtype,
      device=input_embeds.device,
      cache_pos=cache_position,
      batch_size=input_embeds.size(0)
    )

    print(f"attention_mask: {attention_mask.shape}")

    # Forward pass through layers with KVCache
    for layer_idx in range(self.shard.start_layer, self.shard.end_layer):
      print(f"forward layer #{layer_idx}")
      encoder_layer = self.layers[layer_idx]
      print(f"encoder_layer\n{encoder_layer}")
      layer_hidden_state, layer_kv_cache = self.layers[layer_idx](
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings
      )

      hidden_states = layer_hidden_state
      past_kv_cache = layer_kv_cache

      print(f"layer_kv_cache: {layer_kv_cache.size}")

    # Compute prediction score from lm head if at end layer
    if self.shard.is_last_layer():
      # Apply final layer normalization
      hidden_states = self.norm(hidden_states)
      pred_score = self.lm_head(hidden_states[:, -1:, :])
    else:
      pred_score = None

    print(f"end attention_mask: {attention_mask.shape}")

    if pred_score is None:
      return pred_score, hidden_states, past_kv_cache
    else:
      return pred_score, None, past_kv_cache
