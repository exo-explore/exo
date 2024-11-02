"""
llama3 model

Written with pytorch using torchtune and other methods
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchtune.modules as ttm

from exo.inference.shard import Shard
from exo.inference.torch.models.llm_utils import (
  MultiLayerPreceptron,
  RMSNorm
)

class LlamaBlock(nn.Module):
  """
  Encoder block class for the LLaMA model
  """
  def __init__(
    self,
    config,
    mlp,
    self_attn,
    rms_norm_eps=1e-6
  ):
    super(LlamaBlock, self).__init__()
    self.config = config
    self.self_attn = self_attn
    self.mlp = mlp
    self.input_layernorm = RMSNorm(self.config['hidden_size'], eps=rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(self.config['hidden_size'], eps=rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """
    Forward pass with integrated attention, resnet and key-value caching.

    Args:
      hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
      position_ids (Optional[torch.Tensor]): Position IDs tensor of shape (batch_size, seq_len).

    Returns:
      Tuple[torch.Tensor, KVCache]:
        - Output tensor of shape (batch_size, seq_len, dim).
    """
    if isinstance(self.self_attn, ttm.MultiHeadAttention):
      if self.self_attn.kv_cache is None:
        # setup cache
        self.self_attn.setup_cache(
          batch_size=hidden_states.size(0),
          dtype=hidden_states.dtype,
          max_seq_len=2048, #self.config['max_position_embeddings']
        )

    # Reshape `attention_mask` to match the expected shape: [batch_size, seq_len, seq_len]
    if attention_mask is not None:
      attention_mask = attention_mask[:, None, :].expand(-1, hidden_states.size(1), -1).float()
      print(f"reshaped attention_mask: {attention_mask.shape}")

    # setting up resnet
    residual = hidden_states

    # Apply RMSNorm to input
    hidden_states = self.input_layernorm(hidden_states)
    print(f"self.input_layernorm(hidden_states) {hidden_states.shape}")

    hidden_states = self.self_attn(
      x=hidden_states,
      #mask=attention_mask,
      input_pos=position_ids
    )

    # Residual connection
    hidden_states = residual + hidden_states
    residual = hidden_states
    print(f"hidden_states: {hidden_states.shape}")
    print(f"residual: {residual.shape}")
    # Post attention normalization
    hidden_states = self.post_attention_layernorm(hidden_states)
    # Feed-forward network with MLP and residual connection
    hidden_states = self.mlp(hidden_states)
    hidden_states = hidden_states + residual

    return hidden_states

class LlamaModel(nn.Module):
  """
  LlamaModel is a pure PyTorch implementation of the LLaMA architecture
  """

  def __init__(self, config: dict, shard: Shard, is_causal=True):
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
    self.attention_bias = config.get('attention_bias', False)
    self.padding_idx = config.get("pad_token_id")
    self.has_lm_head_weight = False

    # Model layers and methods, order matters
    self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx) 
    self.layers = nn.ModuleList([
      LlamaBlock(
        config=self.config,
        rms_norm_eps=self.rms_norm_eps,
        self_attn=ttm.MultiHeadAttention(
          embed_dim=self.hidden_size,
          num_heads=self.num_heads,
          num_kv_heads=self.num_heads,
          head_dim= self.hidden_size // self.num_heads,
          q_proj=nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias),
          k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias),
          v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias),
          output_proj=nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias),
          max_seq_len=2048, #self.max_position_embeddings,
          is_causal=is_causal,
          attn_dropout=self.attention_dropout
        ),
        mlp=MultiLayerPreceptron(
          input_dim=self.hidden_size,
          hidden_dim=self.intermediate_size,
          activation=self.config.get("hidden_act", "silu"),
          use_bias=self.config.get("mlp_bias", False)
        )
      ) for _ in range(self.num_layers)
    ])
    self.norm = RMSNorm(hidden_size=self.hidden_size, eps=self.rms_norm_eps)
    self.rotary_emb = ttm.RotaryPositionalEmbeddings(
      dim=self.hidden_size // self.num_heads,
      max_seq_len=2048, #self.max_position_embeddings,
      base=self.config.get('rope_theta', 10000)
    )
    self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
  ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    _, seq_len = input_ids.shape

    input_embeds = self.embed_tokens(input_ids)

    if position_ids is None:
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      position_ids = position_ids[:, -seq_len:]

    print(f"LM input_embeds: {input_embeds.shape}")
    print(f"LM attention_mask: {attention_mask.shape}")

    hidden_states = input_embeds

    for layer_idx in range(self.shard.start_layer, self.shard.end_layer):
      #print(f"forward layer #{layer_idx}")
      #print(f"{self.layers[layer_idx]}")
      hidden_states = self.layers[layer_idx](
        hidden_states=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
      )

    if self.shard.is_last_layer():
      pred_score = self.lm_head(self.norm(hidden_states)[:, -1:, :])

      return pred_score, None

    return None, hidden_states
