"""
llama3 model

Written with pytorch using torchtune and other methods
"""
from typing import Optional, Any, Tuple, List

import torch
import torch.nn as nn
import torchtune.modules as ttm
import torchtune.generation as ttg
from torchtune.models.llama3_1 import Llama3ScaledRoPE

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
    max_seq_len: int = 2048
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
          max_seq_len=max_seq_len, #self.config['max_position_embeddings']
        )

    # Apply RMSNorm to input
    hidden_states = self.input_layernorm(hidden_states)
    print(f"self.input_layernorm(hidden_states) {hidden_states.shape}")

    # get causal mask from attention mask
    causal_mask = ttg.get_causal_mask_from_padding_mask(
      attention_mask.bool(),
      max_seq_len
    )

    print(f"causal_mask: {causal_mask.shape}")

    # get position_ids from attention mask
    position_ids = ttg.get_position_ids_from_padding_mask(
      attention_mask.bool()
    )

    print(f"position_ids: {position_ids.shape}")

    hidden_states = self.self_attn(
      x=hidden_states,
      y=hidden_states,
      mask=causal_mask,
      #input_pos=position_ids
    )

    # Residual connection
    print(f"hidden_states: {hidden_states.shape}")
    # Post attention normalization
    hidden_states = self.post_attention_layernorm(hidden_states)
    # Feed-forward network with MLP and residual connection
    hidden_states = self.mlp(hidden_states)

    return hidden_states

def LlamaModel(
  config: dict,
  shard: Shard,
  is_causal: bool=True,
  max_seq_len: int=4096
):
  """
  LlamaModel using torchtune
  """
  # Load configurations from config
  rope_scaling = config.get("rope_scaling")
  hidden_head_dim = config["hidden_size"] // config["num_attention_heads"]

  # Model layers and methods, order matters
  embed_tokens = nn.Embedding(
    config["vocab_size"],
    config["hidden_size"]
  )

  layers = []
  for _ in range(shard.n_layers):
    pos_embeddings = Llama3ScaledRoPE(
      dim=hidden_head_dim,
      max_seq_len=max_seq_len,
      base=config.get('rope_theta', 10000),
      scale_factor=rope_scaling['factor'] if rope_scaling else 32
    )

    self_attn = ttm.MultiHeadAttention(
      embed_dim=config["hidden_size"],
      num_heads=config["num_attention_heads"],
      num_kv_heads=config["num_key_value_heads"],
      head_dim=hidden_head_dim,
      q_proj=nn.Linear(
        config["hidden_size"],
        config["num_attention_heads"] * config["head_dim"],
        bias=config.get('attention_bias', False)
      ),
      k_proj = nn.Linear(
        config["hidden_size"],
        config["num_key_value_heads"] * config["head_dim"],
        bias=config.get('attention_bias', False)
      ),
      v_proj = nn.Linear(
        config["hidden_size"],
        config["num_key_value_heads"] * config["head_dim"],
        bias=config.get('attention_bias', False)
      ),
      output_proj=nn.Linear(
        config["hidden_size"],
        config["hidden_size"],
        bias=config.get('attention_bias', False)
      ),
      max_seq_len=max_seq_len,
      is_causal=is_causal,
      attn_dropout=config.get('attention_dropout', 0.0),
      pos_embeddings=pos_embeddings
    )

    mlp = MultiLayerPreceptron(
      config["hidden_size"],
      config['intermediate_size'],
      'silu'
    )

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
      mlp_norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
    )

    layers.append(layer)

  return ttm.TransformerDecoder(
    tok_embeddings=embed_tokens,
    layers=nn.ModuleList(layers),
    max_seq_len=max_seq_len,
    num_heads=config["num_attention_heads"],
    head_dim=config["head_dim"],
    norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
    output=nn.Linear(config["hidden_size"], config["vocab_size"]),
    num_layers=shard.n_layers,
    #output_hidden_states=list(range(shard.start_layer, shard.end_layer))
  )

class ShardedLlamaModel(nn.Module):
  def __init__(self,
    config: dict,
    shard: Shard,
    tokenizer: Any,
    device: torch.device=torch.device("cpu"),
    hidden_states: Optional[torch.Tensor] = None,
    is_causal=True
  ):
    super(ShardedLlamaModel, self).__init__()

    self.tokenizer = tokenizer
    self.shard = shard
    self.config = config
    self.model = LlamaModel(config, shard, is_causal)
    self.device = device

  def generate(
    self,
    input_tensor: torch.Tensor,
    max_seq_len: int=4096
  ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      input (torch.Tensor) - tokens if initial first layer input and hidden states after
      max_seq_len (int) - Max sequence length of generation, default 4096
    """
    self.model.output_hidden_states = list(range(self.shard.start_layer, self.shard.end_layer))

    if self.shard.is_first_layer():
      tokens = input_tensor

      if tokens.ndim == 1:
        tokens = tokens.view(1, -1)

      _, tokens_length = tokens.size()
      total_response_length = tokens_length + max_seq_len
      resp_max_seq_len = (
        total_response_length
        if not self.model.caches_are_enabled()
        else self.model.decoder_max_cache_seq_len
      )

      # masking for proper attention
      padding_masks = tokens != self.tokenizer.pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, max_seq_len),
          value=True
        )

        masks = ttg.get_causal_mask_from_padding_mask(
          padding_masks,
          target_seq_len=resp_max_seq_len
        )

        input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        masks = torch.tril(
          torch.ones(
            total_response_length,
            resp_max_seq_len if resp_max_seq_len is not None else max_seq_len,
            dtype=torch.bool,
            device=tokens.device,
          )
        ).unsqueeze(0)

      input_pos = torch.arange(
        0, total_response_length, device=tokens.device
      ).unsqueeze(0)

      if self.model.caches_are_enabled():
        curr_masks = masks[:, :tokens_length]
      else:
        curr_masks = masks[:, :tokens_length, :tokens_length]

      model_output = self.model(
        tokens=tokens,
        mask=curr_masks,
        input_pos=input_pos[:, :tokens_length].squeeze()
      )

      model_logits = model_output[-1]
      model_output.pop() # remove logits
      model_hs = model_output # hidden states

      return model_hs, model_logits
    else:
      return None, None
