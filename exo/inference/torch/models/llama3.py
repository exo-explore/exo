"""
llama3 model

Written with pytorch using torchtune and other methods
"""
from typing import Optional, Any, Tuple, List, Union, Callable

import torch
import torch.nn as nn
import torchtune.modules as ttm
import torchtune.generation as ttg
from torchtune.models.llama3_1 import Llama3ScaledRoPE
from torchtune.modules.attention_utils import _MaskType

from exo.inference.shard import Shard
from exo.inference.torch.models.llm_utils import (
  MultiLayerPreceptron,
  RMSNorm
)

class ShardTransformerDecoder(ttm.TransformerDecoder):
  def __init__(
    self,
    *,
    shard: Shard,
    tok_embeddings: nn.Embedding,
    layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    norm: nn.Module,
    output: Union[nn.Linear, Callable],
    num_layers: Optional[int] = None,
    output_hidden_states: Optional[List[int]] = None
  ):
    super().__init__(
      tok_embeddings=tok_embeddings,
      layers=layers,
      max_seq_len=max_seq_len,
      num_heads=num_heads,
      head_dim=head_dim,
      norm=norm,
      output=output,
      num_layers=num_layers,
      output_hidden_states=output_hidden_states,
    )

    self.shard = shard

  def forward(
    self,
    tokens: torch.Tensor,
    *,
    mask: Optional[_MaskType] = None,
    encoder_input: Optional[torch.Tensor] = None,
    encoder_mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, List[torch.Tensor]]:
    # for captured hidden states 
    hidden = []

    # Determine the type of input and shape
    print(f"tokens.ndim: {tokens.ndim}")
    if tokens.ndim == 3:
      h = tokens  # Use directly as hidden states
    else:
      h = self.tok_embeddings(tokens)  # Apply token tok_embeddings

      # capture tok hidden state, if needed
      if 0 in self.output_hidden_states:
        hidden.append(h)

    seq_len = h.shape[1]

    self._validate_inputs(
      seq_len,
      mask=mask,
      encoder_input=encoder_input,
      encoder_mask=encoder_mask,
      input_pos=input_pos,
    )

    # Initialize a list to capture hidden states if requested
    hidden = []
    for i in range(self.shard.start_layer, self.shard.end_layer+1):
      layer = self.layers[i] 

      # Process through each transformer layer
      h = layer(
        h,
        mask=mask,
        encoder_input=encoder_input,
        encoder_mask=encoder_mask,
        input_pos=input_pos,
      )

      # capture wanted hidden states
      if i in self.output_hidden_states:
        hidden.append(h)

      print(f"\n\n\nhidden layer H[{i}]\n{h}\n\n\n")

    # Apply normalization
    h = self.norm(h)

    # Handle chunked output if needed
    if self.num_output_chunks > 0:
        output = self.chunked_output(h)
    else:
        output = self.output(h).float()

    # Return list if hidden states are requested
    output = output if not hidden else [*hidden, output]
    print(f"\n\noutput {output}\n\n")
    return output

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

  return ShardTransformerDecoder(
    tok_embeddings=embed_tokens,
    layers=nn.ModuleList(layers),
    max_seq_len=max_seq_len,
    num_heads=config["num_attention_heads"],
    head_dim=config["head_dim"],
    norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
    output=nn.Linear(config["hidden_size"], config["vocab_size"]),
    num_layers=shard.n_layers,
    #output_hidden_states=list(range(shard.start_layer, shard.end_layer)),
    shard=shard
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
    tokens: torch.Tensor,
    hidden_state: Optional[torch.Tensor] = None,
    max_seq_len: int=4096
  ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      tokens (torch.Tensor) - tokens from prompt tokenization
      hidden_state (torch.Tensor, optional) - hidden state from last activated hidden layer, if any
      max_seq_len (int) - Max sequence length of generation, default 4096
    """
    print(self.shard)
    print(self.shard.is_last_layer())
    if not self.shard.is_last_layer():
      self.model.output_hidden_states = [self.shard.end_layer]

    if tokens.ndim == 1:
      tokens = tokens.view(1, -1)

    _, tokens_length = tokens.size()
    total_response_length = tokens_length + max_seq_len
    resp_max_seq_len = (
      total_response_length
      if not self.model.caches_are_enabled()
      else self.model.decoder_max_cache_seq_len
    )

    # clone tokens
    generated_tokens = tokens.clone()

    # masking for proper attention
    padding_masks = generated_tokens != self.tokenizer.pad_id
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
          resp_max_seq_len if resp_max_seq_len is not None else total_response_length,
          dtype=torch.bool,
          device=tokens.device,
        )
      ).unsqueeze(0)

      input_pos = torch.arange(
        0, total_response_length, device=generated_tokens.device
      ).unsqueeze(0)

    if self.model.caches_are_enabled():
      curr_masks = masks[:, :tokens_length]
    else:
      curr_masks = masks[:, :tokens_length, :tokens_length]

    if hidden_state is not None:
      #_, hs_len, _ = hidden_state.size()
      #total_hidden_length = hs_len + max_seq_len
      #hs_max_seq_len = (
      #  total_response_length
      #  if not self.model.caches_are_enabled()
      #  else self.model.decoder_max_cache_seq_len
      #)

      #hs_mask = torch.tril(
      #  torch.ones(
      #    total_hidden_length,
      #    hs_max_seq_len if hs_max_seq_len is not None else max_seq_len,
      #    dtype=torch.bool,
      #    device=tokens.device,
      #  )
      #).unsqueeze(0)

      #if self.model.caches_are_enabled():
        #hs_curr_masks = hs_mask[:, :hs_len]
      #else:
        #hs_curr_masks = hs_mask[:, :hs_len, :hs_len]

      model_output = self.model(
        tokens=hidden_state,
        mask=curr_masks,
        input_pos=input_pos[:, :tokens_length].squeeze(),
      )
    else:
      model_output = self.model(
        tokens=tokens,
        mask=curr_masks,
        input_pos=input_pos[:, :tokens_length].squeeze()
      )

    print(f"\nmodel_output: {model_output}")

    if isinstance(model_output, list):
      model_logits = model_output[-1]
      model_output.pop() # remove logits
      model_hs = model_output[-1] # get last hidden state
    else:
      model_logits = model_output
      model_hs = None

    return model_hs, model_logits
