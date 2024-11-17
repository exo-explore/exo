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
  RMSNorm,
  get_torch_dtype
)


class ShardTransformerDecoder(ttm.TransformerDecoder):
  """
  ShardTransformerDecorder
  Custom version of torchtune TransformerDecoder to allow for
  sharding of models and passing of hidden layers between shards
  """
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
    output_hidden_states: Optional[List[int]] = None,
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
    # Determine the type of input and shape
    if tokens.ndim == 3:
      h = tokens  # Use directly as hidden states
    else:
      h = self.tok_embeddings(tokens)  # Apply token tok_embeddings

      seq_len = h.shape[1]

      self._validate_inputs(
        seq_len,
        mask=mask,
        encoder_input=encoder_input,
        encoder_mask=encoder_mask,
        input_pos=input_pos,
      )

    # Initialize a list to capture hidden states if requested
    # for captured hidden states
    hidden = []

    for i in range(self.shard.start_layer, self.shard.end_layer + 1):
      layer = self.layers[i]

      print(f"\nhidden layer in H[{i}]\n{h}\nmask\n{mask}\ninput_pos\n{input_pos}\n{self.output_hidden_states}\n")

      # Process through each transformer layer
      h = layer(
        h,
        mask=mask,
        encoder_input=encoder_input,
        encoder_mask=encoder_mask,
        input_pos=input_pos,
      )

      if i in self.output_hidden_states:
        hidden.append(h)

      print(f"\nhidden layer out H[{i}]->H[{i + 1}]\n{h}\n")

    # Apply normalization
    h = self.norm(h)

    # Handle chunked output if needed
    if self.num_output_chunks > 0:
      output = self.chunked_output(h)
    else:
      output = self.output(h).float()

    # Return list if hidden states are requested
    output = [hidden[-1], output] if hidden else output
    print(f"\n\noutput {output}\n\n")
    return output

def LlamaModel(config: dict, shard: Shard):
  """
  LlamaModel using torchtune
  """
  # rope scaling config
  if config["rope_scaling"] is not None:
    scale_factor = config["rope_scaling"].get("factor", 32)

  rope = Llama3ScaledRoPE(
    dim=config["head_dim"],
    max_seq_len=config["max_seq_len"],
    base=config["rope_base"],
    scale_factor=scale_factor,
  )

  layers = []
  for _ in range(shard.n_layers):
    self_attn = ttm.MultiHeadAttention(
      embed_dim=config["embed_dim"],
      num_heads=config["num_heads"],
      num_kv_heads=config["num_kv_heads"],
      head_dim=config["head_dim"],
      q_proj=nn.Linear(
        config["embed_dim"],
        config["num_heads"] * config["head_dim"],
        bias=config["attn_bias"],
      ),
      k_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"] * config["head_dim"],
        bias=config["attn_bias"],
      ),
      v_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"] * config["head_dim"],
        bias=config["attn_bias"],
      ),
      output_proj=nn.Linear(
        config["embed_dim"],
        config["embed_dim"],
        bias=config["attn_bias"],
      ),
      max_seq_len=config["max_seq_len"],
      attn_dropout=config["attn_dropout"],
      pos_embeddings=rope,
    )

    mlp = MultiLayerPreceptron(
      config["embed_dim"],
      config["intermediate_dim"],
      config["hidden_act"]
    )

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
      mlp_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    )

    layers.append(layer)
  
  layers = nn.ModuleList(layers)
  tok_embeddings = nn.Embedding(config["vocab_size"], config["embed_dim"])
  # output_proj = ttm.TiedLinear(tok_embeddings)
  output_proj = nn.Linear(
    config["embed_dim"],
    config["vocab_size"],
    bias=config["attn_bias"],
  )

  return ShardTransformerDecoder(
    tok_embeddings=tok_embeddings,
    shard=shard,
    layers=layers,
    max_seq_len=config["max_seq_len"],
    num_heads=config["num_heads"],
    head_dim=config["head_dim"],
    norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    output=output_proj,
    num_layers=config["num_layers"]
  )

  # return ttm.TransformerDecoder(
  #   tok_embeddings=tok_embeddings,
  #   layers=layers,
  #   max_seq_len=config["max_seq_len"],
  #   num_heads=config["num_heads"],
  #   head_dim=config["head_dim"],
  #   norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
  #   output=output_proj,
  #   num_layers=config["num_layers"],
  # )


class ShardedLlamaModel(nn.Module):
  def __init__(
      self,
      config: dict,
      shard: Shard,
      tokenizer: Any,
      device: Optional[torch.device] = None,
      max_seq_len: Optional[int] = None
    ):
    super(ShardedLlamaModel, self).__init__()

    self.tokenizer = tokenizer
    self.shard = shard
    self.config = config
    self.dtype = get_torch_dtype(self.config["torch_dtype"]) if "torch_dtype" in self.config else torch.float
    self.device = device if device is not None else torch.device("cpu")
    self.use_cache = self.config.get("use_cache", False)
    self.model = LlamaModel(config, self.shard).to(dtype=self.dtype, device=self.device)
    self.max_seq_len = max_seq_len if max_seq_len is not None else 4096

  def generate(self, tokens: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      tokens (torch.Tensor) - tokens from prompt tokenization
      hidden_state (torch.Tensor, optional) - hidden state from last activated hidden layer, if any
      max_seq_len (int) - Max sequence length of generation, default 4096
    """
    print(self.shard)
    print(self.shard.is_last_layer())

    if tokens.ndim == 1:
      tokens = tokens.view(1, -1)

    bsz, tokens_length = tokens.size()

    # setup cache
    if not self.model.caches_are_enabled() and self.use_cache:
      with self.device:
        self.model.setup_caches(bsz, self.dtype, decoder_max_seq_len=self.model.decoder_max_cache_seq_len)

    if not self.shard.is_last_layer():
      self.model.output_hidden_states = [self.shard.end_layer]

    total_response_length = tokens_length + self.max_seq_len
    resp_max_seq_len = total_response_length if not self.model.caches_are_enabled() else self.model.decoder_max_cache_seq_len

    # clone tokens
    generated_tokens = tokens.clone()

    # masking for proper attention
    padding_masks = generated_tokens != self.tokenizer.pad_id
    if not padding_masks.all():
      padding_masks = torch.nn.functional.pad(padding_masks, (0, self.max_seq_len), value=True)

      masks = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=resp_max_seq_len)

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

      input_pos = torch.arange(0, total_response_length, device=generated_tokens.device).unsqueeze(0)
      
    if self.model.caches_are_enabled():
      curr_masks = masks[:, :tokens_length]
    else:
      curr_masks = masks[:, :tokens_length, :tokens_length]

    input_pos = input_pos[:, :tokens_length].squeeze()

    if hidden_state is not None:
      model_output = self.model(
        tokens=hidden_state,
        mask=curr_masks,
        input_pos=input_pos,
      )
    else:
      model_output = self.model(
        tokens=tokens,
        mask=curr_masks,
        input_pos=input_pos,
      )

    print(f"\nmodel_output: {model_output}")

    if isinstance(model_output, list):
      model_logits = model_output[1]
      model_output.pop()  # remove logits
      model_hs = model_output[0]  # get last hidden state
    else:
      model_logits = model_output
      model_hs = None

    return model_hs, model_logits
