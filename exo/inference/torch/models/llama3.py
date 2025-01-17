"""
llama3 model

Written with pytorch using torchtune and other methods
"""
import re

from typing import Optional, Any, Tuple, List, Union, Callable

import torch
import torch.nn as nn
import torchtune.modules as ttm
import torchtune.generation as ttg

from torchtune.modules.attention_utils import _MaskType
from torchtune.modules import RMSNorm
# llama3 torchtune
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
# from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from exo.inference.shard import Shard
from exo.inference.torch.models.llm_utils import (
  llama3_mlp,
  MultiLayerPreceptron,
  # RMSNorm,
)

from exo.helpers import DEBUG


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

  def setup_caches(
    self,
    batch_size: int,
    dtype: torch.dtype,
    *,
    encoder_max_seq_len: Optional[int] = None,
    decoder_max_seq_len: Optional[int] = None,
  ):
    """
    modified version for shard

    assume just decoder layers
    """
    if decoder_max_seq_len is not None:
      self.decoder_max_cache_seq_len = decoder_max_seq_len
    else:
      self.decoder_max_cache_seq_len = self.max_seq_len

    for layer in self.layers:
      if layer is not None:
        layer.setup_caches(
          batch_size,
          dtype,
          encoder_max_seq_len=self.encoder_max_cache_seq_len,
          decoder_max_seq_len=self.decoder_max_cache_seq_len,
        )

  def caches_are_enabled(self) -> bool:
    """
    modified version for shard
    """
    if self.layers[0] is not None:
      return self.layers[0].caches_are_enabled()
    else:
      for layer in self.layers:
        if layer is not None:
          return layer.caches_are_enabled()

    return False

  def reset_caches(self):
    torch.cuda.empty_cache()

    for layer in self.layers:
      if layer is not None:
        layer.reset_cache()

  def check_maxed_cache(self, tokens: torch.Tensor) -> bool:
    """
    Check if cached is maxed out and needs to be reset
    """
    active_layers = [x for x in self.layers if x is not None]
    kv_cache = active_layers[0].attn.kv_cache
    current_pos = kv_cache.cache_pos[0] + tokens.numel() + self.max_seq_len
    k_shape = kv_cache.k_cache.shape[2]

    if DEBUG >= 4:
      print(f"cache current_pos: {current_pos}\nk_shape: {k_shape}")

    if current_pos <= k_shape:
      if DEBUG >= 4:
        print("============ MAX CACHE REACHED CLEAR ==============")

      return True

    return False

  def forward(
    self,
    tokens: torch.Tensor,
    *,
    mask: Optional[_MaskType] = None,
    input_pos: Optional[torch.Tensor] = None,
    hidden_state: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, List[torch.Tensor]]:
    # Determine the type of input and shape
    if DEBUG >= 4:
      print("forward called")
      print(f"tokens [{tokens.shape}]: {tokens}")
      print(f"mask: {mask}")
      print(f"input_pos: {input_pos}")

    if hidden_state is not None:
      h = hidden_state  # Use directly as hidden states
    else:
      seq_len = tokens.shape[1]

      h = self.tok_embeddings(tokens)  # Apply token tok_embeddings

      self._validate_inputs(
        seq_len,
        mask=mask,
        input_pos=input_pos,
      )

    # Initialize a list to capture hidden states if requested
    # for captured hidden states
    hidden = []

    for i in range(self.shard.start_layer, self.shard.end_layer + 1):
      layer = self.layers[i]

      if DEBUG >= 8:
        print(f"\nhidden layer in H[{i}]\n{h}")
        print(f"\nmask\n{mask}\ninput_pos\n{input_pos}")
        print(f"\noutput_hidden_states\n{self.output_hidden_states}\n")

      # Process through each transformer layer
      # with torch.no_grad():
      if layer.caches_are_enabled():
        self.check_maxed_cache(tokens=h)
        try:
          h = layer(
            h,
            mask=mask,
            input_pos=input_pos,
          )
        except AssertionError:
          # assume due to cache
          self.reset_caches()

          h = layer(
            h,
            mask=mask,
            input_pos=input_pos,
          )

      else:
        h = layer(h)

      if i in self.output_hidden_states:
        hidden.append(h)

      if DEBUG >= 8:
        print(f"\nhidden layer out H[{i}]->H[{i + 1}]\n{h}\n")

    if self.shard.is_last_layer():
      # Apply normalization
      h = self.norm(h)

      # Handle chunked output if needed
      output = self.output(h).float()

      if DEBUG >= 4:
        print(f"\n\noutput {output}\n\n")

      return output
    else:
      if DEBUG >= 4:
        print(f"\n\nhidden output {hidden[-1]}\n\n")

      return hidden[-1]


def LlamaModel(config: dict, shard: Shard):
  """
  LlamaModel using torchtune
  """
  # rope scaling config
  scale_factor = 32
  if config["rope_scaling"] is not None:
    scale_factor = config["rope_scaling"].get("factor", 32)

  rope = Llama3ScaledRoPE(
    dim=config["head_dim"],
    max_seq_len=config["max_seq_len"],
    base=config["rope_base"],
    scale_factor=scale_factor,
  )

  # hack to align sharded weights with layers
  # fill unused layer positions with None
  layers = [None for _ in range(shard.n_layers)]

  # build layers
  for i in range(shard.start_layer, shard.end_layer + 1):
    self_attn = ttm.MultiHeadAttention(
      embed_dim=config["embed_dim"],
      num_heads=config["num_heads"],
      num_kv_heads=config["num_kv_heads"],
      head_dim=config["head_dim"],
      q_proj=nn.Linear(
        config["embed_dim"],
        config["num_heads"]*config["head_dim"],
        bias=config["attn_bias"],
      ),
      k_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
        bias=config["attn_bias"],
      ),
      v_proj=nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"]*config["head_dim"],
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

    mlp = llama3_mlp(
      dim=config["embed_dim"],
      hidden_dim=config["intermediate_dim"],
    )

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
      mlp_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    )

    layers[i] = layer

  layers = nn.ModuleList(layers)
  tok_embeddings = nn.Embedding(config["vocab_size"], config["embed_dim"])

  if len(re.findall(r"3\.2", shard.model_id)) > 0:
    output_proj = ttm.TiedLinear(tok_embeddings)
  else:
    output_proj = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

  norm = RMSNorm(config["embed_dim"], eps=config["norm_eps"])

  return ShardTransformerDecoder(
    tok_embeddings=tok_embeddings,
    shard=shard,
    layers=layers,
    max_seq_len=config["max_seq_len"],
    num_heads=config["num_heads"],
    head_dim=config["head_dim"],
    norm=norm,
    output=output_proj,
    num_layers=config["num_layers"],
  )


class ShardedLlamaModel(nn.Module):
  def __init__(
    self,
    config: dict,
    shard: Shard,
    device: Optional[torch.device] = None,
    use_cache: Optional[bool] = False,
    max_generated_tokens: int = 1024,
  ):
    super(ShardedLlamaModel, self).__init__()

    self.shard = shard
    self.config = config
    self.dtype = torch.float16
    self.device = device if device is not None else torch.device("cpu")
    self.max_seq_len = self.config["max_seq_len"]
    self.use_cache = use_cache

    # pad_id maually set as same in all llama models
    self.pad_id = 128004  # from <|finetune_right_pad_id|>

    self.model = LlamaModel(config, self.shard).to(dtype=self.dtype, device=self.device)

    if DEBUG >= 4:
      print("ShardedLlamaModel called")
      print(f"self.model {self.model}")

    # keep track of current position in generation
    self.max_generated_tokens = max_generated_tokens
    self.curr_pos = 0
    self.masks = None
    self.curr_masks = None
    self.input_pos = None
    self.curr_input_pos = None

  def generate(
    self,
    tokens: Optional[torch.Tensor] = None,
    hidden_state: Optional[torch.Tensor] = None,
  ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Generate logits and/or hidden_states from llama model

    Args
      tokens (torch.Tensor) - tokens from prompt tokenization and generation
      hidden_state (torch.Tensor, optional) - hidden state from last activated hidden layer, if any
    """
    if DEBUG >= 4:
      print("generate called")
      print(f"tokens: {tokens}")
      print(f"hidden_state: {hidden_state}")
      print(f"curr_pos: {self.curr_pos}")

    model_hs = None
    model_logits = None

    bsz, tokens_length = tokens.size()

    if tokens_length > 1:
      tokens = tokens.view(1, -1).to(device=self.device) if tokens.ndim == 1 else tokens

      self.curr_pos = tokens_length

      # using self.max_seq_len will take up alot of VRAM
      total_response_length = tokens_length + self.max_generated_tokens

      # setup cache
      if not self.model.caches_are_enabled() and self.use_cache:
        with self.device:
          self.model.setup_caches(
            bsz,
            self.dtype,
            decoder_max_seq_len=tokens.numel() + self.max_generated_tokens,
          )

      if not self.shard.is_last_layer():
        self.model.output_hidden_states = [self.shard.end_layer]

      if not self.model.caches_are_enabled():
        max_seq_len = total_response_length
      else:
        max_seq_len = self.model.decoder_max_cache_seq_len

      # masking for proper attention
      padding_masks = tokens != self.pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.max_generated_tokens),
          value=True,
        )

        self.masks = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.masks = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      if self.model.caches_are_enabled():
        self.curr_masks = self.masks[:, :tokens_length]
      else:
        self.curr_masks = self.masks[:, :tokens_length, :tokens_length]

      self.curr_input_pos = self.input_pos[:, :tokens_length].squeeze()
    else:
      if self.model.caches_are_enabled():
        self.curr_input_pos = self.input_pos[:, self.curr_pos].contiguous()
        self.curr_masks = self.masks[:, self.curr_pos, None, :].contiguous()
      else:
        self.curr_input_pos = self.input_pos[:, :self.curr_pos + 1]
        self.curr_masks = self.masks[:, :self.curr_pos + 1, :self.curr_pos + 1]

    if DEBUG >= 4:
      print("model_input")
      if tokens is not None:
        print(f"tokens: {tokens}")
      if hidden_state is not None:
        print(f"hidden_state: {hidden_state}")
      print(f"mask: {self.curr_masks}")
      print(f"input_pos: {self.curr_input_pos}")

    model_output = self.model(
      tokens=tokens,
      mask=self.curr_masks,
      input_pos=self.curr_input_pos,
      hidden_state=hidden_state,
    )

    self.curr_pos += 1

    if DEBUG >= 4:
      print(f"model_output\n{model_output}")

    if self.shard.is_last_layer():
      model_logits = model_output
    else:
      model_hs = model_output

    return model_hs, model_logits
