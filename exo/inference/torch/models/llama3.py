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
from exo.inference.torch.models.llm_utils import MultiLayerPreceptron, RMSNorm
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
    print(f"current_pos: {current_pos}\nk_shape: {k_shape}")
    if current_pos <= k_shape:
      print("====== MAX CACHE REACHED CLEAR ==============")
      return True

    return False

  def forward(
    self,
    tokens: torch.Tensor,
    *,
    mask: Optional[_MaskType] = None,
    input_pos: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, List[torch.Tensor]]:
    # Determine the type of input and shape
    if DEBUG >= 4:
      print("forward called")
      print(f"tokens [{tokens.shape}]: {tokens}")
      print(f"mask: {mask}")

    if tokens.ndim == 3:
      h = tokens  # Use directly as hidden states
    else:
      h = self.tok_embeddings(tokens)  # Apply token tok_embeddings

      seq_len = h.shape[1]

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
      with torch.no_grad():
        if layer.caches_are_enabled():
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

    # Apply normalization
    h = self.norm(h)

    # Handle chunked output if needed
    if self.num_output_chunks > 0:
      output = self.chunked_output(h)
    else:
      output = self.output(h).float()

    # Return list if hidden states are requested
    output = [hidden[-1], output] if hidden else output

    if DEBUG >= 4:
      print(f"\n\noutput {output}\n\n")

    return output


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

    mlp = MultiLayerPreceptron(config["embed_dim"], config["intermediate_dim"], config["hidden_act"])

    layer = ttm.TransformerSelfAttentionLayer(
      attn=self_attn,
      mlp=mlp,
      sa_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
      mlp_norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    )

    layers[i] = layer

  #for i in range(len(layers)):
  #  print(f"layers[{i}]: {layers[i]}")
  layers = nn.ModuleList(layers)
  tok_embeddings = nn.Embedding(config["vocab_size"], config["embed_dim"])
  output_proj = ttm.TiedLinear(tok_embeddings)
  # output_proj = nn.Linear(
  #   config["embed_dim"],
  #   config["vocab_size"],
  #   bias=config["attn_bias"],
  # )

  return ShardTransformerDecoder(
    tok_embeddings=tok_embeddings,
    shard=shard,
    layers=layers,
    max_seq_len=config["max_seq_len"],
    num_heads=config["num_heads"],
    head_dim=config["head_dim"],
    norm=RMSNorm(config["embed_dim"], eps=config["norm_eps"]),
    output=output_proj,
    num_layers=config["num_layers"],
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
  def __init__(self, config: dict, shard: Shard, device: Optional[torch.device] = None, use_cache: Optional[bool] = False):
    super(ShardedLlamaModel, self).__init__()

    self.shard = shard
    self.config = config
    self.dtype = torch.float16
    self.device = device if device is not None else torch.device("cpu")
    self.max_seq_len = self.config["max_seq_len"]
    self.use_cache = use_cache

    # pad_id maually set as same in all llama models
    self.pad_id = 128004  # from <|finetune_right_pad_id|>

    with torch.no_grad():
      self.model = LlamaModel(config, self.shard).to(dtype=self.dtype, device=self.device)

    if DEBUG >= 8:
      print(f"model loaded: {self.model}\n")
      print(f"device: {self.device}\n")

  def generate(self, tokens: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
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

    curr_masks = None
    input_pos = None

    if tokens is not None:
      if tokens.ndim == 1:
        tokens = tokens.view(1, -1).to(device=self.device)

      bsz, tokens_length = tokens.size()

      # using self.max_seq_len will take up alot of VRAM
      total_response_length = tokens_length + self.max_seq_len

      # setup cache
      if not self.model.caches_are_enabled() and self.use_cache:
        with self.device:
          self.model.setup_caches(bsz, self.dtype, decoder_max_seq_len=tokens.numel() + self.max_seq_len)

      if not self.shard.is_last_layer():
        self.model.output_hidden_states = [self.shard.end_layer]

      resp_max_seq_len = total_response_length if not self.model.caches_are_enabled() else self.model.decoder_max_cache_seq_len

      # clone tokens
      generated_tokens = tokens.clone().to(device=self.device)

      # masking for proper attention
      padding_masks = generated_tokens != self.pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(padding_masks, (0, self.max_seq_len), value=True)

        masks = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=resp_max_seq_len)

        input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        masks = torch.tril(torch.ones(
          total_response_length,
          resp_max_seq_len if resp_max_seq_len is not None else total_response_length,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      if self.model.caches_are_enabled():
        curr_masks = masks[:, :tokens_length]
      else:
        curr_masks = masks[:, :tokens_length, :tokens_length]

      input_pos = input_pos[:, :tokens_length].squeeze()

    if DEBUG >= 4:
      print("model_input")
      if tokens is not None:
        print(f"tokens: {tokens} - {tokens.device}")
        print(f"mask: {curr_masks} - {curr_masks.device}")
        print(f"input_pos: {input_pos} - {input_pos.device}")

      if hidden_state is not None:
        print(f"hidden_state: {hidden_state} - {hidden_state.device}")

    model_output = self.model(
      tokens=hidden_state if hidden_state is not None else tokens,
      mask=curr_masks,
      input_pos=input_pos,
    )

    if DEBUG >= 4:
      print(f"model_output\n{model_output}")

    if isinstance(model_output, list):
      model_logits = model_output[1]
      model_output.pop()  # remove logits
      model_hs = model_output[0]  # get last hidden state
    else:
      model_logits = model_output
      model_hs = None

    return model_hs, model_logits
