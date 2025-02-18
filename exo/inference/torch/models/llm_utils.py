"""
Utility methods used by LLMs
"""
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable

import torch
import torch.nn as nn
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules import FeedForward, TransformerDecoder
from torchtune.models.convert_weights import hf_to_tune

from safetensors.torch import load_file as load_safetensors

from exo.helpers import DEBUG
from exo.inference.shard import Shard

# dtype string to dtype from huggingface type config.json
HF_PRECISION_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


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
    base_config = json.load(f)

    model_config = {
      "rope_scaling": base_config.get("rope_scaling"),
      "embed_dim": base_config["hidden_size"],
      "num_heads": base_config["num_attention_heads"],
      "head_dim": base_config.get(
        "head_dim",
        base_config["hidden_size"] // base_config["num_attention_heads"],
      ),  # Assuming embed_dim = hidden_size
      "num_kv_heads": base_config["num_key_value_heads"],
      "max_seq_len": base_config["max_position_embeddings"],
      "intermediate_dim": base_config["intermediate_size"],
      "attn_dropout": base_config.get("attention_dropout", 0.0),
      "norm_eps": base_config["rms_norm_eps"],
      "rope_base": base_config["rope_theta"],
      "vocab_size": base_config["vocab_size"],
      "num_layers": base_config["num_hidden_layers"],
      "attn_bias": base_config.get("attention_bias", False),
      "hidden_act": base_config.get("hidden_act", "silu"),
      "torch_dtype": HF_PRECISION_STR_TO_DTYPE.get(
        base_config.get("torch_dtype", "float16"),
        torch.float16
      )
    }

    if model_config.get("rope_scaling", None) is not None:
      model_config["rope_scaling_factor"] = model_config["rope_scaling"].get("rope_factor", 32)

    use_org_seq = bool(os.getenv("TORCH_USE_ORG_SEQ", "False").lower() == "true")
    if use_org_seq and model_config.get("rope_scaling", None) is not None:
      model_config["max_seq_len"] = model_config["rope_scaling"]["original_max_position_embeddings"]



  return model_config


def check_weights(model, state_dict):
  """
  Verifies that the weights from the state dictionary are properly loaded into the model.
  """
  model_state_dict = model.state_dict()
  for name, param in model_state_dict.items():
    if name in state_dict:
      loaded_param = state_dict[name]
      if param.shape != loaded_param.shape:
        print(f"Shape mismatch for {name}: expected {param.shape}, got {loaded_param.shape}")
      else:
        print(f"{name}: loaded correctly")

  for name in state_dict:
    if name not in model_state_dict:
      print(f"Unexpected weight {name} found in state_dict")

def load_weights_torch(cache_dir: Path, model: Any, config: Dict):
  # Load weights from safetensors files in the cache directory
  safetensors_files = list(cache_dir.glob("*.safetensors"))
  if not safetensors_files:
    raise FileNotFoundError("No safetensors files found in the cache directory.")

  # Load weights from each found safetensors file
  full_state_dict = {}
  for safetensor_file in safetensors_files:
    state_dict = load_safetensors(safetensor_file)

    if full_state_dict is not None:
      full_state_dict = full_state_dict | state_dict
    else:
      full_state_dict = state_dict

  converted_sd = hf_to_tune(
    state_dict=full_state_dict,
    num_heads=config["num_heads"],
    num_kv_heads=config["num_kv_heads"],
    dim=config["embed_dim"],
    head_dim=config["head_dim"]
  )

  model.load_state_dict(converted_sd, strict=True)

  print("\n--- checking weights ----\n")
  check_weights(model, converted_sd)

def _permute(t, n_heads: int, head_dim: int, dim: int):
  """
  Reshape weight for torchtune
  """
  return (
    t.view(n_heads, 2, head_dim // 2, dim)
    .transpose(1, 2)
    .reshape((head_dim * n_heads), dim)
  )

def load_model_weights_torchtune(
  cache_dir: Path,
  shard: Shard,
  model: Any,
  num_heads: int = 32,
  num_kv_heads: int = 32,
  dim: int = 4096,
  head_dim: int = None
):
  """
  Loads weights from huggingface and changes it to match torchtune naming structure
  """
  if head_dim is None:
    head_dim = dim // num_heads

  model_state_dict = model.state_dict()
  if DEBUG >= 8:
    for name, _ in model_state_dict.items():
      print(f"name {name}")
  # Load weights from safetensors files in the cache directory
  safetensors_files = list(cache_dir.glob("*.safetensors"))
  if not safetensors_files:
    raise FileNotFoundError("No safetensors files found in the cache directory.")

  # Load weights from each found safetensors file

  full_state_dict = {}
  for safetensor_file in safetensors_files:
    state_dict = load_safetensors(safetensor_file)

    if full_state_dict is not None:
      full_state_dict = full_state_dict | state_dict
    else:
      full_state_dict = state_dict

  # remap to work with our model
  remapped_state_dict = {}

  is_llama = True if "llama" in shard.model_id or "Llama" in shard.model_id else False
  
  if DEBUG >= 8 and is_llama:
    print("loading llama type weights")
  elif DEBUG >= 8 and not is_llama:
    print("loading weights")

  for key, value in full_state_dict.items():
    # load layer by shard
    for layer_num in range(shard.start_layer, shard.end_layer + 1):
      # change input layer norm to sa_norm for torchtune
      re_iln = re.findall(rf"model.layers\.{layer_num}\.(input_layernorm)\.weight", key)
      if len(re_iln) != 0:
        new_key = f"model.layers.{layer_num}.sa_norm.scale"
        remapped_state_dict[new_key] = value
        if DEBUG >= 8:
          print(f"{key} == {new_key}")

      # change post attention layernorm to mlp_norm for torchtune
      re_pal = re.findall(rf"model.layers\.{layer_num}\.(post_attention_layernorm)\.weight", key)
      if len(re_pal) != 0:
        new_key = f"model.layers.{layer_num}.mlp_norm.scale"
        remapped_state_dict[new_key] = value
        if DEBUG >= 8:
          print(f"{key} == {new_key}")

      # change self_attn to attn
      # along with changing o_proj to output_proj
      re_attn = re.findall(rf"model\.layers\.{layer_num}.(\w+)\.(\w+)\.(\w+)", key)
      if len(re_attn) != 0 and re_attn[0][0] == "self_attn":
        if re_attn[0][1] == "k_proj" and is_llama:
          value = _permute(
            t=value,
            n_heads=num_kv_heads,
            head_dim=head_dim,
            dim=dim
          )

          new_key = f"model.layers.{layer_num}.attn.{re_attn[0][1]}.{re_attn[0][2]}"
          remapped_state_dict[new_key] = value
        elif re_attn[0][1] == "q_proj" and is_llama:
          value = _permute(
            t=value,
            n_heads=num_heads,
            head_dim=head_dim,
            dim=dim
          )
          new_key = f"model.layers.{layer_num}.attn.{re_attn[0][1]}.{re_attn[0][2]}"
          remapped_state_dict[new_key] = value

        elif re_attn[0][1] == "o_proj":
          new_key = f"model.layers.{layer_num}.attn.output_proj.weight"
          remapped_state_dict[new_key] = value
        else:
          new_key = f"model.layers.{layer_num}.attn.{re_attn[0][1]}.{re_attn[0][2]}"
          remapped_state_dict[new_key] = value
        if DEBUG >= 8:
          print(f"{key} == {new_key}")

      # set mlp weights
      re_mlp = re.findall(rf"model\.layers\.{layer_num}.mlp.(\w+)\.(\w+)", key)
      if len(re_mlp) != 0:
        proj_name = re_mlp[0][0]
        if proj_name == "up_proj":
          proj_name = "w3"
        elif proj_name == "down_proj":
          proj_name = "w2"
        elif proj_name == "gate_proj":
          proj_name = "w1"
        new_key = f"model.layers.{layer_num}.mlp.{proj_name}.weight"
        remapped_state_dict[new_key] = value
        if DEBUG >= 8:
          print(f"{key} == {new_key}")

    # saving embed for paired weights
    if key == "model.embed_tokens.weight":
      remapped_state_dict["model.tok_embeddings.weight"] = value
      if DEBUG >= 8:
        print("model.embed_tokens.weight == model.tok_embeddings.weight")

    if key == "model.norm.weight":
      remapped_state_dict["model.norm.scale"] = value

    if key == "lm_head.weight":
      remapped_state_dict["model.output.weight"] = value
      # saving embed for paired weights
      if key == "model.embed_tokens.weight":
        remapped_state_dict["model.tok_embeddings.weight"] = value
        if DEBUG >= 8:
          print("model.embed_tokens.weight == model.tok_embeddings.weight")

      if key == "model.norm.weight":
        remapped_state_dict["model.norm.scale"] = value

      if key == "lm_head.weight":
        remapped_state_dict["model.output.weight"] = value

  if not remapped_state_dict:
    model.load_state_dict(full_state_dict, strict=True)
  else:
    if DEBUG >= 8:
      print("\nRemapped state dict\n")
      for rsdk in remapped_state_dict.keys():
        print(f"--  {rsdk}")

    # load new weight map
    model.load_state_dict(remapped_state_dict, strict=False)

    if DEBUG >= 8:
      print("\n--- checking weights ----\n")
      check_weights(model, remapped_state_dict)

class ShardTransformerDecoder(TransformerDecoder):
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

  def forward(
    self,
    tokens: torch.Tensor,
    *,
    mask: Optional[_MaskType] = None,
    input_pos: Optional[torch.Tensor] = None,
    hidden_state: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16
  ) -> Union[torch.Tensor, List[torch.Tensor]]:
    # Determine the type of input and shape
    if DEBUG >= 4:
      print("forward called")
      if tokens is not None:
        print(f"tokens [{tokens.shape}]: {tokens}")
        print(f"mask: {mask}")
        print(f"input_pos: {input_pos}")

      if hidden_state is not None:
        print(f"hidden_state [{hidden_state.shape}]: {hidden_state}")

    if hidden_state is not None:
      h = hidden_state  # Use directly as hidden states
    else:
      seq_len = tokens.shape[1]

      self._validate_inputs(
        seq_len,
        mask=mask,
        input_pos=input_pos,
      )

      fl_tokens = tokens.clone()
      h = self.tok_embeddings(fl_tokens).to(dtype=dtype)  # Apply token tok_embeddings

    # Initialize a list to capture hidden states if requested
    # for captured hidden states
    hidden = []
    curr_layers = [self.layers[i] for i in range(self.shard.start_layer, self.shard.end_layer + 1)]
    for i, layer in enumerate(curr_layers):
      if DEBUG >= 8:
        print(f"\nhidden layer in H[{self.shard.start_layer+i}]\n{h}")
        print(f"\nmask\n{mask}\ninput_pos\n{input_pos}")
        print(f"\noutput_hidden_states\n{self.output_hidden_states}\n")

      if i in self.output_hidden_states:
        hidden.append(h)

      # Process through each transformer layer
      h = layer(
        h,
        mask=mask,
        input_pos=input_pos,
      )

      if DEBUG >= 8:
        print(f"\nhidden layer out H[{self.shard.start_layer+i}]->H[{self.shard.start_layer+i+1}]\n{h}\n")

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

class MultiLayerPreceptron(nn.Module):
  def __init__(self, input_dim, hidden_dim, activation="silu", use_bias=False):
    """
    General MLP (Multi-Layer Perceptron) module.

    Args:
      input_dim (int): Dimensionality of the input.
      hidden_dims (int): Hidden layer/intermediate dimensions.
      output_dim (int): Dimensionality of the output.
      activation (str): Activation function ('relu', 'gelu', 'tanh', 'sigmoid', etc.).
      use_bias (bool): Use bias with linearization
    """
    super(MultiLayerPreceptron, self).__init__()

    # Activation function mapping
    activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "leaky_relu": nn.LeakyReLU(0.2), "silu": nn.SiLU()}

    # Ensure valid activation
    if activation not in activations:
      raise ValueError(f"Invalid activation: {activation}. Choose from {list(activations.keys())}")

    # Construct MLP layers
    self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.up_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.down_proj = nn.Linear(hidden_dim, input_dim, bias=use_bias)
    self.act_fn = activations[activation]

  def forward(self, x) -> torch.Tensor:
    return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))

class ShardInferenceState:
  def __init__(
    self,
    tokens: Optional[torch.tensor] = None,
    input_pos: Optional[torch.tensor] = None,
    mask: Optional[torch.tensor] = None,
    curr_pos: int = 0,
    device: torch.device = torch.device("cpu")
  ):
    self.tokens = tokens
    self.input_pos = input_pos
    self.mask = mask
    self.curr_pos = curr_pos
    self.device = device

  def from_dict(self, state_dict):
    """
    Data is stored as torch tensors on needed devices
    """
    self.tokens = torch.tensor(state_dict["tokens"]).to(self.device)
    self.input_pos = torch.tensor(state_dict["input_pos"]).to(self.device)
    self.mask = torch.tensor(state_dict["mask"]).to(self.device)
    self.curr_pos = state_dict["curr_pos"]

  def to_dict(self) -> dict:
    return {
      "tokens": self.tokens.numpy(force=True).tolist(),
      "input_pos": self.input_pos.numpy(force=True).tolist(),
      "mask": self.mask.numpy(force=True).tolist(),
      "curr_pos": self.curr_pos
    }

  def __str__(self) -> str:
    return f"""
    tokens: {self.tokens}
    input_pos: {self.input_pos}
    mask: {self.mask}
    curr_pos: {self.curr_pos}
    """

def layer_mlp(dim: int, hidden_dim: int) -> FeedForward:
  """
  Generalized MLP layer
  Ref: https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3_1/_component_builders.py#L124
  Ref: https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_component_builders.py#L127C1-L134C82
  """
  gate_proj = nn.Linear(dim, hidden_dim, bias=False)
  down_proj = nn.Linear(hidden_dim, dim, bias=False)
  up_proj = nn.Linear(dim, hidden_dim, bias=False)
  return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)