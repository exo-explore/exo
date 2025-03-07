"""
Test of pytorch based mistral models
full layer run
"""

from pathlib import Path
import torch
from huggingface_hub import snapshot_download

import torchtune.generation as ttg

from transformers import AutoTokenizer

from exo.inference.torch.models.general_mha import ShardedGeneralModel
from exo.inference.shard import Shard

from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune
)

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
TEMP = 0.85
TOP_K = 35
MAX_NEW_TOKENS = 200
RAND_SEED = 42


def main(
    model,
    prompt: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16
):
  messages = [{
    "role": "assistant",
    "content": "",
  }, {
    "role": "user",
    "content": prompt,
  }]

  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  tok_out = tokenizer([text], return_tensors="pt")
  print(f"tok_out: {tok_out}")
  tokens = tok_out.input_ids.to(device=device, dtype=torch.int)

  rng = torch.Generator(device=device)
  rng.manual_seed(RAND_SEED)
  
  generated_tokens = tokens.clone()

  print(f"tokens: {tokens}")

  bsz, tokens_length = tokens.size()

  # using self.max_seq_len will take up alot of VRAM
  total_response_length = tokens_length + MAX_NEW_TOKENS

  # setup cache
  if not model.model.caches_are_enabled():
    with device:
      model.model.setup_caches(
        bsz,
        dtype,
        decoder_max_seq_len=total_response_length
      )

  if not model.model.caches_are_enabled():
    max_seq_len = total_response_length
  else:
    max_seq_len = model.model.decoder_max_cache_seq_len

  # masking for proper attention
  
  # select correct pad_id
  if hasattr(tokenizer, "pad_id"):
    pad_id = tokenizer.pad_id
  elif hasattr(tokenizer, "pad_token_id"):
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    if tokenizer.pad_token_id is not None:
      pad_id = tokenizer.pad_token_id
    else:
      pad_id = 0
  else:
    pad_id = 0

  print(f"pad_id: {pad_id}")

  padding_masks = tokens != pad_id
  if not padding_masks.all():
    padding_masks = torch.nn.functional.pad(
      padding_masks,
      (0, MAX_NEW_TOKENS),
      value=True,
    )

    mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

    input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
  else:
    mask = torch.tril(torch.ones(
      total_response_length,
      max_seq_len,
      dtype=torch.bool,
      device=device,
    )).unsqueeze(0)

    input_pos = torch.arange(0, total_response_length, device=device).unsqueeze(0)
    
    print(f"init mask: {mask}")
    print(f"init input_pos: {input_pos}")

  curr_pos = 0

  _, logits = model.generate(
    tokens=tokens,
    mask=mask,
    input_pos=input_pos,
    curr_pos=curr_pos
  )

  curr_pos = tokens_length

  q = torch.empty((
    logits.size(0),
    model.model.tok_embeddings.num_embeddings
  ), device=logits.device).exponential_(1, generator=rng)
  
  tokens = ttg.sample(
    logits=logits[:, -1].clone(),
    temperature=TEMP,
    top_k=TOP_K,
    q=q
  )

  print(f"tokens: {tokens}")

  for i in range(MAX_NEW_TOKENS - 1):
    print(f"gen #{i+1}")

    if tokens.item() == tokenizer.eos_token_id:
      print("stop token hit!")
      break

    tokens = tokens.view(1, -1).to(device=device) if tokens.ndim == 1 else tokens

    _, logits = model.generate(
      tokens=tokens,
      input_pos=input_pos,
      mask=mask,
      curr_pos=curr_pos
    )

    curr_pos += 1

    q = torch.empty(
    (
      logits.size(0),
      model.model.tok_embeddings.num_embeddings
    ), device=logits.device).exponential_(1, generator=rng)

    tokens = ttg.sample(
      logits=logits[:, -1].clone(),
      temperature=TEMP,
      top_k=TOP_K,
      q=q,
    )

    print(f"tokens: {tokens}")

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
    print(f"generated_tokens: {generated_tokens}")

    if not model.model.caches_are_enabled():
      tokens = generated_tokens.clone()

  print(f"\n\n[resp from model]\n\n{tokenizer.decode(generated_tokens.tolist()[0])}\n\n\n")


if __name__ == "__main__":
  # prompt = "Hello, how are you?"
  prompt = "Tell me a joke."
  # prompt = "What is the meaning of exo?"
  # prompt = "Tell me a short 4 line haiku"
  # prompt = "In a single word only, what is the last name of the current president of the USA?"

  # Get the path to the model files from the Hugging Face cache
  cache_dir = Path(snapshot_download(MODEL_NAME))
  print(f"Cache directory: {cache_dir}")

  # Load model configuration
  config = load_model_config(cache_dir/"config.json")

  print(f"current config\n{config}")

  # Setup shard
  n_layers = int(config["num_layers"])
  shard_1 = Shard(
    model_id=MODEL_NAME,
    start_layer=0,
    end_layer=n_layers - 1,
    n_layers=n_layers,
  )

  # Initialize tokenizer
  tokenizer = AutoTokenizer.from_pretrained(cache_dir)

  # Initialize LlamaModel with config and tokenizer
#   device = torch.device("cuda")
  dtype = torch.bfloat16
  device = torch.device("cpu")

  shard_model_1 = ShardedGeneralModel(
    config=config,
    shard=shard_1,
    device=device,
    dtype=config["torch_dtype"],
    use_cache=True,
    max_generated_tokens=MAX_NEW_TOKENS,
  )

  print(f"\nshard_model_1: {shard_model_1}")

  # load_model_weights_torchtune(cache_dir, shard_1, shard_model_1)
  load_model_weights_torchtune(
    cache_dir=cache_dir,
    shard=shard_1,
    model=shard_model_1,
    num_heads=config["num_heads"],
    num_kv_heads=config["num_kv_heads"],
    dim=config["embed_dim"],
    head_dim=config["head_dim"]
  )

  main(shard_model_1, prompt, device, config["torch_dtype"])
