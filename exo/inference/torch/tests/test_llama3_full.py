"""
Test of pytorch based llama3 models
full layer run
"""

from pathlib import Path
import torch
from huggingface_hub import snapshot_download

import torchtune.generation as ttg
from torchtune.models import llama3
from torchtune.data import Message

from transformers import AutoTokenizer

from exo.inference.torch.models.llama3 import ShardedLlamaModel
from exo.inference.shard import Shard

from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_weights_torch,
  load_model_weights_torchtune
)

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEMP = 0.85
TOP_K = 35
MAX_NEW_TOKENS = 200
RAND_SEED = 42


def main(model, prompt: str, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16):
  messages = [{
    "role": "assistant",
    "content": "",
  }, {
    "role": "user",
    "content": prompt,
  }]

  text = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  tok_out = llama_tokenizer([text], return_tensors="pt")
  print(f"tok_out: {tok_out}")
  tokens = tok_out.input_ids.to(device=device, dtype=torch.int)

  # messages = []
  # messages.extend([
  #   Message(role="system", content="You are a helpful and creative AI assistant."),
  #   Message(role="user", content=prompt),
  #   # Empty assistant message to kick-start generation
  #   Message(role="assistant", content=""),
  # ])

  # tokenizer_out = llama_tokenizer({"messages": messages}, inference=True)
  # tokens = torch.tensor([tokenizer_out["tokens"]], dtype=torch.int, device=device)
  

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
  if hasattr(llama_tokenizer, "pad_id"):
    pad_id = llama_tokenizer.pad_id
  elif hasattr(llama_tokenizer, "pad_token_id"):
    print(f"pad_token_id: {llama_tokenizer.pad_token_id}")
    if llama_tokenizer.pad_token_id is not None:
      pad_id = llama_tokenizer.pad_token_id
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

  if model.model.caches_are_enabled():
    curr_mask = mask[:, :tokens_length]
  else:
    curr_mask = mask[:, :tokens_length, :tokens_length]

  curr_pos = 0

  _, logits = model.generate(
    tokens=tokens,
    mask=curr_mask,
    input_pos=input_pos[:, :tokens_length].squeeze(),
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

    if tokens.item() == llama_tokenizer.eos_token_id:
    # if tokens.item() in llama_tokenizer.stop_tokens:
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

  print(f"\n\n[resp from model]\n\n{llama_tokenizer.decode(generated_tokens.tolist()[0])}\n\n\n")


def normal_full(model, user_prompt: str, device: torch.device = torch.device("cpu")):
  # Tokenize input text
  messages = []
  messages.extend([
    Message(role="system", content="You are a helpful and creative AI assistant."),
    Message(role="user", content=user_prompt),
    # Empty assistant message to kick-start generation
    Message(role="assistant", content=""),
  ])

  tokenizer_out = llama_tokenizer({"messages": messages}, inference=True)
  prompt = torch.tensor(tokenizer_out["tokens"], dtype=torch.int, device=device)
  print(f"tokens prompt: {prompt}")
  print(f"pad_id: {llama_tokenizer.pad_id}")
  # messages = [{
  #   "role": "assistant",
  #   "content": "",
  # }, {
  #   "role": "user",
  #   "content": prompt,
  # }]

  # text = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # tok_out = llama_tokenizer([text], return_tensors="pt")
  # prompt_tok = tok_out.input_ids.to(device=device)
  # print(f"tokens prompt: {prompt_tok}")

  generated_tokens, _ = ttg.generate(
    model=model.model,
    prompt=prompt,
    max_generated_tokens=MAX_NEW_TOKENS,
    pad_id=llama_tokenizer.pad_id,
    temperature=TEMP,
    top_k=TOP_K,
    stop_tokens=llama_tokenizer.stop_tokens,
  )

  generated_tokens = generated_tokens[:, -MAX_NEW_TOKENS:].tolist()

  print(f"generated_tokens: {generated_tokens}")

  print(f"\n\n[resp from model]\n\n{llama_tokenizer.decode(generated_tokens[0])}\n\n\n")


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
  # llama_tokenizer_path = f"{cache_dir}/original/tokenizer.model"
  # llama_tokenizer = llama3.llama3_tokenizer(path=llama_tokenizer_path)
  llama_tokenizer = AutoTokenizer.from_pretrained(cache_dir)

  # Initialize LlamaModel with config and tokenizer
  device = torch.device("cuda")
  dtype = torch.bfloat16
  # device = torch.device("cpu")
  shard_model_1 = ShardedLlamaModel(
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

  import time
  time.sleep(5)
  # main(shard_model_1, prompt, device, config["torch_dtype"])
  # normal_full(shard_model_1, prompt, device)
