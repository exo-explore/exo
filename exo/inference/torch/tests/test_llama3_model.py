"""
Test of pytorch based llama3 model
"""
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

import torchtune.generation as ttg
from torchtune.models import llama3
from torchtune.data import Message


from exo.inference.torch.models.llama3 import ShardedLlamaModel
from exo.inference.shard import Shard

from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
)


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEMP=0.6
TOP_K=35
TOP_P=0.9
MAX_SEQ_LEN=2048

def test_generation(text, max_length=10, config=None):
  """
  Test the generation capabilities of the LlamaModel with sample text.
  """
  # Tokenize input text
  messages = []
  messages.extend(
    [
      Message(role="user", content=text),
      # Empty assistant message to kick-start generation
      Message(role="assistant", content=""),
    ]
  )

  tokenizer_out = llama_tokenizer({"messages": messages}, inference=True)
  print(f"tokenizer_out: {tokenizer_out}")
  tokens = tokenizer_out["tokens"]
  prompt = torch.tensor(tokens, dtype=torch.int)

  hidden_states, logits = shard_model.generate(prompt)

  if hidden_states is not None:
    print(f"hidden_states: {hidden_states[0].shape}\n{hidden_states}")

  if logits is not None:
    print(f"logits: {logits.shape}\n{logits}")
  #if prompt.ndim == 1:
  #  prompt = prompt.view(1, -1)

  #bsz, prompt_length = prompt.size()
  #total_response_length = prompt_length + MAX_SEQ_LEN
  #generated_tokens = prompt.clone()
  #resp_max_seq_len = (
  #  total_response_length
  #  if not shard_model.model.caches_are_enabled()
  #  else shard_model.model.decoder_max_cache_seq_len
  #)

  ## masking for proper attention
  #padding_masks = prompt != llama_tokenizer.pad_id
  #if not padding_masks.all():
  #  padding_masks = torch.nn.functional.pad(
  #    padding_masks,
  #    (0, MAX_SEQ_LEN),
  #    value=True
  #  )

  #  masks = ttg.get_causal_mask_from_padding_mask(
  #    padding_masks,
  #    target_seq_len=resp_max_seq_len
  #  )

  #  input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
  #else:
  #  masks = torch.tril(
  #    torch.ones(
  #      total_response_length,
  #      resp_max_seq_len if resp_max_seq_len is not None else MAX_SEQ_LEN,
  #      dtype=torch.bool,
  #      device=prompt.device,
  #    )
  #  ).unsqueeze(0)

  #  input_pos = torch.arange(
  #    0, total_response_length, device=prompt.device
  #  ).unsqueeze(0)

  #if shard_model.model.caches_are_enabled():
  #  curr_masks = masks[:, :prompt_length]
  #else:
  #  curr_masks = masks[:, :prompt_length, :prompt_length]

  #rand_sample = torch.empty(
  #  (
  #    prompt.size(0),
  #    self.model.tok_embeddings.num_embeddings
  #  ), device=prompt.device
  #).exponential_(1, generator=None)

  #print(f"padding_masks: {padding_masks.shape}")
  #print(padding_masks.all())

  ## this can be sepearted out for dist inference
  ## see https://github.com/pytorch/torchtune/blob/bc4acc19ffab2366a14468c97294992dbb7c50d1/torchtune/generation/_generation.py#L66
  #next_token, gen_logits = ttg.generate_next_token(
  #  shard_model.model,
  #  input_pos=input_pos[:, :prompt_length].squeeze(),
  #  x=prompt,
  #  mask=curr_masks,
  #  q=rand_sample
  #)

  #print(f"next_token: {next_token}")

  #generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

  #print(f"generated_tokens: {generated_tokens}")

  #curr_pos = prompt_length

  ## stop tokens logic
  #stop_tokens = None
  #stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
  #stop_tokens = (
  #  torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype)
  #  if stop_tokens
  #  else None
  #)
  #stop_token_mask = torch.ones(
  #  (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
  #)

  ## finish writing stop token logic using torchtune generation
  ## ref https://github.com/pytorch/torchtune/blob/main/torchtune/generation/_generation.py#L337

  #for _ in range(max_length):

  #  if shard_model.model.caches_are_enabled():
  #    curr_input_pos = input_pos[:, curr_pos]
  #    curr_masks = masks[:, curr_pos, None, :]
  #  else:
  #    tokens = generated_tokens.clone()
  #    curr_input_pos = input_pos[:, : curr_pos + 1]
  #    curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

  #generated_tokens = generated_tokens.tolist()
  #print(f"resp: {llama_tokenizer.decode(generated_tokens[0])}")

if __name__ == "__main__":
  print("\nTesting generation:")
  # Get the path to the model files from the Hugging Face cache
  cache_dir = Path(snapshot_download(MODEL_NAME))
  print(f"Cache directory: {cache_dir}")

  # Load model configuration
  config = load_model_config(cache_dir / "config.json")

  print(f"current config\n{config}")

  # Setup shard
  shard = Shard(
    model_id=MODEL_NAME,
    start_layer=0,
    end_layer=4,#int(config["num_hidden_layers"]),
    n_layers=int(config["num_hidden_layers"])
  )

  # Initialize tokenizer
  llama_tokenizer_path = f"{cache_dir}/original/tokenizer.model"
  llama_tokenizer = llama3.llama3_tokenizer(path=llama_tokenizer_path)
  #tokenizer = AutoTokenizer.from_pretrained(
  #  MODEL_NAME,
  #  add_eos_token=True
  #)

  # Initialize LlamaModel with config and tokenizer
  shard_model = ShardedLlamaModel(config, shard, llama_tokenizer)
  print(f"\nshard_model: {shard_model}")
  load_model_weights_torchtune(cache_dir, shard, shard_model)

  # Sample text for testing
  test_text = "Hello"

  test_generation(test_text, 5, config)

