"""
Test of pytorch based llama3 model
"""

from pathlib import Path
import torch
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
TEMP = 0.6
TOP_K = 25
MAX_NEW_TOKENS=10


def test_generation_1(shard_model, prompt):
  """
  Test the generation capabilities of the LlamaModel with sample text.
  """
  # Tokenize input text
  messages = []
  messages.extend([
    Message(role="system", content="You are a helpful and creative AI assistant."),
    Message(role="user", content=prompt),
    # Empty assistant message to kick-start generation
    Message(role="assistant", content=""),
  ])

  print(f"last?: {shard_model.shard.is_last_layer()}")
  tokenizer_out = llama_tokenizer({"messages": messages}, inference=True)
  print(f"tokenizer_out: {tokenizer_out}")
  tokens = torch.tensor(tokenizer_out["tokens"], dtype=torch.int)

  hidden_states, _ = shard_model.generate(tokens)

  if hidden_states is not None:
    print(f"hidden_states[{len(hidden_states)}]: {hidden_states}")

  return hidden_states, tokens


def test_generation_2(shard_model, in_tokens, hidden_state):
  print("Generate with the rest of layers")
  hidden_states, logits = shard_model.generate(tokens=in_tokens, hidden_state=hidden_state)

  if hidden_states is not None:
    print(f"hidden_states {hidden_states.shape}: {hidden_states}")

  if logits is not None:
    print(f"logits: {logits.shape}\n{logits}")

  # rand_sample = torch.empty((
  #     logits.size(0),
  #     shard_model.model.tok_embeddings.num_embeddings
  #   ),
  #   device=logits.device
  # ).exponential_(1, generator=None)

  tokens = ttg.sample(
    logits=logits[:, -1].clone(),
    temperature=TEMP,
    top_k=TOP_K,
    # q=rand_sample
  )

  print(f"tokens: {tokens}")

  generated_tokens = tokens.clone()
  generated_tokens = generated_tokens.tolist()

  print(f"generated_tokens: {generated_tokens}")

  print(f"\n\n[resp from model]\n\n{llama_tokenizer.decode(generated_tokens[0])}\n\n\n")


if __name__ == "__main__":
  print("\nTesting generation:")

  prompt = "Hello, just say 'Hello' back nothing else"

  # Get the path to the model files from the Hugging Face cache
  cache_dir = Path(snapshot_download(MODEL_NAME))

  # Load model configuration
  config = load_model_config(cache_dir / "config.json")

  # Setup shard
  n_layers = int(config["num_layers"])
  s1_end = int(n_layers / 2)
  shard_1 = Shard(model_id=MODEL_NAME, start_layer=0, end_layer=s1_end, n_layers=n_layers)

  shard_2 = Shard(model_id=MODEL_NAME, start_layer=s1_end + 1, end_layer=n_layers - 1, n_layers=n_layers)

  # Initialize tokenizer
  llama_tokenizer_path = f"{cache_dir}/original/tokenizer.model"
  llama_tokenizer = llama3.llama3_tokenizer(path=llama_tokenizer_path)

  # Initialize LlamaModel with config and tokenizer
  shard_model_1 = ShardedLlamaModel(
    config,
    shard_1,
    llama_tokenizer,
    None,
    MAX_NEW_TOKENS
  )
  print(f"\nshard_model_1: {shard_model_1}")
  load_model_weights_torchtune(cache_dir, shard_1, shard_model_1)
  shard_1_hs, shard_1_tokens = test_generation_1(shard_model_1, prompt)

  shard_model_2 = ShardedLlamaModel(
    config,
    shard_2,
    llama_tokenizer,
    None,
    MAX_NEW_TOKENS
  )
  print(f"\nshard_model_2: {shard_model_2}")
  load_model_weights_torchtune(cache_dir, shard_2, shard_model_2)
  test_generation_2(shard_model_2, shard_1_tokens, shard_1_hs)
