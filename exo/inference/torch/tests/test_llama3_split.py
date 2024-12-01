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
TOP_K = 35
MAX_NEW_TOKENS=10


def test_generation_1(shard_model, tokens):
  """
  Test the generation capabilities of the LlamaModel with sample text.
  """

  hidden_states, _ = shard_model.generate(tokens)

  if hidden_states is not None:
    print(f"hidden_states[{len(hidden_states)}]: {hidden_states}")

  return hidden_states

def test_generation_2(shard_model, hidden_state):
  print("Generate with the rest of layers")
  print(f"in hidden_states {hidden_state.shape}: {hidden_state}")
  
  _, logits = shard_model.generate(
    hidden_state=hidden_state
  )

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

  return tokens

if __name__ == "__main__":
  print("\nTesting generation:")

  # prompt = "In a single word only, what is the last name of the current president of the USA?"
  prompt = "In a single word only, what is the capital of france?"

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
  device = torch.device("cuda")
  shard_model_1 = ShardedLlamaModel(
    config=config,
    shard=shard_1,
    device=device,
    use_cache=False
  )

  load_model_weights_torchtune(cache_dir, shard_1, shard_model_1)

  shard_model_2 = ShardedLlamaModel(
    config=config,
    shard=shard_2,
    device=device,
    use_cache=False
  )

  load_model_weights_torchtune(cache_dir, shard_2, shard_model_2)

  # Tokenize input text
  messages = []
  messages.extend([
    Message(role="system", content="You are a helpful and creative AI assistant."),
    Message(role="user", content=prompt),
    # Empty assistant message to kick-start generation
    Message(role="assistant", content=""),
  ])

  tokenizer_out = llama_tokenizer({"messages": messages}, inference=True)
  print(f"tokenizer_out: {tokenizer_out}")
  tokens = torch.tensor([tokenizer_out["tokens"]], dtype=torch.int).to(device=device)

  generated_tokens = tokens.clone().to(device=device)

  for i in range(MAX_NEW_TOKENS):
    print(f"--------- gen #{i} ----------")
    print(f"\n------------ {shard_1.start_layer} - {shard_1.end_layer} ----------\n")
    
    shard_1_hs = test_generation_1(
      shard_model=shard_model_1,
      tokens=tokens
    )

    print(f"\n out shard_1_hs {shard_1_hs}")
    
    print(f"\n------------ {shard_2.start_layer} - {shard_2.end_layer} ----------\n")
    
    tg2_token = test_generation_2(shard_model_2, shard_1_hs)

    if (tg2_token in llama_tokenizer.stop_tokens
      or tg2_token == llama_tokenizer.eos_id):
      print("hit stop token")
      break

    generated_tokens = torch.cat([generated_tokens, tg2_token], dim=-1)
    print(f"\ngenerated_tokens: {generated_tokens}")

    tokens = generated_tokens.clone()

print("\n\n[resp from model]\n\n")
print(f"{llama_tokenizer.decode(generated_tokens.tolist()[0])}")
