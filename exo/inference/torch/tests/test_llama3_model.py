"""
Test of pytorch based llama3 model
"""
from pathlib import Path

import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from exo.inference.torch.models.llm_utils import load_model_config, select_next_token
from exo.inference.torch.models.llama3 import LlamaModel
from exo.inference.shard import Shard

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
TEMP=0.7
TOP_K=25
TOP_P=0.9

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
    else:
       print(f"{name} not found in the state_dict")

  for name in state_dict:
    if name not in model_state_dict:
      print(f"Unexpected weight {name} found in state_dict")

def test_generation(model, tokenizer, text, max_length=10, config=None):
  """
  Test the generation capabilities of the LlamaModel with sample text.
  """
  # Tokenize input text
  prompt = tokenizer.apply_chat_template([
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": text
    }
  ], tokenize=False, add_generation_prompt=True)

  print(f"prompt: {prompt}")

  inputs = tokenizer(prompt, return_tensors="pt")
  input_ids = inputs.get("input_ids")
  attention_mask = inputs.get("attention_mask")

  print(f"input_ids: {input_ids}")
  print(f"attention_mask: {attention_mask}")

  # Start with initial input_ids
  generated_ids = input_ids.clone()

  # Generate tokens step-by-step
  past_kvs = None

  print(f"{model}")

  for _ in range(max_length):
    with torch.no_grad():
      pred_score, hstates, past_kvs = model(
        generated_ids,
        attention_mask=attention_mask,
        past_kv_cache=past_kvs
      )

    print(f"pred_score: {pred_score.shape}")
    print(f"hstates: {hstates.shape if hstates is not None else None}")
    print(f"past_kvs: {past_kvs.size if past_kvs is not None else None}")
    # Select next token using pred_score
    #next_token = select_next_token(pred_score, top_k=TOP_K, top_p=TOP_P, temperature=TEMP, use_max=False)
    next_token = ttg.sample(pred_score, temperature=TEMP, top_k=TOP_K)[:, -1, :]
    print(f"next_token: {next_token}")

    # Update generated_ids
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    print(f"generated_ids: {generated_ids}")

    # Check for EOS token
    print(f"next_token.item(): {next_token.item()}")

    if config:
      print(config["eos_token_id"])
      if next_token.item() in config["eos_token_id"]:
        break
    else:
      if next_token.item() == tokenizer.eos_token_id:
        break

  # Decode generated text
  generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  print(f"\n\n\n\nGenerated Response: {generated_text}")

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
    end_layer=int(config["num_hidden_layers"]) - 1,
    n_layers=int(config["num_hidden_layers"])
  )

  # Initialize tokenizer
  tokenizer = AutoTokenizer.from_pretrained(shard.model_id)

  # Initialize LlamaModel with config and tokenizer
  model = LlamaModel(config, shard)
  print(f"\nmodel: {model}")

  # Load weights from safetensors files in the cache directory
  safetensors_files = list(cache_dir.glob("*.safetensors"))
  if not safetensors_files:
    raise FileNotFoundError("No safetensors files found in the cache directory.")

  # Load weights from each found safetensors file
  for safetensor_file in safetensors_files:
    print(f"Loading weights from: {safetensor_file}")
    state_dict = load_safetensors(safetensor_file)

    # remap to work with our model
    remapped_state_dict = {}
    for key, value in state_dict.items():
      # Remove the 'model.' prefix if it exists
      print(f"remapping: {key}")
      if key.startswith('model.'):
        new_key = key[len('model.'):]  # Remove 'model.'
      else:
        new_key = key

      remapped_state_dict[new_key] = value

    model.load_state_dict(remapped_state_dict, strict=False)

    check_weights(model, remapped_state_dict)

  #exit()
  model.eval()  # Set the model to evaluation mode

  # Sample text for testing
  test_text = "Hello"

  test_generation(model, tokenizer, test_text, 5, config)

