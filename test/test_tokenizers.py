import os
import re
from transformers import AutoTokenizer, AutoProcessor
from exo.models import model_cards


def test_tokenizer(name, tokenizer, verbose=False):
    print(f"--- {name} ({tokenizer.__class__.__name__}) ---")
    text = "Hello! How can I assist you today? Let me know if you need help with something or just want to chat."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"{encoded=}")
    print(f"{decoded=}")

    reconstructed = ""
    for token in encoded:
      if verbose:
        print(f"{token=}")
        print(f"{tokenizer.decode([token])=}")
      reconstructed += tokenizer.decode([token])
    print(f"{reconstructed=}")

    strip_tokens = lambda s: s.lstrip(tokenizer.decode([tokenizer.bos_token_id])).rstrip(tokenizer.decode([tokenizer.eos_token_id]))
    assert text == strip_tokens(decoded) == strip_tokens(reconstructed)

ignore = ["TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R", "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", "llava-hf/llava-1.5-7b-hf", "mlx-community/Qwen*", "dummy", "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", "mlx-community/Phi-3.5-mini-instruct-4bit", "mlx-community/phi-4-4bit", "stabilityai/stable-diffusion-2-1-base"]
ignore_pattern = re.compile(r"^(" + "|".join(model.replace("*", ".*") for model in ignore) + r")")
models = []
for model_id in model_cards:
  for engine_type, repo_id in model_cards[model_id].get("repo", {}).items():
    if not ignore_pattern.match(repo_id):
      models.append(repo_id)
models = list(set(models))

verbose = os.environ.get("VERBOSE", "0").lower() == "1"
for m in models:
    # TODO: figure out why use_fast=False is giving inconsistent behaviour (no spaces decoding invididual tokens) for Mistral-Large-Instruct-2407-4bit
    # test_tokenizer(m, AutoProcessor.from_pretrained(m, use_fast=False), verbose)
    if m not in ["mlx-community/DeepSeek-R1-4bit", "mlx-community/DeepSeek-R1-3bit", "mlx-community/DeepSeek-V3-4bit", "mlx-community/DeepSeek-V3-3bit"]:
      test_tokenizer(m, AutoProcessor.from_pretrained(m, use_fast=True, trust_remote_code=True), verbose)
    test_tokenizer(m, AutoTokenizer.from_pretrained(m, trust_remote_code=True), verbose)
