from transformers import AutoTokenizer, AutoProcessor


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

# test_tokenizer(AutoTokenizer.from_pretrained("mlx-community/Mistral-Nemo-Instruct-2407-4bit"))
models = [
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    "NousResearch/Meta-Llama-3.1-70B",
    "mlx-community/Meta-Llama-3.1-405B-4bit",
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
    # "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    # "llava-hf/llava-1.5-7b-hf",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Mistral-Large-Instruct-2407-4bit",
]

import os
verbose = os.environ.get("VERBOSE", "0").lower() == "1"
for m in models:
    # TODO: figure out why use_fast=False is giving inconsistent behaviour (no spaces decoding invididual tokens) for Mistral-Large-Instruct-2407-4bit
    # test_tokenizer(m, AutoProcessor.from_pretrained(m, use_fast=False), verbose)
    test_tokenizer(m, AutoProcessor.from_pretrained(m, use_fast=True), verbose)
    test_tokenizer(m, AutoTokenizer.from_pretrained(m), verbose)
