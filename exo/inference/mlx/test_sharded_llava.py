import torch
import codecs
import asyncio
import requests
from PIL import Image
from io import BytesIO

import mlx.core as mx
from mlx_lm.models.base import KVCache

from exo.inference.mlx.sharded_model import StatefulShardedModel
from exo.inference.mlx.sharded_utils import load_shard_llava
from exo.inference.shard import Shard

def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))
def generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature):
    kv_heads = (
        [model.language_model.model.n_kv_heads] * len(model.language_model.model.layers)
        if isinstance(model.language_model.model.n_kv_heads, int)
        else model.language_model.model.n_kv_heads
    )
    cache = [KVCache(model.language_model.model.head_dim, n) for n in kv_heads]
    logits = model(input_ids, pixel_values, cache=cache)
    logits = logits[:, -1, :]
    y = sample(logits, temperature=temperature)
    tokens = [y.item()]

    for n in range(max_tokens - 1):
        logits = model.language_model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits, temperature)
        token = y.item()
        if token == processor.tokenizer.eos_token_id:
            break
        tokens.append(token)

    return processor.tokenizer.decode(tokens)

shard_full = Shard("llava", 0, 31, 32)

full_model_shard, full_processor = asyncio.run(load_shard_llava("llava-hf/llava-1.5-7b-hf", shard=shard_full))

full = StatefulShardedModel(shard_full, full_model_shard)

PROMPT = "USER: <image>\nWhat are these?\nASSISTANT:"
IMAGE_FILE = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(IMAGE_FILE)
img = Image.open(BytesIO(response.content))
prompt = codecs.decode(PROMPT, "unicode_escape")
inputs = full_processor(prompt, img, return_tensors="np")
pixel_values = mx.array(inputs["pixel_values"])
input_ids = mx.array(inputs["input_ids"])

print(prompt)
generated_text = generate_text(
    input_ids, pixel_values, full_model_shard, full_processor, 10, 0
)
print(generated_text)