import codecs
import asyncio
import requests
from PIL import Image
from io import BytesIO

import mlx.core as mx
from mlx_lm.models.base import KVCache

from exo.inference.mlx.sharded_model import StatefulShardedModel
from exo.inference.mlx.sharded_utils import load_shard
from exo.inference.shard import Shard

shard_full = Shard("llava", 0, 31, 32)
shard1 = Shard("llava", 0, 12, 32)
shard2 = Shard("llava", 13, 31, 32)

model_path = "llava-hf/llava-1.5-7b-hf"

full_model_shard, full_processor = asyncio.run(load_shard(model_path, shard=shard_full))
model_shard1, processor1 = asyncio.run(load_shard(model_path, shard=shard1))
model_shard2, processor2 = asyncio.run(load_shard(model_path, shard=shard2))

full = StatefulShardedModel(shard_full, full_model_shard)
m1 = StatefulShardedModel(shard1, model_shard1)
m2 = StatefulShardedModel(shard2, model_shard2)

PROMPT = "USER: <image>\nWhat are these?\nASSISTANT:"
IMAGE_FILE = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(IMAGE_FILE)
img = Image.open(BytesIO(response.content))
prompt = codecs.decode(PROMPT, "unicode_escape")
inputs = full_processor(prompt, img, return_tensors="np")
pixel_values = mx.array(inputs["pixel_values"])
input_ids = mx.array(inputs["input_ids"])

print(prompt)
y = full.step("full", input_ids, pixel_values, temp=0)
full_generated_tokens = [y.item()]

for _ in range(13):
    y = full.step("full", y, temp=0)
    full_generated_tokens.append(y.item())

full_response = full_processor.tokenizer.decode(full_generated_tokens)
print("full response:", full_response)

inputs = processor1(prompt, img, return_tensors="np")
pixel_values = mx.array(inputs["pixel_values"])
input_ids = mx.array(inputs["input_ids"])

y = m1.step("shard", input_ids, pixel_values, temp=0)
y = m2.step("shard", y, temp=0)
full_generated_tokens = [y.item()]

for _ in range(13):
    y = m1.step("shard", y, temp=0)
    y = m2.step("shard", y, temp=0)
    full_generated_tokens.append(y.item())

sharded_response = processor2.tokenizer.decode(full_generated_tokens)
print("sharded response:", sharded_response)

assert full_response == sharded_response