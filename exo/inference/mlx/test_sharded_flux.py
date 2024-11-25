import asyncio
import numpy as np
import mlx.core as mx
from exo.inference.mlx.stateful_model import StatefulModel
from exo.inference.mlx.sharded_utils import load_shard
from exo.inference.shard import Shard

shard_full = Shard("flux", 0, 31, 32)
shard1 = Shard("flux", 0, 12, 32)
shard2 = Shard("flux", 13, 31, 32)

model_path = "black-forest-labs/FLUX.1-dev"

full_model_shard, full_tokenizer = asyncio.run(load_shard(model_path, shard=shard_full))
model_shard1, tokenizer1 = asyncio.run(load_shard(model_path, shard=shard1))
model_shard2, tokenizer2 = asyncio.run(load_shard(model_path, shard=shard2))

full = StatefulModel(full_model_shard)
m1 = StatefulModel(model_shard1)
m2 = StatefulModel(model_shard2)

prompt = "Generate an image of a futuristic cityscape with flying cars:"
prompt_tokens = mx.array(full_tokenizer.encode(prompt))
max_tokens = 50

resp = prompt_tokens
full_generated_tokens = []
for _ in range(max_tokens):
    resp = full(resp)
    full_generated_tokens.append(resp.item())

print("full response: ", full_tokenizer.decode(full_generated_tokens))

sharded_generated_tokens = []
sharded_resp = prompt_tokens
for _ in range(max_tokens):
    resp1 = m1(sharded_resp)
    sharded_resp = m2(resp1)
    sharded_generated_tokens.append(sharded_resp.item())

print("sharded response: ", tokenizer1.decode(sharded_generated_tokens))

assert tokenizer1.decode(full_generated_tokens) == tokenizer1.decode(sharded_generated_tokens)
