import mlx.core as mx
from exo.inference.mlx.sharded_model import StatefulShardedModel
from exo.inference.mlx.sharded_utils import load_shard
from exo.inference.shard import Shard

# 79, 80 for Llama-3-70B
shard_full = Shard("llama", 0, 31, 32)
shard1 = Shard("llama", 0, 12, 32)
shard2 = Shard("llama", 13, 31, 32)

full_model_shard, full_tokenizer = load_shard("mlx-community/Meta-Llama-3-8B-Instruct-4bit", shard=shard_full)
model_shard1, tokenizer1 = load_shard("mlx-community/Meta-Llama-3-8B-Instruct-4bit", shard=shard1)
model_shard2, tokenizer2 = load_shard("mlx-community/Meta-Llama-3-8B-Instruct-4bit", shard=shard2)

full = StatefulShardedModel(shard_full, full_model_shard)
m1 = StatefulShardedModel(shard1, model_shard1)
m2 = StatefulShardedModel(shard2, model_shard2)

prompt = "write a beautiful haiku about a utopia where people own their AI with edge intelligence:"
prompt_tokens = mx.array(full_tokenizer.encode(prompt))
max_tokens = 50

resp = prompt_tokens
full_generated_tokens = []
for _ in range(max_tokens):
    resp = full.step(resp)
    full_generated_tokens.append(resp.item())

print("full response: ", full_tokenizer.decode(full_generated_tokens))


sharded_generated_tokens = []
sharded_resp = prompt_tokens
for _ in range(max_tokens):
    resp1 = m1.step(sharded_resp)
    sharded_resp = m2.step(resp1)
    sharded_generated_tokens.append(sharded_resp.item())

print("sharded response: ", tokenizer1.decode(sharded_generated_tokens))

assert tokenizer1.decode(full_generated_tokens) == tokenizer1.decode(sharded_generated_tokens)
