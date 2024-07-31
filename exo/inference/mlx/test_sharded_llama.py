import asyncio
import mlx.core as mx
from exo.inference.mlx.sharded_model import StatefulShardedModel
from exo.inference.mlx.sharded_utils import load_shard
from exo.inference.shard import Shard

# 79, 80 for Llama-3-70B
shard_full = Shard("llama", 0, 31, 32)
shard1 = Shard("llama", 0, 12, 32)
shard2 = Shard("llama", 13, 31, 32)

full_model_shard, full_tokenizer = asyncio.run(load_shard("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", shard=shard_full))
model_shard1, tokenizer1 = asyncio.run(load_shard("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", shard=shard1))
model_shard2, tokenizer2 = asyncio.run(load_shard("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", shard=shard2))

full = StatefulShardedModel(shard_full, full_model_shard)
m1 = StatefulShardedModel(shard1, model_shard1)
m2 = StatefulShardedModel(shard2, model_shard2)

full_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer1.add_special_tokens({'pad_token': '[PAD]'})
tokenizer2.add_special_tokens({'pad_token': '[PAD]'})


prompts = ["write a beautiful haiku about a utopia where people own their AI with edge intelligence:", "write a small rap:"]
prompt_tokens = mx.array(full_tokenizer._tokenizer(prompts, padding=True, truncation=True)["input_ids"])
max_tokens = 30

resp = prompt_tokens

full_generated_tokens = []
for _ in range(max_tokens):
  resp = full.step("full", resp)
  full_generated_tokens.append(resp)

first_gen = [x[0].item() for x in full_generated_tokens]
second_gen = [x[1].item() for x in full_generated_tokens]

print("full response:")
print("Prompt: \n", prompts[0], "\nGeneration: \n", full_tokenizer.decode(first_gen))
print("Prompt: \n", prompts[1], "\nGeneration: \n", full_tokenizer.decode(second_gen))


sharded_generated_tokens = []
sharded_resp = prompt_tokens
for _ in range(max_tokens):
  resp1 = m1.step("shard", sharded_resp)
  sharded_resp = m2.step("shard", resp1)
  sharded_generated_tokens.append(sharded_resp)

first_gen = [x[0].item() for x in sharded_generated_tokens]
second_gen = [x[1].item() for x in sharded_generated_tokens]

print("sharded response:")
print("Prompt: \n", prompts[0], "\nGeneration: \n", tokenizer1.decode(first_gen))
print("Prompt: \n", prompts[1], "\nGeneration: \n", tokenizer1.decode(second_gen))

# assert tokenizer1.decode(full_generated_tokens) == tokenizer1.decode(sharded_generated_tokens)
