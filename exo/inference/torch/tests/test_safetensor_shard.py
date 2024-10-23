"""
Sharding safetensor
"""

import asyncio

from exo.inference.shard import Shard
from exo.inference.torch.models.hf_safe_tensor_shard import HFSafeTensorShard
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.hf.hf_helpers import get_weight_map

from transformers import AutoModelForCausalLM, AutoTokenizer

async def main():
  start_layer = 0
  end_layer = 1

  # Create a Shard object
  shard = Shard(
    model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
    start_layer=start_layer,
    end_layer=end_layer-1,
    n_layers=32
  )

  print(f"Loading shard: {shard}")
  shard_downloader = HFShardDownloader()

  # Ensure shard is downloaded
  model_path = await shard_downloader.ensure_shard(shard)

  # weight map, if any
  model_wm = await get_weight_map(
    repo_id=shard.model_id
  )

  tensor_shard = HFSafeTensorShard(model_path, shard)
  tensor_shard.modify_safetensor()
  tensor_shard.create_safetensor_index()

  # load model and test
  model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=shard.model_id,
    local_files_only=True,
    num_hidden_layers=shard.end_layer - shard.start_layer,
    #device_map="auto",
    torch_dtype="float16"
  ).to("cuda")

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "In one simple word, what is the color of a red apple?"}
  ]

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  model_inputs = tokenizer([text], return_tensors="pt")

  print(f"model_inputs:\n{model_inputs}")

  tensor_shard.restore_backups()

if __name__ == "__main__":
  asyncio.run(main())
