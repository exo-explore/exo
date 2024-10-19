"""
Sharding safetensor
"""

import asyncio

from exo.inference.shard import Shard
from exo.inference.torch.model.hf_safe_tensor_shard import HFSafeTensorShard
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.hf.hf_helpers import get_weight_map


async def main():
  start_layer = 3
  end_layer = 5

  # Create a Shard object
  shard = Shard(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
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
  tensor_shard.restore_backup()


if __name__ == "__main__":
  asyncio.run(main())
