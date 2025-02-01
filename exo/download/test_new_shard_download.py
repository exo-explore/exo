from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard
import asyncio

async def test_new_shard_download():
  shard_downloader = NewShardDownloader()
  shard_downloader.on_progress.register("test").on_next(lambda shard, event: print(shard, event))
  await shard_downloader.ensure_shard(Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=0, n_layers=16), "MLXDynamicShardInferenceEngine")
  async for path, shard_status in shard_downloader.get_shard_download_status("MLXDynamicShardInferenceEngine"):
    print("Shard download status:", path, shard_status)

if __name__ == "__main__":
  asyncio.run(test_new_shard_download())

