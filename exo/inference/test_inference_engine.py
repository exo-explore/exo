from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard
from exo.helpers import DEBUG
import os
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str, n_layers: int):
  prompt = "In a single word only, what is the last name of the current president of the USA?"
  resp_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers), prompt=prompt)
  token_full = await inference_engine_1.sample(resp_full)
  token_full = token_full.reshape(1, -1)
  next_resp_full, _ = await inference_engine_1.infer_tensor(
    "A",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=token_full,
  )

  pp = n_layers // 2
  resp1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers), prompt=prompt)
  resp2, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=resp1,
  )
  tokens2 = await inference_engine_1.sample(resp2)
  tokens2 = tokens2.reshape(1, -1)
  resp3, _ = await inference_engine_1.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers),
    input_data=tokens2,
  )
  resp4, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=resp3,
  )

  assert np.array_equal(resp_full, resp2)
  assert np.array_equal(next_resp_full, resp4)


asyncio.run(test_inference_engine(MLXDynamicShardInferenceEngine(NewShardDownloader()), MLXDynamicShardInferenceEngine(NewShardDownloader()), "llama-3.2-1b", 16))

if os.getenv("RUN_TINYGRAD", default="0") == "1":
  import tinygrad
  import os
  from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
  tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
  asyncio.run(test_inference_engine(TinygradDynamicShardInferenceEngine(NewShardDownloader()), TinygradDynamicShardInferenceEngine(NewShardDownloader()), "llama-3.2-1b", 32))
