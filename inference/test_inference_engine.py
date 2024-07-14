from inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from inference.inference_engine import InferenceEngine
from inference.shard import Shard
import numpy as np

# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine: InferenceEngine, model_id: str, input_data: np.array):
    resp_full, _ = await inference_engine.infer_tensor(shard=Shard(model_id=model_id, start_layer=0, end_layer=1, n_layers=2), input_data=input_data)

    resp1, _ = await inference_engine.infer_tensor(shard=Shard(model_id=model_id, start_layer=0, end_layer=0, n_layers=2), input_data=input_data)
    resp2, _ = await inference_engine.infer_tensor(shard=Shard(model_id=model_id, start_layer=1, end_layer=1, n_layers=2), input_data=resp1)

    assert np.array_equal(resp_full, resp2)

import asyncio

asyncio.run(test_inference_engine(
    MLXDynamicShardInferenceEngine(),
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    [1234]
))
