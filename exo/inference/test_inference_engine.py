from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
import numpy as np

# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine: InferenceEngine, model_id: str):
    prompt = "In a single word only, what is the capital of Japan? "
    resp_full, inference_state_full, _ = await inference_engine.infer_prompt(shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt)

    await inference_engine.reset_shard(shard=Shard(model_id=model_id, start_layer=0, end_layer=10, n_layers=32))
    resp1, inference_state, _ = await inference_engine.infer_prompt(shard=Shard(model_id=model_id, start_layer=0, end_layer=10, n_layers=32), prompt=prompt)

    await inference_engine.reset_shard(shard=Shard(model_id=model_id, start_layer=11, end_layer=31, n_layers=32))
    resp2, _, _ = await inference_engine.infer_tensor(shard=Shard(model_id=model_id, start_layer=11, end_layer=31, n_layers=32), input_data=resp1, inference_state=inference_state)

    assert np.array_equal(resp_full, resp2)

import asyncio

asyncio.run(test_inference_engine(
    MLXDynamicShardInferenceEngine(),
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
))

asyncio.run(test_inference_engine(
    TinygradDynamicShardInferenceEngine(),
    "/Users/alex/Library/Caches/tinygrad/downloads/llama3-8b-sfr",
))