from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(
  inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str
):
  prompt = "In a single word only, what is the last name of the current president of the USA?"
  resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt(
    "A", shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt
  )
  next_resp_full, _next_inference_state_full, _ = await inference_engine_1.infer_tensor(
    "A",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32),
    input_data=resp_full,
    inference_state=inference_state_full,
  )

  resp1, inference_state_1, _ = await inference_engine_1.infer_prompt(
    "B", shard=Shard(model_id=model_id, start_layer=0, end_layer=30, n_layers=32), prompt=prompt
  )
  resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=31, end_layer=31, n_layers=32),
    input_data=resp1,
    inference_state=inference_state_1,
  )
  resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=30, n_layers=32),
    input_data=resp2,
    inference_state=inference_state_2,
  )
  resp4, _inference_state_4, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=31, end_layer=31, n_layers=32),
    input_data=resp3,
    inference_state=inference_state_3,
  )

  assert np.array_equal(resp_full, resp2)
  assert np.array_equal(next_resp_full, resp4)


asyncio.run(
  test_inference_engine(
    MLXDynamicShardInferenceEngine(),
    MLXDynamicShardInferenceEngine(),
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  )
)

# TODO: Need more memory or a smaller model
# asyncio.run(test_inference_engine(
#     TinygradDynamicShardInferenceEngine(),
#     TinygradDynamicShardInferenceEngine(),
#     "llama3-8b-sfr",
# ))
