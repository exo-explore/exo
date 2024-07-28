from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(
  inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str
):
  from exo.inference.tinygrad.inference import Tokenizer
  from pathlib import Path

  _tokenizer = Tokenizer(str(Path(model_id) / "tokenizer.model"))

  prompt = "In a single word only, what is the last name of the president of the United States? "
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

  print(f"{resp2=}")
  print(f"full: {_tokenizer.decode(resp_full)}")
  print(f"next full: {_tokenizer.decode(next_resp_full)}")
  print(f"resp2: {_tokenizer.decode(resp2)}")
  print(f"{resp4=}")
  print(f"resp4: {_tokenizer.decode(resp4)}")

  assert np.array_equal(resp_full, resp2)
  assert np.array_equal(next_resp_full, resp4)


asyncio.run(
  test_inference_engine(
    TinygradDynamicShardInferenceEngine(),
    TinygradDynamicShardInferenceEngine(),
    "llama3-8b-sfr",
  )
)
