"""
Test inference engine and model sharding
"""

import asyncio
from exo.inference.shard import Shard
from exo.inference.torch.inference import TorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
import numpy as np
import time

async def test_inference_engine(
    inference_engine_1: InferenceEngine,
    inference_engine_2: InferenceEngine,
    model_id: str,
    n_layers: int):
  
  prompt = "In a single word only, what is the last name of the current president of the USA?"

  shard = Shard(
    model_id=model_id,
    start_layer=0,
    end_layer=n_layers-1,
    n_layers=n_layers
  )

  resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt(
    "A",
    shard=shard,
    prompt=prompt
  )

  print("\n------------resp_full---------------\n")
  print(resp_full)
  print("\n------------resp_full---------------\n")

  time.sleep(5)

  next_resp_full, _next_inference_state_full, _ = await inference_engine_1.infer_tensor(
    "A",
    shard=shard,
    input_data=resp_full,
    inference_state=inference_state_full,
  )

  print("\n------------next_resp_full---------------\n")
  print(next_resp_full)
  print("\n------------next_resp_full---------------\n")

  time.sleep(5)

  pp = int(n_layers/2)
 
  resp_shard = Shard(
    model_id=model_id, 
    start_layer=0, 
    end_layer=pp, 
    n_layers=n_layers
  )

  resp_shard2 = Shard(
    model_id=model_id, 
    start_layer=pp + 1, 
    end_layer=n_layers-1, 
    n_layers=n_layers
  )

  resp1, inference_state_1, _ = await inference_engine_1.infer_prompt(
    "B", 
    shard=resp_shard,
    prompt=prompt
  )

  print("\n------------resp1---------------\n")
  print(resp1)
  print("\n------------resp1---------------\n")

  time.sleep(5)


  resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=resp_shard2,
    input_data=resp1,
    inference_state=inference_state_1,
  )

  print("\n------------resp2---------------\n")
  print(resp2)
  print("\n------------resp2---------------\n")

  resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(
    "B",
    shard=resp_shard,
    input_data=resp2,
    inference_state=inference_state_2,
  )

  print("\n------------resp3---------------\n")
  print(resp3)
  print("\n------------resp3---------------\n")

  resp4, _inference_state_4, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=resp_shard2,
    input_data=resp3,
    inference_state=inference_state_3,
  )

  print("\n------------resp4---------------\n")
  print(resp4)
  print("\n------------resp4---------------\n")

  assert np.array_equal(resp_full, resp2)
  assert np.array_equal(next_resp_full, resp4)

if __name__ == '__main__':
  # try:
  #   print("\n\n -------- TEST QWEN2 -------- \n\n")
  #   asyncio.run(test_inference_engine(
  #     TorchDynamicShardInferenceEngine(HFShardDownloader()),
  #     TorchDynamicShardInferenceEngine(HFShardDownloader()),
  #     "Qwen/Qwen2-0.5B-Instruct",
  #     24
  #   ))
  # except Exception as err:
  #   print(f"\n\n !!!!!!!!!!! QWEN2 TEST FAILED \n{err}\n")

  #try:
  #  print("\n-------- Test unsloth/Llama-3.2-1B-Instruct ----------\n")
  #  asyncio.run(test_inference_engine(
  #    TorchDynamicShardInferenceEngine(HFShardDownloader()),
  #    TorchDynamicShardInferenceEngine(HFShardDownloader()),
  #    "unsloth/Llama-3.2-1B-Instruct",
  #    24
  #  ))
  #except Exception as err:
  #  print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")

  try:
    print("\n-------- Test unsloth/Meta-Llama-3.1-8B-Instruct ----------\n")
    asyncio.run(test_inference_engine(
      TorchDynamicShardInferenceEngine(HFShardDownloader()),
      TorchDynamicShardInferenceEngine(HFShardDownloader()),
      "unsloth/Meta-Llama-3.1-8B-Instruct",
      32
    ))
  except Exception as err:
    print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")


