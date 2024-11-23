"""
Test inference engine and model sharding
"""
import time
import asyncio

from exo.inference.shard import Shard
from exo.inference.torch.pt_inference import TorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine

import numpy as np

async def test_inference_engine(
    inference_engine_1: InferenceEngine,
    inference_engine_2: InferenceEngine,
    model_id: str,
    n_layers: int):

  prompt = "In a single word only, what is the last name of the current president of the USA?"

  shard = Shard(
    model_id=model_id,
    start_layer=0,
    end_layer=0,
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

if __name__ == '__main__':
  try:
    print("\n\n -------- TEST meta-llama/Llama-3.2-1B-Instruct -------- \n\n")
    asyncio.run(test_inference_engine(
      TorchDynamicShardInferenceEngine(HFShardDownloader()),
      TorchDynamicShardInferenceEngine(HFShardDownloader()),
      "meta-llama/Llama-3.2-1B-Instruct",
      16
    ))
  except Exception as err:
    print(f"\n!!!! LLAMA TEST FAILED \n{err}\n")


