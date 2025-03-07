"""
Test inference engine and model sharding
"""
import pytest
import asyncio

from exo.inference.shard import Shard
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader

import numpy as np

@pytest.mark.asyncio
async def test_inference_engine():
  prompt = "In a single word only, what is the last name of the current president of the USA?"

  shard = Shard(
    model_id="llama-3.2-1b",
    start_layer=0,
    end_layer=8,
    n_layers=16
  )

  shard_2 = Shard(
    model_id="llama-3.2-1b",
    start_layer=9,
    end_layer=15,
    n_layers= 16
  )

  inference_engine = TorchDynamicShardInferenceEngine(HFShardDownloader())

  output_1 = await inference_engine.infer_prompt("test_id", shard, prompt)
  print("\n------------inference_engine.infer_prompt output---------------\n")
  print(output_1)
  print("\n---------------------------\n")

  assert isinstance(output_1, np.ndarray), "Output should be numpy array"

  output_2 = await inference_engine.infer_tensor("test_id", shard, output_1) 
  print("\n------------inference_engine.infer_tensor output---------------\n")
  print(output_2)
  print("\n---------------------------\n")

  assert isinstance(output_2, np.ndarray), "Output should be numpy array" 

if __name__ == '__main__':
  try:
    print("\n\n -------- TEST llama-3.2-1b -------- \n\n")
    asyncio.run(test_inference_engine())
  except Exception as err:
    print(f"\n!!!! TEST FAILED \n{err}\n")


