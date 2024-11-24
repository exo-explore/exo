"""
Test inference engine and model sharding
"""
import pytest
import asyncio

from exo.inference.shard import Shard
from exo.inference.torch.pt_inference import TorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine

import numpy as np

@pytest.mark.asyncio
async def test_inference_engine():
  prompt = "In a single word only, what is the last name of the current president of the USA?"

  shard = Shard(
    model_id="llama-3.2-1b",
    start_layer=0,
    end_layer=15,
    n_layers=16
  )

  inference_engine = TorchDynamicShardInferenceEngine(HFShardDownloader())

  output = await inference_engine.infer_prompt("test_id", shard, prompt)
  print("\n------------inference_engine output---------------\n")
  print(output)
  print("\n---------------------------\n")

  assert isinstance(output, np.ndarray), "Output should be numpy array"

if __name__ == '__main__':
  try:
    print("\n\n -------- TEST unsloth/Llama-3.2-1B-Instruct -------- \n\n")
    asyncio.run(test_inference_engine())
  except Exception as err:
    print(f"\n!!!! LLAMA TEST FAILED \n{err}\n")


