
import asyncio
from exo.inference.shard import Shard
from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.helpers import DEBUG
import os
import numpy as np

async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str, n_layers: int):
    # prompt = "Why is the sky blue?"
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

    next_resp_full = resp_full
    is_finished = False
    while not is_finished:
        next_resp_full, _next_inference_state_full, is_finished = await inference_engine_1.infer_tensor(
            "A",
            shard=shard,
            input_data=next_resp_full,
            inference_state=inference_state_full,
        )

        print("\n------------next_resp_full---------------\n")
        print(next_resp_full)
        print("\n------------next_resp_full---------------\n")
    

   

if __name__ == '__main__':
    # try:
    #     print(f"\n\n -------- TEST QWEN2 -------- \n\n")
    #     asyncio.run(test_inference_engine(
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         "Qwen/Qwen2-0.5B-Instruct",
    #         24
    #     ))
    # except Exception as err:
    #     print(f"\n\n !!!!!!!!!!! QWEN2 TEST FAILED \n{err}\n")

    # try:
    #     print(f"\n\n -------- TEST LLAMA3-1B-Base -------- \n\n")
    #     asyncio.run(test_inference_engine(
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         "andrijdavid/Llama3-1B-Base",
    #         3
    #     ))
    # except Exception as err:
    #     print(f"\n\n !!!!!!!!!!! LLAMA3-1B-Base TEST FAILED \n{err}\n")

    # try:
    #     print(f"\n\n -------- TEST META LLAMA 3.1 8B -------- \n\n")
    #     asyncio.run(test_inference_engine(
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         "meta-llama/Meta-Llama-3.1-8B",
    #         32
    #     ))
    # except Exception as err:
    #     print(f"\n\n !!!!!!!!!!! META LLAMA 3.1 8B TEST FAILED \n{err}\n")

    # try:
    #     print(f"\n\n ------- TEST Chickaboo/ChickaQ-Large -----\n\n")
    #     asyncio.run(test_inference_engine(
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
    #         "Chickaboo/ChickaQ-Large",
    #         24
    #     ))
    # except Exception as err:
    #     print(f"\n\n !!!!!!!!!!! Chickaboo/ChickaQ-Large TEST FAILED \n{err}\n")
    
    try:
        print(f"\n\n --------- TEST ambrosfitz/TinyLlama-1.1B-Chat-yawp -------\n\n")
        asyncio.run(test_inference_engine(
            PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
            PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
            "ambrosfitz/TinyLlama-1.1B-Chat-yawp",
            22
        ))
    except Exception as err:
        print(f"\n\n !!!!!!!!!!! ambrosfitz/TinyLlama-1.1B-Chat-yawp TEST FAILED \n{err}\n")

