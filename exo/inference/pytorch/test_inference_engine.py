
import asyncio
from exo.inference.shard import Shard
from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.helpers import DEBUG
import os
import numpy as np

async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str):
    prompt = "In a single word only, what is the last name of the current president of the USA?"
    resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt)
    next_resp_full, _next_inference_state_full, _ = await inference_engine_1.infer_tensor(
        "A",
        shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32),
        input_data=resp_full,
        inference_state=inference_state_full,
    )

    pp = 15
    resp1, inference_state_1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32), prompt=prompt)
    resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(
        "B",
        shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
        input_data=resp1,
        inference_state=inference_state_1,
    )
    resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(
        "B",
        shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32),
        input_data=resp2,
        inference_state=inference_state_2,
    )
    resp4, _inference_state_4, _ = await inference_engine_2.infer_tensor(
        "B",
        shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
        input_data=resp3,
        inference_state=inference_state_3,
    )

    assert np.array_equal(resp_full, resp2)
    assert np.array_equal(next_resp_full, resp4)

def single_test():
    shard = Shard(
        model_id="meta-llama/Meta-Llama-3.1-8B",
        start_layer=0,
        end_layer=0,
        n_layers=32
    )

    engine = PyTorchDynamicShardInferenceEngine(shard)

   
    # Prepare the prompt
    prompt = "Why is the sky blue?"

    # Run inference
    loop = asyncio.get_event_loop()
    output_data, _, _ = loop.run_until_complete(
        engine.infer_prompt(
            request_id="test_request", shard=shard, prompt=prompt
        )
    )

    assert output_data is not None

if __name__ == '__main__':
    # single_test()
    asyncio.run(test_inference_engine(
        PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
        PyTorchDynamicShardInferenceEngine(HFShardDownloader()),
        "andrijdavid/Llama3-2B-Base",
    ))

