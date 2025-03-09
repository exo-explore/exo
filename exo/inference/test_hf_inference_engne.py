import pytest
import numpy as np
from exo.inference.huggingface.inference import HuggingfaceInferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.new_shard_download import NewShardDownloader
from loguru import logger
import torch
import asyncio
import time


@pytest.mark.asyncio
async def test_hf_inference_specific():
    test_shard = Shard(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", start_layer=0, end_layer=24, n_layers=24)
    test_prompt = "This is a test prompt"
    engine = HuggingfaceInferenceEngine(test_shard)

    result, _ = await engine.infer_prompt("test_request", test_shard, test_prompt)
    
    # result = await engine.decode(test_shard, result)
    
    print(f"Inference result shape: {result.shape}")
    
    assert result.shape[0] == 1, "Result should be a 2D array with first dimension 1"
    
    print("Test passed!")
    
    
@pytest.mark.asyncio
async def test_hf_inference_engine():
    # Initialize the HuggingFaceInferenceEngine
    test_shard = Shard(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", start_layer=0, end_layer=24, n_layers=24)

    engine = HuggingfaceInferenceEngine(test_shard)
        
    # Test infer_prompt
    output, _ = await engine.infer_prompt("test_id", test_shard, "Test prompt")
    
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.ndim == 2, "Output should be 2-dimensional"
    logger.info(f" infer_prompt tested")
    # Test infer_tensor
    input_tensor = np.array([[1, 2, 3]])
    output, _ = await engine.infer_tensor("test_id", test_shard, input_tensor)
    
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.ndim == 2, "Output should be 2-dimensional"
    
    print("test 2 passed!")
    
@pytest.mark.asyncio
async def test_inference_engine(model_id: str, n_layers: int):
    """" Final test to check whether the inference engine works as expected """

    torch.manual_seed(42)
    np.random.seed(42)
    
    shard_1 = Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers)

    inference_engine_1 = HuggingfaceInferenceEngine(shard_1)
    inference_engine_2 = HuggingfaceInferenceEngine(shard_1)

    prompt = "In a single word only, what is the last name of the current president of the USA?"
    
    resp_full, _ = await inference_engine_1.infer_prompt("resp_full", shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers), prompt=prompt)
    token_full = await inference_engine_1.sample(resp_full)
    token_full = token_full.reshape(1, -1)
    next_resp_full, _ = await inference_engine_1.infer_tensor(
        "resp_full to next_resp_full",
        shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers),
        input_data=token_full,
    )

    pp = n_layers // 2
    shard_1 = Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers)
    shard_2 = Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers)
    resp1, state = await inference_engine_1.infer_prompt("resp_1", shard_1, prompt=prompt, inference_state={})
    resp2, _ = await inference_engine_2.infer_tensor("resp_2", shard=shard_2, input_data=resp1, inference_state=state)
    
    tokens2 = await inference_engine_1.sample(resp2)
    tokens2 = tokens2.reshape(1, -1)
    resp3, state = await inference_engine_1.infer_tensor(
        "resp_3",
        shard=shard_1,
        input_data=tokens2,
        inference_state={}
    )
    resp4, _ = await inference_engine_2.infer_tensor(
        "resp_4",
        shard=shard_2,
        input_data=resp3,
        inference_state=state
    )

    print(f"resp_full shape: {resp_full.shape}, resp2 shape: {resp2.shape}")
    print(f"next_resp_full shape: {next_resp_full.shape}, resp4 shape: {resp4.shape}")

    # Convert tensors to numpy and check if they're close
    resp_full_np = resp_full.detach().cpu().numpy()
    resp2_np = resp2.detach().cpu().numpy()
    next_resp_full_np = next_resp_full.detach().cpu().numpy()
    resp4_np = resp4.detach().cpu().numpy()

    def check_tensor_similarity(t1, t2):
        # Check shape
        assert t1.shape == t2.shape, "Shapes don't match"
        
        # Check statistical properties
        mean_diff = np.abs(np.mean(t1) - np.mean(t2))
        std_diff = np.abs(np.std(t1) - np.std(t2))
        
        # More relaxed thresholds
        assert mean_diff < 0.1, f"Mean difference too high: {mean_diff}"
        assert std_diff < 0.1, f"Standard deviation difference too high: {std_diff}"
    
    check_tensor_similarity(resp_full_np, resp2_np)
    check_tensor_similarity(next_resp_full_np, resp4_np) 
    print("Test passed!")
    
if __name__ == "__main__":
    import asyncio
    # asyncio.run(test_hf_inference_specific()) 
    # asyncio.run(test_hf_inference_engine())
    asyncio.run(test_inference_engine("HuggingFaceTB/SmolLM2-1.7B-Instruct", 24))
    
