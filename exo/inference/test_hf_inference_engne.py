import pytest
import numpy as np
from exo.inference.huggingface.inference import HuggingfaceInferenceEngine
from exo.inference.shard import Shard
from exo.download.new_shard_download import NewShardDownloader
from loguru import logger


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
    
    print("All tests passed!")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hf_inference_specific()) 
    asyncio.run(test_hf_inference_engine())
