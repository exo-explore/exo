import asyncio
from pathlib import Path
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine

async def test_model():
    # Define a test shard for llama-3.1-8b which supports tinygrad
    shard = Shard(
        model_id="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",  
        start_layer=0,
        end_layer=31,  
        n_layers=32  
    )

    # Initialize the shard downloader (no need for model_path as it downloads automatically)
    shard_downloader = HFShardDownloader()

    # Create the tinygrad inference engine
    engine = get_inference_engine("tinygrad", shard_downloader)

    # Test prompt
    prompt = "Write a short poem about AI:"
    
    # Run inference
    result, state, is_eos = await engine.infer_prompt("test1", shard, prompt)
    
    # Get the token prediction
    if result.shape == (1, 1):
        next_token = await engine.tokenizer.decode(result[0])
        print(f"Input prompt: {prompt}")
        print(f"Next token: {next_token}")
        print(f"Is end of sequence: {is_eos}")
    else:
        print("Unexpected output shape:", result.shape)

if __name__ == "__main__":
    asyncio.run(test_model())