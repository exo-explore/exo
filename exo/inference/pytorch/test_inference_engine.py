
import asyncio
from exo.inference.shard import Shard
from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine

def main():
    shard = Shard(
        model_id="meta-llama/Meta-Llama-3.1-8B",
        start_layer=0,
        end_layer=0,
        n_layers=12
    )

    engine = PyTorchDynamicShardInferenceEngine(
        shard
    )

   
    # Prepare the prompt
    prompt = "Why is the sky blue?"

    # Run inference
    loop = asyncio.get_event_loop()
    output_data, new_inference_state, is_eos = loop.run_until_complete(
        engine.infer_prompt(
            request_id="test_request", shard=shard, prompt=prompt
        )
    )

    assert output_data is not None

if __name__ == '__main__':
    main()
