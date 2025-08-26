# type: ignore

import contextlib
import os
import time
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model

MODEL_ID = "mlx-community/Llama-3.3-70B-Instruct-4bit"
MODEL_PATH = Path(
    os.path.expanduser("~/.exo/models/mlx-community--Llama-3.3-70B-Instruct-4bit/")
)


def _get_model_size_gb(path: str) -> float:
    """Calculate total size of directory recursively in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert bytes to GB


@pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and _get_model_size_gb(MODEL_PATH) > 30),
    reason=f"This test only runs when model {MODEL_ID} is downloaded",
)
def test_mlx_profiling():
    """
    Test MLX generation directly to profile:
    - Time to first token (TTFT)
    - Prefill tokens per second (TPS)
    - Generation tokens per second (TPS)
    For two consecutive prompts using the 70B Llama model.
    """

    # How much memory to keep "wired" (resident) and how much freed memory MLX should keep cached
    info = mx.metal.device_info()  # returns limits & sizes
    # Start conservatively: e.g., 70–90% of recommended working set
    target_bytes = int(0.8 * info["max_recommended_working_set_size"])

    # Keep more freed buffers around for instant reuse
    mx.set_cache_limit(target_bytes)

    # On macOS 15+ you can wire resident memory to avoid OS paging/compression
    with contextlib.suppress(Exception):
        mx.set_wired_limit(target_bytes)

    print(f"\n=== Loading Model {MODEL_ID} ===")
    load_start = time.time()

    # Load model and tokenizer
    model, _ = load_model(MODEL_PATH, lazy=True, strict=False)
    tokenizer = load_tokenizer(MODEL_PATH)

    # Evaluate model parameters to load them into memory
    mx.eval(model.parameters())

    # Create sampler with temperature 0.7
    sampler = make_sampler(temp=0.7)

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Define test prompts
    prompts = [
        "Write me a haiku about a robot.",
        "Please write a haiku about a flower.",
        "Please write a haiku about headlights.",
    ]

    # Prepare messages in chat format
    test_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    results = []

    for i, (messages, prompt_text) in enumerate(
        zip(test_messages, prompts, strict=False), 1
    ):
        print(f"\n=== Prompt {i}: '{prompt_text}' ===")

        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize to count prompt tokens
        prompt_tokens = tokenizer.encode(formatted_prompt)
        num_prompt_tokens = len(prompt_tokens)

        print(f"Prompt tokens: {num_prompt_tokens}")

        # Start timing
        start_time = time.time()
        first_token_time = None
        tokens_generated = 0
        generated_text = ""

        # Stream generate tokens
        for generation in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=100,
            sampler=sampler,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                print(f"Time to first token: {ttft:.3f}s")

            tokens_generated += 1
            generated_text += generation.text

            # Stop if we hit the finish reason
            if generation.finish_reason:
                break

        total_time = time.time() - start_time
        generation_time = total_time - ttft if first_token_time else total_time

        # Calculate metrics
        prefill_tps = num_prompt_tokens / ttft if ttft > 0 else 0
        generation_tps = (
            tokens_generated / generation_time if generation_time > 0 else 0
        )

        # Store results
        result = {
            "prompt": prompt_text,
            "ttft": ttft,
            "total_time": total_time,
            "generation_time": generation_time,
            "prompt_tokens": num_prompt_tokens,
            "tokens_generated": tokens_generated,
            "prefill_tps": prefill_tps,
            "generation_tps": generation_tps,
            "generated_text": generated_text,
        }
        results.append(result)

        # Print results for this prompt
        print(f"Total completion time: {total_time:.3f}s")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Response length: {len(generated_text)} chars")
        print(
            f"Prefill TPS: {prefill_tps:.1f} tokens/sec ({num_prompt_tokens} prompt tokens / {ttft:.3f}s)"
        )
        print(
            f"Generation TPS: {generation_tps:.1f} tokens/sec ({tokens_generated} tokens / {generation_time:.3f}s)"
        )
        print(f"Generated text preview: {generated_text[:100]}...")

        # Small delay between prompts
        if i < len(prompts):
            time.sleep(3.0)

    # Compare results
    print("\n=== Comparison ===")
    if len(results) == 2:
        r1, r2 = results[0], results[1]

        print(f"Second prompt TTFT: {r2['ttft'] / r1['ttft']:.2f}x the first")
        print(
            f"Second prompt prefill TPS: {r2['prefill_tps'] / r1['prefill_tps']:.2f}x the first"
        )
        print(
            f"Second prompt generation TPS: {r2['generation_tps'] / r1['generation_tps']:.2f}x the first"
        )

        # Performance expectations
        print("\n=== Performance Summary ===")
        print("First prompt:")
        print(f"  TTFT: {r1['ttft']:.3f}s")
        print(f"  Prefill: {r1['prefill_tps']:.1f} tok/s")
        print(f"  Generation: {r1['generation_tps']:.1f} tok/s")

        print("Second prompt (warmed up):")
        print(f"  TTFT: {r2['ttft']:.3f}s")
        print(f"  Prefill: {r2['prefill_tps']:.1f} tok/s")
        print(f"  Generation: {r2['generation_tps']:.1f} tok/s")

    # Basic assertions
    for result in results:
        assert result["ttft"] > 0, "TTFT must be positive"
        assert result["tokens_generated"] > 0, "Must generate at least one token"
        assert len(result["generated_text"]) > 0, "Must generate some text"
        assert result["prefill_tps"] > 0, "Prefill TPS must be positive"
        assert result["generation_tps"] > 0, "Generation TPS must be positive"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_mlx_profiling()
