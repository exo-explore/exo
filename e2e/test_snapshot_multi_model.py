"""Test: Multi-model snapshot tests.
slow

Verifies deterministic output across different model architectures to catch
model-specific regressions. Each model uses its own snapshot file.
Run with: python3 e2e/run_all.py --slow  or  E2E_SLOW=1 python3 e2e/run_all.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snapshot import assert_snapshot

from conftest import Cluster

SEED = 42
PROMPT = "What is the capital of France?"
MAX_TOKENS = 32

MODELS = [
    "mlx-community/SmolLM2-135M-Instruct",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
]


async def main():
    async with Cluster("snapshot_multi_model") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        for model in MODELS:
            short_name = (
                model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            )
            snapshot_name = f"snapshot_multi_{short_name}"

            print(f"  Launching model {model}...")
            await cluster.place_model(model)

            print(f"  Sending chat completion (seed={SEED})...")
            resp = await cluster.chat(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                seed=SEED,
                max_tokens=MAX_TOKENS,
            )

            content = resp["choices"][0]["message"]["content"]
            print(f"  [{short_name}] Response: {content!r}")

            assert_snapshot(
                name=snapshot_name,
                content=content,
                metadata={
                    "model": model,
                    "seed": SEED,
                    "prompt": PROMPT,
                    "max_tokens": MAX_TOKENS,
                },
            )

            print(f"  [{short_name}] PASSED")

        print("PASSED: snapshot_multi_model")


if __name__ == "__main__":
    asyncio.run(main())
