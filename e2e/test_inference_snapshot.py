"""Test: Deterministic inference output (snapshot test).
slow

Sends a chat completion request with a fixed seed,
then verifies the output matches a known-good snapshot. This ensures
inference produces consistent results across runs.

Requires a machine that can run MLX inference at reasonable speed (Apple Silicon).
Run with: python3 e2e/run_all.py --slow  or  E2E_SLOW=1 python3 e2e/run_all.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snapshot import assert_snapshot

from conftest import Cluster

MODEL = "mlx-community/Qwen3-0.6B-4bit"
SEED = 42
PROMPT = "What is 2+2? Reply with just the number."
MAX_TOKENS = 32


async def main():
    async with Cluster("inference_snapshot") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        print(f"  Launching model {MODEL}...")
        await cluster.place_model(MODEL)

        print(f"  Sending chat completion (seed={SEED})...")
        resp = await cluster.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            seed=SEED,
            max_tokens=MAX_TOKENS,
        )

        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content!r}")

        assert_snapshot(
            name="inference_snapshot",
            content=content,
            metadata={
                "model": MODEL,
                "seed": SEED,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
            },
        )

        print("PASSED: inference_snapshot")


if __name__ == "__main__":
    asyncio.run(main())
