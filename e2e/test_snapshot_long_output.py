"""Test: Longer output snapshot.

Verifies deterministic output with a higher max_tokens (128).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snapshot import assert_snapshot

from conftest import Cluster

MODEL = "mlx-community/Qwen3-0.6B-4bit"
SEED = 42
PROMPT = "Explain how a binary search algorithm works."
MAX_TOKENS = 128


async def main():
    async with Cluster("snapshot_long_output") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        print(f"  Launching model {MODEL}...")
        await cluster.place_model(MODEL)

        print(f"  Sending chat completion (seed={SEED}, max_tokens={MAX_TOKENS})...")
        resp = await cluster.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            seed=SEED,
            max_tokens=MAX_TOKENS,
        )

        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content!r}")

        assert_snapshot(
            name="snapshot_long_output",
            content=content,
            metadata={
                "model": MODEL,
                "seed": SEED,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
            },
        )

        print("PASSED: snapshot_long_output")


if __name__ == "__main__":
    asyncio.run(main())
