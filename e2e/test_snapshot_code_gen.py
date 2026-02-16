"""Test: Code generation snapshot.
slow

Verifies deterministic output for a code generation prompt.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snapshot import assert_snapshot

from conftest import Cluster

MODEL = "mlx-community/Qwen3-0.6B-4bit"
SEED = 42
PROMPT = (
    "Write a Python function to reverse a string. Only output the code, no explanation."
)
MAX_TOKENS = 64


async def main():
    async with Cluster("snapshot_code_gen") as cluster:
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
            temperature=0,
            max_tokens=MAX_TOKENS,
        )

        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content!r}")

        assert_snapshot(
            name="snapshot_code_gen",
            content=content,
            metadata={
                "model": MODEL,
                "seed": SEED,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
            },
        )

        print("PASSED: snapshot_code_gen")


if __name__ == "__main__":
    asyncio.run(main())
