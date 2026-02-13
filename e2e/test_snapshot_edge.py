"""Test: Edge case snapshots.
slow

Verifies deterministic output for edge-case prompts: empty-ish input,
very short input, and special characters.
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
MAX_TOKENS = 32

CASES = [
    ("edge_single_word", "Hi"),
    ("edge_special_chars", "What does 2 * (3 + 4) / 7 - 1 equal? Use <math> tags."),
    ("edge_unicode", "Translate 'hello' to Japanese, Chinese, and Korean."),
]


async def main():
    async with Cluster("snapshot_edge") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        print(f"  Launching model {MODEL}...")
        await cluster.place_model(MODEL)

        for snapshot_name, prompt in CASES:
            print(f"  [{snapshot_name}] Sending: {prompt!r}")
            resp = await cluster.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                seed=SEED,
                max_tokens=MAX_TOKENS,
            )

            content = resp["choices"][0]["message"]["content"]
            print(f"  [{snapshot_name}] Response: {content!r}")

            assert_snapshot(
                name=snapshot_name,
                content=content,
                metadata={
                    "model": MODEL,
                    "seed": SEED,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                },
            )

        print("PASSED: snapshot_edge")


if __name__ == "__main__":
    asyncio.run(main())
