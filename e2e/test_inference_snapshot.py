"""Test: Deterministic inference output (snapshot test).

Sends a chat completion request with a fixed seed and temperature=0,
then verifies the output matches a known-good snapshot. This ensures
inference produces consistent results across runs.
"""

import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import Cluster

MODEL = "mlx-community/Qwen3-0.6B-4bit"
SEED = 42
PROMPT = "What is 2+2? Reply with just the number."
MAX_TOKENS = 32
SNAPSHOT_FILE = Path(__file__).parent / "snapshots" / "inference.json"


async def main():
    async with Cluster("inference_snapshot") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        # Launch the model instance (triggers download + placement)
        print(f"  Launching model {MODEL}...")
        await cluster.place_model(MODEL)

        print(f"  Sending chat completion (seed={SEED}, temperature=0)...")
        resp = await cluster.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            seed=SEED,
            temperature=0,
            max_tokens=MAX_TOKENS,
        )

        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content!r}")

        # Load or create snapshot
        if SNAPSHOT_FILE.exists():
            snapshot = json.loads(SNAPSHOT_FILE.read_text())
            expected = snapshot["content"]
            assert content == expected, (
                f"Snapshot mismatch!\n"
                f"  Expected: {expected!r}\n"
                f"  Got:      {content!r}\n"
                f"  Delete {SNAPSHOT_FILE} to regenerate."
            )
            print(f"  Output matches snapshot")
        else:
            SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_FILE.write_text(json.dumps({
                "model": MODEL,
                "seed": SEED,
                "temperature": 0,
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "content": content,
            }, indent=2) + "\n")
            print(f"  Snapshot created: {SNAPSHOT_FILE}")

        print("PASSED: inference_snapshot")


if __name__ == "__main__":
    asyncio.run(main())
