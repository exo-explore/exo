"""Test: Runner chaos — abrupt runner death detection.
slow

Sends a chat completion with the EXO_RUNNER_MUST_DIE trigger, which causes
the runner process to call os._exit(1) (simulating an OOM kill). Verifies that
the RunnerSupervisor health check detects the death and the system doesn't hang.

Requires a machine that can run MLX inference at reasonable speed (Apple Silicon).
Run with: python3 e2e/run_all.py --slow  or  E2E_SLOW=1 python3 e2e/run_all.py
"""

import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest import Cluster

MODEL = "mlx-community/Qwen3-0.6B-4bit"


async def main():
    async with Cluster("runner_chaos") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()

        # Place the model so a runner is loaded and ready
        print(f"  Launching model {MODEL}...")
        await cluster.place_model(MODEL)

        # Send a chat request with the die trigger.
        # The runner will call os._exit(1) mid-inference, simulating OOM kill.
        # The chat request itself will fail — that's expected.
        print("  Sending EXO_RUNNER_MUST_DIE trigger...")
        with contextlib.suppress(Exception):
            await cluster.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "EXO RUNNER MUST DIE"}],
                timeout=60,
            )

        # Wait for the health check to detect the death and emit RunnerFailed
        async def health_check_detected():
            log = await cluster.logs()
            return "runner process died unexpectedly" in log

        await cluster.wait_for(
            "Health check detected runner death",
            health_check_detected,
            timeout=30,
        )

        # Verify RunnerFailed was emitted (visible in logs)
        log = await cluster.logs()
        assert "runner process died unexpectedly" in log, (
            f"Expected health check to detect runner death but it didn't.\nLogs:\n{log}"
        )

        print("PASSED: runner_chaos")


if __name__ == "__main__":
    asyncio.run(main())
