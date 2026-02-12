"""Test: Cluster works without internet access.

Verifies exo functions correctly when containers can talk to each other
but cannot reach the internet (Docker internal network).
"""

import asyncio
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from conftest import Cluster


async def main():
    async with Cluster(
        "no_internet",
        overrides=["tests/no_internet/docker-compose.override.yml"],
    ) as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()
        print("PASSED: no_internet")


if __name__ == "__main__":
    asyncio.run(main())
