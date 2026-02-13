"""Test: Basic cluster formation.

Verifies two nodes discover each other, elect a master, and the API responds.
"""

import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from conftest import Cluster


async def main():
    async with Cluster("cluster_formation") as cluster:
        await cluster.build()
        await cluster.start()
        await cluster.assert_healthy()
        print("PASSED: cluster_formation")


if __name__ == "__main__":
    asyncio.run(main())
