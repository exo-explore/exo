"""Test: Cluster works without internet access.

Verifies exo functions correctly when containers can talk to each other
but cannot reach the internet. Uses iptables to block all outbound traffic
except private subnets and multicast (for mDNS discovery).
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

        # Verify internet is actually blocked from inside the containers
        for node in ["exo-node-1", "exo-node-2"]:
            rc, _ = await cluster.exec(node, "curl", "-sf", "--max-time", "3", "https://huggingface.co", check=False)
            assert rc != 0, f"{node} should not be able to reach the internet"
            print(f"  {node}: internet correctly blocked")

        # Verify exo detected no internet connectivity
        log = await cluster.logs()
        assert "Internet connectivity: False" in log, "exo should detect no internet"
        print("  exo correctly detected no internet connectivity")

        print("PASSED: no_internet")


if __name__ == "__main__":
    asyncio.run(main())
