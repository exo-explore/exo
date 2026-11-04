"""One node for test_wire_gate.py, in its own process: real discovery and zenoh, no exo.

It joins a namespace with ``--wire-version`` and a build, publishes on a test topic for
``--seconds``, and prints one JSON line: how long until it heard another node on the topic
(null if it never did), and the devices on other versions its discovery heard.

The build is ``--build-id``, or with ``--package-version`` that of a source build of that
package version of exo; ``--namespace default`` takes exo's default namespace.
"""

import argparse
import asyncio
import json
import os
import time
from typing import cast

from exo_rs import FromSwarm, NetworkingHandle

TOPIC = "wire_gate_test"


async def run(
    identity: str,
    namespace: str,
    wire_version: int,
    build: str,
    pre_gate_namespace: str | None,
    zenoh_port: int,
    discovery_port: int,
    seconds: float,
) -> None:
    handle = NetworkingHandle.new(
        identity,
        namespace,
        wire_version,
        build,
        pre_gate_namespace,
        zenoh_port,
        discovery_port,
    )
    # Discovery's socket is bound: the next node may start
    print("ready", flush=True)
    start = time.monotonic()
    heard_after: float | None = None

    async def receive() -> None:
        nonlocal heard_after
        while True:
            message = await handle.recv()
            # Only messages from other nodes arrive: zenoh doesn't loop our own back
            if isinstance(message, FromSwarm.Message) and message.topic == TOPIC:
                heard_after = heard_after or time.monotonic() - start

    # The networking handles requests only while something receives from it
    receiving = asyncio.create_task(receive())
    await handle.gossipsub_subscribe(TOPIC)
    while time.monotonic() - start < seconds:
        await handle.gossipsub_publish(TOPIC, b"hello")
        await asyncio.sleep(0.2)
    receiving.cancel()
    report = {
        "heard_after": heard_after,
        "other_versions": handle.devices_on_other_versions(),
    }
    print(json.dumps(report), flush=True)
    # Exit at once: a pending receive from the networking runtime doesn't cancel, and its
    # threads don't stop by themselves
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument(
        "--namespace", required=True, help="or 'default': exo's default"
    )
    parser.add_argument("--wire-version", type=int, required=True)
    parser.add_argument("--build-id", help="the build it claims")
    parser.add_argument(
        "--package-version", help="be a source build of this package version of exo"
    )
    parser.add_argument("--zenoh-port", type=int, required=True)
    parser.add_argument("--discovery-port", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()

    import exo.main
    import exo.routing.router
    import exo.routing.wire

    package_version = cast(str | None, args.package_version)
    if package_version is not None:
        # Each module reads the package version it imported
        exo.main.__version__ = package_version
        exo.routing.wire.__version__ = package_version
        exo.routing.router.__version__ = package_version
    namespace = cast(str, args.namespace)
    if namespace == "default":
        namespace = exo.main.default_namespace({})
    build = cast(str | None, args.build_id) or exo.routing.wire.build_id({})
    pre_gate_namespace = (
        exo.routing.router.__version__
        if namespace == exo.routing.wire.DEFAULT_CLUSTER_NAME
        else None
    )
    asyncio.run(
        run(
            cast(str, args.identity),
            namespace,
            cast(int, args.wire_version),
            build,
            pre_gate_namespace,
            cast(int, args.zenoh_port),
            cast(int, args.discovery_port),
            cast(float, args.seconds),
        )
    )


if __name__ == "__main__":
    main()
