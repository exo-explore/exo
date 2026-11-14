"""Real EXO node; only the native executable is a protocol fixture."""

import sys

import exo.main
import exo.utils.info_gatherer.info_gatherer as info_gatherer
import exo.worker.runner.supervisor as supervisor
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.testkit.node import install
from exo.worker.runner.bootstrap import entrypoint


def main() -> None:
    node_id = NodeId(sys.argv[1])
    install(node_id)
    supervisor.entrypoint = entrypoint
    info_gatherer.cuda_memory_available = lambda: Memory.from_gb(8)
    sys.argv = ["exo", *sys.argv[2:]]
    exo.main.main()


if __name__ == "__main__":
    main()
