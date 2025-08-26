import asyncio
from pathlib import Path

from loguru import logger

from exo.shared.apply import apply
from exo.shared.constants import EXO_WORKER_LOG
from exo.shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from exo.shared.keypair import Keypair, get_node_id_keypair
from exo.shared.logging import logger_setup, logger_cleanup
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    NodePerformanceMeasured,
)
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.worker.ops import (
    ExecuteTaskOp,
    RunnerOp,
)
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.plan import plan
from exo.worker.utils.profile import start_polling_node_metrics
from exo.worker.worker import Worker


async def run(worker: Worker):
    assert worker.global_events is not None

    while True:
        # 1. get latest events
        events = await worker.global_events.get_events_since(
            worker.state.last_event_applied_idx
        )

        # 2. for each event, apply it to the state and run sagas
        for event_from_log in events:
            worker.state = apply(worker.state, event_from_log)

        # 3. based on the updated state, we plan & execute an operation.
        op: RunnerOp | None = plan(
            worker.assigned_runners,
            worker.node_id,
            worker.state.instances,
            worker.state.runners,
            worker.state.tasks,
        )

        # run the op, synchronously blocking for now
        if op is not None:
            logger.info(f"Executing op {op}")
            logger.bind(user_facing=True).debug(f"Worker executing op: {op}")
            try:
                async for event in worker.execute_op(op):
                    await worker.event_publisher(event)
            except Exception as e:
                if isinstance(op, ExecuteTaskOp):
                    generator = worker.fail_task(
                        e, runner_id=op.runner_id, task_id=op.task.task_id
                    )
                else:
                    generator = worker.fail_runner(e, runner_id=op.runner_id)

                async for event in generator:
                    await worker.event_publisher(event)

        await asyncio.sleep(0.01)


async def async_main():
    node_id_keypair: Keypair = get_node_id_keypair()
    node_id = NodeId(node_id_keypair.to_peer_id().to_base58())

    event_log_manager = EventLogManager(EventLogConfig())
    await event_log_manager.initialize()
    shard_downloader = exo_shard_downloader()

    # TODO: add profiling etc to resource monitor
    async def resource_monitor_callback(
        node_performance_profile: NodePerformanceProfile,
    ) -> None:
        await event_log_manager.worker_events.append_events(
            [
                NodePerformanceMeasured(
                    node_id=node_id, node_profile=node_performance_profile
                )
            ],
            origin=node_id,
        )

    asyncio.create_task(start_polling_node_metrics(callback=resource_monitor_callback))

    worker = Worker(
        node_id,
        shard_downloader,
        event_log_manager.worker_events,
        event_log_manager.global_events,
    )

    await run(worker)
    logger_cleanup()


def main(logfile: Path = EXO_WORKER_LOG, verbosity: int = 1):
    logger_setup(logfile, verbosity)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
