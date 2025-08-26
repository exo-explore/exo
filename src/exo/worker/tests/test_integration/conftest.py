import asyncio
from logging import Logger
from typing import Awaitable, Callable

import pytest

from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from exo.shared.logging import logger_test_install
from exo.shared.types.common import NodeId
from exo.worker.download.shard_downloader import NoopShardDownloader
from exo.worker.main import run
from exo.worker.worker import Worker


@pytest.fixture
def worker_running(
    logger: Logger,
) -> Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]]:
    async def _worker_running(
        node_id: NodeId,
    ) -> tuple[Worker, AsyncSQLiteEventStorage]:
        logger_test_install(logger)
        event_log_manager = EventLogManager(EventLogConfig())
        await event_log_manager.initialize()

        global_events = event_log_manager.global_events
        await global_events.delete_all_events()

        shard_downloader = NoopShardDownloader()
        worker = Worker(
            node_id,
            shard_downloader=shard_downloader,
            worker_events=global_events,
            global_events=global_events,
        )
        asyncio.create_task(run(worker))

        return worker, global_events

    return _worker_running
