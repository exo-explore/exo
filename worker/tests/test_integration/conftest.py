import asyncio
from logging import Logger
from typing import Awaitable, Callable

import pytest

from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from shared.types.common import NodeId
from worker.download.shard_downloader import NoopShardDownloader
from worker.main import run
from worker.worker import Worker


@pytest.fixture
def user_message():
    """Override this fixture in tests to customize the message"""
    return "What is the capital of Japan?"


@pytest.fixture
def worker_running(logger: Logger) -> Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]]:
    async def _worker_running(node_id: NodeId) -> tuple[Worker, AsyncSQLiteEventStorage]:
        event_log_manager = EventLogManager(EventLogConfig(), logger)
        await event_log_manager.initialize()

        global_events = event_log_manager.global_events
        await global_events.delete_all_events()

        shard_downloader = NoopShardDownloader()
        worker = Worker(node_id, logger=logger, shard_downloader=shard_downloader, worker_events=global_events, global_events=global_events)
        asyncio.create_task(run(worker, logger))

        return worker, global_events

    return _worker_running