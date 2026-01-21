"""Pytest configuration and shared fixtures for shared package tests."""

import asyncio
from typing import Generator

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_event_loop():
    """Reset the event loop for each test to ensure clean state."""
    # This ensures each test gets a fresh event loop state


def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100000),
            n_layers=32,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=True,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)
