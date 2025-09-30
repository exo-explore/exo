import asyncio
from typing import cast

from loguru import logger
from sqlalchemy.exc import OperationalError

from exo.shared.constants import EXO_HOME
from exo.shared.db.config import EventLogConfig
from exo.shared.db.connector import AsyncSQLiteEventStorage
from exo.utils.fs import ensure_directory_exists


class EventLogManager:
    """
    Manages both worker and global event log connectors.
    Used by both master and worker processes with different access patterns:

    - Worker: writes to worker_events, tails global_events
    - Master (elected): writes to global_events, tails global_events
    - Master (replica): writes to worker_events, tails global_events
    """

    def __init__(self, config: EventLogConfig):
        self._config = config
        self._connector: AsyncSQLiteEventStorage | None = None

        # Ensure base directory exists
        ensure_directory_exists(EXO_HOME)

    # TODO: This seems like it's a pattern to avoid an async __init__ function. But as we know, there's a better pattern for this - using a create() function, like in runner_supervisor.
    async def initialize(self, max_retries: int = 3) -> None:
        """Initialize both connectors with retry logic - call this during startup"""
        # Both master and worker need both connectors
        retry_count: int = 0
        last_error: Exception | None = None

        while retry_count < max_retries:
            try:
                await self.get_connector()
                break
            except OperationalError as e:
                last_error = e
                if "database is locked" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    delay = cast(float, 0.5 * (2**retry_count))
                    logger.warning(
                        f"Database locked while initializing db, retry {retry_count}/{max_retries} after {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.opt(exception=e).error(
                        f"Failed to initialize db after {retry_count + 1} attempts"
                    )
                    raise RuntimeError(
                        f"Could not initialize db after {retry_count + 1} attempts"
                    ) from e
            except Exception as e:
                logger.opt(exception=e).error("Unexpected error initializing db")
                raise

        if retry_count >= max_retries and last_error:
            raise RuntimeError(
                f"Could not initialize db after {max_retries} attempts"
            ) from last_error
        logger.bind(user_facing=True).info("Initialized all event log connectors")

    async def get_connector(self) -> AsyncSQLiteEventStorage:
        """Get or create a connector for the specified log type"""
        if not self._connector:
            db_path = self._config.get_db_path()

            try:
                connector = AsyncSQLiteEventStorage(
                    db_path=db_path,
                    batch_size=self._config.batch_size,
                    batch_timeout_ms=self._config.batch_timeout_ms,
                    debounce_ms=self._config.debounce_ms,
                    max_age_ms=self._config.max_age_ms,
                )

                # Start the connector (creates tables if needed)
                await connector.start()

                self._connector = connector
                logger.bind(user_facing=True).info(
                    f"Initialized db connector at {db_path}"
                )
            except Exception as e:
                logger.bind(user_facing=True).opt(exception=e).error(
                    "Failed to create db connector"
                )
                raise

        return self._connector

    @property
    def events(self) -> AsyncSQLiteEventStorage:
        """Access event log (must call initialize() first)"""
        if not self._connector:
            raise RuntimeError(
                "Event log manager not initialized. Call initialize() first."
            )
        return self._connector

    async def close(self) -> None:
        """Close all open connectors"""
        assert self._connector is not None
        await self._connector.close()
        logger.bind(user_facing=True).info("Closed db connector")
        self._connector = None
