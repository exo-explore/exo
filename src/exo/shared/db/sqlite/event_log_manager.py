import asyncio
from typing import Dict, Optional, cast

from loguru import logger
from sqlalchemy.exc import OperationalError

from exo.shared.constants import EXO_HOME
from exo.shared.db.sqlite.config import EventLogConfig, EventLogType
from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.utils.fs import ensure_directory_exists


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
        self._connectors: Dict[EventLogType, AsyncSQLiteEventStorage] = {}

        # Ensure base directory exists
        ensure_directory_exists(EXO_HOME)

    # TODO: This seems like it's a pattern to avoid an async __init__ function. But as we know, there's a better pattern for this - using a create() function, like in runner_supervisor.
    async def initialize(self, max_retries: int = 3) -> None:
        """Initialize both connectors with retry logic - call this during startup"""
        # Both master and worker need both connectors
        for log_type in [EventLogType.WORKER_EVENTS, EventLogType.GLOBAL_EVENTS]:
            retry_count: int = 0
            last_error: Optional[Exception] = None

            while retry_count < max_retries:
                try:
                    await self.get_connector(log_type)
                    break
                except OperationalError as e:
                    last_error = e
                    if "database is locked" in str(e) and retry_count < max_retries - 1:
                        retry_count += 1
                        delay = cast(float, 0.5 * (2**retry_count))
                        logger.warning(
                            f"Database locked while initializing {log_type.value}, retry {retry_count}/{max_retries} after {delay}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.opt(exception=e).error(
                            f"Failed to initialize {log_type.value} after {retry_count + 1} attempts"
                        )
                        raise RuntimeError(
                            f"Could not initialize {log_type.value} database after {retry_count + 1} attempts"
                        ) from e
                except Exception as e:
                    logger.opt(exception=e).error(
                        f"Unexpected error initializing {log_type.value}"
                    )
                    raise

            if retry_count >= max_retries and last_error:
                raise RuntimeError(
                    f"Could not initialize {log_type.value} database after {max_retries} attempts"
                ) from last_error
        logger.bind(user_facing=True).info("Initialized all event log connectors")

    async def get_connector(self, log_type: EventLogType) -> AsyncSQLiteEventStorage:
        """Get or create a connector for the specified log type"""
        if log_type not in self._connectors:
            db_path = self._config.get_db_path(log_type)

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

                self._connectors[log_type] = connector
                logger.bind(user_facing=True).info(
                    f"Initialized {log_type.value} connector at {db_path}"
                )
            except Exception as e:
                logger.bind(user_facing=True).opt(exception=e).error(
                    f"Failed to create {log_type.value} connector"
                )
                raise

        return self._connectors[log_type]

    @property
    def worker_events(self) -> AsyncSQLiteEventStorage:
        """Access worker events log (must call initialize() first)"""
        if EventLogType.WORKER_EVENTS not in self._connectors:
            raise RuntimeError(
                "Event log manager not initialized. Call initialize() first."
            )
        return self._connectors[EventLogType.WORKER_EVENTS]

    @property
    def global_events(self) -> AsyncSQLiteEventStorage:
        """Access global events log (must call initialize() first)"""
        if EventLogType.GLOBAL_EVENTS not in self._connectors:
            raise RuntimeError(
                "Event log manager not initialized. Call initialize() first."
            )
        return self._connectors[EventLogType.GLOBAL_EVENTS]

    async def close_all(self) -> None:
        """Close all open connectors"""
        for log_type, connector in self._connectors.items():
            await connector.close()
            logger.bind(user_facing=True).info(f"Closed {log_type.value} connector")
        self._connectors.clear()
