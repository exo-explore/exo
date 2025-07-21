from logging import Logger
from typing import Dict

from shared.constants import EXO_HOME
from shared.db.sqlite.config import EventLogConfig, EventLogType
from shared.db.sqlite.connector import AsyncSQLiteEventStorage


class EventLogManager:
    """
    Manages both worker and global event log connectors.
    Used by both master and worker processes with different access patterns:
    
    - Worker: writes to worker_events, tails global_events
    - Master (elected): writes to global_events, tails global_events
    - Master (replica): writes to worker_events, tails global_events
    """
    
    def __init__(self, config: EventLogConfig, logger: Logger):
        self._config = config
        self._logger = logger
        self._connectors: Dict[EventLogType, AsyncSQLiteEventStorage] = {}
        
        # Ensure base directory exists
        EXO_HOME.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize both connectors - call this during startup"""
        # Both master and worker need both connectors
        await self.get_connector(EventLogType.WORKER_EVENTS)
        await self.get_connector(EventLogType.GLOBAL_EVENTS)
        self._logger.info("Initialized all event log connectors")
    
    async def get_connector(self, log_type: EventLogType) -> AsyncSQLiteEventStorage:
        """Get or create a connector for the specified log type"""
        if log_type not in self._connectors:
            db_path = self._config.get_db_path(log_type)
            
            connector = AsyncSQLiteEventStorage(
                db_path=db_path,
                batch_size=self._config.batch_size,
                batch_timeout_ms=self._config.batch_timeout_ms,
                debounce_ms=self._config.debounce_ms,
                max_age_ms=self._config.max_age_ms,
                logger=self._logger
            )
            
            # Start the connector (creates tables if needed)
            await connector.start()
            
            self._connectors[log_type] = connector
            self._logger.info(f"Initialized {log_type.value} connector at {db_path}")
        
        return self._connectors[log_type]
    
    @property
    def worker_events(self) -> AsyncSQLiteEventStorage:
        """Access worker events log (must call initialize() first)"""
        if EventLogType.WORKER_EVENTS not in self._connectors:
            raise RuntimeError("Event log manager not initialized. Call initialize() first.")
        return self._connectors[EventLogType.WORKER_EVENTS]
    
    @property
    def global_events(self) -> AsyncSQLiteEventStorage:
        """Access global events log (must call initialize() first)"""
        if EventLogType.GLOBAL_EVENTS not in self._connectors:
            raise RuntimeError("Event log manager not initialized. Call initialize() first.")
        return self._connectors[EventLogType.GLOBAL_EVENTS]
    
    async def close_all(self) -> None:
        """Close all open connectors"""
        for log_type, connector in self._connectors.items():
            await connector.close()
            self._logger.info(f"Closed {log_type.value} connector")
        self._connectors.clear()