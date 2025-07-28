import asyncio
import contextlib
from enum import Enum
from logging import Logger
from pathlib import Path

from shared.constants import (
    EXO_GLOBAL_EVENT_DB,
    EXO_WORKER_EVENT_DB,
    LIBP2P_GLOBAL_EVENTS_TOPIC,
    LIBP2P_WORKER_EVENTS_TOPIC,
)
from shared.types.common import NodeId


class ForwarderRole(str, Enum):
    """Role determines which forwarding pairs to use"""
    MASTER = "master"
    REPLICA = "replica"


class ForwarderSupervisor:
    """
    Manages the forwarder subprocess for SQLite ↔ libp2p event forwarding.
    The forwarder is a single process that handles multiple forwarding pairs.
    
    Master mode forwards:
    - sqlite:worker_events.db:events → libp2p:worker_events (share local worker events)
    - libp2p:worker_events → sqlite:global_events.db:events (collect network worker events)  
    - sqlite:global_events.db:events → libp2p:global_events (broadcast merged global log)
    
    Replica mode forwards:
    - sqlite:worker_events.db:events → libp2p:worker_events (share local worker events)
    - libp2p:global_events → sqlite:global_events.db:events (receive global log from master)
    """
    
    def __init__(
        self, 
        node_id: NodeId,
        forwarder_binary_path: Path,
        logger: Logger,
        health_check_interval: float = 5.0
    ):
        self.node_id = node_id
        self._binary_path = forwarder_binary_path
        self._logger = logger
        self._health_check_interval = health_check_interval
        self._current_role: ForwarderRole | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._health_check_task: asyncio.Task[None] | None = None
        
    async def notify_role_change(self, new_role: ForwarderRole) -> None:
        """
        Called by external systems (e.g., election handler) when role changes.
        This is the main public interface.
        """
        if self._current_role == new_role:
            self._logger.debug(f"Role unchanged: {new_role}")
            return
            
        self._logger.info(f"Role changing from {self._current_role} to {new_role}")
        self._current_role = new_role
        await self._restart_with_role(new_role)
    
    async def start_as_replica(self) -> None:
        """Convenience method to start in replica mode"""
        await self.notify_role_change(ForwarderRole.REPLICA)
    
    async def stop(self) -> None:
        """Stop forwarder and cleanup"""
        await self._stop_process()
        self._current_role = None
    
    def _get_forwarding_pairs(self, role: ForwarderRole) -> str:
        """
        Generate forwarding pairs based on role.
        Returns list of "source,sink" strings.
        """
        pairs: list[str] = []
        
        # Both master and replica forward local worker events to network
        pairs.append(
            f"sqlite:{EXO_WORKER_EVENT_DB}:events|libp2p:{LIBP2P_WORKER_EVENTS_TOPIC}"
        )
        
        if role == ForwarderRole.MASTER:
            # Master: collect worker events from network into global log
            pairs.append(
                f"libp2p:{LIBP2P_WORKER_EVENTS_TOPIC}|sqlite:{EXO_GLOBAL_EVENT_DB}:events"
            )
            # Master: broadcast global events to network
            pairs.append(
                f"sqlite:{EXO_GLOBAL_EVENT_DB}:events|libp2p:{LIBP2P_GLOBAL_EVENTS_TOPIC}"
            )
        else:  # REPLICA
            # Replica: receive global events from master
            pairs.append(
                f"libp2p:{LIBP2P_GLOBAL_EVENTS_TOPIC}|sqlite:{EXO_GLOBAL_EVENT_DB}:events"
            )
        
        return ','.join(pairs)
    
    async def _restart_with_role(self, role: ForwarderRole) -> None:
        """Internal method to restart forwarder with new role"""
        await self._stop_process()
        
        
        pairs: str = self._get_forwarding_pairs(role)
        self._process = await asyncio.create_subprocess_exec(
            str(self._binary_path),
            f'{pairs}',
            stdout=None,
            stderr=None,
            env={
                "FORWARDER_NODE_ID": str(self.node_id),
            }
        )
        
        self._logger.info(f"Starting forwarder with forwarding pairs: {pairs}")
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(
            self._monitor_health()
        )
    
    async def _stop_process(self) -> None:
        """Stop the forwarder process gracefully"""
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None
        
        if self._process:
            # Check if process is already dead
            if self._process.returncode is None:
                # Process is still alive, terminate it
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._logger.warning("Forwarder didn't terminate, killing")
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    # Process already dead
                    pass
            self._process = None
    
    async def _monitor_health(self) -> None:
        """Monitor process health and restart if it crashes"""
        while self._process and self._current_role:
            try:
                # Check if process is still alive
                retcode = await asyncio.wait_for(
                    self._process.wait(),
                    timeout=self._health_check_interval
                )
                # Process exited
                self._logger.error(f"Forwarder exited with code {retcode}")
                
                # Auto-restart
                await asyncio.sleep(0.2)  # Brief delay before restart
                if self._current_role:  # Still have a role
                    await self._restart_with_role(self._current_role)
                break
                
            except asyncio.TimeoutError:
                # Process still running, continue monitoring
                continue
            except asyncio.CancelledError:
                break
    
    @property
    def is_running(self) -> bool:
        """Check if forwarder process is running"""
        return self._process is not None and self._process.returncode is None
    
    @property
    def current_role(self) -> ForwarderRole | None:
        """Get current forwarder role (for testing)"""
        return self._current_role
    
    @property
    def process_pid(self) -> int | None:
        """Get current process PID (for testing)"""
        return self._process.pid if self._process else None
    
    @property
    def process(self) -> asyncio.subprocess.Process | None:
        """Get current process (for testing)"""
        return self._process
