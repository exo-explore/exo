"""
Comprehensive unit tests for Forwardersupervisor.
Tests basic functionality, process management, and edge cases.
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Callable, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from master.election_callback import ElectionCallbacks
from master.forwarder_supervisor import (
    ForwarderRole,
    ForwarderSupervisor,
)
from shared.constants import (
    EXO_GLOBAL_EVENT_DB,
    EXO_WORKER_EVENT_DB,
    LIBP2P_GLOBAL_EVENTS_TOPIC,
    LIBP2P_WORKER_EVENTS_TOPIC,
)
from shared.types.common import NodeId

# Mock forwarder script content
MOCK_FORWARDER_SCRIPT = '''#!/usr/bin/env python3
"""Mock forwarder for testing."""
import os
import sys
import time
import signal
from pathlib import Path


def log(message: str) -> None:
    """Write to both stdout and a log file for test verification"""
    print(message, flush=True)
    
    # Also write to a file for test verification
    log_file = os.environ.get("MOCK_LOG_FILE")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"{time.time()}: {message}\\n")


def handle_signal(signum: int, frame: object) -> None:
    """Handle termination signals gracefully"""
    log(f"Received signal {signum}")
    sys.exit(0)


def main() -> None:
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    # Log startup with arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    log(f"Mock forwarder started with args: {args}")
    
    # Write PID file if requested (for testing process management)
    pid_file = os.environ.get("MOCK_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()))
    
    # Check for test control environment variables
    exit_after = os.environ.get("MOCK_EXIT_AFTER")
    exit_code = int(os.environ.get("MOCK_EXIT_CODE", "0"))
    hang_mode = os.environ.get("MOCK_HANG_MODE", "false").lower() == "true"
    ignore_signals = os.environ.get("MOCK_IGNORE_SIGNALS", "false").lower() == "true"
    
    if ignore_signals:
        # Ignore SIGTERM for testing force kill scenarios
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        log("Ignoring SIGTERM signal")
    
    # Simulate work
    start_time = time.time()
    while True:
        if exit_after and (time.time() - start_time) >= float(exit_after):
            log(f"Exiting after {exit_after} seconds with code {exit_code}")
            sys.exit(exit_code)
        
        if hang_mode:
            # Simulate a hanging process (no CPU usage but not responding)
            time.sleep(3600)  # Sleep for an hour
        else:
            # Normal operation - small sleep to not consume CPU
            time.sleep(0.1)


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory and clean it up after test."""
    temp_path = Path(tempfile.mkdtemp(prefix="exo_test_"))
    yield temp_path
    # Clean up
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_forwarder_script(temp_dir: Path) -> Path:
    """Create the mock forwarder executable."""
    mock_script = temp_dir / "mock_forwarder.py"
    mock_script.write_text(MOCK_FORWARDER_SCRIPT)
    mock_script.chmod(0o755)
    return mock_script


@pytest.fixture
def test_logger() -> logging.Logger:
    """Create a test logger."""
    logger = logging.getLogger("test_forwarder")
    logger.setLevel(logging.DEBUG)
    
    # Add console handler for debugging
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


@pytest.fixture
def mock_env_vars(temp_dir: Path) -> dict[str, str]:
    """Environment variables for controlling mock forwarder behavior."""
    return {
        "MOCK_LOG_FILE": str(temp_dir / "mock_forwarder.log"),
        "MOCK_PID_FILE": str(temp_dir / "mock_forwarder.pid"),
    }


@pytest_asyncio.fixture
async def cleanup_processes() -> AsyncGenerator[set[int], None]:
    """Track and cleanup any processes created during tests."""
    tracked_pids: set[int] = set()
    
    yield tracked_pids
    
    # Cleanup any remaining processes - simplified to avoid psutil dependency
    import contextlib
    import subprocess
    for pid in tracked_pids:
        with contextlib.suppress(Exception):
            subprocess.run(["kill", str(pid)], check=False, timeout=1)


@pytest.fixture
def track_subprocess(cleanup_processes: set[int]) -> Callable[[asyncio.subprocess.Process], asyncio.subprocess.Process]:
    """Function to track subprocess PIDs for cleanup."""
    def track(process: asyncio.subprocess.Process) -> asyncio.subprocess.Process:
        if process.pid:
            cleanup_processes.add(process.pid)
        return process
    return track


class TestForwardersupervisorBasic:
    """Basic functionality tests for Forwardersupervisor."""
    
    @pytest.mark.asyncio
    async def test_start_as_replica(
        self,
        mock_forwarder_script: Path,
        mock_env_vars: dict[str, str],
        test_logger: logging.Logger,
        track_subprocess: Callable[[asyncio.subprocess.Process], asyncio.subprocess.Process]
    ) -> None:
        """Test starting forwarder in replica mode."""
        # Set environment
        os.environ.update(mock_env_vars)
        
        supervisor = ForwarderSupervisor(NodeId(), mock_forwarder_script, test_logger)
        await supervisor.start_as_replica()
        
        # Track the process for cleanup
        if supervisor.process:
            track_subprocess(supervisor.process)
        
        try:
            # Verify process is running
            assert supervisor.is_running
            assert supervisor.current_role == ForwarderRole.REPLICA
            
            # Wait a bit for log file to be written
            await asyncio.sleep(0.5)
            
            # Verify forwarding pairs in log
            log_content = Path(mock_env_vars["MOCK_LOG_FILE"]).read_text()
            
            # Expected replica forwarding pairs
            expected_pairs = [
                f"sqlite:{EXO_WORKER_EVENT_DB}:events|libp2p:{LIBP2P_WORKER_EVENTS_TOPIC}",
                f"libp2p:{LIBP2P_GLOBAL_EVENTS_TOPIC}|sqlite:{EXO_GLOBAL_EVENT_DB}:events"
            ]
            
            # Check that the forwarder received the correct arguments
            assert all(pair in log_content for pair in expected_pairs)
            
        finally:
            await supervisor.stop()
            assert not supervisor.is_running

    @pytest.mark.asyncio
    async def test_role_change_replica_to_master(
        self,
        mock_forwarder_script: Path,
        mock_env_vars: dict[str, str],
        test_logger: logging.Logger,
        track_subprocess: Callable[[asyncio.subprocess.Process], asyncio.subprocess.Process]
    ) -> None:
        """Test changing role from replica to master."""
        os.environ.update(mock_env_vars)
        
        supervisor = ForwarderSupervisor(NodeId(), mock_forwarder_script, test_logger)
        await supervisor.start_as_replica()
        
        if supervisor.process:
            track_subprocess(supervisor.process)
        
        try:
            # Change to master
            await supervisor.notify_role_change(ForwarderRole.MASTER)
            
            if supervisor.process:
                track_subprocess(supervisor.process)
            
            # Wait for restart
            await asyncio.sleep(0.5)
            
            assert supervisor.is_running
            assert supervisor.current_role == ForwarderRole.MASTER
            
            # Verify new forwarding pairs
            log_content = Path(mock_env_vars["MOCK_LOG_FILE"]).read_text()
            
            # Expected master forwarding pairs
            master_pairs = [
                f"libp2p:{LIBP2P_WORKER_EVENTS_TOPIC}|sqlite:{EXO_GLOBAL_EVENT_DB}:events",
                f"sqlite:{EXO_GLOBAL_EVENT_DB}:events|libp2p:{LIBP2P_GLOBAL_EVENTS_TOPIC}"
            ]
            
            assert all(pair in log_content for pair in master_pairs)
            
        finally:
            await supervisor.stop()

    @pytest.mark.asyncio
    async def test_idempotent_role_change(
        self,
        mock_forwarder_script: Path,
        mock_env_vars: dict[str, str],
        test_logger: logging.Logger,
        track_subprocess: Callable[[asyncio.subprocess.Process], asyncio.subprocess.Process],
    ) -> None:
        """Test that setting the same role twice doesn't restart the process."""
        os.environ.update(mock_env_vars)
        
        supervisor = ForwarderSupervisor(NodeId(), mock_forwarder_script, test_logger)
        await supervisor.start_as_replica()
        
        original_pid = supervisor.process_pid
        if supervisor.process:
            track_subprocess(supervisor.process)
        
        try:
            # Try to change to the same role
            await supervisor.notify_role_change(ForwarderRole.REPLICA)
            
            # Should not restart (same PID)
            assert supervisor.process_pid == original_pid
            
        finally:
            await supervisor.stop()

    @pytest.mark.asyncio
    async def test_process_crash_and_restart(
        self,
        mock_forwarder_script: Path,
        mock_env_vars: dict[str, str],
        test_logger: logging.Logger,
        track_subprocess: Callable[[asyncio.subprocess.Process], asyncio.subprocess.Process]
    ) -> None:
        """Test that Forwardersupervisor restarts the process if it crashes."""
        # Configure mock to exit after 1 second
        mock_env_vars["MOCK_EXIT_AFTER"] = "1"
        mock_env_vars["MOCK_EXIT_CODE"] = "1"
        os.environ.update(mock_env_vars)
        
        supervisor = ForwarderSupervisor(
            NodeId(),
            mock_forwarder_script,
            test_logger,
            health_check_interval=0.5  # Faster health checks for testing
        )
        await supervisor.start_as_replica()
        
        original_pid = supervisor.process_pid
        if supervisor.process:
            track_subprocess(supervisor.process)
        
        try:
            # Wait for first crash
            await asyncio.sleep(1.5)
            
            # Process should have crashed
            assert not supervisor.is_running or supervisor.process_pid != original_pid
            
            # Clear the crash-inducing environment variables so restart works
            if "MOCK_EXIT_AFTER" in os.environ:
                del os.environ["MOCK_EXIT_AFTER"]
            if "MOCK_EXIT_CODE" in os.environ:
                del os.environ["MOCK_EXIT_CODE"]
            
            # Wait for restart
            await asyncio.sleep(1.0)
            
            # Process should have restarted with new PID
            assert supervisor.is_running
            assert supervisor.process_pid != original_pid
            
            # Track new process
            if supervisor.process:
                track_subprocess(supervisor.process)
            
        finally:
            await supervisor.stop()

    @pytest.mark.asyncio
    async def test_nonexistent_binary(
        self, 
        test_logger: logging.Logger, 
        temp_dir: Path
    ) -> None:
        """Test behavior when forwarder binary doesn't exist."""
        nonexistent_path = temp_dir / "nonexistent_forwarder"
        
        supervisor = ForwarderSupervisor(NodeId(), nonexistent_path, test_logger)
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            await supervisor.start_as_replica()


class TestElectionCallbacks:
    """Test suite for ElectionCallbacks."""
    
    @pytest.mark.asyncio
    async def test_on_became_master(self, test_logger: logging.Logger) -> None:
        """Test callback when becoming master."""
        mock_supervisor = MagicMock(spec=ForwarderSupervisor)
        mock_supervisor.notify_role_change = AsyncMock()
        
        callbacks = ElectionCallbacks(mock_supervisor, test_logger)
        await callbacks.on_became_master()
        
        mock_supervisor.notify_role_change.assert_called_once_with(ForwarderRole.MASTER)  # type: ignore

    @pytest.mark.asyncio
    async def test_on_became_replica(self, test_logger: logging.Logger) -> None:
        """Test callback when becoming replica."""
        mock_supervisor = MagicMock(spec=ForwarderSupervisor)
        mock_supervisor.notify_role_change = AsyncMock()
        
        callbacks = ElectionCallbacks(mock_supervisor, test_logger)
        await callbacks.on_became_replica()
        
        mock_supervisor.notify_role_change.assert_called_once_with(ForwarderRole.REPLICA) # type: ignore