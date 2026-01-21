"""FLASH MPI Runner - spawns and monitors FLASH simulations.

Exo-native distributed MPI:
- Exo handles node discovery and coordination
- Coordinator generates hostfile from Exo topology
- mpirun uses exo-rsh (no SSH required) to spawn on remote nodes
- exo-rsh connects to each node's Exo API (/execute endpoint) for remote execution
- Workers just report ready and wait
"""
# ruff: noqa: I001 - Import order intentional (plugin types before shared types)

import os
import shutil
import socket
import subprocess
import threading

from loguru import logger

from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.plugins.implementations.flash.types import FLASHInstance
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.channels import MpReceiver, MpSender

# Find mpirun in PATH, fallback to common locations
MPIRUN_PATH = shutil.which("mpirun") or "/opt/homebrew/bin/mpirun"

# exo-rsh is installed as console script by exo package
_exo_rsh_path = shutil.which("exo-rsh")
if not _exo_rsh_path:
    raise RuntimeError("exo-rsh not found in PATH - this should be installed with exo")
EXO_RSH_PATH: str = _exo_rsh_path


def get_my_rank(instance: FLASHInstance, my_node_id: str) -> int:
    """Determine this node's rank based on position in hosts_by_node."""
    for i, node_id in enumerate(instance.hosts_by_node.keys()):
        if str(node_id) == str(my_node_id):
            return i
    return -1


def get_coordinator_host(instance: FLASHInstance) -> str:
    """Get the IP of the coordinator node."""
    return instance.coordinator_ip


def resolve_host(host: str) -> str:
    """Resolve host string to a usable hostname for MPI hostfile.

    Accepts either an IP address or hostname. For IPs, attempts to resolve
    to a hostname via DNS/mDNS. Hostnames are returned as-is after validation.
    """
    # Check if input is already a hostname (not an IP)
    try:
        socket.inet_aton(host)
        is_ip = True
    except socket.error:
        is_ip = False

    if not is_ip:
        # Already a hostname, verify it resolves and return as-is
        try:
            socket.gethostbyname(host)
            return host
        except socket.gaierror:
            logger.warning(f"Hostname {host} does not resolve, using anyway")
            return host

    # It's an IP address, try to resolve to hostname
    try:
        hostname, _, _ = socket.gethostbyaddr(host)
        hostname = hostname.split(".")[0]
        logger.info(f"Resolved {host} to {hostname}")
        return hostname
    except socket.herror:
        pass

    # Fall back to IP
    logger.warning(f"Could not resolve {host} to hostname, using IP directly")
    return host


def generate_hostfile(instance: FLASHInstance, working_dir: str) -> str:
    """Generate MPI hostfile from instance topology."""
    hostfile_path = os.path.join(working_dir, "flash_hosts.txt")
    with open(hostfile_path, "w") as f:
        for _node_id, hosts in instance.hosts_by_node.items():
            if hosts:
                host = resolve_host(hosts[0].ip)
                f.write(f"{host} slots={instance.ranks_per_node}\n")
    logger.info(f"Generated hostfile at {hostfile_path}")
    with open(hostfile_path, "r") as f:
        logger.info(f"Hostfile contents:\n{f.read()}")
    return hostfile_path


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
) -> None:
    """Main FLASH runner loop.

    Coordinator: generates hostfile and runs mpirun (uses exo-rsh instead of SSH)
    Workers: just report ready and wait for mpirun to spawn processes on them
    """
    assert isinstance(bound_instance.instance, FLASHInstance)
    instance = bound_instance.instance
    runner_id = bound_instance.bound_runner_id
    my_node_id = str(bound_instance.bound_node_id)

    logger.info(f"FLASH runner starting for simulation: {instance.simulation_name}")

    my_rank = get_my_rank(instance, my_node_id)
    world_size = len(instance.hosts_by_node)
    is_coordinator = my_rank == 0
    coordinator_ip = get_coordinator_host(instance)

    logger.info(
        f"FLASH node: rank={my_rank}, world_size={world_size}, coordinator={is_coordinator}"
    )
    logger.info(f"FLASH coordinator IP: {coordinator_ip}")

    process: subprocess.Popen[bytes] | None = None
    current_status: RunnerStatus = RunnerIdle()
    shutdown_requested = False

    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )

    def monitor_output(proc: subprocess.Popen[bytes]) -> None:
        """Monitor FLASH stdout for progress updates."""
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, b""):
            if shutdown_requested:
                break
            try:
                decoded: str = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    logger.info(f"[FLASH] {decoded}")
            except Exception as e:
                logger.warning(f"Error parsing FLASH output: {e}")

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))

            match task:
                case LoadModel() if isinstance(current_status, RunnerIdle):
                    current_status = RunnerLoading()
                    logger.info("Starting FLASH simulation")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    try:
                        if is_coordinator:
                            # Coordinator: generate hostfile and run mpirun
                            hostfile = generate_hostfile(
                                instance, instance.working_directory
                            )

                            iface = instance.network_interface
                            cmd = [
                                MPIRUN_PATH,
                                "-np",
                                str(instance.total_ranks),
                                "--hostfile",
                                hostfile,
                                "--wdir",
                                instance.working_directory,
                                "--oversubscribe",
                                "--mca",
                                "btl",
                                "tcp,self",
                                "--mca",
                                "btl_tcp_if_include",
                                iface,
                                "--mca",
                                "oob_tcp_if_include",
                                iface,
                                "--mca",
                                "plm_rsh_no_tree_spawn",
                                "1",
                            ]

                            # Use exo-rsh for remote execution (no SSH needed)
                            cmd.extend(["--mca", "plm_rsh_agent", EXO_RSH_PATH])

                            cmd.append(instance.flash_executable_path)

                            logger.info(f"FLASH distributed launch: {' '.join(cmd)}")

                            process = subprocess.Popen(
                                cmd,
                                cwd=instance.working_directory,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )

                            monitor_thread = threading.Thread(
                                target=monitor_output, args=(process,), daemon=True
                            )
                            monitor_thread.start()

                            current_status = RunnerRunning()
                            logger.info(
                                f"FLASH running on {world_size} nodes with {instance.total_ranks} ranks"
                            )

                        else:
                            # Worker: mpirun on coordinator will use exo-rsh to spawn processes here
                            logger.info(
                                f"Worker {my_rank}: Ready for mpirun to spawn processes via exo-rsh"
                            )
                            current_status = RunnerRunning()

                    except Exception as e:
                        logger.error(f"Failed to start FLASH: {e}")
                        import traceback

                        logger.error(traceback.format_exc())
                        current_status = RunnerFailed(error_message=str(e))

                case Shutdown():
                    shutdown_requested = True
                    current_status = RunnerShuttingDown()
                    logger.info("FLASH runner shutting down")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    if process and process.poll() is None:
                        logger.info("Terminating FLASH simulation")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            logger.warning("FLASH didn't terminate, killing")
                            process.kill()
                            process.wait()

                    current_status = RunnerShutdown()

                case _:
                    if process and process.poll() is not None:
                        exit_code = process.returncode
                        if exit_code == 0:
                            logger.info("FLASH simulation completed successfully")
                            current_status = RunnerReady()
                        else:
                            logger.error(
                                f"FLASH simulation failed with code {exit_code}"
                            )
                            current_status = RunnerFailed(
                                error_message=f"Exit code {exit_code}"
                            )

            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )

            if isinstance(current_status, RunnerShutdown):
                break

    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=5)

    logger.info("FLASH runner exiting")
