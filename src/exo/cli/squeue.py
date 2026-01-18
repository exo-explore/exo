"""squeue - View the Exo job queue.

Usage:
    exo squeue [options]

Options:
    -l, --long      Show detailed output
    -j, --job ID    Show only this job

Output columns:
    JOBID   - Job identifier (truncated UUID)
    NAME    - Job name
    NODES   - Number of nodes
    STATE   - Job state (PENDING, RUNNING, FAILED, etc.)
"""

import argparse
import sys
from typing import Any, cast

from exo.cli.common import api_request, format_table, truncate_id

# Map Exo runner statuses to SLURM-like states
STATUS_MAP: dict[str, str] = {
    "RunnerIdle": "PENDING",
    "RunnerConnecting": "CONFIGURING",
    "RunnerConnected": "CONFIGURING",
    "RunnerLoading": "CONFIGURING",
    "RunnerLoaded": "CONFIGURING",
    "RunnerWarmingUp": "CONFIGURING",
    "RunnerReady": "COMPLETING",
    "RunnerRunning": "RUNNING",
    "RunnerShuttingDown": "COMPLETING",
    "RunnerShutdown": "COMPLETED",
    "RunnerFailed": "FAILED",
}


def get_job_state(runner_statuses: dict[str, Any]) -> str:
    """Determine overall job state from runner statuses."""
    if not runner_statuses:
        return "PENDING"

    states: set[str] = set()
    for status_val in runner_statuses.values():  # pyright: ignore[reportAny]
        if isinstance(status_val, dict):
            # Extract status type from discriminated union
            type_val = status_val.get("type", "RunnerIdle")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            status_type = str(type_val) if type_val else "RunnerIdle"  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(status_val, str):
            status_type = status_val
        else:
            status_type = "RunnerIdle"
        # Strip parentheses from status strings like "RunnerRunning()"
        if status_type.endswith("()"):
            status_type = status_type[:-2]
        states.add(STATUS_MAP.get(status_type, "UNKNOWN"))

    # Priority order for overall state
    if "FAILED" in states:
        return "FAILED"
    if "RUNNING" in states:
        return "RUNNING"
    if "CONFIGURING" in states:
        return "CONFIGURING"
    if "COMPLETING" in states:
        return "COMPLETING"
    if "COMPLETED" in states:
        return "COMPLETED"
    if "PENDING" in states:
        return "PENDING"
    return "UNKNOWN"


def main(args: list[str]) -> int:
    """Main entry point for squeue command."""
    parser = argparse.ArgumentParser(
        prog="exo squeue",
        description="View the Exo job queue",
    )
    parser.add_argument(
        "-l",
        "--long",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "-j",
        "--job",
        help="Show only this job ID",
    )

    parsed = parser.parse_args(args)

    # Extract typed values
    long_format: bool = parsed.long  # pyright: ignore[reportAny]
    job_filter: str | None = parsed.job  # pyright: ignore[reportAny]

    # Fetch jobs from API - returns list directly
    result = api_request("GET", "/flash/instances")
    # API returns list directly, not {"instances": [...]}
    if isinstance(result, list):
        instances = cast(list[dict[str, Any]], result)
    else:
        instances = cast(list[dict[str, Any]], result.get("instances", []))

    if not instances:
        # No jobs - just print header
        if long_format:
            print("JOBID           NAME            NODES  RANKS  STATE        WORKDIR")
        else:
            print("JOBID       NAME            NODES  STATE")
        return 0

    # Filter by job ID if specified
    if job_filter:
        search = job_filter.lower()
        filtered: list[dict[str, Any]] = []
        for i in instances:
            iid = i.get("instance_id", "")  # pyright: ignore[reportAny]
            if search in str(iid).lower().replace("-", ""):  # pyright: ignore[reportAny]
                filtered.append(i)
        instances = filtered

    # Build table
    rows: list[list[str]] = []

    if long_format:
        headers = ["JOBID", "NAME", "NODES", "RANKS", "STATE", "WORKDIR"]
        for inst in instances:
            iid_val = inst.get("instance_id", "")  # pyright: ignore[reportAny]
            instance_id = str(iid_val) if iid_val else ""  # pyright: ignore[reportAny]
            job_id = truncate_id(instance_id, 12)
            name_val = inst.get("simulation_name", "")  # pyright: ignore[reportAny]
            name = (str(name_val) if name_val else "")[:15]  # pyright: ignore[reportAny]
            runner_statuses = cast(dict[str, Any], inst.get("runner_statuses", {}))
            nodes = str(len(runner_statuses))
            ranks_val = inst.get("total_ranks", 0)  # pyright: ignore[reportAny]
            ranks = str(ranks_val) if ranks_val else "0"  # pyright: ignore[reportAny]
            state = get_job_state(runner_statuses)
            workdir_val = inst.get("working_directory", "")  # pyright: ignore[reportAny]
            workdir = str(workdir_val) if workdir_val else ""  # pyright: ignore[reportAny]
            # Truncate workdir for display
            if len(workdir) > 30:
                workdir = "..." + workdir[-27:]
            rows.append([job_id, name, nodes, ranks, state, workdir])
    else:
        headers = ["JOBID", "NAME", "NODES", "STATE"]
        for inst in instances:
            iid_val = inst.get("instance_id", "")  # pyright: ignore[reportAny]
            instance_id = str(iid_val) if iid_val else ""  # pyright: ignore[reportAny]
            job_id = truncate_id(instance_id, 8)
            name_val = inst.get("simulation_name", "")  # pyright: ignore[reportAny]
            name = (str(name_val) if name_val else "")[:15]  # pyright: ignore[reportAny]
            runner_statuses = cast(dict[str, Any], inst.get("runner_statuses", {}))
            nodes = str(len(runner_statuses))
            state = get_job_state(runner_statuses)
            rows.append([job_id, name, nodes, state])

    print(format_table(headers, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
