"""scancel - Cancel jobs in the Exo queue.

Usage:
    exo scancel <jobid> [<jobid>...]

Arguments:
    jobid   Job ID (or prefix) to cancel. Can specify multiple.

Examples:
    exo scancel abc123          # Cancel job starting with abc123
    exo scancel abc123 def456   # Cancel multiple jobs
"""

import argparse
import sys
from typing import Any, cast

from exo.cli.common import api_request, truncate_id


def main(args: list[str]) -> int:
    """Main entry point for scancel command."""
    parser = argparse.ArgumentParser(
        prog="exo scancel",
        description="Cancel jobs in the Exo queue",
    )
    parser.add_argument(
        "jobids",
        nargs="+",
        help="Job ID(s) to cancel",
    )

    parsed = parser.parse_args(args)
    jobids: list[str] = parsed.jobids  # pyright: ignore[reportAny]

    # Fetch current jobs to resolve partial IDs
    result = api_request("GET", "/flash/instances")
    if isinstance(result, list):
        instances = cast(list[dict[str, Any]], result)
    else:
        instances = cast(list[dict[str, Any]], result.get("instances", []))

    # Build lookup of full IDs
    id_map: dict[str, str] = {}
    for inst in instances:
        iid = inst.get("instance_id", "")  # pyright: ignore[reportAny]
        full_id = str(iid) if iid else ""  # pyright: ignore[reportAny]
        if full_id:
            # Map both full ID and truncated versions
            normalized = full_id.replace("-", "").lower()
            id_map[normalized] = full_id
            # Also map prefixes
            for length in range(4, len(normalized) + 1):
                prefix = normalized[:length]
                if prefix not in id_map:
                    id_map[prefix] = full_id

    cancelled = 0
    errors = 0

    for jobid in jobids:
        search = jobid.lower().replace("-", "")

        # Find matching full ID
        full_id = id_map.get(search)
        if not full_id:
            # Try prefix match
            matches = [fid for key, fid in id_map.items() if key.startswith(search)]
            if len(matches) == 1:
                full_id = matches[0]
            elif len(matches) > 1:
                print(f"Ambiguous job ID: {jobid} matches multiple jobs")
                errors += 1
                continue
            else:
                print(f"Job not found: {jobid}")
                errors += 1
                continue

        # Cancel the job
        try:
            api_request("DELETE", f"/flash/{full_id}")
            print(f"Job {truncate_id(full_id)} cancelled")
            cancelled += 1
        except SystemExit:
            print(f"Failed to cancel job {truncate_id(full_id)}")
            errors += 1

    if errors > 0 and cancelled == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
