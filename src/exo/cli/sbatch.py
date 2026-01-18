"""sbatch - Submit a batch job to Exo.

Usage:
    exo sbatch [options] <script|executable>
    exo sbatch --job-name=NAME --nodes=N <executable>

Options:
    -J, --job-name NAME         Job name
    -N, --nodes N               Number of nodes (default: 1)
    --ntasks-per-node N         Tasks per node (default: 1)
    -D, --chdir DIR             Working directory
    --hosts HOSTS               Comma-separated list of hostnames

Job scripts can contain #SBATCH directives:
    #!/bin/bash
    #SBATCH --job-name=Sod2D
    #SBATCH --nodes=2
    #SBATCH --chdir=/path/to/workdir

    /path/to/flash4
"""

import argparse
import os
import re
import sys

from exo.cli.common import api_request, truncate_id


def parse_job_script(script_path: str) -> tuple[dict[str, str], str | None]:
    """Parse a job script for #SBATCH directives and executable.

    Args:
        script_path: Path to the job script

    Returns:
        Tuple of (directives dict, executable path or None)
    """
    directives: dict[str, str] = {}
    executable: str | None = None

    with open(script_path, "r") as f:
        for line in f:
            line = line.strip()

            # Parse #SBATCH directives
            if line.startswith("#SBATCH"):
                # Handle both --option=value and --option value formats
                match = re.match(r"#SBATCH\s+(-\w|--[\w-]+)(?:=|\s+)(.+)", line)
                if match:
                    opt, val = match.groups()
                    directives[opt.lstrip("-")] = val.strip()
                continue

            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue

            # First non-comment, non-directive line is the executable
            if executable is None:
                # Handle lines like "/path/to/flash4" or "srun /path/to/flash4"
                parts = line.split()
                if parts:
                    # Skip srun/mpirun prefixes if present
                    for part in parts:
                        if not part.startswith("-") and "/" in part:
                            executable = part
                            break
                    if executable is None and parts:
                        executable = parts[-1]  # Last token

    return directives, executable


def main(args: list[str]) -> int:
    """Main entry point for sbatch command."""
    parser = argparse.ArgumentParser(
        prog="exo sbatch",
        description="Submit a batch job to Exo",
    )
    parser.add_argument(
        "script",
        help="Job script or executable path",
    )
    parser.add_argument(
        "-J",
        "--job-name",
        dest="job_name",
        help="Job name",
    )
    parser.add_argument(
        "-N",
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--ntasks-per-node",
        type=int,
        default=1,
        help="Tasks per node (default: 1)",
    )
    parser.add_argument(
        "-D",
        "--chdir",
        help="Working directory",
    )
    parser.add_argument(
        "--hosts",
        help="Comma-separated list of hostnames",
    )

    parsed = parser.parse_args(args)

    # Extract typed values from namespace
    script_path: str = parsed.script  # pyright: ignore[reportAny]
    arg_job_name: str | None = parsed.job_name  # pyright: ignore[reportAny]
    arg_nodes: int = parsed.nodes  # pyright: ignore[reportAny]
    arg_ntasks: int = parsed.ntasks_per_node  # pyright: ignore[reportAny]
    arg_chdir: str | None = parsed.chdir  # pyright: ignore[reportAny]
    arg_hosts: str | None = parsed.hosts  # pyright: ignore[reportAny]

    # Determine if input is a script or direct executable
    executable: str | None = None
    directives: dict[str, str] = {}

    if os.path.isfile(script_path):
        # Check if it's a binary file (executable) or text script
        is_binary = False
        try:
            with open(script_path, "rb") as f:
                chunk = f.read(512)
                # Binary files typically contain null bytes
                is_binary = b"\x00" in chunk
        except OSError:
            pass

        if is_binary:
            # It's a binary executable
            executable = script_path
        else:
            # Try to read as text
            try:
                with open(script_path, "r") as f:
                    first_line = f.readline()
                    f.seek(0)
                    content = f.read(1024)

                if first_line.startswith("#!") or "#SBATCH" in content:
                    # It's a job script - parse it
                    directives, executable = parse_job_script(script_path)
                else:
                    # It's an executable (text but no shebang/directives)
                    executable = script_path
            except UnicodeDecodeError:
                # Can't read as text - treat as binary executable
                executable = script_path
    else:
        # Not a file - treat as executable path
        executable = script_path

    if executable is None:
        print("Error: No executable found in job script", file=sys.stderr)
        return 1

    # Build job parameters - CLI args override script directives
    job_name = arg_job_name or directives.get("job-name") or directives.get("J")
    if not job_name:
        # Generate name from executable
        job_name = os.path.basename(executable).replace(".", "_")

    nodes = arg_nodes
    if "nodes" in directives:
        nodes = int(directives["nodes"])
    if "N" in directives:
        nodes = int(directives["N"])
    if arg_nodes != 1:  # CLI override
        nodes = arg_nodes

    ntasks = arg_ntasks
    if "ntasks-per-node" in directives:
        ntasks = int(directives["ntasks-per-node"])
    if arg_ntasks != 1:  # CLI override
        ntasks = arg_ntasks

    workdir = arg_chdir or directives.get("chdir") or directives.get("D")
    if not workdir:
        workdir = os.getcwd()

    hosts = arg_hosts or directives.get("hosts") or ""

    # Resolve executable to absolute path
    if not os.path.isabs(executable):
        executable = os.path.abspath(os.path.join(workdir, executable))

    # Submit job via API using query parameters
    from urllib.parse import urlencode

    params = {
        "simulation_name": job_name,
        "flash_executable_path": executable,
        "parameter_file_path": "",  # FLASH par file - use default
        "working_directory": workdir,
        "ranks_per_node": str(ntasks),
        "min_nodes": str(nodes),
        "hosts": hosts,
    }

    query_string = urlencode(params)
    result = api_request("POST", f"/flash/launch?{query_string}")

    # Print job submission confirmation
    if isinstance(result, dict):
        instance_id_val = result.get("instance_id")

        if instance_id_val is not None:
            job_id = truncate_id(str(instance_id_val))  # pyright: ignore[reportAny]
            print(f"Submitted batch job {job_id}")
        else:
            # Instance created asynchronously - user should check squeue
            print("Job submitted successfully")
            print("Use 'exo squeue' to view job ID")
    else:
        print("Job submitted successfully")
        print("Use 'exo squeue' to view job ID")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
