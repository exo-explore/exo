"""salloc - Allocate nodes for interactive use.

Usage:
    exo salloc [options] [-- command [args...]]

Options:
    -N, --nodes N       Number of nodes to allocate (default: 1)
    --hosts HOSTS       Comma-separated list of hostnames

If a command is provided after --, it will be executed with
SLURM-like environment variables set:
    SLURM_JOB_NODELIST  - Comma-separated list of allocated nodes
    SLURM_NNODES        - Number of allocated nodes

Examples:
    exo salloc --nodes=2 --hosts=node1,node2 -- mpirun ./my_program
    exo salloc --hosts=localhost -- bash
"""

import argparse
import os
import subprocess
import sys


def main(args: list[str]) -> int:
    """Main entry point for salloc command."""
    # Split args at -- if present
    cmd_args: list[str] = []
    salloc_args = args

    if "--" in args:
        idx = args.index("--")
        salloc_args = args[:idx]
        cmd_args = args[idx + 1 :]

    parser = argparse.ArgumentParser(
        prog="exo salloc",
        description="Allocate nodes for interactive use",
    )
    parser.add_argument(
        "-N",
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to allocate (default: 1)",
    )
    parser.add_argument(
        "--hosts",
        help="Comma-separated list of hostnames (required)",
    )

    parsed = parser.parse_args(salloc_args)

    nodes: int = parsed.nodes  # pyright: ignore[reportAny]
    hosts: str | None = parsed.hosts  # pyright: ignore[reportAny]

    # Require explicit hosts since we can't discover them from topology
    if not hosts:
        print("Error: --hosts is required (e.g., --hosts=node1,node2)", file=sys.stderr)
        print("       The Exo topology doesn't expose hostnames.", file=sys.stderr)
        return 1

    host_list = [h.strip() for h in hosts.split(",") if h.strip()]

    if len(host_list) < nodes:
        print(
            f"Error: Requested {nodes} nodes but only {len(host_list)} hosts provided",
            file=sys.stderr,
        )
        return 1

    # Use first N hosts
    allocated_hosts = host_list[:nodes]
    nodelist = ",".join(allocated_hosts)

    # Set environment variables
    env = os.environ.copy()
    env["SLURM_JOB_NODELIST"] = nodelist
    env["SLURM_NNODES"] = str(nodes)

    print(f"salloc: Granted job allocation on {nodes} node(s)")
    print(f"salloc: Nodes: {nodelist}")

    if cmd_args:
        # Run the command
        print(f"salloc: Running: {' '.join(cmd_args)}")
        result = subprocess.run(cmd_args, env=env)
        return result.returncode
    else:
        # Start interactive shell
        shell = os.environ.get("SHELL", "/bin/bash")
        print(f"salloc: Starting shell {shell}")
        print("salloc: Use 'exit' to release allocation")
        result = subprocess.run([shell], env=env)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
