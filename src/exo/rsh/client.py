#!/usr/bin/env python3
"""exo-rsh - Remote shell client for MPI.

This script is called by mpirun as a replacement for ssh.
Usage: exo-rsh [ssh-options...] hostname command [args...]

It connects to the target node's RSH server (port 52416) and executes the command.
"""

import json
import socket
import sys
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

RSH_PORT = 52416


def resolve_hostname(hostname: str) -> str:
    """Resolve hostname to IP address."""
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        # If resolution fails, try using the hostname directly
        return hostname


def main():
    # Parse arguments - mpirun calls us like: exo-rsh [options] hostname command [args...]
    # SSH options we might see: -x (disable X11), -o options, etc.
    args = sys.argv[1:]

    # Skip SSH-style options
    hostname = None
    command_start = 0

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            # Skip option and its value if needed
            if arg in ("-o", "-i", "-l", "-p", "-F"):
                i += 2  # Skip option and its argument
                continue
            i += 1
            continue
        else:
            # First non-option is the hostname
            hostname = arg
            command_start = i + 1
            break
        i += 1

    if hostname is None or command_start >= len(args):
        print("Usage: exo-rsh [options] hostname command [args...]", file=sys.stderr)
        sys.exit(1)

    command = args[command_start:]

    # Resolve hostname to IP
    ip = resolve_hostname(hostname)

    # Make request to RSH server
    url = f"http://{ip}:{RSH_PORT}/execute"
    data = json.dumps({"command": command}).encode("utf-8")

    try:
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as response:  # pyright: ignore[reportAny]
            response_body: bytes = cast(bytes, response.read())  # pyright: ignore[reportAny]
            result: dict[str, Any] = json.loads(response_body.decode("utf-8"))  # pyright: ignore[reportAny]

        # Output stdout/stderr
        stdout: str = cast(str, result.get("stdout", ""))
        stderr: str = cast(str, result.get("stderr", ""))
        exit_code: int = cast(int, result.get("exit_code", 0))

        if stdout:
            sys.stdout.write(stdout)
            sys.stdout.flush()
        if stderr:
            sys.stderr.write(stderr)
            sys.stderr.flush()

        sys.exit(exit_code)

    except URLError as e:
        print(
            f"exo-rsh: Failed to connect to {hostname}:{RSH_PORT}: {e}", file=sys.stderr
        )
        sys.exit(255)
    except Exception as e:
        print(f"exo-rsh: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
