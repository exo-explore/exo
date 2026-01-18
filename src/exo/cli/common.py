"""Common utilities for Exo CLI commands."""

import json
import os
import urllib.request
from typing import Any
from urllib.error import HTTPError, URLError

# Default API endpoint
DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = 52415


def get_api_base() -> str:
    """Get the API base URL from environment or defaults."""
    host = os.environ.get("EXO_API_HOST", DEFAULT_API_HOST)
    port = os.environ.get("EXO_API_PORT", str(DEFAULT_API_PORT))
    return f"http://{host}:{port}"


def api_request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    """Make an API request to the Exo server.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        path: API path (e.g., "/flash/instances")
        data: Optional JSON data for POST/PUT requests

    Returns:
        Parsed JSON response

    Raises:
        SystemExit: On connection or HTTP errors
    """
    url = f"{get_api_base()}{path}"

    request_data = None
    if data is not None:
        request_data = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=request_data,
        method=method,
    )
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # pyright: ignore[reportAny]
            body: str = response.read().decode("utf-8")  # pyright: ignore[reportAny]
            if body:
                return json.loads(body)  # pyright: ignore[reportAny]
            return {}
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"API error: {e.code} {e.reason}")
        if error_body:
            try:
                error_json: dict[str, str] = json.loads(error_body)  # pyright: ignore[reportAny]
                if "detail" in error_json:
                    print(f"  {error_json['detail']}")
            except json.JSONDecodeError:
                print(f"  {error_body}")
        raise SystemExit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}")
        print(f"Is Exo running at {get_api_base()}?")
        raise SystemExit(1)


def truncate_id(instance_id: str, length: int = 8) -> str:
    """Truncate a UUID for display.

    Args:
        instance_id: Full UUID string
        length: Number of characters to keep

    Returns:
        Truncated ID without hyphens
    """
    return instance_id.replace("-", "")[:length]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as a simple text table.

    Args:
        headers: Column headers
        rows: List of rows, each row is a list of column values

    Returns:
        Formatted table string
    """
    if not rows:
        return "  ".join(f"{h:<10}" for h in headers)

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    # Format output
    lines = [fmt.format(*headers)]
    for row in rows:
        # Pad row if needed
        padded = row + [""] * (len(headers) - len(row))
        lines.append(fmt.format(*padded[: len(headers)]))

    return "\n".join(lines)
