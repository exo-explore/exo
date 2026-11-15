#!/usr/bin/env python3
"""Validate an already running EXO TensorFold deployment using the standard library."""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

from tensorfold_validation.client import Endpoint
from tensorfold_validation.report import qualify


class Arguments(argparse.Namespace):
    exo_url: str = ""
    model: str = ""
    model_revision: str = ""
    hardware: str = ""
    binary: Path | None = None
    api_key_env: str | None = None
    native_url: str | None = None
    native_model: str | None = None
    native_api_key_env: str | None = None
    repeats: int = 3
    concurrency: int = 2
    max_tokens: int = 128
    timeout: int = 300
    output: Path = Path("qualification.json")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def endpoint(url: str, key_name: str | None, timeout: int) -> Endpoint:
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "Use an HTTP(S) base URL without credentials, query or fragment"
        )
    key = os.environ.get(key_name) if key_name is not None else None
    if key_name is not None and not key:
        raise ValueError("The named API key environment variable is empty")
    return Endpoint(url.rstrip("/"), key, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo-url", default="http://127.0.0.1:52415")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact pinned weights revision; recorded as operator supplied",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        help="Machine/GPU and memory, recorded as operator supplied",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Optional matching native binary for its capabilities/version report",
    )
    parser.add_argument(
        "--api-key-env",
        help="Name of an environment variable containing the EXO API token",
    )
    parser.add_argument(
        "--native-url",
        help="Optional separately running direct TensorFold endpoint for comparison",
    )
    parser.add_argument("--native-model")
    parser.add_argument("--native-api-key-env")
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--concurrency", type=positive, default=2)
    parser.add_argument("--max-tokens", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    try:
        report = qualify(
            endpoint(arguments.exo_url, arguments.api_key_env, arguments.timeout),
            arguments.model,
            arguments.model_revision,
            arguments.hardware,
            arguments.repeats,
            arguments.concurrency,
            arguments.max_tokens,
            arguments.binary,
            endpoint(
                arguments.native_url, arguments.native_api_key_env, arguments.timeout
            )
            if arguments.native_url
            else None,
            arguments.native_model,
        )
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"Qualification failed: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Passed: native placement, complete streams, usage, concurrency and post-disconnect request."
    )
    print("Timing receipt saved. Numerical and GPU qualification remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
