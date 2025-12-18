#!/usr/bin/env python3
import json
import os
from typing import NotRequired, TypedDict, cast

import yaml


class MatrixEntry(TypedDict):
    label: str
    index: int


class MatrixInclude(TypedDict):
    label: str
    index: int
    is_primary: bool
    expected_nodes: int


class Config(TypedDict):
    hardware_plan: dict[str, int]
    timeout_seconds: NotRequired[int]
    environment: NotRequired[dict[str, str]]


# Read the config file
config_file: str = os.environ["CONFIG_FILE"]
with open(config_file, "r") as f:
    config: Config = cast(Config, yaml.safe_load(f))

# Extract hardware plan from config
plan: dict[str, int] = config["hardware_plan"]
if not plan:
    raise ValueError(f"No hardware_plan found in {config_file}")

# Build matrix entries
entries: list[MatrixEntry] = []
for label, count in plan.items():
    for idx in range(count):
        entries.append({"label": label, "index": idx})

total_nodes: int = len(entries)
matrix: dict[str, list[MatrixInclude]] = {
    "include": [
        {
            "label": e["label"],
            "index": e["index"],
            "is_primary": (i == 0),
            "expected_nodes": total_nodes,
        }
        for i, e in enumerate(entries)
    ]
}

# Extract other config values
timeout_seconds: int = config.get("timeout_seconds", 600)
environment: dict[str, str] = config.get("environment", {})

# Output to GitHub Actions
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"matrix={json.dumps(matrix)}\n")
    f.write(f"config_file={config_file}\n")
    f.write(f"timeout_seconds={timeout_seconds}\n")
    f.write(f"environment={json.dumps(environment)}\n")

print(f"Matrix: {json.dumps(matrix)}")
print(f"Config file: {config_file}")
print(f"Timeout: {timeout_seconds}")
print(f"Environment: {json.dumps(environment)}")
