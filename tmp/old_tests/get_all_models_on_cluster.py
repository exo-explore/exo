#!/usr/bin/env python3
# pyright: reportAny=false
import json
import subprocess
import sys
from typing import Any, cast
from urllib.request import urlopen

h = sys.argv[1] if len(sys.argv) > 1 else sys.exit(f"USAGE: {sys.argv[0]} host")
ts = subprocess.run(
    ["tailscale", "status"], check=True, text=True, capture_output=True
).stdout.splitlines()
ip = next(
    (sl[0] for line in ts if len(sl := line.split()) >= 2 if sl[1] == h), None
) or sys.exit(f"{h} not found in tailscale")
with urlopen(f"http://{ip}:52415/state", timeout=5) as r:
    data = json.loads(r.read()).get("downloads", {})


def mid(x: dict[str, Any]) -> str | None:
    for k in (
        "DownloadCompleted",
        "shardMetadata",
        "PipelineShardMetadata",
        "modelCard",
        "modelId",
    ):
        x = x.get(k, {})
    return cast(str | None, x if x != {} else None)


common = set[str].intersection(
    *[{m for d in nid if (m := mid(d))} for nid in data.values()]
)
for c in common:
    print(c)
