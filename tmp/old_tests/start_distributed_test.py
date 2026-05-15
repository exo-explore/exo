#!/usr/bin/env python3
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from urllib.request import Request, urlopen

if not (args := sys.argv[1:]):
    sys.exit(
        f"USAGE: {sys.argv[0]} <kind> [host1] [host2] ...\nkind is optional, and should be jaccl or ring"
    )

kind = args[0] if args[0] in ("jaccl", "ring") else "both"
hosts = args[1:] if kind != "both" else args
ts = subprocess.run(
    ["tailscale", "status"], check=True, text=True, capture_output=True
).stdout.splitlines()
ip = {sl[1]: sl[0] for line in ts if len(sl := line.split()) >= 2}
ips = [ip[h] for h in hosts]
devs = [[h, ip[h]] for h in hosts]
n = len(hosts)


def get_tb(a: str) -> list[dict[str, Any]]:
    with urlopen(f"http://{a}:52414/tb_detection", timeout=5) as r:  # pyright: ignore[reportAny]
        return json.loads(r.read())  # pyright: ignore[reportAny]


def get_models(a: str) -> set[str]:
    with urlopen(f"http://{a}:52414/models", timeout=5) as r:  # pyright: ignore[reportAny]
        return set(json.loads(r.read()))  # pyright: ignore[reportAny]


def run(h: str, a: str, body: bytes) -> None:
    with urlopen(
        Request(
            f"http://{a}:52414/run_test",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        ),
        timeout=300,
    ) as r:  # pyright: ignore[reportAny]
        for line in r.read().decode(errors="replace").splitlines():  # pyright: ignore[reportAny]
            print(f"\n{h}@{a}: {line}", flush=True)


with ThreadPoolExecutor(n) as exctr:
    if kind in ("jaccl", "both"):
        payloads = list(exctr.map(get_tb, ips))

        u2e = {
            ident["domainUuid"]: (i, ident["rdmaInterface"])
            for i, p in enumerate(payloads)
            for d in p
            for ident in cast(
                list[dict[str, str]],
                d.get("MacThunderboltIdentifiers", {}).get("idents", []),  # pyright: ignore[reportAny]
            )
        }
        edges = {
            (u2e[s][0], u2e[t][0]): u2e[t][1]
            for p in payloads
            for d in p
            for c in d.get("MacThunderboltConnections", {}).get("conns", [])  # pyright: ignore[reportAny]
            if (s := c["sourceUuid"]) in u2e and (t := c["sinkUuid"]) in u2e  # pyright: ignore[reportAny]
        }
        ibv_devs = [[edges.get((i, j)) for j in range(n)] for i in range(n)]
    else:
        ibv_devs = None

    models = set[str].intersection(*exctr.map(get_models, ips))

    print("\n")
    print("=" * 70)
    print(f"Starting test with {models}")
    print("=" * 70)
    print("\n")
    for model in models:
        body = json.dumps(
            {"devs": devs, "model_id": model, "ibv_devs": ibv_devs, "kind": kind}
        ).encode()
        list(exctr.map(run, hosts, ips, itertools.repeat(body)))
