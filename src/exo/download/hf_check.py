"""`exo-hf-check`: diagnose reachability of HF endpoints and the xet CAS CDN.

Usage:
    uv run exo-hf-check                          # check all configured endpoints
    uv run exo-hf-check --endpoint URL           # check one specific endpoint
    uv run exo-hf-check --repo user/model-id     # use a custom probe repo

The probe transfers <5 KB total: it exercises /api/models/{}/tree/main, HEADs a small
file, HEADs an LFS file (no body), and HEADs cas-bridge.xethub.hf.co to surface the
China-specific xet block. Runs in ~3s.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Literal

from exo.download.download_utils import (
    _fetch_file_list,  # pyright: ignore[reportPrivateUsage]
    create_http_session,
    file_meta,
)
from exo.download.hf_endpoints import get_hf_endpoints
from exo.download.huggingface_utils import get_auth_headers
from exo.shared.types.common import ModelId

# A stable tiny repo we know is served by hf-mirror and huggingface.co.
DEFAULT_PROBE_REPO = ModelId("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
_XET_PROBE_URL = "https://cas-bridge.xethub.hf.co/"

Status = Literal["ok", "fail", "-"]


def _fmt(s: Status) -> str:
    return {"ok": "ok", "fail": "FAIL", "-": "-"}[s]


async def _probe_endpoint(
    endpoint: str, repo: ModelId, timeout_s: float
) -> dict[str, str]:
    result: dict[str, str] = {
        "endpoint": endpoint,
        "server": "-",
        "tree-api": "-",
        "small-file": "-",
        "lfs-meta": "-",
    }

    # 1) Raw HEAD on the root to read `server:` header (distinguishes hf-mirror).
    try:
        async with create_http_session(timeout_profile="short") as session:
            async with asyncio.timeout(timeout_s):
                async with session.head(endpoint) as r:
                    result["server"] = r.headers.get("server", "-")
    except Exception:
        pass

    # 2) Tree API
    try:
        async with asyncio.timeout(timeout_s):
            files = await _fetch_file_list(repo, "main", "", False, endpoint=endpoint)
            if files:
                result["tree-api"] = _fmt("ok")
            else:
                result["tree-api"] = _fmt("fail")
    except Exception as e:
        result["tree-api"] = f"FAIL ({type(e).__name__})"

    # 3) Small-file HEAD (config.json, 307 → relative follow → 200)
    try:
        async with asyncio.timeout(timeout_s):
            length, _ = await file_meta(repo, "main", "config.json", endpoint=endpoint)
            result["small-file"] = _fmt("ok") if length > 0 else _fmt("fail")
    except Exception as e:
        result["small-file"] = f"FAIL ({type(e).__name__})"

    # 4) LFS-meta HEAD (model.safetensors, 302 → header trust). No body transfer.
    try:
        async with asyncio.timeout(timeout_s):
            length, _ = await file_meta(
                repo, "main", "model.safetensors", endpoint=endpoint
            )
            result["lfs-meta"] = _fmt("ok") if length > 0 else _fmt("fail")
    except Exception as e:
        result["lfs-meta"] = f"FAIL ({type(e).__name__})"

    return result


async def _probe_xet_cdn(timeout_s: float) -> str:
    """One-byte Range probe on cas-bridge to detect China-block."""
    try:
        headers = {**(await get_auth_headers()), "Range": "bytes=0-0"}
        async with create_http_session(timeout_profile="short") as session:
            async with asyncio.timeout(timeout_s):
                async with session.head(_XET_PROBE_URL, headers=headers) as r:
                    # Any response (incl. 403 from an unauthenticated root HEAD) means
                    # the host is reachable. Block/filter manifests as timeout/connect error.
                    _ = r.status
                    return "ok"
    except asyncio.TimeoutError:
        return "blocked (timeout)"
    except Exception as e:
        return f"blocked ({type(e).__name__})"


def _print_table(rows: list[dict[str, str]], xet_status: str) -> None:
    headers = ["endpoint", "server", "tree-api", "small-file", "lfs-meta"]
    widths = {
        h: max(len(h), max(len(r[h]) for r in rows) if rows else 0) for h in headers
    }
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))
    print()
    print(f"xet-cdn (cas-bridge.xethub.hf.co): {xet_status}")
    if "blocked" in xet_status:
        print(
            "\nNote: xet-backed models cannot be downloaded through hf-mirror; "
            "the redirect always points at cas-bridge.xethub.hf.co. "
            "Workarounds: use a non-xet model, set HTTPS_PROXY, or pre-stage the "
            "model in $HF_HOME on a machine with HF access."
        )


async def _amain(endpoints: list[str], repo: ModelId, timeout_s: float) -> int:
    rows: list[dict[str, str]] = []
    for ep in endpoints:
        rows.append(await _probe_endpoint(ep, repo, timeout_s))
    xet_status = await _probe_xet_cdn(timeout_s)
    _print_table(rows, xet_status)

    any_ok = any(
        r["tree-api"].startswith("ok") and r["small-file"].startswith("ok")
        for r in rows
    )
    return 0 if any_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="exo-hf-check")
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="Endpoint URL to probe. May be passed multiple times. "
        "If omitted, probes HF_ENDPOINT and HF_MIRROR_ENDPOINT.",
    )
    parser.add_argument(
        "--repo",
        default=str(DEFAULT_PROBE_REPO),
        help=f"Probe repo (default: {DEFAULT_PROBE_REPO}).",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    parsed_endpoint: list[str] | None = args.endpoint  # pyright: ignore[reportAny]
    parsed_repo: str = args.repo  # pyright: ignore[reportAny]
    parsed_timeout: float = args.timeout  # pyright: ignore[reportAny]

    endpoints: list[str] = parsed_endpoint if parsed_endpoint else get_hf_endpoints()
    repo = ModelId(parsed_repo)
    code = asyncio.run(_amain(endpoints, repo, parsed_timeout))
    sys.exit(code)


if __name__ == "__main__":
    main()
