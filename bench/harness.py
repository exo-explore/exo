# type: ignore
from __future__ import annotations

import argparse
import http.client
import json
import os
import time
from typing import Any
from urllib.parse import urlencode

from loguru import logger

_SETTLE_INITIAL_BACKOFF_S = 1.0
_SETTLE_MAX_BACKOFF_S = 60.0
_SETTLE_BACKOFF_MULTIPLIER = 2.0


class ExoHttpError(RuntimeError):
    def __init__(self, status: int, reason: str, body_preview: str):
        super().__init__(f"HTTP {status} {reason}: {body_preview}")
        self.status = status


class ExoClient:
    def __init__(self, host: str, port: int, timeout_s: float = 7200.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        if params:
            path = path + "?" + urlencode(params)

        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            payload: bytes | None = None
            hdrs: dict[str, str] = {"Accept": "application/json"}

            if body is not None:
                payload = json.dumps(body).encode("utf-8")
                hdrs["Content-Type"] = "application/json"
            if headers:
                hdrs.update(headers)

            conn.request(method.upper(), path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""

            if resp.status >= 400:
                raise ExoHttpError(resp.status, resp.reason, text[:300])

            if not text:
                return None
            return json.loads(text)
        finally:
            conn.close()

    def post_bench_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", "/bench/chat/completions", body=payload)


def unwrap_instance(instance: dict[str, Any]) -> dict[str, Any]:
    if len(instance) != 1:
        raise KeyError(f"Expected 1 key, got keys={list(instance.keys())}")

    tag = next(iter(instance))
    inner = instance[tag]
    if not isinstance(inner, dict):
        raise TypeError(f"payload for {tag} must be dict, got {type(inner)}")
    return inner


def instance_id_from_instance(instance: dict[str, Any]) -> str:
    inner = unwrap_instance(instance)
    return str(inner["instanceId"])


def nodes_used_in_instance(instance: dict[str, Any]) -> int:
    inner = unwrap_instance(instance)
    return len(inner["shardAssignments"]["nodeToRunner"])


def runner_ids_from_instance(instance: dict[str, Any]) -> list[str]:
    inner = unwrap_instance(instance)
    runner_to_shard = inner["shardAssignments"]["runnerToShard"]
    return list(runner_to_shard.keys())


def runner_ready(runner: dict[str, Any]) -> bool:
    return "RunnerReady" in runner


def runner_failed(runner: dict[str, Any]) -> bool:
    return "RunnerFailed" in runner


def get_runner_failed_message(runner: dict[str, Any]) -> str | None:
    if "RunnerFailed" in runner:
        return runner["RunnerFailed"].get("errorMessage")
    return None


def wait_for_instance_ready(
    client: ExoClient, instance_id: str, timeout: float = 24000.0
) -> None:
    start_time = time.time()
    instance_existed = False
    while time.time() - start_time < timeout:
        state = client.request_json("GET", "/state")
        instances = state.get("instances", {})

        if instance_id not in instances:
            if instance_existed:
                # Instance was deleted after being created - likely due to runner failure
                raise RuntimeError(
                    f"Instance {instance_id} was deleted (runner may have failed)"
                )
            time.sleep(0.1)
            continue

        instance_existed = True
        instance = instances[instance_id]
        runner_ids = runner_ids_from_instance(instance)
        runners = state.get("runners", {})

        # Check for failed runners first
        for rid in runner_ids:
            runner = runners.get(rid, {})
            if runner_failed(runner):
                error_msg = get_runner_failed_message(runner) or "Unknown error"
                raise RuntimeError(f"Runner {rid} failed: {error_msg}")

        if all(runner_ready(runners.get(rid, {})) for rid in runner_ids):
            return

        time.sleep(0.1)

    raise TimeoutError(f"Instance {instance_id} did not become ready within {timeout=}")


def wait_for_instance_gone(
    client: ExoClient, instance_id: str, timeout: float = 3.0
) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            client.request_json("GET", f"/instance/{instance_id}")
            time.sleep(0.4)
        except ExoHttpError as e:
            if e.status == 404:
                return
            raise

    raise TimeoutError(f"Instance {instance_id} did not get deleted within {timeout=}")


def resolve_model_short_id(client: ExoClient, model_arg: str) -> tuple[str, str]:
    models = client.request_json("GET", "/models") or {}
    data = models.get("data") or []

    for m in data:
        if (m.get("name") or "").lower() == model_arg.lower():
            short_id = str(m["name"])
            full_id = str(m.get("hugging_face_id") or m["name"])
            return short_id, full_id

    for m in data:
        if m.get("hugging_face_id") == model_arg:
            short_id = str(m["name"])
            full_id = str(m["hugging_face_id"])
            return short_id, full_id

    raise ValueError(f"Model not found in /models: {model_arg}")


def placement_filter(instance_meta: str, wanted: str) -> bool:
    s = (instance_meta or "").lower()
    if wanted == "both":
        return ("ring" in s) or ("jaccl" in s)
    return wanted in s


def sharding_filter(sharding: str, wanted: str) -> bool:
    s = (sharding or "").lower()
    if wanted == "both":
        return ("pipeline" in s) or ("tensor" in s)
    return wanted in s


def fetch_and_filter_placements(
    client: ExoClient, full_model_id: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    previews_resp = client.request_json(
        "GET", "/instance/previews", params={"model_id": full_model_id}
    )
    previews = previews_resp.get("previews") or []

    selected: list[dict[str, Any]] = []
    for p in previews:
        if p.get("error") is not None:
            continue
        if not placement_filter(str(p.get("instance_meta", "")), args.instance_meta):
            continue
        if not sharding_filter(str(p.get("sharding", "")), args.sharding):
            continue

        instance = p.get("instance")
        if not isinstance(instance, dict):
            continue

        n = nodes_used_in_instance(instance)
        # Skip tensor ring single node as it is pointless when pipeline ring
        if n == 1 and (
            (args.sharding == "both" and "tensor" in p.get("sharding", "").lower())
            or (
                args.instance_meta == "both"
                and "jaccl" in p.get("instance_meta", "").lower()
            )
        ):
            continue

        if (
            args.skip_pipeline_jaccl
            and (
                args.instance_meta == "both"
                and "jaccl" in p.get("instance_meta", "").lower()
            )
            and (
                args.sharding == "both" and "pipeline" in p.get("sharding", "").lower()
            )
        ):
            continue

        if (
            args.skip_tensor_ring
            and (
                args.instance_meta == "both"
                and "ring" in p.get("instance_meta", "").lower()
            )
            and (args.sharding == "both" and "tensor" in p.get("sharding", "").lower())
        ):
            continue

        if args.min_nodes <= n <= args.max_nodes:
            selected.append(p)

    return selected


def settle_and_fetch_placements(
    client: ExoClient,
    full_model_id: str,
    args: argparse.Namespace,
    settle_timeout: float = 0,
) -> list[dict[str, Any]]:
    selected = fetch_and_filter_placements(client, full_model_id, args)

    if not selected and settle_timeout > 0:
        backoff = _SETTLE_INITIAL_BACKOFF_S
        deadline = time.monotonic() + settle_timeout
        while not selected and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            logger.warning(
                f"No valid placements yet (cluster may still be settling). "
                f"Retrying in {backoff:.1f}s ({remaining:.0f}s remaining)..."
            )
            time.sleep(min(backoff, remaining))
            backoff = min(backoff * _SETTLE_BACKOFF_MULTIPLIER, _SETTLE_MAX_BACKOFF_S)
            selected = fetch_and_filter_placements(client, full_model_id, args)

    return selected


def run_planning_phase(
    client: ExoClient,
    full_model_id: str,
    preview: dict[str, Any],
    danger_delete: bool,
    timeout: float,
    settle_deadline: float | None,
) -> float | None:
    """Check disk space and ensure model is downloaded before benchmarking.

    Returns the wall-clock download duration in seconds if a fresh download
    was needed, or None if the model was already cached on all nodes.
    """
    # Get model size from /models
    models = client.request_json("GET", "/models") or {}
    model_bytes = 0
    for m in models.get("data", []):
        if m.get("hugging_face_id") == full_model_id:
            model_bytes = m.get("storage_size_megabytes", 0) * 1024 * 1024
            break

    if not model_bytes:
        logger.warning(
            f"Could not determine size for {full_model_id}, skipping disk check"
        )
        return None

    # Get nodes from preview
    inner = unwrap_instance(preview["instance"])
    node_ids = list(inner["shardAssignments"]["nodeToRunner"].keys())
    runner_to_shard = inner["shardAssignments"]["runnerToShard"]

    state = client.request_json("GET", "/state")
    downloads = state.get("downloads", {})
    node_disk = state.get("nodeDisk", {})

    needs_download = False

    for node_id in node_ids:
        node_downloads = downloads.get(node_id, [])

        # Check if model already downloaded on this node
        already_downloaded = any(
            "DownloadCompleted" in p
            and unwrap_instance(p["DownloadCompleted"]["shardMetadata"])["modelCard"][
                "modelId"
            ]
            == full_model_id
            for p in node_downloads
        )
        if already_downloaded:
            continue

        needs_download = True

        # Wait for disk info if settle_deadline is set
        disk_info = node_disk.get(node_id, {})
        backoff = _SETTLE_INITIAL_BACKOFF_S
        while not disk_info and settle_deadline and time.monotonic() < settle_deadline:
            remaining = settle_deadline - time.monotonic()
            logger.info(
                f"Waiting for disk info on {node_id} ({remaining:.0f}s remaining)..."
            )
            time.sleep(min(backoff, remaining))
            backoff = min(backoff * _SETTLE_BACKOFF_MULTIPLIER, _SETTLE_MAX_BACKOFF_S)
            state = client.request_json("GET", "/state")
            node_disk = state.get("nodeDisk", {})
            disk_info = node_disk.get(node_id, {})

        if not disk_info:
            logger.warning(f"No disk info for {node_id}, skipping space check")
            continue

        avail = disk_info.get("available", {}).get("inBytes", 0)
        if avail >= model_bytes:
            continue

        if not danger_delete:
            raise RuntimeError(
                f"Insufficient disk on {node_id}: need {model_bytes // (1024**3)}GB, "
                f"have {avail // (1024**3)}GB. Use --danger-delete-downloads to free space."
            )

        # Delete from smallest to largest (skip read-only models from EXO_MODELS_PATH)
        completed = [
            (
                unwrap_instance(p["DownloadCompleted"]["shardMetadata"])["modelCard"][
                    "modelId"
                ],
                p["DownloadCompleted"]["total"]["inBytes"],
            )
            for p in node_downloads
            if "DownloadCompleted" in p
            and not p["DownloadCompleted"].get("readOnly", False)
        ]
        for del_model, size in sorted(completed, key=lambda x: x[1]):
            logger.info(f"Deleting {del_model} from {node_id} ({size // (1024**2)}MB)")
            client.request_json("DELETE", f"/download/{node_id}/{del_model}")
            avail += size
            if avail >= model_bytes:
                break

        if avail < model_bytes:
            raise RuntimeError(f"Could not free enough space on {node_id}")

    # Start downloads (idempotent)
    download_t0 = time.perf_counter() if needs_download else None
    for node_id in node_ids:
        runner_id = inner["shardAssignments"]["nodeToRunner"][node_id]
        shard = runner_to_shard[runner_id]
        client.request_json(
            "POST",
            "/download/start",
            body={
                "targetNodeId": node_id,
                "shardMetadata": shard,
            },
        )
        logger.info(f"Started download on {node_id}")

    # Wait for downloads
    start = time.time()
    while time.time() - start < timeout:
        state = client.request_json("GET", "/state")
        downloads = state.get("downloads", {})
        all_done = True
        for node_id in node_ids:
            done = any(
                "DownloadCompleted" in p
                and unwrap_instance(p["DownloadCompleted"]["shardMetadata"])[
                    "modelCard"
                ]["modelId"]
                == full_model_id
                for p in downloads.get(node_id, [])
            )
            failed = [
                p["DownloadFailed"]["errorMessage"]
                for p in downloads.get(node_id, [])
                if "DownloadFailed" in p
                and unwrap_instance(p["DownloadFailed"]["shardMetadata"])["modelCard"][
                    "modelId"
                ]
                == full_model_id
            ]
            if failed:
                raise RuntimeError(f"Download failed on {node_id}: {failed[0]}")
            if not done:
                all_done = False
        if all_done:
            if download_t0 is not None:
                return time.perf_counter() - download_t0
            return None
        time.sleep(1)

    raise TimeoutError("Downloads did not complete in time")


def add_common_instance_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--host", default=os.environ.get("EXO_HOST", "localhost"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("EXO_PORT", "52415"))
    )
    ap.add_argument("--model", required=True, help="Model short id or huggingface id")
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=4,
        help="Only consider placements using <= this many nodes.",
    )
    ap.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Only consider placements using >= this many nodes.",
    )
    ap.add_argument(
        "--instance-meta", choices=["ring", "jaccl", "both"], default="both"
    )
    ap.add_argument(
        "--sharding", choices=["pipeline", "tensor", "both"], default="both"
    )
    ap.add_argument(
        "--skip-pipeline-jaccl",
        action="store_true",
        help="Skip pipeline+jaccl placements, as it's often pointless.",
    )
    ap.add_argument(
        "--skip-tensor-ring",
        action="store_true",
        help="Skip tensor+ring placements, as it's so slow.",
    )
    ap.add_argument(
        "--timeout", type=float, default=7200.0, help="HTTP timeout (seconds)."
    )
    ap.add_argument(
        "--settle-timeout",
        type=float,
        default=0,
        help="Max seconds to wait for the cluster to produce valid placements (0 = try once).",
    )
    ap.add_argument(
        "--danger-delete-downloads",
        action="store_true",
        help="Delete existing models from smallest to largest to make room for benchmark model.",
    )
