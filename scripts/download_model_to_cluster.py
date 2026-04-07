#!/usr/bin/env python3
"""Download a model to every node in an exo cluster, bypassing placement.

Usage:
    uv run python scripts/download_model_to_cluster.py zai-org/GLM-5.1 --host james

This fetches the ModelCard from HuggingFace locally (to get n_layers),
constructs a full-model PipelineShardMetadata (world_size=1, one shard
covering every layer), and POSTs /download/start to the target exo API
for each node currently in the topology. It then polls /state/downloads
until every node reports DownloadCompleted.

No placement is required. Works with a cluster of any size, including 1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

import httpx
from loguru import logger

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.worker.shards import PipelineShardMetadata


async def fetch_topology_nodes(client: httpx.AsyncClient, base: str) -> list[str]:
    r = await client.get(f"{base}/state/topology")
    r.raise_for_status()
    topology = r.json() or {}
    nodes = topology.get("nodes") or []
    if isinstance(nodes, dict):
        return list(nodes.keys())
    result: list[str] = []
    for n in nodes:
        if isinstance(n, str):
            result.append(n)
        elif isinstance(n, dict):
            result.append(str(n.get("nodeId") or n.get("node_id") or ""))
    return [nid for nid in result if nid]


def build_shard_payload(card: ModelCard) -> dict[str, Any]:
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    return shard.model_dump(mode="json", by_alias=True)


async def ensure_model_card_registered(
    client: httpx.AsyncClient, base: str, model_id: str
) -> None:
    r = await client.get(f"{base}/models")
    r.raise_for_status()
    data = (r.json() or {}).get("data") or []
    for m in data:
        if m.get("hugging_face_id") == model_id or m.get("id") == model_id:
            logger.info(f"Model already registered on cluster: {model_id}")
            return

    logger.info(f"Registering model on cluster via /models/add: {model_id}")
    r = await client.post(f"{base}/models/add", json={"model_id": model_id})
    if r.status_code >= 400:
        raise RuntimeError(f"/models/add failed ({r.status_code}): {r.text}")


def node_model_status(
    downloads_state: dict[str, Any], node_id: str, model_id: str
) -> str:
    entries = downloads_state.get(node_id) or []
    if not isinstance(entries, list):
        return "unknown"
    best = "not_present"
    for entry in entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        [(tag, payload)] = entry.items()
        shard_meta = (payload or {}).get("shardMetadata") or (payload or {}).get(
            "shard_metadata"
        )
        if not isinstance(shard_meta, dict) or len(shard_meta) != 1:
            continue
        [(_, inner)] = shard_meta.items()
        mc = (inner or {}).get("modelCard") or (inner or {}).get("model_card") or {}
        this_id = mc.get("modelId") or mc.get("model_id")
        if this_id != model_id:
            continue
        if tag == "DownloadCompleted":
            return "completed"
        if tag == "DownloadOngoing":
            best = "ongoing"
        elif tag == "DownloadFailed" and best == "not_present":
            best = "failed"
    return best


async def poll_until_complete(
    client: httpx.AsyncClient,
    base: str,
    node_ids: list[str],
    model_id: str,
    timeout_s: float,
) -> None:
    start = time.monotonic()
    while True:
        r = await client.get(f"{base}/state/downloads")
        r.raise_for_status()
        downloads_state = r.json() or {}

        statuses = {
            nid: node_model_status(downloads_state, nid, model_id) for nid in node_ids
        }

        for nid, status in statuses.items():
            entries = downloads_state.get(nid) or []
            if status == "ongoing":
                for entry in entries:
                    if not isinstance(entry, dict) or "DownloadOngoing" not in entry:
                        continue
                    prog = (entry["DownloadOngoing"] or {}).get(
                        "downloadProgress"
                    ) or {}
                    dl_b = (prog.get("downloaded") or {}).get("inBytes") or 0
                    total_b = (prog.get("total") or {}).get("inBytes") or 0
                    pct = (dl_b / total_b * 100) if total_b else 0.0
                    speed = (prog.get("speed") or 0) / (1024 * 1024)
                    logger.info(f"{nid}: {pct:.1f}% @ {speed:.1f} MB/s")
                    break

        if all(s == "completed" for s in statuses.values()):
            logger.info(f"Download complete on all nodes: {list(statuses.keys())}")
            return

        failed = [nid for nid, s in statuses.items() if s == "failed"]
        if failed:
            raise RuntimeError(f"Download failed on nodes: {failed}")

        if time.monotonic() - start > timeout_s:
            pending = [nid for nid, s in statuses.items() if s != "completed"]
            raise TimeoutError(
                f"Downloads did not complete within {timeout_s}s; pending: {pending}"
            )

        await asyncio.sleep(2)


async def run(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}"
    model_id = args.model

    logger.info(f"Fetching ModelCard for {model_id} from HuggingFace...")
    card = await ModelCard.fetch_from_hf(ModelId(model_id))
    logger.info(
        f"Card: n_layers={card.n_layers}, "
        f"storage={card.storage_size.in_gb:.1f}GB, "
        f"quant={card.quantization or '-'}"
    )

    shard_payload = build_shard_payload(card)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        await ensure_model_card_registered(client, base, model_id)

        node_ids = await fetch_topology_nodes(client, base)
        if not node_ids:
            logger.error("No nodes in topology on {}", base)
            return 1
        logger.info(f"Topology has {len(node_ids)} node(s): {node_ids}")

        for node_id in node_ids:
            payload = {"targetNodeId": node_id, "shardMetadata": shard_payload}
            logger.info(f"POST /download/start -> {node_id}")
            r = await client.post(f"{base}/download/start", json=payload)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"/download/start for {node_id} failed "
                    f"({r.status_code}): {r.text}"
                )

        logger.info("Polling for completion...")
        await poll_until_complete(
            client, base, node_ids, model_id, timeout_s=args.timeout
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model id, e.g. zai-org/GLM-5.1")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=52415)
    parser.add_argument(
        "--timeout",
        type=float,
        default=14400.0,
        help="HTTP + overall wait timeout (seconds). Default 4h.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
