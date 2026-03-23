"""
exo Cluster MCP Server

Exposes cluster health and state as MCP resources and tools so Claude can
query worldSize, shard assignments, and node health without manual curl.

Usage:
    uv run python -m exo.mcp_server.server

Or add to .mcp.json:
    "exo-cluster": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "python", "-m", "exo.mcp_server.server"],
        "cwd": "/Users/michaelpuodziukas/exo/exo"
    }
"""
from __future__ import annotations

import http.client
import json
import sys
import urllib.request
from typing import Any, cast

_EXO_API = "http://localhost:52415"


def _urlopen_read(url: str | urllib.request.Request, timeout: int) -> bytes:
    """Open URL, read body, close. Typed to return bytes."""
    conn: http.client.HTTPResponse = cast(
        "http.client.HTTPResponse",
        urllib.request.urlopen(url, timeout=timeout),
    )
    try:
        return conn.read()
    finally:
        conn.close()


def _get(path: str, timeout: int = 4) -> dict[str, Any] | None:
    try:
        raw = _urlopen_read(f"{_EXO_API}{path}", timeout=timeout)
        return cast("dict[str, Any]", json.loads(raw))
    except Exception:
        return None


def _cluster_health() -> dict[str, Any]:
    health = _get("/health")
    state = _get("/state")

    if health is None:
        return {"status": "DOWN", "worldSize": 0, "nodes": []}

    nodes: list[dict[str, Any]] = []
    world_size = 0

    if state is not None:
        instances: dict[str, dict[str, Any]] = cast(
            "dict[str, dict[str, Any]]",
            state.get("instances") or {},
        )
        for inst in instances.values():
            mlx: dict[str, Any] = cast("dict[str, Any]", inst.get("MlxRingInstance") or {})
            shard_assignments: dict[str, Any] = cast("dict[str, Any]", mlx.get("shardAssignments") or {})
            shard_map: dict[str, dict[str, Any]] = cast(
                "dict[str, dict[str, Any]]",
                shard_assignments.get("runnerToShard") or {},
            )
            for node_id, shard in shard_map.items():
                shard_vals: dict[str, Any] = cast("dict[str, Any]", list(shard.values())[0]) if shard else {}
                ws: int = cast("int", shard_vals.get("worldSize") or 0)
                if ws > world_size:
                    world_size = ws
                nodes.append({
                    "node_id": node_id[:12],
                    "rank": shard_vals.get("deviceRank"),
                    "layers": f"{shard_vals.get('startLayer')}-{shard_vals.get('endLayer')}",
                    "worldSize": ws,
                })

    return {
        "status": "UP" if health else "DOWN",
        "worldSize": world_size,
        "nodes": nodes,
        "healthy": world_size == 2,
    }


def _handle_initialize(req_id: str | int | None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"resources": {}, "tools": {}},
            "serverInfo": {"name": "exo-cluster", "version": "1.0.0"},
        },
    }


def _handle_list_resources(req_id: str | int | None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "resources": [
                {
                    "uri": "exo://cluster/health",
                    "name": "Cluster Health",
                    "description": "Current exo cluster health: worldSize, node assignments, status",
                    "mimeType": "application/json",
                },
                {
                    "uri": "exo://cluster/state",
                    "name": "Cluster State",
                    "description": "Raw exo cluster state from /state endpoint",
                    "mimeType": "application/json",
                },
                {
                    "uri": "exo://docs/claude-api",
                    "name": "Claude API Reference",
                    "description": "Full Claude API docs optimized for LLM ingestion — models, tools, streaming, caching",
                    "mimeType": "text/plain",
                },
            ]
        },
    }


def _fetch_claude_api_docs() -> str:
    try:
        raw = _urlopen_read("https://platform.claude.com/llms.txt", timeout=10)
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"# Claude API docs unavailable: {exc}\n\nCheck https://platform.claude.com/docs"


def _handle_read_resource(req_id: str | int | None, uri: str) -> dict[str, Any]:
    if uri == "exo://cluster/health":
        content = json.dumps(_cluster_health(), indent=2)
    elif uri == "exo://cluster/state":
        state = _get("/state") or {"error": "cluster unreachable"}
        content = json.dumps(state, indent=2)
    elif uri == "exo://docs/claude-api":
        content = _fetch_claude_api_docs()
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown resource: {uri}"}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "contents": [{"uri": uri, "mimeType": "application/json", "text": content}]
        },
    }


def _handle_list_tools(req_id: str | int | None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "cluster_health",
                    "description": "Check exo cluster health: worldSize, node ranks, layer assignments. worldSize=2 is healthy.",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": "cluster_inference",
                    "description": "Send a prompt to the exo cluster and get a completion. Uses the currently loaded model.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The prompt to send"},
                            "max_tokens": {"type": "integer", "description": "Max tokens to generate", "default": 256},
                        },
                        "required": ["prompt"],
                    },
                },
            ]
        },
    }


def _handle_call_tool(req_id: str | int | None, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "cluster_health":
        result = _cluster_health()
        text = json.dumps(result, indent=2)
    elif name == "cluster_inference":
        prompt = str(arguments.get("prompt") or "")
        max_tokens = int(cast("int", arguments.get("max_tokens") or 256))
        payload = json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        try:
            req = urllib.request.Request(
                f"{_EXO_API}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp_bytes = _urlopen_read(req, timeout=60)
            data: dict[str, Any] = cast("dict[str, Any]", json.loads(resp_bytes))
            choices: list[dict[str, Any]] = cast("list[dict[str, Any]]", data.get("choices") or [{}])
            message: dict[str, Any] = cast("dict[str, Any]", choices[0].get("message") or {})
            text = str(message.get("content") or "")
        except Exception as exc:
            text = f"Error: {exc}"
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"content": [{"type": "text", "text": text}]},
    }


def serve() -> None:
    """Run the MCP server on stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req: dict[str, Any] = cast("dict[str, Any]", json.loads(line))
        except json.JSONDecodeError:
            continue

        req_id: str | int | None = cast("str | int | None", req.get("id"))
        method: str = str(req.get("method") or "")
        params: dict[str, Any] = cast("dict[str, Any]", req.get("params") or {})

        if method == "initialize":
            resp = _handle_initialize(req_id)
        elif method == "resources/list":
            resp = _handle_list_resources(req_id)
        elif method == "resources/read":
            resp = _handle_read_resource(req_id, str(params.get("uri") or ""))
        elif method == "tools/list":
            resp = _handle_list_tools(req_id)
        elif method == "tools/call":
            resp = _handle_call_tool(
                req_id,
                str(params.get("name") or ""),
                cast("dict[str, Any]", params.get("arguments") or {}),
            )
        elif method == "notifications/initialized":
            continue  # notification, no response needed
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    serve()
