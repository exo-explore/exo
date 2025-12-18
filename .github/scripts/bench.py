#!/usr/bin/env python3

# type: ignore
"""
Unified benchmark script for EXO.
Runs single or multi-stage benchmarks with configurable load patterns.
Requests are fire-and-forget, allowing overlapping execution.

Simple benchmark (1 iteration):    --config .github/configs/bench_simple.yaml
Complex benchmark (multiple stages): --config .github/configs/bench_config.yaml
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _format_http_error(error: Exception) -> str:
    """Format HTTP error with full response details for debugging."""
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unable to read body>"

        headers_str = (
            "\n".join(f"  {k}: {v}" for k, v in error.headers.items())
            if error.headers
            else "<no headers>"
        )

        return (
            f"HTTP {error.code} {error.reason}\n"
            f"URL: {error.url}\n"
            f"Headers:\n{headers_str}\n"
            f"Body: {body}"
        )
    elif isinstance(error, urllib.error.URLError):
        return f"URLError: {error.reason}"
    else:
        return str(error)


def _http_request(
    url: str, *, method: str = "GET", data: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    payload: bytes | None = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:  # nosec - runner-local API
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw": body}
    except Exception as e:
        error_details = _format_http_error(e)
        print(f"HTTP request failed:\n{error_details}")
        raise


async def _http_request_async(
    url: str, *, method: str = "GET", data: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Async version that runs in executor to not block event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: _http_request(url, method=method, data=data)
    )


async def _http_stream_async(
    url: str, *, method: str = "POST", data: Mapping[str, Any], timeout: int = 300
) -> list[tuple[str, float]]:
    """Async streaming request. Returns list of (line, timestamp) tuples."""

    def _stream() -> list[tuple[str, float]]:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        lines: list[tuple[str, float]] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - runner-local API
                for raw_line in resp:
                    timestamp = time.monotonic()
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if line:
                        lines.append((line, timestamp))
            return lines
        except Exception as e:
            error_details = _format_http_error(e)
            print(f"HTTP request failed:\n{error_details}")
            raise

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _stream)


def fetch_state(api_base: str) -> dict[str, Any]:
    return _http_request(f"{api_base}/state")


def unwrap_tagged_union(obj: Any) -> tuple[str | None, Any]:
    """Extract tag and payload from tagged union format {Tag: {fields...}}.

    Returns (tag_name, payload) if the object is a tagged union, otherwise (None, obj).
    """
    if not isinstance(obj, dict):
        return None, obj

    keys = list(obj.keys())
    if len(keys) == 1 and isinstance(keys[0], str):
        tag = keys[0]
        payload = obj[tag]
        return tag, payload

    return None, obj


def collect_metrics_snapshot(state: Mapping[str, Any]) -> MetricsSnapshot:
    """Collect current metrics snapshot from state."""
    timestamp = time.time()

    # Collect memory for each node
    node_memory: dict[str, MemorySnapshot] = {}
    node_profiles: Mapping[str, Any] = state.get("nodeProfiles", {})

    for node_id, profile in node_profiles.items():
        if not isinstance(profile, dict):
            continue

        memory = profile.get("memory", {})
        if not isinstance(memory, dict):
            continue

        # Parse memory values - they're objects with 'inBytes' field
        def get_bytes(mem_obj: Any) -> int:
            if isinstance(mem_obj, dict):
                return int(mem_obj.get("inBytes", 0))
            return 0

        ram_total = get_bytes(memory.get("ramTotal"))
        ram_available = get_bytes(memory.get("ramAvailable"))
        swap_total = get_bytes(memory.get("swapTotal"))
        swap_available = get_bytes(memory.get("swapAvailable"))

        node_memory[node_id] = MemorySnapshot(
            ram_total_bytes=ram_total,
            ram_available_bytes=ram_available,
            ram_used_bytes=max(ram_total - ram_available, 0),
            swap_total_bytes=swap_total,
            swap_available_bytes=swap_available,
            swap_used_bytes=max(swap_total - swap_available, 0),
        )

    # Collect task counts per instance and per node
    instance_tasks: list[InstanceTaskSnapshot] = []
    instances: Mapping[str, Any] = state.get("instances", {})
    tasks: Mapping[str, Any] = state.get("tasks", {})
    print(f"[DEBUG] Num tasks: {len(tasks)}. Num instances: {len(instances)}.")

    # Map instance_id -> node_ids (instances can span multiple nodes)
    instance_to_nodes: dict[str, set[str]] = {}
    for instance_id, instance_wrapped in instances.items():
        # Unwrap tagged Instance union (MlxRingInstance or MlxIbvInstance)
        _instance_tag, instance_data = unwrap_tagged_union(instance_wrapped)
        if not isinstance(instance_data, dict):
            continue

        shard_assignments = instance_data.get("shardAssignments", {})
        if not isinstance(shard_assignments, dict):
            continue

        # Get all nodes that this instance uses
        node_to_runner = shard_assignments.get("nodeToRunner", {})
        if isinstance(node_to_runner, dict):
            instance_to_nodes[instance_id] = set(node_to_runner.keys())

    # Count tasks per instance (only Pending and Running exist in state; completed tasks are deleted)
    instance_task_counts: dict[str, dict[str, int]] = {}
    for instance_id in instances:
        instance_task_counts[instance_id] = {
            "Pending": 0,
            "Running": 0,
        }

    # Iterate through tasks and count by instance and status
    tasks_matched = 0
    tasks_skipped = 0

    for _task_id, task_wrapper in tasks.items():
        if not isinstance(task_wrapper, dict):
            print(f"[DEBUG] Task wrapper is not a dict: {task_wrapper}")
            tasks_skipped += 1
            continue

        # Extract actual task from wrapper (e.g., {"ChatCompletion": {...}})
        if len(task_wrapper) != 1:
            print(
                f"[DEBUG] Task wrapper has unexpected number of keys: {len(task_wrapper)}"
            )
            tasks_skipped += 1
            continue

        _task_type, task_data = next(iter(task_wrapper.items()))

        if not isinstance(task_data, dict):
            print(f"[DEBUG] Task data is not a dict: {task_data}")
            tasks_skipped += 1
            continue

        instance_id = task_data.get("instanceId")
        task_status = task_data.get("taskStatus")

        if not instance_id or instance_id not in instance_task_counts:
            tasks_skipped += 1
            continue

        if task_status not in ["Pending", "Running"]:
            tasks_skipped += 1
            continue

        # Count this task
        instance_task_counts[instance_id][task_status] += 1
        tasks_matched += 1

    if tasks_skipped > 0:
        print(
            f"[DEBUG] Task matching: {tasks_matched} matched, {tasks_skipped} skipped (from {len(tasks)} total)"
        )

    # Build snapshots for each instance (assign to primary node - first in sorted order)
    for instance_id, counts in instance_task_counts.items():
        pending = counts["Pending"]
        running = counts["Running"]
        total_active = pending + running

        node_ids = instance_to_nodes.get(instance_id, set())
        primary_node = sorted(node_ids)[0] if node_ids else "unknown"

        instance_tasks.append(
            InstanceTaskSnapshot(
                instance_id=instance_id,
                node_id=primary_node,
                pending_tasks=pending,
                running_tasks=running,
                total_active_tasks=total_active,
            )
        )

    # Aggregate tasks per node
    node_task_counts: dict[str, dict[str, int]] = {}
    node_instance_counts: dict[str, int] = {}

    for instance_snapshot in instance_tasks:
        node_id = instance_snapshot.node_id

        if node_id not in node_task_counts:
            node_task_counts[node_id] = {
                "Pending": 0,
                "Running": 0,
            }
            node_instance_counts[node_id] = 0

        node_task_counts[node_id]["Pending"] += instance_snapshot.pending_tasks
        node_task_counts[node_id]["Running"] += instance_snapshot.running_tasks
        node_instance_counts[node_id] += 1

    # Build node snapshots
    node_tasks: list[NodeTaskSnapshot] = []
    for node_id, counts in node_task_counts.items():
        pending = counts["Pending"]
        running = counts["Running"]
        total_active = pending + running

        node_tasks.append(
            NodeTaskSnapshot(
                node_id=node_id,
                pending_tasks=pending,
                running_tasks=running,
                total_active_tasks=total_active,
                instance_count=node_instance_counts.get(node_id, 0),
            )
        )

    return MetricsSnapshot(
        timestamp=timestamp,
        node_memory=node_memory,
        instance_tasks=instance_tasks,
        node_tasks=node_tasks,
    )


def get_topology_node_count(state: Mapping[str, Any]) -> int:
    """Get the number of nodes in the topology."""
    topology = state.get("topology", {})
    nodes = topology.get("nodes", [])
    return len(nodes)


def count_instances_by_model(state: Mapping[str, Any], model_id: str) -> int:
    """Count how many instances exist for a given model_id."""
    instances: Mapping[str, Any] = state.get("instances", {})
    count = 0
    for instance_wrapped in instances.values():
        # Unwrap tagged Instance union
        _instance_tag, instance_data = unwrap_tagged_union(instance_wrapped)
        if not isinstance(instance_data, dict):
            continue

        shard = instance_data.get("shardAssignments", {})
        if isinstance(shard, dict) and shard.get("modelId") == model_id:
            count += 1
    return count


def get_all_instance_ids_for_model(
    state: Mapping[str, Any], model_id: str
) -> list[str]:
    """Get all instance IDs for a given model_id."""
    instances: Mapping[str, Any] = state.get("instances", {})
    instance_ids = []
    for instance_id, instance_wrapped in instances.items():
        # Unwrap tagged Instance union
        _instance_tag, instance_data = unwrap_tagged_union(instance_wrapped)
        if not isinstance(instance_data, dict):
            continue

        shard = instance_data.get("shardAssignments", {})
        if isinstance(shard, dict) and shard.get("modelId") == model_id:
            instance_ids.append(instance_id)
    return instance_ids


def count_ready_instances_by_model(state: Mapping[str, Any], model_id: str) -> int:
    """Count how many instances for a model have all their runners ready."""
    instances: Mapping[str, Any] = state.get("instances", {})
    ready_count = 0

    for instance_id, instance_wrapped in instances.items():
        # Unwrap tagged Instance union
        _instance_tag, instance_data = unwrap_tagged_union(instance_wrapped)
        if not isinstance(instance_data, dict):
            continue

        shard = instance_data.get("shardAssignments", {})
        if not isinstance(shard, dict) or shard.get("modelId") != model_id:
            continue

        # Check if all runners for this instance are ready
        runner_ids = get_runner_ids_for_instance(state, instance_id)
        if len(runner_ids) == 0:
            continue

        # Fixed runner status names: RunnerReady and RunnerRunning (not LoadedRunnerStatus/RunningRunnerStatus)
        all_ready = all(
            get_runner_status_kind(state, rid) in {"RunnerReady", "RunnerRunning"}
            for rid in runner_ids
        )

        if all_ready:
            ready_count += 1

    return ready_count


def get_runner_ids_for_instance(
    state: Mapping[str, Any], instance_id: str
) -> list[str]:
    instances: Mapping[str, Any] = state.get("instances", {})
    instance_wrapped = instances.get(instance_id, {})

    # Unwrap tagged Instance union
    _instance_tag, instance_data = unwrap_tagged_union(instance_wrapped)
    if not isinstance(instance_data, dict):
        return []

    shard_assignments = instance_data.get("shardAssignments", {})
    if not isinstance(shard_assignments, dict):
        return []

    r2s = shard_assignments.get("runnerToShard", {})
    if isinstance(r2s, dict):
        return list(r2s.keys())
    return []


def get_runner_status_kind(state: Mapping[str, Any], runner_id: str) -> str | None:
    runners: Mapping[str, Any] = state.get("runners", {})
    status_obj = runners.get(runner_id)
    if not isinstance(status_obj, dict):
        return None
    if len(status_obj) == 1:
        return next(iter(status_obj.keys()))
    return None


async def wait_for_topology_ready(
    api_base: str, expected_nodes: int, timeout_s: int
) -> None:
    """Wait for all expected nodes to appear in the topology."""
    print(
        f"Waiting for {expected_nodes} node(s) to appear in topology (timeout: {timeout_s}s)..."
    )
    start = time.monotonic()
    while True:
        state = fetch_state(api_base)
        node_count = get_topology_node_count(state)
        elapsed = time.monotonic() - start
        print(
            f"  Topology has {node_count}/{expected_nodes} nodes (elapsed: {elapsed:.1f}s)"
        )

        if node_count >= expected_nodes:
            print(f"All {expected_nodes} node(s) are in topology!")
            return

        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for topology. Expected {expected_nodes} nodes, got {node_count}"
            )
        await asyncio.sleep(2)


async def wait_for_instances_ready(
    api_base: str, model_id: str, expected_count: int, timeout_s: int
) -> list[str]:
    """Wait for a specific count of instances for a model to be fully ready."""
    print(
        f"Waiting for {expected_count} instance(s) of {model_id} to be ready (timeout: {timeout_s}s)..."
    )
    start = time.monotonic()
    while True:
        state = fetch_state(api_base)

        total_count = count_instances_by_model(state, model_id)
        ready_count = count_ready_instances_by_model(state, model_id)
        elapsed = time.monotonic() - start

        print(
            f"  Model {model_id}: {ready_count}/{expected_count} ready ({total_count} total) (elapsed: {elapsed:.1f}s)"
        )

        if ready_count >= expected_count:
            instance_ids = get_all_instance_ids_for_model(state, model_id)
            print(
                f"All {expected_count} instance(s) ready! Instance IDs: {instance_ids}"
            )
            return instance_ids

        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for instances. Expected {expected_count} ready instances of {model_id}, "
                f"got {ready_count} ready out of {total_count} total"
            )
        await asyncio.sleep(2)


async def wait_for_all_instances_deleted(api_base: str, model_id: str) -> None:
    """Wait for all instances of a model to be deleted."""
    print(f"Waiting for all instances of {model_id} to be deleted...")
    start = time.monotonic()
    while True:
        state = fetch_state(api_base)
        count = count_instances_by_model(state, model_id)
        if count == 0:
            elapsed = time.monotonic() - start
            print(f"All instances of {model_id} deleted after {elapsed:.1f}s")
            return
        await asyncio.sleep(2)


async def wait_for_tasks_drained(api_base: str, timeout_s: int = 600) -> None:
    """Wait for all tasks in the cluster to be drained (completed or failed).

    Tasks are deleted from state when complete, so we wait until there are no
    pending or running tasks remaining.
    """
    print(f"\n{'=' * 80}")
    print("⏳ WAITING FOR ALL TASKS TO DRAIN")
    print(f"{'=' * 80}")
    start = time.monotonic()

    while True:
        state = fetch_state(api_base)
        snapshot = collect_metrics_snapshot(state)

        # Count total active tasks across all nodes
        total_pending = sum(node.pending_tasks for node in snapshot.node_tasks)
        total_running = sum(node.running_tasks for node in snapshot.node_tasks)
        total_active = total_pending + total_running

        elapsed = time.monotonic() - start

        if total_active == 0:
            print(f"✅ All tasks drained after {elapsed:.1f}s")
            return

        print(
            f"  [{elapsed:.1f}s] Still draining: {total_active} active tasks ({total_pending} pending, {total_running} running)"
        )

        # Print per-node breakdown if there are active tasks
        if snapshot.node_tasks:
            for node_snapshot in snapshot.node_tasks:
                if node_snapshot.total_active_tasks > 0:
                    node_short = node_snapshot.node_id[-4:]
                    print(
                        f"    Node ...{node_short}: {node_snapshot.running_tasks} running, {node_snapshot.pending_tasks} pending"
                    )

        if elapsed > timeout_s:
            print(
                f"⚠️  WARNING: Timed out waiting for tasks to drain after {timeout_s}s"
            )
            print(
                f"   Remaining: {total_active} tasks ({total_pending} pending, {total_running} running)"
            )
            return

        await asyncio.sleep(2)


def generate_prompt(length: int) -> str:
    """Generate a prompt of approximately the specified token length."""
    # Rough approximation: 1 token ≈ 4 characters
    # Use a repeating pattern that's easy to generate
    base_text = "The quick brown fox jumps over the lazy dog. "
    target_chars = length * 4
    repetitions = (target_chars // len(base_text)) + 1
    return (base_text * repetitions)[:target_chars]


@dataclass(frozen=True)
class StageConfig:
    name: str
    prompt_length: int
    generation_length: int
    time_between_requests: float
    iterations: int


@dataclass
class RequestResult:
    request_id: int
    success: bool
    tokens: int
    elapsed_s: float
    started_at: float
    completed_at: float
    time_to_first_token_s: float | None = None
    decode_tps: float | None = None
    error: str | None = None


@dataclass
class StageResult:
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_tokens: int
    total_time: float
    avg_tokens_per_request: float
    avg_time_per_request: float
    avg_time_to_first_token: float | None
    std_time_to_first_token: float | None
    avg_decode_tps: float | None
    avg_ms_per_token: float | None
    std_ms_per_token: float | None
    request_results: list[RequestResult]
    stage_started_at: float
    stage_completed_at: float


@dataclass(frozen=True)
class MemorySnapshot:
    """Memory snapshot for a node at a point in time."""

    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_bytes: int
    swap_total_bytes: int
    swap_available_bytes: int
    swap_used_bytes: int


@dataclass(frozen=True)
class InstanceTaskSnapshot:
    """Task counts for an instance at a point in time.

    Note: Tasks are deleted from state when complete, so we only track active tasks.
    total_active_tasks = pending + running.
    """

    instance_id: str
    node_id: str
    pending_tasks: int
    running_tasks: int
    total_active_tasks: int


@dataclass(frozen=True)
class NodeTaskSnapshot:
    """Task counts for a node at a point in time.

    Note: Tasks are deleted from state when complete, so we only track active tasks.
    total_active_tasks = pending + running across all instances on this node.
    """

    node_id: str
    pending_tasks: int
    running_tasks: int
    total_active_tasks: int
    instance_count: int


@dataclass(frozen=True)
class MetricsSnapshot:
    """System metrics snapshot at a point in time."""

    timestamp: float
    node_memory: dict[str, MemorySnapshot]
    instance_tasks: list[InstanceTaskSnapshot]
    node_tasks: list[NodeTaskSnapshot]


async def run_single_request(
    api_base: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    timeout: int = 300,
) -> RequestResult:
    """Run a single chat completion request and return its result."""
    started_at = time.time()
    start = time.monotonic()
    try:
        lines = await _http_stream_async(
            f"{api_base}/v1/chat/completions",
            method="POST",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=timeout,
        )

        tokens = 0
        got_done = False
        first_token_time: float | None = None
        last_token_time: float | None = None

        for line, timestamp in lines:
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                got_done = True
                break
            try:
                obj = json.loads(payload)
                content = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = timestamp
                    last_token_time = timestamp
                    tokens += 1
            except json.JSONDecodeError:
                continue

        elapsed = time.monotonic() - start
        completed_at = time.time()

        # Calculate TTFT and decode TPS
        time_to_first_token: float | None = None
        decode_tps: float | None = None

        if first_token_time is not None:
            time_to_first_token = first_token_time - start

            # Decode TPS: tokens per second after first token
            if last_token_time is not None and tokens > 1:
                decode_time = last_token_time - first_token_time
                if decode_time > 0:
                    decode_tps = (tokens - 1) / decode_time

        # Request is only successful if we got at least one token AND a [DONE] marker
        if tokens == 0:
            print(
                f"  Request #{request_id}: FAILED - no tokens generated in {elapsed:.2f}s"
            )
            return RequestResult(
                request_id=request_id,
                success=False,
                tokens=0,
                elapsed_s=elapsed,
                started_at=started_at,
                completed_at=completed_at,
                time_to_first_token_s=time_to_first_token,
                decode_tps=decode_tps,
                error="No tokens generated",
            )

        if not got_done:
            print(
                f"  Request #{request_id}: FAILED - incomplete response (no [DONE]) after {elapsed:.2f}s"
            )
            return RequestResult(
                request_id=request_id,
                success=False,
                tokens=tokens,
                elapsed_s=elapsed,
                started_at=started_at,
                completed_at=completed_at,
                time_to_first_token_s=time_to_first_token,
                decode_tps=decode_tps,
                error="Incomplete response (no [DONE] marker)",
            )

        ttft_str = (
            f"{time_to_first_token:.3f}s" if time_to_first_token is not None else "N/A"
        )
        tps_str = f"{decode_tps:.1f} t/s" if decode_tps is not None else "N/A"
        print(
            f"  Request #{request_id}: SUCCESS - {tokens} tokens in {elapsed:.2f}s (TTFT: {ttft_str}, Decode: {tps_str})"
        )
        return RequestResult(
            request_id=request_id,
            success=True,
            tokens=tokens,
            elapsed_s=elapsed,
            started_at=started_at,
            completed_at=completed_at,
            time_to_first_token_s=time_to_first_token,
            decode_tps=decode_tps,
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        completed_at = time.time()
        error_details = _format_http_error(e)
        print(f"  Request #{request_id}: FAILED - {error_details}")
        return RequestResult(
            request_id=request_id,
            success=False,
            tokens=0,
            elapsed_s=elapsed,
            started_at=started_at,
            completed_at=completed_at,
            time_to_first_token_s=None,
            decode_tps=None,
            error=error_details,
        )


async def monitor_metrics(
    api_base: str,
    metrics_snapshots: list[MetricsSnapshot],
    stop_event: asyncio.Event,
    interval_seconds: float = 5.0,
) -> None:
    """Background task that collects metrics snapshots every interval_seconds."""
    print(f"\n{'=' * 80}")
    print(f"🔍 METRICS MONITORING STARTED (polling every {interval_seconds}s)")
    print(f"{'=' * 80}\n")

    snapshot_count = 0
    while not stop_event.is_set():
        try:
            snapshot_count += 1
            state = fetch_state(api_base)
            snapshot = collect_metrics_snapshot(state)
            metrics_snapshots.append(snapshot)

            # Print detailed summary
            node_count = len(snapshot.node_memory)
            instance_count = len(snapshot.instance_tasks)

            # Aggregate task counts from node level (only active tasks in state)
            total_pending = sum(node.pending_tasks for node in snapshot.node_tasks)
            total_running = sum(node.running_tasks for node in snapshot.node_tasks)
            total_active = sum(node.total_active_tasks for node in snapshot.node_tasks)

            # Print detailed breakdown
            print(
                f"\n[METRICS #{snapshot_count}] {node_count} nodes, {instance_count} instances | Active Tasks: {total_active} ({total_pending} pending, {total_running} running)"
            )

            # Print per-node breakdown (only if there are nodes)
            if snapshot.node_tasks:
                for node_snapshot in snapshot.node_tasks:
                    node_short = node_snapshot.node_id[-4:]
                    print(
                        f"  Node ...{node_short}: {node_snapshot.total_active_tasks} active ({node_snapshot.pending_tasks} pending, {node_snapshot.running_tasks} running) across {node_snapshot.instance_count} instances"
                    )

        except Exception as e:
            print(f"[METRICS] Error collecting snapshot: {e}")
            import traceback

            traceback.print_exc()

        # Wait for interval or until stopped
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)


async def run_stage(
    api_base: str,
    model_id: str,
    stage: StageConfig,
    no_overlap: bool = False,
) -> StageResult:
    """Run a single benchmark stage with fire-and-forget requests (or sequential if no_overlap=True)."""
    print("=" * 80)
    print(f"STAGE: {stage.name}")
    print("=" * 80)
    print(f"  Prompt Length:       {stage.prompt_length} tokens")
    print(f"  Generation Length:   {stage.generation_length} tokens")
    print(f"  Time Between Reqs:   {stage.time_between_requests}s")
    print(f"  Iterations:          {stage.iterations}")
    print(f"  No Overlap:          {no_overlap}")
    print("=" * 80)

    stage_started_at = time.time()
    prompt = generate_prompt(stage.prompt_length)
    results: list[RequestResult] = []

    if no_overlap:
        # Sequential execution: wait for each request to complete before starting next
        print("\nRunning requests sequentially (no overlap)...")
        for i in range(stage.iterations):
            result = await run_single_request(
                api_base, model_id, prompt, stage.generation_length, i + 1
            )
            results.append(result)

            # Wait before starting next request (except after last one)
            if i < stage.iterations - 1:
                await asyncio.sleep(stage.time_between_requests)
    else:
        # Concurrent execution: fire-and-forget with delays between starts
        print("\nRunning requests concurrently (with overlap)...")
        tasks: list[asyncio.Task[RequestResult]] = []

        # Fire off requests with delays between them
        for i in range(stage.iterations):
            task = asyncio.create_task(
                run_single_request(
                    api_base, model_id, prompt, stage.generation_length, i + 1
                )
            )
            tasks.append(task)

            # Wait before firing next request (except after last one)
            if i < stage.iterations - 1:
                await asyncio.sleep(stage.time_between_requests)

        # Wait for all requests to complete
        print(f"\nWaiting for all {len(tasks)} HTTP requests to complete...")
        results = list(await asyncio.gather(*tasks))

    # Wait for all tasks in the cluster to be drained
    print("\nHTTP requests completed. Now waiting for cluster tasks to drain...")
    await wait_for_tasks_drained(api_base, timeout_s=600)

    stage_completed_at = time.time()

    # Compute statistics
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    success_rate = successful / len(results) if results else 0.0
    total_tokens = sum(r.tokens for r in results)
    total_time = sum(r.elapsed_s for r in results)
    avg_tokens = total_tokens / successful if successful > 0 else 0.0
    avg_time = total_time / successful if successful > 0 else 0.0

    # Calculate average TTFT and decode TPS for successful requests only
    successful_results = [r for r in results if r.success]

    # Skip first iteration if there are more than 1 iterations (warmup)
    results_for_stats = (
        successful_results[1:] if len(successful_results) > 1 else successful_results
    )

    # TTFT statistics
    ttft_values = [
        r.time_to_first_token_s
        for r in results_for_stats
        if r.time_to_first_token_s is not None
    ]
    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None

    if avg_ttft is not None and len(ttft_values) > 1:
        variance_ttft = sum((x - avg_ttft) ** 2 for x in ttft_values) / len(ttft_values)
        std_ttft = variance_ttft**0.5
    else:
        std_ttft = None

    # Decode TPS and ms per token statistics
    decode_tps_values = [
        r.decode_tps for r in results_for_stats if r.decode_tps is not None
    ]
    avg_decode_tps = (
        sum(decode_tps_values) / len(decode_tps_values) if decode_tps_values else None
    )

    # Convert to ms per token
    ms_per_token_values = (
        [1000.0 / tps for tps in decode_tps_values] if decode_tps_values else []
    )
    avg_ms_per_token = (
        sum(ms_per_token_values) / len(ms_per_token_values)
        if ms_per_token_values
        else None
    )

    if avg_ms_per_token is not None and len(ms_per_token_values) > 1:
        variance_ms_per_token = sum(
            (x - avg_ms_per_token) ** 2 for x in ms_per_token_values
        ) / len(ms_per_token_values)
        std_ms_per_token = variance_ms_per_token**0.5
    else:
        std_ms_per_token = None

    return StageResult(
        name=stage.name,
        total_requests=len(results),
        successful_requests=successful,
        failed_requests=failed,
        success_rate=success_rate,
        total_tokens=total_tokens,
        total_time=total_time,
        avg_tokens_per_request=avg_tokens,
        avg_time_per_request=avg_time,
        avg_time_to_first_token=avg_ttft,
        std_time_to_first_token=std_ttft,
        avg_decode_tps=avg_decode_tps,
        avg_ms_per_token=avg_ms_per_token,
        std_ms_per_token=std_ms_per_token,
        request_results=list(results),
        stage_started_at=stage_started_at,
        stage_completed_at=stage_completed_at,
    )


async def run_benchmark(
    api_base: str,
    config_path: Path,
    expected_nodes: int,
    is_primary: bool,
    timeout_seconds: int,
    results_output_path: Path | None = None,
    git_commit: str | None = None,
    hardware_labels: list[str] | None = None,
) -> int:
    """Run the full staged benchmark."""
    benchmark_started_at = time.time()

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Support both model_id (legacy) and model_ids (new)
    if "model_ids" in config:
        model_ids = config["model_ids"]
    elif "model_id" in config:
        model_ids = [config["model_id"]]
    else:
        raise ValueError("Config must contain either 'model_id' or 'model_ids'")

    # Get sharding and instance_meta (optional, defaults to None if not specified)
    sharding: str | None = config.get("sharding")
    instance_meta: str | None = config.get("instance_meta")

    # Get no_overlap flag (optional, defaults to False)
    no_overlap: bool = config.get("no_overlap", False)

    stages = [StageConfig(**s) for s in config["stages"]]

    print("=" * 80)
    print("EXO BENCHMARK")
    print("=" * 80)
    print(f"Configuration File: {config_path}")
    print(f"Model IDs:          {model_ids}")
    print(f"Instance Count:     {len(model_ids)}")
    print(
        f"Sharding:           {sharding if sharding else 'not specified (defaults to Pipeline)'}"
    )
    print(
        f"Instance Type:      {instance_meta if instance_meta else 'not specified (defaults to MlxRing)'}"
    )
    print(f"No Overlap:         {no_overlap}")
    print(f"Stages:             {len(stages)}")
    print(f"Expected Nodes:     {expected_nodes}")
    print(f"Is Primary:         {is_primary}")
    print("=" * 80)

    try:
        # Wait for all nodes to join the topology first
        await wait_for_topology_ready(
            api_base, expected_nodes, timeout_s=timeout_seconds
        )

        # Add 30 second delay to allow topology to stabilize before creating instances
        print(
            "\nWaiting 30 seconds for topology to stabilize before creating instances..."
        )
        await asyncio.sleep(30)
        print("Proceeding with instance creation\n")

        # Count how many instances we need for each unique model_id
        from collections import Counter

        model_counts = Counter(model_ids)

        print("\nTarget instance counts by model:")
        for model_id, count in model_counts.items():
            print(f"  {model_id}: {count} instance(s)")
        print()

        # Track all instance IDs (collected at the end)
        all_instance_ids: list[str] = []

        if is_primary:
            # Primary: create instances one at a time, waiting for count to increase
            for idx, model_id in enumerate(model_ids):
                # Determine current and target counts for this model
                current_state = fetch_state(api_base)
                current_ready = count_ready_instances_by_model(current_state, model_id)
                target_count = current_ready + 1

                print("=" * 80)
                print(
                    f"[PRIMARY] Creating instance {idx + 1}/{len(model_ids)} for model: {model_id}"
                )
                print(
                    f"[PRIMARY] Current ready count for {model_id}: {current_ready}, target: {target_count}"
                )

                # Build instance creation request data
                instance_data: dict[str, Any] = {"model_id": model_id}
                if sharding is not None:
                    instance_data["sharding"] = sharding
                if instance_meta is not None:
                    instance_data["instance_meta"] = instance_meta

                response = await _http_request_async(
                    f"{api_base}/instance", method="POST", data=instance_data
                )
                print(f"[PRIMARY] Instance creation response: {response}")

                # Wait for one more instance of this model to be ready
                await wait_for_instances_ready(
                    api_base, model_id, target_count, timeout_s=timeout_seconds
                )
                print(f"[PRIMARY] Instance {idx + 1}/{len(model_ids)} is ready")
                print("=" * 80)
        else:
            # Secondary: wait for expected counts of each model to be ready
            print("[SECONDARY] Waiting for all instances to be created and ready...")
            for model_id, expected_count in model_counts.items():
                await wait_for_instances_ready(
                    api_base, model_id, expected_count, timeout_s=timeout_seconds
                )

        # Collect all instance IDs for all models
        state = fetch_state(api_base)
        for model_id in model_counts:
            ids = get_all_instance_ids_for_model(state, model_id)
            all_instance_ids.extend(ids)

        # Count total runners
        total_runners = 0
        for instance_id in all_instance_ids:
            runner_ids = get_runner_ids_for_instance(state, instance_id)
            total_runners += len(runner_ids)

        print(
            f"\nAll {len(all_instance_ids)} instance(s) with {total_runners} total runner(s) are ready!"
        )
        print(f"Instance IDs: {all_instance_ids}")

        if is_primary:
            # Run all stages once (requests will use available instances)
            # We use the first model_id for the benchmark requests
            benchmark_model_id = model_ids[0]
            print(f"\n{'=' * 80}")
            print(f"RUNNING BENCHMARK (using model: {benchmark_model_id})")
            print(f"Instances available: {len(all_instance_ids)}")
            print(f"{'=' * 80}")

            # Start metrics monitoring with 500ms interval to catch fast-completing tasks
            metrics_snapshots: list[MetricsSnapshot] = []
            stop_monitoring = asyncio.Event()
            monitoring_task = asyncio.create_task(
                monitor_metrics(
                    api_base, metrics_snapshots, stop_monitoring, interval_seconds=0.5
                )
            )

            stage_results: list[StageResult] = []
            for stage in stages:
                result = await run_stage(
                    api_base, benchmark_model_id, stage, no_overlap=no_overlap
                )
                stage_results.append(result)

            # Stop metrics monitoring
            print("\nStopping metrics monitoring...")
            stop_monitoring.set()
            await monitoring_task
            print(f"Collected {len(metrics_snapshots)} metrics snapshots")

            # Print final results
            print("\n" + "=" * 80)
            print("BENCHMARK COMPLETE - RESULTS SUMMARY")
            print("=" * 80)
            print(f"Instances tested: {len(all_instance_ids)}")
            print(f"Model IDs: {model_ids}")
            print(f"Instance IDs: {all_instance_ids}")

            for result in stage_results:
                print(f"\nStage: {result.name}")
                print(f"  Total Requests:     {result.total_requests}")
                print(f"  Successful:         {result.successful_requests}")
                print(f"  Failed:             {result.failed_requests}")
                print(f"  Success Rate:       {result.success_rate * 100:.1f}%")
                print(f"  Total Tokens:       {result.total_tokens}")
                print(f"  Avg Tokens/Request: {result.avg_tokens_per_request:.1f}")
                print(f"  Avg Time/Request:   {result.avg_time_per_request:.2f}s")
                if result.avg_time_to_first_token is not None:
                    if result.std_time_to_first_token is not None:
                        print(
                            f"  Avg TTFT:           {result.avg_time_to_first_token:.3f}s ± {result.std_time_to_first_token:.3f}s"
                        )
                    else:
                        print(
                            f"  Avg TTFT:           {result.avg_time_to_first_token:.3f}s"
                        )
                if result.avg_ms_per_token is not None:
                    if result.std_ms_per_token is not None:
                        print(
                            f"  Avg ms/token:       {result.avg_ms_per_token:.2f}ms ± {result.std_ms_per_token:.2f}ms"
                        )
                    else:
                        print(f"  Avg ms/token:       {result.avg_ms_per_token:.2f}ms")
                if result.avg_decode_tps is not None:
                    print(f"  Avg Decode TPS:     {result.avg_decode_tps:.2f} tokens/s")

            benchmark_completed_at = time.time()

            # Build comprehensive results document
            results_doc = {
                "metadata": {
                    "benchmark_started_at": benchmark_started_at,
                    "benchmark_completed_at": benchmark_completed_at,
                    "total_duration_s": benchmark_completed_at - benchmark_started_at,
                    "git_commit": git_commit,
                    "config_file": str(config_path),
                    "hardware_labels": hardware_labels or [],
                    "expected_nodes": expected_nodes,
                    "timeout_seconds": timeout_seconds,
                },
                "cluster": {
                    "model_ids": model_ids,
                    "instance_ids": all_instance_ids,
                    "instance_count": len(all_instance_ids),
                    "runner_count": total_runners,
                    "sharding": sharding,
                    "instance_meta": instance_meta,
                },
                "configuration": {
                    "stages": [
                        {
                            "name": stage.name,
                            "prompt_length": stage.prompt_length,
                            "generation_length": stage.generation_length,
                            "time_between_requests": stage.time_between_requests,
                            "iterations": stage.iterations,
                        }
                        for stage in stages
                    ]
                },
                "results": {
                    "stages": [
                        {
                            "name": r.name,
                            "total_requests": r.total_requests,
                            "successful_requests": r.successful_requests,
                            "failed_requests": r.failed_requests,
                            "success_rate": round(r.success_rate, 4),
                            "total_tokens": r.total_tokens,
                            "avg_tokens_per_request": round(
                                r.avg_tokens_per_request, 2
                            ),
                            "avg_time_per_request": round(r.avg_time_per_request, 3),
                            "avg_time_to_first_token": round(
                                r.avg_time_to_first_token, 3
                            )
                            if r.avg_time_to_first_token is not None
                            else None,
                            "std_time_to_first_token": round(
                                r.std_time_to_first_token, 3
                            )
                            if r.std_time_to_first_token is not None
                            else None,
                            "avg_decode_tps": round(r.avg_decode_tps, 2)
                            if r.avg_decode_tps is not None
                            else None,
                            "avg_ms_per_token": round(r.avg_ms_per_token, 2)
                            if r.avg_ms_per_token is not None
                            else None,
                            "std_ms_per_token": round(r.std_ms_per_token, 2)
                            if r.std_ms_per_token is not None
                            else None,
                            "stage_started_at": r.stage_started_at,
                            "stage_completed_at": r.stage_completed_at,
                            "stage_duration_s": r.stage_completed_at
                            - r.stage_started_at,
                            "requests": [
                                {
                                    "request_id": req.request_id,
                                    "success": req.success,
                                    "tokens": req.tokens,
                                    "elapsed_s": round(req.elapsed_s, 3),
                                    "started_at": req.started_at,
                                    "completed_at": req.completed_at,
                                    "time_to_first_token_s": round(
                                        req.time_to_first_token_s, 3
                                    )
                                    if req.time_to_first_token_s is not None
                                    else None,
                                    "decode_tps": round(req.decode_tps, 2)
                                    if req.decode_tps is not None
                                    else None,
                                    "error": req.error,
                                }
                                for req in r.request_results
                            ],
                        }
                        for r in stage_results
                    ]
                },
                "metrics": {
                    "snapshots": [
                        {
                            "timestamp": snapshot.timestamp,
                            "node_memory": {
                                node_id: {
                                    "ram_total_bytes": mem.ram_total_bytes,
                                    "ram_available_bytes": mem.ram_available_bytes,
                                    "ram_used_bytes": mem.ram_used_bytes,
                                    "swap_total_bytes": mem.swap_total_bytes,
                                    "swap_available_bytes": mem.swap_available_bytes,
                                    "swap_used_bytes": mem.swap_used_bytes,
                                }
                                for node_id, mem in snapshot.node_memory.items()
                            },
                            "instance_tasks": [
                                {
                                    "instance_id": inst.instance_id,
                                    "node_id": inst.node_id,
                                    "pending_tasks": inst.pending_tasks,
                                    "running_tasks": inst.running_tasks,
                                    "total_active_tasks": inst.total_active_tasks,
                                }
                                for inst in snapshot.instance_tasks
                            ],
                            "node_tasks": [
                                {
                                    "node_id": node.node_id,
                                    "pending_tasks": node.pending_tasks,
                                    "running_tasks": node.running_tasks,
                                    "total_active_tasks": node.total_active_tasks,
                                    "instance_count": node.instance_count,
                                }
                                for node in snapshot.node_tasks
                            ],
                        }
                        for snapshot in metrics_snapshots
                    ]
                },
            }

            # Output JSON summary
            print("\n" + "=" * 80)
            print("JSON RESULTS")
            print("=" * 80)
            print(json.dumps(results_doc, indent=2))
            print("=" * 80)

            # Save to file if path provided
            if results_output_path:
                print(f"Saving results to: {results_output_path}")
                with open(results_output_path, "w") as f:
                    json.dump(results_doc, f, indent=2)
                print("Results saved successfully")

            # Cleanup all instances
            for instance_id in all_instance_ids:
                print(f"[PRIMARY] Cleaning up instance: {instance_id}")
                await _http_request_async(
                    f"{api_base}/instance/{instance_id}", method="DELETE"
                )
                print(f"[PRIMARY] Instance {instance_id} deleted successfully")
        else:
            print(
                "[SECONDARY] Waiting with cluster (primary handles benchmark execution)"
            )
            # Secondary nodes wait until all instances of all models are deleted
            for model_id in model_counts:
                await wait_for_all_instances_deleted(api_base, model_id)

        return 0

    except TimeoutError as e:
        print("=" * 80)
        print(f"TIMEOUT ERROR: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print("=" * 80)
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 80)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run unified benchmark for EXO (single or multi-stage)"
    )
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--expected-nodes",
        type=int,
        required=True,
        help="Total number of nodes expected in the cluster",
    )
    parser.add_argument(
        "--is-primary", type=str, choices=["true", "false"], required=True
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--output", type=Path, help="Path to save detailed results JSON"
    )
    parser.add_argument("--git-commit", type=str, help="Git commit hash for metadata")
    parser.add_argument(
        "--hardware-labels", type=str, help="Comma-separated hardware labels"
    )
    args = parser.parse_args()

    api_base = f"http://localhost:{args.api_port}"
    is_primary = args.is_primary.lower() == "true"
    hardware_labels = args.hardware_labels.split(",") if args.hardware_labels else None

    return asyncio.run(
        run_benchmark(
            api_base,
            args.config,
            args.expected_nodes,
            is_primary,
            args.timeout_seconds,
            results_output_path=args.output,
            git_commit=args.git_commit,
            hardware_labels=hardware_labels,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
