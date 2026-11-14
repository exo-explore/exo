import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .client import (
    Endpoint,
    Object,
    Trial,
    decode_object,
    disconnect_after_output,
    measure,
    object_value,
)

PROMPT = (
    "Write a concise explanation of why an engineering benchmark should use the same "
    "model, quantization, prompt, output limit and concurrency for every system."
)


def native_deployments(endpoint: Endpoint, model: str) -> int:
    state = endpoint.get("/state")
    matches: list[str] = []
    for tagged in object_value(state.get("instances", {})).values():
        for name, value in object_value(tagged).items():
            instance = object_value(value)
            assignments = object_value(
                instance.get("shard_assignments", instance.get("shardAssignments", {}))
            )
            if assignments.get("model_id", assignments.get("modelId")) == model:
                matches.append(name)
    if not matches or set(matches) != {"TensorFoldInstance"}:
        raise ValueError(
            "The requested model must have only TensorFold instances in EXO /state"
        )
    return len(matches)


def binary_metadata(binary: Path | None) -> Object | None:
    if binary is None:
        return None
    result = subprocess.run(
        [str(binary.resolve()), "capabilities", "--json"],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    capabilities = decode_object(result.stdout)
    return {
        key: capabilities.get(key)
        for key in ("engine", "version", "backends", "token_stream_versions")
    }


def summary(trials: list[Trial]) -> Object:
    first = sorted(trial.first_output_milliseconds for trial in trials)
    rates = [trial.completion_tokens_per_second for trial in trials]
    return {
        "trials": len(trials),
        "first_output_milliseconds_p50": statistics.median(first),
        "first_output_milliseconds_p95": first[
            max(0, math.ceil(0.95 * len(first)) - 1)
        ],
        "completion_tokens_per_second_p50": statistics.median(rates),
        "total_completion_tokens": sum(trial.completion_tokens for trial in trials),
    }


def run_series(
    endpoint: Endpoint, model: str, repeats: int, concurrency: int, max_tokens: int
) -> Object:
    warmup = measure(endpoint, model, PROMPT, max_tokens)
    measured: list[Trial] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(repeats):
            futures = [
                pool.submit(measure, endpoint, model, PROMPT, max_tokens)
                for _ in range(concurrency)
            ]
            measured.extend(future.result() for future in futures)
    elapsed = time.monotonic() - started
    return {
        "warmup_excluded": warmup.json(),
        "concurrency": concurrency,
        "rounds": repeats,
        "elapsed_seconds": elapsed,
        "aggregate_completion_tokens_per_second": sum(
            trial.completion_tokens for trial in measured
        )
        / elapsed,
        "summary": summary(measured),
        "measurements": [trial.json() for trial in measured],
    }


def qualify(
    endpoint: Endpoint,
    model: str,
    revision: str,
    hardware: str,
    repeats: int,
    concurrency: int,
    max_tokens: int,
    binary: Path | None,
    baseline: Endpoint | None,
    baseline_model: str | None,
) -> Object:
    deployments = native_deployments(endpoint, model)
    report: Object = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "model_revision_declared": revision,
        "hardware_declared": hardware,
        "binary_report_provided": binary_metadata(binary),
        "native_instances_observed": deployments,
        "scope": "Live EXO API, TensorFold placement, response contracts, concurrent requests and HTTP timing. This script does not validate numerical accuracy or GPU kernels.",
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metrics": "First output includes transport and prompt processing. Completion rate includes the entire request; it is not kernel decode throughput. Output is hashed, not retained.",
    }
    report["single_request"] = run_series(endpoint, model, repeats, 1, max_tokens)
    report["concurrent_requests"] = run_series(
        endpoint, model, repeats, concurrency, max_tokens
    )
    disconnect_after_output(
        endpoint,
        model,
        "List the integers from one to five hundred, separated by spaces.",
    )
    report["after_client_disconnect"] = measure(
        endpoint, model, PROMPT, max_tokens
    ).json()
    report["disconnect_scope"] = (
        "Client closed after first output; the next request completed. Confirm native request release with native health or process monitoring."
    )
    if baseline is not None:
        report["direct_native_baseline"] = run_series(
            baseline, baseline_model or model, repeats, 1, max_tokens
        )
        report["baseline_conditions"] = (
            "Operator must ensure this endpoint uses the same pinned weights, native build, context, cache state and settings. No speedup claim is calculated automatically."
        )
    report["status"] = "passed"
    return report
