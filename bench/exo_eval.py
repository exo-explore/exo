#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
exo-eval: Evaluation harness for exo inference system.

Supports multiple evaluation frameworks via TOML configuration:
- lm_eval: Language model evaluation using EleutherAI's lm-evaluation-harness
- swe_bench: SWE-bench evaluation (placeholder for future implementation)
- custom: Custom evaluation scripts

Usage:
    uv run python -m bench.exo_eval --config bench/eval_config.toml --model Llama-3.2-1b-Instruct-4bit
    uv run python -m bench.exo_eval --config bench/eval_config.toml --model Llama-3.2-1b-Instruct-4bit --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Add parent directory to path for direct script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tomlkit
from huggingface_hub import get_token as get_hf_token
from loguru import logger
from tomlkit.exceptions import TOMLKitError

from bench.completions_proxy import tasks_require_completions
from bench.exo_bench import (
    ExoClient,
    ExoHttpError,
    instance_id_from_instance,
    nodes_used_in_instance,
    placement_filter,
    resolve_model_short_id,
    sharding_filter,
    wait_for_instance_gone,
    wait_for_instance_ready,
)

EvalType = Literal["lm_eval", "swe_bench", "custom"]


def load_config(config_path: str) -> dict[str, Any]:
    """Load and parse TOML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        return dict(tomlkit.load(f))


def get_eval_type(config: dict[str, Any]) -> EvalType:
    """Extract evaluation type from config."""
    eval_section = config.get("eval", {})
    eval_type = eval_section.get("type", "lm_eval")
    if eval_type not in ("lm_eval", "swe_bench", "custom"):
        raise ValueError(f"Unknown eval type: {eval_type}")
    return eval_type


def check_hf_token(config: dict[str, Any]) -> bool:
    """Check if HuggingFace token is available when required.

    Returns True if token is available or not required, False otherwise.
    """
    eval_section = config.get("eval", {})
    require_hf_token = eval_section.get("require_hf_token", True)

    if not require_hf_token:
        return True

    token = get_hf_token()
    if token is None:
        logger.error(
            "HuggingFace token not found. "
            "Set HF_TOKEN environment variable or run 'huggingface-cli login'. "
            "To disable this check, set require_hf_token = false in [eval] config."
        )
        return False

    logger.info("HuggingFace token found")
    return True


def select_placement(
    client: ExoClient,
    full_model_id: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Select a placement based on config preferences."""
    instance_config = config.get("instance", {})

    # If explicit instance is provided, use it directly
    if "instance" in instance_config:
        return instance_config["instance"]

    # Otherwise, select from previews based on preferences
    instance_meta_pref = instance_config.get("instance_meta", "ring")
    sharding_pref = instance_config.get("sharding", "pipeline")
    max_nodes = instance_config.get("max_nodes", 4)
    min_nodes = instance_config.get("min_nodes", 1)

    previews_resp = client.request_json(
        "GET", "/instance/previews", params={"model_id": full_model_id}
    )
    previews = previews_resp.get("previews") or []

    selected: list[dict[str, Any]] = []
    for p in previews:
        if p.get("error") is not None:
            continue
        if not placement_filter(str(p.get("instance_meta", "")), instance_meta_pref):
            continue
        if not sharding_filter(str(p.get("sharding", "")), sharding_pref):
            continue

        instance = p.get("instance")
        if not isinstance(instance, dict):
            continue

        n = nodes_used_in_instance(instance)
        if min_nodes <= n <= max_nodes:
            selected.append(p)

    if not selected:
        return None

    # Sort by preference: exact match on sharding/meta, then by node count (descending)
    def sort_key(p: dict[str, Any]) -> tuple[int, int, int]:
        meta_match = (
            1 if instance_meta_pref in str(p.get("instance_meta", "")).lower() else 0
        )
        sharding_match = 1 if sharding_pref in str(p.get("sharding", "")).lower() else 0
        n_nodes = nodes_used_in_instance(p["instance"])
        return (meta_match, sharding_match, n_nodes)

    selected.sort(key=sort_key, reverse=True)
    return selected[0]


def setup_instance(
    client: ExoClient,
    full_model_id: str,
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    """Create and wait for an instance to be ready. Returns (instance_id, preview)."""
    preview = select_placement(client, full_model_id, config)

    if preview is None:
        logger.error("No valid placement found matching config preferences")
        return None, None

    instance_data = preview.get("instance")
    instance: dict[str, Any] = (
        instance_data if isinstance(instance_data, dict) else preview
    )
    instance_id = instance_id_from_instance(instance)

    sharding = str(preview.get("sharding", "unknown"))
    instance_meta = str(preview.get("instance_meta", "unknown"))
    n_nodes = nodes_used_in_instance(instance)

    logger.info(f"Selected placement: {sharding} / {instance_meta} / nodes={n_nodes}")
    logger.info(f"Instance ID: {instance_id}")

    if dry_run:
        logger.info("[dry-run] Would create instance and wait for ready")
        return instance_id, preview

    # Create instance
    client.request_json("POST", "/instance", body={"instance": instance})

    try:
        wait_for_instance_ready(client, instance_id)
        logger.info("Instance is ready")
        time.sleep(1)  # Brief pause after ready
        return instance_id, preview
    except (RuntimeError, TimeoutError) as e:
        logger.error(f"Failed to initialize instance: {e}")
        with contextlib.suppress(ExoHttpError):
            client.request_json("DELETE", f"/instance/{instance_id}")
        return None, None


def teardown_instance(client: ExoClient, instance_id: str) -> None:
    """Delete an instance and wait for it to be gone."""
    try:
        client.request_json("DELETE", f"/instance/{instance_id}")
    except ExoHttpError as e:
        if e.status != 404:
            raise
    wait_for_instance_gone(client, instance_id)
    logger.info(f"Instance {instance_id} deleted")


def build_lm_eval_args(
    config: dict[str, Any],
    base_url: str,
    model: str,
    output_path: str | None,
    limit: int | None,
    use_completions: bool,
) -> list[str]:
    """Build command-line arguments for lm_eval."""
    lm_eval_config = config.get("lm_eval", {})

    # Choose model type based on whether tasks need completions API
    if use_completions:
        model_type = "local-completions"
        endpoint_url = f"{base_url}/v1/completions"
    else:
        model_type = "local-chat-completions"
        endpoint_url = f"{base_url}/v1/chat/completions"

    # Build model_args string with num_concurrent if specified
    model_args_parts = [f"model={model}", f"base_url={endpoint_url}"]
    num_concurrent = lm_eval_config.get("num_concurrent")
    if num_concurrent is not None and num_concurrent > 1:
        model_args_parts.append(f"num_concurrent={num_concurrent}")
    model_args = ",".join(model_args_parts)

    args = [
        sys.executable, "-m", "bench.lm_eval_patched",
        "--model",
        model_type,
        "--model_args",
        model_args,
        "--verbosity",
        "WARNING",
    ]

    # Tasks
    tasks = lm_eval_config.get("tasks", ["mmlu"])
    tasks_str = ",".join(tasks) if isinstance(tasks, list) else str(tasks)
    args.extend(["--tasks", tasks_str])

    # Few-shot
    num_fewshot = lm_eval_config.get("num_fewshot")
    if num_fewshot is not None:
        args.extend(["--num_fewshot", str(num_fewshot)])

    # Batch size (default to 1 for API models, "auto" doesn't work)
    batch_size = lm_eval_config.get("batch_size", 1)
    args.extend(["--batch_size", str(batch_size)])

    # Apply chat template for instruct/chat models (default: true)
    # Only applies to chat completions, but doesn't hurt to include
    apply_chat_template = lm_eval_config.get("apply_chat_template", True)
    if apply_chat_template and not use_completions:
        args.append("--apply_chat_template")

    # Fewshot as multiturn (optional, works with chat template)
    fewshot_as_multiturn = lm_eval_config.get("fewshot_as_multiturn", False)
    if fewshot_as_multiturn and not use_completions:
        args.append("--fewshot_as_multiturn")

    # Limit (command line overrides config)
    effective_limit = limit if limit is not None else lm_eval_config.get("limit")
    if effective_limit is not None:
        args.extend(["--limit", str(effective_limit)])

    # Output path
    effective_output = output_path or lm_eval_config.get("output_path")
    if effective_output:
        args.extend(["--output_path", effective_output])
        # Log model responses for post-hoc analysis when output is saved
        args.append("--log_samples")

    return args


def run_lm_eval(
    config: dict[str, Any],
    host: str,
    port: int,
    model: str,
    output_path: str | None,
    limit: int | None,
    dry_run: bool,
) -> int:
    """Run lm_eval evaluation."""
    lm_eval_config = config.get("lm_eval", {})
    tasks = lm_eval_config.get("tasks", ["mmlu"])
    if isinstance(tasks, str):
        tasks = [tasks]

    # Check if tasks require the completions API
    use_completions = tasks_require_completions(tasks)

    if use_completions:
        logger.info(
            "Tasks require completions API - using native /v1/completions endpoint"
        )

    exo_base_url = f"http://{host}:{port}"

    # Build args - use native completions or chat completions endpoint directly
    args = build_lm_eval_args(
        config, exo_base_url, model, output_path, limit, use_completions=use_completions
    )
    logger.info(f"lm_eval command: {' '.join(args)}")

    if dry_run:
        logger.info("[dry-run] Would execute the above command")
        return 0

    try:
        result = subprocess.run(args, check=False)

        # Print token usage summary from exo
        try:
            import httpx
            usage_resp = httpx.get(f"{exo_base_url}/v1/usage", timeout=5)
            if usage_resp.status_code == 200:
                usage = usage_resp.json()
                logger.info("--- Token Usage (Total) ---")
                logger.info(f"  Requests:          {usage.get('total_requests', 0)}")
                logger.info(f"  Prompt tokens:     {usage.get('total_prompt_tokens', 0)}")
                logger.info(f"  Completion tokens: {usage.get('total_completion_tokens', 0)}")
                logger.info(f"  Reasoning tokens:  {usage.get('total_reasoning_tokens', 0)}")
                logger.info(f"  Total tokens:      {usage.get('total_tokens', 0)}")
                by_model = usage.get("by_model", {})
                if by_model:
                    for model_name, counters in by_model.items():
                        logger.info(f"--- Token Usage ({model_name}) ---")
                        logger.info(f"  Requests:          {counters.get('requests', 0)}")
                        logger.info(f"  Prompt tokens:     {counters.get('prompt_tokens', 0)}")
                        logger.info(f"  Completion tokens: {counters.get('completion_tokens', 0)}")
                        logger.info(f"  Reasoning tokens:  {counters.get('reasoning_tokens', 0)}")
        except Exception:
            pass  # Usage endpoint not available

        return result.returncode
    except FileNotFoundError:
        logger.error("lm_eval not found. Install with: uv sync --extra eval")
        return 1


def run_swe_bench(
    config: dict[str, Any],
    host: str,
    port: int,
    model: str,
    output_path: str | None,
    dry_run: bool,
) -> int:
    """Run SWE-bench evaluation (placeholder)."""
    swe_config = config.get("swe_bench", {})

    dataset = swe_config.get("dataset", "princeton-nlp/SWE-bench_Lite")
    max_workers = swe_config.get("max_workers", 8)
    predictions_path = output_path or swe_config.get(
        "predictions_path", "bench/predictions"
    )

    logger.info("SWE-bench evaluation configuration:")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Model: {model}")
    logger.info(f"  API endpoint: http://{host}:{port}/v1")
    logger.info(f"  Max workers: {max_workers}")
    logger.info(f"  Predictions path: {predictions_path}")

    if dry_run:
        logger.info("[dry-run] SWE-bench evaluation would be executed")
        return 0

    logger.warning(
        "SWE-bench integration is a placeholder. "
        "Implement swebench inference and evaluation logic as needed."
    )
    return 0


def run_custom_eval(
    config: dict[str, Any],
    host: str,
    port: int,
    model: str,
    output_path: str | None,
    dry_run: bool,
) -> int:
    """Run custom evaluation script."""
    custom_config = config.get("custom", {})

    script = custom_config.get("script")
    if not script:
        logger.error("No script specified in [custom] config section")
        return 1

    script_path = Path(script)
    if not script_path.exists():
        logger.error(f"Custom script not found: {script}")
        return 1

    script_args = custom_config.get("args", [])
    if not isinstance(script_args, list):
        script_args = [str(script_args)]

    # Build environment with exo connection info
    env = os.environ.copy()
    env["EXO_HOST"] = host
    env["EXO_PORT"] = str(port)
    env["EXO_MODEL"] = model
    if output_path:
        env["EXO_OUTPUT_PATH"] = output_path

    cmd = [sys.executable, str(script_path), *script_args]
    logger.info(f"Custom eval command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[dry-run] Would execute the above command")
        return 0

    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


def write_results_metadata(
    output_path: str,
    config: dict[str, Any],
    host: str,
    port: int,
    model: str,
    eval_type: EvalType,
    return_code: int,
    preview: dict[str, Any] | None,
) -> None:
    """Write evaluation metadata to a JSON file."""
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_type": eval_type,
        "model": model,
        "api_endpoint": f"http://{host}:{port}/v1",
        "config": config,
        "return_code": return_code,
    }

    if preview:
        metadata["placement"] = {
            "sharding": preview.get("sharding"),
            "instance_meta": preview.get("instance_meta"),
            "instance_id": instance_id_from_instance(preview["instance"])
            if "instance" in preview
            else None,
        }

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "eval_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Wrote evaluation metadata to: {metadata_path}")


def main() -> int:
    """Main entry point for exo-eval."""
    ap = argparse.ArgumentParser(
        prog="exo-eval",
        description="Evaluation harness for exo inference system.",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to TOML configuration file",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("EXO_HOST", "localhost"),
        help="exo API host (default: localhost or EXO_HOST env var)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EXO_PORT", "52415")),
        help="exo API port (default: 52415 or EXO_PORT env var)",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Model name/ID to evaluate",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output path for results (overrides config)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per task (overrides config, lm_eval only)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds (default: 600)",
    )
    ap.add_argument(
        "--skip-instance-setup",
        action="store_true",
        help="Skip instance creation (assume instance already running)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = ap.parse_args()

    logger.info(f"exo-eval starting with config: {args.config}")

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except TOMLKitError as e:
        logger.error(f"Failed to parse config: {e}")
        return 1

    eval_type = get_eval_type(config)
    logger.info(f"Evaluation type: {eval_type}")
    logger.info(f"Model: {args.model}")
    logger.info(f"API endpoint: http://{args.host}:{args.port}/v1")

    # Check HuggingFace token if required
    if not check_hf_token(config):
        return 1

    # Setup instance and resolve model
    instance_id: str | None = None
    preview: dict[str, Any] | None = None
    client: ExoClient | None = None

    if args.skip_instance_setup:
        # Use model name as-is when skipping instance setup
        full_model_id = args.model
        logger.info(f"Using model: {full_model_id} (instance setup skipped)")
    else:
        client = ExoClient(args.host, args.port, timeout_s=args.timeout)

        # Resolve model
        try:
            short_id, full_model_id = resolve_model_short_id(client, args.model)
            logger.info(f"Resolved model: {short_id} -> {full_model_id}")
        except Exception as e:
            logger.error(f"Failed to resolve model: {e}")
            return 1

        instance_id, preview = setup_instance(
            client, full_model_id, config, args.dry_run
        )
        if instance_id is None and not args.dry_run:
            return 1

    try:
        # Run evaluation
        if eval_type == "lm_eval":
            return_code = run_lm_eval(
                config,
                args.host,
                args.port,
                full_model_id,
                args.output,
                args.limit,
                args.dry_run,
            )
        elif eval_type == "swe_bench":
            return_code = run_swe_bench(
                config,
                args.host,
                args.port,
                full_model_id,
                args.output,
                args.dry_run,
            )
        elif eval_type == "custom":
            return_code = run_custom_eval(
                config,
                args.host,
                args.port,
                full_model_id,
                args.output,
                args.dry_run,
            )
        else:
            logger.error(f"Unknown eval type: {eval_type}")
            return 1

        # Write metadata if output path specified and not dry-run
        output_path = args.output or config.get(eval_type, {}).get("output_path")
        if output_path and not args.dry_run:
            write_results_metadata(
                output_path,
                config,
                args.host,
                args.port,
                full_model_id,
                eval_type,
                return_code,
                preview,
            )

        return return_code

    finally:
        # Teardown instance
        if instance_id and client and not args.skip_instance_setup and not args.dry_run:
            teardown_instance(client, instance_id)


if __name__ == "__main__":
    raise SystemExit(main())
