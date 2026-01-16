#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""
exo-eval: Run SWE-bench evaluation against exo using OpenHands SDK (local, no Docker).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


class EvalStatus(str, Enum):
    Resolved = "Resolved"
    Failed = "Failed"
    Error = "Error"
    Timeout = "Timeout"


@dataclass
class EvalResult:
    instance_id: str
    repo: str
    status: EvalStatus
    elapsed_seconds: float
    tests_passed: list[str]
    tests_failed: list[str]
    error_message: str | None = None


def load_swe_bench(
    split: str = "lite",
    limit: int | None = None,
    instance_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-bench dataset from HuggingFace."""
    # SWE-bench Lite is a curated 300-instance subset
    dataset_name = (
        "princeton-nlp/SWE-bench_Lite" if split == "lite" else "princeton-nlp/SWE-bench"
    )
    actual_split = "test" if split == "lite" else split

    ds = load_dataset(dataset_name, split=actual_split)
    instances = [dict(row) for row in ds]

    if instance_ids:
        instances = [i for i in instances if i["instance_id"] in instance_ids]

    if limit:
        instances = instances[:limit]

    return instances


def clone_repo_at_commit(repo: str, commit: str, dest: Path) -> None:
    """Clone a repo at a specific commit."""
    repo_url = f"https://github.com/{repo}.git"

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", commit],
        cwd=dest,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "checkout", commit],
        cwd=dest,
        check=True,
        capture_output=True,
    )


def build_agent_prompt(instance: dict[str, Any]) -> str:
    """Build the prompt for the agent."""
    return f"""You are a software engineer fixing a bug in the {instance['repo']} repository.

## Problem Statement
{instance['problem_statement']}

## Instructions
1. Explore the codebase to understand the issue
2. Identify the files that need to be modified
3. Make the necessary changes to fix the issue
4. The fix should be minimal and targeted

You have access to:
- terminal: Run shell commands (git, grep, python, etc.)
- file_editor: View and edit files

Start by exploring the repository structure to understand where the relevant code is.
"""


def parse_fail_to_pass(fail_to_pass_str: str) -> list[str]:
    """Parse the FAIL_TO_PASS field into a list of test names."""
    try:
        return json.loads(fail_to_pass_str)
    except json.JSONDecodeError:
        return [t.strip() for t in fail_to_pass_str.split(",") if t.strip()]


def run_tests(workspace: Path, tests: list[str]) -> tuple[list[str], list[str]]:
    """Run tests and return (passed, failed) lists."""
    passed = []
    failed = []

    for test in tests:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-xvs", test],
                cwd=workspace,
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0:
                passed.append(test)
            else:
                failed.append(test)
        except subprocess.TimeoutExpired:
            failed.append(test)

    return passed, failed


def run_single_eval(
    instance: dict[str, Any],
    host: str,
    port: int,
    model: str,
    max_turns: int = 30,
    timeout: float = 600.0,
) -> EvalResult:
    """Evaluate a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    fail_to_pass = parse_fail_to_pass(instance["FAIL_TO_PASS"])

    start_time = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "repo"

            # Clone repo at base commit
            logger.info(f"Cloning {repo} at {base_commit[:8]}...")
            clone_repo_at_commit(repo, base_commit, workspace)

            # Setup OpenHands agent
            llm = LLM(
                model=f"openai/{model}",
                base_url=f"http://{host}:{port}/v1",
                api_key="not-needed",
            )

            agent = Agent(
                llm=llm,
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                ],
            )

            # Run agent
            conversation = Conversation(
                agent=agent,
                workspace=str(workspace),
            )

            logger.info(f"Running agent on {instance_id}...")
            conversation.send_message(build_agent_prompt(instance))

            for _turn in range(max_turns):
                if time.perf_counter() - start_time > timeout:
                    return EvalResult(
                        instance_id=instance_id,
                        repo=repo,
                        status=EvalStatus.Timeout,
                        elapsed_seconds=time.perf_counter() - start_time,
                        tests_passed=[],
                        tests_failed=fail_to_pass,
                    )

                result = conversation.run(max_turns=1)
                if result.done:
                    break

            # Run tests to verify
            logger.info(f"Running tests for {instance_id}...")
            passed, failed = run_tests(workspace, fail_to_pass)

            elapsed = time.perf_counter() - start_time
            status = EvalStatus.Resolved if not failed else EvalStatus.Failed

            return EvalResult(
                instance_id=instance_id,
                repo=repo,
                status=status,
                elapsed_seconds=elapsed,
                tests_passed=passed,
                tests_failed=failed,
            )

    except Exception as e:
        return EvalResult(
            instance_id=instance_id,
            repo=repo,
            status=EvalStatus.Error,
            elapsed_seconds=time.perf_counter() - start_time,
            tests_passed=[],
            tests_failed=[],
            error_message=str(e),
        )


def verify_exo_running(host: str, port: int, model: str) -> str:
    """Verify exo is running and return full model ID."""
    import http.client

    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("GET", "/models")
    resp = conn.getresponse()

    if resp.status != 200:
        raise RuntimeError(f"exo not responding at {host}:{port}")

    data = json.loads(resp.read())
    for m in data.get("data", []):
        if m.get("id") == model or m.get("hugging_face_id") == model:
            return m.get("hugging_face_id") or m.get("id")

    raise ValueError(f"Model '{model}' not found in exo")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="exo-eval",
        description="Run SWE-bench evaluation against exo (local, no Docker).",
    )

    ap.add_argument("--host", default=os.environ.get("EXO_HOST", "localhost"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("EXO_PORT", "52415"))
    )
    ap.add_argument("--model", required=True, help="exo model ID")
    ap.add_argument(
        "--split", default="lite", choices=["lite", "dev", "test", "train"]
    )
    ap.add_argument("--limit", type=int, default=10, help="Max instances")
    ap.add_argument("--instance-ids", nargs="+", help="Specific instance IDs")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json-out", default="bench/eval_results.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    # Load dataset first (doesn't require exo to be running)
    logger.info(f"Loading SWE-bench {args.split} dataset...")
    instances = load_swe_bench(
        split=args.split,
        limit=args.limit,
        instance_ids=args.instance_ids,
    )
    logger.info(f"Loaded {len(instances)} instances")

    if args.dry_run:
        print(f"\nSWE-bench {args.split} instances ({len(instances)}):")
        for inst in instances:
            print(f"  {inst['instance_id']} ({inst['repo']})")
        return 0

    # Verify exo is running
    model_id = verify_exo_running(args.host, args.port, args.model)
    logger.info(f"Using model: {model_id}")

    # Run evaluation
    results: list[EvalResult] = []
    for i, instance in enumerate(instances):
        logger.info(f"[{i+1}/{len(instances)}] {instance['instance_id']}")

        result = run_single_eval(
            instance=instance,
            host=args.host,
            port=args.port,
            model=model_id,
            max_turns=args.max_turns,
            timeout=args.timeout,
        )
        results.append(result)

        logger.info(f"  Status: {result.status.value}")
        if result.tests_passed:
            logger.info(f"  Passed: {len(result.tests_passed)} tests")
        if result.tests_failed:
            logger.info(f"  Failed: {len(result.tests_failed)} tests")
        if result.error_message:
            logger.error(f"  Error: {result.error_message}")

    # Compute summary
    total = len(results)
    resolved = sum(1 for r in results if r.status == EvalStatus.Resolved)
    failed = sum(1 for r in results if r.status == EvalStatus.Failed)
    errors = sum(1 for r in results if r.status == EvalStatus.Error)
    timeouts = sum(1 for r in results if r.status == EvalStatus.Timeout)

    summary = {
        "model": model_id,
        "split": args.split,
        "total": total,
        "resolved": resolved,
        "resolved_rate": resolved / total if total else 0,
        "failed": failed,
        "errors": errors,
        "timeouts": timeouts,
    }

    output = {
        "summary": summary,
        "results": [
            {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "status": r.status.value,
                "elapsed_seconds": r.elapsed_seconds,
                "tests_passed": r.tests_passed,
                "tests_failed": r.tests_failed,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }

    Path(args.json_out).write_text(json.dumps(output, indent=2))
    logger.info(f"Results written to {args.json_out}")

    # Print summary
    print("\n" + "=" * 60)
    print("SWE-bench Evaluation Results")
    print("=" * 60)
    print(f"Model:    {model_id}")
    print(f"Split:    {args.split}")
    print(f"Total:    {total}")
    if total:
        print(f"Resolved: {resolved} ({resolved/total*100:.1f}%)")
    else:
        print("Resolved: 0")
    print(f"Failed:   {failed}")
    print(f"Errors:   {errors}")
    print(f"Timeouts: {timeouts}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
