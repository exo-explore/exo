# type: ignore
#!/usr/bin/env python3
"""Quality evaluation for exo using lm-evaluation-harness.

Runs standard LLM benchmarks against exo's OpenAI-compatible API and
optionally compares results across concurrency levels to verify batching
doesn't degrade quality.

NOTE: Only `generate_until` tasks work with chat completions APIs.
`loglikelihood` tasks (MMLU, HellaSwag, ARC) will NOT work.

Only generate_until tasks work with chat completions APIs.

Working tasks:
  gsm8k                        - Grade school math (free-form)
  hendrycks_math_chat          - MATH benchmark, all subjects (custom, uses math_verify)
  hendrycks_math_chat_algebra  - MATH algebra only
  drop                         - Reading comprehension (extractive QA)
  ifeval                       - Instruction following
  truthfulqa_gen               - Truthfulness (generative)
  triviaqa                     - Open-domain trivia QA
  gpqa_diamond_cot_zeroshot    - Graduate-level science QA (chain-of-thought)
  leaderboard_bbh              - BIG-Bench Hard (23 subtasks, CoT)
  humaneval                    - Python code generation
  mbpp                         - Basic Python programming
  aime24                       - Math olympiad (AMC/AIME 2024)
  nq_open                      - Natural Questions (open-domain)
  coqa                         - Conversational QA

NOT working (loglikelihood): mmlu, hellaswag, arc_challenge, winogrande, gpqa_diamond_zeroshot
NOT working (broken extraction): hendrycks_math (use hendrycks_math_chat instead)

Start exo first, then run:
    uv run python exo_eval.py --model <model-id> --tasks gsm8k
    uv run python exo_eval.py --model <model-id> --tasks gsm8k --limit 50 --compare-concurrency 1,4
    uv run python exo_eval.py --model <model-id> --tasks hendrycks_math_chat --limit 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness import (
    ExoClient,
    ExoHttpError,
    add_common_instance_args,
    instance_id_from_instance,
    nodes_used_in_instance,
    resolve_model_short_id,
    run_planning_phase,
    settle_and_fetch_placements,
    wait_for_instance_gone,
    wait_for_instance_ready,
)
from loguru import logger


_AVAILABLE_TASKS = [
    ("gsm8k",                       "Grade school math (free-form)"),
    ("hendrycks_math_chat",         "MATH benchmark, all 7 subjects"),
    ("hendrycks_math_chat_algebra", "MATH — algebra only"),
    ("drop",                        "Reading comprehension (extractive QA)"),
    ("ifeval",                      "Instruction following evaluation"),
    ("truthfulqa_gen",              "Truthfulness (generative)"),
    ("triviaqa",                    "Open-domain trivia QA"),
    ("gpqa_diamond_cot_zeroshot",   "Graduate-level science QA (CoT)"),
    ("leaderboard_bbh",            "BIG-Bench Hard (23 subtasks, CoT)"),
    ("humaneval",                   "Python code generation"),
    ("mbpp",                        "Basic Python programming"),
    ("aime24",                      "Math olympiad (AIME 2024)"),
    ("nq_open",                     "Natural Questions (open-domain)"),
    ("coqa",                        "Conversational QA"),
]


def pick_tasks_interactive() -> str:
    """Show a simple TUI for selecting eval tasks."""
    import termios
    import tty

    if not sys.stdin.isatty():
        logger.error("No --tasks specified and stdin is not a terminal. Use --tasks to specify tasks.")
        return ""

    selected: list[bool] = [False] * len(_AVAILABLE_TASKS)
    cursor = 0
    total_lines = len(_AVAILABLE_TASKS) + 4  # header + blank + tasks + blank + status

    def write(s: str) -> None:
        sys.stdout.write(s)

    def render(first: bool = False) -> None:
        if not first:
            # Move cursor to top of our output area
            write(f"\033[{total_lines}A")
        # Clear from cursor to end of screen
        write("\033[J")
        write("\033[1mSelect tasks\033[0m (up/down move, space toggle, enter confirm, q quit)\r\n\r\n")
        for i, (name, desc) in enumerate(_AVAILABLE_TASKS):
            marker = ">" if i == cursor else " "
            check = "x" if selected[i] else " "
            if i == cursor:
                write(f"\033[7m  {marker} [{check}] {name:<35} {desc}\033[0m\r\n")
            else:
                write(f"  {marker} [{check}] {name:<35} {desc}\r\n")
        n = sum(selected)
        write(f"\r\n  {n} task{'s' if n != 1 else ''} selected\r\n")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Hide cursor
        write("\033[?25l")
        render(first=True)
        while True:
            ch = sys.stdin.read(1)
            if ch == "q" or ch == "\x03":  # q or Ctrl-C
                write("\033[?25h\033[0m\r\n")
                return ""
            elif ch == "\r" or ch == "\n":
                break
            elif ch == " ":
                selected[cursor] = not selected[cursor]
            elif ch == "\x1b":  # escape sequence
                seq = sys.stdin.read(2)
                if seq == "[A":  # up
                    cursor = (cursor - 1) % len(_AVAILABLE_TASKS)
                elif seq == "[B":  # down
                    cursor = (cursor + 1) % len(_AVAILABLE_TASKS)
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Show cursor
        write("\033[?25h\033[0m\r\n")
        sys.stdout.flush()

    chosen = [name for (name, _), sel in zip(_AVAILABLE_TASKS, selected) if sel]
    if not chosen:
        logger.warning("No tasks selected.")
        return ""
    tasks = ",".join(chosen)
    logger.info(f"Selected tasks: {tasks}")
    return tasks


def fetch_context_length(model_id: str) -> int:
    """Fetch max context length from the model's HuggingFace config.json."""
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        for key in ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"):
            if key in config:
                return int(config[key])
    except Exception as e:
        logger.warning(f"Could not fetch context length from HuggingFace: {e}")
    return 4096


def parse_int_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                items.append(int(part))
    return items


def build_lm_eval_cmd(
    *,
    model_id: str,
    base_url: str,
    tasks: str,
    num_concurrent: int,
    timeout: int,
    max_gen_toks: int,
    num_fewshot: int | None,
    limit: int | None,
    output_path: str,
    tasks_dir: str,
    extra_args: list[str],
) -> list[str]:
    model_args = (
        f"model={model_id},"
        f"base_url={base_url},"
        f"num_concurrent={num_concurrent},"
        f"tokenized_requests=False,"
        f"max_gen_toks={max_gen_toks},"
        f"timeout={timeout}"
    )

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--tasks",
        tasks,
        "--model_args",
        model_args,
        "--apply_chat_template",
        "--log_samples",
        # Override task-level stop tokens — many tasks use problematic ones like '.'
        # that break chat completions. <|im_end|> is the correct stop for chat models.
        "--gen_kwargs",
        "until=<|im_end|>",
        # Include custom task configs (e.g. hendrycks_math_chat with math_verify)
        "--include_path",
        tasks_dir,
        "--output_path",
        output_path,
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    cmd.extend(extra_args)
    return cmd


def run_lm_eval(cmd: list[str]) -> int:
    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def load_results(results_dir: str) -> dict[str, Any] | None:
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    # lm-eval writes results to {output_path}/{model_name}/{timestamp}/results.json
    # Find the most recent results.json
    results_files = sorted(results_path.rglob("results.json"), key=lambda p: p.stat().st_mtime)
    if not results_files:
        return None

    with open(results_files[-1]) as f:
        return json.load(f)


def extract_scores(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract task -> {metric: score} from lm-eval results JSON."""
    scores: dict[str, dict[str, float]] = {}
    for task_name, task_data in results.get("results", {}).items():
        task_scores: dict[str, float] = {}
        for key, value in task_data.items():
            if key.endswith(",none") or key.endswith(",flexible-extract") or key.endswith(",strict-match"):
                # Strip the filter suffix for readability
                metric = key.rsplit(",", 1)[0]
                if isinstance(value, (int, float)):
                    task_scores[metric] = float(value)
        if task_scores:
            scores[task_name] = task_scores
    return scores


def print_comparison(
    results_by_concurrency: dict[int, dict[str, dict[str, float]]],
) -> None:
    """Print a comparison table across concurrency levels."""
    concurrency_levels = sorted(results_by_concurrency.keys())

    # Collect all tasks and metrics
    all_tasks: set[str] = set()
    for scores in results_by_concurrency.values():
        all_tasks.update(scores.keys())

    print("\n" + "=" * 80)
    print("QUALITY COMPARISON ACROSS CONCURRENCY LEVELS")
    print("=" * 80)

    header = f"{'Task':<30} {'Metric':<20}"
    for c in concurrency_levels:
        header += f" {'c=' + str(c):>10}"
    if len(concurrency_levels) > 1:
        header += f" {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for task in sorted(all_tasks):
        # Collect metrics from all concurrency levels for this task
        all_metrics: set[str] = set()
        for scores in results_by_concurrency.values():
            if task in scores:
                all_metrics.update(scores[task].keys())

        for metric in sorted(all_metrics):
            row = f"{task:<30} {metric:<20}"
            values: list[float] = []
            for c in concurrency_levels:
                val = results_by_concurrency.get(c, {}).get(task, {}).get(metric)
                if val is not None:
                    row += f" {val:>10.4f}"
                    values.append(val)
                else:
                    row += f" {'N/A':>10}"

            if len(values) >= 2:
                delta = values[-1] - values[0]
                marker = " " if abs(delta) < 0.001 else (" !!!" if abs(delta) > 0.05 else " *")
                row += f" {delta:>+10.4f}{marker}"

            print(row)

    print()
    if len(concurrency_levels) > 1:
        print("Legend: !!! = >5% delta (significant), * = >0.1% delta (minor)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="exo-eval",
        description="Run quality evaluations (MMLU, GSM8K, etc.) against exo.",
    )
    add_common_instance_args(ap)
    ap.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated lm-eval task names. "
            "If omitted, shows an interactive picker. "
            "Only generate_until tasks work with chat APIs."
        ),
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples. If not set, uses lm-eval task default.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples per task (for faster iteration).",
    )
    ap.add_argument(
        "--num-concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests to lm-eval API (default: 1).",
    )
    ap.add_argument(
        "--compare-concurrency",
        nargs="+",
        default=None,
        help="Run eval at multiple concurrency levels and compare quality. E.g. --compare-concurrency 1,4",
    )
    ap.add_argument(
        "--eval-timeout",
        type=int,
        default=7200,
        help="Per-request timeout in seconds for lm-eval API calls (default: 7200).",
    )
    ap.add_argument(
        "--results-dir",
        default="eval_results",
        help="Directory for lm-eval output (default: bench/eval_results).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lm-eval command without running it.",
    )
    ap.add_argument(
        "--skip-instance-setup",
        action="store_true",
        help="Skip instance management (assume model is already running).",
    )

    args, extra_args = ap.parse_known_args()
    extra_args = [a for a in extra_args if a != "--"]

    if args.tasks is None:
        args.tasks = pick_tasks_interactive()
        if not args.tasks:
            return 0

    # Block tasks that don't work with chat completions APIs
    _BROKEN_EXTRACTION = {
        "hendrycks_math": "hendrycks_math_chat",
        "hendrycks_math_algebra": "hendrycks_math_chat_algebra",
        "hendrycks_math_counting_and_prob": "hendrycks_math_chat_counting_and_prob",
        "hendrycks_math_geometry": "hendrycks_math_chat_geometry",
        "hendrycks_math_intermediate_algebra": "hendrycks_math_chat_intermediate_algebra",
        "hendrycks_math_num_theory": "hendrycks_math_chat_num_theory",
        "hendrycks_math_prealgebra": "hendrycks_math_chat_prealgebra",
        "hendrycks_math_precalc": "hendrycks_math_chat_precalc",
    }
    _LOGLIKELIHOOD_TASKS = {
        "mmlu", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
        "piqa", "boolq", "openbookqa", "sciq", "lambada_openai",
        "gpqa_diamond_zeroshot", "gpqa_diamond_n_shot",
        "gpqa_main_zeroshot", "gpqa_main_n_shot",
        "gpqa_extended_zeroshot", "gpqa_extended_n_shot",
    }
    for task in args.tasks.split(","):
        task = task.strip()
        if task in _BROKEN_EXTRACTION:
            logger.error(
                f"Task '{task}' has broken answer extraction for chat models. "
                f"Use '{_BROKEN_EXTRACTION[task]}' instead (same benchmark, uses math_verify)."
            )
            return 1
        if task in _LOGLIKELIHOOD_TASKS:
            logger.error(
                f"Task '{task}' uses loglikelihood scoring which doesn't work with chat completions APIs. "
                f"Use generate_until tasks instead: gsm8k, hendrycks_math_chat, drop, ifeval, "
                f"truthfulqa_gen, triviaqa, gpqa_diamond_cot_zeroshot, leaderboard_bbh, humaneval, mbpp"
            )
            return 1

    client = ExoClient(args.host, args.port, timeout_s=args.timeout)

    instance_id: str | None = None

    if not args.skip_instance_setup:
        short_id, full_model_id = resolve_model_short_id(
            client, args.model, force_download=args.force_download
        )

        selected = settle_and_fetch_placements(
            client, full_model_id, args, settle_timeout=args.settle_timeout
        )

        if not selected:
            logger.error("No valid placements matched your filters.")
            return 1

        selected.sort(
            key=lambda p: (
                str(p.get("instance_meta", "")),
                str(p.get("sharding", "")),
                -nodes_used_in_instance(p["instance"]),
            ),
            reverse=True,
        )

        preview = selected[0]
        instance = preview["instance"]
        instance_id = instance_id_from_instance(instance)
        sharding = str(preview["sharding"])
        instance_meta = str(preview["instance_meta"])
        n_nodes = nodes_used_in_instance(instance)

        logger.info(
            f"PLACEMENT: {sharding} / {instance_meta} / nodes={n_nodes}"
        )

        settle_deadline = (
            time.monotonic() + args.settle_timeout if args.settle_timeout > 0 else None
        )

        logger.info("Planning phase: checking downloads...")
        download_duration = run_planning_phase(
            client,
            full_model_id,
            preview,
            args.danger_delete_downloads,
            args.timeout,
            settle_deadline,
        )
        if download_duration is not None:
            logger.info(f"Download: {download_duration:.1f}s")
        else:
            logger.info("Download: model already cached")

        client.request_json("POST", "/instance", body={"instance": instance})
        try:
            wait_for_instance_ready(client, instance_id)
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Failed to initialize placement: {e}")
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{instance_id}")
            return 1

        time.sleep(1)
    else:
        # No instance management — just use the model arg directly
        full_model_id = args.model

    max_gen_toks = fetch_context_length(full_model_id)
    logger.info(f"max_gen_toks={max_gen_toks} (from model context_length)")

    base_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    tasks_dir = str(Path(__file__).resolve().parent / "tasks")

    try:
        if args.compare_concurrency:
            concurrency_levels = parse_int_list(args.compare_concurrency)
            results_by_concurrency: dict[int, dict[str, dict[str, float]]] = {}

            for concurrency in concurrency_levels:
                output_path = f"{args.results_dir}/c{concurrency}"
                cmd = build_lm_eval_cmd(
                    model_id=full_model_id,
                    base_url=base_url,
                    tasks=args.tasks,
                    num_concurrent=concurrency,
                    timeout=args.eval_timeout,
                    max_gen_toks=max_gen_toks,
                    num_fewshot=args.num_fewshot,
                    limit=args.limit,
                    output_path=output_path,
                    tasks_dir=tasks_dir,
                    extra_args=extra_args,
                )

                if args.dry_run:
                    print(f"[c={concurrency}] {' '.join(cmd)}")
                    continue

                logger.info(f"Running eval at concurrency={concurrency}")
                rc = run_lm_eval(cmd)
                if rc != 0:
                    logger.error(f"lm-eval failed with exit code {rc} at concurrency={concurrency}")
                    continue

                results = load_results(output_path)
                if results:
                    scores = extract_scores(results)
                    results_by_concurrency[concurrency] = scores

            if not args.dry_run and results_by_concurrency:
                print_comparison(results_by_concurrency)

        else:
            cmd = build_lm_eval_cmd(
                model_id=full_model_id,
                base_url=base_url,
                tasks=args.tasks,
                num_concurrent=args.num_concurrent,
                timeout=args.eval_timeout,
                max_gen_toks=max_gen_toks,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                output_path=args.results_dir,
                tasks_dir=tasks_dir,
                extra_args=extra_args,
            )

            if args.dry_run:
                print(" ".join(cmd))
                return 0

            rc = run_lm_eval(cmd)
            if rc != 0:
                logger.error(f"lm-eval failed with exit code {rc}")
                return rc

            results = load_results(args.results_dir)
            if results:
                scores = extract_scores(results)
                # Print a simple summary
                print("\n" + "=" * 60)
                print("EVALUATION RESULTS")
                print("=" * 60)
                for task, metrics in sorted(scores.items()):
                    for metric, value in sorted(metrics.items()):
                        print(f"  {task:<30} {metric:<20} {value:.4f}")
                print()

    finally:
        if instance_id is not None:
            try:
                client.request_json("DELETE", f"/instance/{instance_id}")
            except ExoHttpError as e:
                if e.status != 404:
                    raise
            wait_for_instance_gone(client, instance_id)
            logger.debug(f"Deleted instance {instance_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
