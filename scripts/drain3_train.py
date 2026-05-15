#!/usr/bin/env python3
"""Train Drain3 templates from a collected log file.

Usage:
    uv run --with drain3 python scripts/drain3_train.py /path/to/exo.log
    uv run --with drain3 python scripts/drain3_train.py /path/to/exo.log \
        --message-regex '^[^|]+\\|(?P<message>.*)$'

The script streams the input file into Drain3 training mode, writes mined
templates to JSON and TSV files, and saves a Drain3 state snapshot that can be
loaded by later Drain3 tooling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from exo.utils.drain3_training import (
    DEFAULT_MASKS,
    ClusterRow,
    TrainingSummary,
    default_output_dir,
    extract_message,
    open_log_file,
    write_summary_json,
    write_templates_tsv,
)


class Drain3Modules:
    template_miner: Any
    template_miner_config: Any
    file_persistence: Any
    masking_instruction: Any


def load_drain3_modules() -> Drain3Modules:
    try:
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence
        from drain3.masking import MaskingInstruction
        from drain3.template_miner_config import TemplateMinerConfig
    except ModuleNotFoundError as exc:
        if exc.name == "drain3":
            raise SystemExit(
                "Drain3 is not installed. Run this script with:\n"
                "  uv run --with drain3 python scripts/drain3_train.py <log-file>"
            ) from exc
        raise

    modules = Drain3Modules()
    modules.template_miner = TemplateMiner
    modules.template_miner_config = TemplateMinerConfig
    modules.file_persistence = FilePersistence
    modules.masking_instruction = MaskingInstruction
    return modules


def build_config(args: argparse.Namespace, modules: Drain3Modules) -> Any:
    config = modules.template_miner_config()
    if args.config is not None:
        config.load(str(args.config))
    elif not args.no_default_masks:
        config.mask_prefix = "<:"
        config.mask_suffix = ":>"
        config.masking_instructions = [
            modules.masking_instruction(regex_pattern, mask_with)
            for regex_pattern, mask_with in DEFAULT_MASKS
        ]

    if args.sim_th is not None:
        config.drain_sim_th = args.sim_th
    if args.depth is not None:
        config.drain_depth = args.depth
    if args.max_children is not None:
        config.drain_max_children = args.max_children
    if args.max_clusters is not None:
        config.drain_max_clusters = args.max_clusters
    if args.extra_delimiter:
        config.drain_extra_delimiters = args.extra_delimiter
    if args.profiling:
        config.profiling_enabled = True

    return config


def cluster_rows(template_miner: Any) -> list[ClusterRow]:
    clusters = sorted(
        template_miner.drain.clusters,
        key=lambda cluster: (-cluster.size, cluster.cluster_id),
    )
    return [
        ClusterRow(
            cluster_id=int(cluster.cluster_id),
            size=int(cluster.size),
            template=str(cluster.get_template()),
        )
        for cluster in clusters
    ]


def train_log_file(args: argparse.Namespace) -> TrainingSummary:
    input_path = args.input_log.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input log file not found: {input_path}")

    output_dir = (args.output_dir or default_output_dir(input_path)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = (args.state or output_dir / "drain3_state.bin").resolve()
    templates_json_path = output_dir / "templates.json"
    templates_tsv_path = output_dir / "templates.tsv"
    changes_jsonl_path = output_dir / "template_changes.jsonl"
    matches_jsonl_path = output_dir / "matches.jsonl" if args.write_matches else None

    modules = load_drain3_modules()
    config = build_config(args, modules)
    persistence = modules.file_persistence(str(state_path))
    template_miner = modules.template_miner(persistence_handler=persistence, config=config)
    message_regex = re.compile(args.message_regex) if args.message_regex else None

    change_counts: Counter[str] = Counter()
    lines_read = 0
    messages_trained = 0
    skipped_empty = 0
    skipped_unmatched = 0
    start_time = time.monotonic()

    with (
        open_log_file(input_path, args.encoding) as log_file,
        changes_jsonl_path.open("w", encoding="utf-8") as changes_file,
    ):
        matches_file_context = (
            matches_jsonl_path.open("w", encoding="utf-8")
            if matches_jsonl_path is not None
            else None
        )
        try:
            matches_file = matches_file_context.__enter__() if matches_file_context else None
            for raw_line in log_file:
                lines_read += 1
                if args.max_lines is not None and lines_read > args.max_lines:
                    break

                message = extract_message(
                    raw_line,
                    message_regex,
                    drop_unmatched=args.drop_unmatched,
                )
                if message is None:
                    skipped_unmatched += 1
                    continue
                if not message and not args.include_empty:
                    skipped_empty += 1
                    continue

                result = dict(template_miner.add_log_message(message))
                change_type = str(result.get("change_type", "unknown"))
                change_counts[change_type] += 1
                messages_trained += 1

                if change_type != "none":
                    change_record = {
                        "line_number": lines_read,
                        "message": message,
                        **result,
                    }
                    changes_file.write(json.dumps(change_record, sort_keys=True) + "\n")

                if matches_file is not None:
                    match_record = {
                        "line_number": lines_read,
                        "message": message,
                        **result,
                    }
                    matches_file.write(json.dumps(match_record, sort_keys=True) + "\n")

                if (
                    args.progress_interval > 0
                    and messages_trained % args.progress_interval == 0
                ):
                    elapsed = time.monotonic() - start_time
                    rate = messages_trained / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Processed {messages_trained} messages, "
                        f"{len(template_miner.drain.clusters)} clusters, "
                        f"{rate:.1f} lines/sec",
                        file=sys.stderr,
                    )
        finally:
            if matches_file_context is not None:
                matches_file_context.__exit__(None, None, None)

    if messages_trained == 0:
        raise RuntimeError("No log messages were trained from the input file")

    # Drain3 snapshots on template changes and intervals; force final cluster sizes into state.
    template_miner.save_state("final")

    rows = cluster_rows(template_miner)
    write_templates_tsv(templates_tsv_path, rows)

    elapsed_seconds = time.monotonic() - start_time
    summary = TrainingSummary(
        input_path=input_path,
        output_dir=output_dir,
        state_path=state_path,
        templates_json_path=templates_json_path,
        templates_tsv_path=templates_tsv_path,
        changes_jsonl_path=changes_jsonl_path,
        matches_jsonl_path=matches_jsonl_path,
        lines_read=lines_read,
        messages_trained=messages_trained,
        skipped_empty=skipped_empty,
        skipped_unmatched=skipped_unmatched,
        elapsed_seconds=elapsed_seconds,
        change_counts=change_counts,
        clusters=rows,
    )
    write_summary_json(templates_json_path, summary)
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_log", type=Path, help="Plain text or .gz log file to train on.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for templates.json, templates.tsv, state, and change logs.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Drain3 FilePersistence snapshot path. Defaults to <output-dir>/drain3_state.bin.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional Drain3 .ini file. CLI tuning flags override values loaded from it.",
    )
    parser.add_argument(
        "--message-regex",
        help=(
            "Regex for extracting the unstructured message. Use a named group "
            "'message' or a capture group. With no captures, the matched prefix is stripped."
        ),
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="Skip lines that do not match --message-regex instead of training on the full line.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Train on empty messages instead of skipping blank lines.",
    )
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    parser.add_argument("--max-lines", type=positive_int, help="Stop after reading this many lines.")
    parser.add_argument(
        "--write-matches",
        action="store_true",
        help="Write one matches.jsonl record per trained message. This can be large.",
    )
    parser.add_argument(
        "--progress-interval",
        type=non_negative_int,
        default=10_000,
        help="Print progress every N trained messages. Use 0 to disable.",
    )
    parser.add_argument(
        "--top",
        type=non_negative_int,
        default=25,
        help="Print the top N templates after training. Use 0 to suppress.",
    )
    parser.add_argument("--sim-th", type=float, help="Drain similarity threshold, default 0.4.")
    parser.add_argument("--depth", type=positive_int, help="Drain parse tree depth, default 4.")
    parser.add_argument(
        "--max-children",
        type=positive_int,
        help="Max children per internal Drain node, default 100.",
    )
    parser.add_argument(
        "--max-clusters",
        type=positive_int,
        help="Max clusters to retain. Omit for Drain3's unlimited default.",
    )
    parser.add_argument(
        "--extra-delimiter",
        action="append",
        default=[],
        help="Extra delimiter passed to Drain3. Repeat for multiple delimiters.",
    )
    parser.add_argument(
        "--no-default-masks",
        action="store_true",
        help="Use Drain3's bare defaults instead of this script's timestamp/IP/UUID/number masks.",
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Enable Drain3's profiler output.",
    )
    return parser


def print_summary(summary: TrainingSummary, top: int) -> None:
    rate = summary.messages_trained / summary.elapsed_seconds
    print(
        f"Trained {summary.messages_trained} messages from {summary.lines_read} lines "
        f"in {summary.elapsed_seconds:.2f}s ({rate:.1f} lines/sec)."
    )
    print(f"Clusters: {len(summary.clusters)}")
    print(f"Templates JSON: {summary.templates_json_path}")
    print(f"Templates TSV:  {summary.templates_tsv_path}")
    print(f"Drain3 state:   {summary.state_path}")
    print(f"Changes JSONL:  {summary.changes_jsonl_path}")
    if summary.matches_jsonl_path is not None:
        print(f"Matches JSONL:  {summary.matches_jsonl_path}")
    if summary.skipped_empty or summary.skipped_unmatched:
        print(
            f"Skipped: {summary.skipped_empty} empty, "
            f"{summary.skipped_unmatched} regex-unmatched"
        )

    if top <= 0:
        return

    print()
    print(f"Top {min(top, len(summary.clusters))} templates:")
    for row in summary.clusters[:top]:
        print(f"{row.size:>8}  #{row.cluster_id:<5} {row.template}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = train_log_file(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
