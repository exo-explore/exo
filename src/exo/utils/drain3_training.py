from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


DEFAULT_MASKS: tuple[tuple[str, str], ...] = (
    (
        r"((?<=[^A-Za-z0-9])|^)"
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
        r"(?:Z|[+-]\d{2}:?\d{2})?)"
        r"((?=[^A-Za-z0-9])|$)",
        "TIMESTAMP",
    ),
    (
        r"((?<=[^A-Za-z0-9])|^)"
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
        r"((?=[^A-Za-z0-9])|$)",
        "UUID",
    ),
    (
        r"((?<=[^A-Za-z0-9])|^)"
        r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"((?=[^A-Za-z0-9])|$)",
        "IP",
    ),
    (
        r"((?<=[^A-Za-z0-9])|^)"
        r"((?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4})"
        r"((?=[^A-Za-z0-9])|$)",
        "IPV6",
    ),
    (r"((?<=[^A-Za-z0-9])|^)(0x[a-fA-F0-9]+)((?=[^A-Za-z0-9])|$)", "HEX"),
    (r"((?<=[^A-Za-z0-9])|^)([\-+]?\d+(?:\.\d+)?)((?=[^A-Za-z0-9])|$)", "NUM"),
)


@dataclass(frozen=True)
class ClusterRow:
    cluster_id: int
    size: int
    template: str


@dataclass(frozen=True)
class TrainingSummary:
    input_path: Path
    output_dir: Path
    state_path: Path
    templates_json_path: Path
    templates_tsv_path: Path
    changes_jsonl_path: Path
    matches_jsonl_path: Path | None
    lines_read: int
    messages_trained: int
    skipped_empty: int
    skipped_unmatched: int
    elapsed_seconds: float
    change_counts: Mapping[str, int]
    clusters: Sequence[ClusterRow]


@contextmanager
def open_log_file(path: Path, encoding: str) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding=encoding, errors="replace") as log_file:
            yield log_file
    else:
        with path.open(encoding=encoding, errors="replace") as log_file:
            yield log_file


def extract_message(
    raw_line: str,
    message_regex: re.Pattern[str] | None,
    *,
    drop_unmatched: bool,
) -> str | None:
    line = raw_line.rstrip("\r\n")
    if message_regex is None:
        return line

    match = message_regex.search(line)
    if match is None:
        return None if drop_unmatched else line

    named_message = match.groupdict().get("message")
    if named_message is not None:
        return named_message
    if match.lastindex is not None:
        for group_index in range(1, match.lastindex + 1):
            captured = match.group(group_index)
            if captured is not None:
                return captured

    return line[match.end() :].lstrip()


def default_output_dir(input_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("drain3-output") / f"{input_path.stem}-{timestamp}"


def write_templates_tsv(path: Path, rows: Sequence[ClusterRow]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("cluster_id\tsize\ttemplate\n")
        for row in rows:
            template = row.template.replace("\t", r"\t").replace("\n", r"\n")
            output_file.write(f"{row.cluster_id}\t{row.size}\t{template}\n")


def write_summary_json(path: Path, summary: TrainingSummary) -> None:
    payload = {
        "input_path": str(summary.input_path),
        "output_dir": str(summary.output_dir),
        "state_path": str(summary.state_path),
        "templates_tsv_path": str(summary.templates_tsv_path),
        "changes_jsonl_path": str(summary.changes_jsonl_path),
        "matches_jsonl_path": (
            str(summary.matches_jsonl_path) if summary.matches_jsonl_path else None
        ),
        "lines_read": summary.lines_read,
        "messages_trained": summary.messages_trained,
        "skipped_empty": summary.skipped_empty,
        "skipped_unmatched": summary.skipped_unmatched,
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "lines_per_second": (
            round(summary.messages_trained / summary.elapsed_seconds, 3)
            if summary.elapsed_seconds > 0
            else None
        ),
        "change_counts": dict(summary.change_counts),
        "clusters": [
            {
                "cluster_id": row.cluster_id,
                "size": row.size,
                "template": row.template,
            }
            for row in summary.clusters
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
