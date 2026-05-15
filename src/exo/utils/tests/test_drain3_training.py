from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from exo.utils.drain3_training import (
    ClusterRow,
    TrainingSummary,
    extract_message,
    open_log_file,
    write_summary_json,
    write_templates_tsv,
)


def test_extract_message_uses_named_message_group() -> None:
    pattern = re.compile(r"^\S+\s+\S+\s+(?P<message>.*)$")

    message = extract_message(
        "2026-05-15 INFO worker started\n",
        pattern,
        drop_unmatched=True,
    )

    assert message == "worker started"


def test_extract_message_strips_matched_prefix_without_capture() -> None:
    pattern = re.compile(r"^\[[A-Z]+\]\s+")

    message = extract_message("[ERROR] shard failed", pattern, drop_unmatched=True)

    assert message == "shard failed"


def test_extract_message_can_keep_or_drop_unmatched_lines() -> None:
    pattern = re.compile(r"^structured: (?P<message>.*)$")

    assert (
        extract_message("plain log line", pattern, drop_unmatched=False)
        == "plain log line"
    )
    assert extract_message("plain log line", pattern, drop_unmatched=True) is None


def test_open_log_file_reads_gzip_logs(tmp_path: Path) -> None:
    path = tmp_path / "exo.log.gz"
    with gzip.open(path, mode="wt", encoding="utf-8") as output_file:
        output_file.write("alpha\nbeta\n")

    with open_log_file(path, "utf-8") as input_file:
        assert input_file.read() == "alpha\nbeta\n"


def test_template_outputs_are_stable(tmp_path: Path) -> None:
    rows = [
        ClusterRow(cluster_id=7, size=3, template="worker <*> started"),
        ClusterRow(cluster_id=2, size=1, template="path\tchanged\nagain"),
    ]
    templates_tsv_path = tmp_path / "templates.tsv"
    templates_json_path = tmp_path / "templates.json"
    summary = TrainingSummary(
        input_path=tmp_path / "input.log",
        output_dir=tmp_path,
        state_path=tmp_path / "drain3_state.bin",
        templates_json_path=templates_json_path,
        templates_tsv_path=templates_tsv_path,
        changes_jsonl_path=tmp_path / "template_changes.jsonl",
        matches_jsonl_path=None,
        lines_read=4,
        messages_trained=4,
        skipped_empty=0,
        skipped_unmatched=0,
        elapsed_seconds=2.0,
        change_counts={"none": 2, "cluster_created": 2},
        clusters=rows,
    )

    write_templates_tsv(templates_tsv_path, rows)
    write_summary_json(templates_json_path, summary)

    assert templates_tsv_path.read_text(encoding="utf-8").splitlines() == [
        "cluster_id\tsize\ttemplate",
        "7\t3\tworker <*> started",
        "2\t1\tpath\\tchanged\\nagain",
    ]
    payload = json.loads(templates_json_path.read_text(encoding="utf-8"))
    assert payload["messages_trained"] == 4
    assert payload["lines_per_second"] == 2.0
    assert payload["clusters"][0] == {
        "cluster_id": 7,
        "size": 3,
        "template": "worker <*> started",
    }
