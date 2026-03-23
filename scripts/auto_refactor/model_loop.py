"""
Phases 2, 3, 5: Opus triage → Sonnet execution → Opus error loop.

Uses the Anthropic API via the Max plan (never direct sk-ant- calls).
Model routing: Opus=decisions, Sonnet=diff generation, max 2 error iterations.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.auto_refactor.types import (
    CorrectionNote,
    RefactorCandidate,
    RefactorInstruction,
)

_OPUS_TRIAGE_PROMPT = """\
You are the refactor safety authority for the exo distributed ML inference project.

You will receive a list of proposed refactor candidates from a pre-scanner. Your job:
1. APPROVE candidates that are safe mechanical changes with zero ambiguity
2. REJECT candidates that touch logic, APIs, types, or are unclear
3. For APPROVE candidates: write a single-sentence RefactorInstruction

Rules:
- NEVER approve changes to: apply(), State/Event/Command types, test files, async logic
- ONLY approve: syntax sugar (Optional→X|None, Union→X|Y), bare except→except Exception:, f-string conversions, unnecessary else-after-return
- If ambiguous at all: REJECT

Return JSON array of objects:
[
  {{
    "filepath": "...",
    "lineno": 42,
    "category": "...",
    "decision": "APPROVE" or "REJECT",
    "instruction": "In function foo() at line 42, change Optional[str] to str | None",
    "source_line": "the original line"
  }}
]

Candidates:
{candidates_json}

Source context for each candidate (filepath → relevant lines):
{source_context}
"""

_SONNET_DIFF_PROMPT = """\
You are implementing a safe mechanical code refactor for the exo project.

Apply EXACTLY the instruction below to the file. Return ONLY a unified diff (--- a/path +++ b/path format). No explanation, no markdown, no code blocks — just the raw diff.

Instruction: {instruction}
File: {filepath}
Line: {lineno}

Full file source:
{source}

{correction_note}
Return the unified diff only:
"""

_OPUS_ERROR_PROMPT = """\
A refactor instruction failed validation. Analyze the failure and write a CorrectionNote.

Instruction that failed:
{instruction}

Validator output:
{validator_stderr}

Return JSON:
{{
  "instruction_filepath": "{filepath}",
  "instruction_lineno": {lineno},
  "failure_reason": "brief explanation of what went wrong",
  "correction": "specific instruction for Sonnet on what to do differently"
}}
"""


def _call_claude(model: str, prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude via claude CLI (headless mode) — never direct API."""
    result = subprocess.run(
        [
            "claude",
            "--model", model,
            "--max-turns", "1",
            "--output-format", "text",
            "-p", prompt,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [model_loop] claude CLI failed: {result.stderr[:200]}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _load_source_context(candidates: list[RefactorCandidate], context_lines: int = 5) -> str:
    """Load source lines around each candidate for Opus triage context."""
    context_parts: list[str] = []
    seen_files: set[str] = set()

    for c in candidates:
        if c.filepath in seen_files:
            continue
        seen_files.add(c.filepath)
        try:
            lines = Path(c.filepath).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        start = max(0, c.lineno - context_lines - 1)
        end = min(len(lines), c.lineno + context_lines)
        snippet = "\n".join(
            f"{i+1}: {lines[i]}" for i in range(start, end)
        )
        context_parts.append(f"# {c.filepath} (around line {c.lineno})\n{snippet}")

    return "\n\n".join(context_parts)


def opus_triage(candidates: list[RefactorCandidate]) -> list[RefactorInstruction]:
    """Phase 2: Opus reviews candidates and returns approved instructions."""
    if not candidates:
        return []

    candidates_json = json.dumps([c.model_dump() for c in candidates], indent=2)
    source_context = _load_source_context(candidates)

    prompt = _OPUS_TRIAGE_PROMPT.format(
        candidates_json=candidates_json,
        source_context=source_context,
    )

    response = _call_claude("claude-opus-4-6", prompt, max_tokens=8192)
    raw = _extract_json_array(response)

    instructions: list[RefactorInstruction] = []
    for item in raw:
        if item.get("decision") != "APPROVE":
            continue
        try:
            instructions.append(RefactorInstruction(
                filepath=item["filepath"],
                lineno=int(item["lineno"]),
                category=item["category"],
                instruction=item["instruction"],
                source_line=item.get("source_line", ""),
            ))
        except Exception:
            continue

    return instructions


def sonnet_diff(
    instruction: RefactorInstruction,
    correction: CorrectionNote | None = None,
) -> str:
    """Phase 3: Sonnet produces a unified diff for the instruction."""
    try:
        source = Path(instruction.filepath).read_text(encoding="utf-8")
    except OSError:
        return ""

    correction_note = ""
    if correction:
        correction_note = (
            f"\n\nPrevious attempt failed. Correction: {correction.correction}\n"
            f"Reason it failed: {correction.failure_reason}\n"
        )

    prompt = _SONNET_DIFF_PROMPT.format(
        instruction=instruction.instruction,
        filepath=instruction.filepath,
        lineno=instruction.lineno,
        source=source[:12000],
        correction_note=correction_note,
    )

    return _call_claude("claude-sonnet-4-6", prompt, max_tokens=4096)


def opus_error_analysis(
    instruction: RefactorInstruction,
    validator_stderr: str,
) -> CorrectionNote | None:
    """Phase 5: Opus analyzes validator failure and writes correction note."""
    prompt = _OPUS_ERROR_PROMPT.format(
        instruction=instruction.instruction,
        validator_stderr=validator_stderr[:3000],
        filepath=instruction.filepath,
        lineno=instruction.lineno,
    )

    response = _call_claude("claude-opus-4-6", prompt, max_tokens=1024)
    raw = _extract_json_object(response)
    if not raw:
        return None

    try:
        return CorrectionNote(
            instruction_filepath=raw.get("instruction_filepath", instruction.filepath),
            instruction_lineno=int(raw.get("instruction_lineno", instruction.lineno)),
            failure_reason=raw.get("failure_reason", "unknown"),
            correction=raw.get("correction", ""),
        )
    except Exception:
        return None
