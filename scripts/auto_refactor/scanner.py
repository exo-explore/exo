"""
Phase 1: Qwen pre-scan — finds refactor candidates using local Ollama.

Sends each file to Qwen (http://localhost:11434) and collects proposed
RefactorCandidate objects. Opus/Sonnet see nothing from this phase.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.auto_refactor.types import RefactorCandidate

# Categories Qwen is allowed to propose (must match RefactorCandidate.category)
ALLOWED_CATEGORIES: frozenset[str] = frozenset({
    "optional_syntax",
    "union_syntax",
    "missing_annotation",
    "redundant_pass",
    "bare_exception",
    "f_string_format",
    "unnecessary_else",
})

# Paths that must never be touched
BLOCKLIST_PATTERNS: list[str] = [
    "src/exo/shared/types/",   # Event/State/Command types
    "src/exo/shared/apply.py", # apply() is sacred
    "tests/",
    "rust/",
    "dashboard/",
    "**/tests/**",
    "**/__pycache__/**",
]

_QWEN_PROMPT = """\
You are a conservative Python refactor scanner. Analyze the following Python source file and return ONLY a JSON array of refactor candidates. Each candidate must be a safe mechanical change — no logic changes, no API changes, no architecture changes.

Allowed categories (use these exact strings):
- "optional_syntax": Optional[X] that should be X | None
- "union_syntax": Union[X, Y] that should be X | Y
- "missing_annotation": function parameter or return missing type annotation (only flag if the rest of the function IS annotated)
- "redundant_pass": pass statement that is the only statement in a class/function body and is genuinely redundant
- "bare_exception": bare `except:` that should be `except Exception:`
- "f_string_format": `"{}".format(x)` or `"" % x` that could be an f-string
- "unnecessary_else": else clause after a return/raise that could be eliminated (dedented)

Return ONLY a JSON array. If nothing to report, return [].

File: {filepath}
```python
{source}
```

Return format (JSON array):
[
  {{"filepath": "{filepath}", "lineno": 42, "category": "optional_syntax", "description": "brief description", "source_line": "the exact line"}}
]
"""


def _is_blocked(path: Path, repo_root: Path) -> bool:
    rel = str(path.relative_to(repo_root))
    for pattern in BLOCKLIST_PATTERNS:
        if pattern.endswith("/") and rel.startswith(pattern):
            return True
        if pattern.startswith("**/") and ("/" + pattern[3:]) in ("/" + rel):
            return True
        if rel == pattern:
            return True
    return False


def _qwen_scan(filepath: str, source: str, ollama_url: str) -> list[dict[str, Any]]:
    """Call Qwen via Ollama and parse the JSON response."""
    import urllib.request

    prompt = _QWEN_PROMPT.format(filepath=filepath, source=source[:8000])  # cap at 8K chars
    payload = json.dumps({
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 2048},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        response_text = data.get("response", "")
    except Exception as exc:
        print(f"  [scanner] Qwen call failed for {filepath}: {exc}", file=sys.stderr)
        return []

    # Extract JSON array from response
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not match:
        return []
    try:
        candidates = json.loads(match.group(0))
        if not isinstance(candidates, list):
            return []
        return candidates
    except json.JSONDecodeError:
        return []


def scan_file(
    path: Path,
    repo_root: Path,
    ollama_url: str = "http://localhost:11434",
) -> list[RefactorCandidate]:
    """Scan one file. Returns validated candidates only."""
    if _is_blocked(path, repo_root):
        return []

    source = path.read_text(encoding="utf-8")

    # Quick AST parse check — skip unparseable files
    try:
        ast.parse(source)
    except SyntaxError:
        return []

    raw = _qwen_scan(str(path), source, ollama_url)
    candidates: list[RefactorCandidate] = []
    lines = source.splitlines()

    for item in raw:
        if not isinstance(item, dict):
            continue
        category = item.get("category", "")
        if category not in ALLOWED_CATEGORIES:
            continue
        lineno = int(item.get("lineno", 0))
        if lineno < 1 or lineno > len(lines):
            continue
        try:
            candidates.append(RefactorCandidate(
                filepath=str(path),
                lineno=lineno,
                category=category,  # type: ignore[arg-type]
                description=str(item.get("description", "")),
                source_line=lines[lineno - 1],
            ))
        except Exception:
            continue

    return candidates


def scan_repo(
    repo_root: Path,
    ollama_url: str = "http://localhost:11434",
    verbose: bool = False,
) -> list[RefactorCandidate]:
    """Scan all Python files in repo (respecting blocklist)."""
    all_candidates: list[RefactorCandidate] = []
    py_files = sorted((repo_root / "src" / "exo").rglob("*.py"))

    for path in py_files:
        if _is_blocked(path, repo_root):
            continue
        if verbose:
            print(f"  [scanner] scanning {path.relative_to(repo_root)}")
        candidates = scan_file(path, repo_root, ollama_url)
        all_candidates.extend(candidates)

    # Deduplicate by (filepath, lineno, category)
    seen: set[tuple[str, int, str]] = set()
    deduped: list[RefactorCandidate] = []
    for c in all_candidates:
        key = (c.filepath, c.lineno, c.category)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped
