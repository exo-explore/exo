import fnmatch
import re
from pathlib import Path

from exo.security_gate.issue import Issue


def filter_issues(
    issues: list[Issue],
    source_lines: dict[str, list[str]],
    ignore_file: Path,
) -> list[Issue]:
    """Remove suppressed issues. source_lines maps filepath -> list of source lines."""
    ignore_patterns = _load_ignore_patterns(ignore_file)
    result: list[Issue] = []
    for issue in issues:
        if _is_suppressed_inline(issue, source_lines.get(issue.filepath, [])):
            continue
        if _is_suppressed_by_ignore_file(issue, ignore_patterns):
            continue
        result.append(issue)
    return result


def _load_ignore_patterns(path: Path) -> list[tuple[str, str | None]]:
    """Returns list of (glob_pattern, check_id_or_None) tuples."""
    if not path.exists():
        return []
    patterns: list[tuple[str, str | None]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            glob, check_id = line.split(":", 1)
            patterns.append((glob.strip(), check_id.strip() or None))
        else:
            patterns.append((line, None))
    return patterns


def _is_suppressed_inline(issue: Issue, lines: list[str]) -> bool:
    """Check for # nosec:CHECK_ID on the issue's line (1-indexed)."""
    if issue.lineno < 1 or issue.lineno > len(lines):
        return False
    line = lines[issue.lineno - 1]
    match = re.search(r"#\s*nosec:([A-Z_,]+)", line)
    if not match:
        return False
    suppressed_ids = {s.strip() for s in match.group(1).split(",")}
    return issue.check_id in suppressed_ids


def _is_suppressed_by_ignore_file(
    issue: Issue, patterns: list[tuple[str, str | None]]
) -> bool:
    for glob, check_id in patterns:
        if (fnmatch.fnmatch(issue.filepath, glob) or fnmatch.fnmatch(
            issue.filepath, f"**/{glob}"
        )) and (check_id is None or check_id == issue.check_id):
            return True
    return False
