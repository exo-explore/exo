"""
exo Security Gate — pre-commit security checker.

Usage:
    python -m exo.security_gate.gate [files...]
    python -m exo.security_gate.gate --verbose [files...]

Exit codes:
    0 — no blocking issues
    1 — one or more blocking issues found
    2 — internal error (e.g. all files failed to parse)
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import cast

from exo.security_gate.checks import ALL_CHECKS
from exo.security_gate.issue import Issue
from exo.security_gate.suppressions import filter_issues

# Root of the exo source tree — limit scanning to here when no files are given
_DEFAULT_SCAN_ROOT = Path(__file__).parent.parent.parent / "exo"
# The .security-gate-ignore file lives at the repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_IGNORE_FILE = _REPO_ROOT / ".security-gate-ignore"


def _find_default_files() -> list[Path]:
    """When no filenames are passed, scan all .py files under src/exo/."""
    root = Path(__file__).parent.parent.parent
    return sorted(root.rglob("*.py"))


def _run_checks(filepath: str, source: str, verbose: bool) -> list[Issue]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        print(f"exo-security-gate: WARNING: could not parse {filepath}: {exc}", file=sys.stderr)
        return []

    issues: list[Issue] = []
    for check in ALL_CHECKS:
        try:
            issues.extend(check(filepath, source, tree))
        except Exception as exc:  # noqa: BLE001
            print(
                f"exo-security-gate: WARNING: check {check.__name__} failed on {filepath}: {exc}",
                file=sys.stderr,
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exo-security-gate",
        description="Pre-commit security gate for the exo project",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to check (default: all .py files in src/exo/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show advisory issues in addition to blocking ones",
    )
    parser.add_argument(
        "--ignore-file",
        default=str(_IGNORE_FILE),
        help="Path to .security-gate-ignore file",
    )
    args = parser.parse_args(argv)
    _raw_files: list[str] = cast(list[str], getattr(args, "files", []) or [])
    arg_files: list[str] = _raw_files
    arg_verbose: bool = cast(bool, args.verbose)
    arg_ignore_file: str = cast(str, args.ignore_file)

    paths = [Path(f) for f in arg_files] if arg_files else _find_default_files()

    all_issues: list[Issue] = []
    source_lines_map: dict[str, list[str]] = {}
    parse_failures = 0

    for path in paths:
        filepath = str(path)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"exo-security-gate: WARNING: cannot read {filepath}: {exc}", file=sys.stderr)
            parse_failures += 1
            continue

        source_lines_map[filepath] = source.splitlines()
        issues = _run_checks(filepath, source, arg_verbose)
        all_issues.extend(issues)

    # Apply suppressions
    ignore_file = Path(arg_ignore_file)
    all_issues = filter_issues(all_issues, source_lines_map, ignore_file)

    # Separate blocking from advisory
    blocking = [i for i in all_issues if i.severity == "block"]
    advisory = [i for i in all_issues if i.severity == "advisory"]

    # Build display list
    to_display = blocking + (advisory if arg_verbose else [])

    if to_display:
        print(f"\nexo-security-gate: {len(blocking)} blocking issue(s) found", end="")
        if arg_verbose and advisory:
            print(f" + {len(advisory)} advisory", end="")
        print("\n")

        for issue in to_display:
            print(f"  {issue.filepath}:{issue.lineno}  [{issue.check_id}] {issue.message}")

        print(
            "\nSuppress with `# nosec:CHECK_ID` inline or add path to .security-gate-ignore"
        )
    else:
        if arg_verbose:
            print("exo-security-gate: no issues found")

    # Exit code
    if blocking:
        return 1
    if parse_failures > 0 and len(paths) == parse_failures:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
