"""
Autonomous Refactor Agent — main orchestrator.

Usage:
    uv run python scripts/auto_refactor/orchestrator.py [--dry-run] [--verbose]

Phases:
    0. Pre-flight: stash, branch, capture baseline, abort if tests broken
    1. Qwen pre-scan (local, free tokens)
    2. Opus triage (approve/reject candidates)
    3. Sonnet execution (unified diffs)
    4. Validation gate (ruff + basedpyright + pytest)
    5. Opus error loop (max 2 iterations on failure)
    6. Commit (only changed files, no push)
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from scripts.auto_refactor.model_loop import (
    opus_error_analysis,
    opus_triage,
    sonnet_diff,
)
from scripts.auto_refactor.scanner import scan_repo
from scripts.auto_refactor.types import RefactorResult
from scripts.auto_refactor.validator import (
    capture_baseline,
    check_baseline_clean,
    run_validation,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_MAX_ERROR_ITERATIONS = 2


def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _apply_diff(diff_text: str, dry_run: bool = False) -> bool:
    """Apply a unified diff via `patch`. Returns True on success."""
    if not diff_text.strip():
        return False
    cmd = ["patch", "-p1"]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(
        cmd,
        input=diff_text,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _revert_changes() -> None:
    """Revert all unstaged changes — called before error loop re-attempt."""
    _git(["checkout", "--", "."], check=False)


def _phase_0_preflight(dry_run: bool, verbose: bool) -> tuple[str | None, dict[str, int]]:
    """Stash, branch, baseline. Returns (stash_ref, baseline_counts)."""
    # Check for uncommitted changes
    status = _git(["status", "--porcelain"])
    stash_ref: str | None = None
    if status.stdout.strip() and not dry_run:
        _git(["stash", "push", "-m", "auto-refactor-preflight"])
        stash_ref = "stash@{0}"
        if verbose:
            print("[phase0] stashed uncommitted changes")

    # Create branch
    branch = f"refactor/auto-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not dry_run:
        _git(["checkout", "-b", branch])
        if verbose:
            print(f"[phase0] created branch {branch}")

    # Verify baseline is clean
    if not check_baseline_clean(_REPO_ROOT):
        print("[phase0] ABORT: tests fail on baseline — not safe to refactor", file=sys.stderr)
        if stash_ref and not dry_run:
            _git(["stash", "pop"], check=False)
        sys.exit(1)

    baseline = capture_baseline(_REPO_ROOT)
    if verbose:
        print(f"[phase0] baseline: ruff={baseline.get('ruff', 0)} pyright_errors={baseline.get('pyright_errors', 0)}")

    return stash_ref, baseline


def _phase_6_commit(changed_files: list[str], dry_run: bool, verbose: bool) -> None:
    """Stage only changed files and commit."""
    if dry_run or not changed_files:
        return

    for filepath in changed_files:
        _git(["add", filepath], check=False)

    count = len(changed_files)
    msg = (
        f"refactor(auto): {count} mechanical fix(es)\n\n"
        "Applied by autonomous refactor agent (Opus→Sonnet→Opus loop).\n"
        "Changes: syntax sugar only (Optional→X|None, Union→X|Y, bare except, f-strings).\n"
        "Validated: ruff + basedpyright + pytest all pass.\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    _git(["commit", "-m", msg])
    if verbose:
        print(f"[phase6] committed {count} file(s)")


def run(dry_run: bool = False, verbose: bool = False, ollama_url: str = "http://localhost:11434") -> None:
    print("=== Autonomous Refactor Agent ===")

    # Phase 0
    print("[phase0] pre-flight checks...")
    stash_ref, baseline = _phase_0_preflight(dry_run, verbose)

    # Phase 1: Qwen scan
    print("[phase1] Qwen pre-scan (local)...")
    candidates = scan_repo(_REPO_ROOT, ollama_url=ollama_url, verbose=verbose)
    print(f"[phase1] {len(candidates)} candidate(s) found")

    if not candidates:
        print("[done] no candidates. Nothing to do.")
        return

    # Phase 2: Opus triage
    print("[phase2] Opus triage...")
    instructions = opus_triage(candidates)
    print(f"[phase2] {len(instructions)} instruction(s) approved by Opus")

    if not instructions:
        print("[done] Opus rejected all candidates. Nothing to do.")
        return

    # Phase 3–5: Sonnet execution + validation + error loop
    results: list[RefactorResult] = []
    changed_files: list[str] = []

    for idx, instruction in enumerate(instructions, 1):
        print(f"[phase3] ({idx}/{len(instructions)}) {instruction.filepath}:{instruction.lineno} [{instruction.category}]")

        correction = None
        status = "dropped"
        reason = "max iterations exceeded"

        for iteration in range(1, _MAX_ERROR_ITERATIONS + 2):
            diff = sonnet_diff(instruction, correction)
            if not diff.strip():
                status = "skipped"
                reason = "Sonnet produced empty diff"
                break

            # Dry-run patch check
            if not _apply_diff(diff, dry_run=True):
                status = "skipped"
                reason = "patch --dry-run failed (diff does not apply cleanly)"
                break

            if dry_run:
                status = "applied"
                reason = f"dry-run: diff valid (iteration {iteration})"
                changed_files.append(instruction.filepath)
                break

            # Apply for real
            if not _apply_diff(diff, dry_run=False):
                status = "skipped"
                reason = "patch apply failed"
                break

            # Phase 4: Validation gate
            val = run_validation(_REPO_ROOT, baseline)
            if val.passed:
                status = "applied"
                reason = f"passed validation (iteration {iteration})"
                changed_files.append(instruction.filepath)
                if verbose:
                    print(f"  ✓ applied at iteration {iteration}")
                break

            # Phase 5: Error loop
            if verbose:
                print(f"  ✗ validation failed ({val.tool}) at iteration {iteration}")
            _revert_changes()

            if iteration > _MAX_ERROR_ITERATIONS:
                status = "dropped"
                reason = f"dropped after {_MAX_ERROR_ITERATIONS} failed iterations ({val.tool})"
                break

            correction = opus_error_analysis(instruction, val.stderr)
            if correction is None:
                status = "dropped"
                reason = "Opus could not analyze failure"
                break

        results.append(RefactorResult(
            instruction=instruction,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            iterations=idx,
        ))

    # Phase 6: Commit
    unique_changed = list(dict.fromkeys(changed_files))  # preserve order, dedupe
    if unique_changed and not dry_run:
        print(f"[phase6] committing {len(unique_changed)} file(s)...")
        _phase_6_commit(unique_changed, dry_run, verbose)

    # Restore stash
    if stash_ref and not dry_run:
        _git(["stash", "pop"], check=False)

    # Summary
    applied = [r for r in results if r.status == "applied"]
    skipped = [r for r in results if r.status == "skipped"]
    dropped = [r for r in results if r.status == "dropped"]
    print(f"\n=== Summary: {len(applied)} applied, {len(skipped)} skipped, {len(dropped)} dropped ===")
    for r in results:
        icon = {"applied": "✓", "skipped": "~", "dropped": "✗"}.get(r.status, "?")
        print(f"  {icon} {r.instruction.filepath}:{r.instruction.lineno} [{r.instruction.category}] — {r.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous code refactor agent (Opus→Sonnet→Opus)")
    parser.add_argument("--dry-run", action="store_true", help="Validate diffs but do not apply or commit")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL for Qwen")
    args = parser.parse_args()
    run(dry_run=args.dry_run, verbose=args.verbose, ollama_url=args.ollama_url)


if __name__ == "__main__":
    main()
