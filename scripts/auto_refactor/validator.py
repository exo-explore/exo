"""
Phase 4: Validation gate — ruff + basedpyright + pytest.

Short-circuits on first failure. Captures baseline counts before any
refactoring so we can compare deltas rather than absolute numbers.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    tool: str          # which tool failed (empty if passed)
    stderr: str        # full stderr/stdout for Opus error loop
    new_violations: int = 0


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr


def capture_baseline(repo_root: Path) -> dict[str, int]:
    """Capture baseline error counts BEFORE any changes."""
    baseline: dict[str, int] = {}

    _, ruff_out = _run(["uv", "run", "ruff", "check", "src/"], repo_root)
    baseline["ruff"] = ruff_out.count(": E") + ruff_out.count(": W") + ruff_out.count(": F")

    _, pyright_out = _run(["uv", "run", "basedpyright"], repo_root)
    # basedpyright summary line: "X errors, Y warnings, Z informations"
    import re
    match = re.search(r"(\d+) error", pyright_out)
    baseline["pyright_errors"] = int(match.group(1)) if match else 0

    return baseline


def run_validation(repo_root: Path, baseline: dict[str, int]) -> ValidationResult:
    """Run full validation suite. Returns first failure."""

    # 1. ruff
    rc, ruff_out = _run(["uv", "run", "ruff", "check", "src/"], repo_root)
    if rc != 0:
        new_count = ruff_out.count(": E") + ruff_out.count(": W") + ruff_out.count(": F")
        delta = new_count - baseline.get("ruff", 0)
        if delta > 0:
            return ValidationResult(
                passed=False,
                tool="ruff",
                stderr=ruff_out,
                new_violations=delta,
            )

    # 2. basedpyright
    rc, pyright_out = _run(["uv", "run", "basedpyright"], repo_root)
    if rc != 0:
        import re
        match = re.search(r"(\d+) error", pyright_out)
        current_errors = int(match.group(1)) if match else 0
        if current_errors > baseline.get("pyright_errors", 0):
            return ValidationResult(
                passed=False,
                tool="basedpyright",
                stderr=pyright_out,
                new_violations=current_errors - baseline.get("pyright_errors", 0),
            )

    # 3. pytest
    rc, pytest_out = _run(["uv", "run", "pytest", "-x", "-q", "--timeout=60"], repo_root)
    if rc != 0:
        return ValidationResult(
            passed=False,
            tool="pytest",
            stderr=pytest_out,
        )

    return ValidationResult(passed=True, tool="", stderr="")


def check_baseline_clean(repo_root: Path) -> bool:
    """Verify tests pass before we start. Abort if broken."""
    rc, _ = _run(["uv", "run", "pytest", "-x", "-q", "--timeout=60"], repo_root)
    return rc == 0
