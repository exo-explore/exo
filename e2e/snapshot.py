"""Snapshot testing infrastructure for E2E tests.

Provides deterministic regression testing by comparing inference output
against saved snapshots. On first run, snapshots are created automatically.
Set UPDATE_SNAPSHOTS=1 to regenerate snapshots when output intentionally changes.

Snapshots are stored per-architecture (e.g. snapshots/x86_64/, snapshots/arm64/)
since floating-point results differ between CPU architectures.
"""

import difflib
import json
import os
import platform
from pathlib import Path

ARCH = platform.machine()
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots" / ARCH


def assert_snapshot(
    name: str,
    content: str,
    metadata: dict,
) -> None:
    """Compare content against a saved snapshot, or create one if missing.

    Args:
        name: Snapshot identifier (used as filename: snapshots/{arch}/{name}.json).
        content: The actual inference output to compare.
        metadata: Additional context stored alongside content (model, seed, etc.).
                  Not used for comparison -- purely documentary.

    Raises:
        AssertionError: If content doesn't match the saved snapshot.

    Environment:
        UPDATE_SNAPSHOTS=1: Overwrite existing snapshot with actual content.
    """
    snapshot_file = SNAPSHOTS_DIR / f"{name}.json"
    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"

    if snapshot_file.exists() and not update:
        snapshot = json.loads(snapshot_file.read_text())
        expected = snapshot["content"]
        if content != expected:
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    content.splitlines(),
                    fromfile=f"expected ({snapshot_file.relative_to(SNAPSHOTS_DIR.parent.parent)})",
                    tofile="actual",
                    lineterm="",
                )
            )
            raise AssertionError(
                f"Snapshot mismatch for '{name}' on {ARCH}!\n\n"
                f"{diff}\n\n"
                f"Expected: {expected!r}\n"
                f"Actual:   {content!r}\n\n"
                f"To update: UPDATE_SNAPSHOTS=1 python3 e2e/run_all.py"
            )
        print(f"  Output matches snapshot ({ARCH}/{snapshot_file.name})")
    else:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_data = {**metadata, "arch": ARCH, "content": content}
        snapshot_file.write_text(json.dumps(snapshot_data, indent=2) + "\n")
        action = "Updated" if update else "Created"
        print(f"  {action} snapshot: {ARCH}/{snapshot_file.name}")
