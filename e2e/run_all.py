#!/usr/bin/env python3
"""Discovers and runs all E2E tests in e2e/test_*.py.

Tests with '# slow' on the first line of their docstring are skipped
unless --slow is passed or E2E_SLOW=1 is set.
"""

import os
import subprocess
import sys
from pathlib import Path

E2E_DIR = Path(__file__).parent.resolve()


def is_slow(test_file: Path) -> bool:
    """Check if the test file is marked as slow (has '# slow' in first 3 lines)."""
    with open(test_file) as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                # Read into the docstring
                for doc_line in f:
                    if "slow" in doc_line.lower() and doc_line.strip().startswith(
                        "slow"
                    ):
                        return True
                    if '"""' in doc_line or "'''" in doc_line:
                        break
            break
    return False


def main():
    run_slow = "--slow" in sys.argv or os.environ.get("E2E_SLOW") == "1"
    if "--update-snapshots" in sys.argv:
        os.environ["UPDATE_SNAPSHOTS"] = "1"
    test_files = sorted(E2E_DIR.glob("test_*.py"))
    if not test_files:
        print("No test files found")
        sys.exit(1)

    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for test_file in test_files:
        name = test_file.stem
        if is_slow(test_file) and not run_slow:
            print(f"=== {name} === SKIPPED (slow, use --slow to run)")
            skipped += 1
            continue

        print(f"=== {name} ===")
        result = subprocess.run([sys.executable, str(test_file)])
        if result.returncode == 0:
            passed += 1
        else:
            # Retry once — Docker networking (mDNS) can be slow on first boot
            print(f"\n=== {name} === RETRYING (attempt 2/2)")
            result = subprocess.run([sys.executable, str(test_file)])
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1
                failures.append(name)
        print()

    total = passed + failed + skipped
    print("================================")
    print(
        f"{passed}/{total} tests passed" + (f", {skipped} skipped" if skipped else "")
    )

    if failed:
        print(f"Failed: {' '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
