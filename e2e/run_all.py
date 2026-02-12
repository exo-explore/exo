#!/usr/bin/env python3
"""Discovers and runs all E2E tests in e2e/test_*.py."""

import subprocess
import sys
from pathlib import Path

E2E_DIR = Path(__file__).parent.resolve()


def main():
    test_files = sorted(E2E_DIR.glob("test_*.py"))
    if not test_files:
        print("No test files found")
        sys.exit(1)

    passed = 0
    failed = 0
    failures = []

    for test_file in test_files:
        name = test_file.stem
        print(f"=== {name} ===")
        result = subprocess.run([sys.executable, str(test_file)])
        if result.returncode == 0:
            passed += 1
        else:
            failed += 1
            failures.append(name)
        print()

    total = passed + failed
    print("================================")
    print(f"{passed}/{total} tests passed")

    if failed:
        print(f"Failed: {' '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
