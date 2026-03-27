import sys
from pathlib import Path

import pytest


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip collection of image tests on non-macOS platforms (requires mflux)."""
    if sys.platform != "darwin":
        return True
    return None
