import ast
import re

from exo.security_gate.issue import Issue

_SKIP_PATH_KEYWORDS = ("config", "constants", "settings", "defaults", "test")

_BIND_PATTERN = re.compile(
    r'(?i)(?:bind|host|listen|address)\s*[:=]\s*["\']0\.0\.0\.0["\']'
)
_PRIVATE_IP_PATTERN = re.compile(
    r'["\'](?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})["\']'
)


def _should_skip_file(filepath: str) -> bool:
    path_lower = filepath.lower()
    return any(kw in path_lower for kw in _SKIP_PATH_KEYWORDS)


def check_network_exposure(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    if _should_skip_file(filepath):
        return []

    issues: list[Issue] = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip comment lines
        if stripped.startswith("#"):
            continue

        if _BIND_PATTERN.search(line):
            issues.append(
                Issue(
                    filepath=filepath,
                    lineno=lineno,
                    check_id="NETWORK_EXPOSURE",
                    message="Binding to 0.0.0.0 exposes service to all interfaces — use 127.0.0.1 or make configurable",
                    severity="block",
                )
            )
        elif _PRIVATE_IP_PATTERN.search(line):
            issues.append(
                Issue(
                    filepath=filepath,
                    lineno=lineno,
                    check_id="NETWORK_EXPOSURE",
                    message="Hardcoded private IP address — use configuration or environment variable instead",
                    severity="advisory",
                )
            )

    return issues
