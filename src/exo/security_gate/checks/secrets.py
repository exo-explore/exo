import ast
import binascii
import math
import re

from exo.security_gate.issue import Issue

# Patterns that indicate a line contains a secret
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_key", re.compile(r"(?:AKIA|A3T)[A-Z0-9]{16,}")),
    (
        "generic_secret_assignment",
        re.compile(
            r'(?i)(?:secret|password|passwd|token|api_key|apikey|auth_token|access_token|private_key)\s*[:=]\s*["\'][^"\']{8,}["\']'
        ),
    ),
    (
        "private_key_header",
        re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----"),
    ),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
    ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
]

# Exclusion substrings (case-insensitive) — skip match if found in matched value
_EXCLUSION_SUBSTRINGS = ("example", "placeholder", "test", "dummy", "xxx", "000")

# Minimum length for base64 high-entropy check
_BASE64_MIN_LEN = 40
_BASE64_ENTROPY_THRESHOLD = 4.5


def _shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    freq: dict[str, int] = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


def _is_valid_base64(s: str) -> bool:
    # Must be a multiple of 4 or paddable
    padded = s + "=" * (-len(s) % 4)
    try:
        binascii.a2b_base64(padded)
        return True
    except Exception:
        return False


def _check_base64_high_entropy(line: str) -> list[tuple[int, str]]:
    """Return list of (start_col, matched_string) for high-entropy base64 literals."""
    results: list[tuple[int, str]] = []
    # Match string literals (single or double quoted)
    for m in re.finditer(r'["\']([A-Za-z0-9+/=]{40,})["\']', line):
        value = m.group(1)
        if len(value) < _BASE64_MIN_LEN:
            continue
        if not _is_valid_base64(value):
            continue
        entropy = _shannon_entropy(value)
        if entropy > _BASE64_ENTROPY_THRESHOLD:
            results.append((m.start(), value))
    return results


def _has_exclusion(matched: str) -> bool:
    lower = matched.lower()
    return any(excl in lower for excl in _EXCLUSION_SUBSTRINGS)


def check_secrets(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    issues: list[Issue] = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip pure comment lines
        if stripped.startswith("#"):
            continue
        # Skip lines containing sk-ant- (handled by api_cost_guard.py)
        if "sk-ant-" in line:
            continue

        for pattern_name, pattern in _PATTERNS:
            m = pattern.search(line)
            if m is None:
                continue
            matched = m.group(0)
            if _has_exclusion(matched):
                continue
            issues.append(
                Issue(
                    filepath=filepath,
                    lineno=lineno,
                    check_id="SECRET",
                    message=f"Possible secret ({pattern_name}): {matched[:60]}",
                    severity="block",
                )
            )
            # Only report first pattern match per line to avoid duplicates
            break

        # Base64 high-entropy check (only if no pattern already matched)
        else:
            for _col, value in _check_base64_high_entropy(line):
                if _has_exclusion(value):
                    continue
                issues.append(
                    Issue(
                        filepath=filepath,
                        lineno=lineno,
                        check_id="SECRET",
                        message=f"High-entropy base64 string literal (len={len(value)}, entropy={_shannon_entropy(value):.2f}) — possible embedded secret",
                        severity="block",
                    )
                )
                break  # one per line

    return issues
