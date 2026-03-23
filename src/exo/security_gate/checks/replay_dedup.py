"""Security gate check: Sentinel Replay-Attack Dedup.

Detects apply() / event-handler functions that lack deduplication guards,
which can amplify the damage of a replayed or injected event.

A compliant apply() must contain at least ONE of:
  - early-return if state.last_event_applied_idx >= event.idx  (exo pattern)
  - membership test against a ``seen`` / ``processed`` / ``dedup`` set
  - membership test against any set/frozenset typed variable

Advisory: missing dedup does not block the commit but flags the risk.
"""

from __future__ import annotations

import ast

from exo.security_gate.issue import Issue

# ── helpers ────────────────────────────────────────────────────────────────────

_DEDUP_ATTR_KEYWORDS = frozenset({"seen", "processed", "dedup", "visited", "applied"})


def _looks_like_idx_compare(node: ast.Compare) -> bool:
    """Return True for patterns like ``state.X >= event.Y`` or ``X >= Y``."""
    if len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], (ast.GtE, ast.Gt, ast.Eq)):
        return False
    left, right = node.left, node.comparators[0]
    # attr compare: state.last_event_applied_idx >= event.idx
    if isinstance(left, ast.Attribute) and isinstance(right, ast.Attribute):
        return True
    # name compare: idx >= seen_idx
    return isinstance(left, ast.Name) and isinstance(right, ast.Name)


def _looks_like_dedup_test(node: ast.Compare) -> bool:
    """Return True for ``x in seen`` / ``x in processed_ids`` patterns."""
    if len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], ast.In):
        return False
    container = node.comparators[0]
    if isinstance(container, ast.Name) and any(kw in container.id.lower() for kw in _DEDUP_ATTR_KEYWORDS):
        return True
    return isinstance(container, ast.Attribute) and any(kw in container.attr.lower() for kw in _DEDUP_ATTR_KEYWORDS)


def _has_dedup_guard(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Walk the direct body of ``func_node`` looking for a dedup guard.

    A dedup guard is:
    1. An ``if`` with an idx-compare condition that returns early, OR
    2. An ``if`` with a set-membership test (``x in seen_ids``), OR
    3. Any ``in`` expression whose container has a dedup-like name.
    """
    for stmt in func_node.body:
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test

        # Handle ``not (x in seen)`` or ``x not in seen``
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand

        if isinstance(test, ast.Compare) and (_looks_like_idx_compare(test) or _looks_like_dedup_test(test)):
            # Must contain a return/raise to qualify as an early-return guard
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Return, ast.Raise)):
                    return True

        # BoolOp: ``if state.idx >= event.idx or ...``
        if isinstance(test, ast.BoolOp):
            for value in test.values:
                if isinstance(value, ast.Compare) and (
                    _looks_like_idx_compare(value) or _looks_like_dedup_test(value)
                ):
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.Return, ast.Raise)):
                            return True

    return False


def _is_apply_like(name: str) -> bool:
    """Return True for function names that match the apply-event pattern."""
    return name in ("apply", "apply_event", "handle_event", "process_event", "on_event")


# ── public check ───────────────────────────────────────────────────────────────


def check_replay_dedup(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    """Flag apply-like functions that lack a deduplication early-return guard."""
    issues: list[Issue] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_apply_like(node.name):
            continue
        # Only flag if the function looks like it processes events:
        # it must accept at least 2 positional args (state/self + event)
        total_args = len(node.args.args) + len(node.args.posonlyargs)
        if total_args < 2:
            continue
        if _has_dedup_guard(node):
            continue
        issues.append(
            Issue(
                filepath=filepath,
                lineno=node.lineno,
                check_id="REPLAY_DEDUP",
                message=(
                    f"apply-like function '{node.name}' has no deduplication guard — "
                    "a replayed event will be applied twice. "
                    "Add: ``if state.last_event_applied_idx >= event.idx: return state``"
                ),
                severity="advisory",
            )
        )

    return issues
