"""Tests for the ``EXO_SPEC_DIAG`` env-gated diagnostic helper.

These pin the contract that ``_spec_diag`` is a no-op unless the
``_SPEC_DIAG_ENABLED`` flag (resolved from ``EXO_SPEC_DIAG`` at module
import time) is set. Codex flagged on PR #21 round 3 that several
``[spec-diag] logger.info(...)`` calls in ``generate.py`` were running
unconditionally on every request even though the diagnostics were
intended to be env-gated. After the fix those call sites route
through ``_spec_diag``; this test exercises both states (off / on) and
proves ``generate.py`` reuses the same helper.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Final

import pytest

from exo.worker.engines.mlx.generator import (
    generate,
    pipelined_drafter,
)


class _Recorder:
    """Captures ``info(message)`` calls; substituted in for
    ``pipelined_drafter._diag_logger`` during the gated test.
    """

    def __init__(self) -> None:
        self.entries: list[str] = []

    def info(self, message: str) -> None:
        self.entries.append(message)


def test_spec_diag_is_no_op_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(pipelined_drafter, "_SPEC_DIAG_ENABLED", False)
    monkeypatch.setattr(pipelined_drafter, "_diag_logger", recorder)

    pipelined_drafter._spec_diag("rank 0: must not appear when disabled")

    assert recorder.entries == [], (
        "_spec_diag must short-circuit before touching the logger when "
        "EXO_SPEC_DIAG is unset; got "
        f"{recorder.entries!r}"
    )


def test_spec_diag_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(pipelined_drafter, "_SPEC_DIAG_ENABLED", True)
    monkeypatch.setattr(pipelined_drafter, "_diag_logger", recorder)

    expected: Final[str] = "rank 0: enabled-message"
    pipelined_drafter._spec_diag(expected)

    assert recorder.entries == [expected], (
        "_spec_diag must forward the message to loguru when "
        "EXO_SPEC_DIAG is set; got "
        f"{recorder.entries!r}"
    )


def test_generate_reuses_pipelined_drafter_spec_diag() -> None:
    """``generate.py`` must import ``_spec_diag`` from
    ``pipelined_drafter`` so the four call sites previously written
    as ``logger.info(f\"[spec-diag] ...\")`` are now gated by the
    same helper as ``pipelined_drafter``'s diagnostics. This pins
    that the gating is in place at the symbol level: same function
    object on both modules, no parallel definition.
    """
    assert generate._spec_diag is pipelined_drafter._spec_diag, (
        "generate.py must reuse pipelined_drafter._spec_diag so "
        "EXO_SPEC_DIAG gates ALL spec-decode diagnostic logs"
    )
