"""Shared types for the autonomous refactor agent."""
from __future__ import annotations

from typing import Literal, final

from pydantic import BaseModel


@final
class RefactorCandidate(BaseModel, frozen=True, strict=True):
    """A potential refactor proposed by Qwen during pre-scan."""
    filepath: str
    lineno: int
    category: Literal[
        "optional_syntax",      # Optional[X] -> X | None
        "union_syntax",         # Union[X, Y] -> X | Y
        "missing_annotation",   # unannotated parameter or return
        "redundant_pass",       # empty class/func body with redundant pass
        "bare_exception",       # bare except: -> except Exception:
        "f_string_format",      # "{}".format(x) -> f"{x}"
        "unnecessary_else",     # else after return/raise
    ]
    description: str
    source_line: str


@final
class RefactorInstruction(BaseModel, frozen=True, strict=True):
    """Opus-approved instruction for Sonnet to execute."""
    filepath: str
    lineno: int
    category: str
    instruction: str  # English description of the exact change
    source_line: str


@final
class CorrectionNote(BaseModel, frozen=True, strict=True):
    """Opus analysis of why Sonnet's diff failed."""
    instruction_filepath: str
    instruction_lineno: int
    failure_reason: str
    correction: str  # What Sonnet should do differently


@final
class RefactorResult(BaseModel, frozen=True, strict=True):
    """Outcome of attempting one instruction."""
    instruction: RefactorInstruction
    status: Literal["applied", "skipped", "dropped"]
    reason: str
    iterations: int
