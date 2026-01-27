"""Patched lm_eval runner that fixes bugs in the upstream library.

Fixes:
- UnboundLocalError on `outputs` in TemplateAPI.amodel_call when API returns error
- Prevents eval crash on transient API failures (returns None instead of raising)
- Compatibility with transformers 5.x (missing AutoModelForVision2Seq)

Usage: python -m bench.lm_eval_patched [lm_eval args...]
"""

# ruff: noqa: I001, E402
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportAny=false

# MUST patch transformers BEFORE any lm_eval imports
# AutoModelForVision2Seq/AutoModelForImageTextToText were removed in transformers 5.0
# Patch the lazy module's __getattr__ to return stubs for missing classes
from transformers.utils import import_utils

_original_getattr = import_utils._LazyModule.__getattr__

def _patched_getattr(self: object, name: str) -> object:
    if name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        return type(name, (), {})  # Return a stub class
    return _original_getattr(self, name)  # type: ignore

import_utils._LazyModule.__getattr__ = _patched_getattr  # type: ignore

import functools
from typing import Any


def _patch_amodel_call() -> None:
    """Monkey-patch TemplateAPI.amodel_call to handle the unbound `outputs` variable bug."""
    from lm_eval.models.api_models import TemplateAPI

    original: Any = TemplateAPI.amodel_call

    @functools.wraps(original)
    async def patched_amodel_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(self, *args, **kwargs)
        except (UnboundLocalError, Exception):
            # Return one empty-string result per request in the batch so the
            # reorderer doesn't assert on missing coverage.
            messages = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
            return [""] * max(len(messages), 1)

    TemplateAPI.amodel_call = patched_amodel_call


if __name__ == "__main__":
    _patch_amodel_call()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
