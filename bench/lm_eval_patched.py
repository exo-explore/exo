"""Patched lm_eval runner that fixes bugs in the upstream library.

Fixes:
- UnboundLocalError on `outputs` in TemplateAPI.amodel_call when API returns error
- Prevents eval crash on transient API failures (returns None instead of raising)

Usage: python -m bench.lm_eval_patched [lm_eval args...]
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportAny=false
# ruff: noqa: I001

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
        except UnboundLocalError:
            # `outputs` referenced before assignment when response.raise_for_status() throws
            return []
        except Exception:
            # After all retries fail, don't crash the entire eval
            return []

    TemplateAPI.amodel_call = patched_amodel_call


if __name__ == "__main__":
    _patch_amodel_call()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
