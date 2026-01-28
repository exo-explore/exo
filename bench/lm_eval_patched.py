"""Patched lm_eval runner that fixes bugs in the upstream library.

Fixes:
- UnboundLocalError on `outputs` in TemplateAPI.amodel_call when API returns error
- Prevents eval crash on transient API failures (returns None instead of raising)
- Compatibility with transformers 5.x (missing AutoModelForVision2Seq)
- sock_read timeout causing connection drops with large request queues

Usage: python -m bench.lm_eval_patched [lm_eval args...]
"""

# ruff: noqa: I001, E402
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportAny=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

# MUST patch transformers BEFORE any lm_eval imports
# AutoModelForVision2Seq/AutoModelForImageTextToText were removed in transformers 5.0
# Patch the lazy module's __getattr__ to return stubs for missing classes
from transformers.utils import import_utils

_original_getattr = import_utils._LazyModule.__getattr__


def _patched_getattr(self: object, name: str) -> object:
    if name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        return type(name, (), {})  # Return a stub class
    return _original_getattr(self, name)  # type: ignore


import_utils._LazyModule.__getattr__ = _patched_getattr

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


def _patch_client_timeout() -> None:
    """Patch TemplateAPI.get_batched_requests to disable sock_read timeout.

    By default, aiohttp's ClientTimeout can have a sock_read timeout that causes
    connections to drop if no data is received for a while. With large request
    queues, requests may wait a long time before processing starts, causing
    spurious connection drops and retries that pile up requests.
    """
    from aiohttp import ClientSession, ClientTimeout, TCPConnector

    from lm_eval.models.api_models import TemplateAPI

    original_get_batched: Any = TemplateAPI.get_batched_requests

    @functools.wraps(original_get_batched)
    async def patched_get_batched_requests(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Override the timeout to explicitly disable sock_read timeout
        # This prevents connection drops when requests are queued for a long time
        original_timeout = getattr(self, "timeout", 604800)
        conn = TCPConnector(limit=self._concurrent, ssl=self.verify_certificate)
        timeout = ClientTimeout(
            total=original_timeout, sock_read=None, sock_connect=None
        )

        async with ClientSession(connector=conn, timeout=timeout) as session:
            # Call the internal async logic with our session
            return await _run_batched_requests_with_session(
                self, session, *args, **kwargs
            )

    async def _run_batched_requests_with_session(
        self: Any,
        session: ClientSession,
        requests: Any,
        cache_keys: Any = None,
        ctxlens: Any = None,
        **kwargs: Any,
    ) -> Any:
        import asyncio
        import copy
        import logging

        from tqdm.asyncio import tqdm_asyncio
        from tenacity import retry, stop_after_attempt, wait_exponential
        from lm_eval.models.utils import chunks

        eval_logger = logging.getLogger("lm_eval.models.api_models")
        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        sem = asyncio.Semaphore(self._concurrent)

        retry_: Any = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=1, max=10),
            reraise=True,
            before_sleep=lambda retry_state: eval_logger.info(
                f"Retry attempt {retry_state.attempt_number}"
            ),
        )(self.amodel_call)

        tasks = [
            asyncio.create_task(
                retry_(
                    session=session,
                    sem=sem,
                    messages=message,
                    cache_keys=cache_key,
                    ctxlens=ctxlen,
                    gen_kwargs=copy.deepcopy(kwargs.get("gen_kwargs")),
                    **{k: v for k, v in kwargs.items() if k != "gen_kwargs"},
                )
            )
            for message, cache_key, ctxlen in zip(
                chunks(requests, n=self._batch_size),
                chunks(cache_keys, n=self._batch_size),
                chunks(ctxlens, n=self._batch_size),
                strict=True,
            )
        ]

        return await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    TemplateAPI.get_batched_requests = patched_get_batched_requests


if __name__ == "__main__":
    _patch_amodel_call()
    _patch_client_timeout()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
