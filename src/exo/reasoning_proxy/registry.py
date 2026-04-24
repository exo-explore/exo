import asyncio
import logging
from typing import cast, get_args

import httpx

from exo.reasoning_proxy._helpers import as_dict, as_list, dict_get_str
from exo.shared.types.text_generation import ReasoningDialect

logger = logging.getLogger(__name__)


class DialectRegistry:
    def __init__(self, upstream: str, client: httpx.AsyncClient) -> None:
        self._upstream = upstream.rstrip("/")
        self._client = client
        self._by_model: dict[str, ReasoningDialect] = {}
        self._unknown_logged: set[str] = set()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def refresh(self) -> None:
        await self._fetch()

    async def _fetch(self) -> None:
        try:
            resp = await self._client.get(f"{self._upstream}/v1/models", timeout=10.0)
            resp.raise_for_status()
            body = as_dict(cast(object, resp.json()))
            if body is None:
                return
            data = as_list(body.get("data")) or []
            updated: dict[str, ReasoningDialect] = {}
            for entry_raw in data:
                entry = as_dict(entry_raw)
                if entry is None:
                    continue
                model_id = dict_get_str(entry, "id")
                dialect_raw = entry.get("reasoning_dialect", "none")
                if model_id is not None:
                    updated[model_id] = _coerce_dialect(dialect_raw)
            self._by_model = updated
            self._initialized = True
            logger.info(
                "Loaded %d model dialects from %s", len(updated), self._upstream
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch /v1/models from %s: %s", self._upstream, exc
            )

    async def resolve(self, model_id: str) -> ReasoningDialect:
        async with self._lock:
            if not self._initialized:
                await self._fetch()
            if model_id in self._by_model:
                return self._by_model[model_id]
            await self._fetch()
            if model_id in self._by_model:
                return self._by_model[model_id]
            if model_id not in self._unknown_logged:
                logger.info(
                    "No dialect declared for model %s; passing through", model_id
                )
                self._unknown_logged.add(model_id)
            return "none"


_VALID_DIALECTS: frozenset[str] = frozenset(get_args(ReasoningDialect))


def _coerce_dialect(value: object) -> ReasoningDialect:
    if isinstance(value, str) and value in _VALID_DIALECTS:
        return cast(ReasoningDialect, value)
    return "none"
