"""BenchSession — wires together cluster + client + instance + tokenizer.

Holds the ``EcoSession``, a deployed ``ClusterInfo``, an ``ExoClient`` for
the cluster's primary endpoint, and (for benchmarks that need exact-token
prompts) a lazily-constructed :class:`PromptSizer`.

Benchmarks consume this via :func:`bench.lib.cluster.managed_instance`,
which yields a populated ``BenchSession``. Library helpers (e.g.
``context_scaling.run``) take a ``BenchSession`` and never reach for
global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from exo_tools.client import ExoClient
from exo_tools.cluster import ClusterInfo, EcoSession, make_client_from_url

from .prompt import PromptSizer, load_tokenizer_for_bench


@dataclass
class BenchSession:
    """Bundle of cluster + client + (optional) instance for benchmarks."""

    cluster: ClusterInfo
    eco: EcoSession
    instance_id: str | None = None
    model_id: str | None = None
    full_model_id: str | None = None
    _prompt_sizer: PromptSizer | None = field(default=None, repr=False)

    @property
    def client(self) -> ExoClient:
        return make_client_from_url(self.cluster.api_url)

    def state(self) -> dict[str, Any]:
        raw: Any = self.client.request_json("GET", "/state")  # type: ignore[reportAny]
        if isinstance(raw, dict):
            return cast("dict[str, Any]", raw)
        return {}

    def instances(self) -> dict[str, Any]:
        result: Any = self.state().get("instances", {})  # type: ignore[reportAny]
        if isinstance(result, dict):
            return cast("dict[str, Any]", result)
        return {}

    def get_prompt_sizer(self) -> PromptSizer:
        """Return a cached :class:`PromptSizer` for ``self.full_model_id``.

        Loaded lazily because tokenizer load is expensive and not every
        benchmark needs prompt sizing.
        """
        if self._prompt_sizer is not None:
            return self._prompt_sizer
        if self.full_model_id is None:
            raise RuntimeError(
                "BenchSession.full_model_id is not set; cannot build a PromptSizer."
            )
        tokenizer = load_tokenizer_for_bench(self.full_model_id)
        self._prompt_sizer = PromptSizer(tokenizer)
        return self._prompt_sizer
