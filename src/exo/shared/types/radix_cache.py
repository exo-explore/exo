import hashlib
from typing import Any

from pydantic import ConfigDict, Field

from exo.utils.pydantic_ext import FrozenModel


def compute_token_hash(tokens: tuple[int, ...] | list[int], parent_hash: str = "") -> str:
    """Computes a deterministic SHA256 content-addressable hash for a token sequence."""
    hasher = hashlib.sha256()
    if parent_hash:
        hasher.update(parent_hash.encode("utf-8"))
    for t in tokens:
        hasher.update(t.to_bytes(4, byteorder="big", signed=True))
    return hasher.hexdigest()


class RadixNode(FrozenModel):
    """Represents a compressed radix prefix tree node with content-addressed hash."""

    model_config = ConfigDict(frozen=True, strict=True)

    node_id: str
    parent_id: str | None = None
    token_subsequence: tuple[int, ...] = Field(default_factory=tuple)
    block_ids: tuple[str, ...] = Field(default_factory=tuple)
    node_hash: str = ""
    children: dict[int, str] = Field(default_factory=dict)  # token -> child_node_id
    reference_count: int = 0
    last_accessed_timestamp: float = 0.0
    is_pinned: bool = False

    @property
    def token_length(self) -> int:
        return len(self.token_subsequence)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixMatchResult(FrozenModel):
    """Results from matching a query prompt against the Radix cache tree."""

    model_config = ConfigDict(frozen=True, strict=True)

    matched_tokens_count: int
    matched_block_ids: tuple[str, ...]
    last_matched_node_id: str
    remaining_tokens: tuple[int, ...]
    is_full_match: bool


class RadixTelemetry(FrozenModel):
    """Telemetry capturing token avoidance and Radix prefix cache hit rates."""

    model_config = ConfigDict(frozen=True, strict=True)

    total_queries: int = 0
    total_prefix_hits: int = 0
    total_tokens_avoided: int = 0
    total_tokens_computed: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_prefix_hits / self.total_queries

    @property
    def token_savings_ratio(self) -> float:
        total = self.total_tokens_avoided + self.total_tokens_computed
        if total == 0:
            return 0.0
        return self.total_tokens_avoided / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_prefix_hits": self.total_prefix_hits,
            "hit_rate_pct": round(self.hit_rate * 100.0, 2),
            "total_tokens_avoided": self.total_tokens_avoided,
            "total_tokens_computed": self.total_tokens_computed,
            "token_savings_pct": round(self.token_savings_ratio * 100.0, 2),
        }
