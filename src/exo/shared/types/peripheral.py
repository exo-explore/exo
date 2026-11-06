from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

PeripheralRole = Literal[
    "compute_heavy",
    "tokenization_gateway",
    "sampling_grammar_processor",
    "tiered_cache_swapper",
]


class PeripheralCapability(FrozenModel):
    """Capability specification for peripheral and edge specialization."""

    node_id: NodeId
    roles: list[PeripheralRole] = Field(default_factory=list)
    available_memory_megabytes: int
    has_hardware_accelerator: bool
    network_bandwidth_megabits_per_second: float = 100.0

    @property
    def is_peripheral_specialist(self) -> bool:
        """Determines if the node is best suited for non-GEMM peripheral tasks."""
        return not self.has_hardware_accelerator or self.available_memory_megabytes < 4096


class SamplingConfig(FrozenModel):
    """Pure mathematical sampling configuration for detached peripheral nodes."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None


GrammarType = Literal["none", "json", "regex"]


class GrammarConstraint(FrozenModel):
    """Grammar constraint definition for structured decoding."""

    grammar_type: GrammarType = "none"
    schema_definition: str | None = None


class TruncatedLogitsEntry(FrozenModel):
    """Sparse logits entry for low-bandwidth transfer between compute and peripheral nodes."""

    token_index: int
    logit_value: float


class LogitsPayload(FrozenModel):
    """Wire payload carrying logits from a compute node to a peripheral sampling node."""

    sequence_id: str
    step: int
    vocabulary_size: int
    # Sparse entries if truncated for weak networks, or empty if full dense array is transferred
    sparse_entries: Sequence[TruncatedLogitsEntry] = Field(default_factory=tuple)


class SampledTokenResult(FrozenModel):
    """Result emitted by the peripheral sampling processor."""

    token_id: int
    logprob: float
    is_terminal: bool
    decoded_text: str | None = None
