"""Pure token sampling and stop checks for one tinygrad generation request.

The engine realizes logits and owns the key-value cache. These functions only
choose an id and decide when the completion is finished.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from exo.api.types import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.backends.tinygrad_weights import TinygradWeightError
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import MAX_TOKENS


def completion_token_limit(max_output_tokens: int | None) -> int:
    """Return how many completion tokens a request may emit."""
    if max_output_tokens is None:
        return MAX_TOKENS
    return max_output_tokens


def normalize_stop_strings(stop: str | list[str] | None) -> tuple[str, ...]:
    """Return the non-empty stop strings from a generation request."""
    if stop is None:
        return ()
    if isinstance(stop, str):
        if stop == "":
            return ()
        return (stop,)
    return tuple(item for item in stop if item != "")


def visible_completion_piece(
    previous_text: str,
    piece: str,
    stop_strings: Sequence[str],
) -> tuple[str, str | None]:
    """Return the piece to emit and the stop string that ended the text."""
    updated = previous_text + piece
    for stop in stop_strings:
        index = updated.find(stop)
        if index < 0:
            continue
        visible = updated[:index]
        if visible.startswith(previous_text):
            return visible[len(previous_text) :], stop
        return visible, stop
    return piece, None


def sample_token_id(
    logits: Sequence[float],
    *,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    seed: int,
    draw_index: int,
) -> int:
    """Choose a token id from one logit row.

    Temperature ``None`` or ``0`` is an argmax. A positive temperature draws
    from the tempered distribution after ``top_k`` and ``top_p``. The same
    seed and draw index always select the same id.

    Raises:
        TinygradWeightError: The runner entrypoint handles an empty logit row
            by publishing ``RunnerTerminationError``.
    """
    if not logits:
        raise TinygradWeightError("Tinygrad sampling received no logits")
    if temperature is None or temperature <= 0.0:
        return _argmax(logits)
    weights = _tempered_weights(logits, temperature)
    weights = _apply_top_k(weights, top_k)
    weights = _apply_top_p(weights, top_p)
    return _categorical(weights, seed, draw_index)


def generation_usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    """Build the usage object attached to the finishing token."""
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


def generation_stats(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prefill_seconds: float,
    generation_seconds: float,
    parameter_byte_count: int | None,
) -> GenerationStats:
    """Build timings for the finishing token."""
    return GenerationStats(
        prompt_tps=_tokens_per_second(prompt_tokens, prefill_seconds),
        generation_tps=_tokens_per_second(completion_tokens, generation_seconds),
        prompt_tokens=prompt_tokens,
        generation_tokens=completion_tokens,
        peak_memory_usage=Memory.from_bytes(parameter_byte_count or 0),
    )


def _argmax(logits: Sequence[float]) -> int:
    best_index = 0
    best_value = logits[0]
    for index, value in enumerate(logits):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def _finite(value: float) -> bool:
    return value == value and value != float("inf") and value != float("-inf")


def _tempered_weights(logits: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    finite_values = [value for value in scaled if _finite(value)]
    if not finite_values:
        share = 1.0 / float(len(scaled))
        return [share for _value in scaled]
    peak = max(finite_values)
    exponents: list[float] = []
    for value in scaled:
        if not _finite(value):
            exponents.append(0.0)
            continue
        exponents.append(math.exp(value - peak))
    total = sum(exponents)
    if total == 0.0:
        share = 1.0 / float(len(scaled))
        return [share for _value in scaled]
    return [value / total for value in exponents]


def _apply_top_k(weights: list[float], top_k: int | None) -> list[float]:
    if top_k is None or top_k <= 0 or top_k >= len(weights):
        return weights
    kept = [False for _weight in weights]
    for _choice in range(top_k):
        best_index = -1
        best_weight = 0.0
        for index, weight in enumerate(weights):
            if kept[index]:
                continue
            if best_index < 0 or weight > best_weight:
                best_weight = weight
                best_index = index
        if best_index >= 0:
            kept[best_index] = True
    return [weight if kept[index] else 0.0 for index, weight in enumerate(weights)]


def _apply_top_p(weights: list[float], top_p: float | None) -> list[float]:
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return weights
    total = sum(weights)
    if total <= 0.0:
        return weights

    def weight_at(index: int) -> float:
        return weights[index]

    ordered = sorted(range(len(weights)), key=weight_at, reverse=True)
    kept = [0.0 for _weight in weights]
    running = 0.0
    for index in ordered:
        if running >= top_p and any(value > 0.0 for value in kept):
            break
        kept[index] = weights[index]
        running += weights[index] / total
    return kept


def _categorical(weights: Sequence[float], seed: int, draw_index: int) -> int:
    total = sum(weights)
    if total <= 0.0:
        return 0
    mixed_seed = (seed * 1_000_003 + draw_index) & 0xFFFFFFFF
    draw = random.Random(mixed_seed).random() * total
    running = 0.0
    last_index = 0
    for index, weight in enumerate(weights):
        running += weight
        last_index = index
        if running >= draw:
            return index
    return last_index


def _tokens_per_second(token_count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return float(token_count) / seconds
