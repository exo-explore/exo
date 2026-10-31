"""Penalty processors count the tokens exo's penalties are defined over, in both engines:
presence and frequency penalties the generated tokens, the repetition penalty the prompt and
the generated tokens, each including the token being fed in this step."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    TokenBuffer,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from mlx_lm.sample_utils import make_logits_processors

from exo.worker.engines.mlx.generator.generate import (
    PenaltyProcessor,
    penalty_processors,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    _patched_step,  # pyright: ignore[reportPrivateUsage]
)

VOCAB = 16
PROMPT = [11, 3, 7, 3, 9, 14, 2, 5]
GENERATED = [4, 4, 12, 1, 3, 15]


def recorder(seen: list[list[int]]) -> Callable[[mx.array, mx.array], mx.array]:
    def record(tokens: mx.array, logits: mx.array) -> mx.array:
        seen.append([int(t) for t in cast(list[int], tokens.tolist())])
        return logits

    return record


@pytest.mark.parametrize("prompt_in_history", [1, 2, len(PROMPT)])
@pytest.mark.parametrize("context_size", [3, 20, 0])
def test_processors_see_generated_tokens_or_prompt_and_generated_tokens(
    prompt_in_history: int, context_size: int
) -> None:
    generated_seen: list[list[int]] = []
    everything_seen: list[list[int]] = []
    generated_only = PenaltyProcessor(
        recorder(generated_seen), PROMPT, include_prompt=False, context_size=20
    )
    with_prompt = PenaltyProcessor(
        recorder(everything_seen),
        PROMPT,
        include_prompt=True,
        context_size=context_size,
    )
    logits = mx.zeros((1, VOCAB))
    # An engine hands processors the end of the prompt, then every token generated so far,
    # the latest one (the token being fed) included.
    for step in range(len(GENERATED) + 1):
        history = mx.array(PROMPT[-prompt_in_history:] + GENERATED[:step])
        generated_only(history, logits)
        with_prompt(history, logits)

    assert generated_seen == [GENERATED[:step] for step in range(len(GENERATED) + 1)]
    for step, seen in enumerate(everything_seen):
        everything = PROMPT + GENERATED[:step]
        # mlx-lm's repetition penalty reads the last context_size tokens (all if <= 0)
        window = everything[-context_size:] if context_size > 0 else everything
        assert seen[-len(window) :] == window
        assert len(seen) >= len(window)
        assert seen == everything[len(everything) - len(seen) :]


@pytest.mark.parametrize(
    "penalties",
    [
        (1.1, 20, 1.5, 0.5),
        (1.0, 20, 1.5, None),
        (None, 20, 0.0, 0.3),
        (0.9, 0, None, None),
        (None, None, None, None),
    ],
)
def test_penalty_values_and_order_are_mlx_lm_s(
    penalties: tuple[float | None, int | None, float | None, float | None],
) -> None:
    repetition, context, presence, frequency = penalties
    reference = make_logits_processors(
        repetition_penalty=repetition,
        repetition_context_size=context if context is not None else 20,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    processors = penalty_processors(
        prompt=PROMPT,
        repetition_penalty=repetition,
        repetition_context_size=context,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    kinds = [
        kind
        for kind, penalty in (
            ("repetition", repetition),
            ("presence", presence),
            ("frequency", frequency),
        )
        if penalty is not None and penalty != 0
    ]
    assert len(processors) == len(reference) == len(kinds)
    # Each processor applies mlx-lm's penalty to the tokens it counts: the prompt and the
    # generated tokens for repetition, the generated tokens for presence and frequency.
    for processor, mlx_processor, kind in zip(
        processors, reference, kinds, strict=True
    ):
        processor(mx.array(PROMPT), mx.zeros((1, VOCAB)))  # the first token's call
        counted = PROMPT + GENERATED if kind == "repetition" else GENERATED
        logits = mx.random.normal((1, VOCAB), key=mx.random.key(1))
        expected = mlx_processor(mx.array(counted), mx.array(logits))
        got = processor(mx.array(PROMPT + GENERATED), mx.array(logits))
        assert mx.array_equal(got, expected).item()


def test_batch_step_gives_processors_the_token_being_fed() -> None:
    seen: list[list[int]] = []

    class Model:
        def __call__(self, inputs: mx.array, cache: list[object]) -> mx.array:
            # the next token is always 9
            return (
                mx.zeros((inputs.shape[0], inputs.shape[1], VOCAB)).at[..., 9].add(1.0)
            )

    def greedy(logprobs: mx.array) -> mx.array:
        return mx.argmax(logprobs, axis=-1)

    in_cache = [5, 6]
    batch = SimpleNamespace(
        model=Model(),
        prompt_cache=[],
        logits_processors=[(recorder(seen),)],
        samplers=[greedy],
        fallback_sampler=greedy,
        uids=[0],
        tokens=[list(in_cache)],
        _token_context=[TokenBuffer(list(in_cache))],
        _next_tokens=mx.array([7]),
        _next_logprobs=mx.zeros((1, VOCAB)),
    )

    for _ in range(3):
        _patched_step(batch)  # pyright: ignore[reportArgumentType]

    # each step's processors see the tokens in the cache and the one being fed
    assert seen == [[5, 6, 7], [5, 6, 7, 9], [5, 6, 7, 9, 9]]
    assert cast(list[list[int]], batch.tokens) == [[5, 6, 7, 9, 9]]
