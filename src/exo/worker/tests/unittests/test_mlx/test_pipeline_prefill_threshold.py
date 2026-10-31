"""When pipeline parallelism prefills a prompt in overlapping chunks.

pipeline_parallel_prefill splits a prompt into chunks of prefill_step_size / world_size
(at most a quarter) and overlaps them across the ranks; with fewer than two chunks there
is nothing to overlap. The threshold counts the whole prompt: prefill() is given the
prompt without its last token, which the first decode step processes.
"""

import pytest

from exo.worker.engines.mlx.generator.generate import (
    pipeline_prefill_chunk_size,
    uses_pipeline_prefill,
)


@pytest.mark.parametrize(
    ("world_size", "chunk"), [(2, 2048), (3, 1365), (4, 1024), (8, 1024)]
)
def test_chunk_size(world_size: int, chunk: int) -> None:
    assert pipeline_prefill_chunk_size(4096, world_size) == chunk


@pytest.mark.parametrize(
    ("prompt_length", "world_size", "pipelined"),
    [
        (4095, 2, False),
        # a 4096-token prompt is two chunks of 2048 (it used to run through the ranks
        # one after the other: prefill() sees 4095 tokens)
        (4096, 2, True),
        (6144, 2, True),
        (2729, 3, False),
        (2730, 3, True),
        (2047, 4, False),
        (2048, 4, True),
    ],
)
def test_pipelined_prefill_starts_at_two_chunks(
    prompt_length: int, world_size: int, pipelined: bool
) -> None:
    assert uses_pipeline_prefill(prompt_length, 4096, world_size) == pipelined
