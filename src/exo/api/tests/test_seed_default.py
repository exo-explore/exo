"""The master draws a sampling seed when a request doesn't set one.

Runners fall back to a fixed seed when the task carries none, which makes a
retry of a failed generation replay the exact same failure — so
_send_text_generation_with_images must draw a seed for normal requests while
preserving explicit seeds and leaving bench runs deterministic.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from exo.api.main import API
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _task_params(**overrides: object) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test-org/test-model"),
        input=[InputMessage(role="user", content="hi")],
        **overrides,  # pyright: ignore[reportAny]
    )


async def _send_command(task_params: TextGenerationTaskParams):
    api = SimpleNamespace(_send=AsyncMock())
    return await API._send_text_generation_with_images(api, task_params)  # pyright: ignore[reportPrivateUsage, reportArgumentType]


async def test_missing_seed_gets_drawn():
    command = await _send_command(_task_params())
    assert command.task_params.seed is not None


async def test_explicit_seed_preserved():
    command = await _send_command(_task_params(seed=1234))
    assert command.task_params.seed == 1234


async def test_bench_keeps_seed_unset():
    command = await _send_command(_task_params(bench=True))
    assert command.task_params.seed is None
