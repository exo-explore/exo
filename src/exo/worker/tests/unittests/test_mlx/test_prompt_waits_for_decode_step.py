from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.generate import PromptProcessingBatch

from exo.worker.engines.mlx.patches import opt_batch_gen


def test_prompt_waits_for_the_decode_step_before_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def synchronize(stream: object = None) -> None:
        calls.append("wait")

    def original_prompt(batch: PromptProcessingBatch, tokens: list[list[int]]) -> None:
        calls.append(f"prompt {len(tokens)}")

    monkeypatch.setattr(mx, "synchronize", synchronize)
    monkeypatch.setattr(opt_batch_gen, "_original_prompt", original_prompt)
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)

    opt_batch_gen._prompt_after_decode_step(batch, [[1, 2, 3]])  # pyright: ignore[reportPrivateUsage]
    # mlx-lm calls it every step, mostly with nothing to process: no need to wait then
    opt_batch_gen._prompt_after_decode_step(batch, [])  # pyright: ignore[reportPrivateUsage]

    assert calls == ["wait", "prompt 1", "prompt 0"]


def test_batch_gen_patch_installs_it() -> None:
    opt_batch_gen.apply_batch_gen_patch()
    installed = cast(object, PromptProcessingBatch.prompt)
    assert installed is opt_batch_gen._prompt_after_decode_step  # pyright: ignore[reportPrivateUsage]
