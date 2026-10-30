import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.family import PromptEncoder


def _encoder(exact_think_tags: bool) -> PromptEncoder:
    return PromptEncoder(
        applies_to=lambda _task: True,
        encode=lambda _messages, _task: "",
        exact_think_tags=exact_think_tags,
    )


@pytest.mark.parametrize(
    ("encoder", "expected"),
    [(None, False), (_encoder(False), False), (_encoder(True), True)],
)
def test_think_tags_are_kept_only_when_the_encoder_says_so(
    monkeypatch: pytest.MonkeyPatch, encoder: PromptEncoder | None, expected: bool
) -> None:
    seen: list[str | None] = []

    def lookup(
        _task: TextGenerationTaskParams, model_type: str | None = None
    ) -> PromptEncoder | None:
        seen.append(model_type)
        return encoder

    monkeypatch.setattr(registry, "prompt_encoder_for", lookup)
    task = TextGenerationTaskParams(
        model=ModelId("some/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )
    assert registry.keeps_think_tags(task, "some_type") is expected
    assert seen == ["some_type"]
