"""The generic per-family hooks: they run only for an architecture that sets them, and models
without them load, shard and generate exactly as before."""

from collections.abc import Callable
from pathlib import Path

import mlx.nn as nn
import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.family import Architecture, PromptEncoder


class _Model(nn.Module):
    pass


def _registered(
    architecture: Architecture | None,
) -> Callable[[nn.Module], Architecture | None]:
    def lookup(_model: nn.Module) -> Architecture | None:
        return architecture

    return lookup


def test_post_load_runs_with_the_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[nn.Module, Path]] = []

    def post_load(model: nn.Module, model_path: Path) -> None:
        calls.append((model, model_path))

    architecture = Architecture(
        model_type="hooked", model_class=_Model, post_load=post_load
    )
    monkeypatch.setattr(utils_mlx, "architecture_for_model", _registered(architecture))
    model = _Model()
    utils_mlx.run_post_load(model, Path("/models/hooked"))
    assert calls == [(model, Path("/models/hooked"))]


@pytest.mark.parametrize("registered", [True, False])
def test_post_load_is_a_no_op_without_the_hook(
    monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    architecture = (
        Architecture(model_type="plain", model_class=_Model) if registered else None
    )
    monkeypatch.setattr(utils_mlx, "architecture_for_model", _registered(architecture))
    utils_mlx.run_post_load(_Model(), Path("/models/plain"))


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


@pytest.mark.parametrize(
    ("exact_model_types", "model_type", "expected"),
    [
        (frozenset[str](), "some_type", False),
        (frozenset({"some_type"}), "some_type", True),
        (frozenset({"some_type"}), "other_type", False),
        (frozenset({"some_type"}), None, False),
    ],
)
def test_think_tags_are_kept_when_the_architecture_says_so(
    monkeypatch: pytest.MonkeyPatch,
    exact_model_types: frozenset[str],
    model_type: str | None,
    expected: bool,
) -> None:
    """Models rendered through their Jinja chat template have no prompt encoder; their
    architecture's exact_think_tags decides."""

    def no_encoder(
        _task: TextGenerationTaskParams, _model_type: str | None = None
    ) -> PromptEncoder | None:
        return None

    monkeypatch.setattr(registry, "prompt_encoder_for", no_encoder)
    monkeypatch.setattr(registry, "_EXACT_THINK_TAG_MODEL_TYPES", exact_model_types)
    task = TextGenerationTaskParams(
        model=ModelId("some/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )
    assert registry.keeps_think_tags(task, model_type) is expected
