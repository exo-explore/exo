"""The generic per-family hooks: they run only for an architecture that sets them, and models
without them load, shard and generate exactly as before."""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx import auto_parallel, builder, utils_mlx
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


class _LayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [nn.Linear(2, 2) for _ in range(4)]


def test_pipeline_parallel_override_replaces_the_generic_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, int]] = []

    def pipeline_parallel(
        model: nn.Module,
        group: mx.distributed.Group,
        model_shard_meta: PipelineShardMetadata,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        seen.append((model_shard_meta.start_layer, model_shard_meta.end_layer))
        yield ModelLoadingResponse(layers_loaded=0, total=1)
        return model

    architecture = Architecture(
        model_type="own_pipeline",
        model_class=_LayeredModel,
        pipeline_parallel=pipeline_parallel,
    )
    monkeypatch.setattr(
        auto_parallel, "architecture_for_model", _registered(architecture)
    )
    model = _LayeredModel()
    shard = cast(
        PipelineShardMetadata,
        get_pipeline_shard_metadata(ModelId("own/pipeline"), device_rank=0),
    )
    group = cast(mx.distributed.Group, object())
    generator = auto_parallel.pipeline_auto_parallel(model, group, shard)
    progress: list[ModelLoadingResponse] = []
    while True:
        try:
            progress.append(next(generator))
        except StopIteration as stop:
            result = cast(nn.Module, stop.value)
            break
    assert result is model
    assert progress == [ModelLoadingResponse(layers_loaded=0, total=1)]
    assert seen == [(shard.start_layer, shard.end_layer)]
    # the generic wrappers were not applied
    layers = cast(list[nn.Module], model.model.layers)
    assert all(isinstance(layer, nn.Linear) for layer in layers)


@pytest.mark.parametrize(
    ("batched", "no_batch_env", "expected"),
    [
        (True, False, True),
        (False, False, False),
        (True, True, False),
        (None, False, True),
        (None, True, False),
    ],
)
def test_generates_in_batches(
    monkeypatch: pytest.MonkeyPatch,
    batched: bool | None,
    no_batch_env: bool,
    expected: bool,
) -> None:
    architecture = (
        None
        if batched is None
        else Architecture(
            model_type="m", model_class=_Model, batched_generation=batched
        )
    )
    monkeypatch.setattr(builder, "architecture_for_model", _registered(architecture))
    if no_batch_env:
        monkeypatch.setenv("EXO_NO_BATCH", "1")
    else:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    assert builder.generates_in_batches(_Model()) is expected


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
