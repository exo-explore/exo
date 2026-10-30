"""The generic per-family hooks: they run only for an architecture that sets them, and models
without them load, shard and generate exactly as before."""

from pathlib import Path

import mlx.nn as nn
import pytest

from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.models.family import Architecture


class _Model(nn.Module):
    pass


def test_post_load_runs_with_the_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[nn.Module, Path]] = []

    def post_load(model: nn.Module, model_path: Path) -> None:
        calls.append((model, model_path))

    architecture = Architecture(
        model_type="hooked", model_class=_Model, post_load=post_load
    )
    monkeypatch.setattr(utils_mlx, "architecture_for_model", lambda _: architecture)
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
    monkeypatch.setattr(utils_mlx, "architecture_for_model", lambda _: architecture)
    utils_mlx.run_post_load(_Model(), Path("/models/plain"))
