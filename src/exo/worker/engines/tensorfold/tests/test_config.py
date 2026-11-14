import os
from pathlib import Path

import pytest

from exo.worker.engines.tensorfold.config import (
    NativeSettings,
    validate_single_rank_environment,
)
from exo.worker.engines.tensorfold.process import NativeProcess


@pytest.mark.parametrize(
    "name, value",
    [
        ("TF_GLM_WORLD", "2"),
        ("TF_GLM_RANK", "1"),
        ("TF_GLM_WORLD", "invalid"),
        ("TF_GLM_RANK", ""),
        ("TF_NEMOTRON_RANK", "00"),
        ("TF_NEMOTRON_WORLD", "2"),
        ("TF_MINIMAX_RANK", "1"),
        ("TF_MINIMAX_WORLD", "2"),
        ("TF_TP_WORLD", "4"),
        ("TF_STAGE_WORLD", "2"),
    ],
)
def test_refuses_inherited_distributed_state_before_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, value: str
) -> None:
    for key in tuple(os.environ):
        if key.startswith("TF_") and key.endswith(("_WORLD", "_RANK")):
            monkeypatch.delenv(key)
    monkeypatch.setenv(name, value)
    process = NativeProcess(NativeSettings(binary=tmp_path / "must-not-launch"))
    try:
        with pytest.raises(ValueError, match=name):
            process.start(tmp_path, "model", 8192, None)
        assert not process.socket_path.parent.exists()
    finally:
        process.close()


def test_accepts_explicit_single_rank_and_unrelated_device_selection() -> None:
    validate_single_rank_environment(
        {"TF_GLM_WORLD": "1", "TF_GLM_RANK": "0", "TF_CUDA_DEVICE": "2"}
    )
