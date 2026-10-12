import importlib.machinery
from pathlib import Path

import pytest

import exo.worker.runner.bootstrap as bootstrap
from exo.worker.runner.bootstrap import (
    _ensure_cuda_home,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def cuda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


def _fake_nvidia_spec(base: Path) -> importlib.machinery.ModuleSpec:
    spec = importlib.machinery.ModuleSpec("nvidia", None, is_package=True)
    spec.submodule_search_locations = [str(base)]
    return spec


def test_noop_off_linux(
    cuda_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")
    _ensure_cuda_home()
    assert "CUDA_HOME" not in bootstrap.os.environ


def test_respects_existing_configuration(
    cuda_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
    _ensure_cuda_home()
    assert "CUDA_HOME" not in bootstrap.os.environ


def test_sets_cuda_home_from_nvidia_package(
    cuda_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    include_dir = tmp_path / "cuda_runtime" / "include"
    include_dir.mkdir(parents=True)

    def fake_find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        return _fake_nvidia_spec(tmp_path) if name == "nvidia" else None

    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", fake_find_spec)
    _ensure_cuda_home()
    assert bootstrap.os.environ["CUDA_HOME"] == str(tmp_path / "cuda_runtime")
    monkeypatch.delenv("CUDA_HOME")


def test_noop_without_headers(
    cuda_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        return _fake_nvidia_spec(tmp_path) if name == "nvidia" else None

    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", fake_find_spec)
    _ensure_cuda_home()
    assert "CUDA_HOME" not in bootstrap.os.environ
