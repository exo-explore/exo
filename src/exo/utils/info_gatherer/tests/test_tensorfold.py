import subprocess

import pytest

from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.tensorfold import NativeCapabilities, ServeCapabilities
from exo.utils.info_gatherer import info_gatherer, tensorfold


def capabilities(chip: str | None, backend: str = "cuda") -> NativeCapabilities:
    return NativeCapabilities(
        schema=1,
        engine="zig",
        version="test",
        chip=chip,
        backends=[backend],
        families={"qwen3_5": ["mlx-q4g64"]},
        token_stream_versions=[1],
        serve=ServeCapabilities(flags={"--unix-socket": {}, "--name": {}}),
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("4096, NVIDIA GeForce RTX 4090\n", Memory.from_gb(4)),
        ("[N/A], NVIDIA GB10\n", Memory.from_gb(64)),
        ("[N/A], NVIDIA H100\n", None),
        ("4096, GPU A\n2048, GPU B\n", None),
        ("malformed", None),
    ],
)
def test_gpu_memory_never_sums_devices_or_guesses_vram(
    output: str, expected: Memory | None
) -> None:
    assert tensorfold.parse_cuda_memory(output, 64 * 1024**3) == expected


def test_capabilities_without_a_device_do_not_advertise_a_backend() -> None:
    assert tensorfold.detected_backends(None) == []
    assert tensorfold.detected_backends(capabilities(None)) == []
    assert tensorfold.detected_backends(capabilities("apple-m5")) == []
    assert tensorfold.detected_backends(capabilities("nvidia-sm121")) == [
        Backend.TensorFoldCuda
    ]


def test_metal_requires_the_native_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tensorfold.sys, "platform", "linux")
    assert tensorfold.detected_backends(capabilities("apple-m5", "metal")) == []
    monkeypatch.setattr(tensorfold.sys, "platform", "darwin")
    assert tensorfold.detected_backends(capabilities("apple-m5", "metal")) == [
        Backend.TensorFoldMetal
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        ValueError("version"),
        subprocess.TimeoutExpired("probe", 10),
    ],
)
def test_optional_probe_failure_leaves_mlx_available(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def unavailable() -> NativeCapabilities:
        raise error

    monkeypatch.setattr(tensorfold, "probe_binary", unavailable)
    assert tensorfold.probe_tensorfold() is None


def test_gpu_memory_probe_failure_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> str:
        raise OSError("driver unavailable")

    monkeypatch.setattr(tensorfold, "cuda_bus_id", unavailable)
    assert tensorfold.cuda_memory_available() is None


async def test_native_only_install_does_not_advertise_missing_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(info_gatherer, "_has_mlx", lambda: False)
    monkeypatch.setattr(
        info_gatherer, "probe_tensorfold", lambda: capabilities("nvidia-sm121")
    )
    result = await info_gatherer.NodeBackends.gather()
    assert result.backends == [Backend.TensorFoldCuda]


async def test_mlx_advertisement_remains_available_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(info_gatherer, "_has_mlx", lambda: True)
    monkeypatch.setattr(info_gatherer, "_has_nvml_cuda", lambda: False)
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", True)
    monkeypatch.setattr(info_gatherer, "probe_tensorfold", lambda: None)
    result = await info_gatherer.NodeBackends.gather()
    assert result.backends == [Backend.MlxCpu, Backend.MlxMetal]
