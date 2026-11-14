import pytest

from exo.utils.info_gatherer import metal_memory
from exo.utils.info_gatherer.metal_memory import MetalBindings


def bindings(
    *, size: int, device: int | None = 3, supported: bool = True, failed: bool = False
) -> tuple[MetalBindings, list[str]]:
    events: list[str] = []

    def push_pool() -> int:
        events.append("push")
        return 9

    def create_device() -> int | None:
        events.append("create")
        return device

    def read_size(receiver: int, selector: int) -> int:
        assert receiver == device and selector == 2
        events.append("read")
        if failed:
            raise OSError("query failed")
        return size

    def release(receiver: int) -> None:
        assert receiver == device
        events.append("release")

    def pop_pool(pool: int | None) -> None:
        assert pool == 9
        events.append("pop")

    return MetalBindings(
        create_device=create_device,
        selector=lambda name: 1 if name == b"respondsToSelector:" else 2,
        responds=lambda receiver, selector, getter: (
            receiver == device and selector == 1 and getter == 2 and supported
        ),
        read_size=read_size,
        release=release,
        push_pool=push_pool,
        pop_pool=pop_pool,
    ), events


def test_reads_exact_64_bit_native_limit_and_releases_owned_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, events = bindings(size=17 * 1024**3 + 123)
    monkeypatch.setattr(metal_memory.sys, "platform", "darwin")
    monkeypatch.setattr(metal_memory, "load_bindings", lambda: native)
    memory = metal_memory.get_metal_working_set()
    assert memory is not None and memory.in_bytes == 17 * 1024**3 + 123
    assert events == ["push", "create", "read", "release", "pop"]


@pytest.mark.parametrize("failure", ["no-device", "unsupported", "zero", "query"])
def test_unavailable_memory_is_unknown_and_every_owned_resource_is_released(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    native, events = bindings(
        size=0 if failure == "zero" else 2**36,
        device=None if failure == "no-device" else 3,
        supported=failure != "unsupported",
        failed=failure == "query",
    )
    monkeypatch.setattr(metal_memory.sys, "platform", "darwin")
    monkeypatch.setattr(metal_memory, "load_bindings", lambda: native)
    assert metal_memory.get_metal_working_set() is None
    assert events[-1] == "pop"
    assert events.count("release") == (0 if failure == "no-device" else 1)
    assert ("read" in events) == (failure in {"zero", "query"})


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_framework_failure_and_other_platforms_are_unknown(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    calls: list[bool] = []

    def unavailable() -> MetalBindings:
        calls.append(True)
        raise OSError("Metal framework unavailable")

    monkeypatch.setattr(metal_memory.sys, "platform", platform)
    monkeypatch.setattr(metal_memory, "load_bindings", unavailable)
    assert metal_memory.get_metal_working_set() is None
    assert bool(calls) == (platform == "darwin")
