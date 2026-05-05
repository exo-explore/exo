import os
import sys

import mlx.core as mx
import pytest
from _pytest.capture import CaptureFixture

from exo.utils.mp_stdio_capture import CapturedProcessOptions


def _write_to_stdio(prefix: str, *, stderr_suffix: str) -> None:
    print(f"{prefix}: python stdout")
    print(f"{prefix}: python stderr {stderr_suffix}", file=sys.stderr)
    os.write(1, f"{prefix}: fd stdout\n".encode())
    os.write(2, f"{prefix}: fd stderr {stderr_suffix}\n".encode())


def _write_large_output() -> None:
    os.write(1, b"stdout-0123456789")
    os.write(2, b"stderr-0123456789")


def _raise_after_stderr_write() -> None:
    os.write(2, b"stderr before exception\n")
    raise RuntimeError("child boom")


def _mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    print(f"CHILD: start")

    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)

    print(f"CHILD: end")


@pytest.mark.asyncio
async def test_spawn_process_captures_stdout_and_stderr_separately(
        capfd: CaptureFixture[str],
) -> None:
    options = CapturedProcessOptions(start_method="spawn")
    captured = options.create_process(
        _write_to_stdio,
        "child",
        stderr_suffix="error",
    )

    result = await captured.run()

    parent_output = capfd.readouterr()
    stdout = result.output.stdout_text()
    stderr = result.output.stderr_text()

    assert result.exitcode == 0
    assert "child: python stdout" in stdout
    assert "child: fd stdout" in stdout
    assert "child: python stderr error" in stderr
    assert "child: fd stderr error" in stderr
    assert "child:" not in parent_output.out
    assert "child:" not in parent_output.err


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded.*:DeprecationWarning"
)
async def test_default_options_use_current_multiprocessing_context() -> None:
    result = (
        await CapturedProcessOptions()
        .create_process(
            _write_to_stdio,
            "default",
            stderr_suffix="error",
        )
        .run()
    )

    assert result.exitcode == 0
    assert "default: python stdout" in result.output.stdout_text()
    assert "default: python stderr error" in result.output.stderr_text()


@pytest.mark.asyncio
async def test_capture_can_keep_bounded_tail() -> None:
    options = CapturedProcessOptions(start_method="spawn", max_capture_bytes=8)
    result = await options.create_process(_write_large_output).run()

    assert result.exitcode == 0
    assert result.output.stdout == b"23456789"
    assert result.output.stderr == b"23456789"


@pytest.mark.asyncio
async def test_child_exception_traceback_is_captured_from_stderr() -> None:
    options = CapturedProcessOptions(start_method="spawn")
    result = await options.create_process(_raise_after_stderr_write).run()

    assert result.exitcode == 1
    stderr = result.output.stderr_text()
    assert "stderr before exception" in stderr
    assert "RuntimeError: child boom" in stderr


@pytest.mark.asyncio
async def test_death(capsys: CaptureFixture[str]) -> None:
    with capsys.disabled():
        options = CapturedProcessOptions(start_method="spawn")
        result = await options.create_process(_mlx_force_oom).run()

        print("PARENT: done")

        print("CHILD out:", result.output.stdout_text())
        print("CHILD err:", result.output.stderr_text())
