"""A pipeline rank builds its decode graph while its input activation is still on the way.

On a decode step, PipelineFirstLayer used to wait for the previous rank's activation
before the rest of the shard's graph was even built, so every token paid for building
that graph after the activation arrived. It now posts the receive, the shard's graph is
built on it, and PipelineLastLayer waits for the receive on the CPU, on its own, before
it evaluates anything on the GPU. The values are unchanged, and the GPU still never waits
on the network: a sender that falls behind by more than the GPU watchdog (about 5 s) must
not fail the receiving rank.

Each case runs two ranks over the ring on this machine, one layer per rank.
"""

import ast
import multiprocessing as mp
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models
import pytest

from exo.worker.engines.mlx import models as exo_models
from exo.worker.engines.mlx.sharding import pipeline
from exo.worker.engines.mlx.sharding.pipeline import (
    EAGER_RECEIVE_MODEL_MODULES,
    PipelineFirstLayer,
    PipelineLastLayer,
    receive_can_be_deferred,
)
from exo.worker.engines.mlx.utils_mlx import (
    mx_barrier,
    use_generation_stream_by_default,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer, create_hostfile

# Longer than the GPU watchdog, about 5 s on Apple silicon
SENDER_DELAY = 8.0


class HostReadingLayer(MockLayer):
    """Reads a value computed from its input on the host, as DBRX's MoE does."""

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        total = cast(float, (x * 2).sum().item())
        return x * 2 + 0 * total


def _hold_back_sends(delay: float) -> None:
    """Make this rank's sends leave `delay` seconds late"""
    send = mx.distributed.send

    def late_send(
        x: mx.array, dst: int, *, group: mx.distributed.Group | None = None
    ) -> mx.array:
        time.sleep(delay)
        return send(x, dst, group=group)

    setattr(mx.distributed, "send", late_send)  # noqa: B010


def _run_rank(
    rank: int,
    hostfile_path: str,
    sender_delay: float,
    is_prefill: bool,
    host_read: bool,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        use_generation_stream_by_default()
        inner = HostReadingLayer() if host_read else MockLayer()
        layer = PipelineLastLayer(
            PipelineFirstLayer(inner, r=rank, group=group, defer_receive=not host_read),
            r=rank,
            s=group.size(),
            group=group,
        )
        cast(PipelineFirstLayer, layer.original_layer).is_prefill = is_prefill
        layer.is_prefill = is_prefill
        mx_barrier(group)
        if rank == 0:
            _hold_back_sends(sender_delay)
        outputs: list[object] = []
        for step in range(3):
            output = layer(mx.full((1, 1, 4), float(step + 1)))
            mx.eval(output)
            outputs.append(output.tolist())
        # The last layer has waited for every receive the first layer posted
        assert not pipeline._pending_recvs  # pyright: ignore[reportPrivateUsage]
        results.put((rank, True, outputs))  # pyright: ignore[reportAny]
    except Exception as e:
        results.put((rank, False, repr(e)))  # pyright: ignore[reportAny]


def _run(
    port: int, sender_delay: float, is_prefill: bool, host_read: bool = False
) -> dict[int, object]:
    world_size = 2
    hostfile_path, _ = create_hostfile(world_size, port)
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, hostfile_path, sender_delay, is_prefill, host_read, results),
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        outcomes: dict[int, tuple[bool, object]] = {}
        for _ in range(world_size):
            rank, ok, value = cast(
                tuple[int, bool, object],
                results.get(timeout=120),  # pyright: ignore[reportAny]
            )
            outcomes[rank] = (ok, value)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        os.unlink(hostfile_path)
    failed = {rank: value for rank, (ok, value) in outcomes.items() if not ok}
    assert failed == {}
    return {rank: value for rank, (_, value) in outcomes.items()}


# Each rank doubles its input: the last rank's output is 4x. On a decode step it is
# gathered to every rank; in prefill each rank keeps its own.
EXPECTED = [[[[4.0 * (step + 1)] * 4]] for step in range(3)]
FIRST_RANK_PREFILL = [[[[2.0 * (step + 1)] * 4]] for step in range(3)]


def test_decode_steps_give_the_same_values() -> None:
    values = _run(29540, sender_delay=0.0, is_prefill=False)
    assert values[0] == values[1] == EXPECTED


def test_prefill_steps_give_the_same_values() -> None:
    values = _run(29542, sender_delay=0.0, is_prefill=True)
    assert values[0] == FIRST_RANK_PREFILL
    assert values[1] == EXPECTED


@pytest.mark.slow
@pytest.mark.skipif(not mx.metal.is_available(), reason="the GPU watchdog is Metal's")
def test_a_sender_slower_than_the_gpu_watchdog_does_not_fail_the_receiver() -> None:
    values = _run(29544, sender_delay=SENDER_DELAY, is_prefill=False)
    assert values[0] == values[1] == EXPECTED


@pytest.mark.slow
@pytest.mark.skipif(not mx.metal.is_available(), reason="the GPU watchdog is Metal's")
def test_a_layer_reading_on_the_host_keeps_the_receive_eager() -> None:
    # A host read mid-shard would evaluate a pending receive with the GPU work, so models
    # that do one wait for the receive in the first layer, as before.
    values = _run(29546, sender_delay=SENDER_DELAY, is_prefill=False, host_read=True)
    assert values[0] == values[1] == EXPECTED


def test_models_reading_on_the_host_keep_the_receive_eager() -> None:
    def model_from(module: str) -> nn.Module:
        model_class = type("Model", (nn.Module,), {"__module__": module})
        return cast(nn.Module, model_class())

    assert not receive_can_be_deferred(model_from("mlx_lm.models.dbrx"))
    assert receive_can_be_deferred(model_from("mlx_lm.models.llama"))


# Host reads that only touch values not computed from a layer's input (e.g. cache
# bookkeeping) and need no eager receive: module -> reason. None on this base.
AUDITED_HOST_READS: dict[str, str] = {}
_HOST_READ_METHODS = {"item", "tolist", "numpy"}
_HOST_READ_FUNCTIONS = {
    ("mx", "eval"),
    ("mx", "async_eval"),
    ("mx", "synchronize"),
    ("np", "array"),
    ("np", "asarray"),
}


def _forward_host_reads(package: ModuleType) -> dict[str, list[str]]:
    """Modules of `package` with a host read in a class's `__call__`."""
    root = Path(cast(str, package.__file__)).parent
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        module = ".".join(
            (package.__name__, *path.relative_to(root).with_suffix("").parts)
        )
        tree = ast.parse(path.read_text())
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or method.name != "__call__":
                    continue
                for node in ast.walk(method):
                    if not isinstance(node, ast.Call) or not isinstance(
                        node.func, ast.Attribute
                    ):
                        continue
                    func = node.func
                    name = (
                        (func.value.id, func.attr)
                        if isinstance(func.value, ast.Name)
                        else None
                    )
                    if (func.attr in _HOST_READ_METHODS and not node.args) or (
                        name in _HOST_READ_FUNCTIONS
                    ):
                        found.setdefault(module, []).append(
                            f"{cls.name}.__call__: {ast.unparse(node)}"
                        )
    return found


def test_every_forward_with_a_host_read_keeps_the_receive_eager() -> None:
    # A new model (or an mlx-lm bump) whose forward reads a value on the host must be
    # audited: a read of a value computed from the layer's input needs the eager receive.
    found = {**_forward_host_reads(mlx_lm.models), **_forward_host_reads(exo_models)}
    unlisted = {
        module: reads
        for module, reads in found.items()
        if module not in EAGER_RECEIVE_MODEL_MODULES
        and module not in AUDITED_HOST_READS
    }
    assert unlisted == {}, (
        "host reads in a forward: add the module to EAGER_RECEIVE_MODEL_MODULES "
        "(sharding/pipeline.py) if they read a value computed from the layer's input, "
        f"or to AUDITED_HOST_READS with the reason they don't: {unlisted}"
    )
    assert "mlx_lm.models.dbrx" in found
