import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from .conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    create_hostfile,
    run_gpt_oss_pipeline_device,
    run_gpt_oss_tensor_parallel_device,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _check_model_exists(),
        reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
    ),
]


@dataclass
class DistributedTestResult:
    timed_out: bool
    world_size: int
    results: dict[int, tuple[bool, str]]

    @property
    def all_success(self) -> bool:
        if len(self.results) != self.world_size:
            return False
        return all(r[0] for r in self.results.values())


def run_distributed_test(
    world_size: int,
    port_offset: int,
    process_timeout: int,
    target: Callable[..., None],
    make_args: Callable[[int], tuple[Any, ...]],
) -> DistributedTestResult:
    ctx = mp.get_context("spawn")
    hostfile_path, _ = create_hostfile(
        world_size, DEFAULT_GPT_OSS_CONFIG.base_port + port_offset
    )

    try:
        result_queue: Any = ctx.Queue()
        processes: list[Any] = []

        for rank in range(world_size):
            args = make_args(rank)
            p = ctx.Process(
                target=target,
                args=(rank, world_size, hostfile_path, *args, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=process_timeout)  # pyright: ignore[reportAny]

        timed_out = any(p.is_alive() for p in processes)  # pyright: ignore[reportAny]

        for p in processes:  # pyright: ignore[reportAny]
            if p.is_alive():  # pyright: ignore[reportAny]
                p.terminate()  # pyright: ignore[reportAny]
                p.join(timeout=5)  # pyright: ignore[reportAny]

        results: dict[int, tuple[bool, str]] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            results[rank] = (success, value)

        return DistributedTestResult(
            timed_out=timed_out, world_size=world_size, results=results
        )

    finally:
        os.unlink(hostfile_path)


def run_pipeline_test(
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    prefill_step_size: int,
    use_patch: bool,
    port_offset: int = 0,
    process_timeout: int = 60,
) -> DistributedTestResult:
    def make_args(rank: int) -> tuple[Any, ...]:
        return (
            DEFAULT_GPT_OSS_CONFIG.model_path,
            layer_splits,
            prompt_tokens,
            prefill_step_size,
            use_patch,
        )

    return run_distributed_test(
        world_size=len(layer_splits),
        port_offset=port_offset,
        process_timeout=process_timeout,
        target=run_gpt_oss_pipeline_device,
        make_args=make_args,
    )


def run_tensor_test(
    prompt_tokens: int,
    prefill_step_size: int,
    use_patch: bool,
    port_offset: int = 0,
    process_timeout: int = 60,
) -> DistributedTestResult:
    def make_args(rank: int) -> tuple[Any, ...]:
        return (
            DEFAULT_GPT_OSS_CONFIG.model_path,
            prompt_tokens,
            prefill_step_size,
            use_patch,
        )

    return run_distributed_test(
        world_size=2,
        port_offset=port_offset,
        process_timeout=process_timeout,
        target=run_gpt_oss_tensor_parallel_device,
        make_args=make_args,
    )


class TestPipelineParallelPrefillBug:
    BUG_TRIGGER_SPLITS: list[tuple[int, int]] = [(0, 1), (1, 24)]

    def test_prefill_bug_without_patch(self) -> None:
        result = run_pipeline_test(
            layer_splits=self.BUG_TRIGGER_SPLITS,
            prompt_tokens=100,
            prefill_step_size=64,
            use_patch=False,
            process_timeout=30,
        )
        assert result.timed_out or not result.all_success, (
            "Expected timeout/failure WITHOUT patch. "
            "If this fails, mlx_lm may have been fixed upstream."
        )

    def test_prefill_fixed_with_patch(self) -> None:
        result = run_pipeline_test(
            layer_splits=self.BUG_TRIGGER_SPLITS,
            prompt_tokens=100,
            prefill_step_size=64,
            use_patch=True,
        )
        assert not result.timed_out, "Unexpected timeout with patch"
        assert result.all_success, f"Failures: {result.results}"


class TestPipelineSplitConfigurations:
    @pytest.mark.parametrize(
        "layer_splits",
        [
            [(0, 1), (1, 24)],
            [(0, 6), (6, 24)],
            [(0, 12), (12, 24)],
        ],
        ids=["1_23", "6_18", "12_12"],
    )
    def test_pipeline_splits_with_patch(
        self,
        layer_splits: list[tuple[int, int]],
    ) -> None:
        result = run_pipeline_test(
            layer_splits=layer_splits,
            prompt_tokens=600,
            prefill_step_size=512,
            use_patch=True,
            port_offset=100,
        )
        assert not result.timed_out, f"Timeout with {layer_splits}"
        assert result.all_success, f"Failures with {layer_splits}: {result.results}"


class TestPrefillStepSizeBoundaries:
    @pytest.mark.parametrize(
        "prefill_step_size,prompt_tokens",
        [
            (512, 511),
            (512, 512),
            (512, 513),
            (512, 1024),
        ],
        ids=["under", "exact", "over", "double"],
    )
    def test_boundary_conditions_with_patch(
        self,
        prefill_step_size: int,
        prompt_tokens: int,
    ) -> None:
        result = run_pipeline_test(
            layer_splits=[(0, 12), (12, 24)],
            prompt_tokens=prompt_tokens,
            prefill_step_size=prefill_step_size,
            use_patch=True,
            port_offset=200,
        )
        assert not result.timed_out, f"Timeout: {prompt_tokens=}, {prefill_step_size=}"
        assert result.all_success, f"Failures: {result.results}"


class TestTensorParallelWithPatch:
    """Test that the patch does not break tensor parallelism."""

    def test_tensor_parallel(self) -> None:
        result = run_tensor_test(
            prompt_tokens=100,
            prefill_step_size=64,
            use_patch=True,
            port_offset=400,
        )
        assert not result.timed_out, "Unexpected timeout with patch"
        assert result.all_success, f"Failures: {result.results}"


class TestTensorParallelBoundaries:
    @pytest.mark.parametrize(
        "prefill_step_size,prompt_tokens",
        [
            (512, 511),
            (512, 512),
            (512, 513),
            (512, 1024),
        ],
        ids=["under", "exact", "over", "double"],
    )
    def test_tensor_parallel_boundaries_with_patch(
        self,
        prefill_step_size: int,
        prompt_tokens: int,
    ) -> None:
        result = run_tensor_test(
            prompt_tokens=prompt_tokens,
            prefill_step_size=prefill_step_size,
            use_patch=True,
            port_offset=500,
        )
        assert not result.timed_out, f"Timeout: {prompt_tokens=}, {prefill_step_size=}"
        assert result.all_success, f"Failures: {result.results}"
