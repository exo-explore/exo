import multiprocessing as mp
import os
from typing import Any

import pytest

from .conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    create_hostfile,
    run_gpt_oss_pipeline_device,
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


class TestPipelineParallelPrefillBug:
    BUG_TRIGGER_LAYER_SPLITS: list[tuple[int, int]] = [(0, 1), (1, 24)]

    def test_prefill_bug_without_patch(self) -> None:
        """Replicate bug."""
        ctx = mp.get_context("spawn")
        world_size = 2
        layer_splits = self.BUG_TRIGGER_LAYER_SPLITS
        prompt_tokens = 100
        prefill_step_size = 64

        hostfile_path, _ = create_hostfile(world_size, DEFAULT_GPT_OSS_CONFIG.base_port)

        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []

            for rank in range(world_size):
                p = ctx.Process(
                    target=run_gpt_oss_pipeline_device,
                    args=(
                        rank,
                        world_size,
                        hostfile_path,
                        DEFAULT_GPT_OSS_CONFIG.model_path,
                        layer_splits,
                        prompt_tokens,
                        prefill_step_size,
                        False,
                        result_queue,
                        DEFAULT_GPT_OSS_CONFIG.max_tokens,
                    ),
                )
                p.start()
                processes.append(p)

            process_timeout = 30
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

            all_success = all(
                results.get(r, (False, ""))[0] for r in range(world_size)
            )
            assert timed_out or not all_success, (
                "Expected timeout/failure WITHOUT patch. "
                "If this fails, mlx_lm may have been fixed upstream."
            )

        finally:
            os.unlink(hostfile_path)

    def test_prefill_fixed_with_patch(self) -> None:
        ctx = mp.get_context("spawn")
        world_size = 2
        layer_splits = self.BUG_TRIGGER_LAYER_SPLITS
        prompt_tokens = 100
        prefill_step_size = 64

        hostfile_path, _ = create_hostfile(world_size, DEFAULT_GPT_OSS_CONFIG.base_port)

        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []

            for rank in range(world_size):
                p = ctx.Process(
                    target=run_gpt_oss_pipeline_device,
                    args=(
                        rank,
                        world_size,
                        hostfile_path,
                        DEFAULT_GPT_OSS_CONFIG.model_path,
                        layer_splits,
                        prompt_tokens,
                        prefill_step_size,
                        True,
                        result_queue,
                        DEFAULT_GPT_OSS_CONFIG.max_tokens,
                    ),
                )
                p.start()
                processes.append(p)

            process_timeout = 60
            for p in processes:  # pyright: ignore[reportAny]
                p.join(timeout=process_timeout)  # pyright: ignore[reportAny]

            timed_out = any(p.is_alive() for p in processes)  # pyright: ignore[reportAny]

            for p in processes:  # pyright: ignore[reportAny]
                if p.is_alive():  # pyright: ignore[reportAny]
                    p.terminate()  # pyright: ignore[reportAny]
                    p.join(timeout=5)  # pyright: ignore[reportAny]

            assert not timed_out, "Unexpected timeout with patch"

            results: dict[int, tuple[bool, str]] = {}
            while not result_queue.empty():  # pyright: ignore[reportAny]
                rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
                results[rank] = (success, value)

            all_success = all(
                results.get(r, (False, ""))[0] for r in range(world_size)
            )
            assert all_success, f"Failures: {results}"

        finally:
            os.unlink(hostfile_path)


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
        ctx = mp.get_context("spawn")
        world_size = 2
        prompt_tokens = 600
        prefill_step_size = 512

        hostfile_path, _ = create_hostfile(
            world_size, DEFAULT_GPT_OSS_CONFIG.base_port + 100
        )

        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []

            for rank in range(world_size):
                p = ctx.Process(
                    target=run_gpt_oss_pipeline_device,
                    args=(
                        rank,
                        world_size,
                        hostfile_path,
                        DEFAULT_GPT_OSS_CONFIG.model_path,
                        layer_splits,
                        prompt_tokens,
                        prefill_step_size,
                        True,
                        result_queue,
                        DEFAULT_GPT_OSS_CONFIG.max_tokens,
                    ),
                )
                p.start()
                processes.append(p)

            process_timeout = 60
            for p in processes:  # pyright: ignore[reportAny]
                p.join(timeout=process_timeout)  # pyright: ignore[reportAny]

            timed_out = any(p.is_alive() for p in processes)  # pyright: ignore[reportAny]

            for p in processes:  # pyright: ignore[reportAny]
                if p.is_alive():  # pyright: ignore[reportAny]
                    p.terminate()  # pyright: ignore[reportAny]
                    p.join(timeout=5)  # pyright: ignore[reportAny]

            assert not timed_out, f"Timeout with {layer_splits}"

            results: dict[int, tuple[bool, str]] = {}
            while not result_queue.empty():  # pyright: ignore[reportAny]
                rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
                results[rank] = (success, value)

            all_success = all(
                results.get(r, (False, ""))[0] for r in range(world_size)
            )
            assert all_success, f"Failures with {layer_splits}: {results}"

        finally:
            os.unlink(hostfile_path)


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
        ctx = mp.get_context("spawn")
        world_size = 2
        layer_splits = [(0, 12), (12, 24)]

        hostfile_path, _ = create_hostfile(
            world_size, DEFAULT_GPT_OSS_CONFIG.base_port + 200
        )

        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []

            for rank in range(world_size):
                p = ctx.Process(
                    target=run_gpt_oss_pipeline_device,
                    args=(
                        rank,
                        world_size,
                        hostfile_path,
                        DEFAULT_GPT_OSS_CONFIG.model_path,
                        layer_splits,
                        prompt_tokens,
                        prefill_step_size,
                        True,
                        result_queue,
                        DEFAULT_GPT_OSS_CONFIG.max_tokens,
                    ),
                )
                p.start()
                processes.append(p)

            process_timeout = 60
            for p in processes:  # pyright: ignore[reportAny]
                p.join(timeout=process_timeout)  # pyright: ignore[reportAny]

            timed_out = any(p.is_alive() for p in processes)  # pyright: ignore[reportAny]

            for p in processes:  # pyright: ignore[reportAny]
                if p.is_alive():  # pyright: ignore[reportAny]
                    p.terminate()  # pyright: ignore[reportAny]
                    p.join(timeout=5)  # pyright: ignore[reportAny]

            assert not timed_out, f"Timeout: {prompt_tokens=}, {prefill_step_size=}"

            results: dict[int, tuple[bool, str]] = {}
            while not result_queue.empty():  # pyright: ignore[reportAny]
                rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
                results[rank] = (success, value)

            all_success = all(
                results.get(r, (False, ""))[0] for r in range(world_size)
            )
            assert all_success, f"Failures: {results}"

        finally:
            os.unlink(hostfile_path)
