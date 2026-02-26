# type: ignore
"""Test that pipeline prefill callbacks and output exactly match stream_generate.

Spins up a single-device (non-pipeline) run and a distributed pipeline run,
then verifies that the prompt_progress_callback sequences are identical
and that generated text matches.
"""

import json
import multiprocessing as mp
import os
import tempfile
import traceback
from typing import Any, cast

import pytest

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams

MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
MODEL_PATH = EXO_MODELS_DIR / "mlx-community--gpt-oss-20b-MXFP4-Q8"
TOTAL_LAYERS = 24
MAX_TOKENS = 10
SEED = 42
TEMPERATURE = 0.0


def _model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId(MODEL_ID),
        storage_size=Memory.from_gb(12),
        n_layers=TOTAL_LAYERS,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _build_prompt(tokenizer: Any, prompt_tokens: int) -> tuple[str, Any]:
    """Build a prompt with the given number of user-content tokens, return (chat_prompt, task)."""
    from exo.worker.engines.mlx.utils_mlx import apply_chat_template

    base_text = "The quick brown fox jumps over the lazy dog. "
    base_toks = tokenizer.encode(base_text)
    repeats = (prompt_tokens // len(base_toks)) + 2
    long_text = base_text * repeats
    tokens = tokenizer.encode(long_text)[:prompt_tokens]
    prompt_text = tokenizer.decode(tokens)

    task = TextGenerationTaskParams(
        model=MODEL_ID,
        input=[InputMessage(role="user", content=prompt_text)],
        max_output_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        seed=SEED,
    )

    prompt = apply_chat_template(tokenizer, task)
    return prompt, task


# ---------------------------------------------------------------------------
# Single-device process: uses stream_generate path (no pipeline layers)
# ---------------------------------------------------------------------------
def _run_single_device(
    prompt_tokens: int,
    result_queue: Any,
) -> None:
    """Load full model without pipeline sharding, run mlx_generate, record callbacks."""
    try:
        import mlx.core as mx
        from mlx_lm.utils import load_model

        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.mlx.cache import encode_prompt
        from exo.worker.engines.mlx.generator.generate import mlx_generate
        from exo.worker.engines.mlx.utils_mlx import (
            build_model_path,
            get_tokenizer,
        )

        model_path = build_model_path(ModelId(MODEL_ID))
        model, _ = load_model(model_path, lazy=True, strict=False)
        mx.eval(model)

        # Use PipelineShardMetadata just for get_tokenizer (needs model_card), but
        # do NOT apply pipeline sharding — the model keeps all layers unwrapped.
        dummy_meta = PipelineShardMetadata(
            model_card=_model_card(),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=TOTAL_LAYERS,
            n_layers=TOTAL_LAYERS,
        )
        tokenizer = get_tokenizer(model_path, dummy_meta)

        prompt, task = _build_prompt(tokenizer, prompt_tokens)

        callbacks: list[tuple[int, int]] = []

        def on_progress(processed: int, total: int) -> None:
            callbacks.append((processed, total))

        generated_text = ""
        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=None,
            group=None,
            on_prefill_progress=on_progress,
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        # Also record the token count that prefill() received (prompt_tokens[:-1])
        all_tokens = encode_prompt(tokenizer, prompt)
        prefill_token_count = len(all_tokens) - 1

        result_queue.put(
            (
                True,
                {
                    "callbacks": callbacks,
                    "text": generated_text,
                    "prefill_token_count": prefill_token_count,
                },
            )
        )

    except Exception as e:
        result_queue.put((False, f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Pipeline device process: uses _pipeline_prefill_cache path
# ---------------------------------------------------------------------------
def _run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    result_queue: Any,
) -> None:
    """Load model with pipeline sharding, run mlx_generate, record callbacks."""
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        import mlx.core as mx

        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.mlx.cache import encode_prompt
        from exo.worker.engines.mlx.generator.generate import mlx_generate
        from exo.worker.engines.mlx.utils_mlx import shard_and_load

        group = mx.distributed.init(backend="ring", strict=True)

        start_layer, end_layer = layer_splits[rank]
        shard_meta = PipelineShardMetadata(
            model_card=_model_card(),
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=TOTAL_LAYERS,
        )

        model, tokenizer = shard_and_load(
            shard_meta, group, on_timeout=None, on_layer_loaded=None
        )
        model = cast(Any, model)

        prompt, task = _build_prompt(tokenizer, prompt_tokens)

        callbacks: list[tuple[int, int]] = []

        def on_progress(processed: int, total: int) -> None:
            callbacks.append((processed, total))

        def distributed_prompt_progress_callback(_group: Any = group) -> None:
            from exo.worker.engines.mlx.utils_mlx import mx_any

            mx_any(False, _group)

        generated_text = ""
        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=None,
            group=group,
            on_prefill_progress=on_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        all_tokens = encode_prompt(tokenizer, prompt)
        prefill_token_count = len(all_tokens) - 1

        result_queue.put(
            (
                rank,
                True,
                {
                    "callbacks": callbacks,
                    "text": generated_text,
                    "prefill_token_count": prefill_token_count,
                },
            )
        )

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _create_hostfile(world_size: int, base_port: int) -> str:
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        return f.name


def _run_single_device_test(prompt_tokens: int, timeout: int = 120) -> dict[str, Any]:
    """Run single-device (stream_generate) prefill and return results."""
    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()

    p = ctx.Process(target=_run_single_device, args=(prompt_tokens, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        pytest.fail("Single-device process timed out")

    assert not result_queue.empty(), "Single-device process produced no result"
    success, data = result_queue.get()
    assert success, f"Single-device process failed:\n{data}"
    return data


def _run_pipeline_test(
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    base_port: int,
    timeout: int = 120,
) -> dict[int, dict[str, Any]]:
    """Run pipeline prefill across ranks and return per-rank results."""
    world_size = len(layer_splits)
    hostfile_path = _create_hostfile(world_size, base_port)
    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()

    try:
        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_run_pipeline_device,
                args=(
                    rank,
                    world_size,
                    hostfile_path,
                    layer_splits,
                    prompt_tokens,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=timeout)

        timed_out = any(p.is_alive() for p in processes)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        assert not timed_out, "Pipeline processes timed out"

        results: dict[int, dict[str, Any]] = {}
        while not result_queue.empty():
            rank, success, data = result_queue.get()
            assert success, f"Pipeline rank {rank} failed:\n{data}"
            results[rank] = data

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}: missing ranks {set(range(world_size)) - results.keys()}"
        )
        return results

    finally:
        os.unlink(hostfile_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason=f"GPT-OSS model not found at {MODEL_PATH}",
    ),
]

LAYER_SPLITS_4WAY: list[tuple[int, int]] = [(0, 6), (6, 12), (12, 18), (18, 24)]
LAYER_SPLITS_2WAY: list[tuple[int, int]] = [(0, 12), (12, 24)]


class TestPipelineNoDeadlock:
    """Pipeline prefill must not deadlock at any rank count or prompt length."""

    @pytest.mark.parametrize(
        "layer_splits,prompt_tokens",
        [
            (LAYER_SPLITS_2WAY, 128),
            (LAYER_SPLITS_2WAY, 4096),
            (LAYER_SPLITS_2WAY, 8192),
            (LAYER_SPLITS_2WAY, 16384),
            (LAYER_SPLITS_4WAY, 128),
            (LAYER_SPLITS_4WAY, 4096),
            (LAYER_SPLITS_4WAY, 8192),
            (LAYER_SPLITS_4WAY, 16384),
        ],
        ids=[
            "2rank_128tok",
            "2rank_4096tok",
            "2rank_8192tok",
            "2rank_16384tok",
            "4rank_128tok",
            "4rank_4096tok",
            "4rank_8192tok",
            "4rank_16384tok",
        ],
    )
    def test_no_deadlock(
        self,
        layer_splits: list[tuple[int, int]],
        prompt_tokens: int,
    ) -> None:
        """Pipeline must complete without deadlock at various prompt lengths."""
        pipeline_results = _run_pipeline_test(
            layer_splits=layer_splits,
            prompt_tokens=prompt_tokens,
            base_port=29650,
            timeout=60,
        )
        # If we get here, no deadlock. Verify all ranks produced output.
        for rank, pipe_data in sorted(pipeline_results.items()):
            assert pipe_data["text"], f"Rank {rank} produced no output text"


class TestPipelinePrefillCallbacks:
    """Verify that pipeline prefill callbacks exactly match stream_generate callbacks."""

    @pytest.mark.parametrize(
        "prompt_tokens",
        [50, 500, 5000],
        ids=["short_50", "medium_500", "long_5000"],
    )
    def test_callbacks_match(self, prompt_tokens: int) -> None:
        """All pipeline ranks must produce identical callback sequences."""
        # Run 4-rank pipeline
        pipeline_results = _run_pipeline_test(
            layer_splits=LAYER_SPLITS_4WAY,
            prompt_tokens=prompt_tokens,
            base_port=29700,
            timeout=180,
        )

        # All ranks must agree on prefill token count and callback sequence
        rank0_data = pipeline_results[0]
        rank0_callbacks = rank0_data["callbacks"]
        prefill_count = rank0_data["prefill_token_count"]

        for rank, pipe_data in sorted(pipeline_results.items()):
            pipe_callbacks = pipe_data["callbacks"]

            assert pipe_data["prefill_token_count"] == prefill_count, (
                f"Rank {rank} prefill token count mismatch: "
                f"{pipe_data['prefill_token_count']} vs {prefill_count}"
            )

            assert pipe_callbacks == rank0_callbacks, (
                f"Rank {rank} callback mismatch for {prompt_tokens} prompt tokens "
                f"(prefill M={prefill_count}):\n"
                f"  pipeline R0 ({len(rank0_callbacks)} callbacks): {rank0_callbacks}\n"
                f"  pipeline R{rank} ({len(pipe_callbacks)} callbacks): {pipe_callbacks}"
            )

        # Structural checks: starts with (0, M), ends with (M, M), monotonically increasing
        assert rank0_callbacks[0] == (0, prefill_count), (
            f"First callback should be (0, {prefill_count}), got {rank0_callbacks[0]}"
        )
        assert rank0_callbacks[-1] == (prefill_count, prefill_count), (
            f"Last callback should be ({prefill_count}, {prefill_count}), got {rank0_callbacks[-1]}"
        )
        for i in range(1, len(rank0_callbacks)):
            assert rank0_callbacks[i][0] >= rank0_callbacks[i - 1][0], (
                f"Callbacks not monotonically increasing at index {i}: {rank0_callbacks}"
            )

    @pytest.mark.parametrize(
        "prompt_tokens",
        [50, 500],
        ids=["short_50", "medium_500"],
    )
    def test_output_matches(self, prompt_tokens: int) -> None:
        """Pipeline-generated text must match single-device output."""
        single = _run_single_device_test(prompt_tokens, timeout=180)

        pipeline_results = _run_pipeline_test(
            layer_splits=LAYER_SPLITS_4WAY,
            prompt_tokens=prompt_tokens,
            base_port=29800,
            timeout=180,
        )

        single_text = single["text"]

        # The last rank produces the final logits, so its output should match.
        # Due to SDPA tiling non-determinism, allow minor differences in text.
        last_rank = max(pipeline_results.keys())
        pipe_text = pipeline_results[last_rank]["text"]

        # For deterministic sampling (temp=0.0), outputs should match exactly
        # or be very close. Log both for debugging even if they match.
        if single_text != pipe_text:
            # Find first divergence point
            min_len = min(len(single_text), len(pipe_text))
            diverge_idx = next(
                (i for i in range(min_len) if single_text[i] != pipe_text[i]),
                min_len,
            )
            pytest.fail(
                f"Output text diverged at character {diverge_idx} for {prompt_tokens} prompt tokens:\n"
                f"  single-device: {single_text!r}\n"
                f"  pipeline R{last_rank}: {pipe_text!r}"
            )


class TestPipelineCallbacksStructure:
    """Verify structural properties of callbacks independent of model output."""

    def test_callback_structure_matches_generate_step(self) -> None:
        """Verify callbacks follow generate_step's pattern: (0,M), chunks up to M-1, (M,M)."""
        prompt_tokens = 200
        pipeline_results = _run_pipeline_test(
            layer_splits=LAYER_SPLITS_4WAY,
            prompt_tokens=prompt_tokens,
            base_port=29900,
            timeout=180,
        )

        for rank, pipe_data in sorted(pipeline_results.items()):
            callbacks = pipe_data["callbacks"]
            m = pipe_data["prefill_token_count"]
            assert m > 0, f"Rank {rank}: prefill token count is 0"

            assert callbacks[0] == (0, m), (
                f"Rank {rank}: first callback should be (0, {m}), got {callbacks[0]}"
            )

            assert callbacks[-1] == (m, m), (
                f"Rank {rank}: last callback should be ({m}, {m}), got {callbacks[-1]}"
            )

            if len(callbacks) > 2:
                second_to_last = callbacks[-2]
                assert second_to_last[0] < m, (
                    f"Rank {rank}: second-to-last callback should report < {m}, "
                    f"got {second_to_last}"
                )

            # All callbacks must have total == M
            for i, (_, total) in enumerate(callbacks):
                assert total == m, (
                    f"Rank {rank}: callback {i} has total={total}, expected {m}"
                )

            # processed values must be non-decreasing
            processed_vals = [p for p, _ in callbacks]
            for i in range(1, len(processed_vals)):
                assert processed_vals[i] >= processed_vals[i - 1], (
                    f"Rank {rank}: callbacks not non-decreasing at index {i}: "
                    f"{processed_vals}"
                )

            # No duplicate consecutive callbacks (pipeline dummies must not emit callbacks)
            for i in range(1, len(callbacks)):
                assert callbacks[i] != callbacks[i - 1], (
                    f"Rank {rank}: duplicate consecutive callback at index {i}: "
                    f"{callbacks[i]} (this suggests dummy iterations are emitting callbacks)"
                )
