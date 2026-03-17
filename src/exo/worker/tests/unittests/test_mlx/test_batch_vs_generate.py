import copy
import gc
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.mlx import MLXCacheType, Model
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import CacheSnapshot, KVPrefixCache, cache_length
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    load_tokenizer_for_model_id,
)

from .test_prefix_cache_architectures import (
    ARCHITECTURES,
    ArchSpec,
    _arch_available,  # pyright: ignore[reportPrivateUsage]
    _build_model,  # pyright: ignore[reportPrivateUsage]
    _copy_tokenizer,  # pyright: ignore[reportPrivateUsage]
    _find_snapshot,  # pyright: ignore[reportPrivateUsage]
    _reduce_config,  # pyright: ignore[reportPrivateUsage]
)


def _make_task(
    content: str = "Hello, what is 2+2?",
    max_tokens: int = 10,
    seed: int = 42,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test"),
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=max_tokens,
        temperature=0.7,
        seed=seed,
    )


# ── Helpers ──────────────────────────────────────────────────────────────── #


def _collect_mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    kv_prefix_cache: KVPrefixCache | None,
) -> list[int]:
    """Run mlx_generate and collect output token IDs."""
    prompt = apply_chat_template(tokenizer=tokenizer, task_params=task)
    tokens: list[int] = []
    for resp in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=task,
        prompt=prompt,
        kv_prefix_cache=kv_prefix_cache,
        group=None,
    ):
        tokens.append(resp.token)
        if resp.finish_reason is not None:
            break
    return tokens


def _collect_batch_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
    kv_prefix_cache: KVPrefixCache | None,
) -> list[int]:
    """Run ExoBatchGenerator and collect raw output token IDs"""
    exo_gen = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        kv_prefix_cache=kv_prefix_cache,
    )

    prompt = apply_chat_template(tokenizer=tokenizer, task_params=task_params)
    exo_gen.submit(
        task_id=TaskId("test-single"), task_params=task_params, prompt=prompt
    )

    tokens: list[int] = []
    while exo_gen.has_work:
        results = exo_gen.step()
        for _uid, response in results:
            tokens.append(response.token)

    exo_gen.close()
    return tokens


def _assert_state_equal(sa: object, sb: object, label: str) -> None:
    """Compare two state items, handling both plain arrays and tuples of arrays (CacheList)."""
    if isinstance(sa, tuple):
        assert isinstance(sb, tuple), f"{label}: type mismatch"
        for k, (arr_a, arr_b) in enumerate(
            zip(
                cast(tuple[mx.array, ...], sa),
                cast(tuple[mx.array, ...], sb),
                strict=True,
            )
        ):
            a_f = mx.array(arr_a).astype(mx.float32)
            b_f = mx.array(arr_b).astype(mx.float32)
            if a_f.size == 0:
                assert b_f.size == 0, f"{label}[{k}]: size mismatch"
                continue
            diff = float(mx.max(mx.abs(a_f - b_f)).item())
            assert diff == 0.0, f"{label}[{k}]: max diff {diff}"
    else:
        sa_f = mx.array(cast(mx.array, sa)).astype(mx.float32)
        sb_f = mx.array(cast(mx.array, sb)).astype(mx.float32)
        if sa_f.size == 0:
            assert sb_f.size == 0, f"{label}: size mismatch"
            return
        diff = float(mx.max(mx.abs(sa_f - sb_f)).item())
        assert diff == 0.0, f"{label}: max diff {diff}"


def _compare_cache_arrays(
    cache_a: MLXCacheType,
    cache_b: MLXCacheType,
    label: str = "",
) -> None:
    """Assert two KV caches have identical array values."""
    assert len(cache_a) == len(cache_b), (
        f"{label}Cache layer count: {len(cache_a)} vs {len(cache_b)}"
    )
    for i, (a, b) in enumerate(zip(cache_a, cache_b, strict=True)):
        assert type(a) is type(b), (
            f"{label}Layer {i}: type {type(a).__name__} vs {type(b).__name__}"
        )
        states_a = a.state
        states_b = b.state
        assert len(states_a) == len(states_b), (
            f"{label}Layer {i}: state count {len(states_a)} vs {len(states_b)}"
        )
        for j, (sa, sb) in enumerate(zip(states_a, states_b, strict=True)):
            if sa is None and sb is None:
                continue
            assert sa is not None and sb is not None, (
                f"{label}Layer {i}, state {j}: one is None"
            )
            _assert_state_equal(sa, sb, f"{label}Layer {i}, state {j}")


def _safe_state(cache: object) -> list[object]:
    """Safely access .state on a cache object. Returns [] if uninitialized."""
    # RotatingKVCache.state crashes when keys is None (uninitialized)
    if getattr(cache, "keys", _SENTINEL) is None:
        return []
    try:
        return list(cache.state)  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        return []


_SENTINEL = object()


def _compare_snapshots(
    snaps_a: list[CacheSnapshot] | None,
    snaps_b: list[CacheSnapshot] | None,
    label: str = "",
) -> None:
    """Assert two snapshot lists are identical."""
    if snaps_a is None:
        assert snaps_b is None, f"{label}One side has snapshots, other doesn't"
        return
    assert snaps_b is not None, f"{label}One side has snapshots, other doesn't"
    assert len(snaps_a) == len(snaps_b), (
        f"{label}Snapshot count: {len(snaps_a)} vs {len(snaps_b)}"
    )
    for k, (sa, sb) in enumerate(zip(snaps_a, snaps_b, strict=True)):
        assert sa.token_count == sb.token_count, (
            f"{label}Snapshot {k} token_count: {sa.token_count} vs {sb.token_count}"
        )
        for layer_i, (s1, s2) in enumerate(zip(sa.states, sb.states, strict=True)):
            if s1 is None and s2 is None:
                continue
            assert s1 is not None and s2 is not None, (
                f"{label}Snapshot {k}, layer {layer_i}: one state is None"
            )
            state_a = _safe_state(s1)
            state_b = _safe_state(s2)
            if not state_a and not state_b:
                continue
            assert len(state_a) == len(state_b), (
                f"{label}Snapshot {k}, layer {layer_i}: state length mismatch"
            )
            for st_j, (arr_a, arr_b) in enumerate(zip(state_a, state_b, strict=True)):
                if arr_a is None and arr_b is None:
                    continue
                assert arr_a is not None and arr_b is not None
                _assert_state_equal(
                    arr_a,
                    arr_b,
                    f"{label}Snapshot {k}, layer {layer_i}, state {st_j}",
                )


# ── Test class ────────────────────────────────────────────────────────────── #


@pytest.mark.slow
class TestBatchVsGenerate:
    """Verify BatchGenerator matches mlx_generate for output tokens and prefix cache."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        mx.clear_cache()
        gc.collect()

    @pytest.mark.parametrize(
        "spec",
        ARCHITECTURES,
        ids=[a.name for a in ARCHITECTURES],
    )
    def test_same_output_and_cache(self, spec: ArchSpec) -> None:
        if not _arch_available(spec):
            pytest.skip(f"Model {spec.hub_name} not cached locally")

        snapshot = _find_snapshot(spec.hub_name)
        assert snapshot is not None

        tmpdir = Path(tempfile.mkdtemp(prefix=f"exo_batchtest_{spec.name}_"))
        try:
            # Build reduced config
            with open(snapshot / "config.json") as f:
                cfg = cast(dict[str, Any], json.load(f))
            reduced = _reduce_config(copy.deepcopy(cfg))
            (tmpdir / "config.json").write_text(json.dumps(reduced))

            # Copy tokenizer
            tok_src = snapshot
            if spec.tokenizer_hub is not None:
                alt = _find_snapshot(spec.tokenizer_hub)
                if alt is not None:
                    tok_src = alt
            _copy_tokenizer(tok_src, tmpdir)

            # Load tokenizer, build model with random weights
            model_id = ModelId(f"mlx-community/{spec.hub_name}")
            tokenizer = load_tokenizer_for_model_id(model_id, tmpdir)
            mx.random.seed(0)
            model = _build_model(spec.module, reduced)

            task = _make_task()

            # ── Run mlx_generate path ──
            # Seed is set inside mlx_generate/ExoBatchGenerator.submit from task.seed
            kv_mlx = KVPrefixCache(None)
            mlx_tokens = _collect_mlx_generate(model, tokenizer, task, kv_mlx)

            # ── Run batch generator path ──
            kv_batch = KVPrefixCache(None)
            batch_tokens = _collect_batch_generate(model, tokenizer, task, kv_batch)

            # ── Compare output tokens ──
            assert len(mlx_tokens) > 0, "mlx_generate produced no tokens"
            assert len(batch_tokens) > 0, "BatchGenerator produced no tokens"
            assert mlx_tokens == batch_tokens, (
                f"[{spec.name}] Token mismatch:\n"
                f"  mlx_generate:    {mlx_tokens}\n"
                f"  BatchGenerator:  {batch_tokens}"
            )

            # ── Compare prefix cache KV arrays ──
            assert len(kv_mlx.caches) == 1, "mlx_generate didn't save to prefix cache"
            assert len(kv_batch.caches) == 1, (
                "BatchGenerator didn't save to prefix cache"
            )

            mlx_cache = kv_mlx._get_mlx_cache(0)  # pyright: ignore[reportPrivateUsage]
            batch_cache = kv_batch._get_mlx_cache(0)  # pyright: ignore[reportPrivateUsage]

            _compare_cache_arrays(
                mlx_cache,
                batch_cache,
                label=f"[{spec.name}] ",
            )

            # ── Compare cache lengths ──
            mlx_len = cache_length(mlx_cache)
            batch_len = cache_length(batch_cache)
            assert mlx_len == batch_len, (
                f"[{spec.name}] Cache length: mlx={mlx_len} vs batch={batch_len}"
            )

            # ── Compare snapshots ──
            _compare_snapshots(
                kv_mlx._snapshots[0],  # pyright: ignore[reportPrivateUsage]
                kv_batch._snapshots[0],  # pyright: ignore[reportPrivateUsage]
                label=f"[{spec.name}] ",
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.parametrize(
        "spec",
        ARCHITECTURES,
        ids=[a.name for a in ARCHITECTURES],
    )
    def test_concurrent_batch_completes(self, spec: ArchSpec) -> None:
        """Two requests processed concurrently must both complete without
        crashing and produce non-empty output.

        Note: batch decode logits are NOT bit-exact with sequential because
        Metal's matmul kernel picks different reduction tiling for B=1 vs B=2
        when L=1 (decode step). This introduces sub-ULP float16 diffs in
        gate_proj/down_proj/lm_head which swiglu amplifies by |up_values|.
        With random weights these accumulate into argmax flips; with trained
        weights the diffs are absorbed and output matches exactly (verified
        with real Llama-3.2-1B-Instruct-4bit weights).
        """
        if not _arch_available(spec):
            pytest.skip(f"Model {spec.hub_name} not cached locally")

        snapshot = _find_snapshot(spec.hub_name)
        assert snapshot is not None

        tmpdir = Path(tempfile.mkdtemp(prefix=f"exo_concurrent_{spec.name}_"))
        try:
            with open(snapshot / "config.json") as f:
                cfg = cast(dict[str, Any], json.load(f))
            reduced = _reduce_config(copy.deepcopy(cfg))
            (tmpdir / "config.json").write_text(json.dumps(reduced))

            tok_src = snapshot
            if spec.tokenizer_hub is not None:
                alt = _find_snapshot(spec.tokenizer_hub)
                if alt is not None:
                    tok_src = alt
            _copy_tokenizer(tok_src, tmpdir)

            model_id = ModelId(f"mlx-community/{spec.hub_name}")
            tokenizer = load_tokenizer_for_model_id(model_id, tmpdir)
            mx.random.seed(0)
            model = _build_model(spec.module, reduced)

            # Two different prompts → different prompt lengths.
            task_a = _make_task(content="Hello, what is 2+2?", seed=42)
            task_a = task_a.model_copy(update={"temperature": 0.0})
            task_b = _make_task(
                content="Write a short poem about the ocean and the sky.",
                seed=99,
            )
            task_b = task_b.model_copy(update={"temperature": 0.0})

            # ── Concurrent: submit both to one ExoBatchGenerator ──
            exo_gen = ExoBatchGenerator(
                model=model,
                tokenizer=tokenizer,
                group=None,
                kv_prefix_cache=None,
            )

            prompt_a = apply_chat_template(tokenizer=tokenizer, task_params=task_a)
            prompt_b = apply_chat_template(tokenizer=tokenizer, task_params=task_b)
            tid_a = exo_gen.submit(
                task_id=TaskId("batch-a"), task_params=task_a, prompt=prompt_a
            )
            tid_b = exo_gen.submit(
                task_id=TaskId("batch-b"), task_params=task_b, prompt=prompt_b
            )

            batch_tokens: dict[str, list[int]] = {tid_a: [], tid_b: []}
            finished: set[str] = set()
            while exo_gen.has_work:
                results = exo_gen.step()
                for tid, response in results:
                    batch_tokens[tid].append(response.token)
                    if response.finish_reason is not None:
                        finished.add(tid)

            exo_gen.close()

            # ── Verify both completed ──
            assert len(batch_tokens[tid_a]) > 0, "No tokens for task A"
            assert len(batch_tokens[tid_b]) > 0, "No tokens for task B"
            assert tid_a in finished, "Task A never finished"
            assert tid_b in finished, "Task B never finished"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
