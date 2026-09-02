# pyright: reportAny=false, reportPrivateUsage=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
# pyright: reportCallIssue=false, reportOptionalSubscript=false

"""Tests for RingAttentionLayer and ring-attention causal masking logic."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from copy import deepcopy
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map
from mlx_lm.models import llama, qwen3
from mlx_lm.models.cache import BatchKVCache, KVCache, QuantizedKVCache

import exo.worker.engines.mlx.generator.generate as generate_module
from exo.worker.engines.mlx.generator.generate import prefill
from exo.worker.engines.mlx.ring_attention import (
    RingAttentionLayer,
    _is_attention_layer,
    _make_block_causal_mask,
    ring_auto_parallel,
    set_ring_prefill,
    validate_ring_cache,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


class TestBlockCausalMask:
    def test_kv_entirely_before_returns_none(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=4, kv_seq_len=4, local_offset=4, kv_offset=0, n_heads=2
        )
        assert result is None

    def test_kv_entirely_after_returns_all_false(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=4, kv_seq_len=4, local_offset=0, kv_offset=4, n_heads=2
        )
        assert result is not None
        mx.eval(result)
        assert result.shape == (2, 4, 4)
        assert not bool(mx.any(result).item())

    def test_same_block_causal_mask(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=4, kv_seq_len=4, local_offset=0, kv_offset=0, n_heads=2
        )
        assert result is not None
        mx.eval(result)
        mask = result.tolist()
        for i in range(4):
            for j in range(4):
                assert mask[0][i][j] == (i >= j)

    def test_overlapping_blocks(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=4, kv_seq_len=8, local_offset=4, kv_offset=0, n_heads=1
        )
        assert result is not None
        mx.eval(result)
        mask = result.tolist()
        for i in range(4):
            for j in range(8):
                assert mask[0][i][j] == ((4 + i) >= j)

    def test_mask_shape(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=8, kv_seq_len=6, local_offset=0, kv_offset=0, n_heads=4
        )
        assert result is not None
        assert result.shape == (1, 8, 6)

    def test_mask_dtype_is_bool(self) -> None:
        result = _make_block_causal_mask(
            local_seq_len=4, kv_seq_len=4, local_offset=0, kv_offset=0, n_heads=1
        )
        assert result is not None
        assert result.dtype == mx.bool_


class TestRingAttentionLayerAttributes:
    def test_delegates_attributes(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)
        assert wrapped.custom_attr == "test_value"
        assert wrapped.use_sliding is True

    def test_rank_and_world_size(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)
        assert wrapped.rank == group.rank()
        assert wrapped.world_size == group.size()

    def test_is_prefill_default_false(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)
        assert wrapped.is_prefill is False

    def test_missing_attribute_raises(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)
        with pytest.raises(AttributeError):
            _ = wrapped.nonexistent_attr


class TestRingAttentionLayerSingleDevice:
    def test_single_device_delegates_to_original(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)
        x = mx.ones((1, 4))
        result = wrapped(x)
        mx.eval(result)
        assert (result == 2.0).all()


class TestOnlineAttentionMerge:
    def test_block_merge_matches_full_causal_attention(self) -> None:
        """Ring blocks must share one softmax denominator, not be summed."""
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(MockLayer(), group=group)

        queries = mx.array([[[[0.5, 1.0], [1.5, -0.5]]]])
        first_keys = mx.array([[[[0.0, 1.0], [1.0, 0.0]]]])
        first_values = mx.array([[[[2.0, 0.0], [0.0, 3.0]]]])
        second_keys = mx.array([[[[1.0, 1.0], [-1.0, 0.5]]]])
        second_values = mx.array([[[[1.0, 1.0], [4.0, -1.0]]]])

        running_max, running_sum, running_output = wrapped._accumulate_attention(
            queries,
            first_keys,
            first_values,
            scale=0.5,
            mask=None,
            running_max=None,
            running_sum=None,
            running_output=None,
        )
        second_mask = mx.array([[[True, False], [True, True]]])
        _, running_sum, running_output = wrapped._accumulate_attention(
            queries,
            second_keys,
            second_values,
            scale=0.5,
            mask=second_mask,
            running_max=running_max,
            running_sum=running_sum,
            running_output=running_output,
        )

        merged = running_output / running_sum
        full_keys = mx.concatenate([first_keys, second_keys], axis=2)
        full_values = mx.concatenate([first_values, second_values], axis=2)
        full_mask = mx.array([[[True, True, True, False], [True, True, True, True]]])
        expected = mx.fast.scaled_dot_product_attention(
            queries, full_keys, full_values, scale=0.5, mask=full_mask
        )

        mx.eval(merged, expected)
        assert mx.allclose(merged, expected, atol=1e-5).item()


class TestRingPipelineScheduling:
    def test_posts_communication_and_schedules_compute_before_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeGroup:
            def rank(self) -> int:
                return 0

            def size(self) -> int:
                return 2

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.n_heads = 2
                self.n_kv_heads = 2
                self.head_dim = 4
                self.scale = 0.5

        events: list[str] = []
        array_kinds: dict[int, str] = {}

        def fake_send(
            value: mx.array,
            destination: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            del destination, group, stream
            result = value + 0
            array_kinds[id(result)] = "send"
            events.append("construct_send")
            return result

        def fake_receive(
            template: mx.array,
            source: int,
            *,
            group: object,
            stream: mx.Stream,
        ) -> mx.array:
            del source, group, stream
            result = template + 0
            array_kinds[id(result)] = "receive"
            events.append("construct_receive")
            return result

        def fake_async_eval(*arrays: object) -> None:
            kinds = {
                array_kinds.get(id(array))
                for array in arrays
                if isinstance(array, mx.array)
            }
            if kinds == {"send"}:
                events.append("schedule_send")
            elif kinds == {"receive"}:
                events.append("schedule_receive")
            else:
                events.append("schedule_compute")

        def fake_eval(*arrays: object) -> None:
            kinds = {
                array_kinds.get(id(array))
                for array in arrays
                if isinstance(array, mx.array)
            }
            events.append("wait_receive" if kinds == {"receive"} else "wait_send")

        monkeypatch.setattr(mx.distributed, "send", fake_send)
        monkeypatch.setattr(mx.distributed, "recv_like", fake_receive)
        monkeypatch.setattr(mx, "async_eval", fake_async_eval)
        monkeypatch.setattr(mx, "eval", fake_eval)

        wrapped = RingAttentionLayer(FakeAttention(), FakeGroup())

        def equal_block_lengths(local_sequence_length: int) -> list[int]:
            del local_sequence_length
            return [2, 2]

        monkeypatch.setattr(wrapped, "_block_lengths", equal_block_lengths)
        wrapped._prefill_step(mx.ones((1, 2, 8)), cache=KVCache())

        first_compute = events.index("schedule_compute")
        first_wait = events.index("wait_receive")
        assert events.index("schedule_send") < first_compute
        assert events.index("schedule_receive") < first_compute
        assert first_compute < first_wait
        assert events[-1] == "wait_send"


class TestDecodePassthrough:
    """Decode must forward to the wrapped attention unchanged, so every cache
    type the model supports (including BatchKVCache) works during batched
    generation on ring instances."""

    class _TwoRankGroup:
        def rank(self) -> int:
            return 0

        def size(self) -> int:
            return 2

    @staticmethod
    def _tiny_llama_attention() -> nn.Module:
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
        return llama.Model(args).model.layers[0].self_attn

    def test_decode_with_batch_kv_cache_matches_unwrapped(self) -> None:
        attention = self._tiny_llama_attention()
        wrapped = RingAttentionLayer(attention, self._TwoRankGroup())

        wrapped_cache = BatchKVCache(left_padding=[1, 0])
        unwrapped_cache = BatchKVCache(left_padding=[1, 0])

        for step in range(3):
            x = mx.arange(2 * 16, dtype=mx.float32).reshape(2, 1, 16) / (16 + step)
            wrapped_out = wrapped(x, mask=None, cache=wrapped_cache)
            unwrapped_out = attention(x, mask=None, cache=unwrapped_cache)
            mx.eval(wrapped_out, unwrapped_out)
            assert wrapped_out.shape == (2, 1, 16)
            assert mx.array_equal(wrapped_out, unwrapped_out).item()

    def test_decode_with_standard_kv_cache_matches_unwrapped(self) -> None:
        attention = self._tiny_llama_attention()
        wrapped = RingAttentionLayer(attention, self._TwoRankGroup())

        wrapped_cache = KVCache()
        unwrapped_cache = KVCache()

        for step in range(3):
            x = mx.arange(16, dtype=mx.float32).reshape(1, 1, 16) / (16 + step)
            wrapped_out = wrapped(x, mask=None, cache=wrapped_cache)
            unwrapped_out = attention(x, mask=None, cache=unwrapped_cache)
            mx.eval(wrapped_out, unwrapped_out)
            assert mx.array_equal(wrapped_out, unwrapped_out).item()


class TestSetRingPrefill:
    def test_set_ring_prefill_toggles(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        wrapped = RingAttentionLayer(mock, group=group)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [wrapped]

        model = FakeModel()
        set_ring_prefill(model, is_prefill=True)
        assert wrapped.is_prefill is True
        set_ring_prefill(model, is_prefill=False)
        assert wrapped.is_prefill is False

    def test_set_ring_prefill_ignores_non_ring_layers(self) -> None:
        mock = MockLayer()
        group = mx.distributed.init()
        ring_layer = RingAttentionLayer(mock, group=group)
        plain_layer = MockLayer()

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [ring_layer, plain_layer]

        model = FakeModel()
        set_ring_prefill(model, is_prefill=True)
        assert ring_layer.is_prefill is True

    def test_prefill_restores_ring_state_after_unexpected_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapped = RingAttentionLayer(MockLayer(), group=mx.distributed.init())

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [wrapped]

        def fail_barrier(group: object) -> None:
            del group
            raise RuntimeError("simulated communication failure")

        def force_ring_prefill(*args: object) -> bool:
            del args
            return True

        def accept_cache(cache: object) -> None:
            del cache

        monkeypatch.setattr(
            generate_module, "uses_ring_sequence_parallel_prefill", force_ring_prefill
        )
        monkeypatch.setattr(generate_module, "validate_ring_cache", accept_cache)
        monkeypatch.setattr(generate_module, "mx_barrier", fail_barrier)

        with pytest.raises(RuntimeError, match="communication failure"):
            prefill(
                FakeModel(),
                None,
                None,
                mx.array([1, 2]),
                [],
                None,
                None,
                None,
            )

        assert wrapped.is_prefill is False


class TestIsAttentionLayer:
    def test_detects_q_proj(self) -> None:
        class FakeAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.n_heads = 2
                self.n_kv_heads = 2
                self.head_dim = 4

        assert _is_attention_layer(FakeAttn()) is True

    def test_detects_self_attn(self) -> None:
        class FakeDecoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Linear(8, 8)

        assert _is_attention_layer(FakeDecoder()) is True

    def test_detects_attn(self) -> None:
        class FakeLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.Linear(8, 8)

        assert _is_attention_layer(FakeLayer()) is True

    def test_rejects_plain_mlp(self) -> None:
        class FakeMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 8)
                self.down_proj = nn.Linear(8, 8)

        assert _is_attention_layer(FakeMLP()) is False

    def test_rejects_empty_module(self) -> None:
        class Empty(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        assert _is_attention_layer(Empty()) is False


class TestRingAutoParallel:
    def test_wraps_attention_layers(self) -> None:
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
        model = llama.Model(args)
        group = mx.distributed.init()
        responses = list(ring_auto_parallel(model, group, sequence_block_size=64))
        assert len(responses) == 2
        assert all(r.total == 2 for r in responses)
        assert [r.layers_loaded for r in responses] == [0, 1]
        inner = model.model
        assert isinstance(inner.layers[0].self_attn, RingAttentionLayer)
        assert inner.layers[0].self_attn.sequence_block_size == 64
        assert isinstance(inner.layers[1].self_attn, RingAttentionLayer)

    def test_preserves_non_attention_layers(self) -> None:
        class FakeMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 8)

            def __call__(self, x: mx.array, *a: object, **k: object) -> mx.array:
                return x

        class FakeInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [FakeMLP(), FakeMLP()]

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = FakeInner()

            def __call__(self, x: mx.array, *a: object, **k: object) -> mx.array:
                return x

        model = FakeModel()
        group = mx.distributed.init()
        with pytest.raises(ValueError, match="no supported attention layers"):
            list(ring_auto_parallel(model, group))


class TestRingCompatibility:
    def test_rejects_attention_without_direct_kv_projections(self) -> None:
        class UnsupportedAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = [UnsupportedAttention()]

        with pytest.raises(ValueError, match="direct Q/K/V projection"):
            list(ring_auto_parallel(FakeModel(), mx.distributed.init()))

    def test_rejects_unverified_direct_projection_attention(self) -> None:
        class CustomAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = [CustomAttention()]

        with pytest.raises(ValueError, match="only been verified"):
            list(ring_auto_parallel(FakeModel(), mx.distributed.init()))

    def test_rejects_parent_level_sliding_attention(self) -> None:
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
            sliding_window=16,
        )
        model = llama.Model(args)
        model.model.layers[0].use_sliding = True

        with pytest.raises(ValueError, match="use_sliding"):
            list(ring_auto_parallel(model, mx.distributed.init()))

    def test_rejects_quantized_kv_cache(self) -> None:
        with pytest.raises(TypeError, match="unquantized KVCache"):
            validate_ring_cache([QuantizedKVCache()])

    def test_qwen_qk_norm_matches_native_attention_preparation(self) -> None:
        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
            rope_theta=10000,
            head_dim=8,
            tie_word_embeddings=True,
        )
        attention = qwen3.Attention(args)
        wrapped = RingAttentionLayer(attention, mx.distributed.init())
        x = mx.arange(64, dtype=mx.float32).reshape(1, 4, 16) / 64

        actual_queries, actual_keys, _, _, _, _, _ = wrapped._project_qkv(attention, x)
        expected_queries = attention.q_norm(
            attention.q_proj(x).reshape(1, 4, attention.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        expected_keys = attention.k_norm(
            attention.k_proj(x).reshape(1, 4, attention.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)

        mx.eval(actual_queries, actual_keys, expected_queries, expected_keys)
        assert mx.allclose(actual_queries, expected_queries).item()
        assert mx.allclose(actual_keys, expected_keys).item()


def _run_ring_attention_device(
    rank: int, world_size: int, hostfile_path: str, result_queue: Any
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)

        class MockAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.n_heads = 2
                self.head_dim = 4
                self.scale = 0.5
                for projection in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.o_proj,
                ):
                    projection.weight = mx.arange(64, dtype=mx.float32).reshape(8, 8)
                    projection.bias = mx.zeros(8)

            def __call__(
                self, x: mx.array, *args: object, **kwargs: object
            ) -> mx.array:
                q = self.q_proj(x)
                keys = self.k_proj(x)
                v = self.v_proj(x)
                batch, seq, dim = q.shape
                q = q.reshape(batch, seq, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(batch, seq, self.n_heads, -1).transpose(0, 2, 1, 3)
                v = v.reshape(batch, seq, self.n_heads, -1).transpose(0, 2, 1, 3)
                scores = q @ keys.transpose(0, 1, 3, 2) * self.scale
                attn = mx.softmax(scores, axis=-1)
                out = attn @ v
                out = out.transpose(0, 2, 1, 3).reshape(batch, seq, dim)
                return self.o_proj(out)

        mock_attn = MockAttn()
        wrapped = RingAttentionLayer(mock_attn, group=group)
        wrapped.is_prefill = True
        x = (
            mx.arange(rank * 32, (rank + 1) * 32, dtype=mx.float32).reshape(1, 4, 8)
            / 64
        )
        result = wrapped(x)

        total_sequence_length = world_size * 4
        full_x = (
            mx.arange(total_sequence_length * 8, dtype=mx.float32).reshape(
                1, total_sequence_length, 8
            )
            / 64
        )
        queries = (
            mock_attn.q_proj(full_x)
            .reshape(1, total_sequence_length, 2, 4)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            mock_attn.k_proj(full_x)
            .reshape(1, total_sequence_length, 2, 4)
            .transpose(0, 2, 1, 3)
        )
        values = (
            mock_attn.v_proj(full_x)
            .reshape(1, total_sequence_length, 2, 4)
            .transpose(0, 2, 1, 3)
        )
        causal_mask = mx.tril(
            mx.ones((total_sequence_length, total_sequence_length), dtype=mx.bool_)
        )[None, :, :]
        expected = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=mock_attn.scale, mask=causal_mask
        )
        expected = expected[:, :, rank * 4 : (rank + 1) * 4, :]
        expected = expected.transpose(0, 2, 1, 3).reshape(1, 4, 8)
        expected = mock_attn.o_proj(expected)
        mx.eval(result)
        mx.eval(expected)
        result_queue.put(
            (rank, True, (result.shape, mx.max(mx.abs(result - expected)).item()))
        )
    except Exception as e:
        import traceback

        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def _run_ring_prefill_device(
    rank: int, world_size: int, hostfile_path: str, result_queue: Any
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        mx.random.seed(42)

        class MockAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.n_heads = 2
                self.head_dim = 4
                self.scale = 0.5
                for projection in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.o_proj,
                ):
                    projection.weight = mx.arange(64, dtype=mx.float32).reshape(8, 8)
                    projection.bias = mx.zeros(8)

        class FakeInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [
                    RingAttentionLayer(MockAttn(), group, sequence_block_size=2)
                ]

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = FakeInner()
                self.layers = self.model.layers

            def __call__(self, tokens: mx.array, cache: object) -> mx.array:
                embeddings = mx.eye(8)[tokens]
                return self.model.layers[0](embeddings, cache=cache[0])

        model = FakeModel()
        cache = [KVCache()]
        prompt = mx.array(list(range(8)))
        prefill(
            model,
            None,
            None,
            prompt,
            cache,
            group,
            None,
            None,
        )
        full_embeddings = mx.eye(8)[prompt[None]]
        original = model.layers[0].original_layer
        expected_keys = original.k_proj(full_embeddings)
        expected_keys = expected_keys.reshape(1, 8, 2, 4).transpose(0, 2, 1, 3)
        actual_keys = cache[0].keys[..., : cache[0].offset, :]
        mx.eval(expected_keys, actual_keys)
        result_queue.put(
            (
                rank,
                True,
                (
                    cache[0].offset,
                    mx.allclose(actual_keys, expected_keys).item(),
                    mx.max(mx.abs(actual_keys - expected_keys)).item(),
                ),
            )
        )
    except Exception as e:
        import traceback

        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def _run_llama_ring_device(
    rank: int, world_size: int, hostfile_path: str, result_queue: Any
) -> None:
    """Compare complete two-rank Ring inference against a real tiny Llama model."""
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        mx.random.seed(42)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
        baseline = llama.Model(args)

        def constant_parameter(parameter: mx.array) -> mx.array:
            return mx.full(parameter.shape, 0.01, dtype=parameter.dtype)

        baseline.update(tree_map(constant_parameter, baseline.parameters()))
        ring_model = deepcopy(baseline)
        list(ring_auto_parallel(ring_model, group, sequence_block_size=2))

        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        baseline_cache = baseline.make_cache()
        expected_prefill = baseline(prompt[None], cache=baseline_cache)

        ring_cache = ring_model.make_cache()
        set_ring_prefill(ring_model, is_prefill=True)
        start = (len(prompt) * rank) // world_size
        end = (len(prompt) * (rank + 1)) // world_size
        actual_prefill = ring_model(prompt[start:end][None], cache=ring_cache)
        set_ring_prefill(ring_model, is_prefill=False)

        expected_prefill = expected_prefill[:, start:end, :]
        next_token = mx.array([[9]])
        expected_decode = baseline(next_token, cache=baseline_cache)
        actual_decode = ring_model(next_token, cache=ring_cache)
        mx.eval(actual_prefill, expected_prefill, actual_decode, expected_decode)

        cache_offsets = [entry.offset for entry in ring_cache]
        result_queue.put(
            (
                rank,
                True,
                (
                    mx.max(mx.abs(actual_prefill - expected_prefill)).item(),
                    mx.max(mx.abs(actual_decode - expected_decode)).item(),
                    cache_offsets,
                ),
            )
        )
    except Exception as e:
        import traceback

        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))


class TestRingAttentionDistributed:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_multi_device_ring_attention_completes(self, world_size: int) -> None:
        ctx = mp.get_context("spawn")
        starting_port = 29700 + world_size * 10
        hosts = [f"127.0.0.1:{starting_port + i}" for i in range(world_size)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hosts, f)
            hostfile_path = f.name
        processes: list[Any] = []
        try:
            result_queue: Any = ctx.Queue()
            for rank in range(world_size):
                p = ctx.Process(
                    target=_run_ring_attention_device,
                    args=(rank, world_size, hostfile_path, result_queue),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(timeout=30)
            results: dict[int, Any] = {}
            errors: dict[int, str] = {}
            while not result_queue.empty():
                rank, success, value = result_queue.get()
                if success:
                    results[rank] = value
                else:
                    errors[rank] = value
            assert len(results) == world_size, f"Errors: {errors}"
            for rank in range(world_size):
                assert rank in results, f"Device {rank} failed: {errors.get(rank)}"
                shape, maximum_error = results[rank]
                assert shape == (1, 4, 8)
                assert maximum_error < 1e-5
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            os.unlink(hostfile_path)

    def test_single_device_matches_original(self) -> None:
        group = mx.distributed.init()

        class MockAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.o_proj = nn.Linear(8, 8)
                self.n_heads = 2
                self.head_dim = 4
                self.scale = 0.5

            def __call__(self, x: mx.array, *a: object, **k: object) -> mx.array:
                return self.o_proj(x * 3)

        mock_attn = MockAttn()
        wrapped = RingAttentionLayer(mock_attn, group=group)
        x = mx.ones((1, 4, 8))
        result = wrapped(x)
        mx.eval(result)
        expected = mock_attn(mx.ones((1, 4, 8)))
        mx.eval(expected)
        assert mx.allclose(result, expected).item()

    def test_prefill_partitions_prompt_and_populates_full_cache(self) -> None:
        ctx = mp.get_context("spawn")
        world_size = 2
        hosts = [f"127.0.0.1:{29800 + i}" for i in range(world_size)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hosts, f)
            hostfile_path = f.name
        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []
            for rank in range(world_size):
                process = ctx.Process(
                    target=_run_ring_prefill_device,
                    args=(rank, world_size, hostfile_path, result_queue),
                )
                process.start()
                processes.append(process)
            for process in processes:
                process.join(timeout=30)

            results: dict[int, Any] = {}
            errors: dict[int, str] = {}
            while not result_queue.empty():
                rank, success, value = result_queue.get()
                if success:
                    results[rank] = value
                else:
                    errors[rank] = value

            assert results == {0: (8, True, 0.0), 1: (8, True, 0.0)}, (
                f"Errors: {errors}"
            )
        finally:
            os.unlink(hostfile_path)

    def test_tiny_llama_prefill_and_decode_match_single_device(self) -> None:
        ctx = mp.get_context("spawn")
        world_size = 2
        hosts = [f"127.0.0.1:{29900 + i}" for i in range(world_size)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hosts, f)
            hostfile_path = f.name
        try:
            result_queue: Any = ctx.Queue()
            processes: list[Any] = []
            for rank in range(world_size):
                process = ctx.Process(
                    target=_run_llama_ring_device,
                    args=(rank, world_size, hostfile_path, result_queue),
                )
                process.start()
                processes.append(process)
            for process in processes:
                process.join(timeout=30)

            results: dict[int, Any] = {}
            errors: dict[int, str] = {}
            while not result_queue.empty():
                rank, success, value = result_queue.get()
                if success:
                    results[rank] = value
                else:
                    errors[rank] = value

            assert len(results) == world_size, f"Errors: {errors}"
            for prefill_error, decode_error, cache_offsets in results.values():
                assert prefill_error < 1e-5
                assert decode_error < 1e-5
                assert cache_offsets == [9]
        finally:
            os.unlink(hostfile_path)
