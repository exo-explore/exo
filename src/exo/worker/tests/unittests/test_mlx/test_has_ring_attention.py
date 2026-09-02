"""Tests for has_ring_attention_layer detection helper."""

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.ring_attention import (
    RingAttentionLayer,
    has_ring_attention_layer,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


class TestHasRingAttentionLayer:
    def test_detects_ring_layer(self) -> None:
        group = mx.distributed.init()
        ring_layer = RingAttentionLayer(MockLayer(), group=group)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [ring_layer]

        assert has_ring_attention_layer(FakeModel()) is True

    def test_no_ring_layer_returns_false(self) -> None:
        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [MockLayer(), MockLayer()]

        assert has_ring_attention_layer(FakeModel()) is False

    def test_mixed_layers_detected(self) -> None:
        group = mx.distributed.init()
        ring_layer = RingAttentionLayer(MockLayer(), group=group)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [MockLayer(), ring_layer, MockLayer()]

        assert has_ring_attention_layer(FakeModel()) is True
