"""Tests for RingShardMetadata and the Ring sharding enum."""

import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import (
    RingShardMetadata,
    Sharding,
)


def _make_model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/ring-model"),
        storage_size=Memory.from_gb(1),
        n_layers=4,
        hidden_size=128,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


class TestShardingEnum:
    def test_ring_exists(self) -> None:
        assert Sharding.Ring == "Ring"

    def test_ring_is_distinct_from_others(self) -> None:
        assert Sharding.Ring != Sharding.Tensor
        assert Sharding.Ring != Sharding.Pipeline

    def test_all_sharding_values(self) -> None:
        values = {Sharding.Tensor, Sharding.Pipeline, Sharding.Ring}
        assert len(values) == 3


class TestRingShardMetadata:
    def test_basic_construction(self) -> None:
        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert meta.device_rank == 0
        assert meta.world_size == 2
        assert meta.sequence_block_size == 512

    def test_custom_block_size(self) -> None:
        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=1,
            world_size=3,
            start_layer=0,
            end_layer=4,
            n_layers=4,
            sequence_block_size=1024,
        )
        assert meta.sequence_block_size == 1024

    def test_block_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RingShardMetadata(
                model_card=_make_model_card(),
                device_rank=0,
                world_size=2,
                start_layer=0,
                end_layer=4,
                n_layers=4,
                sequence_block_size=0,
            )

    def test_block_size_must_be_positive_int(self) -> None:
        with pytest.raises(ValidationError):
            RingShardMetadata(
                model_card=_make_model_card(),
                device_rank=0,
                world_size=2,
                start_layer=0,
                end_layer=4,
                n_layers=4,
                sequence_block_size=-1,
            )

    def test_is_first_layer(self) -> None:
        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert meta.is_first_layer is True

    def test_is_last_layer(self) -> None:
        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert meta.is_last_layer is True

    def test_hash_is_deterministic(self) -> None:
        meta1 = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        meta2 = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert hash(meta1) == hash(meta2)

    def test_hash_differs_by_rank(self) -> None:
        meta0 = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        meta1 = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=1,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert hash(meta0) != hash(meta1)

    def test_frozen(self) -> None:
        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        with pytest.raises(ValidationError):
            meta.sequence_block_size = 256

    def test_isinstance_in_shard_metadata_union(self) -> None:
        from exo.shared.types.worker.shards import ShardMetadata

        meta = RingShardMetadata(
            model_card=_make_model_card(),
            device_rank=0,
            world_size=2,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        assert isinstance(meta, ShardMetadata)
