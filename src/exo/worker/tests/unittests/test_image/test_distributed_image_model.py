from __future__ import annotations

from pathlib import Path

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.image.distributed_model import DistributedImageModel


def test_distributed_image_model_raises_when_path_missing(tmp_path: Path) -> None:
    model_id = ModelId("exolabs/FLUX.1-dev")
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
    )
    shard_metadata = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=r"Model path does not exist"):
        _ = DistributedImageModel(
            model_id=str(model_id),
            local_path=missing,
            shard_metadata=shard_metadata,
            group=None,
        )
