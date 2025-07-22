from pathlib import Path

import pytest

from shared.types.models import ModelMetadata
from shared.types.worker.shards import PipelineShardMetadata
from worker.download.model_meta import _get_model_meta  # type: ignore


@pytest.fixture
def model_meta() -> ModelMetadata:
    return _get_model_meta('mlx-community/Llama-3.2-1B-Instruct-4bit') # type: ignore


@pytest.fixture
def pipeline_shard_meta(model_meta: ModelMetadata, tmp_path: Path):
    def _pipeline_shard_meta(
        num_nodes: int = 1, device_rank: int = 0
    ) -> PipelineShardMetadata:
        total_layers = 16
        layers_per_node = total_layers // num_nodes
        start_layer = device_rank * layers_per_node
        end_layer = (
            start_layer + layers_per_node
            if device_rank < num_nodes - 1
            else total_layers
        )

        return PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=device_rank,
            n_layers=total_layers,
            start_layer=start_layer,
            end_layer=end_layer,
            world_size=num_nodes,
        )

    return _pipeline_shard_meta