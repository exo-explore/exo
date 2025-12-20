
import pytest
from exo.worker.download.impl_shard_downloader import exo_shard_downloader, KNOWN_CUSTOM_MODELS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.worker.shards import PipelineShardMetadata

@pytest.mark.asyncio
async def test_custom_model_download_tracking():
    # Use a small validation model
    model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    initial_custom_count = len(KNOWN_CUSTOM_MODELS)
    
    downloader = exo_shard_downloader()
    
    # 1. Trigger ensure_shard
    # We need to build a dummy shard metadata for this model
    model_meta = await get_model_meta(model_id)
    shard = PipelineShardMetadata(
        model_meta=model_meta,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_meta.n_layers,
        n_layers=model_meta.n_layers,
    )
    
    # Start the download (this will add it to tracking)
    await downloader.ensure_shard(shard)
    
    # 2. Verify it was added to KNOWN_CUSTOM_MODELS
    assert model_id in KNOWN_CUSTOM_MODELS
    assert len(KNOWN_CUSTOM_MODELS) > initial_custom_count
    
    # 3. Verify it appears in status
    found_in_status = False
    async for _, progress in downloader.get_shard_download_status():
        if progress.repo_id == model_id:
            found_in_status = True
            break
            
    assert found_in_status, f"Model {model_id} not found in download status stream"
