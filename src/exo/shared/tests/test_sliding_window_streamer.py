import pytest

from exo.shared.types.streaming_layer import (
    LayerDiskMetadata,
    LayerStreamingConfig,
)
from exo.worker.engines.streaming.layer_streamer import SlidingWindowLayerStreamer


def test_pipeline_initialization() -> None:
    streamer = SlidingWindowLayerStreamer()
    layer_size = 1024 * 1024  # 1 MB

    for i in range(4):
        streamer.register_layer(
            LayerDiskMetadata(
                layer_idx=i,
                file_path=f"/models/weights/layer_{i}.bin",
                size_bytes=layer_size,
            )
        )

    buf = streamer.start_pipeline(initial_layer_idx=0)
    assert buf.layer_idx == 0
    assert buf.is_active is True
    assert buf.is_ready is True
    assert streamer.current_active_layer == 0
    assert len(streamer.get_active_layer_bytes()) == layer_size


def test_sequential_layer_rotation_and_eviction() -> None:
    streamer = SlidingWindowLayerStreamer()
    num_layers = 6
    layer_size = 512 * 1024

    for i in range(num_layers):
        streamer.register_layer(
            LayerDiskMetadata(
                layer_idx=i,
                file_path=f"/disk/layer_{i}.bin",
                size_bytes=layer_size,
            )
        )

    streamer.start_pipeline(0)
    visited_layers = [0]

    for _ in range(num_layers - 1):
        next_buf = streamer.step_layer(compute_duration_ms=10.0)
        assert next_buf is not None
        visited_layers.append(next_buf.layer_idx)

    assert visited_layers == list(range(num_layers))

    # Stepping past the last layer should return None
    assert streamer.step_layer() is None


def test_peak_memory_capped_at_two_layers() -> None:
    config = LayerStreamingConfig(enable_clean_page_eviction=True)
    streamer = SlidingWindowLayerStreamer(config)

    # Simulate 32 layers of 10 MB each (total 320 MB)
    num_layers = 32
    layer_bytes = 10 * 1024 * 1024  # 10 MB

    for i in range(num_layers):
        streamer.register_layer(
            LayerDiskMetadata(
                layer_idx=i,
                file_path=f"/ssd/llama_70b/layer_{i}.safetensors",
                size_bytes=layer_bytes,
            )
        )

    streamer.start_pipeline(0)
    for _ in range(num_layers - 1):
        streamer.step_layer(compute_duration_ms=5.0)

    telemetry = streamer.get_telemetry()
    assert telemetry.total_layers_processed == num_layers
    assert telemetry.total_bytes_streamed == num_layers * layer_bytes

    # Memory residence in RAM must be strictly capped at 2 layers (20 MB), never expanding to 320 MB!
    assert telemetry.peak_memory_resident_bytes <= 2 * layer_bytes
    assert telemetry.io_compute_overlap_ratio > 0.0


def test_missing_layer_error_handling() -> None:
    streamer = SlidingWindowLayerStreamer()
    with pytest.raises(KeyError):
        streamer.start_pipeline(initial_layer_idx=999)
