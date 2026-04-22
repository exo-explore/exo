"""Tests for safetensors index file parsing."""

from exo.shared.types.worker.downloads import ModelSafetensorsIndex


def test_safetensors_index_missing_total_size():
    """Index files from image models (e.g. FLUX/mflux) have metadata without
    a total_size field — they use quantization_level and mflux_version instead.

    This must parse successfully — a previous bug caused Pydantic validation to
    fail silently (total_size was a required PositiveInt), which made
    _scan_model_directory skip weight-map checks and report incomplete models
    as complete.
    """
    raw = '{"metadata": {"quantization_level": "4", "mflux_version": "0.3.0"}, "weight_map": {"layer.safetensors": "layer.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is not None
    assert index.metadata.total_size is None
    assert index.weight_map == {"layer.safetensors": "layer.safetensors"}


def test_safetensors_index_valid_total_size():
    """Standard text model index files with a valid total_size should continue
    to parse correctly."""
    raw = '{"metadata": {"total_size": 12345}, "weight_map": {"a.safetensors": "a.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is not None
    assert index.metadata.total_size == 12345


def test_safetensors_index_null_metadata():
    """Index files with null metadata should parse correctly."""
    raw = '{"metadata": null, "weight_map": {"a.safetensors": "a.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is None
