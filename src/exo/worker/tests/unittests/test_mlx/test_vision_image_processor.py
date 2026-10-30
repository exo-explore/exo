"""The image processor loads even when processor_config.json names a class this
transformers release does not have (mlx-community's Qwen3.8 bf16/8bit/mxfp4/nvfp4/mxfp8
conversions name "Qwen3VLImageProcessor"); preprocessor_config.json names one it has."""

import json
from pathlib import Path

from exo.worker.engines.mlx.vision import (
    _load_auto_image_processor,  # pyright: ignore[reportPrivateUsage]
)

PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
PROCESSOR_CONFIG = {
    "image_processor": {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "Qwen3VLImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "max_pixels": 16777216,
        "merge_size": 2,
        "min_pixels": 65536,
        "patch_size": 16,
        "rescale_factor": 0.00392156862745098,
        "temporal_patch_size": 2,
    },
    "processor_class": "Qwen3VLProcessor",
}


def test_falls_back_to_preprocessor_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG))
    (tmp_path / "processor_config.json").write_text(json.dumps(PROCESSOR_CONFIG))

    processor = _load_auto_image_processor(str(tmp_path))

    assert type(processor).__name__.startswith("Qwen2VLImageProcessor")
    assert getattr(processor, "patch_size", None) == 16
    assert getattr(processor, "merge_size", None) == 2
