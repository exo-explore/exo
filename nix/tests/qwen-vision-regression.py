import sys

from exo.shared.models.model_cards import detect_vision_from_config
from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.vision import VisionEncoder

assert len(sys.argv) == 2, "Usage: qwen-vision-regression.py MODEL_ID"
model_id = ModelId(sys.argv[1])
vision_config = detect_vision_from_config(model_id)
assert vision_config is not None, "Expected local model vision metadata"
assert vision_config.model_type == "qwen3_5", "Expected the Qwen3.8 architecture"
encoder = VisionEncoder(vision_config, model_id)
encoder.ensure_loaded()
print(f"Qwen vision loader regression: PASS ({model_id})")
