from exo.worker.engines.image.config import (
    BlockType,
    ImageModelConfig,
    TransformerBlockConfig,
)

QWEN_IMAGE_CONFIG = ImageModelConfig(
    model_family="qwen",
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=60, has_separate_text_output=True
        ),
    ),
    default_steps={"low": 10, "medium": 25, "high": 50},
    num_sync_steps_factor=0.125,  # ~3 sync steps for medium (30 steps)
    guidance_scale=3.5,  # Set to None or < 1.0 to disable CFG
)

QWEN_IMAGE_EDIT_CONFIG = ImageModelConfig(
    model_family="qwen-edit",
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=60, has_separate_text_output=True
        ),
    ),
    default_steps={"low": 10, "medium": 25, "high": 50},
    num_sync_steps_factor=0.125,
    guidance_scale=3.5,
)
