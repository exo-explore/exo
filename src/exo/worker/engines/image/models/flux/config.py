from exo.worker.engines.image.config import (
    BlockType,
    ImageModelConfig,
    TransformerBlockConfig,
)

FLUX_SCHNELL_CONFIG = ImageModelConfig(
    model_family="flux",
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=19, has_separate_text_output=True
        ),
        TransformerBlockConfig(
            block_type=BlockType.SINGLE, count=38, has_separate_text_output=False
        ),
    ),
    default_steps={"low": 1, "medium": 2, "high": 4},
    num_sync_steps_factor=0.5,  # 1 sync step for medium (2 steps)
)


FLUX_DEV_CONFIG = ImageModelConfig(
    model_family="flux",
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=19, has_separate_text_output=True
        ),
        TransformerBlockConfig(
            block_type=BlockType.SINGLE, count=38, has_separate_text_output=False
        ),
    ),
    default_steps={"low": 10, "medium": 25, "high": 50},
    num_sync_steps_factor=0.125,  # ~3 sync steps for medium (25 steps)
)
