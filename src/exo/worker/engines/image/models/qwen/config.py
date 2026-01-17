from exo.worker.engines.image.config import (
    BlockType,
    ImageModelConfig,
    TransformerBlockConfig,
)

# Qwen-Image has 60 joint-style blocks (no single blocks)
# Architecture: 24 heads * 128 dim = 3072 hidden dim
# VAE uses scale factor of 16 (vs Flux's 8)
QWEN_IMAGE_CONFIG = ImageModelConfig(
    model_family="qwen",
    model_variant="image",
    hidden_dim=3072,
    num_heads=24,
    head_dim=128,
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=60, has_separate_text_output=True
        ),
        # Qwen has no single blocks - all blocks process image and text separately
    ),
    patch_size=2,
    vae_scale_factor=16,
    default_steps={"low": 10, "medium": 25, "high": 50},
    num_sync_steps_factor=0.125,  # ~3 sync steps for medium (30 steps)
    uses_attention_mask=True,  # Qwen uses encoder_hidden_states_mask
    guidance_scale=3.5,  # Set to None or < 1.0 to disable CFG
)

# Qwen-Image-Edit uses the same architecture but different processing pipeline
# Uses vision-language encoding and conditioning latents
QWEN_IMAGE_EDIT_CONFIG = ImageModelConfig(
    model_family="qwen-edit",
    model_variant="image-edit",
    hidden_dim=3072,
    num_heads=24,
    head_dim=128,
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=60, has_separate_text_output=True
        ),
    ),
    patch_size=2,
    vae_scale_factor=16,
    default_steps={"low": 10, "medium": 25, "high": 50},
    num_sync_steps_factor=0.125,
    uses_attention_mask=True,
    guidance_scale=3.5,
)
