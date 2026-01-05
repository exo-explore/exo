from enum import Enum
from math import ceil

from pydantic import BaseModel


class BlockType(Enum):
    JOINT = "joint"  # Separate image/text streams
    SINGLE = "single"  # Concatenated streams


class TransformerBlockConfig(BaseModel):
    model_config = {"frozen": True}

    block_type: BlockType
    count: int
    has_separate_text_output: bool  # True for joint blocks that output text separately


class ImageModelConfig(BaseModel):
    model_config = {"frozen": True}

    # Model identification
    model_family: str  # "flux", "fibo", "qwen"
    model_variant: str  # "schnell", "dev", etc.

    # Architecture parameters
    hidden_dim: int
    num_heads: int
    head_dim: int

    # Block configuration - ordered sequence of block types
    block_configs: tuple[TransformerBlockConfig, ...]

    # Tokenization parameters
    patch_size: int  # 2 for Flux/Qwen
    vae_scale_factor: int  # 8 for Flux, 16 for others

    # Inference parameters
    default_steps: dict[str, int]  # {"low": X, "medium": Y, "high": Z}
    num_sync_steps_factor: float  # Fraction of steps for sync phase

    # Feature flags
    uses_attention_mask: bool  # True for Fibo

    @property
    def total_blocks(self) -> int:
        """Total number of transformer blocks."""
        return sum(bc.count for bc in self.block_configs)

    @property
    def joint_block_count(self) -> int:
        """Number of joint transformer blocks."""
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.JOINT
        )

    @property
    def single_block_count(self) -> int:
        """Number of single transformer blocks."""
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.SINGLE
        )

    def get_steps_for_quality(self, quality: str) -> int:
        """Get inference steps for a quality level."""
        return self.default_steps[quality]

    def get_num_sync_steps(self, quality: str) -> int:
        """Get number of synchronous steps based on quality."""
        return ceil(self.default_steps[quality] * self.num_sync_steps_factor)
