from enum import Enum

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
    model_family: str

    block_configs: tuple[TransformerBlockConfig, ...]

    default_steps: dict[str, int]  # {"low": X, "medium": Y, "high": Z}
    num_sync_steps: int  # Number of sync steps for distributed inference

    guidance_scale: float | None = None  # None or <= 1.0 disables CFG

    @property
    def total_blocks(self) -> int:
        return sum(bc.count for bc in self.block_configs)

    @property
    def joint_block_count(self) -> int:
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.JOINT
        )

    @property
    def single_block_count(self) -> int:
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.SINGLE
        )

    def get_steps_for_quality(self, quality: str) -> int:
        return self.default_steps[quality]
