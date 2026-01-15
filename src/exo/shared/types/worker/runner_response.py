from collections.abc import Generator
from typing import Any, Literal

from exo.shared.types.api import FinishReason, GenerationStats
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    pass


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    # logprobs: list[float] | None = None # too big. we can change to be top-k
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None


class ImageGenerationResponse(BaseRunnerResponse):
    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class PartialImageResponse(BaseRunnerResponse):
    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"
    partial_index: int
    total_partials: int

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class FinishedResponse(BaseRunnerResponse):
    pass
