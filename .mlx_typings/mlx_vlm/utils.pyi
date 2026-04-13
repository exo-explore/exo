from pathlib import Path
from typing import Any

class ImageProcessor:
    def preprocess(
        self, images: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]: ...
    def __call__(self, **kwargs: Any) -> dict[str, Any]: ...

def load_image_processor(
    model_path: str | Path, **kwargs: Any
) -> ImageProcessor | None: ...
def load_processor(
    model_path: str | Path, add_detokenizer: bool = ..., **kwargs: Any
) -> ImageProcessor: ...
