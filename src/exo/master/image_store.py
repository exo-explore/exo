import time
import uuid
from pathlib import Path
from typing import NewType

from pydantic import BaseModel

ImageId = NewType("ImageId", str)


class StoredImage(BaseModel, frozen=True):
    image_id: ImageId
    file_path: Path
    content_type: str
    expires_at: float


class ImageStore:
    def __init__(self, storage_dir: Path, default_expiry_seconds: int = 3600) -> None:
        self._storage_dir = storage_dir
        self._default_expiry_seconds = default_expiry_seconds
        self._images: dict[ImageId, StoredImage] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, image_bytes: bytes, content_type: str) -> StoredImage:
        image_id = ImageId(str(uuid.uuid4()))
        extension = _content_type_to_extension(content_type)
        file_path = self._storage_dir / f"{image_id}{extension}"
        file_path.write_bytes(image_bytes)

        stored = StoredImage(
            image_id=image_id,
            file_path=file_path,
            content_type=content_type,
            expires_at=time.time() + self._default_expiry_seconds,
        )
        self._images[image_id] = stored
        return stored

    def get(self, image_id: ImageId) -> StoredImage | None:
        stored = self._images.get(image_id)
        if stored is None:
            return None

        if time.time() > stored.expires_at:
            self._remove(image_id)
            return None

        return stored

    def cleanup_expired(self) -> int:
        now = time.time()
        expired_ids = [
            image_id
            for image_id, stored in self._images.items()
            if now > stored.expires_at
        ]

        for image_id in expired_ids:
            self._remove(image_id)

        return len(expired_ids)

    def _remove(self, image_id: ImageId) -> None:
        stored = self._images.pop(image_id, None)
        if stored is not None and stored.file_path.exists():
            stored.file_path.unlink()


def _content_type_to_extension(content_type: str) -> str:
    extensions = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    return extensions.get(content_type, ".png")
