import time
from pathlib import Path

from pydantic import BaseModel

from exo.shared.types.common import Id


class StoredImage(BaseModel, frozen=True):
    image_id: Id
    file_path: Path
    content_type: str
    expires_at: float


class ImageStore:
    def __init__(self, storage_dir: Path, default_expiry_seconds: int = 3600) -> None:
        self._storage_dir = storage_dir
        self._default_expiry_seconds = default_expiry_seconds
        self._images: dict[Id, StoredImage] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, image_bytes: bytes, content_type: str) -> StoredImage:
        image_id = Id()
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

    def get(self, image_id: Id) -> StoredImage | None:
        stored = self._images.get(image_id)
        if stored is None:
            return None

        if time.time() > stored.expires_at:
            self._remove(image_id)
            return None

        return stored

    def list_images(self) -> list[StoredImage]:
        now = time.time()
        return [stored for stored in self._images.values() if now <= stored.expires_at]

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

    def _remove(self, image_id: Id) -> None:
        stored = self._images.pop(image_id, None)
        if stored is not None and stored.file_path.exists():
            stored.file_path.unlink()


def _content_type_to_extension(
    content_type: str,
) -> str:
    ext = f"{content_type.split('/')[1]}"
    if ext == "jpeg":
        ext = "jpg"

    return f".{ext}"
