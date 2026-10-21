import contextlib
import gzip
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

try:
    import mlx.core as mx
    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
    )
except ImportError:
    mx = None  # type: ignore[assignment]
    ArraysCache = None  # type: ignore[assignment, misc]
    CacheList = None  # type: ignore[assignment, misc]
    KVCache = None  # type: ignore[assignment, misc]
    QuantizedKVCache = None  # type: ignore[assignment, misc]
    RotatingKVCache = None  # type: ignore[assignment, misc]

DEFAULT_OFFLOAD_SUBDIRECTORY_NAME: Final[str] = "exo_tiered_kv_cache"
DEFAULT_COMPRESSION_LEVEL: Final[int] = 3
NOMINAL_OFFLOADED_METADATA_BYTES: Final[int] = 128


class OffloadedCacheMetadata(BaseModel):
    """Immutable metadata tracking an offloaded key-value cache entry."""

    model_config = ConfigDict(frozen=True, strict=True)

    entry_identifier: str
    token_count: int
    file_path: str
    byte_size: int
    created_timestamp: float
    access_counter: int
    prefill_tokens_per_second: float = 0.0


class TieredCacheConfiguration(BaseModel):
    """Configuration settings for tiered key-value cache offloading."""

    model_config = ConfigDict(frozen=True, strict=True)

    storage_directory_path: str = Field(
        default_factory=lambda: os.path.join(
            tempfile.gettempdir(), DEFAULT_OFFLOAD_SUBDIRECTORY_NAME
        )
    )
    compression_enabled: bool = True
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
    cleanup_on_exit: bool = True


def _convert_tensor_to_numpy(tensor_or_array: Any) -> np.ndarray | None:
    """Convert an MLX array or tensor into a detached NumPy array."""
    if tensor_or_array is None:
        return None
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    return np.asarray(tensor_or_array)


def _convert_numpy_to_mlx_array(numpy_array: np.ndarray | None) -> Any:
    """Convert a NumPy array back to an MLX array if MLX is available."""
    if numpy_array is None:
        return None
    if mx is not None:
        return mx.array(numpy_array)
    return numpy_array


def serialize_single_cache_layer(layer: Any) -> dict[str, Any]:
    """Serialize a single attention cache layer into a detached dictionary representation."""
    type_name: str = type(layer).__name__

    if type_name == "KVCache" or (KVCache is not None and isinstance(layer, KVCache)):
        return {
            "cache_type": "KVCache",
            "offset": getattr(layer, "offset", 0),
            "keys": _convert_tensor_to_numpy(getattr(layer, "keys", None)),
            "values": _convert_tensor_to_numpy(getattr(layer, "values", None)),
        }

    if type_name == "RotatingKVCache" or (
        RotatingKVCache is not None and isinstance(layer, RotatingKVCache)
    ):
        return {
            "cache_type": "RotatingKVCache",
            "offset": getattr(layer, "offset", 0),
            "max_size": getattr(layer, "max_size", 0),
            "keep": getattr(layer, "keep", 0),
            "internal_index": getattr(layer, "_idx", 0),
            "keys": _convert_tensor_to_numpy(getattr(layer, "keys", None)),
            "values": _convert_tensor_to_numpy(getattr(layer, "values", None)),
        }

    if type_name == "QuantizedKVCache" or (
        QuantizedKVCache is not None and isinstance(layer, QuantizedKVCache)
    ):
        return {
            "cache_type": "QuantizedKVCache",
            "offset": getattr(layer, "offset", 0),
            "group_size": getattr(layer, "group_size", 64),
            "bits": getattr(layer, "bits", 4),
            "keys": _convert_tensor_to_numpy(getattr(layer, "keys", None)),
            "values": _convert_tensor_to_numpy(getattr(layer, "values", None)),
            "scales": _convert_tensor_to_numpy(getattr(layer, "scales", None)),
            "biases": _convert_tensor_to_numpy(getattr(layer, "biases", None)),
        }

    if type_name == "ArraysCache" or (
        ArraysCache is not None and isinstance(layer, ArraysCache)
    ):
        entries = getattr(layer, "cache", [])
        return {
            "cache_type": "ArraysCache",
            "entries": [_convert_tensor_to_numpy(entry) for entry in entries],
        }

    # Generic or test mock cache (e.g. FakeCache with nbytes)
    generic_attributes: dict[str, Any] = {}
    for attribute_name, attribute_value in vars(layer).items():
        if isinstance(attribute_value, (int, float, str, bool)):
            generic_attributes[attribute_name] = attribute_value
        elif attribute_value is not None:
            generic_attributes[attribute_name] = _convert_tensor_to_numpy(
                attribute_value
            )

    return {
        "cache_type": "GenericCache",
        "original_class_name": type_name,
        "attributes": generic_attributes,
    }


def deserialize_single_cache_layer(serialized_data: dict[str, Any]) -> Any:
    """Deserialize a dictionary representation back into an active cache layer."""
    cache_type: str = serialized_data.get("cache_type", "GenericCache")

    if cache_type == "KVCache" and KVCache is not None:
        layer = KVCache.__new__(KVCache)
        layer.offset = serialized_data["offset"]
        layer.keys = _convert_numpy_to_mlx_array(serialized_data["keys"])
        layer.values = _convert_numpy_to_mlx_array(serialized_data["values"])
        return layer

    if cache_type == "RotatingKVCache" and RotatingKVCache is not None:
        layer = RotatingKVCache.__new__(RotatingKVCache)
        layer.offset = serialized_data["offset"]
        layer.max_size = serialized_data["max_size"]
        layer.keep = serialized_data["keep"]
        layer._idx = serialized_data["internal_index"]
        layer.keys = _convert_numpy_to_mlx_array(serialized_data["keys"])
        layer.values = _convert_numpy_to_mlx_array(serialized_data["values"])
        return layer

    if cache_type == "QuantizedKVCache" and QuantizedKVCache is not None:
        layer = QuantizedKVCache.__new__(QuantizedKVCache)
        layer.offset = serialized_data["offset"]
        layer.group_size = serialized_data["group_size"]
        layer.bits = serialized_data["bits"]
        layer.keys = _convert_numpy_to_mlx_array(serialized_data["keys"])
        layer.values = _convert_numpy_to_mlx_array(serialized_data["values"])
        if serialized_data.get("scales") is not None:
            layer.scales = _convert_numpy_to_mlx_array(serialized_data["scales"])
        if serialized_data.get("biases") is not None:
            layer.biases = _convert_numpy_to_mlx_array(serialized_data["biases"])
        return layer

    if cache_type == "ArraysCache" and ArraysCache is not None:
        raw_entries = serialized_data["entries"]
        layer = ArraysCache(len(raw_entries))
        layer.cache = [_convert_numpy_to_mlx_array(entry) for entry in raw_entries]
        return layer

    # Generic or mock fallback class
    class RestoredGenericCache:
        def __init__(self, attributes: dict[str, Any]) -> None:
            for key, value in attributes.items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, _convert_numpy_to_mlx_array(value))
                else:
                    setattr(self, key, value)

        def trim(self, num_tokens: int) -> None:
            if hasattr(self, "offset"):
                self.offset = max(0, self.offset - num_tokens)

    attributes = serialized_data.get("attributes", {})
    return RestoredGenericCache(attributes)


class TieredKVCacheManager:
    """Manages offloading cold key-value cache entries to disk or remote storage."""

    def __init__(
        self,
        configuration: TieredCacheConfiguration | None = None,
    ) -> None:
        self.configuration = configuration or TieredCacheConfiguration()
        self._storage_directory = Path(self.configuration.storage_directory_path)
        self._ensure_storage_directory_exists()
        self._offloaded_records: dict[str, OffloadedCacheMetadata] = {}

    def _ensure_storage_directory_exists(self) -> None:
        """Create storage directory if not already created."""
        try:
            self._storage_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exception:
            logger.warning(
                f"Failed to create tiered cache storage directory {self._storage_directory}: {exception}"
            )

    def offload_cache_entry(
        self,
        entry_identifier: str,
        token_count: int,
        cache_layers: list[Any],
        snapshots: Any,
        access_counter: int,
        prefill_tokens_per_second: float = 0.0,
    ) -> OffloadedCacheMetadata:
        """Serialize and offload an active key-value cache entry to cold storage."""
        self._ensure_storage_directory_exists()
        file_name = f"{entry_identifier}.kvcache"
        file_path = self._storage_directory / file_name

        serialized_layers = [
            serialize_single_cache_layer(layer) for layer in cache_layers
        ]
        payload: dict[str, Any] = {
            "entry_identifier": entry_identifier,
            "token_count": token_count,
            "layers": serialized_layers,
            "snapshots": snapshots,
            "prefill_tokens_per_second": prefill_tokens_per_second,
            "saved_timestamp": time.time(),
        }

        if self.configuration.compression_enabled:
            with gzip.open(
                file_path, "wb", compresslevel=self.configuration.compression_level
            ) as compressed_file:
                pickle.dump(payload, compressed_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, "wb") as standard_file:
                pickle.dump(payload, standard_file, protocol=pickle.HIGHEST_PROTOCOL)

        byte_size = file_path.stat().st_size
        metadata = OffloadedCacheMetadata(
            entry_identifier=entry_identifier,
            token_count=token_count,
            file_path=str(file_path),
            byte_size=byte_size,
            created_timestamp=time.time(),
            access_counter=access_counter,
            prefill_tokens_per_second=prefill_tokens_per_second,
        )

        self._offloaded_records[entry_identifier] = metadata
        logger.info(
            f"Tiered KV Cache offloaded entry {entry_identifier} ({token_count} tokens, {byte_size} bytes) to {file_path}"
        )
        return metadata

    def restore_cache_entry(
        self,
        metadata: OffloadedCacheMetadata,
    ) -> tuple[list[Any], Any]:
        """Deserialize and restore an offloaded cache entry back into memory."""
        file_path = Path(metadata.file_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Offloaded cache file not found at path: {file_path}"
            )

        if self.configuration.compression_enabled:
            with gzip.open(file_path, "rb") as compressed_file:
                payload: dict[str, Any] = pickle.load(compressed_file)
        else:
            with open(file_path, "rb") as standard_file:
                payload = pickle.load(standard_file)

        raw_layers: list[dict[str, Any]] = payload.get("layers", [])
        restored_layers = [
            deserialize_single_cache_layer(raw_layer) for raw_layer in raw_layers
        ]
        snapshots = payload.get("snapshots")

        logger.info(
            f"Tiered KV Cache hydrated entry {metadata.entry_identifier} ({metadata.token_count} tokens) from {file_path}"
        )
        return restored_layers, snapshots

    def remove_offloaded_entry(
        self,
        metadata: OffloadedCacheMetadata,
    ) -> None:
        """Remove an offloaded cache file from disk and deregister its record."""
        file_path = Path(metadata.file_path)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as exception:
            logger.warning(
                f"Failed to remove offloaded cache file {file_path}: {exception}"
            )

        self._offloaded_records.pop(metadata.entry_identifier, None)

    def clear_all_offloaded_entries(self) -> None:
        """Clean up all offloaded cache files tracked by this manager."""
        for metadata in list(self._offloaded_records.values()):
            self.remove_offloaded_entry(metadata)

        if self._storage_directory.exists():
            with contextlib.suppress(OSError):
                for file_entry in self._storage_directory.glob("*.kvcache"):
                    with contextlib.suppress(OSError):
                        file_entry.unlink()

    def __del__(self) -> None:
        """Destructor to clean up temporary storage if configured."""
        if self.configuration.cleanup_on_exit:
            with contextlib.suppress(OSError):
                self.clear_all_offloaded_entries()

