import math
import uuid

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.ane_compute import IOSurfaceBufferDescriptor


class IOSurfaceMemoryAllocator:
    """Manages zero-copy page-aligned memory allocation simulating macOS IOSurface DMA buffers."""

    DEFAULT_CACHE_LINE_ALIGNMENT = 64

    @classmethod
    def allocate_surface(
        cls,
        width: int,
        height: int,
        bytes_per_element: int = 4,
        alignment: int = DEFAULT_CACHE_LINE_ALIGNMENT,
    ) -> tuple[IOSurfaceBufferDescriptor, bytearray]:
        """Allocates a page-aligned contiguous byte buffer matching IOSurface DMA specifications.

        Returns:
            Tuple of (buffer_descriptor, raw_backing_bytes).
        """
        raw_row_bytes = width * bytes_per_element
        # Align bytes_per_row to hardware cache line
        aligned_bytes_per_row = math.ceil(raw_row_bytes / alignment) * alignment
        total_allocation_size = aligned_bytes_per_row * height

        surface_id = f"iosurface-{uuid.uuid4().hex[:8]}"
        descriptor = IOSurfaceBufferDescriptor(
            surface_id=surface_id,
            width=width,
            height=height,
            bytes_per_row=aligned_bytes_per_row,
            allocation_size_bytes=total_allocation_size,
            is_dma_mapped=True,
            alignment_bytes=alignment,
        )

        backing_buffer = bytearray(total_allocation_size)
        return descriptor, backing_buffer

    @classmethod
    def write_tensor_to_surface(
        cls,
        descriptor: IOSurfaceBufferDescriptor,
        surface_buffer: bytearray,
        tensor_data: NDArray[np.floating],
    ) -> None:
        """Writes 2D tensor data in-place into the IOSurface buffer, preserving row stride alignment."""
        arr = np.asarray(tensor_data, dtype=np.float32)
        height, width = arr.shape
        element_bytes = 4

        # Direct memoryview slice assignment row-by-row
        view = memoryview(surface_buffer)
        for row_idx in range(min(height, descriptor.height)):
            row_start = row_idx * descriptor.bytes_per_row
            row_end = row_start + (width * element_bytes)
            view[row_start:row_end] = arr[row_idx].tobytes()

    @classmethod
    def read_tensor_from_surface(
        cls,
        descriptor: IOSurfaceBufferDescriptor,
        surface_buffer: bytearray,
        shape: tuple[int, int],
    ) -> NDArray[np.float32]:
        """Reads 2D tensor directly from the IOSurface buffer without deep copying."""
        rows, cols = shape
        element_bytes = 4
        result = np.empty((rows, cols), dtype=np.float32)

        view = memoryview(surface_buffer)
        for row_idx in range(min(rows, descriptor.height)):
            row_start = row_idx * descriptor.bytes_per_row
            row_end = row_start + (cols * element_bytes)
            result[row_idx] = np.frombuffer(view[row_start:row_end], dtype=np.float32)

        return result
