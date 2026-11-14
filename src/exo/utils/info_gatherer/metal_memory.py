import ctypes
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from exo.shared.types.memory import Memory


@dataclass(frozen=True)
class MetalBindings:
    create_device: Callable[[], int | None]
    selector: Callable[[bytes], int | None]
    responds: Callable[[int, int, int], bool]
    read_size: Callable[[int, int], int]
    release: Callable[[int], None]
    push_pool: Callable[[], int | None]
    pop_pool: Callable[[int | None], None]


def load_bindings() -> MetalBindings:
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    pointer = ctypes.c_void_p
    return MetalBindings(
        create_device=cast(
            Callable[[], int | None],
            ctypes.CFUNCTYPE(pointer)(("MTLCreateSystemDefaultDevice", metal)),
        ),
        selector=cast(
            Callable[[bytes], int | None],
            ctypes.CFUNCTYPE(pointer, ctypes.c_char_p)(("sel_registerName", objc)),
        ),
        responds=cast(
            Callable[[int, int, int], bool],
            ctypes.CFUNCTYPE(ctypes.c_bool, pointer, pointer, pointer)(
                ("objc_msgSend", objc)
            ),
        ),
        read_size=cast(
            Callable[[int, int], int],
            ctypes.CFUNCTYPE(ctypes.c_uint64, pointer, pointer)(("objc_msgSend", objc)),
        ),
        release=cast(
            Callable[[int], None],
            ctypes.CFUNCTYPE(None, pointer)(("objc_release", objc)),
        ),
        push_pool=cast(
            Callable[[], int | None],
            ctypes.CFUNCTYPE(pointer)(("objc_autoreleasePoolPush", objc)),
        ),
        pop_pool=cast(
            Callable[[int | None], None],
            ctypes.CFUNCTYPE(None, pointer)(("objc_autoreleasePoolPop", objc)),
        ),
    )


def get_metal_working_set() -> Memory | None:
    if sys.platform != "darwin":
        return None
    try:
        bindings = load_bindings()
        pool = bindings.push_pool()
        try:
            device = bindings.create_device()
            if not device:
                return None
            try:
                property_selector = bindings.selector(b"recommendedMaxWorkingSetSize")
                responds_selector = bindings.selector(b"respondsToSelector:")
                if not property_selector or not responds_selector:
                    return None
                if not bindings.responds(device, responds_selector, property_selector):
                    return None
                size = bindings.read_size(device, property_selector)
                return Memory.from_bytes(size) if size > 0 else None
            finally:
                bindings.release(device)
        finally:
            bindings.pop_pool(pool)
    except (OSError, AttributeError, TypeError, ValueError, ctypes.ArgumentError):
        return None
