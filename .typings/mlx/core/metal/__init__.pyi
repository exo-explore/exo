def clear_cache() -> None: ...
def device_info() -> dict[str, str | int]:
    """
    Get information about the GPU device and system settings.

    Currently returns:

    * ``architecture``
    * ``max_buffer_size``
    * ``max_recommended_working_set_size``
    * ``memory_size``
    * ``resource_limit``

    Returns:
        dict: A dictionary with string keys and string or integer values.
    """

def get_active_memory() -> int: ...
def get_cache_memory() -> int: ...
def get_peak_memory() -> int: ...
def is_available() -> bool:
    """Check if the Metal back-end is available."""

def reset_peak_memory() -> None: ...
def set_cache_limit(limit: int) -> int: ...
def set_memory_limit(limit: int) -> int: ...
def set_wired_limit(limit: int) -> int: ...
def start_capture(path: str) -> None:
    """
    Start a Metal capture.

    Args:
      path (str): The path to save the capture which should have
        the extension ``.gputrace``.
    """

def stop_capture() -> None:
    """Stop a Metal capture."""
