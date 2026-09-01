"""The file-log rotation callable used to be a generator that returned True
exactly once and False forever after -- correct for "fresh log per process
start" but silently disabling every later rotation, letting the log file
grow unbounded for the rest of the process's life. These tests cover the
replacement policy: rotate on startup, then rotate again by size.
"""

from exo.shared.logging import _rotation_policy  # pyright: ignore[reportPrivateUsage]


class _FakeFile:
    """Minimal stand-in for the file-like object loguru passes to a
    rotation callable -- only `.tell()` is used by `_rotation_policy`."""

    def __init__(self, size: int):
        self._size = size

    def tell(self) -> int:
        return self._size


class _RaisingFile:
    def tell(self) -> int:
        raise OSError("file closed")


def test_first_call_always_rotates_regardless_of_size():
    should_rotate = _rotation_policy(max_bytes=1000)

    # Even a file well under the size threshold rotates on the first call --
    # this is the "fresh log per process start" behavior the original
    # generator-based rotation intended.
    assert should_rotate("short message", _FakeFile(size=0)) is True


def test_second_call_does_not_rotate_when_under_threshold():
    should_rotate = _rotation_policy(max_bytes=1000)
    should_rotate("first message", _FakeFile(size=0))  # consume the startup rotation

    assert should_rotate("small message", _FakeFile(size=100)) is False


def test_rotates_once_file_plus_message_crosses_threshold():
    should_rotate = _rotation_policy(max_bytes=1000)
    should_rotate("first message", _FakeFile(size=0))  # consume the startup rotation

    # This is the check the original generator never actually performed:
    # without it, the file just keeps growing past max_bytes indefinitely.
    assert should_rotate("x" * 50, _FakeFile(size=980)) is True


def test_only_the_first_call_ever_forces_rotation():
    should_rotate = _rotation_policy(max_bytes=1000)

    assert should_rotate("first message", _FakeFile(size=0)) is True
    # A second small message right after a rotation must NOT force another
    # rotation just because it's early in the policy's lifetime.
    assert should_rotate("second message", _FakeFile(size=0)) is False


def test_tell_raising_oserror_does_not_propagate():
    should_rotate = _rotation_policy(max_bytes=1000)
    should_rotate("first message", _FakeFile(size=0))  # consume the startup rotation

    # A closed/invalid file handle must not crash the logging pipeline --
    # skip rotation for this message rather than raise.
    assert should_rotate("message", _RaisingFile()) is False
