"""
File-based mutex primitive implemented using UNIX-based `flock` syscall.

"""

import contextlib
import errno
import fcntl
import os
import stat
import time
from enum import Enum
from typing import Optional

from exo.shared.utils.fs import StrPath, ensure_parent_directory_exists

# open in read-write mode, creates file if it doesn't exist already,
# closes this file descriptor in any children processes (prevents FD leaking),
# truncates this file on opening (lock-files shouldn't hold content FOR NOW!!!)
# SEE: https://man7.org/linux/man-pages/man2/openat.2.html
OPEN_FLAGS = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_TRUNC

# 0x644 mode flags -> user has read-write permissions, others have read permission only
# SEE: https://man7.org/linux/man-pages/man2/openat.2.html
MODE_FLAGS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH

# default poll-interval for spin-blocking lock
POLL_INTERVAL = 0.05


class LockType(Enum):
    READ = fcntl.LOCK_SH
    WRITE = fcntl.LOCK_EX


class AcquireMode(Enum):
    OS_BLOCKING = 0
    SPIN_BLOCKING = 1
    NON_BLOCKING = 2


class FlockMutex:
    def __init__(self, file_path: StrPath):
        self._file_path = file_path
        self._fd: Optional[int] = None
        self.lock_held: Optional[LockType] = None

    def _open_fd(self):
        assert self._fd is None
        ensure_parent_directory_exists(self._file_path)

        # open file & TRY to change permissions to `MODE_FLAGS` flags
        self._fd = os.open(self._file_path, OPEN_FLAGS, MODE_FLAGS)
        with contextlib.suppress(
            PermissionError
        ):  # This locked is not owned by this UID
            os.chmod(self._fd, MODE_FLAGS)

    def _close_fd(self):
        assert self._fd is not None
        os.close(self._fd)
        self._fd = None

    def _acquire(self, lock_type: LockType, blocking: bool) -> bool:
        assert (self._fd is not None) and (self.lock_held is None)

        # create flags for acquiring lock
        flags = lock_type.value
        if not blocking:
            flags |= fcntl.LOCK_NB

        # continually try to acquire lock (since it may fail due to interrupts)
        while True:
            try:
                fcntl.flock(self._fd, flags)
                break
            except OSError as e:
                if e.errno == errno.EINTR:  # call interrupted by signal -> try again
                    continue
                elif (
                    e.errno == errno.EWOULDBLOCK
                ):  # file is locked & non-blocking is enabled -> return false to indicate
                    return False

                # unhandleable errors -> close FD & raise
                self._close_fd()
                if e.errno == errno.ENOSYS:  # NotImplemented error
                    raise NotImplementedError(
                        "This system doesn't support flock"
                    ) from e
                else:
                    raise

        # set lock-type held
        self.lock_held = lock_type
        return True

    def _release(self):
        assert (self._fd is not None) and (self.lock_held is not None)

        # continually try to release lock (since it may fail due to interrupts)
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                break
            except OSError as e:
                if e.errno == errno.EINTR:  # call interrupted by signal -> try again
                    continue

                # unhandleable errors -> close FD & raise
                self._close_fd()
                if e.errno == errno.ENOSYS:  # NotImplemented error
                    raise NotImplementedError(
                        "This system doesn't support flock"
                    ) from e
                else:
                    raise

        self.lock_held = None

    def acquire(
        self,
        lock_type: LockType = LockType.WRITE,
        acquire_mode: AcquireMode = AcquireMode.SPIN_BLOCKING,
    ) -> bool:
        if self._fd is None:
            self._open_fd()

        # OS-blocking & non-blocking is direct passthrough to private function
        match acquire_mode:
            case AcquireMode.OS_BLOCKING:
                return self._acquire(lock_type, blocking=True)
            case AcquireMode.NON_BLOCKING:
                return self._acquire(lock_type, blocking=False)
            case _:
                pass

        # spin-blocking works by trying to acquire the lock in non-blocking mode, and retrying until success
        while True:
            locked = self._acquire(lock_type, blocking=False)
            if locked:
                return True
            time.sleep(POLL_INTERVAL)

    def release(self):
        self._release()
        self._close_fd()
