import pytest

from exo.shared.ipc.file_mutex.flock_mutex import FlockMutex, LockType
from exo.utils.fs import delete_if_exists, make_temp_path


def test_lock_held():
    path = make_temp_path("testing_flock.lock")
    lock = FlockMutex(path)

    assert lock.lock_held is None

    assert lock.acquire(lock_type=LockType.WRITE)
    assert lock.lock_held == LockType.WRITE
    lock.release()

    assert lock.lock_held is None

    assert lock.acquire(lock_type=LockType.READ)
    assert lock.lock_held == LockType.READ
    lock.release()

    assert lock.lock_held is None

    delete_if_exists(path)


def test_no_reentrant_lock():
    path = make_temp_path("testing_flock.lock")
    lock = FlockMutex(path)

    # no write-lock reentrancy
    lock.acquire(lock_type=LockType.WRITE)
    with pytest.raises(AssertionError):
        lock.acquire(lock_type=LockType.WRITE)
    with pytest.raises(AssertionError):
        lock.acquire(lock_type=LockType.READ)
    lock.release()

    # no read-lock reentrancy
    lock.acquire(lock_type=LockType.READ)
    with pytest.raises(AssertionError):
        lock.acquire(lock_type=LockType.WRITE)
    with pytest.raises(AssertionError):
        lock.acquire(lock_type=LockType.READ)
    lock.release()

    delete_if_exists(path)
