//go:build unix

package ipc

import (
	"errors"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

var (
	ErrFileDescriptorAlreadyOpen = errors.New("file descriptor not open")
	ErrFileDescriptorNotOpen     = errors.New("file descriptor not open")
	ErrLockAlreadyHeld           = errors.New("lock already held")
	ErrLockNotHeld               = errors.New("lock not held")
)

const (
	// open in read-write mode, creates file if it doesn't exist already,
	// closes this file descriptor in any children processes (prevents FD leaking),
	// truncates this file on opening (lock-files shouldn't hold content FOR NOW!!!)
	//
	// SEE: https://man7.org/linux/man-pages/man2/openat.2.html
	flockMutexOpenFlags int = syscall.O_RDWR | syscall.O_CREAT | syscall.O_CLOEXEC | syscall.O_TRUNC

	// 0x644 mode flags -> user has read-write permissions, others have read permission only
	// SEE: https://man7.org/linux/man-pages/man2/openat.2.html
	flockMutexModeFlags uint32 = syscall.S_IRUSR | syscall.S_IWUSR | syscall.S_IRGRP | syscall.S_IROTH

	// default poll-interval for spin-blocking lock
	flockMutexPollInterval = 50 * time.Millisecond
)

type LockType int

const (
	ReadLock    LockType = syscall.LOCK_SH
	WriteLock   LockType = syscall.LOCK_EX
	LockMissing LockType = -1
)

type AcquireMode int

const (
	OsBlocking AcquireMode = iota
	SpinBlocking
	NonBlocking
)

type FlockMutex struct {
	filePath string
	fd       int
	lockHeld LockType
}

func NewFlockMutex(filePath string) *FlockMutex {
	return &FlockMutex{
		filePath: filePath,
		fd:       -1,
		lockHeld: LockMissing,
	}
}

func (mu *FlockMutex) openFd() error {
	if mu.fd != -1 {
		return ErrFileDescriptorAlreadyOpen
	}
	// TODO: ensure_directory_exists(mu.filePath)

	// open file & TRY to change permissions to `modeFlags` flags
	fd, err := unix.Open(mu.filePath, flockMutexOpenFlags, flockMutexModeFlags)
	if err != nil {
		return err
	} else {
		mu.fd = fd
		_ = unix.Fchmod(fd, flockMutexModeFlags) // This locked is not owned by this UID
	}
	return nil
}

func (mu *FlockMutex) closeFd() error {
	if mu.fd == -1 {
		return ErrFileDescriptorNotOpen
	}

	if err := unix.Close(mu.fd); err != nil {
		mu.fd = -1
		return err
	}

	mu.fd = -1
	return nil
}

func (mu *FlockMutex) acquire(lockType LockType, blocking bool) (bool, error) {
	// enforce preconditions/sanity checks
	if mu.fd == -1 {
		return false, ErrFileDescriptorNotOpen
	}
	if mu.lockHeld != LockMissing {
		return false, ErrLockAlreadyHeld
	}

	// create flags for acquiring lock
	var flags = int(lockType)
	if !blocking {
		flags |= syscall.LOCK_NB
	}

	// continually try to acquire lock (since it may fail due to interrupts)
	for {
		if err := unix.Flock(mu.fd, flags); err != nil {
			if errno, ok := err.(unix.Errno); ok {
				// call interrupted by signal -> try again
				if errno == unix.EINTR {
					continue
				}

				// file is locked & non-blocking is enabled -> return false to indicate
				if errno == unix.EWOULDBLOCK {
					return false, nil
				}
			}

			// unhandleable errors -> close FD & return error
			_ = mu.closeFd() // TODO: how to merge Go errors ???
			return false, err
		}
		break
	}

	// set lock-type held
	mu.lockHeld = lockType
	return true, nil
}

func (mu *FlockMutex) release() error {
	// enforce preconditions/sanity checks
	if mu.fd == -1 {
		return ErrFileDescriptorNotOpen
	}
	if mu.lockHeld == LockMissing {
		return ErrLockNotHeld
	}

	// continually try to release lock (since it may fail due to interrupts)
	for {
		if err := unix.Flock(mu.fd, syscall.LOCK_UN); err != nil {
			if errno, ok := err.(unix.Errno); ok {
				// call interrupted by signal -> try again
				if errno == unix.EINTR {
					continue
				}
			}

			// unhandleable errors -> close FD & return error
			mu.lockHeld = LockMissing
			_ = mu.closeFd() // TODO: how to merge Go errors ???
			return err
		}
		break
	}

	mu.lockHeld = LockMissing
	return nil
}

func (mu *FlockMutex) Acquire(lockType LockType, acquireMode AcquireMode) (bool, error) {
	// open file if missing
	if mu.fd == -1 {
		if err := mu.openFd(); err != nil {
			return false, err
		}
	}

	// OS-blocking & non-blocking is direct passthrough to private function
	switch acquireMode {
	case OsBlocking:
		return mu.acquire(lockType, true)
	case NonBlocking:
		return mu.acquire(lockType, false)
	}

	// spin-blocking works by trying to acquire the lock in non-blocking mode, and retrying until success
	for {
		locked, err := mu.acquire(lockType, false)
		if err != nil {
			return false, err
		}
		if locked {
			return true, err
		}
		time.Sleep(flockMutexPollInterval)
	}
}

func (mu *FlockMutex) Release(lockType LockType, acquireMode AcquireMode) error {
	if err := mu.release(); err != nil {
		_ = mu.closeFd() // TODO: how to merge Go errors ???
		return err
	}
	if err := mu.closeFd(); err != nil {
		return err
	}
	return nil
}
