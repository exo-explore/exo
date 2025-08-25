//go:build unix

package ipc

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func check(t *testing.T, err error) {
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func makeTempPath(t *testing.T, pattern string) string {
	f, err := os.CreateTemp("", pattern)
	check(t, err)
	name := f.Name()
	defer os.Remove(name)
	return name
}

func TestLockHeld(t *testing.T) {
	path := makeTempPath(t, "testing_flock.lock")
	defer os.Remove(path)
	mu := NewFlockMutex(path)

	assert.Equal(t, LockMissing, mu.lockHeld)

	acquired, err := mu.Acquire(WriteLock, SpinBlocking)
	check(t, err)
	assert.True(t, acquired)
	assert.Equal(t, WriteLock, mu.lockHeld)
	check(t, mu.release())

	assert.Equal(t, LockMissing, mu.lockHeld)

	acquired, err = mu.Acquire(ReadLock, SpinBlocking)
	check(t, err)
	assert.True(t, acquired)
	assert.Equal(t, ReadLock, mu.lockHeld)
	check(t, mu.release())

	assert.Equal(t, LockMissing, mu.lockHeld)
}

func TestNoReentrantLock(t *testing.T) {
	path := makeTempPath(t, "testing_flock.lock")
	defer os.Remove(path)
	mu := NewFlockMutex(path)

	// no write-lock reentrancy
	acquired, err := mu.Acquire(WriteLock, SpinBlocking)
	check(t, err)
	assert.True(t, acquired)
	{
		acquired, err = mu.Acquire(WriteLock, SpinBlocking)
		assert.False(t, acquired)
		assert.Equal(t, ErrLockAlreadyHeld, err)
	}
	{
		acquired, err = mu.Acquire(ReadLock, SpinBlocking)
		assert.False(t, acquired)
		assert.Equal(t, ErrLockAlreadyHeld, err)
	}
	check(t, mu.release())

	// no read-lock reentrancy
	acquired, err = mu.Acquire(ReadLock, SpinBlocking)
	check(t, err)
	assert.True(t, acquired)
	{
		acquired, err = mu.Acquire(WriteLock, SpinBlocking)
		assert.False(t, acquired)
		assert.Equal(t, ErrLockAlreadyHeld, err)
	}
	{
		acquired, err = mu.Acquire(ReadLock, SpinBlocking)
		assert.False(t, acquired)
		assert.Equal(t, ErrLockAlreadyHeld, err)
	}
	check(t, mu.release())
}
