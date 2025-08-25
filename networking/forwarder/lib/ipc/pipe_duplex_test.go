//go:build unix

package ipc

import (
	"log"
	"os"
	"testing"
	"time"
)

func TestOneTwoThree(t *testing.T) {
	// Avoid SIGPIPE killing the test if a writer outlives its reader.
	// signal.Ignore(syscall.SIGPIPE) TODO: shoudn't sigpipe be handled by the error-code deep inside the duplex??

	// Clean slate before/after.
	onePath := "/tmp/one.pipe"
	twoPath := "/tmp/two.pipe"
	_ = os.Remove(onePath)
	_ = os.Remove(twoPath)
	defer os.Remove(onePath)
	defer os.Remove(twoPath)

	owner, err := NewPipeDuplex(
		onePath, // in
		twoPath, // out
		func(m []byte) error { log.Printf("wow, owner got: [%v]%v", len(m), m); return nil },
	)
	if err != nil {
		t.Fatalf("owner New failed: %v", err)
	}

	time.Sleep(1 * time.Second)

	guest1, err := NewPipeDuplex(
		twoPath, // in
		onePath, // out
		func(m []byte) error { log.Printf("wow, guest1 got: [%v]%v", len(m), m); return nil },
	)
	if err != nil {
		t.Fatalf("guest1 New failed: %v", err)
	}

	if err := owner.SendMessage(make([]byte, 10)); err != nil {
		t.Fatalf("owner SendMessage failed: %v", err)
	}

	// batch send
	if err := guest1.SendMessage(make([]byte, 200)); err != nil {
		t.Fatalf("guest1 SendMessage failed: %v", err)
	}

	time.Sleep(1 * time.Second)

	if err := guest1.Close(); err != nil {
		t.Fatalf("guest1 Close failed: %v", err)
	}

	if err := owner.SendMessage(make([]byte, 21)); err != nil {
		t.Fatalf("owner SendMessage failed: %v", err)
	}

	guest2, err := NewPipeDuplex(
		twoPath, // in
		onePath, // out
		func(m []byte) error { log.Printf("wow, guest2 got: [%v]%v", len(m), m); return nil },
	)
	if err != nil {
		t.Fatalf("guest2 New failed: %v", err)
	}

	if err := guest2.SendMessage(make([]byte, 12)); err != nil {
		t.Fatalf("guest2 SendMessage failed: %v", err)
	}

	time.Sleep(1 * time.Second)

	if err := guest2.Close(); err != nil {
		t.Fatalf("guest2 Close failed: %v", err)
	}
	if err := owner.Close(); err != nil {
		t.Fatalf("owner Close failed: %v", err)
	}
	t.Fail()
}
