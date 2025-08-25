//go:build unix

package ipc

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"lib"
	"log"
	"os"
	"sync"
	"syscall"
	"time"

	"github.com/pdgendt/cobs"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sys/unix"
)

var (
	ErrInOutPipesAreSame   = errors.New("the in-pipe and out-pipe are the same")
	ErrExistingFileNotFifo = errors.New("the existing file is not a FIFO")
)

const (
	pipeDuplexOpenReaderFlags = syscall.O_RDONLY | syscall.O_NONBLOCK
	pipeDuplexOpenWriterFlags = syscall.O_WRONLY | syscall.O_NONBLOCK
	pipeDuplexModeFlags       = syscall.S_IRUSR | syscall.S_IWUSR | syscall.S_IRGRP | syscall.S_IROTH
	pipeDuplexPollInterval    = 50 * time.Millisecond
	pipeDuplex_PIPE_BUF       = 4096
)

// Signal messages range from 1 to 255 & indicate control flow for the bytestream of the pipe.
type SignalMessage byte

const (
	// DISCARD_PREVIOUS tells the receiver to discard previous partial work.
	DiscardPrevious SignalMessage = 0x01
)

type OnMessage = func(msg []byte) error

// Creates a named-pipe communication duplex. Creates a named-pipe communication duplex.
// The reader end is responsible for creating the pipe.
//
// The layers are:
//  1. Raw binary data over pipes
//  2. Variable-length binary packets with COBS
//  3. JSON-like values with Message Pack
type PipeDuplex struct {
	inPath  string
	outPath string

	rawOutMu sync.Mutex
	rawOut   chan []byte

	ctx    context.Context
	cancel context.CancelFunc
	errg   *errgroup.Group
}

func NewPipeDuplex(inPath, outPath string, onMessage OnMessage) (*PipeDuplex, error) {
	// they must be different files
	if inPath == outPath {
		return nil, ErrInOutPipesAreSame
	}
	// pipes should only ever be created, and only by the reader (one-way operations)
	if err := ensureFifoExists(inPath); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())
	errg, ctx := errgroup.WithContext(ctx)
	p := &PipeDuplex{
		inPath:  inPath,
		outPath: outPath,

		rawOut: make(chan []byte, 128), // TODO: decide on size of this w/ constant??

		ctx:    ctx,
		cancel: cancel,
		errg:   errg,
	}
	// Reader
	p.errg.Go(func() error {
		return p.pipeBufferReader(onMessage)
	})

	// Writer
	p.errg.Go(func() error {
		return p.pipeBufferWriter()
	})

	return p, nil
}

// Close stops all goroutines and waits for them to exit.
func (p *PipeDuplex) Close() error {
	p.cancel()

	// this channel is exclusively written to via methods on this object handle, so it is its owner;
	// owners must be the ones to close channels to avoid race conditions
	defer func() {
		// lock channel to avoid race conditions when closing
		p.rawOutMu.Lock()
		defer p.rawOutMu.Unlock()

		close(p.rawOut)
	}()

	return p.errg.Wait()
}

// SendMessage MessagePack-encodes a "value" and enqueues it to the writer.
func (p *PipeDuplex) SendMessage(msg []byte) error {
	// lock channel to avoid race conditions when closing
	p.rawOutMu.Lock()
	defer p.rawOutMu.Unlock()

	// send message bytes over outRaw channel
	select {
	case p.rawOut <- msg:
		// TODO: could this trigger a race condition if calling Close() immediately after SendMessage()???
		//  should I lock p.rawOut w/ a mutex??
		return nil
	case <-p.ctx.Done():
		return nil
	}
}

func (p *PipeDuplex) InPath() string  { return p.inPath }
func (p *PipeDuplex) OutPath() string { return p.outPath }

// ===== Private =====

func ensureFifoExists(path string) error {
	// try to make a file if one doesn't exist already
	// TODO: add equivalent of `ensure_parent_directory_exists(path)` here !!!!!! <- may cause bugs w/out it???
	if err := unix.Mkfifo(path, pipeDuplexModeFlags); err != nil {
		if errno, ok := err.(unix.Errno); ok {
			// misc error, do not handle
			if errno != unix.EEXIST {
				return err
			}

			// ensure the file exists is FIFO
			fi, err := os.Stat(path)
			if err != nil {
				return err // misc error, do not handle
			}
			if fi.Mode()&fs.ModeNamedPipe == 0 {
				return ErrExistingFileNotFifo
			}
			return nil
		} else {
			return err // misc error, do not handle
		}
	}
	return nil
}

func (p *PipeDuplex) pipeBufferReader(onMessage OnMessage) error {
	// open reader in nonblocking mode -> should not fail & immediately open;
	// this marks when the writer process has "started"
	fd, err := unix.Open(p.inPath, pipeDuplexOpenReaderFlags, pipeDuplexModeFlags)
	if err != nil {
		return err
	}
	defer unix.Close(fd)

	// continually pull from the pipe and interpret messages as such:
	//  - all messages are separated/framed by NULL bytes (zero)
	//  - messages with >=2 bytes are COBS-encoded messages, because
	//    the smallest COBS-encoded message is 2 bytes
	//  - 1-byte messages are therefore to be treated as control signals
	var buf []byte // accumulation buffer
	for {
		select { // check for kill-signal
		case <-p.ctx.Done():
			return nil
		default:
		}

		// read available data (and try again if nothing)
		data := make([]byte, pipeDuplex_PIPE_BUF)
		n, err := unix.Read(fd, data)
		if err != nil {
			errno, ok := err.(unix.Errno)
			if !ok || errno != unix.EAGAIN {
				return err
			}

			// if there is a writer connected & the buffer is empty, this would block
			// so we must consume this error gracefully and try again
			time.Sleep(pipeDuplexPollInterval)
			continue
		}
		if n == 0 {
			time.Sleep(pipeDuplexPollInterval)
			continue
		}

		// extend buffer with new data
		buf = append(buf, data[:n]...)

		// if there are no NULL bytes in the buffer, no new message has been formed
		chunks := bytes.Split(buf, []byte{0x00})
		if len(chunks) == 1 {
			continue
		}

		// last chunk is always an unfinished message, so that becomes our new buffer;
		// the rest should be decoded as either signals or COBS and put on queue
		buf = chunks[len(chunks)-1]
		for i := 0; i < len(chunks)-1; i++ {
			chunk := chunks[i]

			// ignore empty messages (they mean nothing)
			if len(chunk) == 0 {
				continue
			}

			// interpret 1-byte messages as signals (they indicate control-flow on messages)
			if len(chunk) == 1 {
				log.Printf("(reader): gotten control signal: %v", chunk[0])
				// TODO: do some kind of stuff here??
				continue
			}

			// interpret >=2 byte messages as COBS-encoded data (decode them)
			decoded, err := cobs.Decode(chunk)
			if err != nil {
				return err
			}

			// call the callback to handle message
			if err := onMessage(decoded); err != nil {
				return err
			}
		}
	}
}

func (p *PipeDuplex) pipeBufferWriter() error {
	log.Printf("(writer): started")

	// continually attempt to open FIFO for reading in nonblocking mode -> will error that:
	//  - ENOENT[2] No such file or directory: until a reader creates FIFO
	//  - ENXIO[6] No such device or address: until a reader opens FIFO
	fd := -1
	for {
		select { // check for kill-signal
		case <-p.ctx.Done():
			return nil
		default:
		}

		tempFd, err := unix.Open(p.outPath, pipeDuplexOpenWriterFlags, pipeDuplexModeFlags)
		if err != nil {
			if errno, ok := err.(unix.Errno); ok {
				// misc error, do not handle
				if !(errno == unix.ENOENT || errno == unix.ENXIO) {
					return err
				}

				// try again if waiting for FIFO creation or reader-end opening
				time.Sleep(pipeDuplexPollInterval)
				continue
			} else {
				return err // misc error, do not handle
			}
		}
		fd = tempFd
		defer unix.Close(fd)

		// ensure the file exists is FIFO
		mode, err := lib.FstatGetMode(fd)
		if err != nil {
			return err // misc error, do not handle
		}
		if mode&fs.ModeNamedPipe == 0 {
			return ErrExistingFileNotFifo
		}

		break // continue logic
	}

	// read bytes from rawOut & write them to pipe
	for {
		select {
		case buf, ok := <-p.rawOut:
			if !ok {
				return nil
			}
			if err := p.writeData(fd, buf); err != nil {
				return err
			}
		case <-p.ctx.Done():
			return nil
		}

	}
}

func (p *PipeDuplex) writeData(fd int, buf []byte) error {
	// COBS-encode the data & append NULL-byte to signify end-of-frame
	buf, err := cobs.Encode(buf)
	if err != nil {
		return err
	}
	buf = append(buf, 0x00)
	total := len(buf)
	sent := 0

	// begin transmission progress
	for sent < total {
		select { // check for kill-signal
		case <-p.ctx.Done():
			return nil
		default:
		}

		// write & progress on happy path
		written, err := unix.Write(fd, buf[sent:])
		if err == nil {
			sent += written
			continue
		}

		// cast to OS error for propper handling
		errno, ok := err.(unix.Errno)
		if !ok {
			return err // misc error, do not handle
		}

		// non-blocking pipe is full, wait a bit and retry
		if errno == syscall.EAGAIN {
			time.Sleep(pipeDuplexPollInterval)
			continue
		}

		// reader disconnected -> handle failure-recovery by doing:
		//  1. signal DISCARD_PREVIOUS to any reader
		//  2. re-setting the progress & trying again
		if errno == syscall.EPIPE {
			if err := p.writeSignal(fd, DiscardPrevious); err != nil {
				return err
			}
			sent = 0
			continue
		}

		return err // misc error, do not handle
	}
	return nil
}

func (p *PipeDuplex) writeSignal(fd int, sig SignalMessage) error {
	signalMessageLength := 2

	// Turn signal-byte into message by terminating with NULL-byte
	buf := []byte{byte(sig), 0x00}
	lib.Assert(len(buf) == signalMessageLength, "this must never NOT be the case")

	// attempt to write until successful
	for {
		select { // check for kill-signal
		case <-p.ctx.Done():
			return nil
		default:
		}

		// small writes (e.g. 2 bytes) should be atomic as per Pipe semantics,
		// meaning IF SUCCESSFUL: the number of bytes written MUST be exactly 2
		written, err := unix.Write(fd, buf)
		if err == nil {
			lib.Assert(written == signalMessageLength, "this must never NOT be the case")
			break
		}

		// cast to OS error for propper handling
		errno, ok := err.(unix.Errno)
		if !ok {
			return err // misc error, do not handle
		}

		// wait a bit and retry if:
		//  - non-blocking pipe is full
		//  - the pipe is broken because of reader disconnection
		if errno == syscall.EAGAIN || errno == syscall.EPIPE {
			time.Sleep(pipeDuplexPollInterval)
			continue
		}

		return err // misc error, do not handle
	}
	return nil
}
