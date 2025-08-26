"""
SEE:
 - https://pubs.opengroup.org/onlinepubs/007904875/functions/open.html
 - https://man7.org/linux/man-pages/man2/openat.2.html
 - https://man7.org/linux/man-pages/man3/mkfifo.3.html
 - https://man7.org/linux/man-pages/man7/pipe.7.html

TODO: add locking on reader/writer ends to prevent multiwriters??
TODO: use signal bytes to ensure proper packet consistency
      +stretch: implement packet IDs, retries, dual-stream confirmations, RPCs & so on

TODO: for more hardening -> check if any of the syscalls used return signal interrupt errors (like in the locking case)
      and interrupt on that happening -> this may not be an issue PER SE but might potentially create insanely bizzare bugs
      if it happens that this behavior DOES occasionally happen for no apparent reason

TODO: maybe consider padding all messages with 0s on both ends ?? so as to prevent ANY ambiguous boundaries ever!!
"""

import errno
import logging
import multiprocessing
import os
import queue
import stat
import threading
import time
from enum import Enum
from multiprocessing.queues import Queue as MQueueT
from multiprocessing.synchronize import Event as MEventT
from threading import Event as TEventT
from typing import Callable

from cobs import cobs  # pyright: ignore[reportMissingTypeStubs]
from pytest import LogCaptureFixture

from exo.shared.utils.fs import (
    StrPath,
    delete_if_exists,
    ensure_parent_directory_exists,
)

OPEN_READER_FLAGS = os.O_RDONLY | os.O_NONBLOCK
OPEN_WRITER_FLAGS = os.O_WRONLY | os.O_NONBLOCK

# 0x644 mode flags -> user has read-write permissions, others have read permission only
MODE_FLAGS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH

POLL_INTERVAL = 0.05  # TODO: maybe parametrize this in classes??
PIPE_BUF = 4096  # size of atomic writes on (most) UNIX pipes


class SignalMessage(Enum):
    """
    Signal messages range from 1 to 255 & indicate control flow for the bytestream of the pipe.

    """

    DISCARD_PREVIOUS = b"\x01"


class PipeDuplex:
    """
    Creates a named-pipe communication duplex. The reader end is responsible for creating the pipe.

    The layers are:
      1. Raw binary data over pipes
      2. Variable-length binary packets with COBS
      3. JSON-like values with Message Pack
    """

    def __init__(
        self,
        in_pipe: StrPath,
        out_pipe: StrPath,
        in_callback: Callable[[bytes], None],
    ):
        assert in_pipe != out_pipe  # they must be different files

        # pipes should only ever be created, and only by the reader (one-way operations)
        _ensure_fifo_exists(in_pipe)  # ensures reader pipe exists

        # create readonly properties (useful for inspection)
        self._in_pipe = in_pipe
        self._out_pipe = out_pipe

        # init synchronisation variables
        self._mkill = multiprocessing.Event()
        self._tkill = threading.Event()
        in_mq: MQueueT[bytes] = multiprocessing.Queue()
        self._out_mq: MQueueT[bytes] = multiprocessing.Queue()
        in_mstarted = multiprocessing.Event()

        # process for reading in binary messages from pipe
        self._p_in = multiprocessing.Process(
            target=_pipe_buffer_reader,
            args=(in_pipe, in_mq, in_mstarted, self._mkill),
            daemon=True,
        )
        self._p_in.start()

        # thread for pulling down binary messages from message queue & calling the callback
        self._t_in = threading.Thread(
            target=_binary_object_dispatcher,
            args=(in_mq, in_callback, self._tkill),
            daemon=True,
        )
        self._t_in.start()

        # process to write binary messages to pipe
        out_mstarted = multiprocessing.Event()
        self._p_out = multiprocessing.Process(
            target=_pipe_buffer_writer,
            args=(out_pipe, self._out_mq, out_mstarted, self._mkill),
            daemon=True,
        )
        self._p_out.start()

        # wait for processes to start properly
        in_mstarted.wait()
        out_mstarted.wait()

    def __del__(self):
        # signal to these processes to die (if they haven't already)
        self._mkill.set()
        self._tkill.set()

    def send_message(self, msg: bytes):
        self._out_mq.put_nowait(msg)

    @property
    def in_pipe(self):
        return self._in_pipe

    @property
    def out_pipe(self):
        return self._out_pipe


def _ensure_fifo_exists(path: StrPath):
    # try to make a file if one doesn't exist already
    ensure_parent_directory_exists(path)
    try:
        os.mkfifo(path, mode=MODE_FLAGS)
    except OSError as e:
        # misc error, do not handle
        if e.errno != errno.EEXIST:
            raise

        # ensure the file exists is FIFO
        st = os.stat(path)
        if stat.S_ISFIFO(st.st_mode):
            return

        # this file is not FIFO
        raise FileExistsError(f"The file '{path}' isn't a FIFO") from e


def _pipe_buffer_reader(
    path: StrPath, mq: MQueueT[bytes], started: MEventT, kill: MEventT
):
    # TODO: right now the `kill` control flow is somewhat haphazard -> ensure every loop-y or blocking part always
    #       checks for kill.is_set() and returns/cleans up early if so

    # open reader in nonblocking mode -> should not fail & immediately open;
    # this marks when the writer process has "started"
    fd = os.open(path, OPEN_READER_FLAGS)
    started.set()
    print("(reader):", "started")

    # continually pull from the pipe and interpret messages as such:
    #  - all messages are separated/framed by NULL bytes (zero)
    #  - messages with >=2 bytes are COBS-encoded messages, because
    #    the smallest COBS-encoded message is 2 bytes
    #  - 1-byte messages are therefore to be treated as control signals
    #
    # TODO: right now i just need to get this to work, but the scheme is fundamentally
    #       extensible for robustness, e.g. signal-bytes can be used to drive state-machines
    #       for ensuring message atomicity/transmission
    #       e.g. we can use single-bytes to discriminate COBS values to say "this is length of upcoming message"
    #            vs. this is the actual content of the message, and so on
    #       .
    #       BUT for now we can just use signal (0xff 0x00) to mean "discard previous message" or similar...
    #       .
    #       BUT in THEORY we very well could have something like
    #       (0x10 0x00)[header signal] + (...)[header data like length & so on]
    #       + (0x20 0x00)[body signal] + (...)[body data]
    #       + (0x30 0x00)[checksum signal] + (...)[checksum data]
    #       And requests to re-send messages that were lost, and so on, like this is a fully 2-layer duplex
    #       communication so we could turn this into a VERY powerful thing some time in the future, like
    #       a whole-ass reimplementation of TCP/PIPES lmaooooo
    buffer = bytearray()
    while not kill.is_set():
        try:
            # read available data (and try again if nothing)
            try:
                data = os.read(fd, PIPE_BUF)
                if data == b"":
                    time.sleep(POLL_INTERVAL)
                    continue
            except OSError as e:
                if e.errno != errno.EAGAIN:
                    raise

                # if there is a writer connected & the buffer is empty, this would block
                # so we must consume this error gracefully and try again
                time.sleep(POLL_INTERVAL)
                continue

            # extend buffer with new data
            buffer.extend(data)

            # if there are no NULL bytes in the buffer, no new message has been formed
            chunks = buffer.split(sep=b"\x00")
            if len(chunks) == 1:
                continue

            # last chunk is always an unfinished message, so that becomes our new buffer;
            # the rest should be decoded as either signals or COBS and put on queue
            buffer = chunks.pop()
            for chunk in chunks:
                chunk = bytes(chunk)

                # ignore empty messages (they mean nothing)
                if chunk == b"":
                    continue

                # interpret 1-byte messages as signals (they indicate control-flow on messages)
                if len(chunk) == 1:
                    print("(reader):", f"gotten control signal: {chunk[0]}")
                    continue  # TODO: right now they should be ignored, since I'm not sure what I want them to do

                # interpret >=2 byte messages as COBS-encoded data (decode them)
                decoded = cobs.decode(chunk)  # pyright: ignore[reportUnknownMemberType]
                mq.put(decoded)
        except BaseException as e:
            # perform cleanup & log before re-raising
            os.close(fd)
            logging.error(msg=f"Error when reading from named pipe at '{path}': {e}")
            raise
    os.close(fd)


def _binary_object_dispatcher(
    mq: MQueueT[bytes], callback: Callable[[bytes], None], kill: TEventT
):
    while not kill.is_set():
        # try to get with timeout (to allow to read the kill-flag)
        try:
            message = mq.get(block=True, timeout=POLL_INTERVAL)
        except queue.Empty:
            continue

        # dispatch binary object with callback
        callback(message)


def _pipe_buffer_writer(
    path: StrPath, mq: MQueueT[bytes], started: MEventT, kill: MEventT
):
    # TODO: right now the `kill` control flow is somewhat haphazard -> ensure every loop-y or blocking part always
    #       checks for kill.is_set() and returns/cleans up early if so

    # for now, started events for writer are rather vacuous: TODO: remove or make more usefull??
    started.set()
    print("(writer):", "started")

    # continually attempt to open FIFO for reading in nonblocking mode -> will error that:
    #  - ENOENT[2] No such file or directory: until a reader creates FIFO
    #  - ENXIO[6] No such device or address: until a reader opens FIFO
    fd = None
    while not kill.is_set():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)

            # ensure the file exists is FIFO
            st = os.fstat(fd)
            print("mode:", st.st_mode & 0o170000)
            if stat.S_ISFIFO(st.st_mode):
                break

            # cleanup on error
            os.close(fd)
            raise FileExistsError(f"The file '{path}' isn't a FIFO")
        except FileExistsError:
            raise  # propagate error
        except OSError as e:
            # misc error, do not handle
            if not (e.errno == errno.ENOENT or e.errno == errno.ENXIO):
                raise

            # try again if waiting for FIFO creation or reader-end opening
            time.sleep(POLL_INTERVAL)
            continue
    assert fd is not None

    while not kill.is_set():
        try:
            # try to get with timeout (to allow to read the kill-flag)
            try:
                data = mq.get(block=True, timeout=POLL_INTERVAL)
            except queue.Empty:
                continue

            # write all data (by continually re-trying until it is done)
            _write_data(fd, data)
        except BaseException as e:
            # perform cleanup & log before re-raising
            os.close(fd)
            logging.error(msg=f"Error when writing to named pipe at '{path}': {e}")
            raise

    os.close(fd)


def _write_data(fd: int, buf: bytes):
    # COBS-encode the data & append NULL-byte to signify end-of-frame
    buf = cobs.encode(buf) + b"\x00"  # pyright: ignore[reportUnknownMemberType]
    total = len(buf)
    sent = 0

    # begin transmission progress
    while sent < total:
        try:
            # Write remaining bytes to the pipe
            written = os.write(fd, buf[sent:])
            sent += written
        except OSError as e:
            # non-blocking pipe is full, wait a bit and retry
            if e.errno == errno.EAGAIN:
                time.sleep(POLL_INTERVAL)
                continue

            # reader disconnected -> handle failure-recovery by doing:
            #  1. signal DISCARD_PREVIOUS to any reader
            #  2. re-setting the progress & trying again
            if e.errno == errno.EPIPE:
                _write_signal(fd, SignalMessage.DISCARD_PREVIOUS)
                sent = 0
                continue

            raise  # misc error, do not handle


def _write_signal(fd: int, signal: SignalMessage):
    signal_message_length = 2

    # Turn signal-byte into message by terminating with NULL-byte
    buf = signal.value + b"\x00"
    assert len(buf) == signal_message_length

    # attempt to write until successful
    while True:
        try:
            # small writes (e.g. 2 bytes) should be atomic as per Pipe semantics,
            # meaning IF SUCCESSFUL: the number of bytes written MUST be exactly 2
            written = os.write(fd, buf)
            assert written == signal_message_length
            break
        except OSError as e:
            # wait a bit and retry if:
            #  - non-blocking pipe is full
            #  - the pipe is broken because of reader disconnection
            if e.errno == errno.EAGAIN or e.errno == errno.EPIPE:
                time.sleep(POLL_INTERVAL)
                continue

            raise  # misc error, do not handle


def _test_one_two_three():
    one_path = "/tmp/one.pipe"
    two_path = "/tmp/two.pipe"
    delete_if_exists(one_path)
    delete_if_exists(two_path)

    owner = PipeDuplex(
        in_pipe=one_path,
        out_pipe=two_path,
        in_callback=lambda x: print(f"wow, owner got: [{len(x)}]{x}"),
    )

    guest = PipeDuplex(
        in_pipe=two_path,
        out_pipe=one_path,
        in_callback=lambda x: print(f"wow, guest1 got: [{len(x)}]{x}"),
    )

    owner.send_message(bytes(0 for _ in range(10)))

    guest.send_message(bytes(0 for _ in range(200)))

    time.sleep(1)

    del guest
    guest = PipeDuplex(
        in_pipe=two_path,
        out_pipe=one_path,
        in_callback=lambda x: print(f"wow, guest2 got: [{len(x)}]{x}"),
    )

    guest.send_message(bytes(0 for _ in range(21)))

    owner.send_message(bytes(0 for _ in range(12)))

    time.sleep(1)

    delete_if_exists(one_path)
    delete_if_exists(two_path)


def test_running_pipe_duplex(caplog: LogCaptureFixture):
    caplog.set_level(logging.INFO)

    _test_one_two_three()
    time.sleep(1)
