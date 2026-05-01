use std::io::{self, ErrorKind, IoSlice, IoSliceMut};
use std::mem::{MaybeUninit, size_of};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use rustix::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd as _, IntoRawFd as _, OwnedFd, RawFd};
use rustix::io::{FdFlags, fcntl_getfd, fcntl_setfd, retry_on_intr};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
    SendAncillaryBuffer, SendAncillaryMessage, SendFlags, SocketFlags, SocketType, recvmsg,
    sendmsg, socketpair,
};
use zerocopy::byteorder::{NetworkEndian, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

const CHANNELS_PER_FRAME: usize = 1;
const FRAME_HEADER_LEN: usize = size_of::<FrameHeader>();

/// Default maximum byte-frame payload accepted by [`UnixPacketChannel::recv`].
///
/// This is a protocol safety limit, not an expected message size. Increase it
/// when the application legitimately needs to pass larger serialized objects.
pub const DEFAULT_MAX_PAYLOAD_SIZE: usize = 64 * 1024 * 1024;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
enum FrameKind {
    Bytes = 0,
    Channel = 1,
}

#[repr(C, packed)]
#[derive(Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
struct FrameHeader {
    raw_kind: u8,
    payload_len: U64<NetworkEndian>,
}

impl FrameHeader {
    fn new(kind: FrameKind, payload_len: usize) -> io::Result<Self> {
        let payload_len = u64::try_from(payload_len).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "payload length does not fit in the wire header",
            )
        })?;

        Ok(Self {
            raw_kind: kind.into(),
            payload_len: U64::new(payload_len),
        })
    }

    fn kind(self) -> io::Result<FrameKind> {
        FrameKind::try_from(self.raw_kind)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid frame kind"))
    }

    fn payload_len(self) -> io::Result<usize> {
        usize::try_from(self.payload_len.get()).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidData,
                "payload length does not fit on this platform",
            )
        })
    }
}

/// One framed message received from a [`UnixPacketChannel`].
#[derive(Debug)]
pub enum Packet {
    Bytes(Vec<u8>),
    Channel(UnixPacketChannel),
}

/// A connected Unix-domain, framed byte/channel transport.
///
/// This is built on an unnamed `AF_UNIX`/`SOCK_STREAM` socket. The stream is
/// framed with a small fixed zerocopy header, so byte payloads are not limited
/// by kernel packet size. Channel endpoints are carried with Unix ancillary
/// data on zero-length channel frames.
#[derive(Debug)]
pub struct UnixPacketChannel {
    fd: OwnedFd,
}

impl UnixPacketChannel {
    /// Create a connected pair of unnamed Unix framed channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the socketpair cannot be created or configured.
    #[inline]
    pub fn pair() -> io::Result<(Self, Self)> {
        let (left, right) = socketpair(
            AddressFamily::UNIX,
            SocketType::STREAM,
            SocketFlags::empty(),
            None,
        )
        .map_err(io::Error::from)?;

        // SAFETY: `socketpair` created both descriptors as connected
        // `AF_UNIX`/`SOCK_STREAM` endpoints, and `OwnedFd` gives this code
        // unique ownership of each descriptor.
        let left = unsafe { Self::from_owned_fd(left)? };
        // SAFETY: Same reasoning as above for the other end of the socketpair.
        let right = unsafe { Self::from_owned_fd(right)? };

        Ok((left, right))
    }

    /// Wrap an owned file descriptor as a Unix framed channel.
    ///
    /// This constructor trusts the caller that `fd` is an `AF_UNIX`/
    /// `SOCK_STREAM` endpoint using this framing protocol. Use
    /// [`Self::from_checked_owned_fd`] when the descriptor comes from an
    /// untrusted source and the host permits socket option inspection.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor cannot be configured.
    ///
    /// # Safety
    ///
    /// `fd` must be an `AF_UNIX`/`SOCK_STREAM` endpoint intended to be used by
    /// this protocol. Use [`Self::from_checked_owned_fd`] when that invariant
    /// has not already been established.
    #[inline]
    pub unsafe fn from_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        configure_socket(&fd)?;
        Ok(Self { fd })
    }

    /// Wrap an owned file descriptor after validating it as a stream socket.
    ///
    /// The descriptor is checked as a `SOCK_STREAM` socket before being
    /// accepted. This intentionally does not validate `SO_DOMAIN`, because some
    /// container/sandbox profiles reject that `getsockopt` even for valid
    /// descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a stream socket or cannot be
    /// configured.
    #[inline]
    pub fn from_checked_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        validate_stream_socket(&fd)?;
        // SAFETY: `validate_stream_socket` checked the descriptor type. The
        // caller supplied `OwnedFd`, so Rust has unique ownership.
        unsafe { Self::from_owned_fd(fd) }
    }

    /// Wrap an inherited raw file descriptor as a Unix framed channel.
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path. After this
    /// function succeeds or fails, the descriptor is owned by Rust and will be
    /// closed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor cannot be configured.
    #[inline]
    pub unsafe fn from_raw_fd(raw_fd: RawFd) -> io::Result<Self> {
        // SAFETY: The caller guarantees that `raw_fd` is open and uniquely owned.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        // SAFETY: The caller also guarantees this descriptor is the expected
        // framed-channel endpoint.
        unsafe { Self::from_owned_fd(fd) }
    }

    /// Wrap an inherited raw file descriptor after validating it as a stream
    /// socket.
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path. After this
    /// function succeeds or fails, the descriptor is owned by Rust and will be
    /// closed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a stream socket or cannot be
    /// configured.
    #[inline]
    pub unsafe fn from_checked_raw_fd(raw_fd: RawFd) -> io::Result<Self> {
        // SAFETY: The caller guarantees that `raw_fd` is open and uniquely owned.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        Self::from_checked_owned_fd(fd)
    }

    /// Return the underlying raw file descriptor.
    #[must_use]
    #[inline]
    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Consume this channel and return its owned file descriptor.
    #[must_use]
    #[inline]
    pub fn into_owned_fd(self) -> OwnedFd {
        self.fd
    }

    /// Consume this channel and return its raw file descriptor.
    #[must_use]
    #[inline]
    pub fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }

    /// Send an arbitrary-sized opaque byte message without attached channels.
    ///
    /// The bytes are length-prefixed and written across the stream as needed.
    /// The receiver still enforces its own maximum accepted payload size.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame cannot be written.
    #[inline]
    pub fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        let header = FrameHeader::new(FrameKind::Bytes, bytes.len())?;

        self.send_all(header.as_bytes())?;
        self.send_all(bytes)
    }

    /// Send a framed-channel endpoint over this channel.
    ///
    /// This consumes `channel`. The kernel duplicates the descriptor into the
    /// receiver's process; consuming the local value gives this method
    /// transfer-of-ownership semantics at the Rust API boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if ancillary data cannot be prepared or the channel
    /// frame cannot be sent. On error, `channel` is still dropped.
    #[inline]
    pub fn send_channel(&mut self, channel: Self) -> io::Result<()> {
        let result = {
            let header = FrameHeader::new(FrameKind::Channel, 0)?;
            let channel_fds = [channel.as_fd()];
            let mut ancillary_storage =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(CHANNELS_PER_FRAME))];
            let mut ancillary = SendAncillaryBuffer::new(&mut ancillary_storage);

            if ancillary.push(SendAncillaryMessage::ScmRights(&channel_fds)) {
                self.send_channel_header(&header, &mut ancillary)
            } else {
                Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "ancillary buffer is too small for channel descriptors",
                ))
            }
        };
        drop(channel);
        result
    }

    /// Receive the next frame with [`DEFAULT_MAX_PAYLOAD_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails, the peer closes, or the next frame
    /// violates the protocol.
    #[inline]
    pub fn recv(&mut self) -> io::Result<Packet> {
        self.recv_with_max_payload_size(DEFAULT_MAX_PAYLOAD_SIZE)
    }

    /// Receive the next frame, allowing at most `max_payload_size` bytes.
    ///
    /// Returns `UnexpectedEof` when the peer has closed the channel. Returns
    /// `InvalidData` if the frame is malformed or exceeds `max_payload_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails, the peer closes, or the next frame
    /// violates the protocol.
    #[inline]
    pub fn recv_with_max_payload_size(&mut self, max_payload_size: usize) -> io::Result<Packet> {
        let mut header_bytes = [0; FRAME_HEADER_LEN];
        let mut channels = self.recv_exact_collecting_channels(&mut header_bytes)?;
        let header = FrameHeader::read_from_bytes(&header_bytes)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid frame header"))?;
        let kind = header.kind()?;
        let payload_len = header.payload_len()?;

        if payload_len > max_payload_size {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "frame payload exceeds maximum size",
            ));
        }

        match kind {
            FrameKind::Bytes => {
                if !channels.is_empty() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "byte frame carried a channel descriptor",
                    ));
                }

                let mut payload = vec![0; payload_len];
                let payload_channels = self.recv_exact_collecting_channels(&mut payload)?;
                if !payload_channels.is_empty() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "byte frame payload carried a channel descriptor",
                    ));
                }

                Ok(Packet::Bytes(payload))
            }
            FrameKind::Channel => {
                if payload_len != 0 {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "channel frame carried a byte payload",
                    ));
                }

                match channels.pop() {
                    Some(channel) if channels.is_empty() => Ok(Packet::Channel(channel)),
                    _ => Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "channel frame did not carry exactly one channel descriptor",
                    )),
                }
            }
        }
    }

    /// Receive the next byte frame with [`DEFAULT_MAX_PAYLOAD_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next frame is not a byte
    /// frame.
    #[inline]
    pub fn recv_bytes(&mut self) -> io::Result<Vec<u8>> {
        self.recv_bytes_with_max_payload_size(DEFAULT_MAX_PAYLOAD_SIZE)
    }

    /// Receive the next byte frame, allowing at most `max_payload_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next frame is not a byte
    /// frame.
    #[inline]
    pub fn recv_bytes_with_max_payload_size(
        &mut self,
        max_payload_size: usize,
    ) -> io::Result<Vec<u8>> {
        match self.recv_with_max_payload_size(max_payload_size)? {
            Packet::Bytes(bytes) => Ok(bytes),
            Packet::Channel(_) => Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a byte frame, received a channel frame",
            )),
        }
    }

    /// Receive the next channel frame.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next frame is not a channel
    /// frame.
    #[inline]
    pub fn recv_channel(&mut self) -> io::Result<Self> {
        match self.recv_with_max_payload_size(0)? {
            Packet::Channel(channel) => Ok(channel),
            Packet::Bytes(_) => Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a channel frame, received a byte frame",
            )),
        }
    }

    fn recv_exact_collecting_channels(&self, mut buffer: &mut [u8]) -> io::Result<Vec<Self>> {
        let mut channels = Vec::new();

        while !buffer.is_empty() {
            let mut ancillary_storage =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(CHANNELS_PER_FRAME))];
            let mut ancillary = RecvAncillaryBuffer::new(&mut ancillary_storage);

            let received = {
                let mut iov = [IoSliceMut::new(buffer)];
                retry_on_intr(|| recvmsg(&self.fd, &mut iov, &mut ancillary, RecvFlags::empty()))
                    .map_err(io::Error::from)?
            };

            if received.bytes == 0 {
                return Err(io::Error::new(ErrorKind::UnexpectedEof, "channel closed"));
            }

            if received.flags.contains(ReturnFlags::TRUNC) {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "stream frame payload truncated",
                ));
            }

            if received.flags.contains(ReturnFlags::CTRUNC) {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "stream frame channel descriptors truncated",
                ));
            }

            for message in ancillary.drain() {
                if let RecvAncillaryMessage::ScmRights(fds) = message {
                    for fd in fds {
                        // SAFETY: Attached descriptors are interpreted as
                        // framed-channel endpoints by this protocol. Malicious
                        // peers can still cause I/O errors, but not Rust memory
                        // unsafety.
                        channels.push(unsafe { Self::from_owned_fd(fd)? });
                    }
                }
            }

            buffer = buffer.get_mut(received.bytes..).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    "stream socket over-reported received bytes",
                )
            })?;
        }

        Ok(channels)
    }

    fn send_all(&self, mut bytes: &[u8]) -> io::Result<()> {
        while !bytes.is_empty() {
            let iov = [IoSlice::new(bytes)];
            let mut ancillary = SendAncillaryBuffer::default();
            let sent = retry_on_intr(|| sendmsg(&self.fd, &iov, &mut ancillary, send_flags()))
                .map_err(io::Error::from)?;
            if sent == 0 {
                return Err(io::Error::new(
                    ErrorKind::WriteZero,
                    "stream socket accepted zero bytes",
                ));
            }

            bytes = bytes.get(sent..).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::WriteZero,
                    "stream socket over-reported sent bytes",
                )
            })?;
        }

        Ok(())
    }

    fn send_channel_header(
        &self,
        header: &FrameHeader,
        ancillary: &mut SendAncillaryBuffer<'_, '_, '_>,
    ) -> io::Result<()> {
        let iov = [IoSlice::new(header.as_bytes())];
        let sent = retry_on_intr(|| sendmsg(&self.fd, &iov, ancillary, send_flags()))
            .map_err(io::Error::from)?;

        if sent != FRAME_HEADER_LEN {
            return Err(io::Error::new(
                ErrorKind::WriteZero,
                "stream socket accepted a partial channel frame header",
            ));
        }

        Ok(())
    }
}

impl AsFd for UnixPacketChannel {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for UnixPacketChannel {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

fn configure_socket<Fd: AsFd>(fd: Fd) -> io::Result<()> {
    set_close_on_exec(fd.as_fd())?;

    #[cfg(target_os = "macos")]
    rustix::net::sockopt::set_socket_nosigpipe(fd.as_fd(), true).map_err(io::Error::from)?;

    Ok(())
}

fn set_close_on_exec<Fd: AsFd>(fd: Fd) -> io::Result<()> {
    let flags = fcntl_getfd(fd.as_fd()).map_err(io::Error::from)?;
    fcntl_setfd(fd.as_fd(), flags | FdFlags::CLOEXEC).map_err(io::Error::from)
}

fn validate_stream_socket(fd: &OwnedFd) -> io::Result<()> {
    let socket_type = rustix::net::sockopt::socket_type(fd).map_err(io::Error::from)?;
    if socket_type != SocketType::STREAM {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "file descriptor is not a SOCK_STREAM socket",
        ));
    }

    Ok(())
}

#[cfg(target_os = "linux")]
const fn send_flags() -> SendFlags {
    SendFlags::NOSIGNAL
}

#[cfg(not(target_os = "linux"))]
const fn send_flags() -> SendFlags {
    SendFlags::empty()
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn sends_and_receives_bytes() -> io::Result<()> {
        let (mut left, mut right) = UnixPacketChannel::pair()?;

        left.send(b"hello")?;
        assert_eq!(right.recv_bytes()?, b"hello");

        Ok(())
    }

    #[test]
    fn sends_and_receives_large_bytes() -> io::Result<()> {
        let (mut left, right) = UnixPacketChannel::pair()?;
        let payload = vec![42; 512 * 1024];
        let max_frame_size = payload.len();

        let receiver_thread = thread::spawn(move || {
            let mut receiver_channel = right;
            receiver_channel.recv_bytes_with_max_payload_size(max_frame_size)
        });
        left.send(&payload)?;

        let received_payload = receiver_thread
            .join()
            .map_err(|_| io::Error::other("receiver thread panicked"))??;
        assert_eq!(received_payload, payload);

        Ok(())
    }

    #[test]
    fn preserves_frame_boundaries() -> io::Result<()> {
        let (mut left, mut right) = UnixPacketChannel::pair()?;

        left.send(b"first")?;
        left.send(b"second")?;

        assert_eq!(right.recv_bytes()?, b"first");
        assert_eq!(right.recv_bytes()?, b"second");

        Ok(())
    }

    #[test]
    fn supports_empty_payloads() -> io::Result<()> {
        let (mut left, mut right) = UnixPacketChannel::pair()?;

        left.send(b"")?;

        assert!(right.recv_bytes()?.is_empty());

        Ok(())
    }

    #[test]
    fn transfers_channels_over_channels() -> io::Result<()> {
        let (mut control_left, mut control_right) = UnixPacketChannel::pair()?;
        let (mut nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_channel(nested_right)?;

        let mut received_nested = control_right.recv_channel()?;
        nested_left.send(b"through nested")?;

        assert_eq!(received_nested.recv_bytes()?, b"through nested");

        Ok(())
    }

    #[test]
    fn preserves_order_across_bytes_and_channels() -> io::Result<()> {
        let (mut control_left, mut control_right) = UnixPacketChannel::pair()?;
        let (mut nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send(b"before")?;
        control_left.send_channel(nested_right)?;
        control_left.send(b"after")?;

        assert_eq!(control_right.recv_bytes()?, b"before");
        let mut received_nested = control_right.recv_channel()?;
        assert_eq!(control_right.recv_bytes()?, b"after");

        nested_left.send(b"through nested")?;
        assert_eq!(received_nested.recv_bytes()?, b"through nested");

        Ok(())
    }

    #[test]
    fn rejects_oversized_payloads() -> io::Result<()> {
        let (mut left, mut right) = UnixPacketChannel::pair()?;

        left.send(b"too large")?;

        let Err(err) = right.recv_with_max_payload_size(3) else {
            return Err(io::Error::other("frame should be too large"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }

    #[test]
    fn rejects_unexpected_frame_kind() -> io::Result<()> {
        let (mut control_left, mut control_right) = UnixPacketChannel::pair()?;
        let (_nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_channel(nested_right)?;

        let Err(err) = control_right.recv_bytes() else {
            return Err(io::Error::other("frame should be a channel"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }
}
