use std::io::{self, ErrorKind, IoSlice, IoSliceMut};
use std::mem::{MaybeUninit, size_of};

use rustix::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd as _, IntoRawFd as _, OwnedFd, RawFd};
use rustix::io::{FdFlags, fcntl_getfd, fcntl_setfd, retry_on_intr};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
    SendAncillaryBuffer, SendAncillaryMessage, SendFlags, SocketType, recvmsg, sendmsg, socketpair,
};
use zerocopy::byteorder::{NetworkEndian, U16, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("packet_channel currently supports only Linux and macOS");

const FRAME_MARKER: u8 = b'P';
const HEADER_MAGIC: [u8; 4] = *b"EXPC";
const HEADER_VERSION: u8 = 1;
const HEADER_LEN: usize = size_of::<FrameHeader>();
const MAX_CHANNELS_PER_PACKET: usize = 16;

/// Default maximum byte payload accepted by [`UnixPacketChannel::recv`].
///
/// This is a protocol safety limit, not an expected message size. Increase it
/// when the application legitimately needs to pass larger serialized objects.
pub const DEFAULT_MAX_PAYLOAD_SIZE: usize = 64 * 1024 * 1024;

#[repr(C, packed)]
#[derive(Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
struct FrameHeader {
    magic: [u8; 4],
    version: u8,
    flags: u8,
    channel_count: U16<NetworkEndian>,
    payload_len: U64<NetworkEndian>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedFrameHeader {
    payload_len: usize,
    channel_count: usize,
}

impl FrameHeader {
    fn new(payload_len: usize, channel_count: usize) -> io::Result<Self> {
        if channel_count > MAX_CHANNELS_PER_PACKET {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "packet carries too many channel descriptors",
            ));
        }

        let payload_len = u64::try_from(payload_len).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "payload length does not fit in the wire header",
            )
        })?;
        let channel_count = u16::try_from(channel_count).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "channel count does not fit in the wire header",
            )
        })?;

        Ok(Self {
            magic: HEADER_MAGIC,
            version: HEADER_VERSION,
            flags: 0,
            channel_count: U16::new(channel_count),
            payload_len: U64::new(payload_len),
        })
    }

    fn decode(bytes: &[u8; HEADER_LEN]) -> io::Result<DecodedFrameHeader> {
        let header = Self::read_from_bytes(bytes)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid packet header"))?;
        if header.magic != HEADER_MAGIC {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "invalid packet header magic",
            ));
        }
        if header.version != HEADER_VERSION {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "unsupported packet header version",
            ));
        }
        if header.flags != 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet header reserved flags are non-zero",
            ));
        }

        let channel_count = usize::from(header.channel_count.get());
        let payload_len = usize::try_from(header.payload_len.get()).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidData,
                "payload length does not fit on this platform",
            )
        })?;

        if channel_count > MAX_CHANNELS_PER_PACKET {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet carries too many channel descriptors",
            ));
        }

        Ok(DecodedFrameHeader {
            payload_len,
            channel_count,
        })
    }
}

/// One logical packet received from a [`UnixPacketChannel`].
///
/// The byte payload and the attached channel descriptors are delivered together.
/// This mirrors Unix socket ancillary data: ordinary bytes are carried in the
/// stream, while descriptors are attached to a specific byte in that stream.
#[derive(Debug)]
pub struct Packet {
    pub bytes: Vec<u8>,
    pub channels: Vec<UnixPacketChannel>,
}

impl Packet {
    #[must_use]
    #[inline]
    pub const fn new(bytes: Vec<u8>, channels: Vec<UnixPacketChannel>) -> Self {
        Self { bytes, channels }
    }

    #[must_use]
    #[inline]
    pub const fn bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            channels: Vec::new(),
        }
    }

    #[must_use]
    #[inline]
    pub fn channel(channel: UnixPacketChannel) -> Self {
        Self {
            bytes: Vec::new(),
            channels: vec![channel],
        }
    }
}

/// A connected Unix-domain byte/channel transport.
///
/// This uses one unnamed `AF_UNIX`/`SOCK_STREAM` socket. Each logical packet is
/// framed as:
///
/// 1. a one-byte marker carrying any `SCM_RIGHTS` ancillary descriptors,
/// 2. a fixed-size header containing byte length and descriptor count,
/// 3. the byte payload.
///
/// Keeping descriptor transfer on a single marker byte avoids retrying
/// descriptor sends after a partial stream write, while `SOCK_STREAM` keeps
/// payload size independent of kernel datagram limits and works on Linux and
/// macOS.
#[derive(Debug)]
pub struct UnixPacketChannel {
    fd: OwnedFd,
}

impl UnixPacketChannel {
    /// Create a connected pair of unnamed Unix packet channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the socketpair cannot be created or configured.
    #[inline]
    pub fn pair() -> io::Result<(Self, Self)> {
        let (left, right) = socketpair(
            AddressFamily::UNIX,
            SocketType::STREAM,
            socket_creation_flags(),
            None,
        )
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("socketpair(AF_UNIX, SOCK_STREAM)", &err))?;

        let left = Self::from_owned_fd(left)?;
        let right = Self::from_owned_fd(right)?;

        Ok((left, right))
    }

    /// Wrap an owned file descriptor after validating it as a connected
    /// `AF_UNIX`/`SOCK_STREAM` socket endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a connected Unix stream socket or cannot
    /// be configured.
    #[inline]
    pub fn from_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        set_close_on_exec(&fd)?;
        validate_channel_fd(&fd)?;
        #[cfg(target_os = "macos")]
        configure_validated_socket(&fd)?;
        Ok(Self { fd })
    }

    /// Compatibility alias for [`Self::from_owned_fd`].
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a connected Unix stream socket or cannot
    /// be configured.
    #[inline]
    pub fn from_checked_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        Self::from_owned_fd(fd)
    }

    /// Wrap an inherited raw file descriptor after validating it as a connected
    /// `AF_UNIX`/`SOCK_STREAM` socket endpoint.
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path. After this
    /// function succeeds or fails, the descriptor is owned by Rust and will be
    /// closed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a connected Unix stream socket or cannot
    /// be configured.
    #[inline]
    pub unsafe fn from_raw_fd(raw_fd: RawFd) -> io::Result<Self> {
        // SAFETY: The caller guarantees that `raw_fd` is open and uniquely owned.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        Self::from_owned_fd(fd)
    }

    /// Compatibility alias for [`Self::from_raw_fd`].
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a connected Unix stream socket or cannot
    /// be configured.
    #[inline]
    pub unsafe fn from_checked_raw_fd(raw_fd: RawFd) -> io::Result<Self> {
        // SAFETY: The caller guarantees that `raw_fd` is open and uniquely owned.
        unsafe { Self::from_raw_fd(raw_fd) }
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

    /// Send an opaque byte message without attached channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet cannot be written.
    #[inline]
    pub fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.send_packet(Packet::bytes(bytes.to_vec()))
    }

    /// Send a packet containing bytes and zero or more channel endpoints.
    ///
    /// This consumes any channels in `packet`. The kernel duplicates descriptors
    /// into the receiving process; consuming local channel values gives this API
    /// transfer-of-ownership semantics at the Rust boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet cannot be written. On error, channels in
    /// `packet` have still been consumed and are dropped.
    #[inline]
    pub fn send_packet(&mut self, packet: Packet) -> io::Result<()> {
        let Packet { bytes, channels } = packet;
        let header = FrameHeader::new(bytes.len(), channels.len())?;

        self.send_marker_with_channels(&channels)?;
        self.send_all(header.as_bytes())?;
        self.send_all(&bytes)
    }

    /// Send a framed-channel endpoint over this channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet cannot be written.
    #[inline]
    pub fn send_channel(&mut self, channel: Self) -> io::Result<()> {
        self.send_packet(Packet::channel(channel))
    }

    /// Receive the next packet with [`DEFAULT_MAX_PAYLOAD_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails, the peer closes, or the next packet
    /// violates the protocol.
    #[inline]
    pub fn recv(&mut self) -> io::Result<Packet> {
        self.recv_with_max_payload_size(DEFAULT_MAX_PAYLOAD_SIZE)
    }

    /// Receive the next packet, allowing at most `max_payload_size` bytes.
    ///
    /// Returns `UnexpectedEof` when the peer has closed the channel. Returns
    /// `InvalidData` if the frame is malformed or exceeds `max_payload_size`.
    /// Protocol errors should be treated as fatal to this channel.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails, the peer closes, or the next packet
    /// violates the protocol.
    #[inline]
    pub fn recv_with_max_payload_size(&mut self, max_payload_size: usize) -> io::Result<Packet> {
        let received_fds = self.recv_marker_with_channels()?;

        let mut header_bytes = [0; HEADER_LEN];
        self.recv_exact_without_channels(&mut header_bytes)?;
        let header = FrameHeader::decode(&header_bytes)?;

        if header.channel_count != received_fds.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet header channel count does not match ancillary data",
            ));
        }
        if header.payload_len > max_payload_size {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet payload exceeds maximum size",
            ));
        }

        let mut payload = vec![0; header.payload_len];
        self.recv_exact_without_channels(&mut payload)?;

        let mut channels = Vec::with_capacity(received_fds.len());
        for fd in received_fds {
            channels.push(Self::from_owned_fd(fd)?);
        }

        Ok(Packet {
            bytes: payload,
            channels,
        })
    }

    /// Receive the next byte-only packet with [`DEFAULT_MAX_PAYLOAD_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next packet carries channels.
    #[inline]
    pub fn recv_bytes(&mut self) -> io::Result<Vec<u8>> {
        self.recv_bytes_with_max_payload_size(DEFAULT_MAX_PAYLOAD_SIZE)
    }

    /// Receive the next byte-only packet, allowing at most `max_payload_size`
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next packet carries channels.
    #[inline]
    pub fn recv_bytes_with_max_payload_size(
        &mut self,
        max_payload_size: usize,
    ) -> io::Result<Vec<u8>> {
        let packet = self.recv_with_max_payload_size(max_payload_size)?;
        if !packet.channels.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a byte-only packet, received channel descriptors",
            ));
        }

        Ok(packet.bytes)
    }

    /// Receive the next channel-only packet.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next packet is not exactly one
    /// channel with no byte payload.
    #[inline]
    pub fn recv_channel(&mut self) -> io::Result<Self> {
        let mut packet = self.recv_with_max_payload_size(0)?;
        if !packet.bytes.is_empty() || packet.channels.len() != 1 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a channel-only packet",
            ));
        }

        Ok(packet.channels.remove(0))
    }

    fn send_marker_with_channels(&self, channels: &[Self]) -> io::Result<()> {
        let borrowed_fds = channels
            .iter()
            .map(AsFd::as_fd)
            .collect::<Vec<BorrowedFd<'_>>>();
        let mut ancillary_storage =
            [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_CHANNELS_PER_PACKET))];
        let mut ancillary = SendAncillaryBuffer::new(&mut ancillary_storage);

        if !borrowed_fds.is_empty()
            && !ancillary.push(SendAncillaryMessage::ScmRights(&borrowed_fds))
        {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "ancillary buffer is too small for channel descriptors",
            ));
        }

        let marker = [FRAME_MARKER];
        let iov = [IoSlice::new(&marker)];
        let sent = retry_on_intr(|| sendmsg(&self.fd, &iov, &mut ancillary, send_flags()))
            .map_err(io::Error::from)?;

        if sent != marker.len() {
            return Err(io::Error::new(
                ErrorKind::WriteZero,
                "stream socket failed to write packet marker",
            ));
        }

        Ok(())
    }

    fn recv_marker_with_channels(&self) -> io::Result<Vec<OwnedFd>> {
        let mut marker = [0];
        let mut ancillary_storage =
            [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_CHANNELS_PER_PACKET))];
        let mut ancillary = RecvAncillaryBuffer::new(&mut ancillary_storage);

        let received = {
            let mut iov = [IoSliceMut::new(&mut marker)];
            retry_on_intr(|| recvmsg(&self.fd, &mut iov, &mut ancillary, recv_flags()))
                .map_err(io::Error::from)?
        };

        if received.bytes == 0 {
            return Err(io::Error::new(ErrorKind::UnexpectedEof, "channel closed"));
        }
        if received.bytes != marker.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "stream socket over-reported marker bytes",
            ));
        }
        check_return_flags(received.flags)?;
        if marker[0] != FRAME_MARKER {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "invalid packet marker",
            ));
        }

        collect_received_fds(&mut ancillary)
    }

    fn recv_exact_without_channels(&self, mut buffer: &mut [u8]) -> io::Result<()> {
        while !buffer.is_empty() {
            let mut ancillary_storage =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_CHANNELS_PER_PACKET))];
            let mut ancillary = RecvAncillaryBuffer::new(&mut ancillary_storage);

            let received = {
                let mut iov = [IoSliceMut::new(buffer)];
                retry_on_intr(|| recvmsg(&self.fd, &mut iov, &mut ancillary, recv_flags()))
                    .map_err(io::Error::from)?
            };

            if received.bytes == 0 {
                return Err(io::Error::new(ErrorKind::UnexpectedEof, "channel closed"));
            }
            check_return_flags(received.flags)?;
            if !collect_received_fds(&mut ancillary)?.is_empty() {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "descriptor ancillary data appeared outside a packet marker",
                ));
            }

            buffer = buffer.get_mut(received.bytes..).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    "stream socket over-reported received bytes",
                )
            })?;
        }

        Ok(())
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

fn collect_received_fds(ancillary: &mut RecvAncillaryBuffer<'_>) -> io::Result<Vec<OwnedFd>> {
    let mut fds = Vec::new();
    for message in ancillary.drain() {
        if let RecvAncillaryMessage::ScmRights(received_fds) = message {
            for fd in received_fds {
                if fds.len() == MAX_CHANNELS_PER_PACKET {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "packet carried too many channel descriptors",
                    ));
                }
                fds.push(fd);
            }
        }
    }

    Ok(fds)
}

fn check_return_flags(flags: ReturnFlags) -> io::Result<()> {
    if flags.contains(ReturnFlags::TRUNC) {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "stream data truncated",
        ));
    }
    if flags.contains(ReturnFlags::CTRUNC) {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "descriptor ancillary data truncated",
        ));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_validated_socket<Fd: AsFd>(_fd: Fd) -> io::Result<()> {
    rustix::net::sockopt::set_socket_nosigpipe(_fd.as_fd(), true)
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("setsockopt(SO_NOSIGPIPE)", &err))?;

    Ok(())
}

fn set_close_on_exec<Fd: AsFd>(fd: Fd) -> io::Result<()> {
    let flags = fcntl_getfd(fd.as_fd())
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("fcntl(F_GETFD)", &err))?;
    fcntl_setfd(fd.as_fd(), flags | FdFlags::CLOEXEC)
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("fcntl(F_SETFD, FD_CLOEXEC)", &err))
}

fn validate_channel_fd(fd: &OwnedFd) -> io::Result<()> {
    let socket_type = rustix::net::sockopt::socket_type(fd)
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("getsockopt(SO_TYPE)", &err))?;
    if socket_type != SocketType::STREAM {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "file descriptor is not a SOCK_STREAM socket",
        ));
    }

    let local_name = rustix::net::getsockname(fd)
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("getsockname", &err))?;
    if local_name.address_family() != AddressFamily::UNIX {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "file descriptor is not an AF_UNIX socket",
        ));
    }

    let peer_name = rustix::net::getpeername(fd)
        .map_err(io::Error::from)
        .map_err(|err| annotate_io_error("getpeername", &err))?
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "file descriptor is not a connected socket",
            )
        })?;
    if peer_name.address_family() != AddressFamily::UNIX {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "file descriptor peer is not an AF_UNIX socket",
        ));
    }

    Ok(())
}

fn annotate_io_error(context: &'static str, err: &io::Error) -> io::Error {
    io::Error::new(err.kind(), format!("{context}: {err}"))
}

#[cfg(target_os = "linux")]
const fn socket_creation_flags() -> rustix::net::SocketFlags {
    rustix::net::SocketFlags::CLOEXEC
}

#[cfg(target_os = "macos")]
const fn socket_creation_flags() -> rustix::net::SocketFlags {
    rustix::net::SocketFlags::empty()
}

#[cfg(target_os = "linux")]
const fn recv_flags() -> RecvFlags {
    RecvFlags::CMSG_CLOEXEC
}

#[cfg(target_os = "macos")]
const fn recv_flags() -> RecvFlags {
    RecvFlags::empty()
}

#[cfg(target_os = "linux")]
const fn send_flags() -> SendFlags {
    SendFlags::NOSIGNAL
}

#[cfg(target_os = "macos")]
const fn send_flags() -> SendFlags {
    SendFlags::empty()
}

#[cfg(test)]
mod tests {
    use std::fs::File;
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
    fn sends_bytes_and_channel_in_one_packet() -> io::Result<()> {
        let (mut control_left, mut control_right) = UnixPacketChannel::pair()?;
        let (mut nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_packet(Packet::new(b"metadata".to_vec(), vec![nested_right]))?;

        let mut packet = control_right.recv()?;
        assert_eq!(packet.bytes, b"metadata");
        assert_eq!(packet.channels.len(), 1);

        let mut received_nested = packet.channels.remove(0);
        nested_left.send(b"nested payload")?;
        assert_eq!(received_nested.recv_bytes()?, b"nested payload");

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
    fn rejects_unexpected_packet_shape() -> io::Result<()> {
        let (mut control_left, mut control_right) = UnixPacketChannel::pair()?;
        let (_nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_channel(nested_right)?;

        let Err(err) = control_right.recv_bytes() else {
            return Err(io::Error::other("packet should carry a channel"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }

    #[test]
    fn rejects_non_channel_descriptors() -> io::Result<()> {
        let (left, mut right) = UnixPacketChannel::pair()?;
        let file = File::open("/dev/null")?;
        let borrowed_fds = [file.as_fd()];

        let mut ancillary_storage =
            [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_CHANNELS_PER_PACKET))];
        let mut ancillary = SendAncillaryBuffer::new(&mut ancillary_storage);
        assert!(ancillary.push(SendAncillaryMessage::ScmRights(&borrowed_fds)));

        let marker = [FRAME_MARKER];
        let iov = [IoSlice::new(&marker)];
        let sent = retry_on_intr(|| sendmsg(&left.fd, &iov, &mut ancillary, send_flags()))
            .map_err(io::Error::from)?;
        assert_eq!(sent, marker.len());

        let header = FrameHeader::new(0, 1)?;
        left.send_all(header.as_bytes())?;

        let Err(_) = right.recv() else {
            return Err(io::Error::other("plain file descriptor should be rejected"));
        };

        Ok(())
    }
}
