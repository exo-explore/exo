use std::io::{self, ErrorKind, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;

use rustix::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd as _, IntoRawFd as _, OwnedFd, RawFd};
use rustix::io::{fcntl_getfd, fcntl_setfd, retry_on_intr, FdFlags};
use rustix::net::{
    recvmsg, sendmsg, socketpair, AddressFamily, RecvAncillaryBuffer,
    RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer, SendAncillaryMessage, SendFlags,
    SocketFlags, SocketType,
};

const CHANNELS_PER_PACKET: usize = 1;
const WIRE_BYTES_TAG: u8 = 0;
const WIRE_CHANNEL_TAG: u8 = 1;

/// One packet received from a [`UnixPacketChannel`].
#[derive(Debug)]
pub enum Packet {
    Bytes(Vec<u8>),
    Channel(UnixPacketChannel),
}

/// A connected Unix-domain, packet-preserving channel.
///
/// This is built on an unnamed `AF_UNIX`/`SOCK_SEQPACKET` socket. Messages retain
/// packet boundaries and can carry other `UnixPacketChannel` endpoints through
/// Unix ancillary data.
#[derive(Debug)]
pub struct UnixPacketChannel {
    fd: OwnedFd,
}

impl UnixPacketChannel {
    /// Create a connected pair of unnamed Unix packet channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the socketpair cannot be created or marked
    /// close-on-exec.
    #[inline]
    pub fn pair() -> io::Result<(Self, Self)> {
        let (left, right) = socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::empty(),
            None,
        )
        .map_err(io::Error::from)?;

        // SAFETY: `socketpair` created both descriptors as connected
        // `AF_UNIX`/`SOCK_SEQPACKET` endpoints, and `OwnedFd` gives this code
        // unique ownership of each descriptor.
        let left = unsafe { Self::from_owned_fd(left)? };
        // SAFETY: Same reasoning as above for the other end of the socketpair.
        let right = unsafe { Self::from_owned_fd(right)? };

        Ok((left, right))
    }

    /// Wrap an owned file descriptor as a Unix packet channel.
    ///
    /// This constructor trusts the caller that `fd` is an `AF_UNIX`/
    /// `SOCK_SEQPACKET` endpoint. Use [`Self::from_checked_owned_fd`] when the
    /// descriptor comes from an untrusted source and the host permits socket
    /// option inspection.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor cannot be marked close-on-exec.
    ///
    /// # Safety
    ///
    /// `fd` must be an `AF_UNIX`/`SOCK_SEQPACKET` endpoint intended to be used
    /// by this protocol. Use [`Self::from_checked_owned_fd`] when that invariant
    /// has not already been established.
    #[inline]
    pub unsafe fn from_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        set_close_on_exec(&fd)?;
        Ok(Self { fd })
    }

    /// Wrap an owned file descriptor after validating it as a packet socket.
    ///
    /// The descriptor is checked as a `SOCK_SEQPACKET` socket before being
    /// accepted. This intentionally does not validate `SO_DOMAIN`, because some
    /// container/sandbox profiles reject that `getsockopt` even for valid
    /// descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not a packet socket or cannot be
    /// marked close-on-exec.
    #[inline]
    pub fn from_checked_owned_fd(fd: OwnedFd) -> io::Result<Self> {
        validate_packet_socket(&fd)?;
        // SAFETY: `validate_packet_socket` checked the descriptor type. The
        // caller supplied `OwnedFd`, so Rust has unique ownership.
        unsafe { Self::from_owned_fd(fd) }
    }

    /// Wrap an inherited raw file descriptor as a Unix packet channel.
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path. After this
    /// function succeeds or fails, the descriptor is owned by Rust and will be
    /// closed on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor cannot be marked close-on-exec.
    #[inline]
    pub unsafe fn from_raw_fd(raw_fd: RawFd) -> io::Result<Self> {
        // SAFETY: The caller guarantees that `raw_fd` is open and uniquely owned.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        // SAFETY: The caller also guarantees this descriptor is the expected
        // packet-channel endpoint.
        unsafe { Self::from_owned_fd(fd) }
    }

    /// Wrap an inherited raw file descriptor after validating it as a packet
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
    /// Returns an error if the descriptor is not a packet socket or cannot be
    /// marked close-on-exec.
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

    /// Send an opaque byte packet without attached channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet cannot be sent.
    #[inline]
    pub fn send(&self, bytes: &[u8]) -> io::Result<()> {
        let mut ancillary = SendAncillaryBuffer::default();
        self.send_packet(WIRE_BYTES_TAG, bytes, &mut ancillary)
    }

    /// Send a packet-channel endpoint over this channel.
    ///
    /// This consumes `channel`. The kernel duplicates the descriptor into the
    /// receiver's process; consuming the local value gives this method
    /// transfer-of-ownership semantics at the Rust API boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if ancillary data cannot be prepared or the channel
    /// packet cannot be sent. On error, `channel` is still dropped.
    #[inline]
    pub fn send_channel(&self, channel: Self) -> io::Result<()> {
        let result = {
            let channel_fds = [channel.as_fd()];
            let mut ancillary_storage =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(CHANNELS_PER_PACKET))];
            let mut ancillary = SendAncillaryBuffer::new(&mut ancillary_storage);

            if ancillary.push(SendAncillaryMessage::ScmRights(&channel_fds)) {
                self.send_packet(WIRE_CHANNEL_TAG, &[], &mut ancillary)
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

    /// Receive the next packet, allowing at most `max_payload_size` bytes.
    ///
    /// Returns `UnexpectedEof` when the peer has closed the channel. Returns
    /// `InvalidData` if the payload or ancillary data was truncated.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails, the peer closes, the packet is
    /// malformed, or the packet exceeds `max_payload_size`.
    #[inline]
    pub fn recv(&self, max_payload_size: usize) -> io::Result<Packet> {
        let buffer_size = max_payload_size
            .checked_add(1)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "packet size overflow"))?;
        let mut buffer = vec![0; buffer_size];
        let mut iov = [IoSliceMut::new(&mut buffer)];
        let mut ancillary_storage =
            [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(CHANNELS_PER_PACKET))];
        let mut ancillary = RecvAncillaryBuffer::new(&mut ancillary_storage);

        let received =
            retry_on_intr(|| recvmsg(&self.fd, &mut iov, &mut ancillary, RecvFlags::empty()))
                .map_err(io::Error::from)?;

        if received.bytes == 0 {
            return Err(io::Error::new(ErrorKind::UnexpectedEof, "channel closed"));
        }

        if received.flags.contains(ReturnFlags::TRUNC) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet payload truncated",
            ));
        }

        if received.flags.contains(ReturnFlags::CTRUNC) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "packet channel descriptors truncated",
            ));
        }

        let tag = buffer
            .first()
            .copied()
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "missing packet tag"))?;
        if tag != WIRE_BYTES_TAG && tag != WIRE_CHANNEL_TAG {
            return Err(io::Error::new(ErrorKind::InvalidData, "invalid packet tag"));
        }

        let mut channels = Vec::new();
        for message in ancillary.drain() {
            if let RecvAncillaryMessage::ScmRights(fds) = message {
                for fd in fds {
                    // SAFETY: Attached descriptors are interpreted as
                    // packet-channel endpoints by this protocol. Malicious
                    // peers can still cause I/O errors, but not Rust memory
                    // unsafety.
                    channels.push(unsafe { Self::from_owned_fd(fd)? });
                }
            }
        }

        let payload = buffer.get(1..received.bytes).ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidData, "invalid packet payload bounds")
        })?;

        match tag {
            WIRE_BYTES_TAG => {
                if !channels.is_empty() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "byte packet carried a channel descriptor",
                    ));
                }

                Ok(Packet::Bytes(payload.to_vec()))
            }
            WIRE_CHANNEL_TAG => {
                if !payload.is_empty() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "channel packet carried a byte payload",
                    ));
                }

                match channels.pop() {
                    Some(channel) if channels.is_empty() => Ok(Packet::Channel(channel)),
                    _ => Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "channel packet did not carry exactly one channel descriptor",
                    )),
                }
            }
            _ => unreachable!("packet tag was validated above"),
        }
    }

    /// Receive the next byte packet.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next packet is not a byte
    /// packet.
    #[inline]
    pub fn recv_bytes(&self, max_payload_size: usize) -> io::Result<Vec<u8>> {
        match self.recv(max_payload_size)? {
            Packet::Bytes(bytes) => Ok(bytes),
            Packet::Channel(_) => Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a byte packet, received a channel packet",
            )),
        }
    }

    /// Receive the next channel packet.
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or the next packet is not a channel
    /// packet.
    #[inline]
    pub fn recv_channel(&self) -> io::Result<Self> {
        match self.recv(0)? {
            Packet::Channel(channel) => Ok(channel),
            Packet::Bytes(_) => Err(io::Error::new(
                ErrorKind::InvalidData,
                "expected a channel packet, received a byte packet",
            )),
        }
    }

    fn send_packet(
        &self,
        tag: u8,
        bytes: &[u8],
        ancillary: &mut SendAncillaryBuffer<'_, '_, '_>,
    ) -> io::Result<()> {
        let tag = [tag];
        let iov = [IoSlice::new(&tag), IoSlice::new(bytes)];

        let expected = bytes
            .len()
            .checked_add(tag.len())
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "packet size overflow"))?;
        let sent = retry_on_intr(|| sendmsg(&self.fd, &iov, ancillary, SendFlags::empty()))
            .map_err(io::Error::from)?;

        if sent != expected {
            return Err(io::Error::new(
                ErrorKind::WriteZero,
                "packet socket accepted a partial packet",
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

fn set_close_on_exec<Fd: AsFd>(fd: Fd) -> io::Result<()> {
    let flags = fcntl_getfd(fd.as_fd()).map_err(io::Error::from)?;
    fcntl_setfd(fd.as_fd(), flags | FdFlags::CLOEXEC).map_err(io::Error::from)
}

fn validate_packet_socket(fd: &OwnedFd) -> io::Result<()> {
    let socket_type = rustix::net::sockopt::socket_type(fd).map_err(io::Error::from)?;
    if socket_type != SocketType::SEQPACKET {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "file descriptor is not a SOCK_SEQPACKET socket",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MAX_PACKET_SIZE: usize = 1024;

    #[test]
    fn sends_and_receives_bytes() -> io::Result<()> {
        let (left, right) = UnixPacketChannel::pair()?;

        left.send(b"hello")?;
        assert_eq!(right.recv_bytes(TEST_MAX_PACKET_SIZE)?, b"hello");

        Ok(())
    }

    #[test]
    fn preserves_packet_boundaries() -> io::Result<()> {
        let (left, right) = UnixPacketChannel::pair()?;

        left.send(b"first")?;
        left.send(b"second")?;

        assert_eq!(right.recv_bytes(TEST_MAX_PACKET_SIZE)?, b"first");
        assert_eq!(right.recv_bytes(TEST_MAX_PACKET_SIZE)?, b"second");

        Ok(())
    }

    #[test]
    fn supports_empty_payloads() -> io::Result<()> {
        let (left, right) = UnixPacketChannel::pair()?;

        left.send(b"")?;

        assert!(right.recv_bytes(TEST_MAX_PACKET_SIZE)?.is_empty());

        Ok(())
    }

    #[test]
    fn transfers_channels_over_channels() -> io::Result<()> {
        let (control_left, control_right) = UnixPacketChannel::pair()?;
        let (nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_channel(nested_right)?;

        let received_nested = control_right.recv_channel()?;
        nested_left.send(b"through nested")?;

        assert_eq!(
            received_nested.recv_bytes(TEST_MAX_PACKET_SIZE)?,
            b"through nested"
        );

        Ok(())
    }

    #[test]
    fn rejects_truncated_payloads() -> io::Result<()> {
        let (left, right) = UnixPacketChannel::pair()?;

        left.send(b"too large")?;

        let Err(err) = right.recv(3) else {
            return Err(io::Error::other("packet should be too large"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }

    #[test]
    fn rejects_unexpected_packet_kind() -> io::Result<()> {
        let (control_left, control_right) = UnixPacketChannel::pair()?;
        let (_nested_left, nested_right) = UnixPacketChannel::pair()?;

        control_left.send_channel(nested_right)?;

        let Err(err) = control_right.recv_bytes(TEST_MAX_PACKET_SIZE) else {
            return Err(io::Error::other("packet should be a channel"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }
}
