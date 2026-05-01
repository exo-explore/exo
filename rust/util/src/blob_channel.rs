use std::io::{self, ErrorKind, Read as _, Write as _};
use std::mem::size_of;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd as _, IntoRawFd as _, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

const LENGTH_PREFIX_SIZE: usize = size_of::<u64>();

/// Default maximum blob size accepted by [`UnixBlobChannel::recv`].
///
/// This is a receiver-side allocation guard, not an expected message size.
pub const DEFAULT_MAX_BLOB_SIZE: usize = 64 * 1024 * 1024;

/// A connected Unix-domain channel for length-prefixed binary blobs.
#[derive(Debug)]
pub struct UnixBlobChannel {
    stream: UnixStream,
}

impl UnixBlobChannel {
    /// Create a connected pair of unnamed Unix blob channels.
    ///
    /// # Errors
    ///
    /// Returns an error if the socketpair cannot be created.
    #[inline]
    pub fn pair() -> io::Result<(Self, Self)> {
        let (left, right) = UnixStream::pair()?;
        Ok((Self { stream: left }, Self { stream: right }))
    }

    /// Wrap an owned file descriptor as a Unix blob channel.
    #[must_use]
    #[inline]
    pub fn from_owned_fd(fd: OwnedFd) -> Self {
        Self {
            stream: UnixStream::from(fd),
        }
    }

    /// Wrap an inherited raw file descriptor as a Unix blob channel.
    ///
    /// # Safety
    ///
    /// `raw_fd` must be open and uniquely owned by this call path. After this
    /// function returns, the descriptor is owned by Rust and will be closed on
    /// drop.
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        Self {
            // SAFETY: The caller guarantees that `raw_fd` is open and uniquely
            // owned by this call path.
            stream: unsafe { UnixStream::from_raw_fd(raw_fd) },
        }
    }

    /// Return the underlying raw file descriptor.
    #[must_use]
    #[inline]
    pub fn raw_fd(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// Consume this channel and return its owned file descriptor.
    #[must_use]
    #[inline]
    pub fn into_owned_fd(self) -> OwnedFd {
        self.stream.into()
    }

    /// Consume this channel and return its raw file descriptor.
    #[must_use]
    #[inline]
    pub fn into_raw_fd(self) -> RawFd {
        self.stream.into_raw_fd()
    }

    /// Send one binary blob.
    ///
    /// # Errors
    ///
    /// Returns an error if the length prefix or blob cannot be written.
    #[inline]
    pub fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        let len = u64::try_from(bytes.len()).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "blob length does not fit in the wire header",
            )
        })?;

        self.stream.write_all(&len.to_be_bytes())?;
        self.stream.write_all(bytes)
    }

    /// Receive one binary blob using [`DEFAULT_MAX_BLOB_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns an error if the length prefix or blob cannot be read, or if the
    /// announced blob length exceeds [`DEFAULT_MAX_BLOB_SIZE`].
    #[inline]
    pub fn recv(&mut self) -> io::Result<Vec<u8>> {
        self.recv_with_max_blob_size(DEFAULT_MAX_BLOB_SIZE)
    }

    /// Receive one binary blob, allowing at most `max_blob_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the length prefix or blob cannot be read, or if the
    /// announced blob length exceeds `max_blob_size`.
    #[inline]
    pub fn recv_with_max_blob_size(&mut self, max_blob_size: usize) -> io::Result<Vec<u8>> {
        let mut len_bytes = [0; LENGTH_PREFIX_SIZE];
        self.stream.read_exact(&mut len_bytes)?;

        let len = u64::from_be_bytes(len_bytes);
        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidData,
                "blob length does not fit on this platform",
            )
        })?;

        if len > max_blob_size {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "blob length exceeds maximum size",
            ));
        }

        let mut bytes = vec![0; len];
        self.stream.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

impl AsFd for UnixBlobChannel {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.stream.as_fd()
    }
}

impl AsRawFd for UnixBlobChannel {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.stream.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn sends_and_receives_bytes() -> io::Result<()> {
        let (mut left, mut right) = UnixBlobChannel::pair()?;

        left.send(b"hello")?;
        assert_eq!(right.recv()?, b"hello");

        Ok(())
    }

    #[test]
    fn sends_and_receives_empty_blob() -> io::Result<()> {
        let (mut left, mut right) = UnixBlobChannel::pair()?;

        left.send(b"")?;
        assert!(right.recv()?.is_empty());

        Ok(())
    }

    #[test]
    fn preserves_blob_boundaries() -> io::Result<()> {
        let (mut left, mut right) = UnixBlobChannel::pair()?;

        left.send(b"first")?;
        left.send(b"second")?;

        assert_eq!(right.recv()?, b"first");
        assert_eq!(right.recv()?, b"second");

        Ok(())
    }

    #[test]
    fn sends_and_receives_large_blob() -> io::Result<()> {
        let (mut left, right) = UnixBlobChannel::pair()?;
        let payload = deterministic_blob(200 * 1024 * 1024);
        let max_blob_size = payload.len();

        let receiver_thread = thread::spawn(move || {
            let mut receiver = right;
            receiver.recv_with_max_blob_size(max_blob_size)
        });
        left.send(&payload)?;

        let received = receiver_thread
            .join()
            .map_err(|_| io::Error::other("receiver thread panicked"))??;
        assert_eq!(received, payload);

        Ok(())
    }

    fn deterministic_blob(len: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut bytes = Vec::with_capacity(len);

        while bytes.len() < len {
            state = state
                .wrapping_mul(0xbf58_476d_1ce4_e5b9)
                .wrapping_add(0x94d0_49bb_1331_11eb);
            bytes.push(state.to_le_bytes()[3]);
        }

        bytes
    }

    #[test]
    fn rejects_oversized_blob() -> io::Result<()> {
        let (mut left, mut right) = UnixBlobChannel::pair()?;

        left.send(b"too large")?;

        let Err(err) = right.recv_with_max_blob_size(3) else {
            return Err(io::Error::other("blob should be too large"));
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        Ok(())
    }
}
