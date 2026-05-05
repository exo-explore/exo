use std::os::fd::RawFd;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::{PyModule, PyModuleMethods as _};
use pyo3::types::{PyBytes, PyBytesMethods as _};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use util::blob_channel::{DEFAULT_MAX_BLOB_SIZE, UnixBlobChannel};

#[gen_stub_pyclass]
#[pyclass(name = "UnixBlobChannel")]
#[derive(Debug)]
pub struct PyUnixBlobChannel {
    channel: Mutex<Option<UnixBlobChannel>>,
}

impl PyUnixBlobChannel {
    const fn new(channel: UnixBlobChannel) -> Self {
        Self {
            channel: Mutex::new(Some(channel)),
        }
    }

    fn lock_channel(&self) -> PyResult<MutexGuard<'_, Option<UnixBlobChannel>>> {
        self.channel
            .lock()
            .map_err(|_| PyRuntimeError::new_err("UnixBlobChannel lock poisoned"))
    }
}

#[allow(
    clippy::multiple_inherent_impl,
    clippy::significant_drop_tightening,
    clippy::use_self,
    clippy::wrong_self_convention
)]
#[gen_stub_pymethods]
#[pymethods]
impl PyUnixBlobChannel {
    /// Create a connected pair of unnamed Unix blob channels.
    #[staticmethod]
    fn pair() -> PyResult<(PyUnixBlobChannel, PyUnixBlobChannel)> {
        let (left, right) = UnixBlobChannel::pair().map_err(PyOSError::new_err)?;
        Ok((Self::new(left), Self::new(right)))
    }

    /// Wrap an inherited raw file descriptor.
    ///
    /// The returned object owns `fd`; do not close or reuse that descriptor
    /// elsewhere after calling this method.
    #[staticmethod]
    fn from_raw_fd(fd: RawFd) -> PyResult<Self> {
        if fd < 0 {
            return Err(PyValueError::new_err(
                "file descriptor must be non-negative",
            ));
        }

        // SAFETY: Python callers use this to adopt an inherited descriptor. The
        // wrapper owns and closes the descriptor after this point.
        Ok(Self::new(unsafe { UnixBlobChannel::from_raw_fd(fd) }))
    }

    /// Return the underlying file descriptor without transferring ownership.
    fn raw_fd(&self) -> PyResult<RawFd> {
        let raw_fd = self
            .lock_channel()?
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("UnixBlobChannel is closed"))?
            .raw_fd();
        Ok(raw_fd)
    }

    /// Alias for [`raw_fd`], matching Python file-like objects.
    fn fileno(&self) -> PyResult<RawFd> {
        self.raw_fd()
    }

    /// Consume this channel and return its file descriptor.
    ///
    /// After this method succeeds, the Python object is closed and the caller
    /// owns the returned descriptor.
    fn into_raw_fd(&self) -> PyResult<RawFd> {
        let channel = self
            .lock_channel()?
            .take()
            .ok_or_else(|| PyValueError::new_err("UnixBlobChannel is closed"))?;
        Ok(channel.into_raw_fd())
    }

    /// Close this channel.
    fn close(&self) -> PyResult<()> {
        drop(self.lock_channel()?.take());
        Ok(())
    }

    /// Return whether this channel has been closed or consumed.
    fn closed(&self) -> PyResult<bool> {
        Ok(self.lock_channel()?.is_none())
    }

    /// Send one binary blob.
    fn send(&self, py: Python<'_>, bytes: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = Vec::from(bytes.as_bytes());
        py.detach(|| {
            {
                let mut guard = self.lock_channel()?;
                let channel = guard
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("UnixBlobChannel is closed"))?;
                channel.send(&bytes)
            }
            .map_err(PyOSError::new_err)
        })
    }

    /// Receive one binary blob using the default maximum blob size.
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| {
            {
                let mut guard = self.lock_channel()?;
                let channel = guard
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("UnixBlobChannel is closed"))?;
                channel.recv()
            }
            .map_err(PyOSError::new_err)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Receive one binary blob, allowing at most `max_blob_size` bytes.
    fn recv_with_max_blob_size<'py>(
        &self,
        py: Python<'py>,
        max_blob_size: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| {
            {
                let mut guard = self.lock_channel()?;
                let channel = guard
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("UnixBlobChannel is closed"))?;
                channel.recv_with_max_blob_size(max_blob_size)
            }
            .map_err(PyOSError::new_err)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Default maximum blob size accepted by `recv`.
    #[staticmethod]
    const fn default_max_blob_size() -> usize {
        DEFAULT_MAX_BLOB_SIZE
    }
}

pub fn blob_channel_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUnixBlobChannel>()?;
    m.add("DEFAULT_MAX_BLOB_SIZE", DEFAULT_MAX_BLOB_SIZE)?;

    Ok(())
}
