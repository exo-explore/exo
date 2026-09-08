use pyo3::exceptions::PyException;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyErr, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::fs;
use std::path::PathBuf;

#[cfg(unix)]
use pidfile_rs::{Pidfile, PidfileError};
#[cfg(unix)]
use std::fs::Permissions;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::prelude::PermissionsExt;

#[cfg(windows)]
mod win {
    use std::fs::{File, OpenOptions};
    use std::io::{self, Seek, SeekFrom, Write};
    use std::os::windows::fs::OpenOptionsExt;
    use std::path::{Path, PathBuf};

    /// ERROR_SHARING_VIOLATION — another process already holds the pidfile.
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const FILE_SHARE_READ: u32 = 0x0000_0001;

    #[derive(Debug)]
    pub enum PidfileError {
        Io(io::Error),
        AlreadyRunning { pid: Option<u32> },
    }

    impl std::fmt::Display for PidfileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Io(err) => write!(f, "{err}"),
                Self::AlreadyRunning { pid: Some(pid) } => {
                    write!(f, "daemon already running with PID {pid}")
                }
                Self::AlreadyRunning { pid: None } => {
                    write!(f, "daemon already running")
                }
            }
        }
    }

    pub struct Pidfile {
        file: Option<File>,
        path: PathBuf,
    }

    impl Pidfile {
        pub fn new(path: &Path) -> Result<Self, PidfileError> {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .share_mode(FILE_SHARE_READ)
                .open(path)
                .map_err(|err| map_open_error(path, err))?;
            Ok(Self {
                file: Some(file),
                path: path.to_path_buf(),
            })
        }

        pub fn write(&mut self) -> Result<(), PidfileError> {
            let file = self.file.as_mut().ok_or_else(|| {
                PidfileError::Io(io::Error::other("pidfile already closed"))
            })?;
            file.set_len(0).map_err(PidfileError::Io)?;
            file.seek(SeekFrom::Start(0)).map_err(PidfileError::Io)?;
            file.write_all(std::process::id().to_string().as_bytes())
                .map_err(PidfileError::Io)?;
            file.flush().map_err(PidfileError::Io)?;
            Ok(())
        }
    }

    impl Drop for Pidfile {
        fn drop(&mut self) {
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn map_open_error(path: &Path, err: io::Error) -> PidfileError {
        if err.raw_os_error() == Some(ERROR_SHARING_VIOLATION) {
            let pid = std::fs::read_to_string(path)
                .ok()
                .and_then(|contents| contents.trim().parse().ok());
            PidfileError::AlreadyRunning { pid }
        } else {
            PidfileError::Io(err)
        }
    }
}

#[cfg(windows)]
use win::{Pidfile, PidfileError};

#[gen_stub_pyclass]
#[pyclass(frozen, extends=PyException, name="PidfileError")]
pub struct PyPidfileError(PidfileError);

impl PyPidfileError {
    // TODO: I actually like this pattern a LOT more but how to abstract??
    fn into_pyerr(self, py: Python) -> PyErr {
        match Bound::new(py, self) {
            Ok(err) => PyErr::from_value(err.into_any()),
            Err(err) => err,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPidfileError {
    fn __repr__(&self) -> String {
        format!("PidfileError(\"{}\")", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

/// A PID file protected with a lock.
///
/// An instance of `Pidfile` can be used to manage a PID file: create it,
/// lock it, detect already running daemons. It is backed by [`pidfile`]
/// functions of `libbsd`/`libutil` which use `flopen` to lock the PID
/// file on Unix. On Windows it uses an exclusive write handle with
/// `FILE_SHARE_READ` so a second process cannot take over the lock.
///
/// When a PID file is created, the process ID of the current process is
/// *not* written there, making it possible to lock the PID file before
/// forking and only write the ID of the forked process when it is ready.
///
/// The PID file is deleted automatically when the `Pidfile` comes out of
/// the scope. To close the PID file without deleting it, for example, in
/// the parent process of a forked daemon, call `close()`.
///
/// [`exit`]: https://doc.rust-lang.org/std/process/fn.exit.html
/// [`pidfile`]: https://linux.die.net/man/3/pidfile
/// [`daemon`(3)]: https://linux.die.net/man/3/daemon
#[gen_stub_pyclass]
#[pyclass(name = "Pidfile")]
pub struct PyPidfile(Option<Pidfile>);

impl PyPidfile {
    #[inline(always)]
    fn get(&self) -> &Pidfile {
        self.0
            .as_ref()
            .expect("cannot use resource after exiting context")
    }

    #[inline(always)]
    fn get_mut(&mut self) -> &mut Pidfile {
        self.0
            .as_mut()
            .expect("cannot use resource after exiting context")
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPidfile {
    /// Creates a new PID file and locks it.
    ///
    /// If the PID file cannot be locked, returns `PidfileError::AlreadyRunning` with
    /// a PID of the already running process, or `None` if no PID has been written to
    /// the PID file yet.
    #[new]
    fn py_new(py: Python, path: PathBuf, mode: u32) -> PyResult<Self> {
        // create all parent directories if don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| PyPidfileError(PidfileError::Io(e)).into_pyerr(py))?;
        }

        #[cfg(unix)]
        let pidfile = Pidfile::new(&path, Permissions::from_mode(mode))
            .map_err(|e| PyPidfileError(e).into_pyerr(py))?;
        #[cfg(windows)]
        let pidfile = {
            let _ = mode;
            Pidfile::new(&path).map_err(|e| PyPidfileError(e).into_pyerr(py))?
        };
        Ok(Self(Some(pidfile)))
    }

    /// Writes the current process ID to the PID file.
    ///
    /// The file is truncated before writing.
    fn write<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
        self.get_mut()
            .write()
            .map_err(|e| PyPidfileError(e).into_pyerr(py))
    }

    /// Extracts the raw file descriptor.
    ///
    /// This function is typically used to **borrow** an owned file descriptor.
    /// When used in this way, this method does **not** pass ownership of the
    /// raw file descriptor to the caller, and the file descriptor is only
    /// guaranteed to be valid while the original object has not yet been
    /// destroyed.
    ///
    /// On Windows this always returns `-1`; daemonization is not supported.
    fn as_raw_fd(&self) -> i32 {
        #[cfg(unix)]
        {
            self.get().as_raw_fd()
        }
        #[cfg(windows)]
        {
            let _ = self.get();
            -1
        }
    }

    /// Closes the PID file and releases associated resources.
    fn close(&mut self) {
        self.0 = None;
    }
}

pub fn pidfile_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPidfileError>()?;
    m.add_class::<PyPidfile>()?;

    Ok(())
}
