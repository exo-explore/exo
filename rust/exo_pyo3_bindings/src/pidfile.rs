use pidfile_rs::{Pidfile, PidfileError};
use pyo3::exceptions::PyException;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{pyclass, pymethods, Bound, PyErr, PyResult, Python};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::fs;
use std::fs::Permissions;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::prelude::PermissionsExt;
use std::path::PathBuf;

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
/// lock it, detect already running daemons. It is backed by [`pidfile`][]
/// functions of `libbsd`/`libutil` which use `flopen` to lock the PID
/// file.
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
pub struct PyPidfile(Pidfile);

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

        Ok(Self(
            Pidfile::new(&path, Permissions::from_mode(mode))
                .map_err(|e| PyPidfileError(e).into_pyerr(py))?,
        ))
    }

    /// Writes the current process ID to the PID file.
    ///
    /// The file is truncated before writing.
    fn write<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
        self.0.write().map_err(|e| PyPidfileError(e).into_pyerr(py))
    }

    /// Extracts the raw file descriptor.
    ///
    /// This function is typically used to **borrow** an owned file descriptor.
    /// When used in this way, this method does **not** pass ownership of the
    /// raw file descriptor to the caller, and the file descriptor is only
    /// guaranteed to be valid while the original object has not yet been
    /// destroyed.
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }

    /// Checks if cleanup is currently enabled.
    ///
    /// Enabled cleanup means normal destructor cleanup logic will run,
    /// and if disabled then it won't run. It should always be `True`
    /// unless dangerously changed by [`Self::_dangerously_set_cleanup`].
    #[getter]
    fn cleanup(&self) -> bool {
        self.0.will_cleanup()
    }

    /// Dangerously configures if destructor cleanup logic should run.
    ///
    /// Disabling cleanup can be used to prevent removal of the file,
    /// for the purposes of e.g. double-fork demonization.
    fn _dangerously_set_cleanup(&mut self, cleanup: bool) {
        unsafe { self.0.config_cleanup(cleanup) }
    }
}

pub fn pidfile_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPidfileError>()?;
    m.add_class::<PyPidfile>()?;

    Ok(())
}
