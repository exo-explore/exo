//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

mod allow_threading;
pub mod config;
mod networking;
mod pidfile;

use crate::config::config_submodule;
use crate::networking::networking_submodule;
use crate::pidfile::pidfile_submodule;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, pymodule};
use pyo3_stub_gen::define_stub_info_gatherer;

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use crate::allow_threading::AllowThreads;
    use extend::ext;
    use pyo3::exceptions::{PyConnectionError, PyRuntimeError};
    use pyo3::types::PyBytes;
    use pyo3::{Py, PyErr, PyResult, Python};
    use tokio::runtime::Runtime;
    use tokio::sync::mpsc;
    use tokio::sync::mpsc::error::TryRecvError;
    use tokio::task::JoinHandle;

    #[ext(pub, name = ByteArrayExt)]
    impl [u8] {
        fn pybytes(&self) -> Py<PyBytes> {
            Python::attach(|py| PyBytes::new(py, self).unbind())
        }
    }

    #[ext(pub, name = ResultExt)]
    impl<T, E> Result<T, E>
    where
        E: ToString,
    {
        fn pyerr(self) -> PyResult<T> {
            self.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }

    pub trait FutureExt: Future + Sized {
        /// SEE: https://pyo3.rs/v0.28.3/async-await#detaching-from-the-interpreter-across-await
        fn allow_threads_py(self) -> AllowThreads<Self>
        where
            AllowThreads<Self>: Future,
        {
            AllowThreads::new(self)
        }
    }

    impl<T: Future> FutureExt for T {}

    #[ext(pub, name = PyErrExt)]
    impl PyErr {
        fn receiver_channel_closed() -> Self {
            PyConnectionError::new_err("Receiver channel closed unexpectedly")
        }
    }

    #[ext(pub, name = PyResultExt)]
    impl<T> PyResult<T> {
        fn write_unraisable(self) -> Option<T> {
            Python::attach(|py| self.write_unraisable_with(py))
        }

        fn write_unraisable_with(self, py: Python<'_>) -> Option<T> {
            match self {
                Ok(v) => Some(v),
                Err(e) => {
                    // write error back to python
                    e.write_unraisable(py, None);
                    None
                }
            }
        }
    }

    #[ext(pub, name = TokioRuntimeExt)]
    impl Runtime {
        #[inline(always)]
        fn spawn_with_scope<F>(&self, py: Python<'_>, future: F) -> PyResult<JoinHandle<F::Output>>
        where
            F: Future + Send + 'static,
            F::Output: Send + 'static,
        {
            use pyo3_async_runtimes::tokio::{get_current_locals, scope};
            let locals = get_current_locals(py)?;
            Ok(self.spawn(scope(locals, future)))
        }

        #[inline(always)]
        async fn run_with_scope<F>(&self, future: F) -> PyResult<F::Output>
        where
            F: Future + Send + 'static,
            F::Output: Send + 'static,
        {
            Python::attach(|py| self.spawn_with_scope(py, future))?
                .allow_threads_py()
                .await
                .pyerr()
        }
    }

    #[ext(pub, name = TokioMpscSenderExt)]
    impl<T> mpsc::Sender<T> {
        /// Sends a value, waiting until there is capacity.
        ///
        /// A successful send occurs when it is determined that the other end of the
        /// channel has not hung up already. An unsuccessful send would be one where
        /// the corresponding receiver has already been closed.
        async fn send_py(&self, value: T) -> PyResult<()> {
            self.send(value)
                .await
                .map_err(|_| PyErr::receiver_channel_closed())
        }
    }

    #[ext(pub, name = TokioMpscReceiverExt)]
    impl<T> mpsc::Receiver<T> {
        /// Receives the next value for this receiver.
        async fn recv_py(&mut self) -> PyResult<T> {
            self.recv().await.ok_or_else(PyErr::receiver_channel_closed)
        }

        /// Receives at most `limit` values for this receiver and returns them.
        ///
        /// For `limit = 0`, an empty collection of messages will be returned immediately.
        /// For `limit > 0`, if there are no messages in the channel's queue this method
        /// will sleep until a message is sent.
        async fn recv_many_py(&mut self, limit: usize) -> PyResult<Vec<T>> {
            // get updates from receiver channel
            let mut updates = Vec::with_capacity(limit);
            let received = self.recv_many(&mut updates, limit).await;

            // if we received zero items, then the channel was unexpectedly closed
            if limit != 0 && received == 0 {
                return Err(PyErr::receiver_channel_closed());
            }

            Ok(updates)
        }

        /// Tries to receive the next value for this receiver.
        fn try_recv_py(&mut self) -> PyResult<Option<T>> {
            match self.try_recv() {
                Ok(v) => Ok(Some(v)),
                Err(TryRecvError::Empty) => Ok(None),
                Err(TryRecvError::Disconnected) => Err(PyErr::receiver_channel_closed()),
            }
        }
    }
}

/// Resolving the version of the python project
pub(crate) mod version {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::PyAnyMethods;
    use pyo3::types::PyModule;
    use pyo3::{PyResult, Python};
    use std::env;
    use std::sync::OnceLock;

    const DEFAULT_VERSION: &str = env!("CARGO_PKG_VERSION");
    static VERSION: OnceLock<String> = OnceLock::new();

    /// Returns either the configured version of Exo (once set by [`set_version_once`])
    /// or falls back to `CARGO_PKG_VERSION` if that hasn't been configured.
    pub fn version() -> &'static str {
        VERSION.get().map_or(DEFAULT_VERSION, String::as_str)
    }

    /// First tries to find `EXO_PKG_VERSION` env-var, falls back to calling Python
    /// `importlib.metadata.version("exo")` to resolve the version of Exo
    pub fn set_version_once(py: Python<'_>) -> PyResult<()> {
        let v = if let Ok(v) = env::var("EXO_PKG_VERSION") {
            v
        } else {
            // essentially runs:
            // ```python
            // from importlib.metadata import version
            // version("exo")
            // ```
            PyModule::import(py, "importlib.metadata")?
                .getattr("version")?
                .call1(("exo",))?
                .extract()?
        };

        // sets version only once
        VERSION
            .set(v)
            .map_err(|_| PyRuntimeError::new_err("Cannot set exo_rs version twice".to_string()))
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "exo_rs", gil_used = true)]
fn main_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // install logger
    pyo3_log::init();

    // resolve version
    version::set_version_once(m.py())?;

    // configure runtime
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_async_runtimes::tokio::init(builder);

    // TODO: for now this is all NOT a submodule, but figure out how to make the submodule system
    //       work with maturin, where the types generate correctly, in the right folder, without
    //       too many importing issues...
    pidfile_submodule(m)?;
    networking_submodule(m)?;
    config_submodule(m)?;

    // top-level constructs
    // TODO: ...

    Ok(())
}

define_stub_info_gatherer!(stub_info);
