//! Python package for EXO Rust bindings.

module_doc!("exo_rs", "Python package for EXO Rust bindings.");

mod allow_threading;
pub mod ident;
pub mod networking;
pub mod pidfile;

use pyo3::{pyclass, pymodule};
use pyo3_stub_gen::{define_stub_info_gatherer, module_doc, reexport_module_members};

/// Namespace for all the constants used by this crate.
pub(crate) mod r#const {
    pub const MPSC_CHANNEL_SIZE: usize = 1024;
}

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
        /// SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await
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
        fn spawn_with_scope<F>(&self, py: Python<'_>, future: F) -> PyResult<JoinHandle<F::Output>>
        where
            F: Future + Send + 'static,
            F::Output: Send + 'static,
        {
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            Ok(self.spawn(pyo3_async_runtimes::tokio::scope(locals, future)))
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

#[pymodule(name = "_core", gil_used = true)]
mod py_exo_rs {
    #[pymodule_export]
    use super::ident::PyKeypair;
    #[pymodule_export]
    use super::networking::{
        PyAllQueuesFullError, PyFromSwarm, PyMessageTooLargeError, PyNetworkingHandle,
        PyNoPeersSubscribedToTopicError,
    };
    #[pymodule_export]
    use super::pidfile::{PyPidfile, PyPidfileError};
    use pyo3::{
        PyResult,
        prelude::{Bound, PyModule},
    };

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // install logger (TODO: change to tracing)
        pyo3_log::init();

        // create pyo3_async_runtimes
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.enable_all();
        pyo3_async_runtimes::tokio::init(builder);

        Ok(())
    }
}

// make sure these re-exports match the #[pymodule_export] from above
reexport_module_members!("exo_rs.ident" from "exo_rs._core";
    "Keypair");
reexport_module_members!("exo_rs.networking" from "exo_rs._core";
    "AllQueuesFullError", "FromSwarm", "MessageTooLargeError", "NetworkingHandle",
    "NoPeersSubscribedToTopicError");
reexport_module_members!("exo_rs.pidfile" from "exo_rs._core";
    "Pidfile", "PidfileError");

define_stub_info_gatherer!(stub_info);
