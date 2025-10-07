//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
#![feature(tuple_trait)]
#![feature(unboxed_closures)]
// #![feature(stmt_expr_attributes)]
// #![feature(assert_matches)]
// #![feature(async_fn_in_dyn_trait)]
// #![feature(async_for_loop)]
// #![feature(auto_traits)]
// #![feature(negative_impls)]

extern crate core;
mod allow_threading;
mod examples;
pub(crate) mod networking;
pub(crate) mod pylibp2p;

use crate::networking::networking_submodule;
use crate::pylibp2p::ident::ident_submodule;
use crate::pylibp2p::multiaddr::multiaddr_submodule;
use pyo3::prelude::PyModule;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, pyclass, pymodule};
use pyo3_stub_gen::define_stub_info_gatherer;

/// Namespace for all the constants used by this crate.
pub(crate) mod r#const {
    pub const MPSC_CHANNEL_SIZE: usize = 1024;
}

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;
    use std::marker::Tuple;

    pub trait SendFn<Args: Tuple + Send + 'static, Output> =
        Fn<Args, Output = Output> + Send + 'static;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use crate::allow_threading::AllowThreads;
    use extend::ext;
    use pyo3::exceptions::{PyConnectionError, PyRuntimeError};
    use pyo3::marker::Ungil;
    use pyo3::types::PyBytes;
    use pyo3::{Py, PyErr, PyResult, Python};
    use tokio::runtime::Runtime;
    use tokio::sync::mpsc;
    use tokio::sync::mpsc::error::TryRecvError;
    use tokio::task::JoinHandle;

    #[ext(pub, name = ByteArrayExt)]
    impl [u8] {
        fn pybytes(&self) -> Py<PyBytes> {
            Python::with_gil(|py| PyBytes::new(py, self).unbind())
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
            Python::with_gil(|py| self.write_unraisable_with(py))
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

pub(crate) mod private {
    use std::marker::Sized;

    /// Sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

/// A wrapper around [`Py`] that implements [`Clone`] using [`Python::with_gil`].
#[repr(transparent)]
pub(crate) struct ClonePy<T>(pub Py<T>);

impl<T> Clone for ClonePy<T> {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self(self.0.clone_ref(py)))
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "exo_pyo3_bindings")]
fn main_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // install logger
    pyo3_log::init();

    // TODO: for now this is all NOT a submodule, but figure out how to make the submodule system
    //       work with maturin, where the types generate correctly, in the right folder, without
    //       too many importing issues...
    ident_submodule(m)?;
    multiaddr_submodule(m)?;
    networking_submodule(m)?;

    // top-level constructs
    // TODO: ...

    Ok(())
}

define_stub_info_gatherer!(stub_info);
