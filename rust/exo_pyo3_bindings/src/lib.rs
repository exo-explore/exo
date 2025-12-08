//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

mod allow_threading;
mod identity;
mod iroh_networking;
// mod examples;

use crate::identity::ident_submodule;
use crate::iroh_networking::networking_submodule;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use crate::allow_threading::AllowThreads;
    use extend::ext;
    use pyo3::exceptions::{PyConnectionError, PyRuntimeError};
    use pyo3::types::PyBytes;
    use pyo3::{Py, PyErr, PyResult, Python};

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
        /// SEE: <https://pyo3.rs/v0.27.1/async-await.html#detaching-from-the-interpreter-across-await>
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
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "exo_pyo3_bindings")]
fn main_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // install logger
    pyo3_log::init();

    ident_submodule(m)?;
    networking_submodule(m)?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
