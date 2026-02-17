//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

mod allow_threading;
mod ident;
mod networking;

use crate::ident::ident_submodule;
use crate::networking::networking_submodule;
use pyo3::prelude::PyModule;
use pyo3::{Bound, PyResult, pyclass, pymodule};
use pyo3_stub_gen::define_stub_info_gatherer;

/// Namespace for all the constants used by this crate.
pub(crate) mod r#const {
    pub const MPSC_CHANNEL_SIZE: usize = 1024;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use crate::allow_threading::AllowThreads;
    use extend::ext;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyBytes;
    use pyo3::{Py, PyResult, Python};

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
        /// SEE: https://pyo3.rs/v0.27.2/async-await.html#detaching-from-the-interpreter-across-await
        fn allow_threads_py(self) -> AllowThreads<Self>
        where
            AllowThreads<Self>: Future,
        {
            AllowThreads(self)
        }
    }

    impl<T: Future> FutureExt for T {}
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
