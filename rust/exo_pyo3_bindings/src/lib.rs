//! TODO: crate documentation
pub(crate) mod allow_threading;

pub(crate) mod networking;
pub(crate) mod take_once {
    use std::sync::Mutex;

    pub struct TakeOnce<T>(Mutex<Option<T>>);
    impl<T> TakeOnce<T> {
        pub fn new(t: T) -> Self {
            Self(Mutex::new(Some(t)))
        }
        pub fn take(&self) -> Option<T> {
            match self.0.try_lock() {
                Ok(mut o) => o.take(),
                Err(_) => None,
            }
        }
    }
}

use pyo3::prelude::*;

use pyo3_stub_gen::define_stub_info_gatherer;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "exo_pyo3_bindings")]
pub fn networking_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // install logger
    pyo3_log::init();
    // setup runtime
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_async_runtimes::tokio::init(builder);

    m.add_class::<networking::PyPeer>()?;
    m.add_class::<networking::PyKeypair>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
