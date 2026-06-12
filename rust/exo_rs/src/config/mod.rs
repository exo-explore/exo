use crate::config::app::app_submodule;
use crate::config::bootstrap::bootstrap_submodule;
use crate::config::cli::cli_submodule;
use clap::ValueEnum;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{pyclass, Bound, PyResult};
use pyo3_stub_gen::derive::gen_stub_pyclass_enum;
use serde::{Deserialize, Serialize};

pub mod app;
pub mod bootstrap;
pub mod cli;
pub mod path;

/// Verbosity level used by EXO's logger.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, ord, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum VerbosityFilter {
    Off = 0,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! {
    r#"
    class VerbosityFilter:
      def __lt__(self, other: object, /) -> bool: ...
      def __le__(self, other: object, /) -> bool: ...
      def __gt__(self, other: object, /) -> bool: ...
      def __ge__(self, other: object, /) -> bool: ...
    "#
    }
}

pub fn config_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    bootstrap_submodule(m)?;
    app_submodule(m)?;
    cli_submodule(m)?;

    m.add_class::<VerbosityFilter>()?;

    Ok(())
}
