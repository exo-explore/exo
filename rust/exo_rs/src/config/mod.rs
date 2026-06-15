use crate::config::app::app_submodule;
use crate::config::bootstrap::bootstrap_submodule;
use crate::config::cli::cli_submodule;
use clap::ValueEnum;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, pyclass};
use pyo3_stub_gen::derive::gen_stub_pyclass_enum;
use serde::{Deserialize, Serialize};

pub mod app;
pub mod bootstrap;
pub mod cli;

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

#[allow(nonstandard_style)]
pub mod default {
    use crate::config::VerbosityFilter;
    use crate::config::app::AppArgs;
    use crate::version;
    // ---- unclassified defaults (belonging to top-level CLI) ----

    /// Port on which the API runs
    pub const API_PORT: u16 = 52415;

    /// Discovery namespace, nodes with different namespaces will not connect.
    pub fn NAMESPACE() -> String {
        version::version().to_string()
    }

    /// Fixed TCP port for zenoh to listen
    pub const ZENOH_PORT: u16 = 52414;

    /// Fixed UDP port for the discovery service
    pub const DISCOVERY_PORT: u16 = 52413;

    /// Default [`AppArgs`] values
    pub const APP_ARGS: AppArgs = AppArgs {
        // verbosity
        verbosity_off: false,
        verbosity: Some(VerbosityFilter::Info),

        // rest
        continuous_batching_enabled: Some(true),
        max_concurrent_requests: Some(8),
        offline: Some(false),
        image_models_enabled: Some(false),
        tracing_enabled: Some(false),
        disaggregation_enabled: Some(false),
        fast_synch: None,
    };
}

pub fn config_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    bootstrap_submodule(m)?;
    app_submodule(m)?;
    cli_submodule(m)?;

    m.add_class::<VerbosityFilter>()?;

    Ok(())
}
