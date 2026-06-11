use crate::config::app::AppArgs;
use crate::config::bootstrap::BootstrapArgs;
use crate::ext::ResultExt;
use crate::{pickle_reduce, version};
use clap::{ArgAction, Parser, ValueEnum};
use pyo3::prelude::{PyAnyMethods, PyModuleMethods};
use pyo3::types::{PyModule, PyTuple};
use pyo3::{Bound, PyAny, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[gen_stub_pyclass]
#[pyclass(module = "exo_rs", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Parser)]
#[command(name = "EXO", version = version::version(), about, long_about = None)]
pub struct CliArgs {
    #[arg(
          short = 'v',
          long,
          value_enum,
          default_value_t = Verbosity::Info,
          value_name = "LEVEL",
          help = "Set the verbosity level"
    )]
    #[pyo3(get, set)]
    pub verbosity: Verbosity,

    #[arg(
        short = 'm',
        long,
        action = ArgAction::SetTrue,
        help = "Force node to be master"
    )]
    #[pyo3(get, set)]
    pub force_master: bool,

    #[arg(
        long = "no-api",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the API"
    )]
    #[pyo3(get, set)]
    pub api_enabled: bool,

    #[arg(
        long,
        default_value_t = 52415,
        value_name = "PORT",
        help = "Port on which the API runs"
    )]
    #[pyo3(get, set)]
    pub api_port: u16,

    #[arg(
        long = "no-worker",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the worker"
    )]
    #[pyo3(get, set)]
    pub worker_enabled: bool,

    #[arg(
        long = "no-downloads",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the download coordinator (node won't download models)"
    )]
    #[pyo3(get, set)]
    pub downloads_enabled: bool,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models"
    )]
    #[pyo3(get, set)]
    pub offline: bool,

    #[arg(
        long = "no-batch",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable continuous batching, use sequential generation"
    )]
    #[pyo3(get, set)]
    pub continuous_batching_enabled: bool,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Run as a legacy SysV-style background daemon using double-fork daemonization"
    )]
    #[pyo3(get, set)]
    pub legacy_daemon: bool,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "MULTIADDRS",
        help = "Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)"
    )]
    #[pyo3(get, set)]
    pub bootstrap_peers: Option<Vec<String>>,

    #[arg(
        long,
        default_value_t = version::version().to_string(),
        value_name = "STRING",
        help = "Discovery namespace, nodes with different namespaces will not connect."
    )]
    #[pyo3(get, set)]
    pub namespace: String,

    #[arg(
        long,
        default_value_t = 52414,
        value_name = "PORT",
        help = "Fixed TCP port for zenoh to listen."
    )]
    #[pyo3(get, set)]
    pub zenoh_port: u16,

    #[arg(
        long,
        default_value_t = 52413,
        value_name = "PORT",
        help = "Fixed UDP port for the discovery service."
    )]
    #[pyo3(get, set)]
    pub discovery_port: u16,

    #[command(flatten)]
    #[pyo3(get)]
    pub bootstrap: BootstrapArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub app: AppArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub deprecated: DeprecatedArgs,
}

#[gen_stub_pymethods]
#[pymethods]
impl CliArgs {
    /// Create only from env-variables
    #[staticmethod]
    pub fn from_env_only() -> Self {
        // parse only from env - no arguments
        CliArgs::parse_from(&["exo"])
    }

    #[staticmethod]
    #[pyo3(name = "parse_from")]
    pub fn py_parse_from(argv: Vec<OsString>) -> Self {
        CliArgs::parse_from(argv)
    }

    #[staticmethod]
    #[pyo3(name = "parse")]
    pub fn py_parse(py: Python<'_>) -> PyResult<Self> {
        // the correct CLI args to parse is `sys.argv`, because the original ones
        // (i.e. `sys.orig_argv`) may contain extra arguments which would mess up parsing
        let argv: Vec<OsString> = PyModule::import(py, "sys")?.getattr("argv")?.extract()?;
        Ok(CliArgs::parse_from(argv))
    }

    pub fn set_bootstrap(&mut self, bootstrap: BootstrapArgs) {
        self.bootstrap = bootstrap;
    }

    pub fn set_app(&mut self, app: AppArgs) {
        self.app = app;
    }

    pub fn set_deprecated(&mut self, deprecated: DeprecatedArgs) {
        self.deprecated = deprecated;
    }

    // -------- SERDE/PICKLING support --------

    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        postcard::to_allocvec(self).pyerr()
    }

    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        postcard::from_bytes(&bytes).pyerr()
    }

    pub fn __reduce__(slf: Bound<'_, Self>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyTuple>)> {
        pickle_reduce(slf, "from_bytes", Self::to_bytes)
    }
}

/// Deprecated arguments go here.
///
/// # Important
///  - Make sure all are `hide = true` so it won't appear in `--help`
///  - Make sure all are [`Option<T>`] so them being missing doesn't cause issues
///  - Edit [`Self::get_error`] to handle changes of new/removed args in here
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct DeprecatedArgs {
    #[arg(long = "libp2p-port", hide = true)]
    #[pyo3(get, set)]
    pub libp2p_port: Option<u16>,
}

impl DeprecatedArgs {
    // TODO: actually run these at some point - maybe automatically..?
    pub fn get_error(&self) -> Option<clap::Error> {
        // destructure: don't change because this becomes compile error when new options are
        // moved into here or removed from here
        let Self { libp2p_port } = self.clone();

        if let Some(_) = libp2p_port {
            Some(clap::Error::raw(
                clap::error::ErrorKind::UnknownArgument,
                "The argument --libp2p-port is deprecated; use --zenoh-port instead",
            ))
        }
        // add more options here
        else {
            None
        }
    }
}

pub fn cli_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Verbosity>()?;
    m.add_class::<CliArgs>()?;
    m.add_class::<DeprecatedArgs>()?;

    Ok(())
}
