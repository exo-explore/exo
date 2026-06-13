use crate::config::app::AppArgs;
use crate::config::bootstrap::BootstrapArgs;
use crate::ext::ResultExt;
use crate::{pickle_reduce, version};
use clap::{ArgAction, Parser};
use pyo3::prelude::{PyAnyMethods, PyModuleMethods};
use pyo3::types::{PyModule, PyTuple};
use pyo3::{Bound, PyAny, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;

// re-export
pub use parser_impl::*;

#[gen_stub_pyclass]
#[pyclass(module = "exo_rs", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Parser)]
#[command(name = "EXO", version = version::version(), about, long_about = None)]
pub struct CliArgs {
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
        help = "Run as a legacy SysV-style background daemon using double-fork daemonization"
    )]
    #[pyo3(get, set)]
    pub legacy_daemon: bool,

    #[arg(
        long,
        env = "EXO_NAMESPACE",
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

    // -------- FLATTENED SUBCOMMANDS --------
    #[command(flatten)]
    #[pyo3(get)]
    pub bootstrap: BootstrapArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub app: AppArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub rejected: RejectedArgs,
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

    pub fn set_rejected(&mut self, rejected: RejectedArgs) {
        self.rejected = rejected;
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

/// Rejected arguments go here.
///
/// # Important
///  - Make sure all are `hide = true` so it won't appear in `--help`
///  - Make sure all are [`Option<T>`] so them being missing doesn't cause issues
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct RejectedArgs {
    // -------- temporarily unavailable --------
    #[arg(
        long,
        env = "EXO_BOOTSTRAP_PEERS",
        value_delimiter = ',',
        value_name = "MULTIADDRS",
        help = "Comma-separated libp2p multiaddrs to dial on startup",
        hide = true,
        value_parser = Rejected::<String>::unavailable(
            Some("--bootstrap-peers"), None, Some("EXO_BOOTSTRAP_PEERS"),
            "bootstrap peers are temporarily removed",
        )
    )]
    #[pyo3(get, set)]
    pub bootstrap_peers: Option<Vec<String>>,

    // -------- deprecated --------
    #[arg(
        long, value_name = "PORT", hide = true,
        value_parser = Rejected::<u16>::deprecated(
            Some("--libp2p-port"), None, None,
            Some("--zenoh-port"), None, None,
        )
    )]
    #[pyo3(get, set)]
    pub libp2p_port: Option<u16>,

    #[arg(
        env = "EXO_LIBP2P_NAMESPACE", value_name = "STRING", hide = true,
        value_parser = Rejected::<String>::deprecated(
            None, None, Some("EXO_LIBP2P_NAMESPACE"),
            Some("--namespace"), None, Some("EXO_NAMESPACE"),
        )
    )]
    #[pyo3(get, set)]
    pub libp2p_namespace: Option<String>,

    #[arg(
        env = "EXO_ZENOH_NAMESPACE", value_name = "STRING", hide = true,
        value_parser = Rejected::<String>::deprecated(
            None, None, Some("EXO_ZENOH_NAMESPACE"),
            Some("--namespace"), None, Some("EXO_NAMESPACE"),
        )
    )]
    #[pyo3(get, set)]
    pub zenoh_namespace: Option<String>,

    #[arg(
        env = "EXO_ENABLE_IMAGE_MODELS", value_name = "BOOL", hide = true,
        value_parser = Rejected::<bool>::deprecated(
            None, None, Some("EXO_ENABLE_IMAGE_MODELS"),
            Some("--enable-image-models"), None, Some("EXO_IMAGE_MODELS_ENABLED"),
        )
    )]
    #[pyo3(get, set)]
    pub enable_image_models: Option<bool>,

    #[arg(
        env = "ENABLE_DISAGGREGATION", value_name = "BOOL", hide = true,
        value_parser = Rejected::<bool>::deprecated(
            None, None, Some("ENABLE_DISAGGREGATION"),
            Some("--enable-disaggregation"), None, Some("EXO_DISAGGREGATION_ENABLED"),
        )
    )]
    #[pyo3(get, set)]
    pub enable_disaggregation: Option<bool>,

    #[arg(
        long = "no-fast-synch", hide = true,
        num_args = 0..=1, default_missing_value = "true",
        value_parser = Rejected::<bool>::deprecated(
            Some("--no-fast-synch"), None, None,
            Some("--fast-synch=false"), None, None,
        )
    )]
    #[pyo3(get, set)]
    pub no_fast_synch: Option<bool>,

    #[arg(
        long = "verbose", hide = true,
        num_args = 0..=1, default_missing_value = "true",
        value_parser = Rejected::<bool>::deprecated(
            Some("--verbose"), None, None,
            Some("--verbosity=debug"), None, Some("EXO_VERBOSITY=debug"),
        )
    )]
    #[pyo3(get, set)]
    pub verbose: Option<bool>,
}

mod parser_impl {
    use clap::builder::TypedValueParser;
    use itertools::Itertools;
    use std::ffi::OsStr;
    use std::marker::PhantomData;

    #[derive(Clone)]
    pub struct Rejected<T> {
        message: String,
        _ty: PhantomData<T>,
    }

    impl<T> Rejected<T> {
        #[inline(always)]
        pub fn new(message: impl Into<String>) -> Self {
            let mut message = message.into();
            if !message.ends_with('\n') {
                message.push('\n');
            }
            Self {
                message,
                _ty: PhantomData,
            }
        }

        #[inline(always)]
        pub fn deprecated(
            old_long: Option<&str>,
            old_short: Option<&str>,
            old_env: Option<&str>,
            new_long: Option<&str>,
            new_short: Option<&str>,
            new_env: Option<&str>,
        ) -> Self {
            let old_names = vec![old_short, old_long, old_env]
                .into_iter()
                .flatten()
                .join("/");
            let new_names = vec![new_short, new_long, new_env]
                .into_iter()
                .flatten()
                .join("/");
            Self::new(format!(
                "the argument {old_names} is deprecated{}",
                if new_names.is_empty() {
                    String::new()
                } else {
                    format!("; use {new_names} instead")
                }
            ))
        }

        #[inline(always)]
        pub fn unavailable(
            long: Option<&str>,
            short: Option<&str>,
            env: Option<&str>,
            reason: impl AsRef<str>,
        ) -> Self {
            let names = vec![short, long, env].into_iter().flatten().join("/");
            Self::new(format!(
                "the argument {names} is unavailable: {}",
                reason.as_ref()
            ))
        }
    }

    impl<T> TypedValueParser for Rejected<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        type Value = T;
        fn parse_ref(
            &self,
            cmd: &clap::Command,
            _arg: Option<&clap::Arg>,
            _value: &OsStr,
        ) -> Result<Self::Value, clap::Error> {
            Err(clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                self.message.clone(),
            )
            .with_cmd(cmd))
        }
    }
}

pub fn cli_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CliArgs>()?;
    m.add_class::<RejectedArgs>()?;

    Ok(())
}
