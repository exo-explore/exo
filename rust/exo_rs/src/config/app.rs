use crate::config::bootstrap::BootstrapSettings;
use crate::config::cli::CliArgs;
use crate::config::{VerbosityFilter, default};
use crate::ext::ResultExt;
use crate::pickle_reduce;
use clap::{
    ArgAction,
    builder::{BoolishValueParser, TypedValueParser},
};
use figment::Figment;
use figment::providers::{Format, Serialized, Toml};
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::types::PyTuple;
use pyo3::{Bound, PyAny, PyResult, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// Arguments that participate in application settings resolution.
///
/// These values may come from defaults, `config.toml`, environment variables, or
/// CLI arguments. Unlike [`BootstrapArgs`](crate::config::bootstrap::BootstrapArgs),
/// they do not participate in finding or loading `config.toml`.
///
/// # Important
///  - Make sure all fields are [`Option<T>`] so they can be layered with other
///    settings sources.
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct AppArgs {
    #[arg(
        short = 'q',
        long = "quiet",
        action = ArgAction::SetTrue,
        conflicts_with = "verbosity",
        help = "Only show error logs (alias for --verbosity=error)"
    )]
    #[serde(skip)]
    pub verbosity_off: bool,
    #[arg(
        short = 'v',
        long,
        env = "EXO_VERBOSITY",
        value_enum,
        default_value_if("verbosity_off", "true", Some("error")),
        value_name = "LEVEL",
        conflicts_with = "verbosity_off",
        help = "Verbosity filter of the application"
    )]
    #[pyo3(get, set)]
    pub verbosity: Option<VerbosityFilter>,

    // this parser cannot use the default boolean parser + ArgAction::SetFalse
    // since it needs to logically invert --no-batch and EXO_NO_BATCH
    #[arg(
      long = "no-batch",
      env = "EXO_NO_BATCH",
      num_args = 0..=1,
      require_equals = true,
      default_missing_value = "true",
      value_parser = BoolishValueParser::new().map(|no_batch| !no_batch),
      value_name = "BOOL",
      help = "Disable continuous batching, use sequential generation"
    )]
    #[pyo3(get, set)]
    pub continuous_batching_enabled: Option<bool>,

    #[arg(
        long,
        env = "EXO_MAX_CONCURRENT_REQUESTS",
        value_parser = clap::value_parser!(u16).range(1..),
        value_name = "NUM",
        help = "Maximum number of concurrent generation requests per runner"
    )]
    #[pyo3(get, set)]
    pub max_concurrent_requests: Option<u16>,

    #[arg(
        long,
        env = "EXO_OFFLINE",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help = "Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models"
    )]
    #[pyo3(get, set)]
    pub offline: Option<bool>,

    #[arg(
        long = "enable-image-models",
        env = "EXO_IMAGE_MODELS_ENABLED",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help = "Enable image model support"
    )]
    #[pyo3(get, set)]
    pub image_models_enabled: Option<bool>,

    #[arg(
        long = "enable-tracing",
        env = "EXO_TRACING_ENABLED",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help = "Enable distributed tracing for performance analysis"
    )]
    #[pyo3(get, set)]
    pub tracing_enabled: Option<bool>,

    #[arg(
        long = "enable-disaggregation",
        env = "EXO_DISAGGREGATION_ENABLED",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help = "Enable prefill/decode disaggregation"
    )]
    #[pyo3(get, set)]
    pub disaggregation_enabled: Option<bool>,

    #[arg(
        long,
        env = "EXO_FAST_SYNCH",
        value_name = "BOOL",
        help = "Force MLX FAST_SYNCH on/off (for JACCL backend)"
    )]
    #[pyo3(get, set)]
    pub fast_synch: Option<bool>,
}

#[gen_stub_pyclass]
#[pyclass(module = "exo_rs", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppSettings {
    #[pyo3(get, set)]
    pub verbosity: VerbosityFilter,
    #[pyo3(get, set)]
    pub continuous_batching_enabled: bool,
    #[pyo3(get, set)]
    pub max_concurrent_requests: u16,
    #[pyo3(get, set)]
    pub offline: bool,
    #[pyo3(get, set)]
    pub image_models_enabled: bool,
    #[pyo3(get, set)]
    pub tracing_enabled: bool,
    #[pyo3(get, set)]
    pub disaggregation_enabled: bool,
    #[pyo3(get, set)]
    pub fast_synch: Option<bool>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AppSettings {
    /// Create default instance.
    #[staticmethod]
    #[pyo3(name = "default")]
    pub fn py_default() -> PyResult<Self> {
        let bootstrap = BootstrapSettings::py_default()?;
        let args = AppArgs::default();
        Self::resolve(&args, &bootstrap)
    }

    /// Create only from environment variables.
    #[staticmethod]
    pub fn from_env_only() -> PyResult<Self> {
        let args = CliArgs::from_env_only();
        let bootstrap = BootstrapSettings::resolve(&args.bootstrap)?;
        Self::resolve(&args.app, &bootstrap)
    }

    #[staticmethod]
    pub fn resolve(args: &AppArgs, bootstrap: &BootstrapSettings) -> PyResult<Self> {
        Figment::new()
            // merge default CLI values
            .merge(Serialized::defaults(default::APP_ARGS))
            // merge configuration file
            .merge(Toml::file(&bootstrap.config_file))
            // merge CLI args (with ENV already merged)
            .merge(Serialized::defaults(args.clone()))
            .extract::<Self>()
            .pyerr()
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

pub fn app_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<AppArgs>()?;
    m.add_class::<AppSettings>()?;

    Ok(())
}
