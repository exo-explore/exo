use crate::config::VerbosityFilter;
use crate::config::cli::CliArgs;
use crate::ext::ResultExt;
use crate::pickle_reduce;
use clap::{
    ArgAction,
    builder::{BoolishValueParser, TypedValueParser},
};
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::types::PyTuple;
use pyo3::{Bound, PyAny, PyResult, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};

/// TODO: once config.toml is integrated, we should still figure out a way to display
///       what would be "default" values here (very informative to user) ((same for BootstrapArgs))
///       or what would be ENV variables that are read - but without those having any effect
///       on what value this one takes.
///       i.e. user sees "Description \[env: FOO_BAR=] \[default: 12345]"
///       and yet those defaults are not ACTUALLY parsed by this CLI and instead consumed
///       later on by the settings merger
///
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
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
    verbosity_off: bool,
    #[arg(
        short = 'v',
        long,
        env = "EXO_VERBOSITY",
        value_enum,
        default_value = "info", // TODO: when config.toml introduced, remove this
        default_value_if("verbosity_off", "true", Some("error")),
        value_name = "LEVEL",
        conflicts_with = "verbosity_off",
        help = "Set the verbosity filter"
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
      default_value = "false", // TODO: when config.toml introduced, remove this
      value_parser = BoolishValueParser::new().map(|no_batch| !no_batch),
      value_name = "BOOL",
      help = "Disable continuous batching, use sequential generation"
    )]
    #[pyo3(get, set)]
    pub continuous_batching_enabled: Option<bool>,

    #[arg(
        long,
        env = "EXO_OFFLINE",
        action = ArgAction::SetTrue,
        default_value = "false", // TODO: when config.toml introduced, remove this
        help = "Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models"
    )]
    #[pyo3(get, set)]
    pub offline: Option<bool>,

    #[arg(
        long = "enable-image-models",
        env = "EXO_IMAGE_MODELS_ENABLED",
        action = ArgAction::SetTrue,
        default_value = "false", // TODO: when config.toml introduced, remove this
        help = "Enable image model support"
    )]
    #[pyo3(get, set)]
    pub image_models_enabled: Option<bool>,

    #[arg(
        long = "enable-tracing",
        env = "EXO_TRACING_ENABLED",
        action = ArgAction::SetTrue,
        default_value = "false", // TODO: when config.toml introduced, remove this
        help = "Enable distributed tracing for performance analysis"
    )]
    #[pyo3(get, set)]
    pub tracing_enabled: Option<bool>,

    #[arg(
        long,
        env = "EXO_FAST_SYNCH",
        value_name = "BOOL",
        help = "Force MLX FAST_SYNCH on/off (for JACCL backend); omit for auto"
    )]
    #[pyo3(get, set)]
    pub fast_synch: Option<bool>,
}

impl Default for AppArgs {
    fn default() -> Self {
        Self {
            // verbosity
            verbosity_off: false,
            verbosity: Some(VerbosityFilter::Info),

            // rest
            continuous_batching_enabled: Some(true),
            offline: Some(false),
            image_models_enabled: Some(false),
            tracing_enabled: Some(false),
            fast_synch: None,
        }
    }
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
    pub offline: bool,
    #[pyo3(get, set)]
    pub image_models_enabled: bool,
    #[pyo3(get, set)]
    pub tracing_enabled: bool,
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
        Self::resolve(&AppArgs::default())
    }

    /// Create only from environment variables.
    #[staticmethod]
    pub fn from_env_only() -> PyResult<Self> {
        Self::resolve(&CliArgs::from_env_only().app)
    }

    #[staticmethod]
    pub fn resolve(args: &AppArgs) -> PyResult<Self> {
        Ok(Self {
            verbosity: args.verbosity.unwrap_or(VerbosityFilter::Info),
            continuous_batching_enabled: args.continuous_batching_enabled.unwrap_or(true),
            offline: args.offline.unwrap_or(false),
            image_models_enabled: args.image_models_enabled.unwrap_or(false),
            tracing_enabled: args.tracing_enabled.unwrap_or(false),
            fast_synch: args.fast_synch,
        })
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
