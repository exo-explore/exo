use crate::config::cli::CliArgs;
use crate::ext::ResultExt;
use crate::pickle_reduce;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::types::PyTuple;
use pyo3::{Bound, PyAny, PyResult, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct AppArgs {
    #[arg(
        long,
        env = "EXO_FAST_SYNCH",
        value_name = "BOOL",
        help = "Force MLX FAST_SYNCH on/off (for JACCL backend); omit for auto"
    )]
    #[pyo3(get, set)]
    pub fast_synch: Option<bool>,
}

#[gen_stub_pyclass]
#[pyclass(module = "exo_rs", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppSettings {
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
