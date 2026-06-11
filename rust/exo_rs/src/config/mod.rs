use crate::config::bootstrap::bootstrap_submodule;
use crate::config::cli::cli_submodule;
use pyo3::prelude::PyModule;
use pyo3::{Bound, PyResult};

pub mod bootstrap;
pub mod cli;
pub mod path;

pub fn config_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    bootstrap_submodule(m)?;
    cli_submodule(m)?;

    Ok(())
}
