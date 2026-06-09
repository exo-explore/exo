use crate::config::cli::{CliArgs, ConfigArgs, DeprecatedArgs, Verbosity};
use crate::config::locator::{LocatorArgs, LocatorConfig};
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult};

pub mod cli;
pub mod defaults;
pub mod locator;
pub mod path;

pub fn config_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Verbosity>()?;
    m.add_class::<CliArgs>()?;
    m.add_class::<LocatorArgs>()?;
    m.add_class::<ConfigArgs>()?;
    m.add_class::<DeprecatedArgs>()?;
    m.add_class::<LocatorConfig>()?;

    Ok(())
}
