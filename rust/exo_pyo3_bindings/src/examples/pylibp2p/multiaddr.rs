use crate::ext::ResultExt as _;
use libp2p::Multiaddr;
use pyo3::prelude::{PyBytesMethods as _, PyModule, PyModuleMethods as _};
use pyo3::types::PyBytes;
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::str::FromStr as _;

/// Representation of a Multiaddr.
#[gen_stub_pyclass]
#[pyclass(name = "Multiaddr", frozen)]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyMultiaddr(pub Multiaddr);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyMultiaddr {
    /// Create a new, empty multiaddress.
    #[staticmethod]
    fn empty() -> Self {
        Self(Multiaddr::empty())
    }

    /// Create a new, empty multiaddress with the given capacity.
    #[staticmethod]
    fn with_capacity(n: usize) -> Self {
        Self(Multiaddr::with_capacity(n))
    }

    /// Parse a `Multiaddr` value from its byte slice representation.
    #[staticmethod]
    fn from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Multiaddr::try_from(bytes).pyerr()?))
    }

    /// Parse a `Multiaddr` value from its string representation.
    #[staticmethod]
    fn from_string(string: String) -> PyResult<Self> {
        Ok(Self(Multiaddr::from_str(&string).pyerr()?))
    }

    /// Return the length in bytes of this multiaddress.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the length of this multiaddress is 0.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return a copy of this [`Multiaddr`]'s byte representation.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self.0.to_vec();
        PyBytes::new(py, &bytes)
    }

    /// Convert a Multiaddr to a string.
    fn to_string(&self) -> String {
        self.0.to_string()
    }

    #[gen_stub(skip)]
    fn __repr__(&self) -> String {
        format!("Multiaddr({})", self.0)
    }

    #[gen_stub(skip)]
    fn __str__(&self) -> String {
        self.to_string()
    }
}

pub fn multiaddr_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiaddr>()?;

    Ok(())
}
