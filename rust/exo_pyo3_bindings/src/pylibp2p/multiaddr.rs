use libp2p::Multiaddr;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// TODO: documentation...
#[gen_stub_pyclass]
#[pyclass(name = "Multiaddr")]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyMultiaddr(pub Multiaddr);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyMultiaddr {
    /// TODO: documentation
    #[staticmethod]
    fn empty() -> Self {
        Self(Multiaddr::empty())
    }

    /// TODO: documentation
    #[staticmethod]
    fn with_capacity(n: usize) -> Self {
        Self(Multiaddr::with_capacity(n))
    }

    /// TODO: documentation
    fn len(&self) -> usize {
        self.0.len()
    }

    /// TODO: documentation
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// TODO: documentation
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self.0.to_vec();
        PyBytes::new(py, &bytes)
    }

    fn __repr__(&self) -> String {
        format!("Multiaddr({})", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

pub fn multiaddr_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiaddr>()?;

    Ok(())
}
