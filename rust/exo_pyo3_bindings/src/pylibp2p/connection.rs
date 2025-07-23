use libp2p::swarm::ConnectionId;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{pyclass, pymethods, Bound, PyResult};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// TODO: documentation...
#[gen_stub_pyclass]
#[pyclass(name = "ConnectionId")]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyConnectionId(pub ConnectionId);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyConnectionId {
    /// TODO: documentation
    #[staticmethod]
    fn new_unchecked(id: usize) -> Self {
        Self(ConnectionId::new_unchecked(id))
    }

    fn __repr__(&self) -> String {
        format!("ConnectionId({})", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

pub fn connection_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnectionId>()?;

    Ok(())
}
