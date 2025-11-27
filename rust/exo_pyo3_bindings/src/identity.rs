use iroh::{EndpointId, SecretKey, endpoint_info::EndpointIdExt};
use postcard::ser_flavors::StdVec;

use crate::ext::ResultExt as _;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rand::rng;

#[gen_stub_pyclass]
#[pyclass(name = "Keypair", frozen)]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyKeypair(pub(crate) SecretKey);

#[gen_stub_pymethods]
#[pymethods]
impl PyKeypair {
    /// Generate a new Ed25519 keypair.
    #[staticmethod]
    fn generate_ed25519() -> Self {
        Self(SecretKey::generate(&mut rng()))
    }
    /// Decode a postcard structure into a keypair
    #[staticmethod]
    fn from_postcard_encoding(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(postcard::from_bytes(&bytes).pyerr()?))
    }
    /// Encode a private key with the postcard format
    fn to_postcard_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = postcard::serialize_with_flavor(&self.0, StdVec::new()).pyerr()?;
        Ok(PyBytes::new(py, &bytes))
    }
    /// Read out the endpoint id corresponding to this keypair
    fn endpoint_id(&self) -> PyEndpointId {
        PyEndpointId(self.0.public())
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "EndpointId", frozen)]
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct PyEndpointId(pub(crate) EndpointId);

#[gen_stub_pymethods]
#[pymethods]
impl PyEndpointId {
    pub fn __str__(&self) -> String {
        self.0.to_z32()
    }
}

impl From<EndpointId> for PyEndpointId {
    fn from(value: EndpointId) -> Self {
        Self(value)
    }
}

pub fn ident_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeypair>()?;
    m.add_class::<PyEndpointId>()?;

    Ok(())
}
