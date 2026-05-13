use crate::ext::ResultExt as _;
use pyo3::types::{PyBytes, PyBytesMethods as _};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Identity keypair of a node.
#[gen_stub_pyclass]
#[pyclass(name = "Keypair", frozen)]
#[repr(transparent)]
pub struct PyKeypair(pub u128);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyKeypair {
    /// Generate a new Ed25519 keypair.
    #[staticmethod]
    fn generate() -> Self {
        Self(rand::random())
    }

    /// Construct an Ed25519 keypair from secret key bytes
    #[staticmethod]
    fn from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(u128::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| "passed too many bytes to from_bytes")
                .pyerr()?,
        )))
    }

    /// Get the secret key bytes underlying the keypair
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.0.to_le_bytes();
        Ok(PyBytes::new(py, &bytes))
    }

    /// Convert the `Keypair` into the corresponding `PeerId` string, which we use as our `NodeId`.
    fn to_node_id(&self) -> String {
        format!("{:x}", self.0)
    }
}
