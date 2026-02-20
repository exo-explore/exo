use crate::ext::ResultExt as _;
use libp2p::identity::Keypair;
use pyo3::types::{PyBytes, PyBytesMethods as _};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Identity keypair of a node.
#[gen_stub_pyclass]
#[pyclass(name = "Keypair", frozen)]
#[repr(transparent)]
pub struct PyKeypair(pub Keypair);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyKeypair {
    /// Generate a new Ed25519 keypair.
    #[staticmethod]
    fn generate() -> Self {
        Self(Keypair::generate_ed25519())
    }

    /// Construct an Ed25519 keypair from secret key bytes
    #[staticmethod]
    fn from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::ed25519_from_bytes(&mut bytes).pyerr()?))
    }

    /// Get the secret key bytes underlying the keypair
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .0
            .clone()
            .try_into_ed25519()
            .pyerr()?
            .secret()
            .as_ref()
            .to_vec();
        Ok(PyBytes::new(py, &bytes))
    }

    /// Convert the `Keypair` into the corresponding `PeerId` string, which we use as our `NodeId`.
    fn to_node_id(&self) -> String {
        self.0.public().to_peer_id().to_base58()
    }
}
