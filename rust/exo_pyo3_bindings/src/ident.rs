use crate::ext::ResultExt as _;
use libp2p::identity::Keypair;
use pyo3::prelude::{PyBytesMethods as _, PyModule, PyModuleMethods as _};
use pyo3::types::PyBytes;
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

    /// Decode a private key from a protobuf structure and parse it as a `Keypair`.
    #[staticmethod]
    fn deserialize(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::from_protobuf_encoding(&bytes).pyerr()?))
    }

    /// Encode a private key as protobuf structure.
    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.0.to_protobuf_encoding().pyerr()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Convert the `Keypair` into the corresponding `PeerId`.
    fn to_string(&self) -> String {
        self.0.public().to_peer_id().to_base58()
    }
}

pub fn ident_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeypair>()?;

    Ok(())
}
