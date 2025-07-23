use crate::ext::ResultExt;
use libp2p::identity::{ecdsa, Keypair};
use libp2p::PeerId;
use pyo3::prelude::{PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// TODO: documentation...
#[gen_stub_pyclass]
#[pyclass(name = "Keypair")]
#[repr(transparent)]
pub struct PyKeypair(pub Keypair);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyKeypair {
    /// TODO: documentation
    #[staticmethod]
    fn generate_ed25519() -> Self {
        Self(Keypair::generate_ed25519())
    }

    /// TODO: documentation
    #[staticmethod]
    fn generate_ecdsa() -> Self {
        Self(Keypair::generate_ecdsa())
    }

    /// TODO: documentation
    #[staticmethod]
    fn generate_secp256k1() -> Self {
        Self(Keypair::generate_secp256k1())
    }

    /// TODO: documentation
    #[staticmethod]
    fn from_protobuf_encoding(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::from_protobuf_encoding(&bytes).pyerr()?))
    }

    /// TODO: documentation
    #[staticmethod]
    fn rsa_from_pkcs8(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::rsa_from_pkcs8(&mut bytes).pyerr()?))
    }

    /// TODO: documentation
    #[staticmethod]
    fn secp256k1_from_der(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::secp256k1_from_der(&mut bytes).pyerr()?))
    }

    /// TODO: documentation
    #[staticmethod]
    fn ed25519_from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::ed25519_from_bytes(&mut bytes).pyerr()?))
    }

    /// TODO: documentation
    #[staticmethod]
    fn ecdsa_from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::from(ecdsa::Keypair::from(
            ecdsa::SecretKey::try_from_bytes(bytes).pyerr()?,
        ))))
    }

    /// TODO: documentation
    fn to_protobuf_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.0.to_protobuf_encoding().pyerr()?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// TODO: documentation...
#[gen_stub_pyclass]
#[pyclass(name = "PeerId")]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyPeerId(pub PeerId);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyPeerId {
    /// TODO: documentation
    #[staticmethod]
    fn random() -> Self {
        Self(PeerId::random())
    }

    /// TODO: documentation
    #[staticmethod]
    fn from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(PeerId::from_bytes(&bytes).pyerr()?))
    }

    /// TODO: documentation
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self.0.to_bytes();
        PyBytes::new(py, &bytes)
    }

    /// TODO: documentation
    fn to_base58(&self) -> String {
        self.0.to_base58()
    }

    fn __repr__(&self) -> String {
        format!("PeerId({})", self.to_base58())
    }

    fn __str__(&self) -> String {
        self.to_base58()
    }
}

pub fn ident_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeypair>()?;
    m.add_class::<PyPeerId>()?;

    Ok(())
}
