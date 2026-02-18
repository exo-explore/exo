use crate::ext::ResultExt as _;
use libp2p::PeerId;
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
    fn generate_ed25519() -> Self {
        Self(Keypair::generate_ed25519())
    }

    /// Generate a new ECDSA keypair.
    #[staticmethod]
    fn generate_ecdsa() -> Self {
        Self(Keypair::generate_ecdsa())
    }

    /// Generate a new Secp256k1 keypair.
    #[staticmethod]
    fn generate_secp256k1() -> Self {
        Self(Keypair::generate_secp256k1())
    }

    /// Decode a private key from a protobuf structure and parse it as a `Keypair`.
    #[staticmethod]
    fn from_protobuf_encoding(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::from_protobuf_encoding(&bytes).pyerr()?))
    }

    /// Decode an keypair from a DER-encoded secret key in PKCS#8 `PrivateKeyInfo`
    /// format (i.e. unencrypted) as defined in [RFC5208].
    ///
    /// [RFC5208]: https://tools.ietf.org/html/rfc5208#section-5
    #[staticmethod]
    fn rsa_from_pkcs8(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::rsa_from_pkcs8(&mut bytes).pyerr()?))
    }

    /// Decode a keypair from a DER-encoded Secp256k1 secret key in an `ECPrivateKey`
    /// structure as defined in [RFC5915].
    ///
    /// [RFC5915]: https://tools.ietf.org/html/rfc5915
    #[staticmethod]
    fn secp256k1_from_der(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::secp256k1_from_der(&mut bytes).pyerr()?))
    }

    #[staticmethod]
    fn ed25519_from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let mut bytes = Vec::from(bytes.as_bytes());
        Ok(Self(Keypair::ed25519_from_bytes(&mut bytes).pyerr()?))
    }

    /// Encode a private key as protobuf structure.
    fn to_protobuf_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.0.to_protobuf_encoding().pyerr()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Convert the `Keypair` into the corresponding `PeerId`.
    fn to_peer_id(&self) -> PyPeerId {
        PyPeerId(self.0.public().to_peer_id())
    }

    // /// Hidden constructor for pickling support. TODO: figure out how to do pickling...
    // #[gen_stub(skip)]
    // #[new]
    // fn py_new(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
    //     Self::from_protobuf_encoding(bytes)
    // }
    //
    // #[gen_stub(skip)]
    // fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
    //     *self = Self::from_protobuf_encoding(state)?;
    //     Ok(())
    // }
    //
    // #[gen_stub(skip)]
    // fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    //     self.to_protobuf_encoding(py)
    // }
    //
    // #[gen_stub(skip)]
    // pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>,)> {
    //     Ok((self.to_protobuf_encoding(py)?,))
    // }
}

/// Identifier of a peer of the network.
///
/// The data is a `CIDv0` compatible multihash of the protobuf encoded public key of the peer
/// as specified in [specs/peer-ids](https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md).
#[gen_stub_pyclass]
#[pyclass(name = "PeerId", frozen)]
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct PyPeerId(pub PeerId);

#[gen_stub_pymethods]
#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyPeerId {
    /// Generates a random peer ID from a cryptographically secure PRNG.
    ///
    /// This is useful for randomly walking on a DHT, or for testing purposes.
    #[staticmethod]
    fn random() -> Self {
        Self(PeerId::random())
    }

    /// Parses a `PeerId` from bytes.
    #[staticmethod]
    fn from_bytes(bytes: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = Vec::from(bytes.as_bytes());
        Ok(Self(PeerId::from_bytes(&bytes).pyerr()?))
    }

    /// Returns a raw bytes representation of this `PeerId`.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self.0.to_bytes();
        PyBytes::new(py, &bytes)
    }

    /// Returns a base-58 encoded string of this `PeerId`.
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
