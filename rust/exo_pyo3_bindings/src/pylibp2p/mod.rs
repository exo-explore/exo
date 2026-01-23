//! A module for exposing Rust's libp2p datatypes over Pyo3
//!
//! TODO: right now we are coupled to libp2p's identity, but eventually we want to create our own
//!       independent identity type of some kind or another. This may require handshaking.
//!

pub mod ident;
pub mod multiaddr;

use std::sync::Mutex;

use cluster_membership::Peer;
use libp2p::identity::ed25519::Keypair;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct PyKeypair(Keypair);

#[gen_stub_pymethods]
#[pymethods]
impl PyKeypair {
    #[staticmethod]
    fn generate() -> Self {
        Self(Keypair::generate())
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct PyPeer(Mutex<Peer>);

#[gen_stub_pymethods]
#[pymethods]
impl PyPeer {
    #[staticmethod]
    fn init(kp: PyKeypair, namespace: String) -> PyResult<Self> {
        Ok(PyPeer(Mutex::new(
            Peer::new(kp.0.secret(), namespace)
                .map_err(|e| e.pyerr())?
                .0,
        )))
    }
}
