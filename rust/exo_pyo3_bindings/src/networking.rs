use crate::allow_threading::AllowThreads;
use crate::take_once::TakeOnce;

use std::pin::pin;

use futures_lite::FutureExt;
use libp2p::{gossipsub::PublishError, identity::Keypair};
use networking::{FromSwarm, Peer, ToSwarm};
use pyo3::{
    coroutine::CancelHandle,
    exceptions::{PyConnectionError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use pyo3_stub_gen::{
    derive::{gen_methods_from_python, gen_stub_pyclass, gen_stub_pymethods},
    inventory::submit,
};
use tokio::sync::{Mutex, mpsc};

#[gen_stub_pyclass]
#[pyclass(name = "Keypair", frozen)]
#[derive(Clone)]
pub struct PyKeypair(Keypair);

#[gen_stub_pymethods]
#[pymethods]
impl PyKeypair {
    /// Generate a new ed25519 keypair
    #[staticmethod]
    fn generate() -> Self {
        Self(Keypair::generate_ed25519())
    }

    /// Decode a private key from a protobuf structure and parse it as a `Keypair`.
    #[staticmethod]
    fn from_protobuf_encoding(bytes: &Bound<'_, PyBytes>) -> Self {
        let bytes = Vec::from(bytes.as_bytes());
        Self(Keypair::from_protobuf_encoding(&bytes).expect("todo"))
    }

    /// Encode a private key to a protobuf structure.
    fn to_protobuf_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match self.0.to_protobuf_encoding() {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes)),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    fn to_string(&self) -> String {
        self.0.public().to_peer_id().to_base58()
    }
}

struct PeerBuilder(
    String,
    Keypair,
    mpsc::Sender<FromSwarm>,
    mpsc::Receiver<ToSwarm>,
);

#[gen_stub_pyclass]
#[pyclass]
pub struct PyPeer {
    peer: TakeOnce<PeerBuilder>,
    to_swarm: mpsc::Sender<ToSwarm>,
    from_swarm: Mutex<mpsc::Receiver<FromSwarm>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPeer {
    #[staticmethod]
    fn new(kp: PyKeypair, namespace: String) -> PyResult<Self> {
        let (to_client, from_swarm) = mpsc::channel(1024);
        let (to_swarm, from_client) = mpsc::channel(1024);
        Ok(Self {
            peer: TakeOnce::new(PeerBuilder(namespace, kp.0, to_client, from_client)),
            to_swarm,
            from_swarm: Mutex::new(from_swarm),
        })
    }

    #[gen_stub(skip)]
    async fn run(&self, #[pyo3(cancel_handle)] mut cancel: CancelHandle) -> PyResult<()> {
        let builder = self
            .peer
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("tried to run peer twice"))?;
        let jh = pyo3_async_runtimes::tokio::get_runtime()
            .spawn(async move {
                let mut peer =
                    Peer::new(builder.0, builder.1, builder.2, builder.3).map_err(|_| {
                        PyConnectionError::new_err("peer failed to listen on default address")
                    })?;
                peer.run()
                    .await
                    .map_err(|()| PyConnectionError::new_err("peer communication closed"))
            })
            .or(async {
                cancel.cancelled().await;
                Ok(Ok(()))
            });
        match AllowThreads(pin!(jh)).await {
            Err(e) if e.is_cancelled() => Ok(()),
            Err(e) if e.is_panic() => Err(PyRuntimeError::new_err(format!("tokio panic {e}"))),
            Err(_) => unreachable!(),
            Ok(res) => res,
        }
    }

    async fn subscribe(&self, topic: String) -> PyResult<()> {
        self.to_swarm
            .send(ToSwarm::Subscribe(topic))
            .await
            .map_err(|_| PyRuntimeError::new_err("swarm communication closed"))
    }
    async fn unsubscribe(&self, topic: String) -> PyResult<()> {
        self.to_swarm
            .send(ToSwarm::Unsubscribe(topic))
            .await
            .map_err(|_| PyRuntimeError::new_err("swarm communication closed"))
    }
    async fn send(&self, topic: String, payload: Py<PyBytes>) -> PyResult<()> {
        // this function attaches to the python interpreter synchronously to avoid holding the GIL
        let bytes = Python::attach(|py| Vec::from(payload.bind(py).as_bytes()));
        self.to_swarm
            .send(ToSwarm::Message(topic, bytes))
            .await
            .map_err(|_| PyRuntimeError::new_err("swarm communication closed"))
    }

    #[gen_stub(skip)]
    async fn recv(
        &self,
        #[pyo3(cancel_handle)] mut cancel: CancelHandle,
    ) -> PyResult<PySwarmEvent> {
        loop {
            return match AllowThreads(pin!(
                self.from_swarm
                    .try_lock()
                    .map_err(|_| PyRuntimeError::new_err("tried to recv twice"))?
                    .recv()
                    .or(async {
                        cancel.cancelled().await;
                        None
                    })
            ))
            .await
            {
                Some(FromSwarm::PublishError(p)) => match p {
                    PublishError::AllQueuesFull(_) => {
                        Err(PyConnectionError::new_err("swarm overloaded"))
                    }
                    PublishError::MessageTooLarge => {
                        Err(PyValueError::new_err("message too large"))
                    }
                    PublishError::NoPeersSubscribedToTopic => {
                        continue;
                    }
                    // TODO(evan): logs here
                    _ => continue,
                },
                None => Err(PyRuntimeError::new_err("swarm communication closed")),
                Some(fs) => Ok(PySwarmEvent(fs)),
            };
        }
    }
}

// Manually submit the run()/recv() stub because the cancelhandle is poorly understood
submit! {
    gen_methods_from_python! {
        r#"
        class PyPeer:
            async def run(self): ...
            async def recv(self) -> PySwarmEvent: ...
        "#
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct PySwarmEvent(FromSwarm);

#[gen_stub_pymethods]
#[pymethods]
impl PySwarmEvent {
    // probably a better way to do this, but...
    fn downcast_discovered(&self) -> Option<String> {
        if let FromSwarm::Discovered(peer_id) = self.0 {
            Some(peer_id.to_base58())
        } else {
            None
        }
    }
    fn downcast_expired(&self) -> Option<String> {
        if let FromSwarm::Expired(peer_id) = self.0 {
            Some(peer_id.to_base58())
        } else {
            None
        }
    }
    fn downcast_message<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(String, String, Bound<'py, PyBytes>)> {
        if let FromSwarm::Message(peer_id, topic, data) = &self.0 {
            Some((peer_id.to_base58(), topic.clone(), PyBytes::new(py, data)))
        } else {
            None
        }
    }
}
