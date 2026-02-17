use crate::r#const::MPSC_CHANNEL_SIZE;
use crate::ext::ResultExt as _;
use crate::ext::{ByteArrayExt as _, FutureExt as _};
use crate::ident::PyKeypair;
use crate::networking::exception::{PyAllQueuesFullError, PyNoPeersSubscribedToTopicError};
use crate::pyclass;
use futures_lite::FutureExt as _;
use networking::swarm::{FromSwarm, Swarm, ToSwarm};
use pyo3::coroutine::CancelHandle;
use pyo3::exceptions::{PyConnectionError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_async_runtimes::tokio::get_runtime;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods};
use std::pin::pin;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

mod exception {
    use pyo3::types::PyTuple;
    use pyo3::{exceptions::PyException, prelude::*};
    use pyo3_stub_gen::derive::*;

    #[gen_stub_pyclass]
    #[pyclass(frozen, extends=PyException, name="NoPeersSubscribedToTopicError")]
    pub struct PyNoPeersSubscribedToTopicError {}

    impl PyNoPeersSubscribedToTopicError {
        const MSG: &'static str = "\
        No peers are currently subscribed to receive messages on this topic. \
        Wait for peers to subscribe or check your network connectivity.";

        ///   Creates a new  [ `PyErr` ]  of this type.
        ///
        ///   [`PyErr`] :  https://docs.rs/pyo3/latest/pyo3/struct.PyErr.html   "PyErr in pyo3"
        pub(crate) fn new_err() -> PyErr {
            PyErr::new::<Self, _>(()) // TODO: check if this needs to be replaced???
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyNoPeersSubscribedToTopicError {
        #[new]
        #[pyo3(signature = (*_a))]
        pub(crate) fn new(_a: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(frozen, extends=PyException, name="AllQueuesFullError")]
    pub struct PyAllQueuesFullError {}

    impl PyAllQueuesFullError {
        const MSG: &'static str =
            "All libp2p peers are unresponsive, resend the message or reconnect.";

        ///   Creates a new  [ `PyErr` ]  of this type.
        ///
        ///   [`PyErr`] :  https://docs.rs/pyo3/latest/pyo3/struct.PyErr.html   "PyErr in pyo3"
        pub(crate) fn new_err() -> PyErr {
            PyErr::new::<Self, _>(()) // TODO: check if this needs to be replaced???
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyAllQueuesFullError {
        #[new]
        #[pyo3(signature = (*_a))]
        pub(crate) fn new(_a: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }
}

#[gen_stub_pyclass]
#[pyclass]
struct PySwarm {
    swarm: Arc<Mutex<Swarm>>,
    from_swarm: Mutex<mpsc::Receiver<FromSwarm>>,
    to_swarm: Mutex<mpsc::Sender<ToSwarm>>,
}

#[gen_stub_pyclass_complex_enum]
#[pyclass]
pub enum PyMessage {
    Connection {
        node_id: String,
        connected: bool,
    },
    Gossip {
        node_id: String,
        topic: String,
        data: Py<PyBytes>,
    },
}
impl TryFrom<FromSwarm> for PyMessage {
    type Error = PyErr;
    fn try_from(value: FromSwarm) -> Result<Self, Self::Error> {
        match value {
            FromSwarm::Discovered(nid) => Ok(PyMessage::Connection {
                node_id: nid.to_base58(),
                connected: true,
            }),
            FromSwarm::Expired(nid) => Ok(PyMessage::Connection {
                node_id: nid.to_base58(),
                connected: false,
            }),
            FromSwarm::Message(nid, topic, data) => Ok(PyMessage::Gossip {
                node_id: nid.to_base58(),
                topic,
                data: data.pybytes(),
            }),
            FromSwarm::PublishError(e) => match e {
                libp2p::gossipsub::PublishError::NoPeersSubscribedToTopic => {
                    Err(PyNoPeersSubscribedToTopicError::new_err())
                }
                libp2p::gossipsub::PublishError::AllQueuesFull(_) => {
                    Err(PyAllQueuesFullError::new_err())
                }
                e => Err(PyRuntimeError::new_err(e.to_string())),
            },
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySwarm {
    #[new]
    fn py_new(identity: Bound<'_, PyKeypair>) -> PyResult<Self> {
        use pyo3_async_runtimes::tokio::get_runtime;

        // get identity
        let identity = identity.borrow().0.clone();

        let (to_swarm, from_client) = mpsc::channel(MPSC_CHANNEL_SIZE);
        let (to_client, from_swarm) = mpsc::channel(MPSC_CHANNEL_SIZE);
        // create networking swarm (within tokio context!! or it crashes)
        let swarm = get_runtime()
            .block_on(async { Swarm::new(identity, from_client, to_client) })
            .pyerr()?;

        Ok(Self {
            swarm: Arc::new(Mutex::new(swarm)),
            from_swarm: Mutex::new(from_swarm),
            to_swarm: Mutex::new(to_swarm),
        })
    }

    #[gen_stub(skip)]
    async fn run(&self, #[pyo3(cancel_handle)] mut cancel: CancelHandle) -> PyResult<()> {
        let copy = Arc::clone(&self.swarm);
        let jh = get_runtime().spawn(async move {
            copy.try_lock()
                .expect("tried to run swarm twice")
                .run()
                .await
        });
        jh.or(async {
            cancel.cancelled().await;
            Ok(())
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    // ---- Connection update receiver methods ----

    /// Receives the next message from networking.
    async fn recv(&self) -> PyResult<PyMessage> {
        let msg = pin!(
            self.from_swarm
                .try_lock()
                .expect("called recv concurrently")
                .recv()
        )
        .allow_threads_py()
        .await;
        match msg {
            None => Err(PyConnectionError::new_err("swarm closed")),
            Some(msg) => msg.try_into(),
        }
    }

    /// Subscribe to a `GossipSub` topic.
    async fn gossipsub_subscribe(&self, topic: String) -> PyResult<()> {
        // send off request to subscribe
        pin!(
            self.to_swarm
                .try_lock()
                .expect("called send concurrently")
                .send(ToSwarm::Subscribe(topic))
        )
        .allow_threads_py() // allow-threads-aware async call
        .await
        .map_err(|_| PyConnectionError::new_err("swarm closed"))
    }

    /// Unsubscribes from a `GossipSub` topic.
    ///
    /// Returns `True` if we were subscribed to this topic. Returns `False` if we were not subscribed.
    async fn gossipsub_unsubscribe(&self, topic: String) -> PyResult<()> {
        // send off request to unsubscribe
        pin!(
            self.to_swarm
                .try_lock()
                .expect("called send concurrently")
                .send(ToSwarm::Unsubscribe(topic))
        )
        .allow_threads_py() // allow-threads-aware async call
        .await
        .map_err(|_| PyConnectionError::new_err("swarm closed"))
    }

    /// Publishes a message to the network on a specific topic.
    async fn gossipsub_publish(&self, topic: String, data: Py<PyBytes>) -> PyResult<()> {
        // send off request to subscribe
        let data = Python::attach(|py| Vec::from(data.as_bytes(py)));
        pin!(
            self.to_swarm
                .try_lock()
                .expect("called send concurrently")
                .send(ToSwarm::Message(topic, data))
        )
        .allow_threads_py() // allow-threads-aware async call
        .await
        .map_err(|_| PyConnectionError::new_err("swarm closed"))
    }
}

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<exception::PyNoPeersSubscribedToTopicError>()?;
    m.add_class::<exception::PyAllQueuesFullError>()?;

    m.add_class::<PySwarm>()?;
    m.add_class::<PyMessage>()?;

    Ok(())
}
