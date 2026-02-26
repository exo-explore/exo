use std::pin::Pin;
use std::sync::Arc;

use crate::r#const::MPSC_CHANNEL_SIZE;
use crate::ext::{ByteArrayExt as _, FutureExt, PyErrExt as _};
use crate::ext::{ResultExt as _, TokioMpscSenderExt as _};
use crate::ident::PyKeypair;
use crate::networking::exception::{
    PyAllQueuesFullError, PyMessageTooLargeError, PyNoPeersSubscribedToTopicError,
};
use crate::pyclass;
use futures_lite::{Stream, StreamExt as _};
use libp2p::gossipsub::PublishError;
use networking::swarm::{FromSwarm, ToSwarm, create_swarm};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{PyModule, PyModuleMethods as _};
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pymethods};
use pyo3_stub_gen::derive::{
    gen_methods_from_python, gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods,
};
use tokio::sync::{Mutex, mpsc, oneshot};

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
        #[pyo3(signature = (*args))]
        #[allow(unused_variables)]
        pub(crate) fn new(args: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __repr__(&self) -> String {
            format!("PeerId(\"{}\")", Self::MSG)
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
        #[pyo3(signature = (*args))]
        #[allow(unused_variables)]
        pub(crate) fn new(args: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __repr__(&self) -> String {
            format!("PeerId(\"{}\")", Self::MSG)
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(frozen, extends=PyException, name="MessageTooLargeError")]
    pub struct PyMessageTooLargeError {}

    impl PyMessageTooLargeError {
        const MSG: &'static str = "Gossipsub message exceeds max_transmit_size. Reduce prompt length or increase the limit.";

        pub(crate) fn new_err() -> PyErr {
            PyErr::new::<Self, _>(())
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyMessageTooLargeError {
        #[new]
        #[pyo3(signature = (*args))]
        #[allow(unused_variables)]
        pub(crate) fn new(args: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __repr__(&self) -> String {
            format!("MessageTooLargeError(\"{}\")", Self::MSG)
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "NetworkingHandle")]
struct PyNetworkingHandle {
    // channels
    pub to_swarm: mpsc::Sender<ToSwarm>,
    pub swarm: Arc<Mutex<Pin<Box<dyn Stream<Item = FromSwarm> + Send>>>>,
}

#[gen_stub_pyclass_complex_enum]
#[pyclass]
enum PyFromSwarm {
    Connection {
        peer_id: String,
        connected: bool,
    },
    Message {
        origin: String,
        topic: String,
        data: Py<PyBytes>,
    },
}
impl From<FromSwarm> for PyFromSwarm {
    fn from(value: FromSwarm) -> Self {
        match value {
            FromSwarm::Discovered { peer_id } => Self::Connection {
                peer_id: peer_id.to_base58(),
                connected: true,
            },
            FromSwarm::Expired { peer_id } => Self::Connection {
                peer_id: peer_id.to_base58(),
                connected: false,
            },
            FromSwarm::Message { from, topic, data } => Self::Message {
                origin: from.to_base58(),
                topic: topic,
                data: data.pybytes(),
            },
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNetworkingHandle {
    // NOTE: `async fn`s here that use `.await` will wrap the future in `.allow_threads_py()`
    //       immediately beforehand to release the interpreter.
    //       SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await

    // ---- Lifecycle management methods ----

    #[new]
    fn py_new(identity: Bound<'_, PyKeypair>) -> PyResult<Self> {
        // create communication channels
        let (to_swarm, from_client) = mpsc::channel(MPSC_CHANNEL_SIZE);

        // get identity
        let identity = identity.borrow().0.clone();

        // create networking swarm (within tokio context!! or it crashes)
        let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        let swarm = create_swarm(identity, from_client).pyerr()?.into_stream();

        Ok(Self {
            swarm: Arc::new(Mutex::new(swarm)),
            to_swarm,
        })
    }

    #[gen_stub(skip)]
    fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let swarm = Arc::clone(&self.swarm);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            swarm
                .try_lock()
                .map_err(|_| PyRuntimeError::new_err("called recv twice concurrently"))?
                .next()
                .await
                .ok_or(PyErr::receiver_channel_closed())
                .map(PyFromSwarm::from)
        })
    }

    // ---- Gossipsub management methods ----

    /// Subscribe to a `GossipSub` topic.
    ///
    /// Returns `True` if the subscription worked. Returns `False` if we were already subscribed.
    async fn gossipsub_subscribe(&self, topic: String) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        self.to_swarm
            .send_py(ToSwarm::Subscribe {
                topic,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors
        rx.allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())?
            .pyerr()
    }

    /// Unsubscribes from a `GossipSub` topic.
    ///
    /// Returns `True` if we were subscribed to this topic. Returns `False` if we were not subscribed.
    async fn gossipsub_unsubscribe(&self, topic: String) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to unsubscribe
        self.to_swarm
            .send_py(ToSwarm::Unsubscribe {
                topic,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & convert any errors
        rx.allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())
    }

    /// Publishes a message with multiple topics to the `GossipSub` network.
    ///
    /// If no peers are found that subscribe to this topic, throws `NoPeersSubscribedToTopicError` exception.
    async fn gossipsub_publish(&self, topic: String, data: Py<PyBytes>) -> PyResult<()> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        let data = Python::attach(|py| Vec::from(data.as_bytes(py)));
        self.to_swarm
            .send_py(ToSwarm::Publish {
                topic,
                data,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors => ignore messageID for now!!!
        let _ = rx
            .allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())?
            .map_err(|e| match e {
                PublishError::AllQueuesFull(_) => PyAllQueuesFullError::new_err(),
                PublishError::MessageTooLarge => PyMessageTooLargeError::new_err(),
                PublishError::NoPeersSubscribedToTopic => {
                    PyNoPeersSubscribedToTopicError::new_err()
                }
                e => PyRuntimeError::new_err(e.to_string()),
            })?;
        Ok(())
    }
}

pyo3_stub_gen::inventory::submit! {
    gen_methods_from_python! {
        r#"
            class PyNetworkingHandle:
                async def recv() -> PyFromSwarm: ...
        "#
    }
}

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<exception::PyNoPeersSubscribedToTopicError>()?;
    m.add_class::<exception::PyAllQueuesFullError>()?;
    m.add_class::<exception::PyMessageTooLargeError>()?;

    m.add_class::<PyNetworkingHandle>()?;
    m.add_class::<PyFromSwarm>()?;

    Ok(())
}
