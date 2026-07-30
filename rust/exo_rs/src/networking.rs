use std::pin::Pin;
use std::sync::Arc;

use crate::ext::{ByteArrayExt as _, FutureExt, PyErrExt as _};
use crate::ext::{ResultExt as _, TokioMpscSenderExt as _};
use futures_lite::{Stream, StreamExt as _};
use networking::swarm::{FromSwarm, Swarm, ToSwarm, create_swarm};
use networking::{Session, is_valid_zid};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods};
use tokio::sync::{Mutex, mpsc, oneshot};

#[gen_stub_pyclass]
#[pyclass(name = "NetworkingHandle")]
pub struct PyNetworkingHandle {
    // channels
    pub to_swarm: mpsc::Sender<ToSwarm>,
    pub swarm: Arc<Mutex<Pin<Box<dyn Stream<Item = FromSwarm> + Send>>>>,
}

#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "FromSwarm")]
pub enum PyFromSwarm {
    Connection { connected: bool },
    Message { topic: String, data: Py<PyBytes> },
}
impl From<FromSwarm> for PyFromSwarm {
    fn from(value: FromSwarm) -> Self {
        match value {
            FromSwarm::Discovered {} => Self::Connection { connected: true },
            FromSwarm::Expired {} => Self::Connection { connected: false },
            FromSwarm::Message { topic, data } => Self::Message {
                topic: topic,
                data: data.pybytes(),
            },
        }
    }
}

impl PyNetworkingHandle {
    pub fn from_session(session: Session) -> Self {
        let (to_swarm, from_client) = mpsc::channel(1024);
        let swarm = Swarm {
            from_client,
            session,
        };
        PyNetworkingHandle {
            swarm: Arc::new(Mutex::new(swarm.into_stream())),
            to_swarm,
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

    #[staticmethod]
    pub fn new(
        identity: &str,
        namespace: &str,
        listen_port: u16,
        discovery_service_port: u16,
    ) -> PyResult<PyNetworkingHandle> {
        // todo: zenoh self assigned peers
        if listen_port == 0 {
            todo!("cannot listen on port 0 yet");
        }
        // create communication channels
        let (to_swarm, from_client) = mpsc::channel(1024);

        // get identity
        if !is_valid_zid(identity) {
            return Err(PyValueError::new_err(format!(
                "{identity} is not a valid zenoh identity"
            )));
        }

        // create networking swarm (within tokio context!! or it crashes)
        let swarm = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(create_swarm(
                identity,
                namespace,
                from_client,
                listen_port,
                discovery_service_port,
            ))
            .pyerr()?;

        Ok(PyNetworkingHandle {
            swarm: Arc::new(Mutex::new(swarm.into_stream())),
            to_swarm,
        })
    }

    #[gen_stub(override_return_type(
        type_repr="typing.Awaitable[FromSwarm]", imports=("typing")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    pub async fn gossipsub_subscribe(&self, topic: String) -> PyResult<bool> {
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
    pub async fn gossipsub_unsubscribe(&self, topic: String) -> PyResult<bool> {
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
    pub async fn gossipsub_publish(&self, topic: String, data: Py<PyBytes>) -> PyResult<()> {
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
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNetworkingHandle>()?;
    m.add_class::<PyFromSwarm>()?;

    Ok(())
}
