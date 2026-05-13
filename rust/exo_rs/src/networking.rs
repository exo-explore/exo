use std::pin::Pin;
use std::sync::Arc;

use crate::ext::{ByteArrayExt as _, FutureExt, PyErrExt as _};
use crate::ext::{ResultExt as _, TokioMpscSenderExt as _};
use futures_lite::{Stream, StreamExt as _};
use networking::swarm::{FromSwarm, ToSwarm, create_swarm};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pymethods};
use pyo3_stub_gen::derive::{
    gen_methods_from_python, gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods,
};
use tokio::sync::{Mutex, mpsc, oneshot};

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

#[gen_stub_pymethods]
#[pymethods]
impl PyNetworkingHandle {
    // NOTE: `async fn`s here that use `.await` will wrap the future in `.allow_threads_py()`
    //       immediately beforehand to release the interpreter.
    //       SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await

    // ---- Lifecycle management methods ----

    #[staticmethod]
    fn new<'py>(
        identity: Bound<'py, PyBytes>,
        bootstrap_peers: Vec<String>,
        listen_port: u16,
    ) -> PyResult<PyNetworkingHandle> {
        // create communication channels
        let (to_swarm, from_client) = mpsc::channel(1024);

        // get identity
        let identity = u128::from_le_bytes(
            identity
                .extract::<'_, Vec<u8>>()?
                .try_into()
                .map_err(|_| PyValueError::new_err("invalid identity bytes"))?,
        );

        // create networking swarm (within tokio context!! or it crashes)
        let swarm = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(create_swarm(
                identity,
                from_client,
                bootstrap_peers,
                listen_port,
            ))
            .pyerr()?;

        Ok(PyNetworkingHandle {
            swarm: Arc::new(Mutex::new(swarm.into_stream())),
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
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
    m.add_class::<PyNetworkingHandle>()?;
    m.add_class::<PyFromSwarm>()?;

    Ok(())
}
