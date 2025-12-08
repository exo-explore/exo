use crate::ext::{ByteArrayExt as _, FutureExt as _, ResultExt as _};
use crate::identity::{PyEndpointId, PyKeypair};
use iroh::SecretKey;
use iroh::discovery::EndpointInfo;
use iroh::discovery::mdns::DiscoveryEvent;
use iroh_gossip::api::{ApiError, Event, GossipReceiver, GossipSender, Message};
use n0_future::{Stream, StreamExt as _};
use networking::ExoNet;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::pin::{Pin, pin};
use std::sync::{Arc, LazyLock};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

#[allow(clippy::expect_used)]
static RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Failed to create tokio runtime"));

#[gen_stub_pyclass]
#[pyclass(name = "IpAddress")]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PyIpAddress {
    inner: SocketAddr,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyIpAddress {
    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }

    pub fn ip_addr(&self) -> String {
        self.inner.ip().to_string()
    }

    pub const fn port(&self) -> u16 {
        self.inner.port()
    }

    pub const fn zone_id(&self) -> Option<u32> {
        match self.inner {
            SocketAddr::V6(ip) => Some(ip.scope_id()),
            SocketAddr::V4(_) => None,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RustNetworkingHandle")]
pub struct PyNetworkingHandle {
    net: Arc<ExoNet>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNetworkingHandle {
    #[staticmethod]
    pub async fn create(identity: PyKeypair, namespace: String) -> PyResult<Self> {
        let loc: SecretKey = identity.0.clone();
        let net = Arc::new(
            RUNTIME
                .spawn(async move { ExoNet::init_iroh(loc, &namespace).await })
                .await
                // todo: pyerr better
                .pyerr()?
                .pyerr()?,
        );
        let cloned = Arc::clone(&net);
        RUNTIME.spawn(async move { cloned.start_auto_dialer().await });

        Ok(Self { net })
    }

    async fn subscribe(&self, topic: String) -> PyResult<(PySender, PyReceiver)> {
        let fut = self.net.subscribe(&topic);
        let (send, recv) = pin!(fut).allow_threads_py().await.pyerr()?;
        Ok((PySender { inner: send }, PyReceiver { inner: recv }))
    }

    async fn get_connection_receiver(&self) -> PyResult<PyConnectionReceiver> {
        let fut = self.net.connection_info();
        let stream = pin!(fut).allow_threads_py().await;
        Ok(PyConnectionReceiver {
            inner: Mutex::new(Box::pin(stream)),
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RustConnectionMessage")]
pub struct PyConnectionMessage {
    #[pyo3(get)]
    pub endpoint_id: PyEndpointId,
    #[pyo3(get)]
    pub current_transport_addrs: Option<BTreeSet<PyIpAddress>>,
}

#[gen_stub_pyclass]
#[pyclass(name = "RustSender")]
struct PySender {
    inner: GossipSender,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySender {
    async fn send(&mut self, message: Py<PyBytes>) -> PyResult<()> {
        let bytes = Python::attach(|py| message.as_bytes(py).to_vec());
        let broadcast_fut = self.inner.broadcast(bytes.into());
        pin!(broadcast_fut).await.pyerr()
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RustReceiver")]
struct PyReceiver {
    inner: GossipReceiver,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReceiver {
    async fn receive(&mut self) -> PyResult<Py<PyBytes>> {
        loop {
            let next_fut = self.inner.next();
            match pin!(next_fut).allow_threads_py().await {
                // Successful cases
                Some(Ok(Event::Received(Message { content, .. }))) => {
                    return Ok(content.to_vec().pybytes());
                }
                Some(Ok(other)) => log::info!("Dropping gossip event {other:?}"),
                None => return Err(PyStopAsyncIteration::new_err("")),
                Some(Err(ApiError::Closed { .. })) => {
                    return Err(PyStopAsyncIteration::new_err(""));
                }

                // Failure case
                Some(Err(other)) => {
                    return Err(PyRuntimeError::new_err(other.to_string()));
                }
            }
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "RustConnectionReceiver")]
struct PyConnectionReceiver {
    inner: Mutex<Pin<Box<dyn Stream<Item = DiscoveryEvent> + Send>>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConnectionReceiver {
    async fn receive(&mut self) -> PyResult<PyConnectionMessage> {
        let mg_fut = self.inner.lock();
        let mut lock = pin!(mg_fut).allow_threads_py().await;
        match lock.next().allow_threads_py().await {
            // Successful cases
            Some(DiscoveryEvent::Discovered {
                endpoint_info: EndpointInfo { endpoint_id, data },
                ..
            }) => Ok(PyConnectionMessage {
                endpoint_id: endpoint_id.into(),
                current_transport_addrs: Some(
                    data.ip_addrs()
                        .map(|inner| PyIpAddress { inner: *inner })
                        .collect(),
                ),
            }),
            Some(DiscoveryEvent::Expired { endpoint_id }) => Ok(PyConnectionMessage {
                endpoint_id: endpoint_id.into(),
                current_transport_addrs: None,
            }),
            // Failure case
            None => Err(PyStopAsyncIteration::new_err("")),
        }
    }
}

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnectionMessage>()?;
    m.add_class::<PyReceiver>()?;
    m.add_class::<PySender>()?;
    m.add_class::<PyConnectionReceiver>()?;
    m.add_class::<PyNetworkingHandle>()?;

    Ok(())
}
