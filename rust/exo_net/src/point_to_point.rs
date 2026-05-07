use std::sync::Arc;

use pyo3::exceptions::PyConnectionError;
use pyo3::types::PyBytes;
use pyo3::types::PyNone;
use pyo3::{BoundObject, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::Result;
use zenoh::{
    handlers::FifoChannelHandler,
    pubsub::{Publisher, Subscriber},
    sample::Sample,
};

use crate::ext::ByteArrayExt;

#[gen_stub_pyclass]
#[pyclass]
pub struct NetReceiver {
    pub subscriber: Subscriber<FifoChannelHandler<Sample>>,
}
#[gen_stub_pymethods]
#[pymethods]
impl NetReceiver {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[bytes | None]",
        imports=("collections.abc")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, {
            assert!(
                self.subscriber.receiver_count() == 1,
                "tried to receive twice on the same receiver"
            );
            let subscriber = self.subscriber.clone();
            async move {
                match subscriber.recv_async().await {
                    Err(_) => {
                        // stream closed;
                        Ok(Python::attach(|py| PyNone::get(py).unbind()).into_any())
                    }
                    Ok(sample) => Ok(sample.payload().to_bytes().to_vec().pybytes().into_any()),
                }
            }
        })
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct NetSender {
    pub publisher: Arc<Publisher<'static>>,
    pub first: bool,
}
#[gen_stub_pymethods]
#[pymethods]
impl NetSender {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[bool]",
        imports=("collections.abc")
    ))]
    pub fn send<'py>(
        &'py mut self,
        py: Python<'py>,
        data: Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let is_first = self.first;
        self.first = false;
        pyo3_async_runtimes::tokio::future_into_py(py, {
            let publisher = Arc::clone(&self.publisher);
            // clone the data so py can have it back
            let bytes = data.as_bytes().to_vec();
            async move {
                if is_first {
                    wait_for_listener(&*publisher)
                        .await
                        .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
                }
                if !publisher
                    .matching_status()
                    .await
                    .map_err(|e| PyConnectionError::new_err(e.to_string()))?
                    .matching()
                {
                    return Ok(false);
                }
                publisher
                    .put(&bytes)
                    .await
                    .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
                Ok(true)
            }
        })
    }
}

async fn wait_for_listener<'a>(publisher: &Publisher<'a>) -> Result<()> {
    let matcher = publisher.matching_listener().await?;
    if publisher.matching_status().await?.matching() {
        return Ok(());
    }
    while let Ok(status) = matcher.recv_async().await {
        if status.matching() {
            break;
        }
    }
    Ok(())
}
