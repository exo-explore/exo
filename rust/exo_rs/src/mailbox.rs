use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::pubsub::Subscriber;
use zenoh::{
    handlers::FifoChannelHandler,
    sample::{Sample, SampleKind},
};

use crate::sample_to_string;

#[gen_stub_pyclass]
#[pyclass]
pub struct Mailbox {
    pub subscriber: Subscriber<FifoChannelHandler<Sample>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Mailbox {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[str | None]",
        imports=("collections.abc")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, {
            if self.subscriber.receiver_count() != 1 {
                return Err(PyRuntimeError::new_err(
                    "tried to receive twice on the same receiver",
                ));
            }
            let subscriber = self.subscriber.clone();
            let kexpr = self.subscriber.key_expr().clone();
            async move {
                loop {
                    match subscriber.recv_async().await {
                        Ok(sample) if sample.kind() == SampleKind::Delete => continue,
                        Err(_) => {
                            return Ok(None);
                        }
                        Ok(sample) => {
                            if *sample.key_expr() != kexpr {
                                continue;
                            }
                            return Ok(Some(sample_to_string(sample)));
                        }
                    }
                }
            }
        })
    }
}

pub fn mailbox_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mailbox>()?;
    Ok(())
}
