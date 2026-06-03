use networking::{AbortOnDrop, Session, liveliness_aggregator::LivelinessAggregator};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::sync::mpsc;
use zenoh::{Result as ZResult, Wait};

use parking_lot::Mutex;
use pyo3::{
    exceptions::{PyConnectionError, PyRuntimeError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::{
    handlers::FifoChannelHandler,
    sample::{Sample, SampleKind},
};
use zenoh_ext::{
    AdvancedPublisher, AdvancedSubscriber, AdvancedSubscriberBuilderExt, HistoryConfig,
};

use crate::sample_to_string;

#[gen_stub_pyclass]
#[pyclass]
pub struct LVAggregator {
    pub prefix: Arc<str>,
    pub store: Arc<Mutex<HashMap<String, String>>>,
    pub current_live: LivelinessAggregator,
}

pub fn spawn_lv_aggregator_onto(session: &Session, prefix: Arc<str>) -> ZResult<LVAggregator> {
    // nota bene: config must be kept in line with SessionHandle::last_value_receiver
    let store = Arc::new(Mutex::new(HashMap::default()));
    session
        .z
        .declare_subscriber(format!("{prefix}/**"))
        .advanced()
        .history(
            HistoryConfig::default()
                .max_samples(1)
                .detect_late_publishers(),
        )
        .callback({
            let store = Arc::clone(&store);
            let prefix = Arc::clone(&prefix);
            move |sample| {
                if let Some(s) = sample
                    .key_expr()
                    .to_string()
                    .strip_prefix(&*prefix)
                    .and_then(|it| it.strip_prefix('/'))
                {
                    let s = s.to_string();
                    match sample.kind() {
                        SampleKind::Put => {
                            store.lock().insert(s, sample_to_string(sample));
                        }
                        SampleKind::Delete => {
                            store.lock().remove(&s);
                        }
                    }
                };
            }
        })
        .background()
        .wait()?;
    Ok(LVAggregator {
        prefix,
        store,
        current_live: session.liveliness_aggregator.clone(),
    })
}

#[gen_stub_pymethods]
#[pymethods]
impl LVAggregator {
    pub fn dump(&self) -> HashMap<String, String> {
        let mut store = self.store.lock();
        let currently_alive: HashSet<String> = self.current_live.dump();
        // remove any keys that are no longer live
        store.retain(|key, _| {
            currently_alive.iter().any(|node_id| {
                key.strip_prefix(node_id)
                    .is_some_and(|rest| rest.starts_with("/"))
            })
        });
        store.clone()
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct ClearingLVSubscriber {
    pub receiver: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<(String, Option<String>)>>>,
    pub handle: AbortOnDrop,
}
#[gen_stub_pymethods]
#[pymethods]
impl ClearingLVSubscriber {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[tuple[str, str | None] | None]",
        imports=("collections.abc")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, {
            let receiver = Arc::clone(&self.receiver);
            async move { Ok(receiver.lock().await.recv().await) }
        })
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct LVSubscriber {
    pub subscriber: AdvancedSubscriber<FifoChannelHandler<Sample>>,
}
#[gen_stub_pymethods]
#[pymethods]
impl LVSubscriber {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[tuple[str, str | None] | None]",
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
            async move {
                Ok(subscriber.recv_async().await.ok().map(|sample| {
                    (
                        sample.key_expr().to_string(),
                        match sample.kind() {
                            SampleKind::Put => Some(sample_to_string(sample)),
                            SampleKind::Delete => None,
                        },
                    )
                }))
            }
        })
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct LVPublisher {
    pub state: Arc<AdvancedPublisher<'static>>,
}
impl LVPublisher {
    pub fn new(publisher: AdvancedPublisher<'static>) -> Self {
        Self {
            state: Arc::new(publisher),
        }
    }
}
#[gen_stub_pymethods]
#[pymethods]
impl LVPublisher {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn put<'py>(&'py self, py: Python<'py>, data: String) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&self.state);
        pyo3_async_runtimes::tokio::future_into_py(py, {
            // clone the data so py can have it back
            async move {
                state
                    .put(data)
                    .await
                    .map_err(|e| PyConnectionError::new_err(e.to_string()))
            }
        })
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn delete<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&self.state);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            state
                .delete()
                .await
                .map_err(|e| PyConnectionError::new_err(e.to_string()))
        })
    }
}

pub fn lv_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LVPublisher>()?;
    m.add_class::<LVSubscriber>()?;
    m.add_class::<LVAggregator>()?;
    Ok(())
}
