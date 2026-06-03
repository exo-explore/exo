use networking::{AbortOnDrop, Session};
use parking_lot::Mutex;
use pyo3::{
    exceptions::{PyConnectionError, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::sync::mpsc;
use zenoh::{Wait, qos::CongestionControl, sample::SampleKind};
use zenoh_ext::{
    AdvancedPublisherBuilderExt, AdvancedSubscriberBuilderExt, CacheConfig, HistoryConfig,
    MissDetectionConfig,
};

use crate::{
    last_value::{
        ClearingLVSubscriber, LVAggregator, LVPublisher, LVSubscriber, spawn_lv_aggregator_onto,
    },
    mailbox::Mailbox,
    networking::PyNetworkingHandle,
    sample_to_string,
    storage::Storage,
    task::{TaskRequester, TaskResponder},
};

#[gen_stub_pyclass]
#[pyclass]
pub struct SessionHandle {
    pub session: Session,
}

#[gen_stub_pymethods]
#[pymethods]
impl SessionHandle {
    #[staticmethod]
    pub fn new<'py>(
        identity: &str,
        namespace: &str,
        listen_port: u16,
        discovery_service_port: u16,
    ) -> PyResult<(SessionHandle, PyNetworkingHandle)> {
        // get identity
        if !identity
            .chars()
            .all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
            || identity.len() > 32
        {
            return Err(PyValueError::new_err(format!(
                "{identity} is not a valid zenoh identity"
            )));
        }

        let cfg = networking::cfg(identity, listen_port).map_err(|e| {
            PyValueError::new_err(format!("failed to write config: {}", e.to_string()))
        })?;
        let session = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(networking::open(
                cfg,
                namespace,
                listen_port,
                discovery_service_port,
            ))
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "failed to spawn networking on tokio runtime: {}",
                    e.to_string()
                ))
            })?;
        let legacy = PyNetworkingHandle::from_session(session.clone());
        Ok((Self { session }, legacy))
    }

    pub fn last_value_aggregator(&self, prefix: String) -> PyResult<LVAggregator> {
        spawn_lv_aggregator_onto(&self.session, prefix.into()).map_err(|e| {
            PyConnectionError::new_err(format!("failed to spawn liveliness aggregator: {e}"))
        })
    }

    pub fn last_value_subscriber(&self, kexpr: &str) -> PyResult<LVSubscriber> {
        // nota bene: config must be kept in track with the LVAggregator
        self.session
            .z
            .declare_subscriber(kexpr)
            .advanced()
            .history(
                HistoryConfig::default()
                    .max_samples(1)
                    .detect_late_publishers(),
            )
            .wait()
            .map_err(|e| PyConnectionError::new_err(format!("failed to declare subscriber: {e}")))
            .map(|subscriber| LVSubscriber { subscriber })
    }

    pub fn last_value_publisher(&self, kexpr: String) -> PyResult<LVPublisher> {
        self.session
            .z
            .declare_publisher(kexpr)
            .advanced()
            .publisher_detection()
            .sample_miss_detection(MissDetectionConfig::default())
            .cache(CacheConfig::default().max_samples(1))
            .wait()
            .map_err(|e| PyConnectionError::new_err(format!("failed to declare publisher: {e}")))
            .map(LVPublisher::new)
    }

    /// An LV subscriber which synthesizes delete events for offline nodes
    pub fn clearing_last_value_subscriber(&self, kexpr: &str) -> PyResult<ClearingLVSubscriber> {
        // nota bene: config must be kept in track with the LVAggregator
        let (send, recv) = mpsc::unbounded_channel();
        let subscriber = self
            .session
            .z
            .declare_subscriber(kexpr)
            .advanced()
            .history(
                HistoryConfig::default()
                    .max_samples(1)
                    .detect_late_publishers(),
            )
            .wait()
            .map_err(|e| {
                PyConnectionError::new_err(format!("failed to declare subscriber: {e}"))
            })?;
        let liveliness_subscriber = self
            .session
            .z
            .liveliness()
            .declare_subscriber("live/*")
            .history(true)
            .wait()
            .map_err(|e| {
                PyConnectionError::new_err(format!("failed to declare liveliness subscriber: {e}"))
            })?;
        let handle = AbortOnDrop(tokio::task::spawn(async move {
            let mut seen_keys = HashSet::<String>::new();

            loop {
                tokio::select! {
                    sample = subscriber.recv_async() => {
                        let Ok(sample) = sample else {
                            break;
                        };

                        let key = sample.key_expr().to_string();

                        match sample.kind() {
                            SampleKind::Put => {
                                let value = sample_to_string(sample);
                                seen_keys.insert(key.clone());
                                let _ = send.send((key, Some(value)));
                            }
                            SampleKind::Delete => {
                                seen_keys.remove(&key);
                                let _ = send.send((key, None));
                            }
                        }
                    }

                    sample = liveliness_subscriber.recv_async() => {
                        let Ok(sample) = sample else {
                            break;
                        };
                        if sample.kind() == SampleKind::Put {
                            continue;
                        }
                        let kexpr = sample.key_expr().to_string();
                        let Some(node_id) = kexpr.strip_prefix("live/") else {
                            continue;
                        };

                        let deleted: Vec<String> = seen_keys
                            .iter()
                            .filter(|key| key.contains(node_id))
                            .cloned()
                            .collect();

                        for key in deleted {
                            seen_keys.remove(&key);
                            let _ = send.send((key, None));
                        }
                    }
                }
            }
        }));
        Ok(ClearingLVSubscriber {
            receiver: Arc::new(tokio::sync::Mutex::new(recv)),
            handle,
        })
    }
    pub fn storage_interface(&self) -> Storage {
        Storage {
            session: self.session.z.clone(),
        }
    }

    pub fn task_requester(&self) -> TaskRequester {
        TaskRequester {
            session: self.session.z.clone(),
        }
    }

    pub fn task_responder(&self, instance_id: String) -> PyResult<TaskResponder> {
        let queryable = self
            .session
            .z
            .declare_queryable(format!("task/instances/{instance_id}/tasks/*"))
            .complete(true)
            .wait()
            .map_err(|e| {
                PyConnectionError::new_err(format!("failed to declare task responder: {e}"))
            })?;
        Ok(TaskResponder {
            instance_id,
            queryable,
            session: self.session.z.clone(),
            assignments: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn send_mail<'py>(
        &'py self,
        py: Python<'py>,
        node_ids: Vec<String>,
        payload: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.z.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                for node_id in node_ids {
                    // TIL: session.z.put() just. is blocking with CongestionControl::Block. Even through its async apis.
                    // TODO: migrate more(?) zenoh calls to a dedicated thread **if necessary**. I do it here as we spin up N puts simultaneously.
                    session
                        .put(format!("mail/{node_id}"), payload.as_bytes())
                        .congestion_control(CongestionControl::Block)
                        .wait()
                        .map_err(|e| {
                            PyConnectionError::new_err(format!(
                                "failed to declare task responder: {e}"
                            ))
                        })?;
                }
                Ok(())
            })
            .await
            .expect("panic in worker thread")
        })
    }

    pub fn mailbox(&self, node_id: String) -> PyResult<Mailbox> {
        let subscriber = self
            .session
            .z
            .declare_subscriber(format!("mail/{node_id}"))
            .wait()
            .map_err(|e| {
                PyConnectionError::new_err(format!("failed to declare task responder: {e}"))
            })?;
        Ok(Mailbox { subscriber })
    }
}

pub fn session_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SessionHandle>()?;
    Ok(())
}
