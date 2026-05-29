use networking::Session;
use pyo3::{
    exceptions::{PyConnectionError, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::Wait;
use zenoh_ext::{
    AdvancedPublisherBuilderExt, AdvancedSubscriberBuilderExt, CacheConfig, HistoryConfig,
    MissDetectionConfig,
};

use crate::{
    last_value::{LVAggregator, LVPublisher, LVSubscriber, spawn_lv_aggregator_onto},
    networking::PyNetworkingHandle,
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
}

pub fn session_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SessionHandle>()?;
    Ok(())
}
