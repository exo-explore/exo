use std::{collections::HashMap, time::Duration};

use networking::STORAGE_PREFIX;
use pyo3::{
    exceptions::{PyConnectionError, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::{
    Session as ZSession, Wait, handlers::FifoChannelHandler, query::Reply, sample::SampleKind,
};

use crate::sample_to_string;

#[gen_stub_pyclass]
#[pyclass]
pub struct Storage {
    pub session: ZSession,
}

#[gen_stub_pymethods]
#[pymethods]
impl Storage {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[str | None]",
        imports=("collections.abc")
    ))]
    pub fn get<'py>(&'py self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        if key.contains('*') {
            return Err(PyValueError::new_err(format!(
                "{key} is invalid -- Storage.get only supports fixed keys"
            )));
        }
        let session = self.session.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let subscriber = session
                .get(format!("{STORAGE_PREFIX}/{key}"))
                //.allowed_destination(Locality::SessionLocal)
                .await
                .map_err(|e| PyConnectionError::new_err(format!("failed to query storage: {e}")))?;
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    Ok(None)
                }
                reply = subscriber.recv_async() => {
                    Ok(reply.ok()
                        .and_then(|reply| reply.into_result().ok())
                        .filter(|sample| sample.kind() == SampleKind::Put)
                        .map(sample_to_string)
                    )
                }
            }
        })
    }
    pub fn get_many(&self, key: String) -> PyResult<StorageGetter> {
        self.session
            .get(key)
            .wait()
            .map_err(|e| PyConnectionError::new_err(format!("failed to query storage: {e}")))
            .map(StorageGetter)
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn put<'py>(
        &'py self,
        py: Python<'py>,
        key: String,
        data: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        if key.contains('*') {
            return Err(PyValueError::new_err(format!(
                "{key} is invalid -- Storage.put only supports fixed keys"
            )));
        }
        let session = self.session.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            session
                .put(format!("{STORAGE_PREFIX}/{key}"), data)
                .await
                .map_err(|e| PyConnectionError::new_err(format!("failed to query storage: {e}")))
        })
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn delete<'py>(&'py self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        if key.contains('*') {
            return Err(PyValueError::new_err(format!(
                "{key} is invalid -- Storage.delete only supports fixed keys"
            )));
        }
        let session = self.session.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            session
                .delete(format!("{STORAGE_PREFIX}/{key}"))
                .await
                .map_err(|e| PyConnectionError::new_err(format!("failed to query storage: {e}")))
        })
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[dict[str, str]]",
        imports=("collections.abc")
    ))]
    pub fn dump<'py>(&'py self, py: Python<'py>, prefix: String) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(networking::read_raw_memory_storage()
                .await
                .into_iter()
                .filter_map(|(key, value)| {
                    Some((
                        key?.as_str().strip_prefix(prefix.as_str())?.to_string(),
                        value
                            .payload
                            .try_to_string()
                            .expect("we only use utf8 encoded strings. someone messed up")
                            .to_string(),
                    ))
                })
                .collect::<HashMap<String, String>>())
        })
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct StorageGetter(FifoChannelHandler<Reply>);

#[gen_stub_pymethods]
#[pymethods]
impl StorageGetter {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[tuple[str, str] | None]",
        imports=("collections.abc")
    ))]
    fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.0.receiver_count() != 1 {
            return Err(PyRuntimeError::new_err(
                "Tried to call StorageGetter.recv twice concurrently",
            ));
        }
        let dupe = self.0.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let sample = loop {
                match dupe.recv_async().await {
                    Err(_) => return Ok(None),
                    Ok(reply) => match reply.into_result() {
                        Err(e) => {
                            log::warn!("Ignoring reply error: {e}");
                            continue;
                        }
                        Ok(sample) => match sample.kind() {
                            SampleKind::Put => break sample,
                            SampleKind::Delete => {
                                log::warn!(
                                    "Received unexpected DELETE from queryable: {}",
                                    sample.key_expr()
                                );
                                continue;
                            }
                        },
                    },
                };
            };

            let key = sample
                .key_expr()
                .to_string()
                .strip_prefix(format!("{STORAGE_PREFIX}/").as_str())
                .expect("invalid storage format encountered")
                .to_string();

            Ok(Some((key, sample_to_string(sample))))
        })
    }
}

pub fn storage_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Storage>()?;
    m.add_class::<StorageGetter>()?;
    Ok(())
}
