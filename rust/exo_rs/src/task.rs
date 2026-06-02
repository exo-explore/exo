use std::{sync::Arc, time::Duration};

use pyo3::{
    exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zenoh::{
    Session as ZSession, Wait,
    handlers::FifoChannelHandler,
    pubsub::{Publisher, Subscriber},
    qos::CongestionControl,
    query::{ConsolidationMode, Query, Queryable},
    sample::{Sample, SampleKind},
};

#[gen_stub_pyclass]
#[pyclass]
pub struct TaskRequester {
    pub session: ZSession,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct TaskResponder {
    pub instance_id: String,
    pub queryable: Queryable<FifoChannelHandler<Query>>,
    pub session: ZSession,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct TaskRequest {
    pub query: Query,
    pub key: String,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct TaskChunkSender {
    pub publisher: Arc<Publisher<'static>>,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct TaskStream {
    pub receiver: Subscriber<FifoChannelHandler<Sample>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TaskRequester {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[TaskStream]",
        imports=("collections.abc")
    ))]
    pub fn submit<'py>(
        &'py self,
        py: Python<'py>,
        instance_id: String,
        command_id: String,
        command: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let receiver = declare_task_stream(&session, &command_id)?;

            request_task_admission(&session, instance_id, command_id, command).await?;
            Ok(TaskStream { receiver })
        })
    }

    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn interrupt<'py>(
        &'py self,
        py: Python<'py>,
        instance_id: String,
        command_id: String,
        command: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            request_task_admission(&session, instance_id, command_id, command).await
        })
    }
}

fn task_key(instance_id: &str, command_id: &str) -> String {
    format!("task/instances/{instance_id}/tasks/{command_id}")
}

fn task_chunks_key(command_id: &str) -> String {
    format!("task/commands/{command_id}/chunks")
}

fn declare_task_stream(
    session: &ZSession,
    command_id: &str,
) -> PyResult<Subscriber<FifoChannelHandler<Sample>>> {
    session
        .declare_subscriber(task_chunks_key(command_id))
        .wait()
        .map_err(|e| PyConnectionError::new_err(format!("failed to declare task stream: {e}")))
}

async fn request_task_admission(
    session: &ZSession,
    instance_id: String,
    command_id: String,
    command: String,
) -> PyResult<()> {
    let replies = session
        .get(task_key(&instance_id, &command_id))
        .payload(command)
        .congestion_control(CongestionControl::Block)
        .consolidation(ConsolidationMode::None)
        .timeout(Duration::from_secs(5))
        .wait()
        .map_err(|e| PyConnectionError::new_err(format!("failed to submit task: {e}")))?;

    let reply = replies.recv_async().await.map_err(|e| {
        PyConnectionError::new_err(format!("task admission stream closed early: {e}"))
    })?;

    match reply.into_result() {
        Ok(sample) => {
            if sample.kind() == SampleKind::Delete {
                Err(PyConnectionError::new_err(
                    "task admission replied with delete",
                ))
            } else {
                let _ = sample_to_string(sample);
                Ok(())
            }
        }
        Err(error) => {
            if let Ok(payload) = error.payload().try_to_string() {
                if payload == "Timeout" {
                    Err(PyTimeoutError::new_err("task admission timed out"))
                } else {
                    Err(PyRuntimeError::new_err(format!(
                        "task admission rejected: {payload}"
                    )))
                }
            } else {
                Err(PyRuntimeError::new_err(format!(
                    "task admission failed: {error}"
                )))
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TaskResponder {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[tuple[TaskRequest, TaskChunkSender, str | None] | None]",
        imports=("collections.abc")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.queryable.receiver_count() != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "called recv twice concurrently",
            ));
        }
        let queryable = self.queryable.clone();
        let session = self.session.clone();
        let key_prefix = format!("task/instances/{}/tasks/", self.instance_id);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                match queryable.recv_async().await {
                    Ok(query) => {
                        let query_key = query.key_expr().as_str();
                        if !query_key.starts_with(&key_prefix) {
                            continue;
                        }
                        let key = query_key.to_string();
                        let command_id = query_key[key_prefix.len()..].to_string();
                        let payload = query.payload().map(|payload| {
                            payload
                                .try_to_string()
                                .expect("we only use utf8 encoded strings. someone messed up")
                                .to_string()
                        });
                        let publisher = session
                            .declare_publisher(task_chunks_key(&command_id))
                            .wait()
                            .map_err(|e| {
                                PyConnectionError::new_err(format!(
                                    "failed to declare task chunk sender: {e}"
                                ))
                            })?;
                        return Ok(Some((
                            TaskRequest { query, key },
                            TaskChunkSender {
                                publisher: Arc::new(publisher),
                            },
                            payload,
                        )));
                    }
                    Err(_) => return Ok(None),
                }
            }
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TaskChunkSender {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[None]",
        imports=("collections.abc")
    ))]
    pub fn send<'py>(&'py self, py: Python<'py>, chunk: String) -> PyResult<Bound<'py, PyAny>> {
        let publisher = Arc::clone(&self.publisher);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            wait_for_matching_subscriber(&publisher).await?;
            publisher
                .put(chunk)
                .await
                .map_err(|e| PyConnectionError::new_err(format!("failed to send task chunk: {e}")))
        })
    }
}

async fn wait_for_matching_subscriber(publisher: &Publisher<'static>) -> PyResult<()> {
    let status = publisher.matching_status().await.map_err(|e| {
        PyConnectionError::new_err(format!("failed to check task chunk subscribers: {e}"))
    })?;
    if status.matching() {
        return Ok(());
    }

    let listener = publisher.matching_listener().await.map_err(|e| {
        PyConnectionError::new_err(format!("failed to listen for task chunk subscribers: {e}"))
    })?;
    loop {
        let status = listener.recv_async().await.map_err(|e| {
            PyConnectionError::new_err(format!("task chunk subscriber listener closed: {e}"))
        })?;
        if status.matching() {
            return Ok(());
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TaskRequest {
    pub fn reply(&self, payload: String) -> PyResult<()> {
        self.query
            .reply(&self.key, payload)
            .wait()
            .map_err(|e| PyConnectionError::new_err(format!("failed to reply to task query: {e}")))
    }

    pub fn reply_err(&self, payload: String) -> PyResult<()> {
        if payload == "Timeout" {
            return Err(PyValueError::new_err(
                "Timeout is reserved for zenoh query timeouts",
            ));
        }
        self.query
            .reply_err(payload)
            .wait()
            .map_err(|e| PyConnectionError::new_err(format!("failed to reject task query: {e}")))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TaskStream {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[str | None]",
        imports=("collections.abc")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.receiver.receiver_count() != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "called recv twice concurrently",
            ));
        }
        let receiver = self.receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(receiver.recv_async().await.ok().map(sample_to_string))
        })
    }
}

fn sample_to_string(sample: Sample) -> String {
    sample
        .payload()
        .try_to_string()
        .expect("we only use utf8 encoded strings. someone messed up")
        .to_string()
}

pub fn task_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TaskChunkSender>()?;
    m.add_class::<TaskRequester>()?;
    m.add_class::<TaskRequest>()?;
    m.add_class::<TaskResponder>()?;
    m.add_class::<TaskStream>()?;
    Ok(())
}
