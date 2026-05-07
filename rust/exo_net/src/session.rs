use std::sync::Arc;

use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use zenoh::Session;
use zenoh::Wait;
use zenoh::qos::CongestionControl;

use crate::{
    point_to_point::{NetReceiver, NetSender},
    state::StateProxy,
};

#[gen_stub_pyclass]
#[pyclass]
pub struct PySession {
    pub session: Session,
}
#[gen_stub_pymethods]
#[pymethods]
impl PySession {
    /* for now construct with NetworkingHandle
    #[staticmethod]
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[PySession]",
        imports=("collections.abc")
    ))]
    pub fn init<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(Self {
                session: networking::open(
                    networking::cfg(rand::random(), 0).expect("default cfg is valid"),
                )
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            })
        })
    }
    */

    pub fn net_receiver<'py>(&self, key: String) -> PyResult<NetReceiver> {
        Ok(NetReceiver {
            subscriber: self
                .session
                .declare_subscriber(key)
                .wait()
                // C5: key format error
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    pub fn net_sender<'py>(&self, key: String) -> PyResult<NetSender> {
        Ok(NetSender {
            publisher: Arc::new(
                self.session
                    .declare_publisher(key)
                    .congestion_control(CongestionControl::Block)
                    .wait()
                    // C5: key format error, could be declaration error
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            first: true,
        })
    }

    pub fn state_proxy(&self) -> StateProxy {
        StateProxy {
            session: self.session.clone(),
        }
    }
}
