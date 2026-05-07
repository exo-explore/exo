use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde_json::{Map, Value};
use zenoh::{Result, Session, sample::SampleFields};

#[gen_stub_pyclass]
#[pyclass]
pub struct StateProxy {
    pub session: Session,
}
#[gen_stub_pymethods]
#[pymethods]
impl StateProxy {
    #[gen_stub(override_return_type(
        type_repr="collections.abc.Awaitable[str]",
        imports=("collections.abc")
    ))]
    pub fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, {
            let session = self.session.clone();
            async move {
                Self::_snapshot(session)
                    .await
                    .map_err(|e| PyValueError::new_err(e.to_string()))
                    .map(|v| v.to_string())
            }
        })
    }
}

impl StateProxy {
    async fn _snapshot(session: Session) -> Result<Value> {
        let q = session.get("storage/mem1/**").await?;

        let mut v = Value::Object(Map::default());

        while let Ok(sample) = q.recv_async().await {
            let mut cur_v = &mut v;
            let Ok(sample) = sample.into_result() else {
                continue;
            };
            // skip storage/mem1
            let SampleFields {
                payload, key_expr, ..
            } = sample.into();
            let mut iter = key_expr.split('/').skip(2).peekable();
            loop {
                let Some(p) = iter.next() else {
                    break;
                };
                if iter.peek().is_none() {
                    // terminal; write value into json
                    let existing = cur_v
                        .as_object_mut()
                        .expect("path terminated unexpectedly - value stored at some/path and some/path/two")
                        .insert(p.to_owned(), Value::String(payload.try_to_string()?.to_string()));

                    if let Some(value) = existing {
                        assert!(value.is_string())
                        // could log, but string overwrites are fine
                    }
                } else {
                    // non-terminal; ensure key exists in v, then replace cur with that object
                    cur_v = cur_v
                        .as_object_mut()
                        .expect("path terminated unexpectedly - value stored at some/path and some/path/two")
                        .entry(p)
                        .or_insert(Value::Object(Map::default()));
                    assert!(
                        cur_v.is_object(),
                        "path terminated unexpectedly - value stored at some/path and some/path/two"
                    )
                }
            }
        }
        Ok(v)
    }
}
