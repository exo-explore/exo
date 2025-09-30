//! This module exists to hold examples of some pyo3 patterns that may be too complex to
//! re-create from scratch, but too inhomogenous to create an abstraction/wrapper around.
//!
//! Pattern examples include:
//!  - Async task handles: with GC-integrated cleanup
//!  - Sync/async callbacks from python: with propper eventloop handling
//!
//! Mutability pattern: https://pyo3.rs/v0.26.0/async-await.html#send--static-constraint
//!  - Store mutable fields in tokio's `Mutex<T>`
//!  - For async code: take `&self` and `.lock().await`
//!  - For sync code: take `&mut self` and `.get_mut()`

use crate::ext::{PyResultExt as _, ResultExt as _, TokioRuntimeExt as _};
use futures::FutureExt as _;
use futures::future::BoxFuture;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{PyModule, PyModuleMethods as _};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, PyTraverseError, PyVisit, Python, pyclass, pymethods,
};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;

fn needs_tokio_runtime() {
    tokio::runtime::Handle::current();
}

type SyncCallback = Box<dyn Fn() + Send + Sync>;
type AsyncCallback = Box<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>;

enum AsyncTaskMessage {
    SyncCallback(SyncCallback),
    AsyncCallback(AsyncCallback),
}

async fn async_task(
    sender: mpsc::UnboundedSender<()>,
    mut receiver: mpsc::UnboundedReceiver<AsyncTaskMessage>,
) {
    log::info!("RUST: async task started");

    // task state
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    let mut sync_cbs: Vec<SyncCallback> = vec![];
    let mut async_cbs: Vec<AsyncCallback> = vec![];

    loop {
        tokio::select! {
            // handle incoming messages from task-handle
            message = receiver.recv() => {
                // handle closed channel by exiting
                let Some(message) = message else {
                    log::info!("RUST: channel closed");
                    break;
                };

                // dispatch incoming event
                match message {
                    AsyncTaskMessage::SyncCallback(cb) => {
                        sync_cbs.push(cb);
                    }
                    AsyncTaskMessage::AsyncCallback(cb) => {
                        async_cbs.push(cb);
                    }
                }
            }

            // handle all other events
            _ = interval.tick() => {
                log::info!("RUST: async task tick");

                // call back all sync callbacks
                for cb in &sync_cbs {
                    cb();
                }

                // call back all async callbacks
                for cb in &async_cbs {
                    cb().await;
                }

                // send event on unbounded channel
                sender.send(()).expect("handle receiver cannot be closed/dropped");
            }
        }
    }

    log::info!("RUST: async task stopped");
}

// #[gen_stub_pyclass]
#[pyclass(name = "AsyncTaskHandle")]
#[derive(Debug)]
struct PyAsyncTaskHandle {
    sender: Option<mpsc::UnboundedSender<AsyncTaskMessage>>,
    receiver: mpsc::UnboundedReceiver<()>,
}

#[allow(clippy::expect_used)]
impl PyAsyncTaskHandle {
    const fn sender(&self) -> &mpsc::UnboundedSender<AsyncTaskMessage> {
        self.sender
            .as_ref()
            .expect("The sender should only be None after de-initialization.")
    }

    const fn sender_mut(&mut self) -> &mpsc::UnboundedSender<AsyncTaskMessage> {
        self.sender
            .as_mut()
            .expect("The sender should only be None after de-initialization.")
    }

    const fn new(
        sender: mpsc::UnboundedSender<AsyncTaskMessage>,
        receiver: mpsc::UnboundedReceiver<()>,
    ) -> Self {
        Self {
            sender: Some(sender),
            receiver,
        }
    }
}

// #[gen_stub_pymethods]
#[pymethods]
impl PyAsyncTaskHandle {
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        use pyo3_async_runtimes::tokio::get_runtime;

        // create communication channel TOWARDS our task
        let (h_sender, t_receiver) = mpsc::unbounded_channel::<AsyncTaskMessage>();

        // create communication channel FROM our task
        let (t_sender, h_receiver) = mpsc::unbounded_channel::<()>();

        // perform necessary setup within tokio context - or it crashes
        let () = get_runtime().block_on(async { needs_tokio_runtime() });

        // spawn tokio task with this thread's task-locals - without this, async callbacks on the new threads will not work!!
        _ = get_runtime().spawn_with_scope(py, async move {
            async_task(t_sender, t_receiver).await;
        });
        Ok(Self::new(h_sender, h_receiver))
    }

    /// NOTE: exceptions in callbacks are silently ignored until end of execution
    fn add_sync_callback(
        &self,
        // #[gen_stub(override_type(
        //     type_repr="collections.abc.Callable[[], None]",
        //     imports=("collections.abc")
        // ))]
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        // blocking call to async method -> can do non-blocking if needed
        self.sender()
            .send(AsyncTaskMessage::SyncCallback(Box::new(move || {
                _ = Python::with_gil(|py| callback.call0(py).write_unraisable_with(py));
            })))
            .pyerr()?;
        Ok(())
    }

    /// NOTE: exceptions in callbacks are silently ignored until end of execution
    fn add_async_callback(
        &self,
        // #[gen_stub(override_type(
        //     type_repr="collections.abc.Callable[[], collections.abc.Awaitable[None]]",
        //     imports=("collections.abc")
        // ))]
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        // blocking call to async method -> can do non-blocking if needed
        self.sender()
            .send(AsyncTaskMessage::AsyncCallback(Box::new(move || {
                let c = Python::with_gil(|py| callback.clone_ref(py));
                async move {
                    if let Some(f) = Python::with_gil(|py| {
                        let coroutine = c.call0(py).write_unraisable_with(py)?;
                        pyo3_async_runtimes::tokio::into_future(coroutine.into_bound(py))
                            .write_unraisable_with(py)
                    }) {
                        _ = f.await.write_unraisable();
                    }
                }
                .boxed()
            })))
            .pyerr()?;
        Ok(())
    }

    async fn receive_unit(&mut self) -> PyResult<()> {
        self.receiver
            .recv()
            .await
            .ok_or(PyErr::new::<PyRuntimeError, _>(
                "cannot receive unit on closed channel",
            ))
    }

    fn drain_units(&mut self) -> PyResult<i32> {
        let mut cnt = 0;
        loop {
            match self.receiver.try_recv() {
                Err(TryRecvError::Disconnected) => {
                    return Err(PyErr::new::<PyRuntimeError, _>(
                        "cannot receive unit on closed channel",
                    ));
                }
                Err(TryRecvError::Empty) => return Ok(cnt),
                Ok(()) => {
                    cnt += 1;
                    continue;
                }
            }
        }
    }

    // #[gen_stub(skip)]
    const fn __traverse__(&self, _visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(()) // This is needed purely so `__clear__` can work
    }

    // #[gen_stub(skip)]
    fn __clear__(&mut self) {
        // TODO: may or may not need to await a "kill-signal" oneshot channel message,
        //       to ensure that the networking task is done BEFORE exiting the clear function...
        //       but this may require GIL?? and it may not be safe to call GIL here??
        self.sender = None; // Using Option<T> as a trick to force `sender` channel to be dropped
    }
}

pub fn examples_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsyncTaskHandle>()?;

    Ok(())
}
