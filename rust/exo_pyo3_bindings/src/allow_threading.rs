//! SEE: <https://pyo3.rs/v0.27.1/async-await.html#detaching-from-the-interpreter-across-await>

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::{
    future::Future,
    pin::{Pin, pin},
    task::{Context, Poll},
};

#[repr(transparent)]
pub struct AllowThreads<F>(F);

impl<F> AllowThreads<F>
where
    Self: Future,
{
    pub(crate) const fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> Future for AllowThreads<F>
where
    F: Future + Unpin + Send,
    F::Output: Send,
{
    type Output = Result<F::Output, PyErr>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let waker = cx.waker();
        match Python::try_attach(|py| {
            py.detach(|| pin!(&mut self.0).poll(&mut Context::from_waker(waker)))
        }) {
            Some(Poll::Pending) => Poll::Pending,
            Some(Poll::Ready(t)) => Poll::Ready(Ok(t)),
            // TODO: this doesn't actually work - graceful py shutdown handling
            None => Poll::Ready(Err(PyRuntimeError::new_err(
                "Python runtime shutdown while awaiting a future",
            ))),
        }
    }
}
