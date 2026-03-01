//! SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await
//!

use pin_project::pin_project;
use pyo3::prelude::*;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

/// SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await
#[pin_project]
#[repr(transparent)]
pub(crate) struct AllowThreads<F>(#[pin] F);

impl<F> AllowThreads<F>
where
    Self: Future,
{
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> Future for AllowThreads<F>
where
    F: Future + Send,
    F::Output: Send,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let waker = cx.waker();
        Python::attach(|py| py.detach(|| self.project().0.poll(&mut Context::from_waker(waker))))
    }
}
