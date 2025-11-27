//! SEE: https://pyo3.rs/v0.27.1/async-await.html#detaching-from-the-interpreter-across-await

use pyo3::prelude::*;
use std::{
    future::Future,
    pin::{Pin, pin},
    task::{Context, Poll},
};

#[repr(transparent)]
pub(crate) struct AllowThreads<F>(F);

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
    F: Future + Unpin + Send,
    F::Output: Send,
{
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let waker = cx.waker();
        Python::attach(|py| py.detach(|| pin!(&mut self.0).poll(&mut Context::from_waker(waker))))
    }
}
