//! See: <https://pyo3.rs/v0.27.2/async-await.html#detaching-from-the-interpreter-across-await>
use pyo3::prelude::*;
use std::{
    future::Future,
    pin::{Pin, pin},
    task::{Context, Poll},
};

pub struct AllowThreads<F>(pub(crate) F);

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
