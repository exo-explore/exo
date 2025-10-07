use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::task::{Context, Waker};

/// A wrapper around [`VecDeque`] which wakes (if it can) on any `push_*` methods,
/// and updates the internally stored waker by consuming [`Context`] on any `pop_*` methods.
pub struct WakerDeque<T> {
    waker: Option<Waker>,
    deque: VecDeque<T>,
}

impl<T: Debug> Debug for WakerDeque<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.deque.fmt(f)
    }
}

impl<T> WakerDeque<T> {
    pub fn new() -> Self {
        Self {
            waker: None,
            deque: VecDeque::new(),
        }
    }

    fn update(&mut self, cx: &mut Context<'_>) {
        self.waker = Some(cx.waker().clone());
    }

    fn wake(&mut self) {
        let Some(ref mut w) = self.waker else { return };
        w.wake_by_ref();
        self.waker = None;
    }

    pub fn pop_front(&mut self, cx: &mut Context<'_>) -> Option<T> {
        self.update(cx);
        self.deque.pop_front()
    }

    pub fn pop_back(&mut self, cx: &mut Context<'_>) -> Option<T> {
        self.update(cx);
        self.deque.pop_back()
    }

    pub fn push_front(&mut self, value: T) {
        self.wake();
        self.deque.push_front(value);
    }

    pub fn push_back(&mut self, value: T) {
        self.wake();
        self.deque.push_back(value);
    }
}
