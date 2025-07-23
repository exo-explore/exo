//! TODO: crate documentation
//!
//! this is here as a placeholder documentation

// enable Rust-unstable features for convenience
#![feature(tuple_trait)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(unsized_fn_params)] // this is fine because I am PURELY wrapping around existing `Fn*` traits
// global lints
#![allow(internal_features)]
#![allow(clippy::arbitrary_source_item_ordering)]

use fn_pipe_proc::impl_fn_pipe_for_tuple;
use std::marker::Tuple;

/// A trait representing a pipe of functions, where the output of one will
/// be fed as the input of another, until the entire pipe ran
pub trait FnPipe<Args: Tuple>: FnMutPipe<Args> {
    extern "rust-call" fn run(&self, args: Args) -> Self::Output;
}

pub trait FnMutPipe<Args: Tuple>: FnOncePipe<Args> {
    extern "rust-call" fn run_mut(&mut self, args: Args) -> Self::Output;
}

pub trait FnOncePipe<Args: Tuple> {
    type Output;

    extern "rust-call" fn run_once(self, args: Args) -> Self::Output;
}

// implement `Fn/Pipe*` variants for tuples of upto length 26,
// can be increased in the future
impl_fn_pipe_for_tuple!(26usize);
