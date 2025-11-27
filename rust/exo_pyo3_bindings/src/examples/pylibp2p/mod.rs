//! A module for exposing Rust's libp2p datatypes over Pyo3
//!
//! TODO: right now we are coupled to libp2p's identity, but eventually we want to create our own
//!       independent identity type of some kind or another. This may require handshaking.
//!

pub mod ident;
pub mod multiaddr;
