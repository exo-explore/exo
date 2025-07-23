//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
// #![feature(stmt_expr_attributes)]
// #![feature(unboxed_closures)]
// #![feature(assert_matches)]
// #![feature(async_fn_in_dyn_trait)]
// #![feature(async_for_loop)]
// #![feature(auto_traits)]
// #![feature(negative_impls)]

use crate::participant::ParticipantId;

pub mod cel;
mod communicator;
mod participant;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {}

pub(crate) mod private {
    /// Sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

pub enum ElectionMessage {
    /// Announce election
    Election {
        candidate: ParticipantId,
    },
    Alive,
    Victory {
        coordinator: ParticipantId,
    },
}
