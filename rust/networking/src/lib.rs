//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!
pub mod swarm;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}
