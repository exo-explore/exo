//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
#![feature(tuple_trait)]
#![feature(unboxed_closures)]
// #![feature(stmt_expr_attributes)]
// #![feature(assert_matches)]
// #![feature(async_fn_in_dyn_trait)]
// #![feature(async_for_loop)]
// #![feature(auto_traits)]
// #![feature(negative_impls)]

extern crate core;
pub(crate) mod discovery;
pub(crate) mod pylibp2p;

use crate::discovery::discovery_submodule;
use crate::pylibp2p::connection::connection_submodule;
use crate::pylibp2p::ident::ident_submodule;
use crate::pylibp2p::multiaddr::multiaddr_submodule;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{prelude::*, types::*};
use pyo3::{pyclass, pymodule, Bound, PyResult};
use pyo3_stub_gen::define_stub_info_gatherer;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;
    use std::marker::Tuple;

    pub trait SendFn<Args: Tuple + Send + 'static, Output> =
        Fn<Args, Output = Output> + Send + 'static;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use extend::ext;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::PyErr;

    #[ext(pub, name = ResultExt)]
    impl<T, E> Result<T, E>
    where
        E: ToString,
    {
        fn pyerr(self) -> Result<T, PyErr> {
            self.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }
}

pub(crate) mod private {
    use std::marker::Sized;

    /// Sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

pub(crate) const MPSC_CHANNEL_SIZE: usize = 8;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "exo_pyo3_bindings")]
fn main_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // install logger
    pyo3_log::init();

    // TODO: for now this is all NOT a submodule, but figure out how to make the submodule system
    //       work with maturin, where the types generate correctly, in the right folder, without
    //       too many importing issues...
    connection_submodule(m)?;
    ident_submodule(m)?;
    multiaddr_submodule(m)?;
    discovery_submodule(m)?;

    // top-level constructs
    // TODO: ...

    Ok(())
}

define_stub_info_gatherer!(stub_info);

/// Test of unit test for testing link problem
#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        assert_eq!(2 + 2, 4);
    }
}
