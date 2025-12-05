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

pub mod discovery;
pub mod keep_alive;
pub mod swarm;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {
    use extend::ext;
    use libp2p::Multiaddr;
    use libp2p::multiaddr::Protocol;
    use std::net::IpAddr;

    #[ext(pub, name = MultiaddrExt)]
    impl Multiaddr {
        /// If the multiaddress corresponds to a TCP address, extracts it
        fn try_to_tcp_addr(&self) -> Option<(IpAddr, u16)> {
            let mut ps = self.into_iter();
            let ip = if let Some(p) = ps.next() {
                match p {
                    Protocol::Ip4(ip) => IpAddr::V4(ip),
                    Protocol::Ip6(ip) => IpAddr::V6(ip),
                    _ => return None,
                }
            } else {
                return None;
            };
            let Some(Protocol::Tcp(port)) = ps.next() else {
                return None;
            };
            Some((ip, port))
        }
    }
}

pub(crate) mod private {
    #![allow(dead_code)]

    /// Sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}
