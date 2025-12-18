//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
#![feature(stmt_expr_attributes)]
#![feature(type_alias_impl_trait)]
#![feature(specialization)]
#![feature(unboxed_closures)]
#![feature(const_trait_impl)]
#![feature(fn_traits)]

pub mod nonempty;
pub mod wakerdeque;

pub(crate) mod private {
    // sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {}

/// Namespace for crate-wide extension traits/methods
pub mod ext {
    use extend::ext;

    #[ext(pub, name = BoxedSliceExt)]
    impl<T> Box<[T]> {
        #[inline]
        fn map<B, F>(self, f: F) -> Box<[B]>
        where
            F: FnMut(T) -> B,
        {
            self.into_iter().map(f).collect()
        }
    }

    #[ext(pub, name = VecExt)]
    impl<T> Vec<T> {
        #[inline]
        fn map<B, F>(self, f: F) -> Vec<B>
        where
            F: FnMut(T) -> B,
        {
            self.into_iter().map(f).collect()
        }
    }
}
