// enable Rust-unstable features for convenience
#![feature(trait_alias)]

pub mod wakerdeque;

/// Namespace for crate-wide extension traits/methods
pub mod ext {
    use extend::ext;

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
