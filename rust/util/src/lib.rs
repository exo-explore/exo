use extend::ext;
use std::collections::HashSet;
use std::hash::Hash;

pub mod path;

#[ext(pub, name = VecExt)]
impl<T> Vec<T> {
    /// Deduplicates vector while preserving the order.
    #[inline(always)]
    fn dedup_preserve_order(&mut self)
    where
        T: Eq + Hash + Clone,
    {
        let mut set = HashSet::new();
        self.retain(|x| set.insert(x.clone()));
    }
}
