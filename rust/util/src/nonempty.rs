use std::slice::SliceIndex;
use std::{ops, slice};
use thiserror::Error;

#[derive(Error, Debug)]
#[error("Cannot create to `NonemptyArray` because the supplied slice is empty")]
pub struct EmptySliceError;

/// A pointer to a non-empty fixed-size slice allocated on the heap.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonemptyArray<T>(Box<[T]>);

#[allow(clippy::arbitrary_source_item_ordering)]
impl<T> NonemptyArray<T> {
    #[inline]
    pub fn singleton(value: T) -> Self {
        Self(Box::new([value]))
    }

    #[allow(clippy::missing_errors_doc)]
    #[inline]
    pub fn try_from_boxed_slice<S: Into<Box<[T]>>>(
        boxed_slice: S,
    ) -> Result<Self, EmptySliceError> {
        let boxed_slice = boxed_slice.into();
        if boxed_slice.is_empty() {
            Err(EmptySliceError)
        } else {
            Ok(Self(boxed_slice))
        }
    }

    #[must_use]
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.0
    }

    #[must_use]
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.to_vec()
    }

    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        &self.0
    }

    #[allow(clippy::indexing_slicing)]
    #[must_use]
    #[inline]
    pub fn first(&self) -> &T {
        &self.0[0]
    }

    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    #[must_use]
    #[inline]
    pub fn last(&self) -> &T {
        &self.0[self.0.len() - 1]
    }

    #[must_use]
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.0.get(index)
    }

    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[allow(clippy::iter_without_into_iter)]
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.0.iter()
    }

    #[allow(clippy::iter_without_into_iter)]
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    #[inline]
    #[must_use]
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> NonemptyArray<U> {
        NonemptyArray(self.0.into_iter().map(f).collect())
    }
}

impl<T> From<NonemptyArray<T>> for Box<[T]> {
    #[inline]
    fn from(value: NonemptyArray<T>) -> Self {
        value.into_boxed_slice()
    }
}

impl<T> ops::Index<usize> for NonemptyArray<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T> IntoIterator for NonemptyArray<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_boxed_slice().into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a NonemptyArray<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
