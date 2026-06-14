use crate::ext::ResultExt;
use clap::{ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use pyo3::pyclass::boolean_struct::False;
use pyo3::{
    Borrowed, FromPyObject, IntoPyObject, Py, PyAny, PyClass, PyClassInitializer, PyErr, PyResult,
    Python,
};
use pyo3_stub_gen::{PyStubType, TypeInfo};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ffi::OsString;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

/// Wrapper around [`Py`] to provide integration with other libraries.
#[repr(transparent)]
pub struct NewPy<T>(Py<T>);

impl<T: Debug> Debug for NewPy<T>
where
    T: PyClass,
{
    #[inline(always)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Python::attach(|py| (&*self.borrow(py)).fmt(f))
    }
}

impl<T: Clone> Clone for NewPy<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self::new(Python::attach(|py| self.clone_ref(py)))
    }
}

impl<T: PartialEq> PartialEq for NewPy<T>
where
    T: PyClass,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        Python::attach(|py| &*self.borrow(py) == &*other.borrow(py))
    }
}

impl<T: Eq> Eq for NewPy<T> where T: PyClass {}

impl<T: Display> Display for NewPy<T>
where
    T: PyClass,
{
    #[inline(always)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Python::attach(|py| (&*self.borrow(py)).fmt(f))
    }
}

impl<T: FromStr> FromStr for NewPy<T>
where
    T: PyClass + Into<PyClassInitializer<T>>,
    T::Err: ToString,
{
    type Err = PyErr;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::py_try_new(<T as FromStr>::from_str(s).pyerr()?)
    }
}

impl<T> From<Py<T>> for NewPy<T> {
    #[inline(always)]
    fn from(inner: Py<T>) -> Self {
        Self::new(inner)
    }
}

impl<T> Deref for NewPy<T> {
    type Target = Py<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl<T> DerefMut for NewPy<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner_mut()
    }
}

impl<'a, 'py, T> FromPyObject<'a, 'py> for NewPy<T>
where
    Py<T>: FromPyObject<'a, 'py>,
{
    type Error = <Py<T> as FromPyObject<'a, 'py>>::Error;

    #[inline(always)]
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        <Py<T> as FromPyObject<'a, 'py>>::extract(ob).map(Self::new)
    }
}

impl<'py, T> IntoPyObject<'py> for NewPy<T>
where
    Py<T>: IntoPyObject<'py>,
{
    type Target = <Py<T> as IntoPyObject<'py>>::Target;
    type Output = <Py<T> as IntoPyObject<'py>>::Output;
    type Error = <Py<T> as IntoPyObject<'py>>::Error;

    #[inline(always)]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        <Py<T> as IntoPyObject<'py>>::into_pyobject(self.into_inner(), py)
    }
}

impl<T> Serialize for NewPy<T>
where
    Py<T>: Serialize,
{
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        <Py<T> as Serialize>::serialize(&*self, serializer)
    }
}

impl<'de, T> Deserialize<'de> for NewPy<T>
where
    Py<T>: Deserialize<'de>,
{
    #[inline(always)]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        <Py<T> as Deserialize<'de>>::deserialize(deserializer).map(Self::new)
    }
}

impl<T> PyStubType for NewPy<T>
where
    Py<T>: PyStubType,
{
    #[inline(always)]
    fn type_output() -> TypeInfo {
        <Py<T> as PyStubType>::type_output()
    }
}

impl<T> NewPy<T> {
    #[inline(always)]
    fn new(inner: impl Into<Py<T>>) -> Self {
        Self(inner.into())
    }

    #[inline(always)]
    pub fn py_try_new_with(
        py: Python<'_>,
        value: impl Into<PyClassInitializer<T>>,
    ) -> PyResult<Self>
    where
        T: PyClass,
    {
        Py::new(py, value).map(Self)
    }

    #[inline(always)]
    pub fn py_try_new(value: impl Into<PyClassInitializer<T>>) -> PyResult<Self>
    where
        T: PyClass,
    {
        Python::attach(|py| Self::py_try_new_with(py, value))
    }

    #[inline(always)]
    pub fn clap_try_new(value: impl Into<PyClassInitializer<T>>) -> Result<Self, clap::Error>
    where
        T: PyClass,
    {
        Self::py_try_new(value).map_err(|e| clap::Error::raw(clap::error::ErrorKind::Io, e))
    }

    #[inline(always)]
    pub fn inner(&self) -> &Py<T> {
        &self.0
    }

    #[inline(always)]
    pub fn inner_mut(&mut self) -> &mut Py<T> {
        &mut self.0
    }

    #[inline(always)]
    pub fn into_inner(self) -> Py<T> {
        self.0
    }
}

impl<T: Parser> Parser for NewPy<T>
where
    T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
{
    fn try_parse() -> Result<Self, clap::Error> {
        <T as Parser>::try_parse().and_then(Self::clap_try_new)
    }

    fn try_parse_from<I, It>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = It>,
        It: Into<OsString> + Clone,
    {
        <T as Parser>::try_parse_from(itr).and_then(Self::clap_try_new)
    }
}

impl<T: CommandFactory> CommandFactory for NewPy<T> {
    fn command() -> clap::Command {
        <T as CommandFactory>::command()
    }
    fn command_for_update() -> clap::Command {
        <T as CommandFactory>::command_for_update()
    }
}

impl<T: FromArgMatches> FromArgMatches for NewPy<T>
where
    T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
{
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        <T as FromArgMatches>::from_arg_matches(matches).and_then(Self::clap_try_new)
    }
    fn from_arg_matches_mut(matches: &mut ArgMatches) -> Result<Self, clap::Error> {
        <T as FromArgMatches>::from_arg_matches_mut(matches).and_then(Self::clap_try_new)
    }
    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        Python::attach(|py| {
            <T as FromArgMatches>::update_from_arg_matches(&mut *self.borrow_mut(py), matches)
        })
    }
    fn update_from_arg_matches_mut(&mut self, matches: &mut ArgMatches) -> Result<(), clap::Error> {
        Python::attach(|py| {
            <T as FromArgMatches>::update_from_arg_matches_mut(&mut *self.borrow_mut(py), matches)
        })
    }
}

impl<T: Args> Args for NewPy<T>
where
    T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
{
    fn augment_args(cmd: clap::Command) -> clap::Command {
        <T as Args>::augment_args(cmd)
    }
    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        <T as Args>::augment_args_for_update(cmd)
    }
}

impl<T: Subcommand> Subcommand for NewPy<T>
where
    T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
{
    fn augment_subcommands(cmd: clap::Command) -> clap::Command {
        <T as Subcommand>::augment_subcommands(cmd)
    }
    fn augment_subcommands_for_update(cmd: clap::Command) -> clap::Command {
        <T as Subcommand>::augment_subcommands_for_update(cmd)
    }
    fn has_subcommand(name: &str) -> bool {
        <T as Subcommand>::has_subcommand(name)
    }
}
