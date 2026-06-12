use crate::config::app::AppArgs;
use crate::config::bootstrap::BootstrapArgs;
use crate::ext::ResultExt;
use crate::{pickle_reduce, version};
use clap::{ArgAction, Parser};
use pyo3::prelude::{PyAnyMethods, PyModuleMethods};
use pyo3::types::{PyModule, PyTuple};
use pyo3::{pyclass, pymethods, Bound, PyAny, PyResult, Python};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;

#[gen_stub_pyclass]
#[pyclass(module = "exo_rs", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Parser)]
#[command(name = "EXO", version = version::version(), about, long_about = None)]
pub struct CliArgs {
    #[arg(
        short = 'm',
        long,
        action = ArgAction::SetTrue,
        help = "Force node to be master"
    )]
    #[pyo3(get, set)]
    pub force_master: bool,

    #[arg(
        long = "no-api",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the API"
    )]
    #[pyo3(get, set)]
    pub api_enabled: bool,

    #[arg(
        long,
        default_value_t = 52415,
        value_name = "PORT",
        help = "Port on which the API runs"
    )]
    #[pyo3(get, set)]
    pub api_port: u16,

    #[arg(
        long = "no-worker",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the worker"
    )]
    #[pyo3(get, set)]
    pub worker_enabled: bool,

    #[arg(
        long = "no-downloads",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the download coordinator (node won't download models)"
    )]
    #[pyo3(get, set)]
    pub downloads_enabled: bool,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Run as a legacy SysV-style background daemon using double-fork daemonization"
    )]
    #[pyo3(get, set)]
    pub legacy_daemon: bool,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "MULTIADDRS",
        help = "Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)"
    )]
    #[pyo3(get, set)]
    pub bootstrap_peers: Option<Vec<String>>,

    #[arg(
        long,
        env = "EXO_NAMESPACE",
        default_value_t = version::version().to_string(),
        value_name = "STRING",
        help = "Discovery namespace, nodes with different namespaces will not connect."
    )]
    #[pyo3(get, set)]
    pub namespace: String,

    #[arg(
        long,
        default_value_t = 52414,
        value_name = "PORT",
        help = "Fixed TCP port for zenoh to listen."
    )]
    #[pyo3(get, set)]
    pub zenoh_port: u16,

    #[arg(
        long,
        default_value_t = 52413,
        value_name = "PORT",
        help = "Fixed UDP port for the discovery service."
    )]
    #[pyo3(get, set)]
    pub discovery_port: u16,

    // -------- FLATTENED SUBCOMMANDS --------
    #[command(flatten)]
    #[pyo3(get)]
    pub bootstrap: BootstrapArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub app: AppArgs,

    #[command(flatten)]
    #[pyo3(get)]
    pub deprecated: DeprecatedArgs,
}

#[gen_stub_pymethods]
#[pymethods]
impl CliArgs {
    /// Create only from env-variables
    #[staticmethod]
    pub fn from_env_only() -> Self {
        // parse only from env - no arguments
        CliArgs::parse_from(&["exo"])
    }

    #[staticmethod]
    #[pyo3(name = "parse_from")]
    pub fn py_parse_from(argv: Vec<OsString>) -> Self {
        CliArgs::parse_from(argv)
    }

    #[staticmethod]
    #[pyo3(name = "parse")]
    pub fn py_parse(py: Python<'_>) -> PyResult<Self> {
        // the correct CLI args to parse is `sys.argv`, because the original ones
        // (i.e. `sys.orig_argv`) may contain extra arguments which would mess up parsing
        let argv: Vec<OsString> = PyModule::import(py, "sys")?.getattr("argv")?.extract()?;
        Ok(CliArgs::parse_from(argv))
    }

    pub fn set_bootstrap(&mut self, bootstrap: BootstrapArgs) {
        self.bootstrap = bootstrap;
    }

    pub fn set_app(&mut self, app: AppArgs) {
        self.app = app;
    }

    pub fn set_deprecated(&mut self, deprecated: DeprecatedArgs) {
        self.deprecated = deprecated;
    }

    // -------- SERDE/PICKLING support --------

    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        postcard::to_allocvec(self).pyerr()
    }

    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        postcard::from_bytes(&bytes).pyerr()
    }

    pub fn __reduce__(slf: Bound<'_, Self>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyTuple>)> {
        pickle_reduce(slf, "from_bytes", Self::to_bytes)
    }
}

/// Deprecated arguments go here.
///
/// # Important
///  - Make sure all are `hide = true` so it won't appear in `--help`
///  - Make sure all are [`Option<T>`] so them being missing doesn't cause issues
///  - Edit [`Self::get_error`] to handle changes of new/removed args in here
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct DeprecatedArgs {
    #[arg(long = "libp2p-port", hide = true)]
    #[pyo3(get, set)]
    pub libp2p_port: Option<u16>,
}

impl DeprecatedArgs {
    // TODO: actually run these at some point - maybe automatically..?
    pub fn get_error(&self) -> Option<clap::Error> {
        // destructure: don't change because this becomes compile error when new options are
        // moved into here or removed from here
        let Self { libp2p_port } = self.clone();

        if let Some(_) = libp2p_port {
            Some(clap::Error::raw(
                clap::error::ErrorKind::UnknownArgument,
                "The argument --libp2p-port is deprecated; use --zenoh-port instead",
            ))
        }
        // add more options here
        else {
            None
        }
    }
}

// pub mod cli_py {
//     use crate::ext::ResultExt;
//     use clap::{ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand};
//     use pyo3::pyclass::boolean_struct::False;
//     use pyo3::{Py, PyClass, PyClassInitializer, PyErr, PyResult, Python};
//     use std::ffi::OsString;
//     use std::fmt::{Debug, Display, Formatter};
//     use std::ops::{Deref, DerefMut};
//     use std::str::FromStr;
//
//     #[repr(transparent)]
//     pub struct CliPy<T>(Py<T>);
//
//     impl<T: Debug> Debug for CliPy<T>
//     where
//         T: PyClass,
//     {
//         #[inline(always)]
//         fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//             Python::attach(|py| (&*self.borrow(py)).fmt(f))
//         }
//     }
//
//     impl<T: Clone> Clone for CliPy<T> {
//         #[inline(always)]
//         fn clone(&self) -> Self {
//             Self::new(Python::attach(|py| self.clone_ref(py)))
//         }
//     }
//
//     impl<T: PartialEq> PartialEq for CliPy<T>
//     where
//         T: PyClass,
//     {
//         #[inline(always)]
//         fn eq(&self, other: &Self) -> bool {
//             Python::attach(|py| &*self.borrow(py) == &*other.borrow(py))
//         }
//     }
//
//     impl<T: Eq> Eq for CliPy<T> where T: PyClass {}
//
//     impl<T: Display> Display for CliPy<T>
//     where
//         T: PyClass,
//     {
//         #[inline(always)]
//         fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//             Python::attach(|py| (&*self.borrow(py)).fmt(f))
//         }
//     }
//
//     impl<T: FromStr> FromStr for CliPy<T>
//     where
//         T: PyClass + Into<PyClassInitializer<T>>,
//         T::Err: ToString,
//     {
//         type Err = PyErr;
//
//         #[inline(always)]
//         fn from_str(s: &str) -> Result<Self, Self::Err> {
//             Self::py_try_new(<T as FromStr>::from_str(s).pyerr()?)
//         }
//     }
//
//     impl<T> From<Py<T>> for CliPy<T> {
//         #[inline(always)]
//         fn from(inner: Py<T>) -> Self {
//             Self::new(inner)
//         }
//     }
//
//     impl<T> Deref for CliPy<T> {
//         type Target = Py<T>;
//         #[inline(always)]
//         fn deref(&self) -> &Self::Target {
//             self.inner()
//         }
//     }
//
//     impl<T> DerefMut for CliPy<T> {
//         #[inline(always)]
//         fn deref_mut(&mut self) -> &mut Self::Target {
//             self.inner_mut()
//         }
//     }
//
//     impl<T> CliPy<T> {
//         #[inline(always)]
//         fn new(inner: impl Into<Py<T>>) -> Self {
//             Self(inner.into())
//         }
//
//         #[inline(always)]
//         pub fn py_try_new_with(
//             py: Python<'_>,
//             value: impl Into<PyClassInitializer<T>>,
//         ) -> PyResult<Self>
//         where
//             T: PyClass,
//         {
//             Py::new(py, value).map(Self)
//         }
//
//         #[inline(always)]
//         pub fn py_try_new(value: impl Into<PyClassInitializer<T>>) -> PyResult<Self>
//         where
//             T: PyClass,
//         {
//             Python::attach(|py| Self::py_try_new_with(py, value))
//         }
//
//         #[inline(always)]
//         pub fn clap_try_new(value: impl Into<PyClassInitializer<T>>) -> Result<Self, clap::Error>
//         where
//             T: PyClass,
//         {
//             Self::py_try_new(value).map_err(|e| clap::Error::raw(clap::error::ErrorKind::Io, e))
//         }
//
//         #[inline(always)]
//         pub fn inner(&self) -> &Py<T> {
//             &self.0
//         }
//
//         #[inline(always)]
//         pub fn inner_mut(&mut self) -> &mut Py<T> {
//             &mut self.0
//         }
//
//         #[inline(always)]
//         pub fn into_inner(self) -> Py<T> {
//             self.0
//         }
//     }
//
//     impl<T: Parser> Parser for CliPy<T>
//     where
//         T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
//     {
//         fn try_parse() -> Result<Self, clap::Error> {
//             <T as Parser>::try_parse().and_then(Self::clap_try_new)
//         }
//
//         fn try_parse_from<I, It>(itr: I) -> Result<Self, clap::Error>
//         where
//             I: IntoIterator<Item = It>,
//             It: Into<OsString> + Clone,
//         {
//             <T as Parser>::try_parse_from(itr).and_then(Self::clap_try_new)
//         }
//     }
//
//     impl<T: CommandFactory> CommandFactory for CliPy<T> {
//         fn command() -> clap::Command {
//             <T as CommandFactory>::command()
//         }
//         fn command_for_update() -> clap::Command {
//             <T as CommandFactory>::command_for_update()
//         }
//     }
//
//     impl<T: FromArgMatches> FromArgMatches for CliPy<T>
//     where
//         T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
//     {
//         fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
//             <T as FromArgMatches>::from_arg_matches(matches).and_then(Self::clap_try_new)
//         }
//         fn from_arg_matches_mut(matches: &mut ArgMatches) -> Result<Self, clap::Error> {
//             <T as FromArgMatches>::from_arg_matches_mut(matches).and_then(Self::clap_try_new)
//         }
//         fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
//             Python::attach(|py| {
//                 <T as FromArgMatches>::update_from_arg_matches(&mut *self.borrow_mut(py), matches)
//             })
//         }
//         fn update_from_arg_matches_mut(
//             &mut self,
//             matches: &mut ArgMatches,
//         ) -> Result<(), clap::Error> {
//             Python::attach(|py| {
//                 <T as FromArgMatches>::update_from_arg_matches_mut(
//                     &mut *self.borrow_mut(py),
//                     matches,
//                 )
//             })
//         }
//     }
//
//     impl<T: Args> Args for CliPy<T>
//     where
//         T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
//     {
//         fn augment_args(cmd: clap::Command) -> clap::Command {
//             <T as Args>::augment_args(cmd)
//         }
//         fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
//             <T as Args>::augment_args_for_update(cmd)
//         }
//     }
//
//     impl<T: Subcommand> Subcommand for CliPy<T>
//     where
//         T: PyClass<Frozen = False> + Into<PyClassInitializer<T>>,
//     {
//         fn augment_subcommands(cmd: clap::Command) -> clap::Command {
//             <T as Subcommand>::augment_subcommands(cmd)
//         }
//         fn augment_subcommands_for_update(cmd: clap::Command) -> clap::Command {
//             <T as Subcommand>::augment_subcommands_for_update(cmd)
//         }
//         fn has_subcommand(name: &str) -> bool {
//             <T as Subcommand>::has_subcommand(name)
//         }
//     }
// }

pub fn cli_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CliArgs>()?;
    m.add_class::<DeprecatedArgs>()?;

    Ok(())
}
