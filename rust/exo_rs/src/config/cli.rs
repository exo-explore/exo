use crate::config::app::AppArgs;
use crate::config::bootstrap::BootstrapArgs;
use crate::ext::ResultExt;
use crate::{pickle_reduce, version};
use clap::{ArgAction, Parser};
use pyo3::prelude::{PyAnyMethods, PyModuleMethods};
use pyo3::types::{PyModule, PyTuple};
use pyo3::{Bound, PyAny, PyResult, Python, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;

// re-export
pub use parser_impl::*;

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
    pub rejected: RejectedArgs,
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

    pub fn set_rejected(&mut self, rejected: RejectedArgs) {
        self.rejected = rejected;
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

/// Rejected arguments go here.
///
/// # Important
///  - Make sure all are `hide = true` so it won't appear in `--help`
///  - Make sure all are [`Option<T>`] so them being missing doesn't cause issues
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct RejectedArgs {
    // -------- temporarily unavailable --------
    #[arg(
        long,
        env = "EXO_BOOTSTRAP_PEERS",
        value_delimiter = ',',
        value_name = "MULTIADDRS",
        help = "Comma-separated libp2p multiaddrs to dial on startup",
        hide = true,
        value_parser = Rejected::<String>::unavailable(
            Some("--bootstrap-peers"), None, Some("EXO_BOOTSTRAP_PEERS"),
            "bootstrap peers are temporarily removed",
        )
    )]
    #[pyo3(get, set)]
    pub bootstrap_peers: Option<Vec<String>>,

    // -------- deprecated --------
    #[arg(
        long, value_name = "PORT", hide = true,
        value_parser = Rejected::<u16>::deprecated(
            Some("--libp2p-port"), None, None,
            Some("--zenoh-port"), None, None,
        )
    )]
    #[pyo3(get, set)]
    pub libp2p_port: Option<u16>,

    #[arg(
        env = "EXO_LIBP2P_NAMESPACE", value_name = "STRING", hide = true,
        value_parser = Rejected::<String>::deprecated(
            None, None, Some("EXO_LIBP2P_NAMESPACE"),
            Some("--namespace"), None, Some("EXO_NAMESPACE"),
        )
    )]
    #[pyo3(get, set)]
    pub libp2p_namespace: Option<String>,

    #[arg(
        env = "EXO_ENABLE_IMAGE_MODELS", value_name = "BOOL", hide = true,
        value_parser = Rejected::<bool>::deprecated(
            None, None, Some("EXO_ENABLE_IMAGE_MODELS"),
            Some("--enable-image-models"), None, Some("EXO_IMAGE_MODELS_ENABLED"),
        )
    )]
    #[pyo3(get, set)]
    pub enable_image_models: Option<bool>,

    #[arg(
        env = "ENABLE_DISAGGREGATION", value_name = "BOOL", hide = true,
        value_parser = Rejected::<bool>::deprecated(
            None, None, Some("ENABLE_DISAGGREGATION"),
            Some("--enable-disaggregation"), None, Some("EXO_DISAGGREGATION_ENABLED"),
        )
    )]
    #[pyo3(get, set)]
    pub enable_disaggregation: Option<bool>,
}

mod parser_impl {
    use clap::builder::TypedValueParser;
    use itertools::Itertools;
    use std::ffi::OsStr;
    use std::marker::PhantomData;

    #[derive(Clone)]
    pub struct Rejected<T> {
        message: String,
        _ty: PhantomData<T>,
    }

    impl<T> Rejected<T> {
        #[inline(always)]
        pub fn new(message: impl Into<String>) -> Self {
            let mut message = message.into();
            if !message.ends_with('\n') {
                message.push('\n');
            }
            Self {
                message,
                _ty: PhantomData,
            }
        }

        #[inline(always)]
        pub fn deprecated(
            old_long: Option<&str>,
            old_short: Option<&str>,
            old_env: Option<&str>,
            new_long: Option<&str>,
            new_short: Option<&str>,
            new_env: Option<&str>,
        ) -> Self {
            let old_names = vec![old_short, old_long, old_env]
                .into_iter()
                .flatten()
                .join("/");
            let new_names = vec![new_short, new_long, new_env]
                .into_iter()
                .flatten()
                .join("/");
            Self::new(format!(
                "the argument {old_names} is deprecated{}",
                if new_names.is_empty() {
                    String::new()
                } else {
                    format!("; use {new_names} instead")
                }
            ))
        }

        #[inline(always)]
        pub fn unavailable(
            long: Option<&str>,
            short: Option<&str>,
            env: Option<&str>,
            reason: impl AsRef<str>,
        ) -> Self {
            let names = vec![short, long, env].into_iter().flatten().join("/");
            Self::new(format!(
                "the argument {names} is unavailable: {}",
                reason.as_ref()
            ))
        }
    }

    impl<T> TypedValueParser for Rejected<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        type Value = T;
        fn parse_ref(
            &self,
            cmd: &clap::Command,
            _arg: Option<&clap::Arg>,
            _value: &OsStr,
        ) -> Result<Self::Value, clap::Error> {
            Err(clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                self.message.clone(),
            )
            .with_cmd(cmd))
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
    m.add_class::<RejectedArgs>()?;

    Ok(())
}
