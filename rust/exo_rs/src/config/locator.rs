use crate::config::path::{parse_path, PathBufValueParserExt};
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::{pyclass, pymethods, Bound, PyResult};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{fs, io};
use util::path::PathExt;
use util::VecExt;

/// Arguments that are needed to resolve paths to files go here.
///
/// This is needed for such things as resolving the path of the configuration `.toml` file,
/// therefore any args here cannot be specified by the configuration `.toml` file.
///
/// By default, any path-like argument goes here, but can be moved to [`ConfigArgs`] if needed.
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
#[command(about = None, long_about = None)]
pub struct LocatorArgs {
    #[arg(
        long,
        env = "EXO_HOME",
        value_parser = parse_path().create_dir(),
        value_name = "PATH",
        help = "Path to Exo's home directory"
    )]
    #[pyo3(get)]
    pub exo_home: Option<PathBuf>,

    #[arg(
        long,
        env = "EXO_DEFAULT_MODELS_DIR",
        value_parser = parse_path().dir_exists(),
        value_name = "PATH",
        help = "Default models directory; always included as first entry in writable models directories"
    )]
    #[pyo3(get)]
    pub default_models_dir: Option<PathBuf>,

    #[arg(
        long,
        value_delimiter = ':',
        env = "EXO_MODELS_READ_ONLY_DIRS",
        value_parser = parse_path().dir_exists(),
        value_name = "PATHS",
        help = "Read-only model directories (colon-separated); never written to or deleted from"
    )]
    #[pyo3(get)]
    pub models_read_only_dirs: Option<Vec<PathBuf>>,

    #[arg(
        long,
        value_delimiter = ':',
        env = "EXO_MODELS_DIRS",
        value_parser = parse_path().dir_exists(),
        value_name = "PATHS",
        help = "Writable model directories (colon-separated); default directory is always prepended"
    )]
    #[pyo3(get)]
    pub models_dirs: Option<Vec<PathBuf>>,

    #[arg(
        long,
        env = "EXO_CONFIG_FILE",
        value_parser = parse_path().file_exists(),
        value_name = "PATH",
        help = "Path to Exo's .toml config file"
    )]
    #[pyo3(get)]
    pub config_file: Option<PathBuf>,
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocatorConfig {
    #[pyo3(get)]
    pub exo_home: ExoHome,

    #[pyo3(get)]
    pub models_dirs: ModelsDirs,

    #[pyo3(get)]
    pub log_files: LogFiles,

    // other
    #[pyo3(get)]
    pub pid_file: PathBuf,

    #[pyo3(get)]
    pub node_zid: PathBuf,

    #[pyo3(get)]
    pub config_file: PathBuf,

    #[pyo3(get)]
    pub custom_model_cards_dir: PathBuf,

    #[pyo3(get)]
    pub event_log_dir: PathBuf,
    #[pyo3(get)]
    pub image_cache_dir: PathBuf,
    #[pyo3(get)]
    pub tracing_cache_dir: PathBuf,
}

#[gen_stub_pymethods]
#[pymethods]
impl LocatorConfig {
    #[staticmethod]
    pub fn resolve(args: &LocatorArgs) -> PyResult<Self> {
        let exo_home = ExoHome::resolve(args)?;
        let models_dirs = ModelsDirs::resolve(args, &exo_home)?;
        let log_files = LogFiles::resolve(&exo_home)?;

        // PID file
        let pid_file = exo_home.cache.join("exo.pid");

        // Identity (config)
        let node_zid = exo_home.cache.join("node_zid");
        let config_file = args
            .config_file
            .clone()
            .unwrap_or_else(|| exo_home.config.join("config.toml"));
        config_file.create_file_if_not_found()?;

        // custom model card dirs TODO: see model_cards.py "todo"
        let custom_model_cards_dir = exo_home.data.join("custom_model_cards");

        let event_log_dir = exo_home.data.join("event_log");
        let image_cache_dir = exo_home.cache.join("images");
        let tracing_cache_dir = exo_home.cache.join("traces");

        Ok(Self {
            exo_home,
            models_dirs,
            log_files,
            pid_file,
            node_zid,
            config_file,
            custom_model_cards_dir,
            event_log_dir,
            image_cache_dir,
            tracing_cache_dir,
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExoHome {
    #[pyo3(get)]
    pub config: PathBuf,
    #[pyo3(get)]
    pub data: PathBuf,
    #[pyo3(get)]
    pub cache: PathBuf,
}

impl ExoHome {
    /// Get (and create if missing) the home directory for a specific purpose, with this precedence:
    ///  1. Prioritize `exo_home` if set
    ///  2. Fall back to "`<dir>`/exo" if specified; should be [XDG Directories] on Linux,
    ///     and [Standard Directories] on macOS
    ///  3. Fall back to "$HOME/.exo" if all else fails
    ///
    /// [XDG Directories]: https://specifications.freedesktop.org/basedir/latest/
    /// [Standard Directories]: https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html#//apple_ref/doc/uid/TP40010672-CH2-SW6
    fn get_home_dir(
        exo_home: &Option<PathBuf>,
        get_dir: impl FnOnce() -> Option<PathBuf>,
    ) -> io::Result<PathBuf> {
        // TODO: This splits the folders into separate ones in macOS **too**
        //       so we can have persistent IDs in cache folder and avoid the copy bug Evan mentioned
        //       BUT the user encountered the bug when he used "macOS time machine" or something
        //       so test that the "macOS time machine" doesn't copy the cache folder

        let home = exo_home
            .clone()
            .or_else(|| get_dir().map(|p| p.join("exo")))
            .or_else(|| dirs::home_dir().map(|p| p.join(".exo")))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "no home EXO home directory found: none specified, and $HOME directory doesn't exist",
                )
            })?;
        fs::create_dir_all(&home)?;
        Ok(home)
    }

    pub fn resolve(args: &LocatorArgs) -> io::Result<Self> {
        // create config/data/cache folders which the rest of the paths are derived from
        Ok(Self {
            config: Self::get_home_dir(&args.exo_home, dirs::config_dir)?,
            data: Self::get_home_dir(&args.exo_home, dirs::data_dir)?,
            cache: Self::get_home_dir(&args.exo_home, dirs::cache_dir)?,
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsDirs {
    #[pyo3(get)]
    pub default_models_dir: PathBuf,
    #[pyo3(get)]
    pub models_read_only_dirs: Vec<PathBuf>,
    #[pyo3(get)]
    pub models_dirs: Vec<PathBuf>,
}

impl ModelsDirs {
    pub fn resolve(args: &LocatorArgs, exo_home: &ExoHome) -> io::Result<Self> {
        // create default models dir
        let default_models_dir = args
            .default_models_dir
            .clone()
            .unwrap_or_else(|| exo_home.data.join("models"));
        fs::create_dir_all(&default_models_dir)?;

        // set of read-only directories
        let mut models_read_only_dirs = args.models_read_only_dirs.clone().unwrap_or_else(Vec::new);
        models_read_only_dirs.dedup_preserve_order();

        // set of mutable directories includes default directory and excludes read-only ones
        let mut models_dirs = vec![default_models_dir.clone()];
        if let Some(ref dirs) = args.models_dirs {
            models_dirs.extend(dirs.clone())
        };
        models_dirs.dedup_preserve_order();
        models_dirs.retain(|d| !models_read_only_dirs.contains(d));

        Ok(Self {
            default_models_dir,
            models_read_only_dirs,
            models_dirs,
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogFiles {
    #[pyo3(get)]
    pub exo_log_dir: PathBuf,
    #[pyo3(get)]
    pub exo_log: PathBuf,
    #[pyo3(get)]
    pub exo_runner_log_dir: PathBuf,
    #[pyo3(get)]
    pub exo_runner_stdout_log: PathBuf,
    #[pyo3(get)]
    pub exo_runner_stderr_log: PathBuf,
}

impl LogFiles {
    pub fn resolve(exo_home: &ExoHome) -> io::Result<Self> {
        // Exo log
        let exo_log_dir = exo_home.cache.join("exo_log");
        fs::create_dir_all(&exo_log_dir)?;
        let exo_log = exo_log_dir.join("exo.log");

        // Exo runner log
        let exo_runner_log_dir = exo_log_dir.join("runner_log");
        fs::create_dir_all(&exo_runner_log_dir)?;
        let exo_runner_stdout_log = exo_runner_log_dir.join("stdout.log");
        let exo_runner_stderr_log = exo_runner_log_dir.join("stderr.log");

        Ok(Self {
            exo_log_dir,
            exo_log,
            exo_runner_log_dir,
            exo_runner_stdout_log,
            exo_runner_stderr_log,
        })
    }
}

pub fn locator_submodule(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<LocatorArgs>()?;
    m.add_class::<LocatorConfig>()?;
    m.add_class::<ExoHome>()?;
    m.add_class::<ModelsDirs>()?;
    m.add_class::<LogFiles>()?;

    Ok(())
}
