use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{fs, io};

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
        value_name = "PATH",
        help = "Path to Exo's home directory"
    )]
    #[pyo3(get)]
    pub exo_home: Option<PathBuf>,

    #[arg(
        long,
        env = "EXO_CONFIG_FILE",
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

    // other
    #[pyo3(get)]
    pub config_file: PathBuf,
}

impl LocatorConfig {
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
    ) -> PathBuf {
        // TODO: This splits the folders into seperate ones in macOS **too**
        //       so we can have persistent IDs in cache folder and avoid the copy bug Evan mentioned
        //       BUT the user encountered the bug when he used "macOS time machine" or something
        //       so test that the "macOS time machine" doesn't copy the cache folder

        // TODO: better error handling and bubbling up
        let home = exo_home
            .clone()
            .or_else(|| get_dir().map(|p| p.join("exo")))
            .or_else(|| dirs::home_dir().map(|p| p.join(".exo")))
            .expect("One of these should've succeeded: TODO better error handling");
        fs::create_dir_all(&home).expect("could not create home directory for whatever reason");
        home
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl LocatorConfig {
    #[staticmethod]
    pub fn resolve(args: &LocatorArgs) -> Self {
        // validate EXO_HOME if specified
        let exo_home = args
            .exo_home
            .clone()
            .map(|p| -> io::Result<PathBuf> {
                // TODO: better error propagation (we should bubble it up with result type)
                let p = p.canonicalize().expect("Path could not be canonicalized");
                return Ok(p);
            })
            .transpose()
            .expect("This is entirely a placeholder for error handling");

        // create config/data/cache folders which the rest of the paths are derived from
        let exo_home = ExoHome {
            config: Self::get_home_dir(&exo_home, dirs::config_dir),
            data: Self::get_home_dir(&exo_home, dirs::data_dir),
            cache: Self::get_home_dir(&exo_home, dirs::cache_dir),
        };

        // Log files (data/logs or cache)
        let exo_log_dir = exo_home.cache.join("exo_log");
        let exo_log = exo_log_dir.join("exo.log");
        let exo_runner_log_dir = exo_log_dir.join("runner_log");
        let exo_runner_stdout_log = exo_runner_log_dir.join("stdout.log");
        let exo_runner_stderr_log = exo_runner_log_dir.join("stderr.log");
        let exo_pid = exo_home.cache.join("exo.pid");

        // Identity (config)
        let exo_node_zid = exo_home.cache.join("node_zid");
        let exo_config_file = args
            .config_file
            .clone()
            .unwrap_or_else(|| exo_home.config.join("config.toml"));
        // TODO: check if exists (if it was arg) otherwise create if it doesn't exist (default location)

        Self {
            exo_home,
            config_file: exo_config_file,
        }
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDirs {
    #[pyo3(get)]
    pub default_models_dir: PathBuf,
    #[pyo3(get)]
    pub models_dirs: Vec<PathBuf>,
    #[pyo3(get)]
    pub models_read_only_dirs: Vec<PathBuf>,
}
