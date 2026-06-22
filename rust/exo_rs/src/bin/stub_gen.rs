use clap::Parser;
use exo_rs::config::app::AppSettings;
use exo_rs::config::bootstrap::BootstrapSettings;
use exo_rs::config::cli::CliArgs;
use pyo3::Python;
use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    Python::initialize();

    Python::attach(|py| -> Result<()> {
        let a = CliArgs::parse();
        println!("{a:?}\n");
        let b = BootstrapSettings::resolve(py, &a.bootstrap.borrow(py))?;
        println!("{b:?}\n");
        let app = AppSettings::resolve(&a.app.borrow(py), &b)?;
        println!("{app:?}\n");
        Ok(())
    })?;

    // return Ok(());

    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
    let stub = exo_rs::stub_info()?;
    stub.generate()?;
    Ok(())
}
