use clap::Parser;
use exo_rs::config::CliArgs;
use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let a = CliArgs::parse();
    println!("{a:?}");
    println!("{:?}", a.verbosity);
    return Ok(());

    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
    let stub = exo_rs::stub_info()?;
    stub.generate()?;
    Ok(())
}
