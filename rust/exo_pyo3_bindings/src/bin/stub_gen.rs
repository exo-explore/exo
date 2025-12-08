use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .filter_or("RUST_LOG", "warning,exo_pyo3_bindings=info,networking=info"),
    )
    .init();
    let stub = exo_pyo3_bindings::stub_info()?;
    stub.generate()?;
    Ok(())
}
