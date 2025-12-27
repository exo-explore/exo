use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = exo_pyo3_bindings::stub_info()?;
    stub.generate()?;
    Ok(())
}
