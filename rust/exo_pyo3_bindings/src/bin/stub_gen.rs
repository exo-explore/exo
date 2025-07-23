use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let body = async {
        env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info"))
            .init();
        let stub = exo_pyo3_bindings::stub_info()?;
        stub.generate()?;
        Ok(())
    };
    #[allow(
        clippy::expect_used,
        clippy::diverging_sub_expression,
        clippy::needless_return
    )]
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed building the Runtime");

        let a = runtime.handle();

        return runtime.block_on(body);
    }
}

// fn main() -> Result<()> {
//     let stub = python_bindings::stub_info()?;
//     stub.generate()?;
//     Ok(())
// }
