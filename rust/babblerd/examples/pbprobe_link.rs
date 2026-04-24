fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    babblerd::profiling::pbprobe::standalone::run_from_env()
}
