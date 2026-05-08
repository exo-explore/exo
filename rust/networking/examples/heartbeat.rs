use env_logger::Env;
use log::info;
use networking;
use zenoh::{Result, Wait};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::try_init_from_env(Env::new().default_filter_or("info")).expect("logger failed");
    info!("Opening session...");
    let cfg = networking::cfg(&format!("{:x}", rand::random::<u128>()), 52414)?;
    let session = networking::open(cfg, "exo", 52414, 52413).await?;
    let _tok = session
        .z
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.z.zid()))
        .wait()?;
    session
        .z
        .liveliness()
        .declare_subscriber("**")
        .history(true)
        .callback(|tok| info!("{}: {}", tok.kind(), tok.key_expr().to_string()))
        .background()
        .wait()?;
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = session.z.put("hello", "world") => {},
        }
    }
    Ok(())
}
