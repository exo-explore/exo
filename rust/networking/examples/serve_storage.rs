use networking;
use tracing::info;
use zenoh::Result;

#[tokio::main]
async fn main() -> Result<()> {
    zenoh::init_log_from_env_or("info");
    info!("Opening session...");
    let cfg = networking::cfg(rand::random(), 0)?;
    let session = networking::open(cfg).await?;
    let _tok = session
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.zid()))
        .await?;
    let _sub = session
        .liveliness()
        .declare_subscriber("nodes/*/live")
        .history(true)
        .callback(|tok| println!("{tok:?}"))
        .await?;
    tokio::signal::ctrl_c().await?;
    Ok(())
}
