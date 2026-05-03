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
    let key_expr = "storage/mem1/name";
    let payload = "me";

    info!("Putting Data ('{key_expr}': '{payload}')...");
    session.put(key_expr, payload).await?;
    tokio::signal::ctrl_c().await?;
    Ok(())
}
