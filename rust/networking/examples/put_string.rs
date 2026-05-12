use log::info;
use networking;
use zenoh::Result;

#[tokio::main]
async fn main() -> Result<()> {
    zenoh::init_log_from_env_or("info");
    info!("Opening session...");
    let cfg = networking::cfg(rand::random(), 52414)?;
    let session = networking::open(cfg, 52414).await?;
    let _tok = session
        .z
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.z.zid()))
        .await?;
    let key_expr = "storage/mem1/name";
    let payload = "me";

    info!("Putting Data ('{key_expr}': '{payload}')...");
    session.z.put(key_expr, payload).await?;
    tokio::signal::ctrl_c().await?;
    Ok(())
}
