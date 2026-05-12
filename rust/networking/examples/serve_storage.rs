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
    let _sub = session
        .z
        .liveliness()
        .declare_subscriber("nodes/*/live")
        .history(true)
        .callback(|tok| println!("{tok:?}"))
        .await?;

    let _sub2 = session
        .z
        .declare_subscriber("storage/mem1/**")
        .callback(|sample| {
            info!(
                "receiverd {} bytes on {}",
                sample.payload().len(),
                sample.key_expr()
            )
        });

    tokio::signal::ctrl_c().await?;
    Ok(())
}
