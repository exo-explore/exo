use networking;
use tracing::{info, warn};
use zenoh::{Result, Wait};

#[tokio::main]
async fn main() -> Result<()> {
    zenoh::init_log_from_env_or("info");
    info!("Opening session...");
    let cfg = networking::cfg(&format!("{:x}", rand::random::<u128>()), 52414)?;
    // Their own cluster name: exo nodes on the network don't see them as another version
    let session = networking::open(
        cfg,
        "exo-networking-example",
        0,
        "example",
        None,
        52414,
        52413,
    )
    .await?;
    let _tok = session
        .z
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.z.zid()))
        .wait()?;
    let subs = session
        .z
        .liveliness()
        .declare_subscriber("**")
        .history(true)
        .wait()?;
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            s = subs.recv_async() => {
                match s {
                    Err(e) => warn!("{e}"),
                    Ok(s) => info!("{}: {}", s.kind(), s.key_expr().to_string().split("/").nth(1).unwrap()),
                }
            }
        }
    }
    Ok(())
}
