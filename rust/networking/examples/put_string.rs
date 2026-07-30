use std::{env, time::Duration};

use env_logger::Env;
use log::info;
use networking;
use zenoh::Result;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::try_init_from_env(Env::new().default_filter_or("info")).expect("logger failed");
    let n_bytes = env::args()
        .nth(1)
        .and_then(|it| it.parse::<usize>().ok())
        .expect("USAGE: put_string <n> -- pub a string of n bytes into stream/data");
    info!("Opening session...");
    let cfg = networking::cfg(&format!("{:x}", rand::random::<u128>()), 52414)?;
    let session = networking::open(cfg, "exo", 52414, 52413).await?;
    let _tok = session
        .z
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.z.zid()))
        .await?;
    let key_expr = "stream/data";
    let payload = "n".repeat(n_bytes);

    let pubs = session
        .z
        .declare_publisher(key_expr)
        .congestion_control(zenoh::qos::CongestionControl::Block)
        .await?;
    let pubs_l = pubs.matching_listener().await?;
    if !pubs.matching_status().await?.matching() {
        while !pubs_l.recv_async().await?.matching() {}
    }

    tokio::time::sleep(Duration::from_secs(1)).await;
    info!("Putting Data ('{key_expr}': '{}')...", payload.len());
    for _ in 0..10 {
        let t = tokio::time::Instant::now();
        for _ in 0..5000 {
            pubs.put(payload.clone()).await?;
        }
        info!("{:?}", t.elapsed());
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    tokio::signal::ctrl_c().await?;
    Ok(())
}
