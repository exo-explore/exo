use std::{borrow::Cow, env};

use env_logger::Env;
use log::{info, warn};
use networking;
use zenoh::{Result, Wait};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::try_init_from_env(Env::new().default_filter_or("info")).expect("logger failed");
    info!("Opening session...");
    let cfg = networking::cfg(rand::random(), 52414)?;
    let session = networking::open(cfg, 52414).await?;
    let other_live = session
        .z
        .liveliness()
        .declare_subscriber("**")
        .history(true)
        .wait()?;
    _ = other_live.recv_async().await?;
    let other_live = session.z.liveliness().get("**").wait()?;
    while let Ok(s) = other_live.recv_async().await {
        info!("{s:?}");
    }
    let query = env::args().nth(1).expect("USAGE: z_get [query]");
    info!("Querying {query}");
    let subs = session.z.liveliness().get(query).await?;
    while let Ok(r) = subs.recv_async().await {
        match r.into_result() {
            Ok(s) => info!(
                "{}: {}",
                s.key_expr(),
                s.payload()
                    .try_to_string()
                    .unwrap_or_else(|_| Cow::Borrowed("-bytes-"))
            ),
            Err(e) => warn!("{e}"),
        }
    }
    Ok(())
}
