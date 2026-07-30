use std::time::Duration;

use env_logger::Env;
use log::info;
use networking;
use zenoh::Result;

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
        .await?;
    let _sub = session
        .z
        .liveliness()
        .declare_subscriber("nodes/*/live")
        .history(true)
        .callback(|tok| {
            info!(
                "{}: {}",
                tok.kind(),
                tok.key_expr()
                    .to_string()
                    .strip_prefix("nodes/")
                    .and_then(|it| it.strip_suffix("/live"))
                    .unwrap()
            )
        })
        .await?;

    let watch = async {
        for _ in 0..1000 {
            tokio::time::sleep(Duration::from_secs(1)).await;
            session
                .z
                .get("**")
                .callback(|reply| {
                    let sample = reply.into_result().expect("no errs");
                    info!(
                        "got {} bytes on {}",
                        sample.payload().len(),
                        sample.key_expr()
                    )
                })
                .await?;
        }
        Result::<()>::Ok(())
    };
    let subs = session.z.declare_subscriber("**").await?;

    let mut i = 0;
    let _a = async {
        while let Ok(sample) = subs.recv_async().await {
            i += 1;
            info!(
                "[{i}] received {} bytes on {}",
                sample.payload().len(),
                sample.key_expr()
            )
        }
    };
    tokio::select! {
        _ = watch => {},
        _ = _a => {},
        _ = tokio::signal::ctrl_c() => {},
    }
    Ok(())
}
