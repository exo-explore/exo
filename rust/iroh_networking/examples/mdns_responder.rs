use iroh::{SecretKey, endpoint_info::EndpointIdExt};
use iroh_networking::ExoNet;
use n0_future::StreamExt;

// Launch a mock version of iroh for testing purposes
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let key = SecretKey::generate(&mut rand::rng());
    let dbg_key = key.public().to_z32();
    println!("Starting with pk: {dbg_key}");
    let net = ExoNet::init_iroh(key, "").await.unwrap();

    let mut conn_info = net.connection_info().await;

    let task = tokio::task::spawn(async move {
        println!("Inner task started!");
        loop {
            dbg!(conn_info.next().await);
        }
    });

    println!("Task started!");

    task.await.unwrap();
}
