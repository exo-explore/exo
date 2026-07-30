#[cfg(test)]
mod tests {
    use core::mem::drop;
    use core::option::Option::Some;
    use core::time::Duration;
    use tokio;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_drop_channel() {
        struct Ping;

        let (tx, mut rx) = mpsc::channel::<Ping>(10);

        let _ = tokio::spawn(async move {
            println!("TASK: entered");

            loop {
                tokio::select! {
                    result = rx.recv() => {
                        match result {
                            Some(_) => {
                                println!("TASK: pinged");
                            }
                            None => {
                                println!("TASK: closing channel");
                                break;
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs_f32(0.1)) => {
                        println!("TASK: heartbeat");
                    }
                }
            }

            println!("TASK: exited");
        });

        let tx2 = tx.clone();

        tokio::time::sleep(Duration::from_secs_f32(0.11)).await;

        tx.send(Ping).await.expect("Should not fail");
        drop(tx);

        tokio::time::sleep(Duration::from_secs_f32(0.11)).await;

        tx2.send(Ping).await.expect("Should not fail");
        drop(tx2);

        tokio::time::sleep(Duration::from_secs_f32(0.11)).await;
    }
}
