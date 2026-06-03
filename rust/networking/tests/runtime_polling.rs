use std::time::Duration;

use zenoh::Wait;

fn unique_key(name: &str) -> String {
    format!("test/zenoh-runtime-polling/{}/{}", std::process::id(), name)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_and_recv_work_on_tokio_baseline() {
    let session = zenoh::open(zenoh::Config::default())
        .await
        .expect("open session");

    let key = unique_key("tokio-baseline");
    let reply_key = key.clone();

    let _queryable = session
        .declare_queryable(key.clone())
        .callback(move |query| {
            query
                .reply(reply_key.clone(), "hello-from-queryable")
                .wait()
                .expect("reply from queryable");
        })
        .await
        .expect("declare queryable");

    let replies = session.get(key).await.expect("get");

    let reply = tokio::time::timeout(Duration::from_secs(5), replies.recv_async())
        .await
        .expect("timed out waiting for reply")
        .expect("reply channel closed");

    let sample = reply.result().expect("reply result was error");
    let payload = sample
        .payload()
        .try_to_string()
        .expect("payload should be utf8");

    assert_eq!(payload.as_ref(), "hello-from-queryable");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_and_recv_work_when_polled_by_smol_without_tokio_context() {
    let session = zenoh::open(zenoh::Config::default())
        .await
        .expect("open session under tokio");

    let key = unique_key("smol-no-tokio-context");
    let reply_key = key.clone();

    let _queryable = session
        .declare_queryable(key.clone())
        .callback(move |query| {
            query
                .reply(reply_key.clone(), "hello-from-queryable")
                .wait()
                .expect("reply from queryable");
        })
        .await
        .expect("declare queryable under tokio");

    let session_for_smol = session.clone();

    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        // This thread was not entered by Tokio.
        // If Zenoh's get/recv path requires an ambient Tokio Handle in the polling thread,
        // this is where it should panic, hang, or error.
        let result = {
            smol::block_on(async move {
                let replies = session_for_smol.get(key).await.expect("get under smol");

                let reply = replies.recv_async().await.expect("reply channel closed");

                let sample = reply.result().expect("reply result was error");
                let payload = sample
                    .payload()
                    .try_to_string()
                    .expect("payload should be utf8");

                payload.to_string()
            })
        };

        tx.send(result).expect("send test result");
    });

    let result = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("smol thread timed out; likely hung waiting for get/reply");

    let payload = result;

    assert_eq!(payload, "hello-from-queryable");
}
