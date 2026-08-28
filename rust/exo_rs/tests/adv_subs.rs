use std::time::{Duration, SystemTime, UNIX_EPOCH};

use zenoh_ext::{
    AdvancedPublisherBuilderExt, AdvancedSubscriber, AdvancedSubscriberBuilderExt, CacheConfig,
    HistoryConfig, MissDetectionConfig,
};

use zenoh::{handlers::FifoChannelHandler, sample::Sample};

// Adjust these imports to your crate/module paths.
use exo_rs::{
    last_value::{LVPublisher, LVSubscriber},
    session::SessionHandle,
};
async fn expect_two_values(
    sub: &AdvancedSubscriber<FifoChannelHandler<Sample>>,
    key_a: &str,
    val_a: &str,
    key_b: &str,
    val_b: &str,
) {
    use std::collections::HashMap;
    use tokio::time::{Duration, Instant, timeout};
    use zenoh::sample::SampleKind;

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen: HashMap<String, String> = HashMap::new();

    loop {
        if seen.get(key_a).map(String::as_str) == Some(val_a)
            && seen.get(key_b).map(String::as_str) == Some(val_b)
        {
            return;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "timed out waiting for both historical samples; expected {key_a}={val_a}, {key_b}={val_b}; seen = {seen:?}"
        );

        match timeout(remaining.min(Duration::from_millis(750)), sub.recv_async()).await {
            Ok(Ok(sample)) => {
                if sample.kind() == SampleKind::Delete {
                    continue;
                }

                let key = sample.key_expr().to_string();
                let value = sample
                    .payload()
                    .try_to_string()
                    .expect("payload should be UTF-8")
                    .to_string();

                if key == key_a || key == key_b {
                    eprintln!("received relevant {key} = {value}");
                    seen.insert(key, value);
                } else {
                    eprintln!("received unrelated {key} = {value}");
                }
            }
            Ok(Err(e)) => panic!("subscriber receive failed: {e}"),
            Err(_) => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lv_subscriber_receives_last_value_from_multiple_publishers() {
    let cfg =
        networking::cfg(&format!("{:x}", rand::random::<u128>()), 52414).expect("create config");
    let n_session = networking::open(cfg, "exo", 52414, 52413)
        .await
        .expect("open session");

    let session = SessionHandle { session: n_session };

    let run_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let base = format!("zenoh_advanced_history_test/{run_id}");

    let key_a = format!("{base}/a");
    let key_b = format!("{base}/b");
    let sub_key = format!("{base}/*");

    let pub1: LVPublisher = session
        .last_value_publisher(key_a.clone())
        .expect("declare LV publisher a");

    pub1.state.put("aa").await.expect("publish aa");

    let pub2: LVPublisher = session
        .last_value_publisher(key_b.clone())
        .expect("declare LV publisher b");

    pub2.state.put("bb").await.expect("publish bb");

    // Let publisher detection / cache metadata settle before the late subscriber joins.
    tokio::time::sleep(Duration::from_millis(250)).await;

    let sub: LVSubscriber = session
        .last_value_subscriber(&*sub_key)
        .expect("declare LV subscriber");

    expect_two_values(&sub.subscriber, &*key_a, "aa", &*key_b, "bb").await
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wildcard_advanced_subscriber_receives_history_from_both_publishers() {
    let cfg =
        networking::cfg(&format!("{:x}", rand::random::<u128>()), 52412).expect("create config");
    let n_session = networking::open(cfg, "exo", 52412, 52411)
        .await
        .expect("open session");
    let session = n_session.z.clone();

    // Unique prefix so the wildcard subscriber cannot accidentally see unrelated traffic.
    let run_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let base = format!("zenoh_advanced_history_test/{run_id}");
    let key_a = format!("{base}/a");
    let key_b = format!("{base}/b");
    let sub_key = format!("{base}/*");

    let pub1 = session
        .declare_publisher(key_a.clone())
        .advanced()
        .publisher_detection()
        .sample_miss_detection(MissDetectionConfig::default())
        .cache(CacheConfig::default().max_samples(1))
        .await
        .expect("declare advanced publisher a");

    pub1.put("aa").await.expect("publish aa");

    let pub2 = session
        .declare_publisher(key_b.clone())
        .advanced()
        .sample_miss_detection(MissDetectionConfig::default())
        .publisher_detection()
        .cache(CacheConfig::default().max_samples(1))
        .await
        .expect("declare advanced publisher b");

    pub2.put("bb").await.expect("publish bb");

    // Give liveliness/cache declarations a brief chance to settle before declaring
    // the late-joining advanced subscriber.
    tokio::time::sleep(Duration::from_millis(250)).await;

    let sub = session
        .declare_subscriber(sub_key)
        .advanced()
        .history(
            HistoryConfig::default()
                .max_samples(1)
                .detect_late_publishers(),
        )
        .await
        .expect("declare advanced subscriber");

    expect_two_values(&sub, &*key_a, "aa", &*key_b, "bb").await
}
