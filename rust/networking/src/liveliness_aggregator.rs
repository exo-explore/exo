use std::{collections::HashSet, sync::Arc};

use parking_lot::Mutex;
use zenoh::{Result, Session, Wait, sample::SampleKind};

pub fn spawn_liveliness_aggregator(session: &Session) -> Result<LivelinessAggregator> {
    let store = Arc::new(Mutex::new(HashSet::default()));
    session
        .liveliness()
        .declare_subscriber("live/*")
        .history(true)
        .callback({
            let store = Arc::clone(&store);
            move |sample| {
                let Some(nid) = sample
                    .key_expr()
                    .to_string()
                    .strip_prefix("live/")
                    .map(str::to_owned)
                else {
                    return;
                };
                let mut mg = store.lock();
                match sample.kind() {
                    SampleKind::Put => mg.insert(nid),
                    SampleKind::Delete => mg.remove(&nid),
                };
            }
        })
        .background()
        .wait()?;
    Ok(LivelinessAggregator { store })
}

#[derive(Clone)]
pub struct LivelinessAggregator {
    // need two arcs as the sub owns an arc to the store.
    store: Arc<Mutex<HashSet<String>>>,
}
impl LivelinessAggregator {
    pub fn dump(&self) -> HashSet<String> {
        self.store.lock().clone()
    }
}
