//! Rust's log records reach Python's `logging` from a thread of their own.
//!
//! `pyo3_log` hands each record to Python's logging on the thread that logs, and Python's
//! logging needs the GIL. So every Rust thread that logs (zenoh's, whose `tracing` events
//! arrive here too, the membership's timer, discovery) waited for Python whenever a Python
//! thread held the GIL: a busy Python stalled networking, and a thread that logged while
//! holding a lock that a GIL-holding thread waited for stopped the process for good.
//!
//! `pyo3_log` has no setting that avoids the GIL for a record it passes on: its cache only
//! skips the GIL for levels Python ignores. So here a log call only queues the record, and one
//! thread passes the records on to `pyo3_log`, in order. Python's levels still filter as before.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_log::{Caching, Logger};

/// Records waiting for Python. A log call never waits: when this many are waiting, a record is
/// dropped and counted, and the count is logged when there is room again. A busy node logs
/// tens of records a second, so this covers minutes of Python holding the GIL.
const QUEUE_CAPACITY: usize = 16_384;

/// The most detailed level passed on, as `pyo3_log` does by default; Python's levels filter
/// further.
const MAX_LEVEL: LevelFilter = LevelFilter::Debug;

/// A record, owned so it can wait in the queue
struct Queued {
    level: Level,
    target: String,
    message: String,
    module_path: Option<String>,
    file: Option<String>,
    line: Option<u32>,
}

struct QueuedLogger {
    python: Arc<Logger>,
    queue: SyncSender<Queued>,
    dropped: Arc<AtomicU64>,
}

impl Log for QueuedLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        // pyo3_log answers this from its cache, without the GIL
        self.python.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let queued = Queued {
            level: record.level(),
            target: record.target().to_owned(),
            message: record.args().to_string(),
            module_path: record.module_path().map(str::to_owned),
            file: record.file().map(str::to_owned),
            line: record.line(),
        };
        if let Err(TrySendError::Full(_)) = self.queue.try_send(queued) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {}
}

/// Installs the logger. Call once, when the module is imported.
pub fn init(py: Python<'_>) -> PyResult<()> {
    let python = Arc::new(Logger::new(py, Caching::LoggersAndLevels)?.filter(MAX_LEVEL));
    let (queue, records) = sync_channel(QUEUE_CAPACITY);
    let dropped = Arc::new(AtomicU64::new(0));
    std::thread::Builder::new()
        .name("exo_rs-log".to_owned())
        .spawn({
            let python = Arc::clone(&python);
            let dropped = Arc::clone(&dropped);
            move || pass_on(&python, &records, &dropped)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("couldn't start the log thread: {e}")))?;
    if log::set_boxed_logger(Box::new(QueuedLogger {
        python,
        queue,
        dropped,
    }))
    .is_ok()
    {
        log::set_max_level(MAX_LEVEL);
    }
    Ok(())
}

/// Passes each queued record on to Python (taking the GIL), and says when some were dropped
fn pass_on(python: &Logger, records: &Receiver<Queued>, dropped: &AtomicU64) {
    for record in records {
        let lost = dropped.swap(0, Ordering::Relaxed);
        if lost > 0 {
            python.log(
                &Record::builder()
                    .level(Level::Warn)
                    .target("exo_rs")
                    .args(format_args!(
                        "{lost} log records from Rust were dropped while Python was busy"
                    ))
                    .build(),
            );
        }
        python.log(
            &Record::builder()
                .level(record.level)
                .target(&record.target)
                .module_path(record.module_path.as_deref())
                .file(record.file.as_deref())
                .line(record.line)
                .args(format_args!("{}", record.message))
                .build(),
        );
    }
}
