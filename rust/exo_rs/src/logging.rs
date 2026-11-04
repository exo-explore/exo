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
//!
//! What can't wait for Python doesn't: an error record is also written to stderr at once, and a
//! panic is written straight into exo's log file (see `write_panics_to`). At exit, the queue is
//! drained for at most `EXIT_DRAIN_LIMIT`.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_log::{Caching, Logger};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Records waiting for Python. A log call never waits: when this many are waiting, the record
/// is dropped (so the newest ones are lost) and counted, and the count is logged once the
/// waiting ones are out. A busy node logs tens of records a second, so this covers minutes of
/// Python holding the GIL.
const QUEUE_CAPACITY: usize = 16_384;

/// How long exo waits at exit for the records still queued to reach Python's logging. Through
/// exo's own log (a file and the console) records pass on at about 1,500 a second, so this lets
/// about 3,000 out: far more than a node logs while it shuts down, though not a full queue. A
/// quit is held up by this at most; a record still queued after it is lost (an error was
/// already written to stderr).
pub const EXIT_DRAIN_LIMIT: Duration = Duration::from_secs(2);

/// How often the exit drain looks whether the queue is empty
const EXIT_DRAIN_POLL: Duration = Duration::from_millis(10);

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

/// What the logger and the thread that passes records on share
#[derive(Default)]
struct Shared {
    /// Records queued and not yet passed on
    waiting: AtomicUsize,
    dropped: AtomicU64,
    /// When the first and the latest record since the last notice were dropped, in ms since the
    /// Unix epoch
    first_drop: AtomicU64,
    last_drop: AtomicU64,
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// Where panics are written, once exo has set up its log file
static PANIC_LOG: Mutex<Option<PathBuf>> = Mutex::new(None);

struct QueuedLogger {
    python: Arc<Logger>,
    queue: SyncSender<Queued>,
    shared: Arc<Shared>,
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
        let message = record.args().to_string();
        if let Some(line) = stderr_line(record.level(), record.target(), &message) {
            // An error is also written at once: the process may not live to pass it on
            _ = std::io::stderr().write_all(line.as_bytes());
        }
        let queued = Queued {
            level: record.level(),
            target: record.target().to_owned(),
            message,
            module_path: record.module_path().map(str::to_owned),
            file: record.file().map(str::to_owned),
            line: record.line(),
        };
        // Counted before it is sent, so the drain never sees it passed on before it is counted
        self.shared.waiting.fetch_add(1, Ordering::SeqCst);
        match self.queue.try_send(queued) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.shared.waiting.fetch_sub(1, Ordering::SeqCst);
                let now = epoch_millis(SystemTime::now());
                if self.shared.dropped.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.shared.first_drop.store(now, Ordering::SeqCst);
                }
                self.shared.last_drop.store(now, Ordering::SeqCst);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.shared.waiting.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    fn flush(&self) {}
}

/// Installs the logger and the panic hook. Call once, when the module is imported.
pub fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let python = Arc::new(Logger::new(py, Caching::LoggersAndLevels)?.filter(MAX_LEVEL));
    let (queue, records) = sync_channel(QUEUE_CAPACITY);
    let shared = Arc::clone(SHARED.get_or_init(Arc::default));
    std::thread::Builder::new()
        .name("exo_rs-log".to_owned())
        .spawn({
            let python = Arc::clone(&python);
            let shared = Arc::clone(&shared);
            move || pass_on(&python, &records, &shared)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("couldn't start the log thread: {e}")))?;
    if log::set_boxed_logger(Box::new(QueuedLogger {
        python,
        queue,
        shared,
    }))
    .is_ok()
    {
        log::set_max_level(MAX_LEVEL);
    }
    install_panic_hook();
    module.add_function(wrap_pyfunction!(flush_logs, module)?)?;
    module.add_function(wrap_pyfunction!(write_panics_to, module)?)?;
    // A last chance for the records still queued when Python exits (exo itself flushes them
    // before it closes its log, see exo.shared.logging)
    py.import("atexit")?
        .call_method1("register", (module.getattr("flush_logs")?,))?;
    Ok(())
}

/// Waits (without the GIL) until the Rust log records queued so far have reached Python's
/// logging, for at most `EXIT_DRAIN_LIMIT`. Returns whether they all did.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn flush_logs(py: Python<'_>) -> bool {
    let Some(shared) = SHARED.get() else {
        return true;
    };
    py.detach(|| {
        let deadline = Instant::now() + EXIT_DRAIN_LIMIT;
        while shared.waiting.load(Ordering::SeqCst) > 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(EXIT_DRAIN_POLL);
        }
        true
    })
}

/// Writes every Rust panic from now on straight into the log file at `path` too (it also goes
/// to stderr, as before): a panic can end the process before a queued record reaches Python.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn write_panics_to(path: String) {
    if let Ok(mut panic_log) = PANIC_LOG.lock() {
        *panic_log = Some(PathBuf::from(path));
    }
}

/// Passes each queued record on to Python (taking the GIL), and says when some were dropped
fn pass_on(python: &Logger, records: &Receiver<Queued>, shared: &Shared) {
    for record in records {
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
        // The drop notice goes after the records that were queued before the drops, which are
        // older than every dropped one
        if shared.waiting.fetch_sub(1, Ordering::SeqCst) == 1 {
            let lost = shared.dropped.swap(0, Ordering::SeqCst);
            if lost > 0 {
                let first = shared.first_drop.load(Ordering::SeqCst);
                let last = shared.last_drop.load(Ordering::SeqCst);
                python.log(
                    &Record::builder()
                        .level(Level::Warn)
                        .target("exo_rs")
                        .args(format_args!("{}", drop_notice(lost, first, last)))
                        .build(),
                );
            }
        }
    }
}

/// What to write to stderr at once for a record, if anything: errors only
fn stderr_line(level: Level, target: &str, message: &str) -> Option<String> {
    (level == Level::Error).then(|| {
        format!(
            "[ {} | ERROR    | {target} ] {message}\n",
            local_time(SystemTime::now())
        )
    })
}

fn drop_notice(lost: u64, first_ms: u64, last_ms: u64) -> String {
    format!(
        "{lost} log records from Rust were dropped between {} and {} while Python was busy \
         (the queue keeps the oldest {QUEUE_CAPACITY})",
        local_time(UNIX_EPOCH + Duration::from_millis(first_ms)),
        local_time(UNIX_EPOCH + Duration::from_millis(last_ms)),
    )
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        let Some(path) = PANIC_LOG.lock().ok().and_then(|path| path.clone()) else {
            return;
        };
        let thread = std::thread::current();
        let line = format!(
            "[ {} | CRITICAL | exo_rs:panic ] Rust panicked on thread {}: {info}\n{}\n",
            local_time(SystemTime::now()),
            thread.name().unwrap_or("unnamed"),
            std::backtrace::Backtrace::force_capture(),
        );
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            _ = file.write_all(line.as_bytes());
        }
    }));
}

fn epoch_millis(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

/// `time` as exo's log writes times: local, to the millisecond
fn local_time(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = libc::time_t::try_from(since.as_secs()).unwrap_or(0);
    // SAFETY: an all-zero `tm` is a valid value, and localtime_r only writes into it
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are to live values for the duration of the call
    if unsafe { libc::localtime_r(&raw const seconds, &raw mut tm) }.is_null() {
        return format!("{}.{:03}", since.as_secs(), since.subsec_millis());
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        since.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_errors_are_written_to_stderr_at_once() {
        assert!(stderr_line(Level::Warn, "networking", "a warning").is_none());
        let line = stderr_line(Level::Error, "networking::swarm", "it broke").expect("an error");
        assert!(
            line.contains("| ERROR    | networking::swarm ] it broke"),
            "{line}"
        );
    }

    #[test]
    fn the_drop_notice_says_how_many_and_when() {
        let first = epoch_millis(SystemTime::now());
        let notice = drop_notice(3_616, first, first + 1_500);
        assert!(notice.starts_with("3616 log records from Rust were dropped between "));
        assert!(notice.contains(&local_time(UNIX_EPOCH + Duration::from_millis(first))));
        assert!(notice.contains(&local_time(
            UNIX_EPOCH + Duration::from_millis(first + 1_500)
        )));
    }

    #[test]
    fn local_time_is_written_as_exos_log_writes_it() {
        let written = local_time(SystemTime::now());
        // YYYY-MM-DD HH:MM:SS.mmm
        assert_eq!(written.len(), 23, "{written}");
        assert_eq!(&written[4..5], "-");
        assert_eq!(&written[19..20], ".");
    }
}
