use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ext::{ByteArrayExt as _, FutureExt, PyErrExt as _};
use crate::ext::{ResultExt as _, TokioMpscSenderExt as _};
use futures_lite::{Stream, StreamExt as _};
use networking::membership::{
    GRACE, Member, MemberState, Quorum, Reason, SPLIT_HOLD, STALL_THRESHOLD, Snapshot,
    WEDGE_TIMEOUT,
};
use networking::swarm::{FromExo, FromSwarm, Members, Swarm, ToSwarm, create_swarm};
use networking::{Session, is_valid_zid};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pymethods};
use pyo3_stub_gen::derive::{
    gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pyclass_enum, gen_stub_pymethods,
};
use tokio::sync::{Mutex, mpsc, oneshot};

#[gen_stub_pyclass]
#[pyclass(name = "NetworkingHandle")]
pub struct PyNetworkingHandle {
    // channels
    pub to_swarm: mpsc::Sender<ToSwarm>,
    pub swarm: Arc<Mutex<Pin<Box<dyn Stream<Item = FromSwarm> + Send>>>>,
    members: Members,
}

/// A node's state as this node sees it
#[gen_stub_pyclass_enum]
#[pyclass(name = "MemberState", eq, eq_int, frozen, hash, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyMemberState {
    Alive,
    /// Unreachable now; Gone if it stays so for the grace period
    Suspect,
    Gone,
    /// Only ever this node's own state, for a while after it was stopped (frozen or asleep)
    Resuming,
}

/// Why a node isn't Alive
#[gen_stub_pyclass_enum]
#[pyclass(name = "LossReason", eq, eq_int, frozen, hash, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyLossReason {
    /// Its token was deleted, or its connection closed and it didn't answer
    Disconnected,
    /// It said it was leaving
    Left,
    /// Its transport is up but exo on it stopped reporting
    Wedged,
}

/// How much of the cluster this node can reach
#[gen_stub_pyclass_enum]
#[pyclass(name = "Quorum", eq, eq_int, frozen, hash, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyQuorum {
    /// More than half of the nodes it knows
    Majority,
    /// Exactly half: two nodes with one away, or any even split
    Half,
    /// Less than half
    Minority,
}

impl From<Quorum> for PyQuorum {
    fn from(value: Quorum) -> Self {
        match value {
            Quorum::Majority => Self::Majority,
            Quorum::Half => Self::Half,
            Quorum::Minority => Self::Minority,
        }
    }
}

impl From<Reason> for PyLossReason {
    fn from(value: Reason) -> Self {
        match value {
            Reason::Disconnected => Self::Disconnected,
            Reason::Left => Self::Left,
            Reason::Wedged => Self::Wedged,
        }
    }
}

/// The state, and the reason for one that isn't Alive
fn split(state: MemberState) -> (PyMemberState, Option<PyLossReason>) {
    match state {
        MemberState::Alive => (PyMemberState::Alive, None),
        MemberState::Resuming => (PyMemberState::Resuming, None),
        MemberState::Suspect(reason) => (PyMemberState::Suspect, Some(reason.into())),
        MemberState::Gone(reason) => (PyMemberState::Gone, Some(reason.into())),
    }
}

/// Seconds since the Unix epoch
fn timestamp(time: SystemTime) -> f64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64())
}

/// A node and the state it has been in since `since` (seconds since the Unix epoch)
#[gen_stub_pyclass]
#[pyclass(name = "Member", frozen, get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMember {
    node: String,
    state: PyMemberState,
    reason: Option<PyLossReason>,
    since: f64,
    /// Whether it counts in the quorum read: a node Gone for a while (see FORGET_AFTER in the
    /// membership module), or that left, doesn't, though it stays in the view until removed
    counted: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMember {
    #[new]
    #[pyo3(signature = (node, state, reason, since, counted=true))]
    fn new(
        node: String,
        state: PyMemberState,
        reason: Option<PyLossReason>,
        since: f64,
        counted: bool,
    ) -> Self {
        Self {
            node,
            state,
            reason,
            since,
            counted,
        }
    }
}

impl From<Member> for PyMember {
    fn from(member: Member) -> Self {
        let (state, reason) = split(member.state);
        Self {
            node: member.node.to_string(),
            state,
            reason,
            since: timestamp(member.since),
            counted: member.counted,
        }
    }
}

/// Every node this node knows of, with how much of the cluster it reaches
#[gen_stub_pyclass]
#[pyclass(name = "MembershipSnapshot", frozen, get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMembershipSnapshot {
    /// This node first, then its peers
    members: Vec<PyMember>,
    quorum: PyQuorum,
    /// The Alive peers, plus this node unless it is Resuming
    reachable: usize,
    /// This node and every peer seen Alive, except those that left
    last_known: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMembershipSnapshot {
    #[new]
    #[pyo3(signature = (members, quorum, reachable, last_known))]
    fn new(members: Vec<PyMember>, quorum: PyQuorum, reachable: usize, last_known: usize) -> Self {
        Self {
            members,
            quorum,
            reachable,
            last_known,
        }
    }
}

impl From<Snapshot> for PyMembershipSnapshot {
    fn from(snapshot: Snapshot) -> Self {
        Self {
            members: snapshot.members.into_iter().map(PyMember::from).collect(),
            quorum: snapshot.quorum.into(),
            reachable: snapshot.reachable,
            last_known: snapshot.last_known,
        }
    }
}

#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "FromSwarm")]
pub enum PyFromSwarm {
    Message {
        topic: String,
        data: Py<PyBytes>,
    },
    /// A node's state changed, as this node sees it (this node's own state too, first)
    Membership {
        node: String,
        state: PyMemberState,
        reason: Option<PyLossReason>,
        /// Seconds since the Unix epoch
        since: f64,
    },
}
impl From<FromSwarm> for PyFromSwarm {
    fn from(value: FromSwarm) -> Self {
        match value {
            FromSwarm::Membership(member) => {
                let PyMember {
                    node,
                    state,
                    reason,
                    since,
                    ..
                } = member.into();
                Self::Membership {
                    node,
                    state,
                    reason,
                    since,
                }
            }
            FromSwarm::Message { topic, data } => Self::Message {
                topic: topic,
                data: data.pybytes(),
            },
        }
    }
}

impl PyNetworkingHandle {
    pub fn from_session(session: Session) -> Self {
        let (to_swarm, from_client) = mpsc::channel(1024);
        let swarm = Swarm::new(session, from_client);
        PyNetworkingHandle {
            members: swarm.members(),
            swarm: Arc::new(Mutex::new(swarm.into_stream())),
            to_swarm,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNetworkingHandle {
    // NOTE: `async fn`s here that use `.await` will wrap the future in `.allow_threads_py()`
    //       immediately beforehand to release the interpreter.
    //       SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await

    // ---- Lifecycle management methods ----

    #[staticmethod]
    pub fn new(
        py: Python<'_>,
        identity: &str,
        namespace: &str,
        listen_port: u16,
        discovery_service_port: u16,
    ) -> PyResult<PyNetworkingHandle> {
        // todo: zenoh self assigned peers
        if listen_port == 0 {
            todo!("cannot listen on port 0 yet");
        }
        // create communication channels
        let (to_swarm, from_client) = mpsc::channel(1024);

        // get identity
        if !is_valid_zid(identity) {
            return Err(PyValueError::new_err(format!(
                "{identity} is not a valid zenoh identity"
            )));
        }

        // create networking swarm (within tokio context!! or it crashes), without the GIL:
        // starting zenoh and discovery runs other threads, which may need the GIL to log
        let swarm = py
            .detach(|| {
                pyo3_async_runtimes::tokio::get_runtime().block_on(create_swarm(
                    identity,
                    namespace,
                    from_client,
                    listen_port,
                    discovery_service_port,
                ))
            })
            .pyerr()?;

        Ok(PyNetworkingHandle {
            members: swarm.members(),
            swarm: Arc::new(Mutex::new(swarm.into_stream())),
            to_swarm,
        })
    }

    /// This node's view of the cluster right now: every node it knows of, and the quorum read
    pub fn members(&self, py: Python<'_>) -> PyMembershipSnapshot {
        // Read without the GIL: the membership's own thread may hold the lock for a moment,
        // and must never wait for a thread that waits for it with the GIL held
        py.detach(|| self.members.snapshot()).into()
    }

    /// The coordinator removed `node` from the cluster: it stops counting in the quorum, unless
    /// this node still reaches it. Ignored for a node id that isn't valid.
    pub fn forget(&self, node: &str) {
        if let Ok(zid) = node.parse() {
            self.members.tell(FromExo::Removed(zid));
        }
    }

    #[gen_stub(override_return_type(
        type_repr="typing.Awaitable[FromSwarm]", imports=("typing")
    ))]
    pub fn recv<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let swarm = Arc::clone(&self.swarm);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            swarm
                .try_lock()
                .map_err(|_| PyRuntimeError::new_err("called recv twice concurrently"))?
                .next()
                .await
                .ok_or(PyErr::receiver_channel_closed())
                .map(PyFromSwarm::from)
        })
    }

    // ---- Gossipsub management methods ----

    /// Subscribe to a `GossipSub` topic.
    ///
    /// With `high_priority`, messages on this topic are sent ahead of normal traffic.
    ///
    /// Returns `True` if the subscription worked. Returns `False` if we were already subscribed.
    #[pyo3(signature = (topic, high_priority=false))]
    pub async fn gossipsub_subscribe(&self, topic: String, high_priority: bool) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        self.to_swarm
            .send_py(ToSwarm::Subscribe {
                topic,
                high_priority,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors
        rx.allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())?
            .pyerr()
    }

    /// Unsubscribes from a `GossipSub` topic.
    ///
    /// Returns `True` if we were subscribed to this topic. Returns `False` if we were not subscribed.
    pub async fn gossipsub_unsubscribe(&self, topic: String) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to unsubscribe
        self.to_swarm
            .send_py(ToSwarm::Unsubscribe {
                topic,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & convert any errors
        rx.allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())
    }

    /// Publishes a message with multiple topics to the `GossipSub` network.
    ///
    /// If no peers are found that subscribe to this topic, throws `NoPeersSubscribedToTopicError` exception.
    pub async fn gossipsub_publish(&self, topic: String, data: Py<PyBytes>) -> PyResult<()> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        let data = Python::attach(|py| Vec::from(data.as_bytes(py)));
        self.to_swarm
            .send_py(ToSwarm::Publish {
                topic,
                data,
                result_sender: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors => ignore messageID for now!!!
        let _ = rx
            .allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

// The membership's timers that exo's Python code acts on, in seconds (see
// rust/networking/src/membership.rs for what each is and why)
pyo3_stub_gen::module_variable!("exo_rs", "GRACE_SECONDS", f64);
pyo3_stub_gen::module_variable!("exo_rs", "WEDGE_TIMEOUT_SECONDS", f64);
pyo3_stub_gen::module_variable!("exo_rs", "SPLIT_HOLD_SECONDS", f64);
pyo3_stub_gen::module_variable!("exo_rs", "STALL_THRESHOLD_SECONDS", f64);

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("GRACE_SECONDS", GRACE.as_secs_f64())?;
    m.add("WEDGE_TIMEOUT_SECONDS", WEDGE_TIMEOUT.as_secs_f64())?;
    m.add("SPLIT_HOLD_SECONDS", SPLIT_HOLD.as_secs_f64())?;
    m.add("STALL_THRESHOLD_SECONDS", STALL_THRESHOLD.as_secs_f64())?;
    m.add_class::<PyNetworkingHandle>()?;
    m.add_class::<PyFromSwarm>()?;
    m.add_class::<PyMemberState>()?;
    m.add_class::<PyLossReason>()?;
    m.add_class::<PyQuorum>()?;
    m.add_class::<PyMember>()?;
    m.add_class::<PyMembershipSnapshot>()?;

    Ok(())
}
