#![allow(
    clippy::multiple_inherent_impl,
    clippy::unnecessary_wraps,
    clippy::unused_self,
    clippy::needless_pass_by_value
)]

use crate::r#const::MPSC_CHANNEL_SIZE;
use crate::ext::{ByteArrayExt as _, FutureExt, PyErrExt as _};
use crate::ext::{ResultExt as _, TokioMpscReceiverExt as _, TokioMpscSenderExt as _};
use crate::pyclass;
use crate::pylibp2p::ident::{PyKeypair, PyPeerId};
use libp2p::futures::StreamExt as _;
use libp2p::gossipsub::{IdentTopic, Message, MessageId, PublishError};
use libp2p::swarm::SwarmEvent;
use libp2p::{gossipsub, mdns};
use networking::discovery;
use networking::swarm::create_swarm;
use pyo3::prelude::{PyModule, PyModuleMethods as _};
use pyo3::types::PyBytes;
use pyo3::{Bound, Py, PyErr, PyResult, PyTraverseError, PyVisit, Python, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::net::IpAddr;
use tokio::sync::{Mutex, mpsc, oneshot};
use util::ext::VecExt as _;

mod exception {
    use pyo3::types::PyTuple;
    use pyo3::{PyErrArguments, exceptions::PyException, prelude::*};
    use pyo3_stub_gen::derive::*;

    #[gen_stub_pyclass]
    #[pyclass(frozen, extends=PyException, name="NoPeersSubscribedToTopicError")]
    pub struct PyNoPeersSubscribedToTopicError {}

    impl PyNoPeersSubscribedToTopicError {
        const MSG: &'static str = "\
        No peers are currently subscribed to receive messages on this topic. \
        Wait for peers to subscribe or check your network connectivity.";

        ///   Creates a new  [ `PyErr` ]  of this type.
        ///
        ///   [`PyErr`] :  https://docs.rs/pyo3/latest/pyo3/struct.PyErr.html   "PyErr in pyo3"
        pub(crate) fn new_err() -> PyErr {
            PyErr::new::<Self, _>(()) // TODO: check if this needs to be replaced???
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyNoPeersSubscribedToTopicError {
        #[new]
        #[pyo3(signature = (*args))]
        #[allow(unused_variables)]
        pub(crate) fn new(args: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __repr__(&self) -> String {
            format!("PeerId(\"{}\")", Self::MSG)
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(frozen, extends=PyException, name="AllQueuesFullError")]
    pub struct PyAllQueuesFullError {}

    impl PyAllQueuesFullError {
        const MSG: &'static str =
            "All libp2p peers are unresponsive, resend the message or reconnect.";

        ///   Creates a new  [ `PyErr` ]  of this type.
        ///
        ///   [`PyErr`] :  https://docs.rs/pyo3/latest/pyo3/struct.PyErr.html   "PyErr in pyo3"
        pub(crate) fn new_err() -> PyErr {
            PyErr::new::<Self, _>(()) // TODO: check if this needs to be replaced???
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyAllQueuesFullError {
        #[new]
        #[pyo3(signature = (*args))]
        #[allow(unused_variables)]
        pub(crate) fn new(args: &Bound<'_, PyTuple>) -> Self {
            Self {}
        }

        fn __repr__(&self) -> String {
            format!("PeerId(\"{}\")", Self::MSG)
        }

        fn __str__(&self) -> String {
            Self::MSG.to_string()
        }
    }
}

/// Connection or disconnection event discriminant type.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, name = "ConnectionUpdateType")]
#[derive(Debug, Clone, PartialEq)]
enum PyConnectionUpdateType {
    Connected = 0,
    Disconnected,
}

#[gen_stub_pyclass]
#[pyclass(frozen, name = "ConnectionUpdate")]
#[derive(Debug, Clone)]
struct PyConnectionUpdate {
    /// Whether this is a connection or disconnection event
    #[pyo3(get)]
    update_type: PyConnectionUpdateType,

    /// Identity of the peer that we have connected to or disconnected from.
    #[pyo3(get)]
    peer_id: PyPeerId,

    /// Remote connection's IPv4 address.
    #[pyo3(get)]
    remote_ipv4: String,

    /// Remote connection's TCP port.
    #[pyo3(get)]
    remote_tcp_port: u16,
}

enum ToTask {
    GossipsubSubscribe {
        topic: String,
        result_tx: oneshot::Sender<PyResult<bool>>,
    },
    GossipsubUnsubscribe {
        topic: String,
        result_tx: oneshot::Sender<bool>,
    },
    GossipsubPublish {
        topic: String,
        data: Vec<u8>,
        result_tx: oneshot::Sender<PyResult<MessageId>>,
    },
}

#[allow(clippy::enum_glob_use)]
async fn networking_task(
    mut swarm: networking::swarm::Swarm,
    mut to_task_rx: mpsc::Receiver<ToTask>,
    connection_update_tx: mpsc::Sender<PyConnectionUpdate>,
    gossipsub_message_tx: mpsc::Sender<(String, Vec<u8>)>,
) {
    use SwarmEvent::*;
    use ToTask::*;
    use mdns::Event::*;
    use networking::swarm::BehaviourEvent::*;

    log::info!("RUST: networking task started");

    loop {
        tokio::select! {
            message = to_task_rx.recv() => {
                // handle closed channel
                let Some(message) = message else {
                    log::info!("RUST: channel closed");
                    break;
                };

                // dispatch incoming messages
                match message {
                    GossipsubSubscribe { topic, result_tx } => {
                        // try to subscribe
                        let result = swarm.behaviour_mut()
                            .gossipsub.subscribe(&IdentTopic::new(topic));

                        // send response oneshot
                        if let Err(e) = result_tx.send(result.pyerr()) {
                            log::error!("RUST: could not subscribe to gossipsub topic since channel already closed: {e:?}");
                            continue;
                        }
                    }
                    GossipsubUnsubscribe { topic, result_tx } => {
                        // try to unsubscribe from the topic
                        let result = swarm.behaviour_mut()
                            .gossipsub.unsubscribe(&IdentTopic::new(topic));

                        // send response oneshot (or exit if connection closed)
                        if let Err(e) = result_tx.send(result) {
                            log::error!("RUST: could not unsubscribe from gossipsub topic since channel already closed: {e:?}");
                            continue;
                        }
                    }
                    GossipsubPublish { topic, data, result_tx } => {
                        // try to publish the data -> catch NoPeersSubscribedToTopic error & convert to correct exception
                        let result = swarm.behaviour_mut().gossipsub.publish(
                            IdentTopic::new(topic), data);
                        let pyresult: PyResult<MessageId> = if let Err(PublishError::NoPeersSubscribedToTopic) = result {
                            Err(exception::PyNoPeersSubscribedToTopicError::new_err())
                        } else if let Err(PublishError::AllQueuesFull(_)) = result {
                            Err(exception::PyAllQueuesFullError::new_err())
                        } else {
                            result.pyerr()
                        };

                        // send response oneshot (or exit if connection closed)
                        if let Err(e) = result_tx.send(pyresult) {
                            log::error!("RUST: could not publish gossipsub message since channel already closed: {e:?}");
                            continue;
                        }
                    }
                }
            }

            // architectural solution to this problem:
            // create keep_alive behavior who's job it is to dial peers discovered by mDNS (and drop when expired)
            //   -> it will emmit TRUE connected/disconnected events consumable elsewhere
            //
            // gossipsub will feed off-of dial attempts created by networking, and that will bootstrap its' peers list
            // then for actual communication it will dial those peers if need-be
            swarm_event = swarm.select_next_some() => {
                match swarm_event {
                    Behaviour(Gossipsub(gossipsub::Event::Message {
                        message: Message {
                            topic,
                            data,
                            ..
                        },
                        ..
                    })) => {
                        // topic-ID is just the topic hash!!! (since we used identity hasher)
                        let message = (topic.into_string(), data);

                        // send incoming message to channel (or exit if connection closed)
                        if let Err(e) = gossipsub_message_tx.send(message).await {
                            log::error!("RUST: could not send incoming gossipsub message since channel already closed: {e}");
                            continue;
                        }
                    },
                    Behaviour(Discovery(discovery::Event::ConnectionEstablished { peer_id, remote_ip, remote_tcp_port, .. })) => {
                        // grab IPv4 string
                        let remote_ipv4 = match remote_ip {
                            IpAddr::V4(ip) => ip.to_string(),
                            IpAddr::V6(ip) => {
                                log::warn!("RUST: ignoring connection to IPv6 address: {ip}");
                                continue;
                            }
                        };

                        // send connection event to channel (or exit if connection closed)
                        if let Err(e) = connection_update_tx.send(PyConnectionUpdate {
                            update_type: PyConnectionUpdateType::Connected,
                            peer_id: PyPeerId(peer_id),
                            remote_ipv4,
                            remote_tcp_port,
                        }).await {
                            log::error!("RUST: could not send connection update since channel already closed: {e}");
                            continue;
                        }
                    },
                    Behaviour(Discovery(discovery::Event::ConnectionClosed { peer_id, remote_ip, remote_tcp_port, .. })) => {
                        // grab IPv4 string
                        let remote_ipv4 = match remote_ip {
                            IpAddr::V4(ip) => ip.to_string(),
                            IpAddr::V6(ip) => {
                                log::warn!("RUST: ignoring disconnection from IPv6 address: {ip}");
                                continue;
                            }
                        };

                        // send disconnection event to channel (or exit if connection closed)
                        if let Err(e) = connection_update_tx.send(PyConnectionUpdate {
                            update_type: PyConnectionUpdateType::Disconnected,
                            peer_id: PyPeerId(peer_id),
                            remote_ipv4,
                            remote_tcp_port,
                        }).await {
                            log::error!("RUST: could not send connection update since channel already closed: {e}");
                            continue;
                        }
                    },
                    e => {
                        log::info!("RUST: other event {e:?}");
                    }
                }
            }
        }
    }

    log::info!("RUST: networking task stopped");
}

#[gen_stub_pyclass]
#[pyclass(name = "NetworkingHandle")]
#[derive(Debug)]
struct PyNetworkingHandle {
    // channels
    to_task_tx: Option<mpsc::Sender<ToTask>>,
    connection_update_rx: Mutex<mpsc::Receiver<PyConnectionUpdate>>,
    gossipsub_message_rx: Mutex<mpsc::Receiver<(String, Vec<u8>)>>,
}

impl Drop for PyNetworkingHandle {
    fn drop(&mut self) {
        // TODO: may or may not need to await a "kill-signal" oneshot channel message,
        //       to ensure that the networking task is done BEFORE exiting the clear function...
        //       but this may require GIL?? and it may not be safe to call GIL here??
        self.to_task_tx = None; // Using Option<T> as a trick to force channel to be dropped
    }
}

#[allow(clippy::expect_used)]
impl PyNetworkingHandle {
    fn new(
        to_task_tx: mpsc::Sender<ToTask>,
        connection_update_rx: mpsc::Receiver<PyConnectionUpdate>,
        gossipsub_message_rx: mpsc::Receiver<(String, Vec<u8>)>,
    ) -> Self {
        Self {
            to_task_tx: Some(to_task_tx),
            connection_update_rx: Mutex::new(connection_update_rx),
            gossipsub_message_rx: Mutex::new(gossipsub_message_rx),
        }
    }

    const fn to_task_tx(&self) -> &mpsc::Sender<ToTask> {
        self.to_task_tx
            .as_ref()
            .expect("The sender should only be None after de-initialization.")
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNetworkingHandle {
    // NOTE: `async fn`s here that use `.await` will wrap the future in `.allow_threads_py()`
    //       immediately beforehand to release the interpreter.
    //       SEE: https://pyo3.rs/v0.26.0/async-await.html#detaching-from-the-interpreter-across-await

    // ---- Lifecycle management methods ----

    #[new]
    fn py_new(identity: Bound<'_, PyKeypair>) -> PyResult<Self> {
        use pyo3_async_runtimes::tokio::get_runtime;

        // create communication channels
        let (to_task_tx, to_task_rx) = mpsc::channel(MPSC_CHANNEL_SIZE);
        let (connection_update_tx, connection_update_rx) = mpsc::channel(MPSC_CHANNEL_SIZE);
        let (gossipsub_message_tx, gossipsub_message_rx) = mpsc::channel(MPSC_CHANNEL_SIZE);

        // get identity
        let identity = identity.borrow().0.clone();

        // create networking swarm (within tokio context!! or it crashes)
        let swarm = get_runtime()
            .block_on(async { create_swarm(identity) })
            .pyerr()?;

        // spawn tokio task running the networking logic
        get_runtime().spawn(async move {
            networking_task(
                swarm,
                to_task_rx,
                connection_update_tx,
                gossipsub_message_tx,
            )
            .await;
        });
        Ok(Self::new(
            to_task_tx,
            connection_update_rx,
            gossipsub_message_rx,
        ))
    }

    #[gen_stub(skip)]
    const fn __traverse__(&self, _visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(()) // This is needed purely so `__clear__` can work
    }

    #[gen_stub(skip)]
    fn __clear__(&mut self) {
        // TODO: may or may not need to await a "kill-signal" oneshot channel message,
        //       to ensure that the networking task is done BEFORE exiting the clear function...
        //       but this may require GIL?? and it may not be safe to call GIL here??
        self.to_task_tx = None; // Using Option<T> as a trick to force channel to be dropped
    }

    // ---- Connection update receiver methods ----

    /// Receives the next `ConnectionUpdate` from networking.
    async fn connection_update_recv(&self) -> PyResult<PyConnectionUpdate> {
        self.connection_update_rx
            .lock()
            .allow_threads_py() // allow-threads-aware async call
            .await
            .recv_py()
            .allow_threads_py() // allow-threads-aware async call
            .await
    }

    /// Receives at most `limit` `ConnectionUpdate`s from networking and returns them.
    ///
    /// For `limit = 0`, an empty collection of `ConnectionUpdate`s will be returned immediately.
    /// For `limit > 0`, if there are no `ConnectionUpdate`s in the channel's queue this method
    /// will sleep until a `ConnectionUpdate`s is sent.
    async fn connection_update_recv_many(&self, limit: usize) -> PyResult<Vec<PyConnectionUpdate>> {
        self.connection_update_rx
            .lock()
            .allow_threads_py() // allow-threads-aware async call
            .await
            .recv_many_py(limit)
            .allow_threads_py() // allow-threads-aware async call
            .await
    }

    // TODO: rn this blocks main thread if anything else is awaiting the channel (bc its a mutex)
    //       so its too dangerous to expose just yet. figure out a better semantics for handling this,
    //       so things don't randomly block
    // /// Tries to receive the next `ConnectionUpdate` from networking.
    // fn connection_update_try_recv(&self) -> PyResult<Option<PyConnectionUpdate>> {
    //     self.connection_update_rx.blocking_lock().try_recv_py()
    // }
    //
    // /// Checks if the `ConnectionUpdate` channel is empty.
    // fn connection_update_is_empty(&self) -> bool {
    //     self.connection_update_rx.blocking_lock().is_empty()
    // }
    //
    // /// Returns the number of `ConnectionUpdate`s in the channel.
    // fn connection_update_len(&self) -> usize {
    //     self.connection_update_rx.blocking_lock().len()
    // }

    // ---- Gossipsub management methods ----

    /// Subscribe to a `GossipSub` topic.
    ///
    /// Returns `True` if the subscription worked. Returns `False` if we were already subscribed.
    async fn gossipsub_subscribe(&self, topic: String) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        self.to_task_tx()
            .send_py(ToTask::GossipsubSubscribe {
                topic,
                result_tx: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors
        rx.allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())?
    }

    /// Unsubscribes from a `GossipSub` topic.
    ///
    /// Returns `True` if we were subscribed to this topic. Returns `False` if we were not subscribed.
    async fn gossipsub_unsubscribe(&self, topic: String) -> PyResult<bool> {
        let (tx, rx) = oneshot::channel();

        // send off request to unsubscribe
        self.to_task_tx()
            .send_py(ToTask::GossipsubUnsubscribe {
                topic,
                result_tx: tx,
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
    async fn gossipsub_publish(&self, topic: String, data: Py<PyBytes>) -> PyResult<()> {
        let (tx, rx) = oneshot::channel();

        // send off request to subscribe
        let data = Python::with_gil(|py| Vec::from(data.as_bytes(py)));
        self.to_task_tx()
            .send_py(ToTask::GossipsubPublish {
                topic,
                data,
                result_tx: tx,
            })
            .allow_threads_py() // allow-threads-aware async call
            .await?;

        // wait for response & return any errors => ignore messageID for now!!!
        let _ = rx
            .allow_threads_py() // allow-threads-aware async call
            .await
            .map_err(|_| PyErr::receiver_channel_closed())??;
        Ok(())
    }

    // ---- Gossipsub message receiver methods ----

    /// Receives the next message from the `GossipSub` network.
    async fn gossipsub_recv(&self) -> PyResult<(String, Py<PyBytes>)> {
        self.gossipsub_message_rx
            .lock()
            .allow_threads_py() // allow-threads-aware async call
            .await
            .recv_py()
            .allow_threads_py() // allow-threads-aware async call
            .await
            .map(|(t, d)| (t, d.pybytes()))
    }

    /// Receives at most `limit` messages from the `GossipSub` network and returns them.
    ///
    /// For `limit = 0`, an empty collection of messages will be returned immediately.
    /// For `limit > 0`, if there are no messages in the channel's queue this method
    /// will sleep until a message is sent.
    async fn gossipsub_recv_many(&self, limit: usize) -> PyResult<Vec<(String, Py<PyBytes>)>> {
        Ok(self
            .gossipsub_message_rx
            .lock()
            .allow_threads_py() // allow-threads-aware async call
            .await
            .recv_many_py(limit)
            .allow_threads_py() // allow-threads-aware async call
            .await?
            .map(|(t, d)| (t, d.pybytes())))
    }

    // TODO: rn this blocks main thread if anything else is awaiting the channel (bc its a mutex)
    //       so its too dangerous to expose just yet. figure out a better semantics for handling this,
    //       so things don't randomly block
    // /// Tries to receive the next message from the `GossipSub` network.
    // fn gossipsub_try_recv(&self) -> PyResult<Option<(String, Py<PyBytes>)>> {
    //     Ok(self
    //         .gossipsub_message_rx
    //         .blocking_lock()
    //         .try_recv_py()?
    //         .map(|(t, d)| (t, d.pybytes())))
    // }
    //
    // /// Checks if the `GossipSub` message channel is empty.
    // fn gossipsub_is_empty(&self) -> bool {
    //     self.gossipsub_message_rx.blocking_lock().is_empty()
    // }
    //
    // /// Returns the number of `GossipSub` messages in the channel.
    // fn gossipsub_len(&self) -> usize {
    //     self.gossipsub_message_rx.blocking_lock().len()
    // }
}

pub fn networking_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<exception::PyNoPeersSubscribedToTopicError>()?;
    m.add_class::<exception::PyAllQueuesFullError>()?;

    m.add_class::<PyConnectionUpdateType>()?;
    m.add_class::<PyConnectionUpdate>()?;
    m.add_class::<PyConnectionUpdateType>()?;
    m.add_class::<PyNetworkingHandle>()?;

    Ok(())
}
