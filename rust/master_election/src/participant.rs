use crate::communicator::Communicator;
use crate::ElectionMessage;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};

// trait ParticipantState {} // TODO: make sealed or something??
//
// struct Coordinator; // TODO: change to master
// struct Candidate; // i.e. election candidate
// struct Transient; // transient state, e.g. waiting for election results, declaring themselves winner, etc
// struct Follower; // i.e. a follower of an existing coordinator
//
// mod participant_impl {
//     use crate::participant::{Candidate, Coordinator, Follower, ParticipantState, Transient};
//
//     impl ParticipantState for Coordinator {}
//     impl ParticipantState for Candidate {}
//     impl ParticipantState for Transient {}
//     impl ParticipantState for Follower {}
// }

pub type ParticipantSelf = Arc<Mutex<Participant>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ParticipantId(pub u128);

#[derive(Debug, Clone, Copy)]
pub enum ParticipantState {
    Coordinator,                    // i.e. master
    ElectionCandidate, // after noticing a master went down, become candidate and `Election` message to all nodes higher than itself
    Waiting,           // when lower nodes are waiting for results of an election to conclude
    Follower { id: ParticipantId }, // when a participant is following a coordinator
    Transient,         // when the participant is in a neutral/uninitialized state
}

pub struct Participant {
    id: ParticipantId,
    state: ParticipantState,
    on_message_sent: Vec<Box<dyn FnOnce(ElectionMessage, ParticipantId)>>,
}

mod impls {
    use crate::participant::{Participant, ParticipantId, ParticipantSelf, ParticipantState};
    use crate::ElectionMessage;

    impl Participant {
        pub fn new_with(id: ParticipantId, state: ParticipantState) -> Self {
            Self {
                id,
                state,
                on_message_sent: vec![],
            }
        }

        pub fn add_on_message_sent<F>(&mut self, callback: F)
        where
            F: FnOnce(ElectionMessage, ParticipantId) + Send + 'static,
        {
            self.on_message_sent.push(Box::new(callback));
        }

        pub async fn receive_message(mut self_: ParticipantSelf, message: ElectionMessage) {
            let foo = self_.lock_owned().await;
        }
    }
}

pub const TASK_CHANNEL_SIZE: usize = 8;
pub const ELECTION_VICTORY_TIMEOUT: Duration = Duration::from_secs(1);
pub const VICTORY_WAITING_TIMEOUT: Duration = Duration::from_secs(1);
pub const HEARTBEAT_RECEIVE_TIMEOUT: Duration = Duration::from_secs(2);
pub const HEARTBEAT_SEND_TIMEOUT: Duration = Duration::from_secs(1);

pub enum InMessage {
    ElectionMessage(ElectionMessage),
    Heartbeat,
}

pub enum OutMessage {
    ElectionMessage(ElectionMessage),
    Heartbeat,
}

#[derive(Error, Debug)]
pub enum ParticipantError {
    #[error("could not send out-message: `{0}`")]
    SendError(#[from] mpsc::error::SendError<OutMessage>),
}

pub async fn participant_task<C: Communicator>(
    mut in_channel: mpsc::Receiver<InMessage>,
    out_channel: mpsc::Sender<OutMessage>,
    communicator: C,
) -> Result<(), ParticipantError> {
    // task state
    let participant_id: ParticipantId = ParticipantId(1234u128); // TODO: replace with dependency injection
    let mut participant_state: ParticipantState = ParticipantState::Transient;

    // TODO: slot this logic into this somewhere...
    //       4. If P receives an Election message from another process with a lower ID it sends an Answer message
    //       back and if it has not already started an election, it starts the election process at the beginning,
    //       by sending an Election message to higher-numbered processes.

    loop {
        match participant_state {
            ParticipantState::Transient => {
                // When a process P recovers from failure, or the failure detector indicates
                // that the current coordinator has failed, P performs the following actions:
                //
                // 1A) If P has the highest process ID, it sends a Victory message to all other
                // processes and becomes the new Coordinator.
                let max_id = communicator
                    .all_participants()
                    .iter()
                    .max()
                    .unwrap_or(&ParticipantId(0u128));
                if max_id <= &participant_id {
                    participant_state = ParticipantState::Coordinator;
                    communicator.broadcast_message(
                        ElectionMessage::Victory {
                            coordinator: participant_id,
                        },
                        communicator.all_participants(),
                    );
                    continue;
                }

                // 1B) Otherwise, P broadcasts an Election message to all other processes with
                // higher process IDs than itself
                participant_state = ParticipantState::ElectionCandidate;
                communicator.broadcast_message(
                    ElectionMessage::Election {
                        candidate: participant_id,
                    },
                    &communicator
                        .all_participants()
                        .iter()
                        .filter(|&p| p > &participant_id)
                        .copied()
                        .collect::<Vec<_>>(),
                );
            }
            ParticipantState::ElectionCandidate => {
                tokio::select! {
                    // 2. If P receives no Answer after sending an Election message, then it broadcasts
                    // a Victory message to all other processes and becomes the Coordinator.
                    _ = tokio::time::sleep(ELECTION_VICTORY_TIMEOUT) => {
                        participant_state = ParticipantState::Coordinator;
                        communicator.broadcast_message(
                            ElectionMessage::Victory {
                                coordinator: participant_id,
                            },
                            communicator.all_participants(),
                        );
                    }

                    // 3A. If P receives an Answer from a process with a higher ID, it sends no further
                    // messages for this election and waits for a Victory message. (If there is no Victory
                    // message after a period of time, it restarts the process at the beginning.)
                    Some(InMessage::ElectionMessage(ElectionMessage::Alive)) = in_channel.recv() => {
                        participant_state = ParticipantState::Waiting;
                    } // TODO: handle all other branches, e.g. channel closure, different messages & so on
                }
            }
            ParticipantState::Waiting => {
                tokio::select! {
                    // 3B. If there is no Victory message after a period of time, it restarts the process
                    // at the beginning.
                    _ = tokio::time::sleep(VICTORY_WAITING_TIMEOUT) => {
                        participant_state = ParticipantState::Transient;
                    }

                    // 5. If P receives a Victory message, it treats the sender as the coordinator.
                    Some(InMessage::ElectionMessage(ElectionMessage::Victory { coordinator })) = in_channel.recv() => {
                        participant_state = ParticipantState::Follower { id: coordinator };
                    } // TODO: handle all other branches, e.g. channel closure, different messages & so on
                }
            }
            ParticipantState::Follower { id: coordinator_id } => {
                tokio::select! {
                    // If we do not receive a heartbeat from the coordinator, trigger new election
                    _ = tokio::time::sleep(VICTORY_WAITING_TIMEOUT) => {
                        participant_state = ParticipantState::Transient;
                    }

                    // If we do receive a heartbeat - keep going
                    Some(InMessage::Heartbeat) = in_channel.recv() => {
                    } // TODO: handle all other branches, e.g. channel closure, different messages & so on
                }
            }
            ParticipantState::Coordinator => {
                // If we are coordinator - send heart beats
                {
                    out_channel.send(OutMessage::Heartbeat).await?;
                    tokio::time::sleep(HEARTBEAT_SEND_TIMEOUT).await;
                }
            }
        }
    }
}
