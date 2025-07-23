//! Communicator is an abstraction that allows me to "mock" speaking to the network
//!

use crate::participant::{Participant, ParticipantId};
use crate::ElectionMessage;

pub trait Communicator {
    fn all_participants(&self) -> &[ParticipantId];
    fn broadcast_message(&self, message: ElectionMessage, recipients: &[ParticipantId]) -> ();
    fn register_participant(&mut self, participant: &Participant) -> ParticipantId;
}

mod communicator_impls {
    macro_rules! as_ref_impl {
        () => {
            #[inline]
            fn all_participants(&self) -> &[ParticipantId] {
                self.as_ref().all_participants()
            }

            #[inline]
            fn broadcast_message(&self, message: Message, recipients: &[ParticipantId]) {
                self.as_ref().broadcast_message(message, recipients);
            }
        };
    }

    // impl<C: Communicator> Communicator for Box<C> {
    //     as_ref_impl!();
    // }
    //
    // impl<C: Communicator> Communicator for Arc<C> {
    //     as_ref_impl!();
    // }
}
