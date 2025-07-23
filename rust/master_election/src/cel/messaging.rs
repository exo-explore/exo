use crate::cel::messaging::data::Probability;
use crate::cel::KnowledgeMessage;

mod data {
    use ordered_float::OrderedFloat;
    use thiserror::Error;

    #[derive(Error, Debug, Copy, Clone, PartialEq, PartialOrd)]
    #[error("Floating number `{0}` is not a probability")]
    #[repr(transparent)]
    pub struct NotProbabilityError(f64);

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
    #[repr(transparent)]
    pub struct Probability(OrderedFloat<f64>);

    impl Probability {
        const MIN_P: OrderedFloat<f64> = OrderedFloat(0.0);
        const MAX_P: OrderedFloat<f64> = OrderedFloat(1.0);

        pub fn new(p: f64) -> Result<Self, NotProbabilityError> {
            let p = OrderedFloat(p);
            if Self::MIN_P <= p && p <= Self::MAX_P {
                Ok(Self(p))
            } else {
                Err(NotProbabilityError(p.0))
            }
        }

        pub const fn into_f64(self) -> f64 {
            self.0.0
        }
    }

    impl From<Probability> for f64 {
        fn from(value: Probability) -> Self {
            value.into_f64()
        }
    }

    impl TryFrom<f64> for Probability {
        type Error = NotProbabilityError;
        fn try_from(value: f64) -> Result<Self, Self::Error> {
            Self::new(value)
        }
    }
}

/// Haas et al. proposed several gossip protocols for *ad hoc networks* that use probabilities.
/// Combined with the number of hops or the number of times the same message is received, the
/// protocols choose if a node broadcast a message to all its neighbors or not, reducing thus
/// the number of messages propagated in the system. The authors show that gossiping with a
/// probability between 0.6 and 0.8 ensures that almost every node of the system gets the message,
/// with up to 35% fewer messages in some networks compared to flooding.
pub fn local_broadcast(message: KnowledgeMessage, rho: Probability) {
    //
}
