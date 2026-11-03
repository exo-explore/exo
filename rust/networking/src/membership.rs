//! Which nodes this one counts as part of the cluster, and when one joins or leaves.
//!
//! The main signal is each node's liveliness token. It isn't always enough: in a cluster of
//! three or more, zenoh can keep the token of a node that crashed (or lost power, or was cut
//! off) for good, and the rest of the cluster then never learns that the node has gone. When
//! the lost node was the master, nothing works until a heartbeat timeout much later.
//!
//! So a closed connection counts too. Every node is normally connected to every other one; when
//! the connection to a node closes, it is asked whether it is still there (it may still be
//! reachable through another node), and if it doesn't answer within [`PROBE_TIMEOUT`] it counts
//! as gone. The answer comes from zenoh's own threads, not from exo's Python code, so a node
//! that is busy is never taken for one that has gone.

use std::collections::HashSet;
use std::time::Duration;

use zenoh::config::ZenohId;

/// Every node answers queries on this key, followed by its id, to show it is still there.
pub const ALIVE_KEY_PREFIX: &str = "exo/alive";

/// How long a node whose connection closed has to answer before it counts as gone. A healthy
/// node answers in milliseconds, at a priority ahead of other traffic; a node that crashed
/// never does. This adds to the time it takes to notice a lost master, so it is short.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// A node joining or leaving, as this node sees it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    Joined(ZenohId),
    Left(ZenohId),
}

#[derive(Debug, Default)]
pub struct Membership {
    /// The nodes reported as joined (and not since as left)
    present: HashSet<ZenohId>,
    /// The nodes this one has a connection to
    connected: HashSet<ZenohId>,
    /// The nodes being asked whether they are still there
    probing: HashSet<ZenohId>,
}

impl Membership {
    pub fn token_declared(&mut self, zid: ZenohId) -> Option<Change> {
        self.joined(zid)
    }

    pub fn token_deleted(&mut self, zid: ZenohId) -> Option<Change> {
        self.left(zid)
    }

    /// A connection to a node opened. A node that restarts with the same id may come back
    /// before its old token is gone, and then no new token is seen: the connection shows it.
    pub fn connection_opened(&mut self, zid: ZenohId) -> Option<Change> {
        self.connected.insert(zid);
        self.joined(zid)
    }

    /// A connection to a node closed. Returns whether to ask the node if it is still there.
    pub fn connection_closed(&mut self, zid: ZenohId) -> bool {
        self.connected.remove(&zid);
        self.present.contains(&zid) && self.probing.insert(zid)
    }

    /// The answer to whether a node whose connection closed is still there
    pub fn probe_finished(&mut self, zid: ZenohId, answered: bool) -> Option<Change> {
        self.probing.remove(&zid);
        if answered || self.connected.contains(&zid) {
            return None;
        }
        self.left(zid)
    }

    fn joined(&mut self, zid: ZenohId) -> Option<Change> {
        self.present.insert(zid).then_some(Change::Joined(zid))
    }

    fn left(&mut self, zid: ZenohId) -> Option<Change> {
        self.present.remove(&zid).then_some(Change::Left(zid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zid(n: u8) -> ZenohId {
        format!("{:x}", u128::from(n) + 1)
            .parse()
            .expect("valid zid")
    }

    #[test]
    fn a_node_joins_once_whether_its_token_or_its_connection_comes_first() {
        let mut membership = Membership::default();
        assert_eq!(
            membership.connection_opened(zid(1)),
            Some(Change::Joined(zid(1)))
        );
        assert_eq!(membership.token_declared(zid(1)), None);

        assert_eq!(
            membership.token_declared(zid(2)),
            Some(Change::Joined(zid(2)))
        );
        assert_eq!(membership.connection_opened(zid(2)), None);
    }

    #[test]
    fn a_node_whose_connection_closed_leaves_if_it_doesnt_answer() {
        let mut membership = Membership::default();
        membership.token_declared(zid(1));
        membership.connection_opened(zid(1));

        assert!(membership.connection_closed(zid(1)));
        // Its token is never deleted (the zenoh bug this works around)
        assert_eq!(
            membership.probe_finished(zid(1), false),
            Some(Change::Left(zid(1)))
        );
        // ... and a later deletion doesn't report it twice
        assert_eq!(membership.token_deleted(zid(1)), None);
    }

    #[test]
    fn a_node_that_answers_stays() {
        let mut membership = Membership::default();
        membership.token_declared(zid(1));
        membership.connection_opened(zid(1));

        // Reachable through another node, or its connection reopened while it was asked
        assert!(membership.connection_closed(zid(1)));
        assert_eq!(membership.probe_finished(zid(1), true), None);

        assert!(membership.connection_closed(zid(1)));
        membership.connection_opened(zid(1));
        assert_eq!(membership.probe_finished(zid(1), false), None);
    }

    #[test]
    fn a_node_whose_token_went_first_isnt_asked() {
        let mut membership = Membership::default();
        membership.token_declared(zid(1));
        membership.connection_opened(zid(1));

        assert_eq!(membership.token_deleted(zid(1)), Some(Change::Left(zid(1))));
        assert!(!membership.connection_closed(zid(1)));
    }

    #[test]
    fn a_node_is_asked_once_at_a_time() {
        let mut membership = Membership::default();
        membership.token_declared(zid(1));
        membership.connection_opened(zid(1));

        assert!(membership.connection_closed(zid(1)));
        membership.connection_opened(zid(1));
        assert!(!membership.connection_closed(zid(1)));
    }

    #[test]
    fn a_node_that_left_and_reconnects_with_its_old_token_still_known_joins_again() {
        let mut membership = Membership::default();
        membership.token_declared(zid(1));
        membership.connection_opened(zid(1));
        membership.connection_closed(zid(1));
        membership.probe_finished(zid(1), false);

        assert_eq!(
            membership.connection_opened(zid(1)),
            Some(Change::Joined(zid(1)))
        );
    }
}
