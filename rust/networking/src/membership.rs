//! Which nodes this one counts as part of the cluster: one state per node, and every timer that
//! decides it.
//!
//! Each peer is Alive, Suspect or Gone, with the reason it isn't Alive:
//!
//! ```text
//!                 token put | connection opened
//!   (unknown) ------------------------------------------------------> Alive
//!   Alive -- token deleted | connection closed, probe unanswered ---> Suspect(Disconnected)
//!   Alive -- transport up, no report from exo on it for WEDGE_TIMEOUT -> Suspect(Wedged)
//!   Alive | Suspect -- it announced that it is leaving --------------> Gone(Left)
//!   Suspect(Disconnected) -- token put | connection opened | probe answered -> Alive
//!   Suspect(Wedged) -- a report -------------------------------------> Alive
//!   Suspect -- Suspect for GRACE ------------------------------------> Gone(same reason)
//!   Gone(Disconnected | Left) -- token put | connection opened ------> Alive (a rejoin)
//!   Gone(Wedged) -- a report ----------------------------------------> Alive
//! ```
//!
//! A loss is undone by the kind of signal that showed it: a lost transport by the transport,
//! a wedged exo by a report from it. Each change is reported once, whichever input shows it
//! first.
//!
//! The liveliness token is the main signal, but in a cluster of three or more zenoh can keep
//! the token of a node that crashed (or lost power, or was cut off) for good. So a closed
//! connection counts too: every node is normally connected to every other one, and when the
//! connection to a node closes it is asked whether it is still there (it may still be reachable
//! through another node). If it doesn't answer within [`PROBE_TIMEOUT`] it is Suspect. The
//! answer comes from zenoh's own threads, not from exo's Python code, so a busy node is never
//! taken for one that has gone.
//!
//! This node watches its own clocks too (see [`Membership::tick`]): after it was stopped
//! (frozen, starved, or asleep) it is Resuming for a while, and the peers it loses then aren't
//! timed toward Gone, because losing them says nothing about them.
//!
//! The state machine reads no clock: every input carries the time ([`Now`]), so it is tested
//! with made-up times.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime};

use zenoh::config::ZenohId;

/// Every node answers queries on this key, followed by its id, to show it is still there.
pub const ALIVE_KEY_PREFIX: &str = "exo/alive";

// ---- Every liveness timer, in one place --------------------------------------------------

/// zenoh's link lease: a connection that carries nothing (not even one of the 4 keep-alives
/// per lease) for this long is closed. A frozen, sleeping or cut-off peer is noticed after
/// this. It is zenoh's default, set explicitly in [`crate::cfg`] so it is visible and tested.
pub const LEASE: Duration = Duration::from_secs(10);

/// How long a node whose connection closed has to answer before it is Suspect. A healthy node
/// answers in milliseconds, at a priority ahead of other traffic, from zenoh's threads; a node
/// that crashed never does. When zenoh keeps a crashed node's token (in clusters of three or
/// more) the crash is noticed only when this runs out, so it is short.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// How often the membership checks its timers and its own clocks.
pub const TICK: Duration = Duration::from_secs(1);

/// A tick this late means this node itself was stopped (its process was frozen or starved, or
/// the machine slept), and peers may have expired their sessions with it while it still holds
/// its own. Well above the jitter of a 1 s tick on a busy machine, well below `LEASE`.
pub const STALL_THRESHOLD: Duration = Duration::from_secs(5);

/// After such a stall discovery neither answers nor dials peers for this long, so that the
/// stale sessions are closed (after `LEASE`) before new links are made. zenoh adds a new link
/// to a peer's existing session instead of starting a new one, and when the peer had already
/// dropped that session it never receives this node's declarations again: a one-way split.
pub const HOLD_OFF: Duration = Duration::from_secs(12);

/// How long peers take to reconnect once the hold-off is over: measured 1.6-4.4 s between
/// machines on a LAN and 4.8-6.2 s with every node on one machine.
pub const REJOIN: Duration = Duration::from_secs(6);

/// How long a peer is Suspect before it is Gone (G in the design). A peer frozen just past
/// the lease has its sessions closed (`LEASE`), holds off when it resumes (`HOLD_OFF`) and then
/// reconnects (`REJOIN`); with 2 s to spare it is back before it counts as gone.
pub const GRACE: Duration =
    Duration::from_secs(LEASE.as_secs() + HOLD_OFF.as_secs() + REJOIN.as_secs() + 2);

/// How long a peer whose transport is up may go without a report from exo on it before it is
/// Suspect(Wedged) (T_WEDGE in the design): the same as `GRACE` on purpose, so a wedged node
/// and a lost one are given the same time. Nodes report about once a second; only a node that
/// reported at least once is checked, so nothing is Wedged until reports are fed in.
pub const WEDGE_TIMEOUT: Duration = GRACE;

/// How long a node with less than half the cluster reachable waits before it counts only what
/// it can reach (used by the coordinator, later). Far longer than the faults the grace rides out
/// (a lid close, a Wi-Fi drop, a router restart), so by then the others are away for a while.
pub const MINORITY_HOLD_LIMIT: Duration = Duration::from_secs(5 * 60);

// ---- States --------------------------------------------------------------------------------

/// Why a node isn't Alive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// Its token was deleted, or its connection closed and it didn't answer
    Disconnected,
    /// It said it was leaving
    Left,
    /// Its transport is up but exo on it stopped reporting
    Wedged,
}

/// A node's state as this node sees it
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemberState {
    Alive,
    /// Unreachable now, for the reason given; Gone if it stays so for `GRACE`
    Suspect(Reason),
    Gone(Reason),
    /// Only ever this node's own state: it was stopped, and for `HOLD_OFF + REJOIN` after it
    /// resumed what it sees of its peers says little about them
    Resuming,
}

/// How much of the cluster this node can reach
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quorum {
    /// More than half
    Majority,
    /// Exactly half: two nodes with one away, or any even split
    Half,
    /// Less than half
    Minority,
}

/// A reading of both clocks. The monotonic clock times everything. The wall clock shows a
/// sleep, which macOS doesn't count on its monotonic clock, and dates what is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Now {
    pub monotonic: Instant,
    pub wall: SystemTime,
}

impl Now {
    pub fn read() -> Self {
        Self {
            monotonic: Instant::now(),
            wall: SystemTime::now(),
        }
    }
}

/// A node and the state it has been in since `since`. Reported once for each change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub node: ZenohId,
    pub state: MemberState,
    pub since: SystemTime,
}

/// Every node this node knows of, with how much of the cluster it reaches
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// This node first, then its peers
    pub members: Vec<Member>,
    pub quorum: Quorum,
    /// The Alive peers, plus this node unless it is Resuming
    pub reachable: usize,
    /// The nodes counted in the quorum: this node and every peer seen Alive, except those that
    /// left or were forgotten
    pub last_known: usize,
}

#[derive(Debug)]
struct Peer {
    state: MemberState,
    /// When it entered `state`
    since: Now,
    /// Whether a connection to it is open
    connected: bool,
    /// Whether it is being asked whether it is still there
    probing: bool,
    /// When exo on it last reported; `None` until it first does, and then it isn't checked
    last_heard: Option<Instant>,
}

#[derive(Debug)]
pub struct Membership {
    own: ZenohId,
    own_state: MemberState,
    own_since: SystemTime,
    grace: Duration,
    /// Every peer ever seen Alive, unless forgotten since
    peers: BTreeMap<ZenohId, Peer>,
    last_tick: Now,
    /// When this node's latest stall stops counting: until then it is Resuming, and a peer's
    /// grace (or wedge timeout) is counted from no earlier than this
    resumed_at: Option<Instant>,
}

impl Membership {
    /// `grace` is how long a peer is Suspect before it is Gone: `GRACE`, except in tests.
    pub fn new(own: ZenohId, grace: Duration, now: Now) -> Self {
        Self {
            own,
            own_state: MemberState::Alive,
            own_since: now.wall,
            grace,
            peers: BTreeMap::new(),
            last_tick: now,
            resumed_at: None,
        }
    }

    pub const fn own(&self) -> Member {
        Member {
            node: self.own,
            state: self.own_state,
            since: self.own_since,
        }
    }

    pub fn token_put(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        self.reachable_again(node, now)
    }

    pub fn token_deleted(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        self.lost(node, Reason::Disconnected, now)
    }

    /// A connection to a node opened. A node that restarts with the same id may come back
    /// before its old token is gone, and then no new token is seen: the connection shows it.
    pub fn connection_opened(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        if node == self.own {
            return None;
        }
        let change = self.reachable_again(node, now);
        if let Some(peer) = self.peers.get_mut(&node) {
            peer.connected = true;
        }
        change
    }

    /// A connection to a node closed. Returns whether to ask the node if it is still there
    /// (and then `probe_finished` must follow).
    pub fn connection_closed(&mut self, node: ZenohId) -> bool {
        let Some(peer) = self.peers.get_mut(&node) else {
            return false;
        };
        peer.connected = false;
        if peer.state != MemberState::Alive || peer.probing {
            return false;
        }
        peer.probing = true;
        true
    }

    /// The answer to whether a node whose connection closed is still there
    pub fn probe_finished(&mut self, node: ZenohId, answered: bool, now: Now) -> Option<Member> {
        let peer = self.peers.get_mut(&node)?;
        peer.probing = false;
        if answered {
            // Reachable through another node: the probe undoes a lost transport, nothing else
            return (peer.state == MemberState::Suspect(Reason::Disconnected))
                .then(|| self.set(node, MemberState::Alive, now));
        }
        if peer.connected {
            // Its connection opened again while it was asked
            return None;
        }
        self.lost(node, Reason::Disconnected, now)
    }

    /// The node said it is leaving (a graceful quit): it is Gone at once, without a grace.
    pub fn left(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        let peer = self.peers.get(&node)?;
        (peer.state != MemberState::Gone(Reason::Left))
            .then(|| self.set(node, MemberState::Gone(Reason::Left), now))
    }

    /// exo on the node reported (its periodic node info). Only a report undoes a wedge.
    pub fn heard_from(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        let peer = self.peers.get_mut(&node)?;
        peer.last_heard = Some(now.monotonic);
        matches!(
            peer.state,
            MemberState::Suspect(Reason::Wedged) | MemberState::Gone(Reason::Wedged)
        )
        .then(|| self.set(node, MemberState::Alive, now))
    }

    /// The node was removed from the cluster (by the coordinator): it no longer counts in the
    /// quorum, and if it comes back it is new.
    pub fn forget(&mut self, node: ZenohId) {
        self.peers.remove(&node);
    }

    /// Called every `TICK`: notices this node's own stalls, and peers that are Suspect for
    /// `grace` (Gone) or Alive but unheard for `WEDGE_TIMEOUT` (Suspect(Wedged)).
    ///
    /// A tick more than `STALL_THRESHOLD` late on either clock means this node was stopped.
    /// Both clocks are read because each misses a case: when a Mac sleeps its monotonic clock
    /// stops and only the wall clock shows the gap, and the wall clock can be set back while
    /// the process is frozen. This node is then Resuming for `HOLD_OFF + REJOIN` (discovery
    /// holds off, then the peers reconnect), and no peer's grace or wedge timeout counts the
    /// time before that: a peer it lost meanwhile gets a full grace from the end of it.
    pub fn tick(&mut self, now: Now) -> Vec<Member> {
        let mut changes = Vec::new();
        let late = elapsed(self.last_tick, now).saturating_sub(TICK);
        self.last_tick = now;
        if late >= STALL_THRESHOLD {
            log::warn!(
                "this node was unresponsive for {late:.0?}; resuming for {:?} before it times \
                 any lost peer",
                HOLD_OFF + REJOIN
            );
            self.resumed_at = Some(now.monotonic + HOLD_OFF + REJOIN);
            if self.own_state != MemberState::Resuming {
                changes.push(self.set_own(MemberState::Resuming, now));
            }
        } else if self.own_state == MemberState::Resuming
            && self.resumed_at.is_none_or(|at| now.monotonic >= at)
        {
            changes.push(self.set_own(MemberState::Alive, now));
        }

        let timed_from = |instant: Instant| self.resumed_at.map_or(instant, |at| instant.max(at));
        let due: Vec<(ZenohId, MemberState)> = self
            .peers
            .iter()
            .filter_map(|(&node, peer)| match peer.state {
                MemberState::Suspect(reason)
                    if now.monotonic >= timed_from(peer.since.monotonic) + self.grace =>
                {
                    Some((node, MemberState::Gone(reason)))
                }
                MemberState::Alive
                    if peer.last_heard.is_some_and(|heard| {
                        now.monotonic >= timed_from(heard) + WEDGE_TIMEOUT
                    }) =>
                {
                    Some((node, MemberState::Suspect(Reason::Wedged)))
                }
                MemberState::Alive
                | MemberState::Suspect(_)
                | MemberState::Gone(_)
                | MemberState::Resuming => None,
            })
            .collect();
        changes.extend(
            due.into_iter()
                .map(|(node, state)| self.set(node, state, now)),
        );
        changes
    }

    pub fn quorum(&self) -> Quorum {
        quorum(self.reachable(), self.last_known())
    }

    pub fn snapshot(&self) -> Snapshot {
        let members = std::iter::once(self.own())
            .chain(self.peers.iter().map(|(&node, peer)| Member {
                node,
                state: peer.state,
                since: peer.since.wall,
            }))
            .collect();
        Snapshot {
            members,
            quorum: self.quorum(),
            reachable: self.reachable(),
            last_known: self.last_known(),
        }
    }

    fn reachable(&self) -> usize {
        let own = usize::from(self.own_state != MemberState::Resuming);
        own + self
            .peers
            .values()
            .filter(|peer| peer.state == MemberState::Alive)
            .count()
    }

    fn last_known(&self) -> usize {
        1 + self
            .peers
            .values()
            .filter(|peer| peer.state != MemberState::Gone(Reason::Left))
            .count()
    }

    /// The transport shows the node: it joins, rejoins, or is back from a lost transport
    fn reachable_again(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        if node == self.own {
            return None;
        }
        let Some(peer) = self.peers.get(&node) else {
            self.peers.insert(
                node,
                Peer {
                    state: MemberState::Alive,
                    since: now,
                    connected: false,
                    probing: false,
                    last_heard: None,
                },
            );
            return Some(self.member(node));
        };
        match peer.state {
            MemberState::Suspect(Reason::Disconnected)
            | MemberState::Gone(Reason::Disconnected | Reason::Left) => {
                Some(self.set(node, MemberState::Alive, now))
            }
            // A wedge is only undone by a report; Suspect(Left) is never kept
            MemberState::Alive
            | MemberState::Suspect(Reason::Wedged | Reason::Left)
            | MemberState::Gone(Reason::Wedged)
            | MemberState::Resuming => None,
        }
    }

    fn lost(&mut self, node: ZenohId, reason: Reason, now: Now) -> Option<Member> {
        let peer = self.peers.get(&node)?;
        (peer.state == MemberState::Alive)
            .then(|| self.set(node, MemberState::Suspect(reason), now))
    }

    /// Moves a known peer to `state` and returns the change
    fn set(&mut self, node: ZenohId, state: MemberState, now: Now) -> Member {
        if let Some(peer) = self.peers.get_mut(&node) {
            peer.state = state;
            peer.since = now;
            if state == MemberState::Alive {
                // It couldn't report while it was away: its wedge timeout starts again
                peer.last_heard = peer.last_heard.map(|_| now.monotonic);
            }
        }
        self.member(node)
    }

    fn set_own(&mut self, state: MemberState, now: Now) -> Member {
        self.own_state = state;
        self.own_since = now.wall;
        self.own()
    }

    fn member(&self, node: ZenohId) -> Member {
        self.peers.get(&node).map_or_else(
            || Member {
                node,
                state: MemberState::Gone(Reason::Disconnected),
                since: SystemTime::UNIX_EPOCH,
            },
            |peer| Member {
                node,
                state: peer.state,
                since: peer.since.wall,
            },
        )
    }
}

/// The quorum read: how `reachable` nodes compare to half of `last_known`
pub fn quorum(reachable: usize, last_known: usize) -> Quorum {
    match reachable.saturating_mul(2).cmp(&last_known) {
        Ordering::Greater => Quorum::Majority,
        Ordering::Equal => Quorum::Half,
        Ordering::Less => Quorum::Minority,
    }
}

/// The time from `before` to `now` on whichever clock shows more of it. A wall clock set back
/// counts as no time.
fn elapsed(before: Now, now: Now) -> Duration {
    let monotonic = now.monotonic.saturating_duration_since(before.monotonic);
    let wall = now.wall.duration_since(before.wall).unwrap_or_default();
    monotonic.max(wall)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zid(n: u8) -> ZenohId {
        format!("{:x}", u128::from(n) + 1)
            .parse()
            .expect("valid zid")
    }

    const OWN: u8 = 0;

    /// A clock the tests move by hand: both clocks advance together unless told otherwise
    #[derive(Clone, Copy)]
    struct Clock {
        now: Now,
    }

    impl Clock {
        fn new() -> Self {
            Self {
                now: Now {
                    monotonic: Instant::now(),
                    wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
                },
            }
        }

        /// Both clocks move on by `by` (a running or frozen process)
        fn advance(&mut self, by: Duration) -> Now {
            self.now.monotonic += by;
            self.now.wall += by;
            self.now
        }

        /// Only the wall clock moves on (a Mac asleep: its monotonic clock stops)
        fn sleep(&mut self, by: Duration) -> Now {
            self.now.wall += by;
            self.now
        }

        /// The wall clock is set back (and the monotonic clock moves on by `by`)
        fn set_wall_back(&mut self, back: Duration, by: Duration) -> Now {
            self.now.monotonic += by;
            self.now.wall -= back;
            self.now
        }
    }

    fn membership(clock: &Clock) -> Membership {
        Membership::new(zid(OWN), GRACE, clock.now)
    }

    fn state(change: Option<Member>) -> Option<(ZenohId, MemberState)> {
        change.map(|member| (member.node, member.state))
    }

    fn states(changes: Vec<Member>) -> Vec<(ZenohId, MemberState)> {
        changes
            .into_iter()
            .map(|member| (member.node, member.state))
            .collect()
    }

    /// Ticks every second for `duration`, collecting what changed
    fn run_for(
        membership: &mut Membership,
        clock: &mut Clock,
        duration: Duration,
    ) -> Vec<(ZenohId, MemberState)> {
        let mut changes = Vec::new();
        let mut waited = Duration::ZERO;
        while waited < duration {
            waited += TICK;
            changes.extend(states(membership.tick(clock.advance(TICK))));
        }
        changes
    }

    fn alive(n: u8) -> (ZenohId, MemberState) {
        (zid(n), MemberState::Alive)
    }

    fn suspect(n: u8, reason: Reason) -> (ZenohId, MemberState) {
        (zid(n), MemberState::Suspect(reason))
    }

    fn gone(n: u8, reason: Reason) -> (ZenohId, MemberState) {
        (zid(n), MemberState::Gone(reason))
    }

    fn connected_peer(membership: &mut Membership, clock: &Clock, n: u8) {
        membership.token_put(zid(n), clock.now);
        membership.connection_opened(zid(n), clock.now);
    }

    // ---- Joining, and reporting each change once ----

    #[test]
    fn a_node_joins_once_whether_its_token_or_its_connection_comes_first() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        assert_eq!(
            state(membership.connection_opened(zid(1), clock.now)),
            Some(alive(1))
        );
        assert_eq!(state(membership.token_put(zid(1), clock.now)), None);

        assert_eq!(
            state(membership.token_put(zid(2), clock.now)),
            Some(alive(2))
        );
        assert_eq!(state(membership.connection_opened(zid(2), clock.now)), None);
    }

    #[test]
    fn this_node_is_never_its_own_peer() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        assert_eq!(membership.token_put(zid(OWN), clock.now), None);
        assert_eq!(membership.connection_opened(zid(OWN), clock.now), None);
        assert_eq!(membership.token_deleted(zid(OWN), clock.now), None);
        assert_eq!(
            states(membership.snapshot().members),
            vec![alive(OWN)],
            "only this node, Alive"
        );
    }

    #[test]
    fn signals_about_unknown_nodes_change_nothing() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        assert_eq!(membership.token_deleted(zid(1), clock.now), None);
        assert!(!membership.connection_closed(zid(1)));
        assert_eq!(membership.probe_finished(zid(1), false, clock.now), None);
        assert_eq!(membership.left(zid(1), clock.now), None);
        assert_eq!(membership.heard_from(zid(1), clock.now), None);
        assert_eq!(membership.snapshot().members.len(), 1);
    }

    #[test]
    fn a_loss_is_reported_once_whichever_signal_shows_it_first() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        // The connection closes and the probe fails, then the token goes too
        assert!(membership.connection_closed(zid(1)));
        assert_eq!(
            state(membership.probe_finished(zid(1), false, clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
        assert_eq!(membership.token_deleted(zid(1), clock.now), None);

        // And the other way round
        connected_peer(&mut membership, &clock, 2);
        assert_eq!(
            state(membership.token_deleted(zid(2), clock.now)),
            Some(suspect(2, Reason::Disconnected))
        );
        assert!(
            !membership.connection_closed(zid(2)),
            "a node whose token went first isn't asked"
        );
    }

    // ---- Alive -> Suspect ----

    #[test]
    fn a_node_whose_token_is_deleted_is_suspect() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert_eq!(
            state(membership.token_deleted(zid(1), clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
    }

    #[test]
    fn a_node_whose_connection_closed_is_suspect_if_it_doesnt_answer() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        assert!(membership.connection_closed(zid(1)));
        // Its token is never deleted (the zenoh bug this works around)
        assert_eq!(
            state(membership.probe_finished(zid(1), false, clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
    }

    #[test]
    fn a_node_that_answers_stays_alive() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        // Reachable through another node
        assert!(membership.connection_closed(zid(1)));
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);

        // ... or its connection reopened while it was asked
        assert!(membership.connection_closed(zid(1)));
        assert_eq!(membership.connection_opened(zid(1), clock.now), None);
        assert_eq!(membership.probe_finished(zid(1), false, clock.now), None);
        assert_eq!(membership.quorum(), Quorum::Majority);
    }

    #[test]
    fn a_node_is_asked_once_at_a_time() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        assert!(membership.connection_closed(zid(1)));
        membership.connection_opened(zid(1), clock.now);
        assert!(!membership.connection_closed(zid(1)));
    }

    #[test]
    fn a_node_that_announces_its_leave_is_gone_at_once() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert_eq!(
            state(membership.left(zid(1), clock.now)),
            Some(gone(1, Reason::Left))
        );
        assert_eq!(membership.left(zid(1), clock.now), None);
        // Its token and connection going afterwards change nothing
        assert_eq!(membership.token_deleted(zid(1), clock.now), None);
        assert!(!membership.connection_closed(zid(1)));

        // A suspect node that announces its leave is gone at once too
        connected_peer(&mut membership, &clock, 2);
        membership.token_deleted(zid(2), clock.now);
        assert_eq!(
            state(membership.left(zid(2), clock.now)),
            Some(gone(2, Reason::Left))
        );
    }

    #[test]
    fn a_node_that_stops_reporting_with_its_transport_up_is_wedged() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        membership.heard_from(zid(1), clock.now);

        // Node 2 never reported: it isn't checked (reports aren't fed in yet)
        let changes = run_for(&mut membership, &mut clock, WEDGE_TIMEOUT - TICK);
        assert_eq!(changes, vec![]);
        assert_eq!(
            run_for(&mut membership, &mut clock, TICK),
            vec![suspect(1, Reason::Wedged)]
        );
    }

    #[test]
    fn a_node_that_keeps_reporting_is_never_wedged() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        for _ in 0..(3 * WEDGE_TIMEOUT.as_secs()) {
            membership.heard_from(zid(1), clock.now);
            assert_eq!(states(membership.tick(clock.advance(TICK))), vec![]);
        }
    }

    // ---- Suspect -> Alive ----

    #[test]
    fn a_suspect_node_is_alive_again_when_its_transport_is_back() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        membership.token_deleted(zid(1), clock.now);
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );

        membership.token_deleted(zid(1), clock.now);
        assert_eq!(
            state(membership.connection_opened(zid(1), clock.now)),
            Some(alive(1))
        );

        // The probe was asked before the token went, and answered after
        assert!(membership.connection_closed(zid(1)));
        membership.token_deleted(zid(1), clock.now);
        assert_eq!(
            state(membership.probe_finished(zid(1), true, clock.now)),
            Some(alive(1))
        );
    }

    #[test]
    fn only_a_report_undoes_a_wedge() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.heard_from(zid(1), clock.now);
        assert_eq!(
            run_for(&mut membership, &mut clock, WEDGE_TIMEOUT),
            vec![suspect(1, Reason::Wedged)]
        );

        // The transport says nothing about exo on the node
        assert_eq!(membership.token_put(zid(1), clock.now), None);
        assert_eq!(membership.connection_opened(zid(1), clock.now), None);
        assert!(!membership.connection_closed(zid(1)));
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);

        assert_eq!(
            state(membership.heard_from(zid(1), clock.now)),
            Some(alive(1))
        );
        // ... and its wedge timeout starts again
        assert_eq!(
            run_for(&mut membership, &mut clock, WEDGE_TIMEOUT - TICK),
            vec![]
        );
    }

    #[test]
    fn a_node_back_from_a_lost_transport_has_a_fresh_wedge_timeout() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.heard_from(zid(1), clock.now);
        membership.token_deleted(zid(1), clock.now);
        run_for(&mut membership, &mut clock, GRACE - TICK);

        // Back just before the grace: it couldn't report while it was away
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );
        assert_eq!(
            run_for(&mut membership, &mut clock, WEDGE_TIMEOUT - TICK),
            vec![]
        );
    }

    // ---- Suspect -> Gone, and back ----

    #[test]
    fn a_suspect_node_is_gone_after_the_grace() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.token_deleted(zid(1), clock.now);

        assert_eq!(run_for(&mut membership, &mut clock, GRACE - TICK), vec![]);
        assert_eq!(
            run_for(&mut membership, &mut clock, TICK),
            vec![gone(1, Reason::Disconnected)]
        );
        // Reported once
        assert_eq!(run_for(&mut membership, &mut clock, GRACE), vec![]);
    }

    #[test]
    fn a_wedged_node_is_gone_after_the_grace_and_back_when_it_reports() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.heard_from(zid(1), clock.now);
        run_for(&mut membership, &mut clock, WEDGE_TIMEOUT);
        assert_eq!(
            run_for(&mut membership, &mut clock, GRACE),
            vec![gone(1, Reason::Wedged)]
        );
        assert_eq!(membership.token_put(zid(1), clock.now), None);
        assert_eq!(
            state(membership.heard_from(zid(1), clock.now)),
            Some(alive(1))
        );
    }

    #[test]
    fn the_grace_can_be_shortened() {
        let mut clock = Clock::new();
        let mut membership = Membership::new(zid(OWN), Duration::from_secs(3), clock.now);
        connected_peer(&mut membership, &clock, 1);
        membership.token_deleted(zid(1), clock.now);
        assert_eq!(
            run_for(&mut membership, &mut clock, Duration::from_secs(3)),
            vec![gone(1, Reason::Disconnected)]
        );
    }

    #[test]
    fn a_gone_node_rejoins_when_its_transport_is_back() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.connection_closed(zid(1));
        membership.probe_finished(zid(1), false, clock.now);
        run_for(&mut membership, &mut clock, GRACE);

        // A probe answer doesn't bring back a Gone node: only the transport does
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);
        assert_eq!(
            state(membership.connection_opened(zid(1), clock.now)),
            Some(alive(1))
        );

        // ... and one that left comes back the same way, when it restarts
        membership.left(zid(1), clock.now);
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );
    }

    #[test]
    fn a_node_back_before_the_grace_is_never_gone() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.token_deleted(zid(1), clock.now);
        run_for(&mut membership, &mut clock, GRACE - TICK);
        membership.token_put(zid(1), clock.now);

        // Lost again: the grace starts over
        membership.token_deleted(zid(1), clock.now);
        assert_eq!(run_for(&mut membership, &mut clock, GRACE - TICK), vec![]);
    }

    #[test]
    fn a_forgotten_node_is_new_when_it_comes_back() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.token_deleted(zid(1), clock.now);
        membership.forget(zid(1));
        assert_eq!(membership.snapshot().last_known, 1);
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );
    }

    // ---- This node's own stalls ----

    #[test]
    fn ticks_on_time_are_no_stall() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        // Late, but less than the threshold
        let late = TICK + STALL_THRESHOLD - Duration::from_millis(1);
        assert_eq!(membership.tick(clock.advance(late)), vec![]);
        assert_eq!(run_for(&mut membership, &mut clock, GRACE), vec![]);
        assert_eq!(membership.own().state, MemberState::Alive);
    }

    #[test]
    fn a_frozen_node_is_resuming_for_the_hold_off_and_the_rejoin() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        // Frozen (SIGSTOP, a starved runtime): both clocks move on
        assert_eq!(
            states(membership.tick(clock.advance(TICK + STALL_THRESHOLD))),
            vec![(zid(OWN), MemberState::Resuming)]
        );
        assert_eq!(membership.own().state, MemberState::Resuming);
        assert_eq!(
            run_for(&mut membership, &mut clock, HOLD_OFF + REJOIN - TICK),
            vec![]
        );
        assert_eq!(run_for(&mut membership, &mut clock, TICK), vec![alive(OWN)]);
    }

    #[test]
    fn a_sleeping_mac_is_resuming_though_its_monotonic_clock_stopped() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        membership.tick(clock.advance(TICK));
        // Asleep for a minute: the next tick is one monotonic second later
        clock.sleep(Duration::from_secs(60));
        assert_eq!(
            states(membership.tick(clock.advance(TICK))),
            vec![(zid(OWN), MemberState::Resuming)]
        );
    }

    #[test]
    fn a_wall_clock_set_back_while_frozen_is_still_a_stall() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        assert_eq!(
            states(
                membership
                    .tick(clock.set_wall_back(Duration::from_secs(60), Duration::from_secs(30)))
            ),
            vec![(zid(OWN), MemberState::Resuming)]
        );
    }

    #[test]
    fn a_wall_clock_set_back_while_running_is_no_stall() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        assert_eq!(
            membership.tick(clock.set_wall_back(Duration::from_secs(3600), TICK)),
            vec![]
        );
        assert_eq!(run_for(&mut membership, &mut clock, TICK * 3), vec![]);
    }

    #[test]
    fn a_stall_while_resuming_extends_it_and_is_reported_once() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        membership.tick(clock.advance(Duration::from_secs(20)));
        run_for(&mut membership, &mut clock, Duration::from_secs(10));
        assert_eq!(
            membership.tick(clock.advance(Duration::from_secs(20))),
            vec![]
        );
        assert_eq!(
            run_for(&mut membership, &mut clock, HOLD_OFF + REJOIN - TICK),
            vec![]
        );
        assert_eq!(run_for(&mut membership, &mut clock, TICK), vec![alive(OWN)]);
    }

    #[test]
    fn peers_lost_while_resuming_get_a_full_grace_after_it() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        // Node 1 was lost before this node froze for 40 s, node 2 just after it resumed
        membership.token_deleted(zid(1), clock.now);
        run_for(&mut membership, &mut clock, Duration::from_secs(5));
        membership.tick(clock.advance(Duration::from_secs(40)));
        // Its loss is reported at once (and starts an election as before): only its grace waits
        assert_eq!(
            state(membership.token_deleted(zid(2), clock.now)),
            Some(suspect(2, Reason::Disconnected))
        );

        let resuming = HOLD_OFF + REJOIN;
        assert_eq!(
            run_for(&mut membership, &mut clock, resuming + GRACE - TICK),
            vec![alive(OWN)],
            "nobody is Gone within a grace of the end of resuming"
        );
        let mut gone_after = run_for(&mut membership, &mut clock, TICK);
        gone_after.sort_by_key(|&(node, _)| node);
        let mut expected = vec![gone(1, Reason::Disconnected), gone(2, Reason::Disconnected)];
        expected.sort_by_key(|&(node, _)| node);
        assert_eq!(gone_after, expected);
    }

    #[test]
    fn peers_arent_wedged_by_a_stall_of_this_node() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.heard_from(zid(1), clock.now);
        // Frozen for longer than the wedge timeout: its reports were waiting for this node
        membership.tick(clock.advance(WEDGE_TIMEOUT * 2));
        assert_eq!(
            run_for(
                &mut membership,
                &mut clock,
                HOLD_OFF + REJOIN + WEDGE_TIMEOUT - TICK
            ),
            vec![alive(OWN)]
        );
    }

    // ---- The quorum read ----

    /// A cluster of `total` nodes (this one included) of which `away` peers are Suspect and
    /// `gone` are Gone
    fn cluster(total: u8, away: u8, gone: u8) -> Membership {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        for n in 1..total {
            connected_peer(&mut membership, &clock, n);
        }
        for n in 1..=(away + gone) {
            membership.token_deleted(zid(n), clock.now);
        }
        run_for(&mut membership, &mut clock, GRACE - TICK);
        for n in 1..=away {
            membership.token_put(zid(n), clock.now);
            membership.token_deleted(zid(n), clock.now);
        }
        run_for(&mut membership, &mut clock, TICK);
        let snapshot = membership.snapshot();
        let count = |wanted: fn(MemberState) -> bool| {
            snapshot
                .members
                .iter()
                .filter(|member| wanted(member.state))
                .count()
        };
        assert_eq!(
            count(|s| matches!(s, MemberState::Suspect(_))),
            usize::from(away)
        );
        assert_eq!(
            count(|s| matches!(s, MemberState::Gone(_))),
            usize::from(gone)
        );
        membership
    }

    #[test]
    fn quorum_reads_for_one_to_five_nodes() {
        use Quorum::{Half, Majority, Minority};
        // (nodes, unreachable peers, expected)
        let cases = [
            (1, 0, Majority),
            (2, 0, Majority),
            (2, 1, Half),
            (3, 0, Majority),
            (3, 1, Majority),
            (3, 2, Minority),
            (4, 0, Majority),
            (4, 1, Majority),
            (4, 2, Half),
            (4, 3, Minority),
            (5, 0, Majority),
            (5, 1, Majority),
            (5, 2, Majority),
            (5, 3, Minority),
            (5, 4, Minority),
        ];
        for (total, unreachable, expected) in cases {
            // Suspect and Gone peers count the same: both were members and aren't reachable
            for gone in 0..=unreachable {
                let membership = cluster(total, unreachable - gone, gone);
                let snapshot = membership.snapshot();
                assert_eq!(
                    (snapshot.quorum, snapshot.last_known, snapshot.reachable),
                    (
                        expected,
                        usize::from(total),
                        usize::from(total - unreachable)
                    ),
                    "{total} nodes, {unreachable} unreachable ({gone} gone)"
                );
            }
        }
    }

    #[test]
    fn quorum_reads_for_even_splits() {
        for total in [2_usize, 4, 6, 8] {
            assert_eq!(quorum(total / 2, total), Quorum::Half, "{total} nodes");
            assert_eq!(quorum(total / 2 + 1, total), Quorum::Majority);
            assert_eq!(quorum(total / 2 - 1, total), Quorum::Minority);
        }
    }

    #[test]
    fn nodes_that_left_dont_count_in_the_quorum() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        membership.left(zid(1), clock.now);
        membership.left(zid(2), clock.now);
        let snapshot = membership.snapshot();
        assert_eq!((snapshot.reachable, snapshot.last_known), (1, 1));
        assert_eq!(snapshot.quorum, Quorum::Majority);
    }

    #[test]
    fn a_resuming_node_doesnt_count_itself_as_reachable() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert_eq!(membership.quorum(), Quorum::Majority);
        membership.tick(clock.advance(Duration::from_secs(20)));
        assert_eq!(membership.quorum(), Quorum::Half);
        assert_eq!(membership.snapshot().reachable, 1);
    }

    #[test]
    fn the_snapshot_lists_this_node_first_with_the_times_of_each_change() {
        let mut clock = Clock::new();
        let start = clock.now.wall;
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 2);
        connected_peer(&mut membership, &clock, 1);
        let lost_at = clock.advance(Duration::from_secs(3));
        membership.token_deleted(zid(2), lost_at);
        let snapshot = membership.snapshot();
        assert_eq!(
            snapshot.members,
            vec![
                Member {
                    node: zid(OWN),
                    state: MemberState::Alive,
                    since: start
                },
                Member {
                    node: zid(1),
                    state: MemberState::Alive,
                    since: start
                },
                Member {
                    node: zid(2),
                    state: MemberState::Suspect(Reason::Disconnected),
                    since: lost_at.wall
                },
            ]
        );
    }

    // ---- The timers' derivations ----

    #[test]
    fn the_timers_keep_their_derivations() {
        assert!(GRACE >= LEASE + HOLD_OFF + REJOIN, "{GRACE:?}");
        assert_eq!(GRACE, Duration::from_secs(30));
        assert_eq!(WEDGE_TIMEOUT, GRACE);
        assert!(
            HOLD_OFF > LEASE,
            "stale sessions must close before reconnecting"
        );
        assert!(STALL_THRESHOLD < LEASE);
        assert!(STALL_THRESHOLD > TICK);
        assert!(MINORITY_HOLD_LIMIT > GRACE);
    }
}
