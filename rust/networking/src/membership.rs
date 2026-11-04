//! Which nodes this one counts as part of the cluster: one state per node, and every timer that
//! decides it.
//!
//! Each peer is Alive, Suspect or Gone, with the reason it isn't Alive:
//!
//! ```text
//!                 token put | connection opened
//!   (unknown) ------------------------------------------------------> Alive
//!   Alive -- token deleted, no connection | probe unanswered -------> Suspect(Disconnected)
//!   Alive -- transport up, no report from exo on it for WEDGE_TIMEOUT -> Suspect(Wedged)
//!   Alive | Suspect -- it announced that it is leaving --------------> Gone(Left), forgotten
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
//! exo tells it what only exo knows: that exo on a node reported ([`Membership::heard_from`]),
//! that the cluster's state lists a node ([`Membership::listed`]), and that the coordinator
//! removed one ([`Membership::forget`]).
//!
//! The quorum read counts this node and the peers the cluster's state lists (seen here or not:
//! see [`Membership::listed`]), and a peer stops counting (it stays in the view, as Gone, until
//! it is removed) when it leaves, and when it has been Gone for [`FORGET_AFTER`]. The
//! coordinator removing it (`forget`, unless this node still reaches it) takes it out of the
//! view too, once this node reads Majority with it still counted: a node cut off from most of
//! the cluster (or either side of an even split) may be told by its own coordinator to remove
//! the others, and honouring that at once would make it read Majority before the split hold
//! ends. Otherwise every restart of a node, which comes back with a new id, would count as one
//! more node for good. A peer that comes back counts again.
//!
//! The liveliness token is the main signal, but in a cluster of three or more zenoh can keep
//! the token of a node that crashed (or lost power, or was cut off) for good. So a closed
//! connection counts too: every node is normally connected to every other one, and when the
//! connection to a node closes it is asked whether it is still there (it may still be reachable
//! through another node). If it doesn't answer within [`PROBE_TIMEOUT`] it is Suspect. The
//! answer comes from zenoh's own threads, not from exo's Python code, so a busy node is never
//! taken for one that has gone. A node is asked too when its token is deleted while its
//! connection is still open, and, every [`REPROBE_INTERVAL`], while it is Alive without a
//! connection of its own: nothing else would show that it crashed. The probes to make are taken
//! with [`Membership::take_probes`], and each one's answer comes back to `probe_finished`.
//!
//! This node watches its own clocks too (see [`Membership::tick`]): after it was stopped
//! (frozen, starved, or asleep) it is Resuming for a while, and the peers it loses then aren't
//! timed toward Gone, because losing them says nothing about them.
//!
//! The state machine reads no clock: every input carries the time ([`Now`]), so it is tested
//! with made-up times. It doesn't log either (its caller does, see [`Membership::take_stall`]).

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

/// zenoh recomputes its routes this long after a link changes (`TREES_COMPUTATION_DELAY_MS`,
/// zenoh's own constant). Until then a query to a node that is now reachable only through
/// another one finds no route and ends at once, with no answer.
const ZENOH_ROUTES_RECOMPUTED_AFTER: Duration = Duration::from_millis(100);

/// How long after a connection closed its node is first asked whether it is still there, and
/// how long to wait before asking again each time a query ends without an answer, until
/// `PROBE_TIMEOUT`. Three times zenoh's route recomputation delay, so that the other nodes'
/// news of the closed link has reached this node too. Asked once and at once, a node still
/// reachable through a third one was taken for lost in every one of 12 trials on one machine;
/// asked this way, in none of 12.
pub const PROBE_FIRST_ASK_DELAY: Duration = ZENOH_ROUTES_RECOMPUTED_AFTER.saturating_mul(3);

/// How often a node that is Alive without a direct connection (it is reached through another
/// node) is asked again whether it is still there. When such a node crashes no connection of
/// this node closes, and in a cluster of three or more zenoh may keep its token, so only asking
/// shows it. Asked every lease, its crash is noticed within `REPROBE_INTERVAL + TICK +
/// PROBE_TIMEOUT` (13 s), about when a frozen peer with a direct connection is (`LEASE +
/// PROBE_TIMEOUT`, 12 s). Each ask is one small query routed through another node.
pub const REPROBE_INTERVAL: Duration = LEASE;

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

/// How much longer than `LEASE` a peer can be frozen (or cut off) and still be back before it
/// is Gone: `GRACE` rides out a freeze of up to `LEASE + FREEZE_RIDDEN_OUT` (22 s) in all.
const FREEZE_RIDDEN_OUT: Duration = Duration::from_secs(12);

/// How long a peer is Suspect before it is Gone. It is Suspect once its link
/// was silent for `LEASE` (and the probe ran out); a peer frozen just past that resumes, holds
/// off for `HOLD_OFF` and reconnects within `REJOIN`, so it is back `HOLD_OFF + REJOIN` = 18 s
/// after it was Suspect (measured: 18.3 s for a freeze 5.6 s past the lease). The rest is
/// `FREEZE_RIDDEN_OUT`. A lost peer is Gone 40-42 s after it was lost (`LEASE` or
/// `PROBE_TIMEOUT`, then this).
pub const GRACE: Duration =
    Duration::from_secs(HOLD_OFF.as_secs() + REJOIN.as_secs() + FREEZE_RIDDEN_OUT.as_secs());

/// How long a peer must stay Alive for its earlier losses to stop counting toward `GRACE`. A peer
/// lost again sooner adds its new time away to the earlier one, so a link that drops every few
/// seconds ends Gone once it has been away for `GRACE` in all, instead of never (each return
/// used to start the grace again). The same length as `GRACE`: a peer back for as long as this
/// node waits for a lost one is steady, and its next loss gets a full grace.
pub const STEADY_AFTER: Duration = GRACE;

/// How long a peer whose transport is up may go without a report from exo on it before it is
/// Suspect(Wedged). Nodes report about once a second, and a busy
/// node's reports were up to 6.3 s apart: about 5 times that, so a slow exo isn't taken for a
/// stuck one. A wedged node is then Gone 60 s after its last report (this, then `GRACE`),
/// later than a lost one: a slow exo is harder to tell from a stuck one than a lost transport
/// is. Only a node that reported at least once is checked, so nothing is Wedged until reports
/// are fed in.
pub const WEDGE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the side of a split without a majority (an even split, or the smaller side) holds
/// before it takes over the work of the nodes it lost: placing their models again elsewhere,
/// and counting only the nodes it reaches (see `FORGET_AFTER`). The other side may still be
/// serving them. The longest outages that healed by themselves in fault tests were network cuts
/// of 90 s and a Mac that rejoined 72 s after waking: about 3 times that, so a blip never moves
/// a model, and still short enough that the work of a node that really died moves within
/// minutes. One constant for how long an even split, or a node cut off from most of the
/// cluster, waits before acting on the nodes it can't reach.
pub const SPLIT_HOLD: Duration = Duration::from_secs(5 * 60);

/// How long a peer is Gone before it no longer counts in the quorum read (it stays in the view, as
/// Gone, so that a coordinator can still remove it). Not counting it makes this side of a split a
/// majority of fewer nodes, which ends the hold for its work, so it is the split hold: never
/// sooner, or the hold would end early. A peer that left stops counting at once (`left`).
pub const FORGET_AFTER: Duration = SPLIT_HOLD;

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

/// A reading of the clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Now {
    /// Times everything. It stops while the machine sleeps (Rust's `Instant`: the uptime clock on
    /// macOS, `CLOCK_MONOTONIC` on Linux), so a grace doesn't run out during a sleep.
    pub awake: Instant,
    /// A clock that keeps counting while the machine sleeps (`CLOCK_MONOTONIC` on macOS,
    /// `CLOCK_BOOTTIME` on Linux), as the time since some fixed point. Only compared with
    /// `awake` to see that this node slept: unlike the wall clock it never jumps, so neither an
    /// NTP step nor someone setting the time looks like a stall.
    pub with_sleep: Duration,
    /// Dates what is reported; never used to time anything
    pub wall: SystemTime,
}

impl Now {
    pub fn read() -> Self {
        Self {
            awake: Instant::now(),
            with_sleep: clock_with_sleep(),
            wall: SystemTime::now(),
        }
    }
}

#[cfg(target_os = "macos")]
const CLOCK_WITH_SLEEP: libc::clockid_t = libc::CLOCK_MONOTONIC;
#[cfg(target_os = "linux")]
const CLOCK_WITH_SLEEP: libc::clockid_t = libc::CLOCK_BOOTTIME;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
const CLOCK_WITH_SLEEP: libc::clockid_t = libc::CLOCK_MONOTONIC;

/// The clock that keeps counting while the machine sleeps; zero if it can't be read (then only
/// a frozen process shows as a stall, not a sleep)
fn clock_with_sleep() -> Duration {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime only writes into `time`, which outlives the call
    if unsafe { libc::clock_gettime(CLOCK_WITH_SLEEP, &raw mut time) } != 0 {
        return Duration::ZERO;
    }
    Duration::new(
        u64::try_from(time.tv_sec).unwrap_or(0),
        u32::try_from(time.tv_nsec).unwrap_or(0),
    )
}

/// Whether the time from the tick at `before` to the one at `now` means this node was stopped:
/// more than `STALL_THRESHOLD` late, on the clock that stops in a sleep (a frozen or starved
/// process) or on the one that doesn't (a sleep). Returns how late it was. Membership and
/// discovery both use this, so they agree on what a stall is.
pub fn stall(before: Now, now: Now) -> Option<Duration> {
    let awake = now.awake.saturating_duration_since(before.awake);
    let with_sleep = now.with_sleep.saturating_sub(before.with_sleep);
    let late = awake.max(with_sleep).saturating_sub(TICK);
    (late >= STALL_THRESHOLD).then_some(late)
}

/// A node and the state it has been in since `since`. Reported once for each change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub node: ZenohId,
    pub state: MemberState,
    pub since: SystemTime,
    /// Whether it counts in the quorum read: a peer counts once the cluster's state lists it;
    /// one Gone for `FORGET_AFTER`, or that left, doesn't (it stays in the view until the
    /// coordinator removes it)
    pub counted: bool,
}

/// Every node this node knows of, with how much of the cluster it reaches
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// This node first, then its peers
    pub members: Vec<Member>,
    pub quorum: Quorum,
    /// The counted Alive peers, plus this node unless it is Resuming
    pub reachable: usize,
    /// The nodes counted in the quorum: this node and every counted peer (see `FORGET_AFTER`)
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
    /// Whether to ask it again as soon as the current probe ends: something happened while it
    /// was asked (its connection closed again, or its token went) that the answer may predate
    probe_again: bool,
    /// Whether a connection to it opened while it was asked
    opened_while_probing: bool,
    /// When to ask it again, while it is Alive without a connection of its own
    next_probe: Option<Instant>,
    /// When exo on it last reported; `None` until it first does, and then it isn't checked
    last_heard: Option<Instant>,
    /// Its time Suspect before the current state, since it was last Alive for `STEADY_AFTER`:
    /// it counts toward `GRACE` with the current time Suspect
    away_before: Duration,
    /// Whether it counts in the quorum read (see `FORGET_AFTER`): from when it is `listed`
    counted: bool,
    /// The coordinator removed it: it leaves the view once it isn't Alive here and this node
    /// reads Majority (see `forget`), unless the cluster lists it again first (see `listed`)
    removed: bool,
    /// Whether the cluster's state lists it (it reported, or a snapshot named it). Until then
    /// it doesn't count: a node that leaves before it ever reported (it quit within seconds of
    /// starting) isn't in the cluster's state, so nothing removes it, and it would otherwise
    /// count until `FORGET_AFTER`. Once Gone, a peer the state never listed leaves the view.
    listed: bool,
}

#[derive(Debug)]
pub struct Membership {
    own: ZenohId,
    own_state: MemberState,
    own_since: SystemTime,
    grace: Duration,
    /// Every peer seen Alive, until it is forgotten (see `FORGET_AFTER`)
    peers: BTreeMap<ZenohId, Peer>,
    /// The previous tick; `None` before the first, which never counts as late (a node may
    /// start ticking well after the membership was made)
    last_tick: Option<Now>,
    /// When this node's latest stall stops counting: until then it is Resuming, and a peer's
    /// grace (or wedge timeout) is counted from no earlier than this
    resumed_at: Option<Instant>,
    /// How late the latest tick was, when that was a stall, until `take_stall` reads it
    stalled_for: Option<Duration>,
    /// The nodes to ask whether they are still there, until `take_probes` reads them
    probes: Vec<ZenohId>,
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
            last_tick: None,
            resumed_at: None,
            stalled_for: None,
            probes: Vec::new(),
        }
    }

    pub const fn own(&self) -> Member {
        Member {
            node: self.own,
            state: self.own_state,
            since: self.own_since,
            counted: true,
        }
    }

    pub fn token_put(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        self.reachable_again(node, now)
    }

    /// A node's token was deleted. With its connection still open, it is asked first: zenoh
    /// can drop a token while the node is there.
    pub fn token_deleted(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        let peer = self.peers.get(&node)?;
        if peer.state == MemberState::Alive && peer.connected {
            self.ask(node);
            return None;
        }
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
            peer.opened_while_probing |= peer.probing;
            peer.next_probe = None;
        }
        change
    }

    /// A connection to a node closed: an Alive node is asked whether it is still there (it may be
    /// reachable through another node; see `take_probes`).
    pub fn connection_closed(&mut self, node: ZenohId) {
        let Some(peer) = self.peers.get_mut(&node) else {
            return;
        };
        peer.connected = false;
        if peer.state == MemberState::Alive {
            self.ask(node);
        }
    }

    /// The answer to whether a node is still there, for each node `take_probes` gave
    pub fn probe_finished(&mut self, node: ZenohId, answered: bool, now: Now) -> Option<Member> {
        let peer = self.peers.get_mut(&node)?;
        peer.probing = false;
        let ask_again = std::mem::take(&mut peer.probe_again);
        let reopened = std::mem::take(&mut peer.opened_while_probing);
        let change = if answered {
            // Reachable (perhaps through another node): the probe undoes a lost transport,
            // nothing else
            (peer.state == MemberState::Suspect(Reason::Disconnected))
                .then(|| self.set(node, MemberState::Alive, now))
        } else if reopened && peer.connected {
            // Its connection opened again while it was asked
            None
        } else {
            self.lost(node, Reason::Disconnected, now)
        };
        if ask_again
            && self
                .peers
                .get(&node)
                .is_some_and(|peer| peer.state == MemberState::Alive && !peer.connected)
        {
            self.ask(node);
        }
        change
    }

    /// The nodes to ask whether they are still there (each answer goes to `probe_finished`)
    pub fn take_probes(&mut self) -> Vec<ZenohId> {
        std::mem::take(&mut self.probes)
    }

    /// The node said it is leaving (a graceful quit): it is Gone at once, without a grace, and
    /// stops counting in the quorum read (it stays in the view until it is removed).
    pub fn left(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        let peer = self.peers.get_mut(&node)?;
        peer.counted = false;
        (peer.state != MemberState::Gone(Reason::Left))
            .then(|| self.set(node, MemberState::Gone(Reason::Left), now))
    }

    /// exo on the node reported (its periodic node info). Only a report undoes a wedge.
    pub fn heard_from(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        let peer = self.peers.get_mut(&node)?;
        peer.last_heard = Some(now.awake);
        matches!(
            peer.state,
            MemberState::Suspect(Reason::Wedged) | MemberState::Gone(Reason::Wedged)
        )
        .then(|| self.set(node, MemberState::Alive, now))
    }

    /// The cluster's state lists the node: it reported to the coordinator, or a snapshot of the
    /// state names it. It counts in the quorum read from then on (until it leaves, or has been
    /// Gone for `FORGET_AFTER`), seen here or not: a node that joins one side of a split
    /// adopts a state that lists the other side's nodes, and counting them it reads that side
    /// as the minority it is, not as a majority of the nodes it happens to have seen.
    ///
    /// A node this one has never seen is Suspect(Disconnected) from now, and Gone after the
    /// grace unless its transport shows it, so a node that went away before this one started
    /// (a new coordinator can inherit such nodes, or a late report can bring one back) is
    /// removed like any other instead of staying listed for good.
    ///
    /// A removal still waiting (the node was Alive here, or this node didn't read Majority) no
    /// longer holds: the node reported after it, and is back in the state; taking it out of
    /// the view would leave the state listing a node that, lost, never shows as Gone to be
    /// removed.
    pub fn listed(&mut self, node: ZenohId, now: Now) -> Option<Member> {
        if node == self.own {
            return None;
        }
        if let Some(peer) = self.peers.get_mut(&node) {
            peer.removed = false;
            if !peer.listed {
                peer.listed = true;
                peer.counted = peer.state != MemberState::Gone(Reason::Left);
            }
            return None;
        }
        self.peers.insert(
            node,
            Peer {
                state: MemberState::Suspect(Reason::Disconnected),
                since: now,
                connected: false,
                probing: false,
                probe_again: false,
                opened_while_probing: false,
                next_probe: None,
                last_heard: None,
                away_before: Duration::ZERO,
                counted: true,
                listed: true,
                removed: false,
            },
        );
        Some(self.member(node))
    }

    /// The coordinator removed the node from the cluster: it leaves the view (and no longer
    /// counts), and if it comes back it is new. Only once it isn't Alive here, and this node reads
    /// Majority with it still counted, now or later:
    /// - a quitting node announces its departure (a removal) while it is still Alive here, and its
    ///   token goes a few seconds later; a node this node still reaches for good (wrongly
    ///   removed) stays, and a node that is back after a real loss is no longer removed;
    /// - a coordinator on the smaller side of a split, or on either side of an even one, also
    ///   removes the nodes it lost, and honouring that at once would make this node read Majority
    ///   before the split hold ends.
    pub fn forget(&mut self, node: ZenohId) {
        if let Some(peer) = self.peers.get_mut(&node) {
            peer.removed = true;
            self.apply_removals();
        }
    }

    /// Takes the nodes the coordinator removed, and that aren't Alive here, out of the view, if
    /// this node reads Majority
    fn apply_removals(&mut self) {
        if self.quorum() == Quorum::Majority {
            self.peers
                .retain(|_, peer| !(peer.removed && peer.state != MemberState::Alive));
        }
    }

    /// Called every `TICK`: notices this node's own stalls, and peers that are Suspect for
    /// `grace` (Gone) or Alive but unheard for `WEDGE_TIMEOUT` (Suspect(Wedged)).
    ///
    /// A late tick (see `stall`) means this node was stopped: frozen, starved or asleep. It is
    /// then Resuming for `HOLD_OFF + REJOIN` (discovery holds off, then the peers reconnect),
    /// and no peer's grace or wedge timeout counts the time before that: a peer it lost
    /// meanwhile gets a full grace from the end of it.
    pub fn tick(&mut self, now: Now) -> Vec<Member> {
        let mut changes = Vec::new();
        let stalled = self
            .last_tick
            .replace(now)
            .and_then(|before| stall(before, now));
        if let Some(late) = stalled {
            self.stalled_for = Some(late);
            self.resumed_at = Some(now.awake + HOLD_OFF + REJOIN);
            if self.own_state != MemberState::Resuming {
                changes.push(self.set_own(MemberState::Resuming, now));
            }
        } else if self.own_state == MemberState::Resuming
            && self.resumed_at.is_none_or(|at| now.awake >= at)
        {
            changes.push(self.set_own(MemberState::Alive, now));
        }

        let timed_from = |instant: Instant| self.resumed_at.map_or(instant, |at| instant.max(at));
        let due: Vec<(ZenohId, MemberState)> = self
            .peers
            .iter()
            .filter_map(|(&node, peer)| match peer.state {
                MemberState::Suspect(reason)
                    if peer.away_before
                        + now
                            .awake
                            .saturating_duration_since(timed_from(peer.since.awake))
                        >= self.grace =>
                {
                    Some((node, MemberState::Gone(reason)))
                }
                MemberState::Alive
                    if peer
                        .last_heard
                        .is_some_and(|heard| now.awake >= timed_from(heard) + WEDGE_TIMEOUT) =>
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

        // A peer Gone for FORGET_AFTER (not counting this node's stalls) no longer counts; it
        // stays in the view until the coordinator removes it
        let resumed_at = self.resumed_at;
        for peer in self.peers.values_mut() {
            if let MemberState::Gone(_) = peer.state {
                let from = resumed_at.map_or(peer.since.awake, |at| peer.since.awake.max(at));
                if now.awake >= from + FORGET_AFTER {
                    peer.counted = false;
                }
            }
        }
        // A Gone peer the cluster's state never listed has nothing to remove: it leaves the view
        self.peers
            .retain(|_, peer| peer.listed || !matches!(peer.state, MemberState::Gone(_)));

        // Ask again, every REPROBE_INTERVAL, each node Alive without a connection of its own
        let mut ask = Vec::new();
        for (&node, peer) in &mut self.peers {
            if peer.state != MemberState::Alive || peer.connected || peer.probing {
                continue;
            }
            match peer.next_probe {
                None => peer.next_probe = Some(now.awake + REPROBE_INTERVAL),
                Some(at) if now.awake >= at => {
                    peer.next_probe = None;
                    ask.push(node);
                }
                Some(_) => {}
            }
        }
        for node in ask {
            self.ask(node);
        }
        self.apply_removals();
        changes
    }

    /// How late the latest tick was, if it was a stall that hasn't been asked about yet. The
    /// caller logs it: this module never logs, so it can be used under a lock that Python-facing
    /// code also takes (logging can wait for Python's GIL).
    pub const fn take_stall(&mut self) -> Option<Duration> {
        self.stalled_for.take()
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
                counted: peer.counted,
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
            .filter(|peer| peer.state == MemberState::Alive && peer.counted)
            .count()
    }

    fn last_known(&self) -> usize {
        1 + self.peers.values().filter(|peer| peer.counted).count()
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
                    probe_again: false,
                    opened_while_probing: false,
                    next_probe: None,
                    last_heard: None,
                    away_before: Duration::ZERO,
                    // Until the cluster's state lists it (see `listed`)
                    counted: false,
                    listed: false,
                    removed: false,
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

    /// Asks the node whether it is still there, or again once the probe in flight ends
    fn ask(&mut self, node: ZenohId) {
        let Some(peer) = self.peers.get_mut(&node) else {
            return;
        };
        if peer.probing {
            peer.probe_again = true;
        } else {
            peer.probing = true;
            self.probes.push(node);
        }
    }

    fn lost(&mut self, node: ZenohId, reason: Reason, now: Now) -> Option<Member> {
        let peer = self.peers.get(&node)?;
        (peer.state == MemberState::Alive)
            .then(|| self.set(node, MemberState::Suspect(reason), now))
    }

    /// Moves a known peer to `state` and returns the change
    fn set(&mut self, node: ZenohId, state: MemberState, now: Now) -> Member {
        let resumed_at = self.resumed_at;
        if let Some(peer) = self.peers.get_mut(&node) {
            let held = now.awake.saturating_duration_since(peer.since.awake);
            match (peer.state, state) {
                // Back from Suspect: its time away counts toward a next loss's grace
                (MemberState::Suspect(_), MemberState::Alive) => {
                    let from = resumed_at.map_or(peer.since.awake, |at| peer.since.awake.max(at));
                    peer.away_before += now.awake.saturating_duration_since(from);
                }
                // Lost after a steady while: a full grace again
                (MemberState::Alive, MemberState::Suspect(_)) if held >= STEADY_AFTER => {
                    peer.away_before = Duration::ZERO;
                }
                // Gone, or back from Gone: whatever came before no longer counts
                (_, MemberState::Gone(_)) | (MemberState::Gone(_), _) => {
                    peer.away_before = Duration::ZERO;
                }
                _ => {}
            }
            peer.state = state;
            peer.since = now;
            peer.next_probe = None;
            if state == MemberState::Alive {
                // Back: it counts again (once the cluster's state lists it), and isn't removed
                peer.counted = peer.listed;
                peer.removed = false;
            }
            if state == MemberState::Alive {
                // It couldn't report while it was away: its wedge timeout starts again
                peer.last_heard = peer.last_heard.map(|_| now.awake);
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
                counted: false,
            },
            |peer| Member {
                node,
                state: peer.state,
                since: peer.since.wall,
                counted: peer.counted,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn zid(n: u8) -> ZenohId {
        format!("{:x}", u128::from(n) + 1)
            .parse()
            .expect("valid zid")
    }

    const OWN: u8 = 0;

    /// A clock the tests move by hand: every clock advances together unless told otherwise
    #[derive(Clone, Copy)]
    struct Clock {
        now: Now,
    }

    impl Clock {
        fn new() -> Self {
            Self {
                now: Now {
                    awake: Instant::now(),
                    with_sleep: Duration::from_secs(1_000),
                    wall: SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000),
                },
            }
        }

        /// Every clock moves on by `by` (a running or frozen process)
        fn advance(&mut self, by: Duration) -> Now {
            self.now.awake += by;
            self.now.with_sleep += by;
            self.now.wall += by;
            self.now
        }

        /// The machine sleeps for `by`: the clock that stops in a sleep doesn't move
        fn sleep(&mut self, by: Duration) -> Now {
            self.now.with_sleep += by;
            self.now.wall += by;
            self.now
        }

        /// Someone (or NTP) sets the wall clock by `by`, forward or back; nothing else moves
        fn set_wall(&mut self, forward: bool, by: Duration) -> Now {
            if forward {
                self.now.wall += by;
            } else {
                self.now.wall -= by;
            }
            self.now
        }
    }

    /// A membership that has ticked once, at the clock's time
    fn membership(clock: &Clock) -> Membership {
        let mut membership = Membership::new(zid(OWN), GRACE, clock.now);
        membership.tick(clock.now);
        membership
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

    /// The quorum read, and the nodes reachable and counted
    fn read(membership: &Membership) -> (Quorum, usize, usize) {
        let snapshot = membership.snapshot();
        (snapshot.quorum, snapshot.reachable, snapshot.last_known)
    }

    /// A peer that joined and reported: its transport is up and the cluster's state lists it
    fn connected_peer(membership: &mut Membership, clock: &Clock, n: u8) {
        membership.token_put(zid(n), clock.now);
        membership.connection_opened(zid(n), clock.now);
        membership.listed(zid(n), clock.now);
    }

    /// Closes the node's connection, and says whether that made the membership ask it
    fn closed(membership: &mut Membership, n: u8) -> bool {
        membership.connection_closed(zid(n));
        membership.take_probes().contains(&zid(n))
    }

    /// The node's connection closes and it doesn't answer
    fn lose(membership: &mut Membership, now: Now, n: u8) -> Option<Member> {
        membership.connection_closed(zid(n));
        membership.take_probes();
        membership.probe_finished(zid(n), false, now)
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
        assert!(!closed(&mut membership, 1));
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
        assert!(closed(&mut membership, 1));
        assert_eq!(
            state(membership.probe_finished(zid(1), false, clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
        assert_eq!(membership.token_deleted(zid(1), clock.now), None);

        // And the other way round: the token of a node without a connection of its own goes
        membership.token_put(zid(2), clock.now);
        assert_eq!(
            state(membership.token_deleted(zid(2), clock.now)),
            Some(suspect(2, Reason::Disconnected))
        );
        assert!(
            !closed(&mut membership, 2),
            "a node already suspect isn't asked"
        );
    }

    // ---- Alive -> Suspect ----

    #[test]
    fn a_node_whose_token_is_deleted_without_a_connection_is_suspect() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        membership.token_put(zid(1), clock.now);
        assert_eq!(
            state(membership.token_deleted(zid(1), clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
    }

    #[test]
    fn a_node_whose_token_goes_while_its_connection_is_open_is_asked_first() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);

        // It answers: it is still there, whatever zenoh did with its token
        assert_eq!(membership.token_deleted(zid(1), clock.now), None);
        assert_eq!(membership.take_probes(), vec![zid(1)]);
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);

        // It doesn't (its connection still open): Suspect
        assert_eq!(membership.token_deleted(zid(2), clock.now), None);
        assert_eq!(membership.take_probes(), vec![zid(2)]);
        assert_eq!(
            state(membership.probe_finished(zid(2), false, clock.now)),
            Some(suspect(2, Reason::Disconnected))
        );
    }

    #[test]
    fn a_node_whose_connection_closed_is_suspect_if_it_doesnt_answer() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        assert!(closed(&mut membership, 1));
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
        assert!(closed(&mut membership, 1));
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);

        // ... or its connection reopened while it was asked
        assert!(closed(&mut membership, 1));
        assert_eq!(membership.connection_opened(zid(1), clock.now), None);
        assert_eq!(membership.probe_finished(zid(1), false, clock.now), None);
        assert_eq!(membership.quorum(), Quorum::Majority);
    }

    #[test]
    fn a_node_is_asked_once_at_a_time() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        assert!(closed(&mut membership, 1));
        membership.connection_opened(zid(1), clock.now);
        assert!(!closed(&mut membership, 1), "one probe at a time");
        // It closed again while it was asked: asked again as soon as the answer is in
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);
        assert_eq!(membership.take_probes(), vec![zid(1)]);
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
        assert!(!closed(&mut membership, 1));

        // A suspect node that announces its leave is gone at once too
        connected_peer(&mut membership, &clock, 2);
        lose(&mut membership, clock.now, 2);
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

    /// Answers every probe the membership asks for, as a node that is up (`true`) or crashed,
    /// collecting what changed
    fn answer_probes(
        membership: &mut Membership,
        now: Now,
        answered: bool,
    ) -> Vec<(ZenohId, MemberState)> {
        let mut changes = Vec::new();
        for node in membership.take_probes() {
            changes.extend(state(membership.probe_finished(node, answered, now)));
        }
        changes
    }

    /// Ticks for `duration`, answering every probe as `answered`
    fn run_answering(
        membership: &mut Membership,
        clock: &mut Clock,
        duration: Duration,
        answered: bool,
    ) -> Vec<(ZenohId, MemberState)> {
        let mut changes = Vec::new();
        let mut waited = Duration::ZERO;
        while waited < duration {
            waited += TICK;
            changes.extend(states(membership.tick(clock.advance(TICK))));
            changes.extend(answer_probes(membership, clock.now, answered));
        }
        changes
    }

    #[test]
    fn a_node_reached_only_through_another_is_asked_again_until_it_crashes() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // Its direct connection closes; it answers through another node
        assert!(closed(&mut membership, 1));
        membership.probe_finished(zid(1), true, clock.now);

        // It keeps answering: it stays Alive, asked about every REPROBE_INTERVAL
        let mut asked = 0;
        for _ in 0..(3 * REPROBE_INTERVAL.as_secs()) {
            membership.tick(clock.advance(TICK));
            let probes = membership.take_probes();
            asked += probes.len();
            for node in probes {
                assert_eq!(membership.probe_finished(node, true, clock.now), None);
            }
        }
        assert_eq!(asked, 2, "asked about every REPROBE_INTERVAL (plus a tick)");

        // It crashes (no connection of this node closes, its token stays): Suspect at the next
        // ask, Gone a grace later
        let changes = run_answering(
            &mut membership,
            &mut clock,
            REPROBE_INTERVAL + TICK + GRACE + TICK,
            false,
        );
        assert_eq!(
            changes,
            vec![
                suspect(1, Reason::Disconnected),
                gone(1, Reason::Disconnected)
            ]
        );
    }

    #[test]
    fn a_node_that_answered_before_its_token_went_is_asked_again() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert!(closed(&mut membership, 1));
        // It crashed just after answering: its token goes before the answer is in
        membership.token_deleted(zid(1), clock.now);
        assert_eq!(
            state(membership.probe_finished(zid(1), true, clock.now)),
            Some(alive(1))
        );
        // Nothing would show its crash but asking again
        let changes = run_answering(
            &mut membership,
            &mut clock,
            REPROBE_INTERVAL + TICK + GRACE + TICK,
            false,
        );
        assert_eq!(
            changes,
            vec![
                suspect(1, Reason::Disconnected),
                gone(1, Reason::Disconnected)
            ]
        );
    }

    #[test]
    fn a_node_whose_link_closed_again_while_it_was_asked_is_asked_again_at_once() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert!(closed(&mut membership, 1));
        // The link is back, and lost again (a crash), before the first answer is in
        membership.connection_opened(zid(1), clock.now);
        assert!(!closed(&mut membership, 1));
        // That answer predates the second loss: it is asked again, and doesn't answer
        assert_eq!(membership.probe_finished(zid(1), true, clock.now), None);
        assert_eq!(membership.take_probes(), vec![zid(1)]);
        assert_eq!(
            state(membership.probe_finished(zid(1), false, clock.now)),
            Some(suspect(1, Reason::Disconnected))
        );
    }

    #[test]
    fn a_node_with_a_connection_of_its_own_isnt_asked_again() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        run_for(&mut membership, &mut clock, REPROBE_INTERVAL * 3);
        assert_eq!(membership.take_probes(), vec![]);
    }

    // ---- Suspect -> Alive ----

    #[test]
    fn a_suspect_node_is_alive_again_when_its_transport_is_back() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);

        lose(&mut membership, clock.now, 1);
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );

        lose(&mut membership, clock.now, 1);
        assert_eq!(
            state(membership.connection_opened(zid(1), clock.now)),
            Some(alive(1))
        );

        // The probe was asked before the token went, and answered after
        assert!(closed(&mut membership, 1));
        lose(&mut membership, clock.now, 1);
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
        assert!(!closed(&mut membership, 1));
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
        lose(&mut membership, clock.now, 1);
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
        lose(&mut membership, clock.now, 1);

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
        lose(&mut membership, clock.now, 1);
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
    fn a_flapping_node_is_gone_once_it_has_been_away_for_the_grace_in_all() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // Away 4 s, back 1 s, again and again: each return used to start the grace again
        let mut changes = Vec::new();
        let mut cycles = 0;
        while !changes.contains(&gone(1, Reason::Disconnected)) && cycles < 20 {
            cycles += 1;
            changes.extend(state(lose(&mut membership, clock.now, 1)));
            changes.extend(run_for(&mut membership, &mut clock, Duration::from_secs(4)));
            changes.extend(state(membership.connection_opened(zid(1), clock.now)));
            changes.extend(run_for(&mut membership, &mut clock, TICK));
        }
        // Gone in the cycle in which its time away reached GRACE (30 s: the 8th)
        assert_eq!(cycles, 8, "{changes:?}");
    }

    #[test]
    fn a_node_lost_now_and_then_gets_a_full_grace_each_time() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        for _ in 0..5 {
            lose(&mut membership, clock.now, 1);
            assert_eq!(run_for(&mut membership, &mut clock, GRACE - TICK), vec![]);
            membership.connection_opened(zid(1), clock.now);
            run_for(&mut membership, &mut clock, STEADY_AFTER);
        }
        assert_eq!(
            states(membership.snapshot().members),
            vec![alive(OWN), alive(1)]
        );
    }

    #[test]
    fn a_node_back_before_the_grace_is_never_gone() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, GRACE - TICK);
        membership.token_put(zid(1), clock.now);
        assert_eq!(run_for(&mut membership, &mut clock, STEADY_AFTER), vec![]);

        // Lost again after a steady while: the grace starts over
        lose(&mut membership, clock.now, 1);
        assert_eq!(run_for(&mut membership, &mut clock, GRACE - TICK), vec![]);
    }

    #[test]
    fn a_node_lost_again_soon_after_it_was_back_has_only_the_rest_of_the_grace() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, GRACE - TICK * 5);
        membership.connection_opened(zid(1), clock.now);
        run_for(&mut membership, &mut clock, TICK);
        lose(&mut membership, clock.now, 1);
        assert_eq!(run_for(&mut membership, &mut clock, TICK * 4), vec![]);
        assert_eq!(
            run_for(&mut membership, &mut clock, TICK),
            vec![gone(1, Reason::Disconnected)]
        );
    }

    /// Whether the node is in the view, and if so whether it counts in the quorum read
    fn counted(membership: &Membership, n: u8) -> Option<bool> {
        membership
            .snapshot()
            .members
            .into_iter()
            .find(|member| member.node == zid(n))
            .map(|member| member.counted)
    }

    #[test]
    fn a_node_that_left_stops_counting_at_once_and_stays_in_view_until_removed() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        membership.left(zid(1), clock.now);
        let snapshot = membership.snapshot();
        assert_eq!(
            states(snapshot.members),
            vec![alive(OWN), gone(1, Reason::Left)]
        );
        assert_eq!(snapshot.last_known, 1);
        assert_eq!(counted(&membership, 1), Some(false));
        membership.forget(zid(1));
        assert_eq!(counted(&membership, 1), None);
    }

    #[test]
    fn a_node_gone_for_forget_after_stays_in_view_uncounted_until_removed_or_back() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, GRACE + FORGET_AFTER);
        // Still in the view as Gone, so a coordinator can still remove it, but not counted
        assert_eq!(
            states(membership.snapshot().members),
            vec![alive(OWN), gone(1, Reason::Disconnected)]
        );
        assert_eq!(counted(&membership, 1), Some(false));
        assert_eq!(membership.quorum(), Quorum::Majority);

        // Back: it counts again
        membership.connection_opened(zid(1), clock.now);
        assert_eq!(counted(&membership, 1), Some(true));
        assert_eq!(membership.snapshot().last_known, 2);

        // Lost and removed by the coordinator: out of the view once this node reads Majority (with
        // a third node back; alone it reads Half and keeps it)
        lose(&mut membership, clock.now, 1);
        membership.forget(zid(1));
        assert_eq!(counted(&membership, 1), Some(true));
        connected_peer(&mut membership, &clock, 2);
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(counted(&membership, 1), None);
    }

    #[test]
    fn the_coordinator_removing_a_node_this_one_reaches_changes_nothing() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        membership.forget(zid(1));
        run_for(&mut membership, &mut clock, GRACE);
        assert_eq!(membership.snapshot().last_known, 3);
    }

    #[test]
    fn a_node_back_after_its_removal_is_no_longer_removed() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // Lost and removed while this node, alone, reads Half: the removal waits
        lose(&mut membership, clock.now, 1);
        membership.forget(zid(1));
        // Back after that real loss: the removal no longer holds
        membership.connection_opened(zid(1), clock.now);
        lose(&mut membership, clock.now, 1);
        connected_peer(&mut membership, &clock, 2);
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(counted(&membership, 1), Some(true));
    }

    #[test]
    fn a_pair_stays_a_majority_when_the_other_node_restarts_under_new_ids() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        for (old, new) in [(1, 2), (2, 3), (3, 4)] {
            // It quits; the coordinator removes it (on announce-leave a quitting node asks it to
            // at once; otherwise its silence timeout does); it comes back with a new id
            lose(&mut membership, clock.now, old);
            run_for(&mut membership, &mut clock, Duration::from_secs(15));
            membership.forget(zid(old));
            // Alone, this node reads Half: the removal waits for a Majority read
            assert_eq!(counted(&membership, old), Some(true));
            connected_peer(&mut membership, &clock, new);
            run_for(&mut membership, &mut clock, TICK);
            let snapshot = membership.snapshot();
            assert_eq!(
                (snapshot.quorum, snapshot.reachable, snapshot.last_known),
                (Quorum::Majority, 2, 2),
                "after the restart as node {new}"
            );
            assert_eq!(counted(&membership, old), None, "removed once Majority");
        }
    }

    #[test]
    fn a_removal_of_a_node_still_alive_here_applies_once_it_leaves() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        // A quitting node announces its departure (a removal) before its token goes
        membership.forget(zid(1));
        assert_eq!(counted(&membership, 1), Some(true), "still Alive here");
        run_for(&mut membership, &mut clock, Duration::from_secs(3));
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(counted(&membership, 1), None);
        assert_eq!(membership.snapshot().last_known, 2);
    }

    #[test]
    fn a_removal_waiting_while_the_node_is_alive_is_called_off_when_it_is_listed_again() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        // Removed while this node still reaches it, and then it reports again
        membership.forget(zid(1));
        membership.listed(zid(1), clock.now);
        // Lost later: it stays in the view and counts, to be Gone and removed like any other
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(counted(&membership, 1), Some(true));
        run_for(&mut membership, &mut clock, GRACE + TICK);
        let member = membership
            .snapshot()
            .members
            .into_iter()
            .find(|m| m.node == zid(1));
        assert_eq!(
            member.map(|m| m.state),
            Some(MemberState::Gone(Reason::Disconnected))
        );
    }

    #[test]
    fn a_pair_whose_other_node_announces_each_quit_stays_a_majority() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        for (old, new) in [(1, 2), (2, 3), (3, 4)] {
            run_for(&mut membership, &mut clock, Duration::from_secs(20));
            // It quits: its departure first, while it is still Alive here; its token 3 s later
            membership.forget(zid(old));
            run_for(&mut membership, &mut clock, Duration::from_secs(3));
            lose(&mut membership, clock.now, old);
            run_for(&mut membership, &mut clock, Duration::from_secs(10));
            // It comes back under a new id
            connected_peer(&mut membership, &clock, new);
            run_for(&mut membership, &mut clock, TICK);
            let snapshot = membership.snapshot();
            assert_eq!(
                (snapshot.quorum, snapshot.reachable, snapshot.last_known),
                (Quorum::Majority, 2, 2),
                "after the restart as node {new}"
            );
        }
    }

    #[test]
    fn a_node_cut_off_from_two_others_keeps_them_until_the_split_hold() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        lose(&mut membership, clock.now, 1);
        lose(&mut membership, clock.now, 2);
        // Its own coordinator (itself, cut off) removes the other two at its silence timeout
        run_for(&mut membership, &mut clock, Duration::from_secs(15));
        membership.forget(zid(1));
        membership.forget(zid(2));
        // It reads Minority, and keeps both, until they stop counting at the split hold
        run_for(
            &mut membership,
            &mut clock,
            GRACE + SPLIT_HOLD - Duration::from_secs(16),
        );
        let snapshot = membership.snapshot();
        assert_eq!(
            (snapshot.quorum, snapshot.reachable, snapshot.last_known),
            (Quorum::Minority, 1, 3)
        );
        run_for(&mut membership, &mut clock, Duration::from_secs(2));
        assert_eq!(membership.quorum(), Quorum::Majority, "after the hold");
    }

    #[test]
    fn neither_side_of_a_two_node_cut_drops_the_other_before_the_split_hold() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, Duration::from_secs(15));
        // Each side's coordinator (each node, for its side) removes the other
        membership.forget(zid(1));
        run_for(
            &mut membership,
            &mut clock,
            GRACE + SPLIT_HOLD - Duration::from_secs(16),
        );
        assert_eq!(membership.quorum(), Quorum::Half);
        assert_eq!(counted(&membership, 1), Some(true));
    }

    #[test]
    fn a_majority_removes_a_crashed_node_at_once() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        lose(&mut membership, clock.now, 2);
        membership.forget(zid(2));
        let snapshot = membership.snapshot();
        assert_eq!(counted(&membership, 2), None);
        assert_eq!(
            (snapshot.quorum, snapshot.reachable, snapshot.last_known),
            (Quorum::Majority, 2, 2)
        );
    }

    #[test]
    fn restarts_nobody_removed_stop_counting_once_gone_for_forget_after() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // The other node of a pair crashes and comes back under a new id, three times, and
        // nobody removes the old ids (the coordinator is this node, and it didn't)
        for (old, new) in [(1, 2), (2, 3), (3, 4)] {
            lose(&mut membership, clock.now, old);
            run_for(&mut membership, &mut clock, GRACE + TICK);
            connected_peer(&mut membership, &clock, new);
        }
        // Meanwhile the old ids count: 2 of 5 (with the removals a node makes, never)
        assert_eq!(membership.quorum(), Quorum::Minority);
        // Once each old id has been Gone for FORGET_AFTER it stops counting
        run_for(&mut membership, &mut clock, FORGET_AFTER);
        let snapshot = membership.snapshot();
        assert_eq!(
            (snapshot.quorum, snapshot.reachable, snapshot.last_known),
            (Quorum::Majority, 2, 2)
        );
    }

    #[test]
    fn a_gone_node_counts_until_forget_after() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, GRACE);
        assert_eq!(membership.quorum(), Quorum::Half);
        run_for(&mut membership, &mut clock, FORGET_AFTER - TICK);
        assert_eq!(
            membership.quorum(),
            Quorum::Half,
            "the split hold isn't cut short"
        );
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(membership.quorum(), Quorum::Majority);
    }

    // ---- Nodes the cluster's state lists ----

    #[test]
    fn a_listed_node_never_seen_is_suspect_and_gone_after_the_grace() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        assert_eq!(
            state(membership.listed(zid(2), clock.now)),
            Some(suspect(2, Reason::Disconnected))
        );
        // Listed again (it is in every report and snapshot): reported once
        assert_eq!(membership.listed(zid(2), clock.now), None);
        // It counts: the cluster's state lists it
        let snapshot = membership.snapshot();
        assert_eq!((snapshot.reachable, snapshot.last_known), (2, 3));
        assert_eq!(run_for(&mut membership, &mut clock, GRACE - TICK), vec![]);
        assert_eq!(
            run_for(&mut membership, &mut clock, TICK),
            vec![gone(2, Reason::Disconnected)]
        );
    }

    #[test]
    fn a_listed_node_counts_at_once_and_is_reachable_once_its_transport_shows_it() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        membership.listed(zid(1), clock.now);
        assert_eq!(read(&membership), (Quorum::Half, 1, 2));
        assert_eq!(
            state(membership.token_put(zid(1), clock.now)),
            Some(alive(1))
        );
        assert_eq!(read(&membership), (Quorum::Majority, 2, 2));
        // A report alone doesn't show it: only the transport does
        membership.listed(zid(2), clock.now);
        assert_eq!(membership.heard_from(zid(2), clock.now), None);
        assert_eq!(read(&membership), (Quorum::Majority, 2, 3));
    }

    #[test]
    fn a_node_that_joins_the_smaller_side_of_a_split_reads_it_as_a_minority() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        // Its side: node 1. The state it adopts also lists nodes 2, 3 and 4, which are on the
        // other side
        connected_peer(&mut membership, &clock, 1);
        for n in 2..=4 {
            membership.listed(zid(n), clock.now);
        }
        assert_eq!(read(&membership), (Quorum::Minority, 2, 5));
        // Still a minority once they are Gone, until they have been Gone for FORGET_AFTER (as
        // on that side's other node, which lost them)
        run_for(&mut membership, &mut clock, GRACE + FORGET_AFTER - TICK);
        assert_eq!(membership.quorum(), Quorum::Minority);
        run_for(&mut membership, &mut clock, TICK);
        assert_eq!(read(&membership), (Quorum::Majority, 2, 2));
    }

    #[test]
    fn listing_a_known_node_or_this_one_changes_nothing() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        lose(&mut membership, clock.now, 1);
        assert_eq!(membership.listed(zid(1), clock.now), None);
        assert_eq!(membership.listed(zid(OWN), clock.now), None);
        assert_eq!(
            states(membership.snapshot().members),
            vec![alive(OWN), suspect(1, Reason::Disconnected)]
        );
    }

    #[test]
    fn a_listed_node_is_timed_from_the_end_of_resuming() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        membership.tick(clock.now);
        membership.tick(clock.advance(Duration::from_secs(20)));
        membership.listed(zid(1), clock.now);
        assert_eq!(
            run_for(
                &mut membership,
                &mut clock,
                HOLD_OFF + REJOIN + GRACE - TICK
            ),
            vec![alive(OWN)]
        );
        assert_eq!(
            run_for(&mut membership, &mut clock, TICK),
            vec![gone(1, Reason::Disconnected)]
        );
    }

    #[test]
    fn a_node_that_leaves_before_the_cluster_lists_it_never_counts() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // Node 2 starts and quits within seconds, before it ever reports
        membership.token_put(zid(2), clock.now);
        let snapshot = membership.snapshot();
        assert_eq!((snapshot.reachable, snapshot.last_known), (2, 2));
        membership.token_deleted(zid(2), clock.now);
        assert_eq!(membership.quorum(), Quorum::Majority);
        // Gone, and never listed: nothing to remove, so it leaves the view
        run_for(&mut membership, &mut clock, GRACE);
        assert_eq!(
            states(membership.snapshot().members),
            vec![alive(OWN), alive(1)]
        );
    }

    #[test]
    fn a_node_counts_once_it_is_alive_and_listed_in_either_order() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        membership.token_put(zid(1), clock.now);
        assert_eq!(membership.snapshot().last_known, 1);
        membership.listed(zid(1), clock.now);
        assert_eq!(membership.snapshot().last_known, 2);
        membership.listed(zid(2), clock.now);
        membership.token_put(zid(2), clock.now);
        let snapshot = membership.snapshot();
        assert_eq!((snapshot.reachable, snapshot.last_known), (3, 3));
    }

    #[test]
    fn a_pair_whose_other_node_restarts_quickly_three_times_stays_a_majority() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        // It quits, and comes back under new ids that quit again before they report
        lose(&mut membership, clock.now, 1);
        for n in 2..=4 {
            run_for(&mut membership, &mut clock, TICK);
            membership.token_put(zid(n), clock.now);
            run_for(&mut membership, &mut clock, TICK * 3);
            membership.token_deleted(zid(n), clock.now);
        }
        // The last one stays and reports
        connected_peer(&mut membership, &clock, 5);
        assert_eq!(membership.quorum(), Quorum::Majority);
        let snapshot = membership.snapshot();
        // This node, the original peer (until the coordinator removes it) and the last one
        assert_eq!((snapshot.reachable, snapshot.last_known), (2, 3));
    }

    #[test]
    fn a_node_waking_from_a_real_sleep_blames_no_peer_for_it() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        membership.heard_from(zid(1), clock.now);
        membership.heard_from(zid(2), clock.now);
        membership.tick(clock.now);
        // Asleep for two minutes: only the clock that counts sleep moves
        clock.sleep(Duration::from_secs(120));
        assert_eq!(
            states(membership.tick(clock.advance(TICK))),
            vec![(zid(OWN), MemberState::Resuming)]
        );
        // Its peers dropped their sessions with it meanwhile: it sees them close on waking
        lose(&mut membership, clock.now, 1);
        lose(&mut membership, clock.now, 2);
        // Neither Gone nor wedged within a grace of the end of Resuming
        assert_eq!(
            run_for(
                &mut membership,
                &mut clock,
                HOLD_OFF + REJOIN + GRACE - TICK
            ),
            vec![alive(OWN)]
        );
        // They are back once discovery lets them in again
        membership.token_put(zid(1), clock.now);
        membership.token_put(zid(2), clock.now);
        assert_eq!(membership.quorum(), Quorum::Majority);
        assert_eq!(membership.snapshot().last_known, 3);
    }

    #[test]
    fn a_forgotten_node_is_new_when_it_comes_back() {
        let clock = Clock::new();
        let mut membership = membership(&clock);
        connected_peer(&mut membership, &clock, 1);
        connected_peer(&mut membership, &clock, 2);
        lose(&mut membership, clock.now, 1);
        membership.forget(zid(1));
        assert_eq!(membership.snapshot().last_known, 2);
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
        // How late it was is there to log, once
        assert_eq!(membership.take_stall(), Some(STALL_THRESHOLD));
        assert_eq!(membership.take_stall(), None);
        assert_eq!(membership.own().state, MemberState::Resuming);
        assert_eq!(
            run_for(&mut membership, &mut clock, HOLD_OFF + REJOIN - TICK),
            vec![]
        );
        assert_eq!(run_for(&mut membership, &mut clock, TICK), vec![alive(OWN)]);
    }

    #[test]
    fn a_sleeping_machine_is_resuming_though_its_awake_clock_stopped() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        membership.tick(clock.advance(TICK));
        // Asleep for a minute: the next tick is one awake second later
        clock.sleep(Duration::from_secs(60));
        assert_eq!(
            states(membership.tick(clock.advance(TICK))),
            vec![(zid(OWN), MemberState::Resuming)]
        );
    }

    #[test]
    fn a_wall_clock_set_while_frozen_is_still_a_stall() {
        let mut clock = Clock::new();
        let mut membership = membership(&clock);
        clock.set_wall(false, Duration::from_secs(60));
        assert_eq!(
            states(membership.tick(clock.advance(Duration::from_secs(30)))),
            vec![(zid(OWN), MemberState::Resuming)]
        );
    }

    #[test]
    fn a_wall_clock_set_while_running_is_no_stall() {
        // An NTP step, or someone setting the time, either way
        for forward in [true, false] {
            let mut clock = Clock::new();
            let mut membership = membership(&clock);
            clock.set_wall(forward, Duration::from_secs(3600));
            assert_eq!(membership.tick(clock.advance(TICK)), vec![]);
            assert_eq!(run_for(&mut membership, &mut clock, TICK * 3), vec![]);
            assert_eq!(membership.own().state, MemberState::Alive);
        }
    }

    #[test]
    fn the_clock_that_counts_sleep_is_read_and_moves_with_the_awake_one() {
        let before = Now::read();
        std::thread::sleep(Duration::from_millis(50));
        let after = Now::read();
        assert!(before.with_sleep > Duration::ZERO, "clock_gettime failed");
        let awake = after.awake - before.awake;
        let with_sleep = after.with_sleep - before.with_sleep;
        assert!(
            with_sleep.abs_diff(awake) < Duration::from_millis(20),
            "{awake:?} vs {with_sleep:?}"
        );
        assert_eq!(stall(before, after), None);
    }

    #[test]
    fn the_first_tick_is_never_late() {
        let mut clock = Clock::new();
        // Made long before it first ticks (exo read the stream late)
        let mut membership = Membership::new(zid(OWN), GRACE, clock.now);
        assert_eq!(
            membership.tick(clock.advance(Duration::from_secs(60))),
            vec![]
        );
        assert_eq!(membership.own().state, MemberState::Alive);
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
        lose(&mut membership, clock.now, 1);
        run_for(&mut membership, &mut clock, Duration::from_secs(5));
        membership.tick(clock.advance(Duration::from_secs(40)));
        // Its loss is reported at once (and starts an election as before): only its grace waits
        assert_eq!(
            state(lose(&mut membership, clock.now, 2)),
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
        // The last `gone` peers were lost a grace ago, the first `away` ones just now
        for n in (away + 1)..=(away + gone) {
            lose(&mut membership, clock.now, n);
        }
        run_for(&mut membership, &mut clock, GRACE);
        for n in 1..=away {
            lose(&mut membership, clock.now, n);
        }
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
        lose(&mut membership, lost_at, 2);
        let snapshot = membership.snapshot();
        assert_eq!(
            snapshot.members,
            vec![
                Member {
                    node: zid(OWN),
                    state: MemberState::Alive,
                    since: start,
                    counted: true,
                },
                Member {
                    node: zid(1),
                    state: MemberState::Alive,
                    since: start,
                    counted: true,
                },
                Member {
                    node: zid(2),
                    state: MemberState::Suspect(Reason::Disconnected),
                    since: lost_at.wall,
                    counted: true,
                },
            ]
        );
    }

    // ---- The timers' derivations ----

    #[test]
    fn the_timers_keep_their_derivations() {
        // A peer frozen just past the lease is back HOLD_OFF + REJOIN after it was Suspect
        assert_eq!(GRACE, HOLD_OFF + REJOIN + FREEZE_RIDDEN_OUT);
        assert_eq!(GRACE, Duration::from_secs(30));
        // About 5 times the longest gap between a busy node's reports (6.3 s)
        assert!(WEDGE_TIMEOUT >= Duration::from_millis(6_300) * 4);
        assert!(
            HOLD_OFF > LEASE,
            "stale sessions must close before reconnecting"
        );
        assert!(STALL_THRESHOLD < LEASE);
        assert!(STALL_THRESHOLD > TICK);
        // A split hold outlasts any blip the grace rides out, and the 90 s outages that healed
        assert!(SPLIT_HOLD > GRACE);
        // A crash of a node reached only through another is noticed about when a frozen peer is
        assert!(REPROBE_INTERVAL + TICK + PROBE_TIMEOUT <= LEASE + PROBE_TIMEOUT + TICK);
        assert!(SPLIT_HOLD >= Duration::from_secs(90) * 3);
        // Forgetting a lost node ends the hold for its work: never before the hold ends
        assert!(FORGET_AFTER >= SPLIT_HOLD);
        // Back for as long as a lost node is waited for: steady
        assert_eq!(STEADY_AFTER, GRACE);
        assert_eq!(PROBE_FIRST_ASK_DELAY, ZENOH_ROUTES_RECOMPUTED_AFTER * 3);
        assert!(
            PROBE_FIRST_ASK_DELAY * 4 <= PROBE_TIMEOUT,
            "a node is asked several times before it is suspect"
        );
    }
}
