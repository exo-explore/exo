pub mod centrality;
pub mod messaging;

use crate::cel::data::{Map, Set};
use std::collections::VecDeque;

pub mod data {
    use std::marker::PhantomData;

    #[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
    pub struct Set<V>(PhantomData<V>);

    impl<V> Set<V> {
        pub fn new() -> Self {
            todo!()
        }

        pub fn add(&mut self, value: V) -> bool {
            todo!()
        }

        pub fn remove(&mut self, value: V) {}

        pub fn add_all(&mut self, other: &Set<V>) {}

        pub fn values_mut(&mut self) -> &mut [V] {
            todo!()
        }

        pub fn values(&self) -> &[V] {
            todo!()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
    pub struct Map<K, V>(PhantomData<(K, V)>);

    impl<K, V> Map<K, V> {
        pub fn new() -> Self {
            todo!()
        }

        pub fn set(&mut self, key: K, value: V) {}

        pub fn get(&self, key: K) -> &V {
            todo!()
        }

        pub fn get_mut(&mut self, key: K) -> &mut V {
            todo!()
        }

        pub fn kv_mut(&mut self) -> &mut [(K, V)] {
            todo!()
        }

        pub fn contains_key(&self, key: K) -> bool {
            todo!()
        }

        pub fn not_contains_key(&self, key: K) -> bool {
            !self.contains_key(key)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct ID(pub u128);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct Clock(pub u64);

impl Clock {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn plus_one(self) -> Self {
        Self(self.0 + 1)
    }
}

/// `CEL` uses a data structure called a `view`
///
/// A `view` associated to node  is composed of two elements:
/// 1) A logical `clock` value, acting as a timestamp and incremented at each connection and disconnection.
/// 2) A set of node `identifiers`, which are the current neighbors of `i` (this node).
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct View {
    /// Logical clock
    clock: Clock,

    /// Neighbors set
    neigh: Set<ID>,
}

impl View {
    pub fn new(clock: Clock, neigh: Set<ID>) -> Self {
        Self { clock, neigh }
    }
}

/// The only type of message exchanged between neighbors is the `knowledge` message.
/// It contains the current topological knowledge that the sender node has of the network,
/// i.e. its `known` variable.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct KnowledgeMessage {
    pub known: Map<ID, View>,
}

/// Each node `i` maintains a local variable called `known`.
///
/// This variable represents the current topological knowledge that `i` has of its current
/// component (including itself). It is implemented  as a map of `view` indexed by node `identifier`.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Node {
    id: ID,
    known: Map<ID, View>,
}

impl Node {
    /// Firstly, node  initializes its `known` variable with its own identifier (`i`),
    /// and sets its logical clock to `0`.
    pub fn initialization(this_id: ID) -> Self {
        let mut neigh = Set::new(); // neigh = \{ i \}
        neigh.add(this_id);

        let mut known = Map::<ID, View>::new();
        known.set(this_id, View::new(Clock::ZERO, neigh));

        Self { id: this_id, known }
    }

    /// When a new node `j` appears in the transmission range of `i`, the crosslayer mechanism of
    /// `i` detects `j`, and triggers the `Connection` method.
    ///
    /// Node `j` is added to the neighbors set of node `i`. As the knowledge of  has been updated,
    /// its logical clock is incremented.
    ///
    /// Since links are assumed bidirectional, i.e. the emission range equals the reception range,
    /// if node `i` has no previous knowledge of `j`, the neighbor-aware mechanism adds both
    /// `i` and `j` in the set of neighbors of `j`. Then, `i` sets the clock value of `j` to `1`,
    /// as `i` was added to the knowledge of node `j`. On the other hand, if node `i` already has
    /// information about `j`, `i` is added to the neighbors of `j`, and the logical clock of
    /// node `j` is incremented.
    ///
    /// Finally, by calling `LocalBroadcast` method, node `i` shares its
    /// knowledge with `j` and informs its neighborhood of its new neighbor `j`.
    /// Note that such a method sends a knowledge message to the neighbors
    /// of node `i`, with a gossip probability `\rho`, as seen in `Section 2.8`.
    /// However, for the first hop, `\rho` is set to `1` to make sure that all neighbors of `i`
    /// will be aware of its new neighbor `j`. Note that the cross-layer mechanism
    /// of node `j` will also trigger its `Connection` method, and the respective
    /// steps will also be achieved on node `j`.
    pub fn node_connection(&mut self, other_id: ID) {
        let this_known = self.known.get_mut(self.id);
        this_known.neigh.add(other_id);
        this_known.clock = this_known.clock.plus_one();

        if self.known.not_contains_key(other_id) {
            let mut other_neigh = Set::new(); // neigh = \{ j, i \}
            other_neigh.add(self.id);
            other_neigh.add(other_id);

            self.known.set(other_id, View::new(Clock::ONE, other_neigh));
        } else {
            let other_known = self.known.get_mut(other_id);
            other_known.neigh.add(self.id);
            other_known.clock = other_known.clock.plus_one();
        }

        // TODO: `LocalBroadcast(knowlege<known>, 1)`
    }

    /// When a node `j` disappears from the transmission range of node `i`,
    /// the cross-layer mechanism stops receiving beacon messages at the
    /// MAC level, and triggers the `Disconnection` method. Node `j` is
    /// then removed from the knowledge of node `i`, and its clock
    /// is incremented as its knowledge was modified.
    ///
    /// Then, the neighbor-aware mechanism assumes that node `i` will also disconnect
    /// from `j`. Therefore, `i` is removed from the neighborhood of `j` in the
    /// knowledge of node `i`, and the corresponding clock is incremented.
    ///
    /// Finally, node `i` broadcasts its updated knowledge to its neighbors.
    pub fn node_disconected(&mut self, other_id: ID) {
        let this_known = self.known.get_mut(self.id);
        this_known.neigh.remove(other_id);
        this_known.clock = this_known.clock.plus_one();

        let other_known = self.known.get_mut(other_id);
        other_known.neigh.remove(self.id);
        other_known.clock = other_known.clock.plus_one();

        // TODO: `LocalBroadcast(knowlege<known>, 1)`
    }

    /// When node  receives a knowledge message `known_j`, from node `j`,
    /// it looks at each node `n` included in `known_j`. If `n` is an
    /// unknown node for `i`, or if `n` is known by node `i` and has a
    /// more recent clock value in `known_j`, the clock and neighbors of
    /// node `n` are updated in the knowledge of `i`.
    ///
    /// Note that a clock value of `n` higher than the one currently known by
    /// node `i` means that node `n` made some connections and/or
    /// disconnections of which node `i` is not aware. Then, the `UpdateNeighbors`
    /// method is called to update the knowledge of `i` regarding the neighbors
    /// of `n`. If the clock value of node `n` is identical to the one of
    /// both the knowledge of node `i` and `known_j`, the neighbor-aware
    /// mechanism merges the neighbors of node `n` from `known_j` with the
    /// known neighbors of `n` in the knowledge of `i`.
    ///
    /// Remark that the clock of node `n` is not updated by the neighbor-aware
    /// mechanism, otherwise, `n` would not be able to override this view in the
    /// future with more recent information. The `UpdateNeighbors` method is
    /// then called. Finally, node `i` broadcasts its knowledge only if
    /// this latter was modified.
    pub fn receive_knowledge(
        &mut self,
        other_id: ID,
        KnowledgeMessage {
            known: mut other_known,
        }: KnowledgeMessage,
    ) {
        let mut this_known_updated = false;

        for (n, other_known_n) in other_known.kv_mut() {
            if self.known.not_contains_key(*n) || other_known_n.clock > self.known.get(*n).clock {
                self.known.set(*n, other_known_n.clone());
                // TODO: UpdateNeighbors(known_j, n)
            } else if other_known_n.clock == self.known.get(*n).clock {
                self.known.get_mut(*n).neigh.add_all(&other_known_n.neigh);
                // TODO: UpdateNeighbors(known_j, n)
            }
        }

        // TODO: figure out what constitutes "updated", i.e. should any of the two branches count?
        //       or should each atomic update-op be checked for "change"??
        if this_known_updated {
            // TODO: TopologicalBroadcast()
        }
    }

    /// The `UpdateNeighbors` method updates the knowledge of `i` with
    /// information about the neighbors of node `n`. If the neighbor `k`
    /// is an unknown node for `i`, or if `k` is known by `i` but has a
    /// more recent clock value in `known_j` (line 38), the clock and neighbors
    /// of node `k` are added or updated in the knowledge of node `i`.
    /// Otherwise, if the clock of node `k` is identical in the knowledge of node
    /// `i` and in `known_j`, the neighbor-aware mechanism merges the
    /// neighbors of node `k` in the knowledge of `i`.
    fn update_neighbors(&mut self, other_known: &mut Map<ID, View>, n: ID) {
        for k in other_known.get(n).neigh.values() {
            if self.known.not_contains_key(*k)
                || other_known.get(*k).clock > self.known.get(*k).clock
            {
                self.known.set(*k, other_known.get(*k).clone());
            } else if other_known.get(*k).clock == self.known.get(*k).clock {
                self.known
                    .get_mut(*k)
                    .neigh
                    .add_all(&other_known.get(*k).neigh);
            }
        }
    }

    /// The `TopologicalBroadcast` method uses a self-pruning approach to broadcast
    /// or not the updated knowledge of node `i`, after the reception of a `knowledge`
    /// from a neighbor `j`. To this end, node `i` checks whether each of its neighbors
    /// has the same neighborhood as itself. In this case, node `n` is supposed to have
    /// also received the knowledge message from neighbor node `j`. Therefore, among the
    /// neighbors having the same neighborhood than `i`, only the one with
    /// the smallest identifier will broadcast the knowledge, with a
    /// gossip probability `\rho`. Note that this topological self-pruning
    /// mechanism reaches the same neighborhood as multiple broadcasts.
    fn topological_broadcast(&self) {
        for n in self.known.get(self.id).neigh.values() {
            // TODO: ensure this is a value-equality comparison
            if self.known.get(*n).neigh == self.known.get(self.id).neigh {
                if *n < self.id {
                    return;
                }
            }
        }

        // TODO: `LocalBroadcast(knowlege<known>, \rho)`
    }

    /// The leader is elected when a process running on node `i` calls the `Leader`
    /// function. This function returns the most central leader in the component
    /// according the closeness centrality, as seen in Section 2.7, using the
    /// knowledge of node `i`. The closeness centrality is chosen instead of the
    /// betweenness centrality, because it is faster to compute and requires fewer
    /// computational steps, therefore consuming less energy from the mobile node
    /// batteries than the latter.
    ///
    /// First, node `i` rebuilds its component according to its topological knowledge.
    /// To do so, it computes the entire set of reachable nodes, by adding
    /// neighbors, neighbors of its neighbors, and so on.
    /// Then, it evaluates the shortest distance between each reachable node and the
    /// other ones, and computes the closeness centrality for each of them.
    /// Finally, it returns the node having the highest closeness value as the
    /// leader. The highest node identifier is used to break ties among
    /// identical centrality values. If all nodes of the component have the same
    /// topological knowledge, the `Leader()` function will return the same leader
    /// node when invoked. Otherwise, it may return different leader nodes.
    /// However, when the network topology stops changing, the algorithm
    /// ensures that all nodes of a component will eventually have the same
    /// topological knowledge and therefore, the `Leader()` function will return
    /// the same leader node for all of them.
    fn leader(&self) -> ID {
        // this just computes the transitive closure of the adj-list graph starting from node `i`
        // TODO: its an inefficient BFS impl, swap to better later!!!
        let mut component = Set::new();

        let mut process_queue =
            VecDeque::from_iter(self.known.get(self.id).neigh.values().iter().cloned());
        while let Some(j) = process_queue.pop_front() {
            let successfully_added = component.add(j);

            // was already processed, so don't add neighbors
            if !successfully_added {
                continue;
            }

            process_queue.extend(self.known.get(j).neigh.values().iter().cloned());
        }

        let leader: ID = todo!(); // TODO: `Max (ClosenessCentrality (component))`
        return leader;
    }
}
